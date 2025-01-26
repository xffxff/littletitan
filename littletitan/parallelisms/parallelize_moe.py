import torch.nn as nn
from torch.distributed import DeviceMesh
from torchtitan.config_manager import TORCH_DTYPE_MAP, JobConfig
from torchtitan.logging import logger
from torchtitan.parallelisms.parallel_dims import ParallelDims
from torchtitan.parallelisms.parallelize_llama import apply_ac, apply_ddp
from typing import Optional
from torch.distributed.tensor.parallel import parallelize_module
import torch

from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)


def parallelize_moe(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.ep_enabled:
        apply_ep(model, world_mesh["ep"])

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if (
        parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        expert_dp_shard_mesh = None
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                expert_dp_shard_mesh_dim_names = ("dp_replicate", "expert_dp_shard")
            else:
                expert_dp_shard_mesh_dim_names = ("expert_dp_shard",)
            expert_dp_shard_mesh = world_mesh[tuple(expert_dp_shard_mesh_dim_names)]

        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            ep_enabled=parallel_dims.ep_enabled,
            expert_dp_shard_mesh=expert_dp_shard_mesh,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.experimental.enable_compiled_autograd,
        )


def apply_ep(model: nn.Module, ep_mesh: Optional[DeviceMesh] = None):
    from littletitan.models.token_dispatcher import EPTokenDispatcher
    from littletitan.models.model import MoELayer

    assert ep_mesh is not None

    for _, transformer_block in model.layers.items():
        moe_layer: MoELayer = transformer_block.feed_forward

        moe_layer.dispatcher = EPTokenDispatcher(
            moe_layer.top_k, moe_layer.num_experts, ep_mesh
        )
        sparse_experts = transformer_block.feed_forward.sparse_experts
        for name, param in sparse_experts.named_parameters(recurse=False):
            dist_param = nn.Parameter(
                distribute_tensor(param, ep_mesh, [Shard(0)]).to_local()
            )
            sparse_experts.register_parameter(name, dist_param)


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    ep_enabled: bool = False,
    expert_dp_shard_mesh: Optional[DeviceMesh] = None,
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for layer_id, transformer_block in model.layers.items():
        if pp_enabled:
            # For PP, do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = False
        else:
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.layers) - 1

        if ep_enabled:
            fsdp_expert_shard_config = fsdp_config.copy()
            fsdp_expert_shard_config["mesh"] = expert_dp_shard_mesh
            fully_shard(
                transformer_block.feed_forward.sparse_experts,
                **fsdp_expert_shard_config,
                reshard_after_forward=reshard_after_forward,
            )

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)
