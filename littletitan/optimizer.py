from typing import List

from torch.nn import Module
from torchtitan.config_manager import JobConfig
from torchtitan.optimizer import OptimizersContainer, OptimizersInBackwardContainer


def build_optimizers(
    model_parts: List[Module], job_config: JobConfig
) -> OptimizersContainer:
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    fused = job_config.optimizer.fused
    optimizer_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": False,  # as not all parameters are DTensors on the same meshes (e.g. MoE non-shared experts and other params are on different FSDP meshes
    }
    return (
        OptimizersContainer(model_parts, optimizer_kwargs, name)
        if not optim_in_bwd
        else OptimizersInBackwardContainer(model_parts, optimizer_kwargs, name)
    )
