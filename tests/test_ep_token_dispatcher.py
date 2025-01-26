import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from littletitan.models.token_dispatcher import EPTokenDispatcher
from datetime import timedelta
import os


def test_ep_token_dispatcher_basic():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))

    seq_len = 8
    top_k = 2
    dim = 3
    num_experts = 4
    dispatcher = EPTokenDispatcher(top_k=top_k, num_experts=num_experts, ep_mesh=mesh)

    x = torch.arange(seq_len).unsqueeze(1).repeat(1, dim).to("cuda")
    indices = torch.tensor(
        [[0, 1], [2, 3], [1, 3], [0, 2], [3, 3], [2, 1], [1, 0], [1, 0]]
    ).to("cuda")

    permuted_tokens = dispatcher.dispatch(x, indices)

    if mesh.get_rank() == 1:
        expected_permuted_tokens = torch.tensor(
            [
                [1, 1, 1],
                [3, 3, 3],
                [5, 5, 5],
                [1, 1, 1],
                [3, 3, 3],
                [5, 5, 5],
                [1, 1, 1],
                [2, 2, 2],
                [4, 4, 4],
                [4, 4, 4],
                [1, 1, 1],
                [2, 2, 2],
                [4, 4, 4],
                [4, 4, 4],
            ]
        )
    else:
        expected_permuted_tokens = torch.tensor(
            [
                [0, 0, 0],
                [3, 3, 3],
                [6, 6, 6],
                [7, 7, 7],
                [0, 0, 0],
                [3, 3, 3],
                [6, 6, 6],
                [7, 7, 7],
                [0, 0, 0],
                [2, 2, 2],
                [5, 5, 5],
                [6, 6, 6],
                [7, 7, 7],
                [0, 0, 0],
                [2, 2, 2],
                [5, 5, 5],
                [6, 6, 6],
                [7, 7, 7],
            ]
        )
    assert torch.allclose(permuted_tokens, expected_permuted_tokens.cuda())

    scores = torch.ones((seq_len, top_k)) / top_k
    tokens = dispatcher.combine(permuted_tokens, scores.cuda())

    torch.distributed.destroy_process_group()

    assert torch.allclose(tokens, x)


def test_ep_token_dispatcher():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size = 2
    top_k = 6
    dim = 1024
    seq_len = 4096
    num_experts = 64
    device = "cuda"

    # torch.random.manual_seed(42)

    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=5))
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    mesh = init_device_mesh(device, (2,), mesh_dim_names=("ep",))

    dispatcher = EPTokenDispatcher(top_k=top_k, num_experts=num_experts, ep_mesh=mesh)
    x = torch.randn(batch_size, seq_len, dim, device=device)

    indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k), device=device)
    permuted_tokens, _ = dispatcher.dispatch(x, indices)
    assert (
        permuted_tokens.shape[1] == dim
    )  # Only check the dimension size as the number of tokens can vary

    scores = torch.ones((batch_size * seq_len, top_k), device=device) / top_k
    combined_tokens = dispatcher.combine(permuted_tokens, scores)
    assert combined_tokens.shape == (batch_size, seq_len, dim)

    assert torch.allclose(
        combined_tokens, x, atol=1e-5
    )  # Allow a small tolerance for floating point operations
