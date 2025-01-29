import torch

from littletitan.models.token_dispatcher import TokenDispatcher


def test_token_dispatcher_basic():
    seq_len = 8
    top_k = 2
    dim = 3
    dispatcher = TokenDispatcher(top_k=top_k)

    x = torch.arange(seq_len).unsqueeze(1).repeat(1, dim)
    indices = torch.tensor(
        [[0, 1], [2, 3], [1, 1], [0, 2], [3, 3], [2, 1], [1, 0], [1, 0]]
    )

    permuted_tokens = dispatcher.dispatch(x, indices)

    expected_output = torch.tensor(
        [
            [0, 0, 0],
            [3, 3, 3],
            [6, 6, 6],
            [7, 7, 7],
            [0, 0, 0],
            [2, 2, 2],
            [2, 2, 2],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [1, 1, 1],
            [3, 3, 3],
            [5, 5, 5],
            [1, 1, 1],
            [4, 4, 4],
            [4, 4, 4],
        ]
    )

    assert permuted_tokens.shape == (seq_len * top_k, dim)

    assert torch.allclose(permuted_tokens, expected_output)

    scores = torch.ones((seq_len, top_k)) / top_k

    combined_tokens = dispatcher.combine(permuted_tokens, scores)

    assert combined_tokens.shape == (seq_len, dim)

    assert torch.allclose(combined_tokens, x)


def test_token_dispatcher():
    batch_size = 2
    top_k = 6
    dim = 1024
    seq_len = 4096
    num_experts = 64
    device = "cuda"
    dispatcher = TokenDispatcher(top_k=top_k)
    x = torch.randn(batch_size, seq_len, dim, device=device)

    indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k), device=device)
    permuted_tokens = dispatcher.dispatch(x, indices)
    assert permuted_tokens.shape == (batch_size * seq_len * top_k, dim)

    scores = torch.ones((batch_size * seq_len, top_k), device=device) / top_k
    combined_tokens = dispatcher.combine(permuted_tokens, scores)
    assert combined_tokens.shape == (batch_size, seq_len, dim)

    assert torch.allclose(combined_tokens, x)
