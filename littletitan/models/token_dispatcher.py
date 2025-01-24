import torch
from torch.distributed.device_mesh import DeviceMesh
from typing import Optional


def print_rank(*args, rank=0):
    if torch.distributed.get_rank() == rank:
        print(*args)


def permute(x: torch.Tensor, indices: torch.Tensor):
    """
    Permute the tokens in x according to the indices.

    Args:
        x (torch.Tensor): shape (batch_size, seq_len, dim) or (batch_size * seq_len, dim)
        indices (torch.Tensor): shape (batch_size, seq_len, top_k) or (batch_size * seq_len, top_k), where each token is dispatched to an expert

    Returns:
        torch.Tensor: shape (batch_size * seq_len * top_k, dim)
    """
    assert (
        x.dim() == indices.dim()
    ), f"x.shape: {x.shape}, indices.shape: {indices.shape}"

    x = x.view(-1, x.size(-1))
    topk = indices.size(-1)
    flatten_indices = indices.flatten()
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = x.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


class TokenDispatcher:
    def __init__(self, top_k: int, num_experts: int):
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_state_shape = None
        self.reversed_input_permutation_mapping = None

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor):
        self.hidden_state_shape = x.shape
        permuted_tokens, sorted_indices = permute(x, indices)
        self.reversed_input_permutation_mapping = sorted_indices

        tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)
        return permuted_tokens, tokens_per_expert

    def combine(
        self, permuted_tokens: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        num_unpermuted_tokens = scores.numel()
        unpermuted_tokens = torch.zeros(
            (num_unpermuted_tokens, permuted_tokens.size(1)),
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        unpermuted_tokens.index_copy_(
            0, self.reversed_input_permutation_mapping, permuted_tokens
        )
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, self.top_k, permuted_tokens.size(1)
        )

        scores = scores.view(-1, self.top_k, 1)
        unpermuted_tokens = unpermuted_tokens * scores
        unpermuted_tokens = unpermuted_tokens.sum(dim=1).type_as(permuted_tokens)
        output = unpermuted_tokens.view(self.hidden_state_shape)
        return output


class EPTokenDispatcher:
    def __init__(
        self, top_k: int, num_experts: int, ep_mesh: Optional[DeviceMesh] = None
    ):
        self.top_k = top_k
        self.num_experts = num_experts
        self.ep_mesh = ep_mesh
        self.num_local_experts = num_experts // ep_mesh.size(0)
        assert self.num_local_experts * ep_mesh.size(0) == self.num_experts

        self.hidden_state_shape = None
        self.reversed_local_tokens_permutation_mapping = None
        self.reversed_global_tokens_permutation_mapping = None
        self.dispatch_output_split_sizes = None
        self.dispatch_input_split_sizes = None

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor):
        self.hidden_state_shape = x.shape
        permuted_tokens, sorted_indices = permute(x, indices)

        tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)
        global_tokens_per_expert = torch.zeros(
            tokens_per_expert.size(0), device=x.device, dtype=tokens_per_expert.dtype
        )
        torch.distributed.all_to_all_single(
            output=global_tokens_per_expert,
            input=tokens_per_expert,
            group=self.ep_mesh.get_group(),
            async_op=False,
        )

        global_tokens = torch.empty(
            (global_tokens_per_expert.sum(), x.size(-1)),
            device=x.device,
            dtype=x.dtype,
        )

        output_split_sizes = (
            global_tokens_per_expert.view(-1, self.num_local_experts).sum(-1).tolist()
        )
        input_split_sizes = (
            tokens_per_expert.view(-1, self.num_local_experts).sum(-1).tolist()
        )

        torch.distributed.all_to_all_single(
            output=global_tokens,
            input=permuted_tokens,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.ep_mesh.get_group(),
            async_op=False,
        )

        global_token_indices = (
            torch.arange(global_tokens_per_expert.size(0), device=x.device)
            % self.num_local_experts
        ).repeat_interleave(global_tokens_per_expert)

        global_tokens, sorted_global_token_indices = permute(
            global_tokens, global_token_indices.unsqueeze(-1)
        )

        self.reversed_local_tokens_permutation_mapping = sorted_indices
        self.reversed_global_tokens_permutation_mapping = sorted_global_token_indices
        self.dispatch_output_split_sizes = output_split_sizes
        self.dispatch_input_split_sizes = input_split_sizes

        tokens_per_expert = torch.bincount(
            global_token_indices.view(-1), minlength=self.num_local_experts
        )
        return global_tokens, tokens_per_expert

    def combine(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        all-to-all combine

        x: shape (tokens_dispatched_to_the_current_ep_rank, dim), tokens_dispatched_to_the_current_ep_rank
            is not a fixed number
        scores: shape (tokens_dispatched_to_the_current_ep_rank, top_k)
        """
        unpermuted_global_tokens = torch.zeros_like(x)

        unpermuted_global_tokens.index_copy_(
            0, self.reversed_global_tokens_permutation_mapping, x
        )

        local_tokens_num = sum(self.dispatch_input_split_sizes)
        permuted_local_tokens = torch.empty(
            (local_tokens_num, x.size(1)),
            device=x.device,
            dtype=x.dtype,
        )

        torch.distributed.all_to_all_single(
            output=permuted_local_tokens,
            input=unpermuted_global_tokens,
            output_split_sizes=self.dispatch_input_split_sizes,
            input_split_sizes=self.dispatch_output_split_sizes,
            group=self.ep_mesh.get_group(),
            async_op=False,
        )

        unpermuted_local_tokens = torch.zeros_like(permuted_local_tokens)
        unpermuted_local_tokens.index_copy_(
            0, self.reversed_local_tokens_permutation_mapping, permuted_local_tokens
        )

        unpermuted_local_tokens = unpermuted_local_tokens.reshape(
            -1, self.top_k, permuted_local_tokens.size(-1)
        )

        scores = scores.view(-1, self.top_k, 1)
        unpermuted_local_tokens = unpermuted_local_tokens * scores
        unpermuted_local_tokens = unpermuted_local_tokens.sum(dim=1).type_as(
            permuted_local_tokens
        )
        output = unpermuted_local_tokens.view(self.hidden_state_shape)
        return output
