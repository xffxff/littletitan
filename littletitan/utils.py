import torch

def get_activated_params(model: torch.nn.Module, model_config, exclude_embedding: bool = False) -> int:
    sparse_experts_params = sum(p.numel() for n, p in model.named_parameters() if "experts.experts" in n)
    non_sparse_experts_params = sum(p.numel() for n, p in model.named_parameters() if "experts.experts" not in n)
    activated_params = sparse_experts_params * model_config.moe_top_k / model_config.moe_num_experts + non_sparse_experts_params
    if exclude_embedding:
        activated_params -= model.tok_embeddings.weight.numel()
    return activated_params
