
from littletitan.models.model import Transformer, ModelArgs


moe_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "deepseek-moe-16b": ModelArgs(
        dim=2048,
        n_layers=12, # 28, set to 12 for local testing
        n_heads=16,
        n_kv_heads=16,
        multiple_of=128,
        max_seq_len=4096,
        moe_expert_dim=1408,
        moe_num_experts=64,
        moe_top_k=6,
        moe_num_shared_experts=2,
    )
}


models_config = {
    "moe": moe_configs,
}

model_name_to_cls = {"moe": Transformer}

model_name_to_tokenizer = {
    "moe": "tiktoken",
}
