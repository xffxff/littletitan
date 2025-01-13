
from littletitan.models.model import Transformer, ModelArgs


moe_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
}


models_config = {
    "moe": moe_configs,
}

model_name_to_cls = {"moe": Transformer}

model_name_to_tokenizer = {
    "moe": "tiktoken",
}
