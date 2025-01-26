from torchtitan.parallelisms.pipeline_llama import pipeline_llama

from littletitan.parallelisms.parallel_dims import ParallelDims
from littletitan.parallelisms.parallelize_moe import parallelize_moe

__all__ = [
    "models_parallelize_fns",
    "models_pipelining_fns",
    "ParallelDims",
]

models_parallelize_fns = {
    "moe": parallelize_moe,
}
models_pipelining_fns = {
    "moe": pipeline_llama,
}
