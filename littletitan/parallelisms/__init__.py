# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.parallelisms.parallel_dims import ParallelDims
from torchtitan.parallelisms.parallelize_llama import parallelize_llama
from torchtitan.parallelisms.pipeline_llama import pipeline_llama


__all__ = [
    "models_parallelize_fns",
    "models_pipelining_fns",
    "ParallelDims",
]

models_parallelize_fns = {
    "moe": parallelize_llama,
}
models_pipelining_fns = {
    "moe": pipeline_llama,
}
