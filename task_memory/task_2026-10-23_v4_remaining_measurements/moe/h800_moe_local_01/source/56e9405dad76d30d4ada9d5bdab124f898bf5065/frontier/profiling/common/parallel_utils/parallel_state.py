# Copyright 2023 The Sarathi team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Simplified parallel state management for profiling.

For profiling purposes, we use a simplified version that supports:
- Single-device profiling with simulated parallelism
- Optional multi-GPU profiling when torch.distributed is initialized
"""

from typing import Optional

import torch

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
) -> None:
    """Initialize model parallel groups for profiling.
    
    For profiling, we only support tensor parallelism.
    Pipeline parallelism is not needed for single-operation profiling.
    
    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
    """
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    
    # For single-device profiling, we simulate parallelism
    if not torch.distributed.is_initialized():
        _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = tensor_model_parallel_size
        _MPU_TENSOR_MODEL_PARALLEL_RANK = 0
        _TENSOR_MODEL_PARALLEL_GROUP = None
        return
    
    # For multi-GPU profiling, use actual distributed groups
    world_size: int = torch.distributed.get_world_size()
    rank: int = torch.distributed.get_rank()
    
    if world_size % tensor_model_parallel_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by "
            f"tensor_model_parallel_size ({tensor_model_parallel_size})"
        )
    
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    
    # Build the tensor model-parallel groups.
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _MPU_TENSOR_MODEL_PARALLEL_RANK = rank % tensor_model_parallel_size
            _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = tensor_model_parallel_size


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return 1


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return 0


def destroy_model_parallel():
    """Set the groups to none and reset ranks."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    
    _TENSOR_MODEL_PARALLEL_GROUP = None
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None

