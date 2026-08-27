"""
MoE profiling input parameter definitions.

This module defines the parameter space for MoE compute operations profiling.
Following the design principle: EP (expert_parallel_size) is a distribution parameter,
not a compute parameter, so we use num_experts_per_device instead.
"""

from dataclasses import dataclass
from typing import List

from frontier.moe_load_imbalance import MoELoadImbalanceInput


@dataclass
class MoEProfilingConfig:
    """Configuration for MoE profiling parameters."""
    
    # Gating network parameters
    num_tokens_list: List[int]
    num_experts_list: List[int]  # Total number of experts
    router_topk_list: List[int]
    
    # Grouped GEMM parameters
    num_experts_per_device_list: List[int]  # Number of experts per device (= total_experts / EP)
    
    # Model dimensions
    hidden_dim: int
    expert_hidden_dim: int
    
    # Parallelism
    tensor_parallel_size_list: List[int]


def get_default_moe_profiling_config(
    max_tokens: int = 4096,
    num_experts: int = 8,
    router_topk: int = 2,
    hidden_dim: int = 4096,
    expert_hidden_dim: int = 11008,
) -> MoEProfilingConfig:
    """
    Get default MoE profiling configuration.
    
    Args:
        max_tokens: Maximum number of tokens to profile
        num_experts: Total number of experts in the model
        router_topk: Number of experts selected per token
        hidden_dim: Model hidden dimension
        expert_hidden_dim: Expert FFN hidden dimension
    
    Returns:
        MoEProfilingConfig with default parameter ranges
    """
    # Token range (similar to dense FFN profiling)
    num_tokens_list = (
        [1, 2, 4]
        + list(range(8, 1024, 8))
        + list(range(1024, 2 * 1024 + 1, 16))
        + list(range(2 * 1024, max_tokens + 1, 32))
    )
    num_tokens_list = [t for t in num_tokens_list if t <= max_tokens]
    num_tokens_list.sort()
    
    # Expert configurations
    num_experts_list = [num_experts]  # Typically fixed per model
    router_topk_list = [router_topk]  # Typically fixed per model
    
    # Number of experts per device (simulates different EP configurations)
    # EP=1: all experts on one device
    # EP=2: half experts per device
    # EP=4: quarter experts per device
    # EP=8: 1/8 experts per device
    num_experts_per_device_list = []
    for divisor in [1, 2, 4, 8]:
        if num_experts % divisor == 0:
            num_experts_per_device_list.append(num_experts // divisor)
    
    # Tensor parallelism configurations
    tensor_parallel_size_list = [1, 2, 4, 8]
    
    return MoEProfilingConfig(
        num_tokens_list=num_tokens_list,
        num_experts_list=num_experts_list,
        router_topk_list=router_topk_list,
        num_experts_per_device_list=num_experts_per_device_list,
        hidden_dim=hidden_dim,
        expert_hidden_dim=expert_hidden_dim,
        tensor_parallel_size_list=tensor_parallel_size_list,
    )

