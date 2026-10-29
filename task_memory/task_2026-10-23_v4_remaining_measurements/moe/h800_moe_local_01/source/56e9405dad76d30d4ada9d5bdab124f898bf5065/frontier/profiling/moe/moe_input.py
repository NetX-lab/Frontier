"""
MoE profiling input parameter definitions.

This module defines the parameter space for MoE compute operations profiling.
Following the design principle: EP (expert_parallel_size) is a distribution parameter,
not a compute parameter, so we use num_experts_per_device instead.
"""

import math
import operator
from dataclasses import dataclass
from typing import Iterable, List, Optional

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


def get_runtime_legal_expert_parallel_sizes(num_experts: int) -> List[int]:
    """Return every positive EP divisor accepted by the runtime topology check."""
    if isinstance(num_experts, bool):
        raise ValueError("num_experts must be a positive integer")
    try:
        normalized_num_experts = operator.index(num_experts)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("num_experts must be a positive integer") from exc
    if normalized_num_experts <= 0:
        raise ValueError(
            f"num_experts must be a positive integer, got {normalized_num_experts}"
        )

    divisors = set()
    for candidate in range(1, math.isqrt(normalized_num_experts) + 1):
        if normalized_num_experts % candidate != 0:
            continue
        divisors.add(candidate)
        divisors.add(normalized_num_experts // candidate)
    return sorted(divisors)


def resolve_moe_expert_parallel_sizes(
    num_experts: int,
    requested_sizes: Optional[Iterable[int]] = None,
) -> List[int]:
    """Resolve an optional EP selection against the runtime-legal domain."""
    legal_sizes = get_runtime_legal_expert_parallel_sizes(num_experts)
    if requested_sizes is None:
        return legal_sizes

    try:
        requested_values = list(requested_sizes)
    except TypeError as exc:
        raise ValueError(
            "expert_parallel_sizes must be an iterable of positive integer divisors"
        ) from exc

    if not requested_values:
        raise ValueError(
            "expert_parallel_sizes must contain at least one positive integer divisor"
        )

    resolved_sizes = []
    for value in requested_values:
        if isinstance(value, bool):
            raise ValueError(
                "expert_parallel_sizes must contain positive integer divisors; "
                f"got {value!r}"
            )
        try:
            normalized_value = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "expert_parallel_sizes must contain positive integer divisors; "
                f"got {value!r}"
            ) from exc
        if normalized_value not in legal_sizes:
            raise ValueError(
                f"expert_parallel_size={normalized_value} must be a positive divisor "
                f"of num_experts={num_experts}; legal values are {legal_sizes}"
            )
        resolved_sizes.append(normalized_value)

    return list(dict.fromkeys(resolved_sizes))


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
    if max_tokens > 0:
        num_tokens_list.append(max_tokens)
    num_tokens_list = sorted(set(num_tokens_list))
    
    # Expert configurations
    num_experts_list = [num_experts]  # Typically fixed per model
    router_topk_list = [router_topk]  # Typically fixed per model
    
    # Include every runtime-legal EP divisor so the profiling envelope is a
    # superset of the runtime topology domain.
    num_experts_per_device_list = [
        num_experts // expert_parallel_size
        for expert_parallel_size in get_runtime_legal_expert_parallel_sizes(num_experts)
    ]
    
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
