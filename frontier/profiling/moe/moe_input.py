"""
MoE profiling input parameter definitions.

This module defines the parameter space for MoE compute operations profiling.
Following the design principle: EP (expert_parallel_size) is a distribution parameter,
not a compute parameter, so we use num_experts_per_device instead.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


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


@dataclass
class MoELoadImbalanceInput:
    """
    Extended MoE input with load imbalance support.
    
    This class extends the basic MoE profiling parameters to include
    expert load distribution information, enabling profiling of grouped GEMM
    performance under various load imbalance scenarios.
    
    Backward compatible: If expert_token_counts is not provided, it automatically
    generates uniform distribution (preserving original behavior).
    """
    # Basic configuration (same as before)
    num_tokens: int
    num_experts_per_device: int
    hidden_dim: int
    expert_hidden_dim: int
    router_topk: int
    
    # Load distribution parameters (new)
    load_distribution: str = "uniform"  # uniform/skewed/extremely_skewed
    expert_token_counts: Optional[List[int]] = None  # If None, auto-generate uniform
    seed: Optional[int] = None
    
    # Tensor parallelism
    tensor_parallel_size: int = 1
    
    def __post_init__(self):
        """Auto-generate expert_token_counts if not provided (backward compatibility)."""
        if self.expert_token_counts is None:
            # Generate uniform distribution
            total_routed_tokens = self.num_tokens * self.router_topk
            tokens_per_expert = total_routed_tokens // self.num_experts_per_device
            remainder = total_routed_tokens % self.num_experts_per_device
            
            self.expert_token_counts = [tokens_per_expert] * self.num_experts_per_device
            # Distribute remainder tokens to first few experts
            for i in range(remainder):
                self.expert_token_counts[i] += 1
    
    # Core features (4)
    @property
    def total_routed_tokens(self) -> int:
        """Total number of tokens after top-k routing."""
        return sum(self.expert_token_counts)
    
    @property
    def tokens_per_expert_avg(self) -> float:
        """Average tokens per expert."""
        return self.total_routed_tokens / self.num_experts_per_device
    
    @property
    def model_expansion_ratio(self) -> float:
        """Expert hidden dim / model hidden dim."""
        return self.expert_hidden_dim / self.hidden_dim
    
    # Workload features (2)
    @property
    def tokens_to_experts_ratio(self) -> float:
        """Ratio of total tokens to number of experts (workload density)."""
        return self.total_routed_tokens / self.num_experts_per_device
    
    # Load distribution features (4)
    @property
    def expert_utilization(self) -> float:
        """Fraction of experts with non-zero load."""
        counts = np.array(self.expert_token_counts)
        active_experts = np.sum(counts > 0)
        return float(active_experts / self.num_experts_per_device)
    
    @property
    def min_load_ratio(self) -> float:
        """Minimum load / average load."""
        counts = np.array(self.expert_token_counts)
        mean = counts.mean()
        return float(counts.min() / mean) if mean > 0 else 0.0
    
    @property
    def load_imbalance_cv(self) -> float:
        """Coefficient of variation (CV) of expert loads."""
        counts = np.array(self.expert_token_counts)
        mean = counts.mean()
        std = counts.std()
        return float(std / mean) if mean > 0 else 0.0
    
    @property
    def max_load_ratio(self) -> float:
        """Maximum load / average load."""
        counts = np.array(self.expert_token_counts)
        mean = counts.mean()
        return float(counts.max() / mean) if mean > 0 else 0.0
    
    # Distribution statistics (2)
    @property
    def load_entropy(self) -> float:
        """Shannon entropy of load distribution (higher = more uniform)."""
        counts = np.array(self.expert_token_counts, dtype=float)
        total = counts.sum()
        if total == 0:
            return 0.0
        
        probs = counts / total
        probs = probs + 1e-12  # Avoid log(0)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)
    
    @property
    def load_gini_coefficient(self) -> float:
        """Gini coefficient of load distribution (0=perfect equality, 1=perfect inequality)."""
        counts = np.array(self.expert_token_counts, dtype=float)
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        total = sorted_counts.sum()
        
        if total == 0:
            return 0.0
        
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * total) - (n + 1) / n
        return float(gini)
    
    def to_features_dict(self) -> dict:
        """
        Export all features as a dictionary for training and prediction.
        
        Returns 15 features total:
        - 4 core features
        - 2 config features
        - 2 workload features
        - 4 load features
        - 2 distribution features
        - 1 metadata
        """
        return {
            # Core features (4)
            "total_routed_tokens": self.total_routed_tokens,
            "num_experts_per_device": self.num_experts_per_device,
            "hidden_dim": self.hidden_dim,
            "expert_hidden_dim": self.expert_hidden_dim,
            
            # Config features (2)
            "router_topk": self.router_topk,
            "model_expansion_ratio": self.model_expansion_ratio,
            
            # Workload features (2)
            "tokens_per_expert_avg": self.tokens_per_expert_avg,
            "tokens_to_experts_ratio": self.tokens_to_experts_ratio,
            
            # Load features (4)
            "expert_utilization": self.expert_utilization,
            "min_load_ratio": self.min_load_ratio,
            "load_imbalance_cv": self.load_imbalance_cv,
            "max_load_ratio": self.max_load_ratio,
            
            # Distribution features (2)
            "load_entropy": self.load_entropy,
            "load_gini_coefficient": self.load_gini_coefficient,
            
            # Metadata
            "load_distribution": self.load_distribution,
            "seed": self.seed,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"MoELoadImbalanceInput("
            f"num_tokens={self.num_tokens}, "
            f"num_experts={self.num_experts_per_device}, "
            f"distribution={self.load_distribution}, "
            f"cv={self.load_imbalance_cv:.3f}, "
            f"gini={self.load_gini_coefficient:.3f})"
    )

