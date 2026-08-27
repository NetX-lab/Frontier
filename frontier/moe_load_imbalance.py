"""Runtime-owned MoE load-imbalance feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MoELoadImbalanceInput:
    """Feature input shared by runtime prediction and offline MoE profiling.

    The class contains deterministic feature derivation only. Profiling
    sampling configuration and benchmark execution remain in
    ``frontier.profiling.moe``.
    """

    num_tokens: int
    num_experts_per_device: int
    hidden_dim: int
    expert_hidden_dim: int
    router_topk: int
    load_distribution: str = "uniform"
    expert_token_counts: Optional[List[int]] = None
    seed: Optional[int] = None
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        """Generate a uniform routing distribution when counts are omitted."""
        if self.expert_token_counts is None:
            total_routed_tokens = self.num_tokens * self.router_topk
            tokens_per_expert = total_routed_tokens // self.num_experts_per_device
            remainder = total_routed_tokens % self.num_experts_per_device
            self.expert_token_counts = [
                tokens_per_expert
            ] * self.num_experts_per_device
            for index in range(remainder):
                self.expert_token_counts[index] += 1

    @property
    def total_routed_tokens(self) -> int:
        """Return the total number of tokens after top-k routing."""
        return sum(self.expert_token_counts)

    @property
    def tokens_per_expert_avg(self) -> float:
        """Return the average routed-token count per expert."""
        return self.total_routed_tokens / self.num_experts_per_device

    @property
    def model_expansion_ratio(self) -> float:
        """Return expert hidden dimension divided by model hidden dimension."""
        return self.expert_hidden_dim / self.hidden_dim

    @property
    def tokens_to_experts_ratio(self) -> float:
        """Return routed-token density per expert."""
        return self.total_routed_tokens / self.num_experts_per_device

    @property
    def expert_utilization(self) -> float:
        """Return the fraction of experts with non-zero load."""
        counts = np.array(self.expert_token_counts)
        active_experts = np.sum(counts > 0)
        return float(active_experts / self.num_experts_per_device)

    @property
    def min_load_ratio(self) -> float:
        """Return minimum expert load divided by average load."""
        counts = np.array(self.expert_token_counts)
        mean = counts.mean()
        return float(counts.min() / mean) if mean > 0 else 0.0

    @property
    def load_imbalance_cv(self) -> float:
        """Return the coefficient of variation of expert loads."""
        counts = np.array(self.expert_token_counts)
        mean = counts.mean()
        std = counts.std()
        return float(std / mean) if mean > 0 else 0.0

    @property
    def max_load_ratio(self) -> float:
        """Return maximum expert load divided by average load."""
        counts = np.array(self.expert_token_counts)
        mean = counts.mean()
        return float(counts.max() / mean) if mean > 0 else 0.0

    @property
    def load_entropy(self) -> float:
        """Return Shannon entropy of the expert-load distribution."""
        counts = np.array(self.expert_token_counts, dtype=float)
        total = counts.sum()
        if total == 0:
            return 0.0
        probabilities = counts / total
        probabilities = probabilities + 1e-12
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        return float(entropy)

    @property
    def load_gini_coefficient(self) -> float:
        """Return the Gini coefficient of expert loads."""
        counts = np.array(self.expert_token_counts, dtype=float)
        sorted_counts = np.sort(counts)
        num_experts = len(sorted_counts)
        total = sorted_counts.sum()
        if total == 0:
            return 0.0
        gini = (
            2 * np.sum((np.arange(num_experts) + 1) * sorted_counts)
        ) / (num_experts * total) - (num_experts + 1) / num_experts
        return float(gini)

    def to_features_dict(self) -> dict:
        """Return the feature vector consumed by grouped-GEMM predictors."""
        return {
            "total_routed_tokens": self.total_routed_tokens,
            "num_experts_per_device": self.num_experts_per_device,
            "hidden_dim": self.hidden_dim,
            "expert_hidden_dim": self.expert_hidden_dim,
            "router_topk": self.router_topk,
            "model_expansion_ratio": self.model_expansion_ratio,
            "tokens_per_expert_avg": self.tokens_per_expert_avg,
            "tokens_to_experts_ratio": self.tokens_to_experts_ratio,
            "expert_utilization": self.expert_utilization,
            "min_load_ratio": self.min_load_ratio,
            "load_imbalance_cv": self.load_imbalance_cv,
            "max_load_ratio": self.max_load_ratio,
            "load_entropy": self.load_entropy,
            "load_gini_coefficient": self.load_gini_coefficient,
            "load_distribution": self.load_distribution,
            "seed": self.seed,
        }

    def __str__(self) -> str:
        """Return a concise human-readable description."""
        return (
            f"MoELoadImbalanceInput("
            f"num_tokens={self.num_tokens}, "
            f"num_experts={self.num_experts_per_device}, "
            f"distribution={self.load_distribution}, "
            f"cv={self.load_imbalance_cv:.3f}, "
            f"gini={self.load_gini_coefficient:.3f})"
        )
