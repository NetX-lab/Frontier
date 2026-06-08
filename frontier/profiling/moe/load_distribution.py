
"""
Load distribution generators for MoE profiling.

This module provides functions to generate different expert load distributions
for profiling MoE grouped GEMM performance under various load imbalance scenarios.

Supported distributions:
- uniform: All experts receive roughly equal number of tokens
- skewed: Some experts are more popular than others (power law distribution)
- extremely_skewed: Majority of tokens go to a small subset of experts (80-20 rule)
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


def generate_expert_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    load_distribution: str = "uniform",
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate expert routing data (topk_weights and topk_ids) with specified load distribution.
    
    This function simulates the output of a MoE router/gating network, producing
    routing decisions that result in different expert load distributions.
    
    Args:
        num_tokens: Number of input tokens
        num_experts: Total number of experts
        top_k: Number of experts selected per token
        load_distribution: Distribution type ("uniform", "skewed", "extremely_skewed")
        seed: Random seed for reproducibility
        dtype: Data type for routing weights
    
    Returns:
        topk_weights: Routing weights [num_tokens, top_k], normalized to sum to 1 per token
        topk_ids: Selected expert indices [num_tokens, top_k]
    
    Raises:
        ValueError: If load_distribution is not recognized
    
    Examples:
        >>> weights, ids = generate_expert_routing(1024, 8, 2, "uniform", seed=42)
        >>> weights.shape, ids.shape
        (torch.Size([1024, 2]), torch.Size([1024, 2]))
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # check top_k
    if top_k > num_experts:
        raise ValueError(
            f"top_k ({top_k}) cannot exceed num_experts ({num_experts}). "
            f"This typically happens when EP is too large. "
            f"Reduce EP so that num_experts_per_device >= router_topk."
        )
    
    if load_distribution == "uniform":
        # Uniform distribution: each expert has equal probability of being selected
        # Generate random scores and select top-k
        random_matrix = torch.rand(num_tokens, num_experts, device=device)
        topk_ids = torch.argsort(random_matrix, dim=1, descending=True)[:, :top_k].int()
        
        # Generate random weights and normalize
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
        
    elif load_distribution == "skewed":
        # Skewed distribution: some experts are more popular (power law)
        # Use sqrt(expert_id) as probability weight - earlier experts are more likely
        expert_probs = torch.pow(
            torch.arange(num_experts, dtype=torch.float, device=device), 
            0.5  # Power factor: 0.5 gives moderate skew
        )
        expert_probs = expert_probs / expert_probs.sum()
        
        # Sample experts according to probability distribution
        topk_ids = torch.multinomial(
            expert_probs.unsqueeze(0).expand(num_tokens, -1),
            top_k,
            replacement=False
        ).int()
        
        # Generate random weights and normalize
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
        
    elif load_distribution == "extremely_skewed":
        # Extremely skewed: 80% of tokens use only a small subset of experts
        # This simulates scenarios where certain experts become "hot"
        popular_experts = min(num_experts // 4, 8)  # Use at most 1/4 of experts or 8
        
        # 80% of tokens use popular experts, 20% use all experts
        use_popular = torch.rand(num_tokens, device=device) < 0.8
        
        # Generate routing for popular experts subset
        if popular_experts >= top_k:
            popular_random = torch.rand(num_tokens, popular_experts, device=device)
            popular_topk = torch.argsort(popular_random, dim=1, descending=True)[:, :top_k]
        else:
            # If popular subset is smaller than top_k, fall back to all experts
            popular_random = torch.rand(num_tokens, num_experts, device=device)
            popular_topk = torch.argsort(popular_random, dim=1, descending=True)[:, :top_k]
        
        # Generate routing for all experts
        all_random = torch.rand(num_tokens, num_experts, device=device)
        all_topk = torch.argsort(all_random, dim=1, descending=True)[:, :top_k]
        
        # Select which distribution to use based on the 80-20 split
        topk_ids = torch.where(
            use_popular.unsqueeze(1).expand(-1, top_k),
            popular_topk,
            all_topk
        ).int()
        
        # Generate random weights and normalize
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    
    else:
        raise ValueError(
            f"Unknown load_distribution: {load_distribution}. "
            f"Must be one of: uniform, skewed, extremely_skewed"
        )
    
    return topk_weights, topk_ids


def compute_expert_token_counts(
    topk_ids: torch.Tensor,
    num_experts: int
) -> List[int]:
    """
    Compute the number of tokens assigned to each expert from routing decisions.
    
    This function counts how many times each expert appears in the topk_ids tensor,
    which represents the load distribution across experts.
    
    Args:
        topk_ids: Selected expert indices [num_tokens, top_k]
        num_experts: Total number of experts
    
    Returns:
        expert_token_counts: List of token counts for each expert [num_experts]
    
    Examples:
        >>> topk_ids = torch.tensor([[0, 1], [1, 2], [0, 2]])  # 3 tokens, top_k=2
        >>> compute_expert_token_counts(topk_ids, num_experts=3)
        [2, 2, 2]  # Each expert gets 2 tokens
    """
    # Flatten topk_ids and count occurrences of each expert
    topk_ids_flat = topk_ids.flatten().cpu()
    counts = torch.bincount(topk_ids_flat, minlength=num_experts)
    return counts.numpy().astype(int).tolist()


def analyze_load_distribution(expert_token_counts: List[int]) -> dict:
    """
    Analyze the load distribution across experts and compute statistics.
    
    This function computes various metrics to characterize the load imbalance,
    including coefficient of variation, entropy, and Gini coefficient.
    
    Args:
        expert_token_counts: Number of tokens assigned to each expert
    
    Returns:
        Dictionary containing load distribution statistics:
        - mean_load: Average tokens per expert
        - std_load: Standard deviation of load
        - cv: Coefficient of variation (std/mean)
        - min_load: Minimum tokens assigned to any expert
        - max_load: Maximum tokens assigned to any expert
        - active_experts: Number of experts with non-zero load
        - utilization: Fraction of experts with non-zero load
        - entropy: Shannon entropy of load distribution
        - gini: Gini coefficient (0=perfect equality, 1=perfect inequality)
    
    Examples:
        >>> counts = [100, 100, 100, 100]  # Perfectly balanced
        >>> stats = analyze_load_distribution(counts)
        >>> stats['cv']  # Should be close to 0
        0.0
    """
    counts = np.array(expert_token_counts, dtype=float)
    num_experts = len(counts)
    total_tokens = counts.sum()
    
    # Basic statistics
    mean_load = counts.mean()
    std_load = counts.std()
    cv = std_load / mean_load if mean_load > 0 else 0.0
    
    # Active experts
    active_experts = np.sum(counts > 0)
    utilization = active_experts / num_experts if num_experts > 0 else 0.0
    
    # Entropy (measure of distribution uniformity)
    if total_tokens > 0:
        probs = counts / total_tokens
        probs = probs + 1e-12  # Avoid log(0)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
    else:
        entropy = 0.0
    
    # Gini coefficient (measure of inequality)
    if total_tokens > 0:
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * total_tokens) - (n + 1) / n
    else:
        gini = 0.0
    
    return {
        "mean_load": float(mean_load),
        "std_load": float(std_load),
        "cv": float(cv),
        "min_load": int(counts.min()),
        "max_load": int(counts.max()),
        "active_experts": int(active_experts),
        "utilization": float(utilization),
        "entropy": float(entropy),
        "gini": float(gini),
    }


if __name__ == "__main__":
    # Simple test to verify the functions work
    print("Testing load distribution generators...")
    
    for dist_type in ["uniform", "skewed", "extremely_skewed"]:
        print(f"\n{dist_type.upper()} distribution:")
        weights, ids = generate_expert_routing(
            num_tokens=1000,
            num_experts=8,
            top_k=2,
            load_distribution=dist_type,
            seed=42
        )
        
        counts = compute_expert_token_counts(ids, num_experts=8)
        stats = analyze_load_distribution(counts)
        
        print(f"  Expert token counts: {counts}")
        print(f"  CV: {stats['cv']:.3f}")
        print(f"  Gini: {stats['gini']:.3f}")
        print(f"  Entropy: {stats['entropy']:.3f}")
        print(f"  Utilization: {stats['utilization']:.3f}")

