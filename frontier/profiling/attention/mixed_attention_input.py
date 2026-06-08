"""
Mixed-length batch attention input for profiling.

This module defines the input structure for profiling attention performance
with mixed-length sequences in a single batch.
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class MixedAttentionInput:
    """
    Input specification for mixed-length batch attention profiling.
    
    This class represents a batch of sequences with potentially different lengths,
    used to profile attention performance under realistic mixed-batch scenarios.
    
    Attributes:
        seq_lens: List of sequence lengths for each sequence in the batch.
                 e.g., [128, 256, 512] represents a batch of 3 sequences.
        kv_cache_size: Size of KV cache for all sequences (simplified to be uniform).
        mode: Generation mode label for the mixed-batch sample.
              Supported values:
                - "even": all sequences have same length
                - "random": randomized heterogeneous lengths
                - "online_grid_balanced": deterministic balanced shape for online grid
                - "online_grid_skewed": deterministic skewed shape for online grid
    """
    
    seq_lens: List[int]
    kv_cache_size: int = 0
    mode: str = "even"
    
    def __post_init__(self):
        """Validate and normalize input after initialization."""
        if not self.seq_lens:
            raise ValueError("seq_lens cannot be empty")
        
        if any(s <= 0 for s in self.seq_lens):
            raise ValueError("All sequence lengths must be positive")
        
        if self.kv_cache_size < 0:
            raise ValueError("kv_cache_size cannot be negative")
        
        valid_modes = [
            "even",
            "random",
            "online_grid_balanced",
            "online_grid_skewed",
        ]
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got '{self.mode}'"
            )
    
    @property
    def batch_size(self) -> int:
        """Number of sequences in the batch."""
        return len(self.seq_lens)
    
    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all sequences."""
        return sum(self.seq_lens)
    
    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length in the batch."""
        return max(self.seq_lens)
    
    @property
    def min_seq_len(self) -> int:
        """Minimum sequence length in the batch."""
        return min(self.seq_lens)
    
    @property
    def avg_seq_len(self) -> float:
        """Average sequence length in the batch."""
        return sum(self.seq_lens) / len(self.seq_lens)
    
    @property
    def equal_seq_len(self) -> int:
        """
        Equivalent sequence length using FLOP-based calculation.
        
        This is computed as sqrt(sum(s_i^2)) to preserve total FLOPs,
        since attention complexity is O(n^2) for sequence length n.
        
        This metric is used to compare mixed-batch performance with
        single-sequence performance that has equivalent computational cost.
        """
        return int(np.sqrt(sum(x**2 for x in self.seq_lens)))
    
    @property
    def seq_len_variance(self) -> float:
        """
        Variance of sequence lengths.
        
        Measures how spread out the sequence lengths are.
        Higher variance indicates more heterogeneity in the batch.
        """
        if len(self.seq_lens) == 1:
            return 0.0
        return float(np.var(self.seq_lens))
    
    @property
    def seq_len_std(self) -> float:
        """Standard deviation of sequence lengths."""
        return float(np.sqrt(self.seq_len_variance))
    
    @property
    def seq_len_cv(self) -> float:
        """
        Coefficient of variation (CV) of sequence lengths.
        
        CV = std / mean, a normalized measure of dispersion.
        Useful for comparing variability across different batch sizes.
        """
        avg = self.avg_seq_len
        if avg == 0:
            return 0.0
        return self.seq_len_std / avg
    
    def is_valid(self, max_seq_len: int, max_batch_size: int) -> bool:
        """
        Check if this input is valid for profiling.
        
        Args:
            max_seq_len: Maximum allowed sequence length.
            max_batch_size: Maximum allowed batch size.
        
        Returns:
            True if valid, False otherwise.
        """
        if self.batch_size == 0:
            return False
        
        if self.batch_size > max_batch_size:
            return False
        
        if self.max_seq_len > max_seq_len:
            return False
        
        # Each sequence must fit within max_seq_len including cache
        if any(s + self.kv_cache_size > max_seq_len for s in self.seq_lens):
            return False
        
        return True
    
    def is_under_memory_limit(self, max_num_tokens: int) -> bool:
        """
        Check if this input fits within memory constraints.
        
        Args:
            max_num_tokens: Maximum total tokens (including KV cache).
        
        Returns:
            True if within limits, False otherwise.
        """
        total_with_cache = self.total_tokens + self.batch_size * self.kv_cache_size
        return total_with_cache <= max_num_tokens
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"MixedAttentionInput("
            f"batch_size={self.batch_size}, "
            f"seq_lens={self.seq_lens}, "
            f"total_tokens={self.total_tokens}, "
            f"equal_len={self.equal_seq_len}, "
            f"variance={self.seq_len_variance:.1f}, "
            f"mode={self.mode})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"MixedAttentionInput("
            f"seq_lens={self.seq_lens}, "
            f"kv_cache_size={self.kv_cache_size}, "
            f"mode='{self.mode}')"
        )
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for CSV output.
        
        Returns:
            Dictionary with all relevant fields for profiling results.
        """
        return {
            "batch_size": self.batch_size,
            "seq_lens": self.seq_lens,
            "total_tokens": self.total_tokens,
            "max_seq_len": self.max_seq_len,
            "min_seq_len": self.min_seq_len,
            "avg_seq_len": self.avg_seq_len,
            "equal_seq_len": self.equal_seq_len,
            "seq_len_variance": self.seq_len_variance,
            "seq_len_std": self.seq_len_std,
            "seq_len_cv": self.seq_len_cv,
            "kv_cache_size": self.kv_cache_size,
            "mode": self.mode,
            "is_mixed_batch": True,
            "is_prefill": True,
        }
