"""Input specification for true mixed prefill+decode attention profiling."""

from dataclasses import dataclass
from typing import List


@dataclass
class TrueMixedBatchInput:
    """Represents one true mixed attention batch (prefill + decode sequences).

    - Prefill sequences consume ``prefill_seq_len`` new tokens.
    - Decode sequences consume exactly one new token each with existing KV cache.
    """

    prefill_seq_lens: List[int]
    prefill_kv_cache_sizes: List[int]
    decode_kv_cache_sizes: List[int]

    def __post_init__(self) -> None:
        if not self.prefill_seq_lens:
            raise ValueError("prefill_seq_lens cannot be empty for true mixed batch")
        if not self.decode_kv_cache_sizes:
            raise ValueError("decode_kv_cache_sizes cannot be empty for true mixed batch")
        if len(self.prefill_seq_lens) != len(self.prefill_kv_cache_sizes):
            raise ValueError(
                "prefill_seq_lens and prefill_kv_cache_sizes must have the same length"
            )
        if any(x <= 0 for x in self.prefill_seq_lens):
            raise ValueError("All prefill_seq_lens must be positive")
        if any(x < 0 for x in self.prefill_kv_cache_sizes):
            raise ValueError("All prefill_kv_cache_sizes must be non-negative")
        if any(x < 0 for x in self.decode_kv_cache_sizes):
            raise ValueError("All decode_kv_cache_sizes must be non-negative")

    @property
    def num_prefill_seqs(self) -> int:
        return len(self.prefill_seq_lens)

    @property
    def num_decode_seqs(self) -> int:
        return len(self.decode_kv_cache_sizes)

    @property
    def decode_batch_size(self) -> int:
        return self.num_decode_seqs

    @property
    def total_batch_size(self) -> int:
        return self.num_prefill_seqs + self.num_decode_seqs

    @property
    def total_prefill_tokens(self) -> int:
        return int(sum(self.prefill_seq_lens))

    @property
    def total_decode_tokens(self) -> int:
        return int(self.num_decode_seqs)

    @property
    def total_tokens(self) -> int:
        return self.total_prefill_tokens + self.total_decode_tokens

    @property
    def decode_avg_kv_cache_size(self) -> int:
        if self.num_decode_seqs == 0:
            return 0
        return int(sum(self.decode_kv_cache_sizes) / self.num_decode_seqs)

    @property
    def batch_composition_ratio(self) -> float:
        if self.total_batch_size == 0:
            return 0.0
        return float(self.num_prefill_seqs) / float(self.total_batch_size)

    def is_valid(self, max_seq_len: int, max_batch_size: int) -> bool:
        if self.total_batch_size > max_batch_size:
            return False

        prefill_total_lens = [
            seq_len + kv_cache
            for seq_len, kv_cache in zip(
                self.prefill_seq_lens,
                self.prefill_kv_cache_sizes,
            )
        ]
        decode_total_lens = [kv_cache + 1 for kv_cache in self.decode_kv_cache_sizes]

        return (
            all(x <= max_seq_len for x in prefill_total_lens)
            and all(x <= max_seq_len for x in decode_total_lens)
        )

    def is_under_memory_limit(self, max_num_tokens: int) -> bool:
        total_with_cache = (
            self.total_prefill_tokens
            + sum(self.prefill_kv_cache_sizes)
            + self.total_decode_tokens
            + sum(self.decode_kv_cache_sizes)
        )
        return total_with_cache <= max_num_tokens

    def to_dict(self) -> dict:
        return {
            "prefill_seq_lens": self.prefill_seq_lens,
            "prefill_kv_cache_sizes": self.prefill_kv_cache_sizes,
            "decode_kv_cache_sizes": self.decode_kv_cache_sizes,
            "num_prefill_seqs": self.num_prefill_seqs,
            "num_decode_seqs": self.num_decode_seqs,
            "decode_batch_size": self.decode_batch_size,
            "total_batch_size": self.total_batch_size,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "total_tokens": self.total_tokens,
            "decode_avg_kv_cache_size": self.decode_avg_kv_cache_size,
            "batch_composition_ratio": self.batch_composition_ratio,
            "mode": "true_mixed",
            "is_true_mixed_batch": True,
            # Keep compatibility fields for downstream readers.
            "is_mixed_batch": False,
            "is_prefill": True,
            "batch_size": self.total_batch_size,
            "prefill_chunk_size": 0,
            "kv_cache_size": 0,
        }

    def __repr__(self) -> str:
        return (
            "TrueMixedBatchInput("
            f"num_prefill_seqs={self.num_prefill_seqs}, "
            f"num_decode_seqs={self.num_decode_seqs}, "
            f"total_tokens={self.total_tokens}, "
            f"decode_avg_kv_cache_size={self.decode_avg_kv_cache_size}"
            ")"
        )
