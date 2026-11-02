from math import ceil
from typing import Optional


class AttentionInput:
    def __init__(
        self,
        prefill_chunk_size: int,
        kv_cache_size: int,
        batch_size: int,
        is_prefill: bool,
    ):
        self.prefill_chunk_size = prefill_chunk_size
        self.kv_cache_size = kv_cache_size
        self.batch_size = batch_size
        self.is_prefill = is_prefill

    def is_valid(
        self,
        max_seq_len: int,
        max_model_len: Optional[int] = None,
    ):
        runtime_max_len = max_seq_len if max_model_len is None else max_model_len
        if self.is_prefill:
            if self.batch_size != 1:
                return False
            elif self.prefill_chunk_size == 0:
                return False
            elif self.prefill_chunk_size + self.kv_cache_size > max_seq_len:
                return False
            elif self.prefill_chunk_size + self.kv_cache_size > runtime_max_len:
                return False
        else:
            if self.prefill_chunk_size > 0:
                return False
            elif self.kv_cache_size < 0:
                return False
            elif self.kv_cache_size + 1 > runtime_max_len:
                return False
        return True

    def is_under_memory_limit(self, max_num_blocks: int, block_size: int):
        # The profiling wrapper gives every sequence its own blocks.
        current_tokens = self.prefill_chunk_size if self.is_prefill else 1
        blocks_per_sequence = ceil((self.kv_cache_size + current_tokens) / block_size)
        return self.batch_size * blocks_per_sequence <= max_num_blocks

    def __str__(self):
        return f"prefill_chunk_size: {self.prefill_chunk_size}, kv_cache_size: {self.kv_cache_size}, batch_size: {self.batch_size}, is_prefill: {self.is_prefill}"
