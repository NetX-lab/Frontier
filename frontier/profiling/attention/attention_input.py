from math import ceil
from numbers import Integral
from typing import Iterable


def required_blocks_for_lengths(
    sequence_lengths: Iterable[int],
    *,
    block_size: int,
) -> int:
    """Return the total physical KV blocks for independent sequences.

    KV pages are allocated per sequence.  Summing token lengths before taking
    ``ceil`` undercounts fragmented batches, because every sequence consumes at
    least one block.  Callers must pass each sequence's *total* length,
    including any newly generated decode token.
    """

    if isinstance(block_size, bool) or not isinstance(block_size, Integral):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")

    required_blocks = 0
    for sequence_length in sequence_lengths:
        if isinstance(sequence_length, bool) or not isinstance(
            sequence_length, Integral
        ):
            raise ValueError(
                "sequence lengths must be positive integers, "
                f"got {sequence_length!r}."
            )
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError(
                "sequence lengths must be positive integers, "
                f"got {sequence_length!r}."
            )
        required_blocks += ceil(sequence_length / block_size)
    return required_blocks


def validate_profile_kv_capacity(
    *,
    max_num_blocks: int,
    profile_max_seq_len: int,
    block_size: int,
) -> int:
    """Validate physical KV capacity for the declared profiling domain."""

    if isinstance(max_num_blocks, bool) or not isinstance(max_num_blocks, Integral):
        raise ValueError(
            f"max_num_blocks must be a positive integer, got {max_num_blocks!r}."
        )
    if isinstance(profile_max_seq_len, bool) or not isinstance(
        profile_max_seq_len, Integral
    ) or int(profile_max_seq_len) <= 0:
        raise ValueError(
            "profile_max_seq_len must be a positive integer, "
            f"got {profile_max_seq_len!r}."
        )
    if isinstance(block_size, bool) or not isinstance(block_size, Integral) or int(block_size) <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    required_blocks = ceil(int(profile_max_seq_len) / int(block_size))
    if int(max_num_blocks) < required_blocks:
        raise ValueError(
            "Physical KV-cache capacity cannot cover profile_max_seq_len: "
            f"max_num_blocks={max_num_blocks}, "
            f"required_blocks_per_sequence={required_blocks}, "
            f"profile_max_seq_len={profile_max_seq_len}, "
            f"block_size={block_size}. Reduce the profiling range or allocate "
            "enough GPU memory."
        )
    return int(max_num_blocks)


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

    def is_valid(self, max_seq_len: int): 
        if self.is_prefill:
            if self.batch_size != 1:
                return False
            elif self.prefill_chunk_size == 0:
                return False
            elif self.prefill_chunk_size + self.kv_cache_size > max_seq_len:
                return False
        else:
            if self.prefill_chunk_size > 0:
                return False
            elif self.kv_cache_size < 0:
                return False
            elif self.kv_cache_size + 1 > max_seq_len:
                return False
        return True

    def required_blocks(self, *, block_size: int) -> int:
        """Return total KV blocks required by this batch.

        Prefill sequences consume ``prefill_chunk_size + kv_cache_size``
        tokens.  Decode sequences consume ``kv_cache_size + 1`` tokens because
        the new token must also have a KV slot.  Block allocation is performed
        independently for each sequence.
        """

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size!r}.")
        sequence_length = self.kv_cache_size + (
            self.prefill_chunk_size if self.is_prefill else 1
        )
        return required_blocks_for_lengths(
            [sequence_length] * self.batch_size,
            block_size=block_size,
        )

    def is_under_memory_limit(self, max_num_blocks: int, *, block_size: int) -> bool:
        """Return whether this batch fits the available physical KV blocks."""

        if isinstance(max_num_blocks, bool) or not isinstance(max_num_blocks, Integral):
            raise ValueError(
                "max_num_blocks must be a non-negative integer, "
                f"got {max_num_blocks!r}."
            )
        if int(max_num_blocks) < 0:
            raise ValueError(
                "max_num_blocks must be a non-negative integer, "
                f"got {max_num_blocks!r}."
            )
        return self.required_blocks(block_size=block_size) <= int(max_num_blocks)

    def __str__(self):
        return f"prefill_chunk_size: {self.prefill_chunk_size}, kv_cache_size: {self.kv_cache_size}, batch_size: {self.batch_size}, is_prefill: {self.is_prefill}"
