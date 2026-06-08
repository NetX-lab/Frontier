"""Simplified sequence metadata for profiling.

This module provides a minimal SequenceMetadata implementation for profiling purposes.
It does not include the full Sequence object from Sarathi, only the essential fields
needed for attention profiling.
"""

from typing import List, Optional


class SimpleSequence:
    """Simplified sequence object for profiling.

    This is a minimal version of Sarathi's Sequence class, containing only
    the fields needed for attention profiling.
    """

    def __init__(
        self,
        seq_id: str,
        num_prompt_tokens_processed: int = 0,
        total_len: int = 0,
    ):
        self.seq_id = seq_id
        self._num_prompt_tokens_processed = num_prompt_tokens_processed
        self._total_len = total_len

    def get_num_prompt_tokens_processed(self) -> int:
        """Get the number of prompt tokens that have been processed."""
        return self._num_prompt_tokens_processed

    def get_len(self) -> int:
        """Get the total length of the sequence (prompt + generated tokens)."""
        return self._total_len


class SequenceMetadata:
    """Metadata for a sequence. Used for attention profiling.

    This is a simplified version of Sarathi's SequenceMetadata class,
    containing only the fields needed for attention profiling.

    Args:
        seq: The sequence object (or a simple proxy).
        block_table: The block table for the sequence (list of block indices).
        prompt_chunk_len: The size of the prompt chunk.
    """

    def __init__(
        self,
        seq: SimpleSequence,
        block_table: Optional[List[int]],
        prompt_chunk_len: int,
    ) -> None:
        self.seq = seq
        self.block_table = block_table
        self.prompt_chunk_len = prompt_chunk_len

    @property
    def num_prompt_tokens(self) -> int:
        """Get the number of prompt tokens in this chunk."""
        return self.prompt_chunk_len

    @property
    def is_prompt(self) -> bool:
        """Check if this is a prompt (prefill) chunk."""
        return self.prompt_chunk_len > 0

    @property
    def num_output_tokens(self) -> int:
        """Get the number of output (decode) tokens."""
        if self.prompt_chunk_len > 0:
            return 0
        return 1

    @property
    def num_tokens(self) -> int:
        """Get the total number of tokens (prompt or decode)."""
        return max(self.prompt_chunk_len, 1)

    def __str__(self) -> str:
        return (
            f"SequenceMetadata(seq_id={self.seq.seq_id}, "
            f"prompt_chunk_len={self.prompt_chunk_len})"
        )

    def __repr__(self) -> str:
        return self.__str__()

