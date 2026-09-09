"""Workload descriptions for stateful GDN profiling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GDNProfileInput:
    """One decode-first vLLM GDN batch.

    vLLM's common attention metadata places single-token decodes before
    multi-token prefills. Frontier makes that ordering explicit instead of
    silently rearranging caller input.
    """

    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.query_lens:
            raise ValueError("GDN profile input must contain at least one sequence")
        if len(self.query_lens) != len(self.context_lens):
            raise ValueError("query_lens and context_lens must have equal length")
        if any(type(length) is not int or length <= 0 for length in self.query_lens):
            raise ValueError(f"query_lens must be positive ints: {self.query_lens}")
        if any(
            type(length) is not int or length < 0 for length in self.context_lens
        ):
            raise ValueError(
                f"context_lens must be non-negative ints: {self.context_lens}"
            )

        seen_prefill = False
        for query_len in self.query_lens:
            is_decode = query_len == 1
            if not is_decode:
                seen_prefill = True
            elif seen_prefill:
                raise ValueError(
                    "GDN mixed batches must be decode-first: all query_len=1 "
                    "sequences must precede multi-token prefills"
                )

    @property
    def batch_size(self) -> int:
        return len(self.query_lens)

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def num_decode_tokens(self) -> int:
        return sum(query_len == 1 for query_len in self.query_lens)

    @property
    def num_prefill_tokens(self) -> int:
        return self.num_tokens - self.num_decode_tokens

    @property
    def phase(self) -> str:
        if self.num_decode_tokens == self.num_tokens:
            return "decode"
        if self.num_decode_tokens == 0:
            return "prefill"
        return "mixed"

    @property
    def has_initial_state(self) -> bool:
        return any(context_len > 0 for context_len in self.context_lens)

    @property
    def max_query_len(self) -> int:
        return max(self.query_lens)

    @property
    def state_block_ids(self) -> tuple[int, ...]:
        """Return request-state blocks with vLLM's null block reserved.

        vLLM defines cache block zero as ``NULL_BLOCK_ID``. Real request state
        must therefore start at block one; assigning a request to zero causes
        the causal-convolution kernel to skip it.
        """

        return tuple(range(1, self.batch_size + 1))

    @classmethod
    def prefill(
        cls,
        *,
        seq_len: int,
        batch_size: int = 1,
        context_len: int = 0,
    ) -> "GDNProfileInput":
        if seq_len <= 1:
            raise ValueError("GDN prefill seq_len must be greater than one")
        return cls(
            query_lens=(int(seq_len),) * int(batch_size),
            context_lens=(int(context_len),) * int(batch_size),
        )

    @classmethod
    def decode(
        cls,
        *,
        batch_size: int,
        context_len: int,
    ) -> "GDNProfileInput":
        if context_len <= 0:
            raise ValueError("GDN decode context_len must be positive")
        return cls(
            query_lens=(1,) * int(batch_size),
            context_lens=(int(context_len),) * int(batch_size),
        )

    @classmethod
    def mixed(
        cls,
        *,
        decode_batch_size: int,
        decode_context_len: int,
        prefill_seq_len: int,
        prefill_context_len: int = 0,
    ) -> "GDNProfileInput":
        return cls(
            query_lens=(1,) * int(decode_batch_size) + (int(prefill_seq_len),),
            context_lens=(int(decode_context_len),) * int(decode_batch_size)
            + (int(prefill_context_len),),
        )
