"""Workload descriptions and schema constants for standard GDN profiling."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


_GDN_FEATURE_COLUMNS = (
    "measurement_type",
    "model_architecture_profile",
    "quant_signature",
    "device",
    "runtime_stack_signature",
    "gdn_runtime_backend",
    "gdn_rank_aggregation",
    "gdn_prefill_backend",
    "gdn_decode_backend",
    "gqa_interleaved_layout",
    "packed_recurrent_decode",
    "model_dtype",
    "conv_state_dtype",
    "recurrent_state_dtype",
    "num_tensor_parallel_workers",
    "hidden_size",
    "conv_kernel_size",
    "key_head_dim",
    "value_head_dim",
    "num_key_heads",
    "num_value_heads",
    "batch_size",
    "batch_num_tokens",
    "batch_num_prefill_tokens",
    "batch_num_decode_tokens",
    "max_query_len",
    "has_initial_state",
)


def get_required_gdn_profiling_columns() -> tuple[str, ...]:
    """Return the stable raw GDN metadata/feature schema."""

    return _GDN_FEATURE_COLUMNS


@dataclass(frozen=True)
class GDNProfileInput:
    """One GDN workload with explicit per-request query/context lengths."""

    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    logical_phase: str
    prefill_request_mask: tuple[bool, ...] | None = None
    physical_batch_size: int | None = None

    def __post_init__(self) -> None:
        if not self.query_lens:
            raise ValueError("GDN profile input must contain at least one sequence")
        if len(self.query_lens) != len(self.context_lens):
            raise ValueError("query_lens and context_lens must have equal length")
        if any(type(length) is not int or length <= 0 for length in self.query_lens):
            raise ValueError(f"query_lens must be positive ints: {self.query_lens}")
        if any(type(length) is not int or length < 0 for length in self.context_lens):
            raise ValueError(
                f"context_lens must be non-negative ints: {self.context_lens}"
            )
        phase = str(self.logical_phase).strip().lower()
        if phase not in {"prefill", "decode", "mixed"}:
            raise ValueError(
                "logical_phase must be prefill, decode, or mixed, "
                f"got {self.logical_phase!r}"
            )
        object.__setattr__(self, "logical_phase", phase)

        mask = self.prefill_request_mask
        if mask is None:
            if phase == "prefill":
                mask = (True,) * len(self.query_lens)
            elif phase == "decode":
                mask = (False,) * len(self.query_lens)
            else:
                raise ValueError(
                    "mixed GDN inputs require an explicit prefill_request_mask"
                )
        else:
            mask = tuple(bool(value) for value in mask)
            if len(mask) != len(self.query_lens):
                raise ValueError(
                    "prefill_request_mask must match query_lens length"
                )
        if phase == "prefill" and not all(mask):
            raise ValueError("prefill phase requires every request to be marked prefill")
        if phase == "decode" and any(mask):
            raise ValueError("decode phase requires every request to be marked decode")
        if phase == "mixed" and (all(mask) or not any(mask)):
            raise ValueError("mixed phase requires both prefill and decode requests")
        if any(not is_prefill and query_len != 1 for is_prefill, query_len in zip(mask, self.query_lens)):
            raise ValueError(
                "decode requests must carry exactly one query token; provide an "
                "explicit prefill mask for one-token continuations"
            )
        object.__setattr__(self, "prefill_request_mask", mask)
        physical_batch_size = self.physical_batch_size
        if physical_batch_size is None:
            physical_batch_size = len(self.query_lens)
        if type(physical_batch_size) is not int or physical_batch_size < len(self.query_lens):
            raise ValueError(
                "physical_batch_size must be an int >= logical batch size, "
                f"got {physical_batch_size!r} for batch_size={len(self.query_lens)}"
            )
        object.__setattr__(self, "physical_batch_size", physical_batch_size)

    @property
    def batch_size(self) -> int:
        return len(self.query_lens)

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def num_decode_tokens(self) -> int:
        return sum(
            query_len
            for query_len, is_prefill in zip(
                self.query_lens, self.prefill_request_mask
            )
            if not is_prefill
        )

    @property
    def num_prefill_tokens(self) -> int:
        return self.num_tokens - self.num_decode_tokens

    @property
    def max_query_len(self) -> int:
        return max(self.query_lens)

    @property
    def query_len_cv(self) -> float:
        mean = self.num_tokens / self.batch_size
        if mean == 0:
            return 0.0
        variance = sum((length - mean) ** 2 for length in self.query_lens) / self.batch_size
        return sqrt(variance) / mean

    @property
    def num_stateful_requests(self) -> int:
        return sum(context_len > 0 for context_len in self.context_lens)

    @property
    def phase(self) -> str:
        return self.logical_phase

    @property
    def has_initial_state(self) -> bool:
        return self.num_stateful_requests > 0

    @property
    def prefill_mask(self) -> tuple[bool, ...]:
        """Identify requests whose current logical work is prefill."""

        return self.prefill_request_mask

    @property
    def state_init_mode(self) -> str:
        """Record state provenance without changing the timed workload."""

        return "primed_prefix" if self.has_initial_state else "zero"

    @property
    def state_block_ids(self) -> tuple[int, ...]:
        """Return vLLM state pages, reserving page zero as the null page."""

        return tuple(range(1, self.physical_batch_size + 1))

    def require_supported_phase(self) -> None:
        """Reject same-batch prefill/decode GDN execution."""

        if self.phase == "mixed":
            raise ValueError(
                "same-batch GDN prefill/decode mixed execution is unsupported"
            )

    @classmethod
    def prefill(
        cls, *, seq_len: int, batch_size: int = 1, context_len: int = 0
    ) -> "GDNProfileInput":
        if int(seq_len) <= 0:
            raise ValueError("GDN prefill seq_len must be positive")
        if int(batch_size) <= 0:
            raise ValueError("GDN prefill batch_size must be positive")
        return cls(
            query_lens=(int(seq_len),) * int(batch_size),
            context_lens=(int(context_len),) * int(batch_size),
            logical_phase="prefill",
        )

    @classmethod
    def decode(cls, *, batch_size: int, context_len: int) -> "GDNProfileInput":
        if int(batch_size) <= 0:
            raise ValueError("GDN decode batch_size must be positive")
        if int(context_len) <= 0:
            raise ValueError("GDN decode context_len must be positive")
        return cls(
            query_lens=(1,) * int(batch_size),
            context_lens=(int(context_len),) * int(batch_size),
            logical_phase="decode",
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
            logical_phase="mixed",
            prefill_request_mask=(False,) * int(decode_batch_size) + (True,),
        )


def build_profile_inputs(
    *,
    prefill_lengths: tuple[int, ...] = (128, 512, 2048),
    decode_batch_sizes: tuple[int, ...] = (1, 8, 32),
    decode_context_lengths: tuple[int, ...] = (128, 2048, 8192),
) -> tuple[GDNProfileInput, ...]:
    """Build deterministic cold/hot prefill and decode inputs for CPU planning."""

    if len(decode_batch_sizes) != len(decode_context_lengths):
        raise ValueError("decode batch sizes and contexts must have equal length")
    inputs = [
        GDNProfileInput.prefill(seq_len=int(seq_len), context_len=0)
        for seq_len in prefill_lengths
    ]
    inputs.extend(
        GDNProfileInput.prefill(seq_len=int(seq_len), context_len=int(seq_len))
        for seq_len in prefill_lengths
    )
    inputs.extend(
        GDNProfileInput.decode(
            batch_size=int(batch_size), context_len=int(context_len)
        )
        for batch_size, context_len in zip(
            decode_batch_sizes, decode_context_lengths
        )
    )
    return tuple(inputs)
