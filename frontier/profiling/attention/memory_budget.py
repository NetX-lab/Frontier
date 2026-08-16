"""Physical memory-budget helpers for attention profiling backends."""

from __future__ import annotations

import os
from collections.abc import Mapping
from math import ceil
from numbers import Integral


def _positive_environment_integer(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw_value = environ.get(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.")
    return value


def get_flashinfer_workspace_sizes_bytes(
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Return the float and integer workspace size for one FlashInfer wrapper."""

    source = os.environ if environ is None else environ
    workspace_gb = _positive_environment_integer(
        source,
        "FRONTIER_FLASHINFER_WORKSPACE_GB",
        4,
    )
    int_workspace_mb = _positive_environment_integer(
        source,
        "FRONTIER_FLASHINFER_INT_WORKSPACE_MB",
        512,
    )
    return workspace_gb * 1024**3, int_workspace_mb * 1024**2


def get_attention_backend_workspace_reservation_bytes(
    attention_backend: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Return bytes that must remain available before allocating the KV cache."""

    raw_backend = getattr(attention_backend, "value", attention_backend)
    backend = str(raw_backend).strip().upper()
    if backend == "FLASHINFER":
        workspace_bytes, int_workspace_bytes = get_flashinfer_workspace_sizes_bytes(
            environ=environ
        )
        # FlashInfer allocates one float and one integer workspace for prefill,
        # and another pair for decode.
        return 2 * (workspace_bytes + int_workspace_bytes)
    if backend in {"NO_OP", "FLASHINFER_MLA"}:
        return 0
    raise ValueError(
        f"Unsupported attention backend for memory budgeting: {attention_backend!r}."
    )


def resolve_requested_max_num_blocks(
    *,
    physical_max_num_blocks: int,
    requested_max_num_blocks: int | None,
    required_max_num_blocks: int | None = None,
    profile_max_seq_len: int,
    block_size: int,
) -> int:
    """Resolve an optional explicit KV allocation cap without clamping.

    ``physical_max_num_blocks`` is derived from free GPU memory after backend
    workspace reservation.  A user-provided cap selects a smaller allocation,
    but it may neither exceed that physical maximum nor fail to hold one
    sequence at ``profile_max_seq_len``.  Requested multi-sequence shapes are
    checked separately by the input-family capacity validators.
    """

    for name, value in (
        ("physical_max_num_blocks", physical_max_num_blocks),
        ("profile_max_seq_len", profile_max_seq_len),
        ("block_size", block_size),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if requested_max_num_blocks is not None:
        if (
            isinstance(requested_max_num_blocks, bool)
            or not isinstance(requested_max_num_blocks, Integral)
            or int(requested_max_num_blocks) <= 0
        ):
            raise ValueError(
                "requested_max_num_blocks must be a positive integer or None, "
                f"got {requested_max_num_blocks!r}."
            )
        requested = int(requested_max_num_blocks)
        if requested > int(physical_max_num_blocks):
            raise ValueError(
                "requested_max_num_blocks exceeds the computed physical maximum: "
                f"requested={requested}, physical maximum={physical_max_num_blocks}."
            )
    else:
        requested = int(physical_max_num_blocks)

    minimum = ceil(int(profile_max_seq_len) / int(block_size))
    if required_max_num_blocks is not None:
        if (
            isinstance(required_max_num_blocks, bool)
            or not isinstance(required_max_num_blocks, Integral)
            or int(required_max_num_blocks) <= 0
        ):
            raise ValueError(
                "required_max_num_blocks must be a positive integer or None, "
                f"got {required_max_num_blocks!r}."
            )
        minimum = max(minimum, int(required_max_num_blocks))
    if requested < minimum:
        raise ValueError(
            "requested_max_num_blocks cannot cover the declared profiling workload: "
            f"requested={requested}, required_max_num_blocks={minimum}, "
            f"profile_max_seq_len={profile_max_seq_len}, block_size={block_size}."
        )
    return requested
