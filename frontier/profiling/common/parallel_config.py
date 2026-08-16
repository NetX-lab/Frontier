"""Simple parallel configuration for profiling."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Iterable


# The release profiling database is materialized for these tensor-parallel
# sizes.  A caller must add matching profiling data and metadata before using
# another TP size; silently accepting an unsupported value only defers the
# failure until model loading or runtime lookup.
SUPPORTED_PROFILE_TP_SIZES: tuple[int, ...] = (1, 2, 4, 8)


def validate_prediction_min_kv_cache_size(
    value: object,
    *,
    argument_name: str = "prediction_min_kv_cache_size",
) -> int:
    """Validate the explicit lower bound for attention prediction grids."""

    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(
            f"{argument_name} must be a non-negative integer, got {value!r}."
        )
    return int(value)


def _cluster_name(cluster_type: object | None) -> str:
    """Normalize enum/string cluster labels without importing runtime config."""

    if cluster_type is None:
        return "monolithic"
    name = getattr(cluster_type, "name", None)
    return str(name if name is not None else cluster_type).strip().lower()


def validate_profile_tp_sizes(
    values: Iterable[int] | None,
    *,
    argument_name: str = "num_tensor_parallel_workers",
) -> list[int]:
    """Validate TP sizes accepted by the profile-backed compute release.

    The input is returned as a list to preserve the CLI ordering and duplicate
    handling used by existing profiling entrypoints.  Validation is strict:
    non-integral, non-positive, and TP sizes without checked-in profile data
    are rejected before any GPU work starts.
    """

    if values is None:
        raise ValueError(
            f"{argument_name} must be a non-empty list of supported TP sizes "
            f"{SUPPORTED_PROFILE_TP_SIZES}."
        )

    normalized = list(values)
    if not normalized:
        raise ValueError(
            f"{argument_name} must be a non-empty list of supported TP sizes "
            f"{SUPPORTED_PROFILE_TP_SIZES}."
        )

    invalid = [
        value
        for value in normalized
        if isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) not in SUPPORTED_PROFILE_TP_SIZES
    ]
    if invalid:
        raise ValueError(
            f"{argument_name} must contain only supported TP sizes "
            f"{SUPPORTED_PROFILE_TP_SIZES}; got invalid values {invalid}."
        )

    canonical = [int(value) for value in normalized]
    duplicates = sorted(
        {value for value in canonical if canonical.count(value) > 1}
    )
    if duplicates:
        raise ValueError(
            f"{argument_name} contains duplicate TP sizes {duplicates}; "
            "each supported TP size must be profiled once per invocation."
        )

    return canonical


def validate_profile_backed_runtime_tp_sizes(
    replica_config: object,
    *,
    cluster_type: object | None = None,
    model_is_moe: bool | None = None,
    enable_dummy_mode: bool = False,
    communication_only: bool = False,
) -> None:
    """Validate only the compute TP fields used by a runtime cluster.

    The checked-in compute profile database covers TP sizes ``1/2/4/8``.
    This guard is intentionally narrower than ``ReplicaConfig`` validation:
    communication-only/transfer clusters and dummy execution do not consume
    profile-backed compute rows, and PD-AF role clusters have disjoint
    attention/FFN fields (unused fields may legitimately be zero).
    """

    if enable_dummy_mode or communication_only:
        return

    cluster_name = _cluster_name(cluster_type)
    if cluster_name in {"trans", "transfer", "communication", "communication_only"}:
        return

    # ``model_is_moe`` is optional for lightweight callers.  If absent, validate
    # both fields because the caller has not declared which compute family it
    # will consume; this preserves fail-fast behavior for direct predictor use.
    is_moe = None if model_is_moe is None else bool(model_is_moe)
    use_attention = cluster_name in {
        "monolithic",
        "prefill",
        "decode",
        "decode_attn",
    }
    use_moe = cluster_name in {"monolithic", "prefill", "decode", "decode_ffn"}

    if cluster_name not in {
        "monolithic",
        "prefill",
        "decode",
        "decode_attn",
        "decode_ffn",
        "trans",
        "transfer",
        "communication",
        "communication_only",
    } and is_moe is None:
        # Lightweight/unit callers may use a symbolic cluster label without
        # model metadata.  Validate both profile-backed fields in that case.
        use_attention = True
        use_moe = True

    if is_moe is False and cluster_name in {"monolithic", "prefill", "decode"}:
        # Dense models use the attention TP for their linear/MLP profiles; the
        # MoE TP field is not a compute lookup dimension in these paths.
        use_moe = False
    elif is_moe is False and cluster_name == "decode_ffn":
        # Dense PD-AF DECODE_FFN still executes dense FFN linear profiles.  The
        # predictor keys those rows by attention TP; the MoE TP field is an
        # unused role value and must not be validated here.
        use_attention = True
        use_moe = False
    elif is_moe is True and cluster_name == "decode_attn":
        # DECODE_ATTN executes attention only; its MoE TP field is intentionally
        # unused (often zero in PD-AF configs).
        use_moe = False

    if use_attention:
        validate_profile_tp_sizes(
            [getattr(replica_config, "attn_tensor_parallel_size", None)],
            argument_name=f"{cluster_name} attn_tensor_parallel_size",
        )
    if use_moe:
        validate_profile_tp_sizes(
            [getattr(replica_config, "moe_tensor_parallel_size", None)],
            argument_name=f"{cluster_name} moe_tensor_parallel_size",
        )


@dataclass
class ParallelConfig:
    """Simple parallel configuration for profiling.
    
    This is a lightweight version of Sarathi's ParallelConfig,
    containing only the fields needed for profiling operations.
    """
    
    pipeline_parallel_size: int = field(
        default=2, metadata={"help": "Number of pipeline parallel groups."}
    )
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of tensor parallel groups."}
    )

    def __post_init__(self):
        validate_profile_tp_sizes(
            [self.tensor_parallel_size], argument_name="tensor_parallel_size"
        )
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size
