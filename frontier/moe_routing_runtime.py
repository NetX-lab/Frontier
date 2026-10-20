from __future__ import annotations

import pandas as pd


STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH = "standard_fused_topk"
UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH = "uniform_topk"
SUPPORTED_MOE_GATING_ROUTING_RUNTIME_PATHS = {
    STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
    UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH,
}

# Only classified producer backends may change the standalone shuffle charge.
MOE_BACKEND_FUSES_PERMUTATION = {
    "FLASHINFER_CUTLASS": True,
    "MARLIN": False,
}


def moe_permutation_is_fused(df: pd.DataFrame) -> bool:
    """Return the selected expert rows' permutation ownership."""

    if "moe_native_backend" not in df.columns:
        return False
    backends = set(df["moe_native_backend"].dropna().astype(str))
    unknown = backends - MOE_BACKEND_FUSES_PERMUTATION.keys()
    if unknown:
        raise ValueError(f"Unclassified moe_native_backend values: {sorted(unknown)}")
    fused = {MOE_BACKEND_FUSES_PERMUTATION[name] for name in backends}
    if df["moe_native_backend"].isna().any():
        fused.add(False)
    if len(fused) != 1:
        raise ValueError("MoE rows mix fused and standalone permutation measurements")
    return fused.pop()


def validate_moe_gating_routing_runtime_path(requested_runtime_path: str) -> str:
    normalized_path = str(requested_runtime_path).strip()
    if normalized_path not in SUPPORTED_MOE_GATING_ROUTING_RUNTIME_PATHS:
        raise ValueError(
            f"Unsupported routing_runtime_path={requested_runtime_path!r}. "
            f"Expected one of {sorted(SUPPORTED_MOE_GATING_ROUTING_RUNTIME_PATHS)}."
        )
    return normalized_path


def resolve_moe_gating_routing_runtime_path(
    moe_routing_distribution_type: str,
) -> str:
    """Map the canonical routing distribution to profiling runtime metadata."""
    normalized_distribution = str(moe_routing_distribution_type).strip().lower()
    if normalized_distribution in {"balanced", "skewed", "zipf"}:
        return STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH
    if normalized_distribution == "random":
        return UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH
    raise ValueError(
        f"Unsupported moe_routing_distribution_type={moe_routing_distribution_type!r}. "
        "Expected 'balanced', 'random', 'skewed', or 'zipf'."
    )


def filter_moe_gating_routing_topk_rows(
    df: pd.DataFrame,
    *,
    requested_runtime_path: str,
    source_name: str,
) -> pd.DataFrame:
    normalized_runtime_path = validate_moe_gating_routing_runtime_path(
        requested_runtime_path
    )
    if "routing_runtime_path" not in df.columns:
        raise ValueError(
            "MoE routing-path metadata is missing for moe_gating_routing_topk, "
            f"but runtime requires routing_runtime_path={normalized_runtime_path!r}. "
            f"Source: {source_name}"
        )

    filtered_df = df[
        df["routing_runtime_path"].astype(str) == normalized_runtime_path
    ].copy()
    if len(filtered_df) == 0:
        available_paths = sorted(
            df["routing_runtime_path"].dropna().astype(str).unique().tolist()
        )
        raise ValueError(
            "No moe_gating_routing_topk profiling rows match the requested "
            f"routing_runtime_path={normalized_runtime_path!r}. "
            f"Available paths: {available_paths}. Source: {source_name}"
        )
    return filtered_df
