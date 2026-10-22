from __future__ import annotations

import pandas as pd
from frontier.moe_routing_policy import ROUTING_COUNT_POLICIES, resolve_routing_count_policy


STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH = "standard_fused_topk"
UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH = "uniform_topk"
SUPPORTED_MOE_GATING_ROUTING_RUNTIME_PATHS = {policy.runtime_path for policy in ROUTING_COUNT_POLICIES.values()}


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
    return resolve_routing_count_policy(moe_routing_distribution_type).runtime_path


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
