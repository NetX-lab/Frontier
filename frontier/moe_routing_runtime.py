from __future__ import annotations

import pandas as pd
import math
from numbers import Real
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


def filter_moe_operator_rows(df, *, operator_facts, requested_runtime_path, source_name):
    """Admit an already bound physical scope without resolving another operator.

    Native row metadata is supplied by the caller from its selected contract;
    neither weight precision nor the presence of a timing column selects it.
    """
    path = validate_moe_gating_routing_runtime_path(requested_runtime_path)
    if path != "position_cyclic":
        return filter_moe_gating_routing_topk_rows(
            df, requested_runtime_path=path, source_name=source_name,
        ) if operator_facts["spec"].name == "moe_gating_routing_topk" else df
    from frontier.operators.families import MOE_FAMILY, get_moe_profiling_operators
    from frontier.operators.typed_contracts import parse_typed_operator_contracts, validate_typed_operator_metadata
    from frontier.moe_gating_runtime import validate_moe_gating_runtime_context

    spec, binding, contract = (operator_facts[key] for key in ("spec", "binding", "layer_contract"))
    context = operator_facts["native_context"]
    selected = get_moe_profiling_operators(context)
    if (context["model_architecture_profile"] != "step5" or context["routing_runtime_path"] != path
            or spec not in selected or binding.operator != spec or binding.family_id != "moe"):
        raise ValueError("MoE operator_facts do not bind an applicable Step physical scope")
    scope = operator_facts["frame_scope"]
    expected_scope = "source_local" if spec.ep_agnostic else "dp_gathered"
    if (scope != expected_scope or contract.tensor_parallel_mode != spec.tp_mode
            or (contract.expert_parallel_size is None) != spec.ep_agnostic):
        raise ValueError("MoE bound frame/TP/EP facts conflict with the selected spec")
    metadata = operator_facts["row_metadata"]
    if not metadata:
        raise ValueError("MoE position rows require selected native row metadata")
    required = {"model_architecture_profile", "routing_runtime_path", "moe_frame_scope",
                "typed_operator_contracts", "num_tokens", "live_routing_token_count",
                "masked_routing_token_count", "num_tensor_parallel_workers", "expert_parallel_size",
                "gating_runtime_context", "profiling_precision", *metadata}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"MoE physical scope metadata missing {missing}. Source: {source_name}")
    rows = df.loc[df["routing_runtime_path"].eq(path) & df["model_architecture_profile"].eq("step5")].copy()
    for name, expected in metadata.items():
        rows = rows.loc[rows[name].eq(expected)]
    rows = rows.loc[rows["num_tensor_parallel_workers"].eq(contract.tensor_parallel_size)]
    if contract.expert_parallel_size is not None:
        rows = rows.loc[rows["expert_parallel_size"].eq(contract.expert_parallel_size)]
    target = spec.profiling_name()
    for _, row in rows.iterrows():
        row_scope = row["moe_frame_scope"]
        if row_scope not in ("source_local", "dp_gathered"):
            raise ValueError("MoE row has an unknown physical frame")
        names = {op.profiling_name() for op in selected
                 if ("source_local" if op.ep_agnostic else "dp_gathered") == row_scope}
        typed = parse_typed_operator_contracts(row["typed_operator_contracts"])
        if set(typed) != names:
            raise ValueError("MoE typed targets conflict with the selected native physical frame")
        for name in {op.profiling_name() for op in MOE_FAMILY.operators if op.profiling_target}:
            for stat in ("median", "kernel_count"):
                value = row.get(f"time_stats.{name}.{stat}")
                if name not in names:
                    if value is not None and not pd.isna(value):
                        raise ValueError("MoE opposite-frame/inapplicable target must be absent, not zero")
                elif (not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value))
                      or value <= 0 or (stat == "kernel_count" and value != int(value))):
                    raise ValueError(f"MoE {name} requires positive paired {stat} coverage")
        physical, live, masked = (row[name] for name in
                                 ("num_tokens", "live_routing_token_count", "masked_routing_token_count"))
        if any(not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or value != int(value)
               or value < 0 for value in (physical, live, masked)) or physical <= 0 or physical != live + masked:
            raise ValueError("MoE physical rows must equal live plus masked rows")
        validate_moe_gating_runtime_context(row["gating_runtime_context"])
        if row_scope == scope:
            validate_typed_operator_metadata(typed[target], operator_name=target,
                                            expected_metadata=contract.typed_metadata_identity())
    rows = rows.loc[rows["moe_frame_scope"].eq(scope)]
    if spec.precision_name() == "moe_gating":
        rows = rows.loc[rows["gating_runtime_context"].eq(operator_facts["gating_runtime_context"])]
    if rows.empty:
        raise ValueError(f"MoE {target} has no measured {scope} coverage. Source: {source_name}")
    if rows["profiling_precision"].isna().any():
        raise ValueError("MoE selected physical rows lack source precision")
    if "source_profiling_precision" not in rows:
        rows["source_profiling_precision"] = rows["profiling_precision"]
    rows["profiling_precision"] = operator_facts["precision"].name
    return rows
