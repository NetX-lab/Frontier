#!/usr/bin/env python3
"""Compare aggregate per-operation latency between vLLM CUDA logs and Frontier op traces."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontier.config.model_config import BaseModelConfig


OP_NAME_MAP_BY_PROFILE: dict[str, dict[str, list[str]]] = {
    "dense": {
        "input_layernorm": ["input_layernorm"],
        "attn_pre_proj": ["attn_pre_proj"],
        "attn_rope": ["attn_rope"],
        "attn_kv_cache_save": ["attn_kv_cache_save"],
        "attn_prefill": ["attn_prefill"],
        "attn_decode": ["attn_decode"],
        "attn_post_proj": ["attn_post_proj"],
        "post_attention_layernorm": ["post_attention_layernorm"],
        "mlp_up_proj": ["mlp_up_proj"],
        "mlp_act": ["mlp_act"],
        "mlp_down_proj": ["mlp_down_proj"],
        "attn_post_proj_tp_allreduce": [
            "attn_post_proj_tp_allreduce",
            "attn_tensor_parallel_allreduce",
            "attn_tp_allreduce",
            "tensor_parallel_allreduce",
        ],
        "mlp_down_proj_tp_allreduce": [
            "mlp_down_proj_tp_allreduce",
            "mlp_tensor_parallel_allreduce",
            "mlp_tp_allreduce",
            "tensor_parallel_allreduce",
        ],
        "embedding_tp_allreduce": [
            "embedding_tp_allreduce",
            "tensor_parallel_allreduce",
        ],
    },
    "moe": {
        "input_layernorm": ["input_layernorm"],
        "attn_pre_proj": ["attn_pre_proj"],
        "attn_kv_cache_save": ["attn_kv_cache_save"],
        "attn_prefill": ["attn_prefill"],
        "attn_decode": ["attn_decode"],
        "attn_post_proj": ["attn_post_proj"],
        "post_attention_layernorm": ["post_attention_layernorm"],
        "moe_gating": ["moe_gating"],
        "moe_shuffling": ["moe_shuffling"],
        "moe_grouped_gemm": ["moe_grouped_gemm"],
        "moe_tensor_parallel_allreduce": [
            "moe_tensor_parallel_allreduce",
            "tensor_parallel_allreduce",
        ],
        "expert_parallel_allreduce": ["expert_parallel_allreduce"],
        "add": ["add"],
    },
}

VIRTUAL_FRONTIER_OP_COMPONENTS: dict[str, list[list[str]]] = {
    "moe_gating": [
        ["moe_gating"],
        ["moe_gating_linear", "moe_gating_routing_topk"],
        ["moe_gating_routing_topk"],
    ],
    "add": [
        ["add"],
        ["add_attn_residual", "add_ffn_residual"],
        ["add_ffn_residual"],
        ["add_attn_residual"],
    ],
}

FRONTIER_MODEL_LAYER_REPEATED_OPS = {
    "input_layernorm",
    "attn_pre_proj",
    "attn_rope",
    "attn_kv_cache_save",
    "attn_prefill",
    "attn_decode",
    "attn_post_proj",
    "post_attention_layernorm",
    "mlp_up_proj",
    "mlp_act",
    "mlp_down_proj",
    "attn_post_proj_tp_allreduce",
    "attn_tp_allreduce",
    "mlp_down_proj_tp_allreduce",
    "mlp_tp_allreduce",
    "add_attn_residual",
}

FRONTIER_MOE_LAYER_REPEATED_OPS = {
    "moe_gating",
    "moe_gating_linear",
    "moe_gating_routing_topk",
    "moe_shuffling",
    "moe_grouped_gemm",
    "moe_tensor_parallel_allreduce",
    "expert_parallel_allreduce",
    "add_ffn_residual",
}

FRONTIER_SINGLE_INVOCATION_OPS = {
    "embedding_tp_allreduce",
}


@lru_cache(maxsize=None)
def _load_model_config_by_name(model_name: str) -> BaseModelConfig:
    return BaseModelConfig.create_from_name(model_name)


def _frontier_full_layer_multiplier(record: dict[str, Any]) -> int:
    layer_id = _to_int(record.get("layer_id"), default=-1)
    if layer_id >= 0:
        return 1

    meta = record.get("meta", {}) or {}
    if not isinstance(meta, dict):
        return 1

    recorded_layer_depth = _to_int(meta.get("num_layers"), default=1)
    if recorded_layer_depth > 1:
        return 1

    batch_size, total_tokens = _normalize_frontier_bucket_signature(record)
    if batch_size <= 0 or total_tokens <= batch_size:
        return 1

    op_name = str(record.get("name") or "")
    if op_name in FRONTIER_SINGLE_INVOCATION_OPS:
        return 1

    model_name = str(meta.get("model_name") or "").strip()
    if not model_name:
        raise ValueError(
            f"missing model_name in Frontier trace event requiring layer multiplier: {record}"
        )
    model_config = _load_model_config_by_name(model_name)

    if op_name in FRONTIER_MOE_LAYER_REPEATED_OPS:
        num_moe_layers = int(model_config.get_num_moe_layers())
        if num_moe_layers <= 0:
            raise ValueError(
                f"model {model_name} has no MoE layers for Frontier op {op_name}: {record}"
            )
        return num_moe_layers

    if op_name in FRONTIER_MODEL_LAYER_REPEATED_OPS:
        num_layers = int(model_config.num_layers)
        if num_layers <= 0:
            raise ValueError(
                f"model {model_name} has invalid num_layers={num_layers} for Frontier op {op_name}: {record}"
            )
        return num_layers

    return 1


PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_MIXED = "mixed"
PHASE_UNKNOWN = "unknown"

FULL_LAYER_CONTEXT = "full_layer_total"
SINGLE_LAYER_CONTEXT = "single_layer_avg"

MEASUREMENT_SCOPE_ACTIONABILITY_ACTIONABLE = "ACTIONABLE"
MEASUREMENT_SCOPE_ACTIONABILITY_NON_ACTIONABLE_NESTED_SCOPE = (
    "NON_ACTIONABLE_NESTED_SCOPE"
)
FUSED_ADD_NORM_NESTED_OPS = {"add"}
FUSED_ADD_NORM_NESTED_SCOPE_REASON = (
    "vLLM records add as a nested scope inside fused add+norm kernels; "
    "it is diagnostic-only and must not be treated as an independent additive residual."
)
COLLECTIVE_WAIT_INCLUSIVE_SPREAD_THRESHOLD_MS = 0.5
COLLECTIVE_ALIGNMENT_MODE_KERNEL_ONLY = "kernel_only"
COLLECTIVE_ALIGNMENT_MODE_WAIT_INCLUSIVE = "wait_inclusive"
COLLECTIVE_ALIGNMENT_MODE_RUNTIME_WRAPPER = "runtime_wrapper"
COLLECTIVE_SCOPE_ACTIONABILITY_ACTIONABLE = "ACTIONABLE"
COLLECTIVE_SCOPE_ACTIONABILITY_NON_ACTIONABLE = "NON_ACTIONABLE_SCOPE_MISMATCH"
COLLECTIVE_COMPONENT_SEMANTICS_PER_SCOPE_VALID = "per_scope_valid"
COLLECTIVE_COMPONENT_SEMANTICS_DEGRADED_BATCH_SUM = "degraded_batch_sum"
COLLECTIVE_COMPONENT_ACTIONABILITY_ACTIONABLE = "ACTIONABLE"
COLLECTIVE_COMPONENT_ACTIONABILITY_NON_ACTIONABLE = "NON_ACTIONABLE_BATCH_SUM_ARTIFACT"
VLLM_BATCH_DUPLICATE_STAGE_TIMESTAMP_SLACK_S = 1e-3
COLLECTIVE_SCOPE_ACTIONABILITY_NON_ACTIONABLE_RUNTIME_WRAPPER = (
    "NON_ACTIONABLE_RUNTIME_WRAPPER_SCOPE"
)

_EP_SHARED_DOMAIN_COLLECTIVE_NAMES = {
    "expert_parallel_allreduce",
}

_TP_RUNTIME_WRAPPER_COLLECTIVE_NAMES = {
    "attn_post_proj_tp_allreduce",
    "attn_tensor_parallel_allreduce",
    "attn_tp_allreduce",
    "mlp_down_proj_tp_allreduce",
    "mlp_tensor_parallel_allreduce",
    "mlp_tp_allreduce",
    "embedding_tp_allreduce",
    "moe_tensor_parallel_allreduce",
    "tensor_parallel_allreduce",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-op-log", type=Path, required=True)
    parser.add_argument("--vllm-batch-log", type=Path, default=None)
    parser.add_argument(
        "--vllm-op-log-override",
        action="append",
        default=[],
        help=(
            "Override selected vLLM ops from a dedicated probe dir. "
            "Format: op_a,op_b=/path/to/probe_dir. The directory must contain "
            "vllm_cuda_ops.jsonl and vllm_batch_log.jsonl."
        ),
    )
    parser.add_argument(
        "--vllm-clean-batch-log",
        type=Path,
        default=None,
        help=(
            "Optional clean-run vLLM batch log used only for decode batch execution "
            "evidence, while --vllm-batch-log remains the op-log time-window anchor."
        ),
    )
    parser.add_argument("--frontier-op-traces", type=Path, required=True)
    parser.add_argument(
        "--model-profile",
        type=str,
        choices=sorted(OP_NAME_MAP_BY_PROFILE.keys()),
        default="dense",
    )
    parser.add_argument(
        "--fused-add-norm-scope",
        type=str,
        choices=["auto", "enabled", "disabled"],
        default="auto",
        help=(
            "Controls whether nested fused add+norm scopes such as MoE 'add' are "
            "excluded from additive per-op error closure. 'auto' enables this for "
            "the MoE profile used by Qwen3 MoE comparison cases."
        ),
    )
    parser.add_argument(
        "--schedule-summary-json",
        type=Path,
        default=None,
        help="Optional comparison_summary.json used for path-divergence comparability gating.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--threshold-percent", type=float, default=5.0)
    parser.add_argument(
        "--collective-bucket-mode",
        type=str,
        choices=["raw", "request_level"],
        default="raw",
        help=(
            "Collective bucket remap policy for decode bs>1,tok/request=1 cases. "
            "Default keeps raw collective buckets."
        ),
    )
    parser.add_argument(
        "--top-bucket-rows-in-md",
        type=int,
        default=60,
        help="Max bucket rows to render in markdown (full list always kept in JSON).",
    )
    return parser.parse_args()


def _to_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        return int(float(text))
    if isinstance(value, list):
        if not value:
            return default
        return _to_int(value[0], default=default)
    return default


def _safe_rel_error(vllm_value: float, frontier_value: float) -> float | None:
    if vllm_value <= 0:
        return None
    return abs(frontier_value - vllm_value) / vllm_value * 100.0


def _safe_delta_percent(vllm_count: int, frontier_count: int) -> float | None:
    if vllm_count <= 0:
        if frontier_count == 0:
            return 0.0
        return None
    return (frontier_count - vllm_count) / vllm_count * 100.0


def _is_collective_scope_name(op_name: str) -> bool:
    normalized = op_name.strip().lower()
    return (
        normalized.endswith("allreduce")
        or normalized.endswith("allgather")
        or normalized.endswith("reduce_scatter")
        or normalized.endswith("broadcast")
        or "alltoall" in normalized
    )


def _collective_domain_hint(op_name: str) -> str | None:
    normalized = op_name.strip().lower()
    if normalized in _EP_SHARED_DOMAIN_COLLECTIVE_NAMES:
        return "EP_SHARED_DOMAIN"
    if normalized in _TP_RUNTIME_WRAPPER_COLLECTIVE_NAMES:
        return "TP"
    return None


def _new_collective_component_metrics() -> dict[str, dict[str, float]]:
    return {
        "kernel_only": _new_metrics(),
        "related_wait": _new_metrics(),
        "wait_inclusive": _new_metrics(),
    }


def _finalize_collective_component_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | int | None]]:
    return {
        component: _finalize_metrics(component_metrics)
        for component, component_metrics in metrics.items()
    }


def _collective_component_metric_value(
    component_metrics: dict[str, dict[str, float | int | None]] | None,
    component: str,
    context: str,
) -> float | None:
    if not isinstance(component_metrics, dict):
        return None
    metrics = component_metrics.get(component)
    if not isinstance(metrics, dict):
        return None
    if context == FULL_LAYER_CONTEXT:
        value = metrics.get("full_layer_total_ms")
    elif context == SINGLE_LAYER_CONTEXT:
        value = metrics.get("single_layer_avg_ms")
    else:
        value = None
    return None if value is None else float(value)


def _vllm_collective_component_semantics(
    logical_invocation: dict[str, Any],
) -> str:
    aggregation_mode = str(logical_invocation.get("aggregation_mode") or "").strip()
    if aggregation_mode == "per_scope":
        return COLLECTIVE_COMPONENT_SEMANTICS_PER_SCOPE_VALID
    return COLLECTIVE_COMPONENT_SEMANTICS_DEGRADED_BATCH_SUM


def _build_vllm_collective_component_summary(
    logical_invocations: list[dict[str, Any]],
    *,
    request_level_decode_bucketing: bool = False,
) -> tuple[
    dict[str, dict[str, dict[str, float | int | None]]],
    dict[tuple[str, int, int], dict[str, dict[str, float | int | None]]],
]:
    total_metrics_raw: dict[str, dict[str, dict[str, float]]] = {}
    bucket_metrics_raw: dict[
        tuple[str, int, int], dict[str, dict[str, float]]
    ] = {}

    for logical_invocation in logical_invocations:
        op_name = str(logical_invocation.get("op_name", "")).strip()
        if not _is_collective_scope_name(op_name):
            continue

        duration_ms = float(logical_invocation["duration_ms"])
        component_semantics = _vllm_collective_component_semantics(logical_invocation)
        kernel_only_ms: float | None = None
        related_wait_ms: float | None = None
        if component_semantics == COLLECTIVE_COMPONENT_SEMANTICS_PER_SCOPE_VALID:
            kernel_only_ms = float(
                logical_invocation.get("rank_time_ms_min", duration_ms) or duration_ms
            )
            related_wait_ms = max(0.0, duration_ms - kernel_only_ms)
        layer_depth = int(logical_invocation["layer_depth"])

        component_metrics = total_metrics_raw.setdefault(
            op_name, _new_collective_component_metrics()
        )
        if kernel_only_ms is not None:
            _accumulate_metrics(component_metrics["kernel_only"], kernel_only_ms, layer_depth)
        if related_wait_ms is not None:
            _accumulate_metrics(component_metrics["related_wait"], related_wait_ms, layer_depth)
        _accumulate_metrics(
            component_metrics["wait_inclusive"], duration_ms, layer_depth
        )

        batch_size = int(logical_invocation["batch_size"])
        total_tokens = int(logical_invocation["total_tokens"])
        bucket_candidates: list[tuple[int, int, int]] = []
        if batch_size > 0 and total_tokens > 0:
            bucket_candidates.append((batch_size, total_tokens, 1))
            if (
                request_level_decode_bucketing
                and component_semantics == COLLECTIVE_COMPONENT_SEMANTICS_PER_SCOPE_VALID
                and str(logical_invocation.get("phase", PHASE_UNKNOWN)) == PHASE_DECODE
                and _to_int(logical_invocation.get("batch_num_prefill_tokens"), default=0)
                <= 0
                and _to_int(logical_invocation.get("batch_num_decode_tokens"), default=0)
                > 0
                and batch_size > 1
                and total_tokens % batch_size == 0
            ):
                per_request_tokens = total_tokens // batch_size
                if per_request_tokens > 0:
                    bucket_candidates = [(1, per_request_tokens, batch_size)]

        for bucket_batch_size, bucket_total_tokens, bucket_weight in bucket_candidates:
            bucket_key = (op_name, bucket_batch_size, bucket_total_tokens)
            bucket_components = bucket_metrics_raw.setdefault(
                bucket_key, _new_collective_component_metrics()
            )
            for _ in range(bucket_weight):
                if kernel_only_ms is not None:
                    _accumulate_metrics(
                        bucket_components["kernel_only"], kernel_only_ms, layer_depth
                    )
                if related_wait_ms is not None:
                    _accumulate_metrics(
                        bucket_components["related_wait"], related_wait_ms, layer_depth
                    )
                _accumulate_metrics(
                    bucket_components["wait_inclusive"], duration_ms, layer_depth
                )

    total_metrics = {
        op_name: _finalize_collective_component_metrics(metrics)
        for op_name, metrics in total_metrics_raw.items()
    }
    bucket_metrics = {
        bucket_key: _finalize_collective_component_metrics(metrics)
        for bucket_key, metrics in bucket_metrics_raw.items()
    }
    return total_metrics, bucket_metrics


def _frontier_collective_base_op_name(event: dict[str, Any]) -> str:
    meta = event.get("meta", {}) or {}
    if isinstance(meta, dict):
        base_op_name = str(meta.get("collective_base_op_name") or "").strip()
        if base_op_name:
            return base_op_name
    name = str(event.get("name") or "").strip()
    if name.endswith("_wait"):
        return name[: -len("_wait")]
    return name


def _build_frontier_collective_component_summary(
    trace_events: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, float | int | None]]],
    dict[tuple[str, int, int], dict[str, dict[str, float | int | None]]],
]:
    total_metrics_raw: dict[str, dict[str, dict[str, float]]] = {}
    bucket_metrics_raw: dict[
        tuple[str, int, int], dict[str, dict[str, float]]
    ] = {}
    wait_counts: dict[str, int] = defaultdict(int)
    wait_event_names: dict[str, set[str]] = defaultdict(set)

    for event in trace_events:
        if str(event.get("type") or "").strip() != "COMM":
            continue

        base_op_name = _frontier_collective_base_op_name(event)
        if not _is_collective_scope_name(base_op_name):
            continue

        meta = event.get("meta", {}) or {}
        component = "kernel_only"
        if (
            isinstance(meta, dict)
            and str(meta.get("collective_scope_component") or "").strip()
            == "related_wait"
        ) or str(event.get("name") or "").strip().endswith("_wait"):
            component = "related_wait"

        duration = float(event["duration_ms"])
        recorded_layer_depth = _frontier_layer_depth(event)
        full_layer_multiplier = _frontier_full_layer_multiplier(event)
        layer_depth = max(recorded_layer_depth, full_layer_multiplier)

        component_metrics = total_metrics_raw.setdefault(
            base_op_name, _new_collective_component_metrics()
        )
        _accumulate_metrics(
            component_metrics[component],
            duration,
            layer_depth,
            full_layer_multiplier=full_layer_multiplier,
        )

        batch_size, total_tokens = _normalize_frontier_bucket_signature(event)
        if batch_size > 0 and total_tokens > 0:
            bucket_key = (base_op_name, batch_size, total_tokens)
            bucket_components = bucket_metrics_raw.setdefault(
                bucket_key, _new_collective_component_metrics()
            )
            _accumulate_metrics(
                bucket_components[component],
                duration,
                layer_depth,
                full_layer_multiplier=full_layer_multiplier,
            )

        if component == "related_wait":
            wait_counts[base_op_name] += 1
            wait_event_names[base_op_name].add(str(event.get("name") or "").strip())

    total_metrics: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for op_name, component_metrics in total_metrics_raw.items():
        finalized = _finalize_collective_component_metrics(component_metrics)
        finalized["wait_inclusive"] = _merge_finalized_metrics(
            [finalized["kernel_only"], finalized["related_wait"]]
        )
        total_metrics[op_name] = finalized

    bucket_metrics: dict[
        tuple[str, int, int], dict[str, dict[str, float | int | None]]
    ] = {}
    for bucket_key, component_metrics in bucket_metrics_raw.items():
        finalized = _finalize_collective_component_metrics(component_metrics)
        finalized["wait_inclusive"] = _merge_finalized_metrics(
            [finalized["kernel_only"], finalized["related_wait"]]
        )
        bucket_metrics[bucket_key] = finalized

    ops: list[dict[str, Any]] = []
    for op_name in sorted(total_metrics):
        components = total_metrics[op_name]
        ops.append(
            {
                "op_name": op_name,
                "kernel_only_total_ms": float(
                    components["kernel_only"]["full_layer_total_ms"]
                ),
                "related_wait_total_ms": float(
                    components["related_wait"]["full_layer_total_ms"]
                ),
                "wait_inclusive_total_ms": float(
                    components["wait_inclusive"]["full_layer_total_ms"]
                ),
                "kernel_only_single_layer_avg_ms": components["kernel_only"][
                    "single_layer_avg_ms"
                ],
                "related_wait_single_layer_avg_ms": components["related_wait"][
                    "single_layer_avg_ms"
                ],
                "wait_inclusive_single_layer_avg_ms": components["wait_inclusive"][
                    "single_layer_avg_ms"
                ],
                "num_wait_rows": int(wait_counts.get(op_name, 0)),
                "wait_event_names": sorted(wait_event_names.get(op_name, set())),
            }
        )

    return {
        "status": "PASS",
        "num_collective_ops": len(ops),
        "ops": ops,
    }, total_metrics, bucket_metrics


def _resolve_collective_scope_alignment(
    *,
    vllm_collective_scope_audit: dict[str, Any],
    frontier_collective_component_summary: dict[str, Any],
) -> tuple[str, str, list[str]]:
    flagged_ops = [
        str(op_name)
        for op_name in vllm_collective_scope_audit.get("flagged_op_names", [])
    ]
    if not flagged_ops:
        return (
            COLLECTIVE_ALIGNMENT_MODE_KERNEL_ONLY,
            COLLECTIVE_SCOPE_ACTIONABILITY_ACTIONABLE,
            [],
        )

    frontier_wait_rows_by_op = {
        str(row.get("op_name")): int(row.get("num_wait_rows", 0))
        for row in frontier_collective_component_summary.get("ops", [])
        if isinstance(row, dict)
    }
    missing_wait_ops = [
        op_name for op_name in flagged_ops if frontier_wait_rows_by_op.get(op_name, 0) <= 0
    ]
    if missing_wait_ops:
        return (
            COLLECTIVE_ALIGNMENT_MODE_WAIT_INCLUSIVE,
            COLLECTIVE_SCOPE_ACTIONABILITY_NON_ACTIONABLE,
            missing_wait_ops,
        )
    return (
        COLLECTIVE_ALIGNMENT_MODE_WAIT_INCLUSIVE,
        COLLECTIVE_SCOPE_ACTIONABILITY_ACTIONABLE,
        [],
    )


def _resolve_row_collective_scope_contract(
    *,
    op_name: str,
    audit_row: dict[str, Any] | None,
    collective_scope_actionability: str,
    missing_frontier_wait_ops: list[str],
) -> tuple[str, str, bool]:
    if not _is_collective_scope_name(op_name) or audit_row is None:
        return (
            COLLECTIVE_ALIGNMENT_MODE_KERNEL_ONLY,
            COLLECTIVE_SCOPE_ACTIONABILITY_ACTIONABLE,
            False,
        )

    classification = str(audit_row.get("classification") or "")
    if classification == "WAIT_INCLUSIVE_DUPLICATED_RANK_SCOPE":
        return (
            COLLECTIVE_ALIGNMENT_MODE_WAIT_INCLUSIVE,
            collective_scope_actionability,
            op_name in missing_frontier_wait_ops,
        )
    if classification == "RUNTIME_WRAPPER_DUPLICATED_RANK_SCOPE":
        return (
            COLLECTIVE_ALIGNMENT_MODE_RUNTIME_WRAPPER,
            COLLECTIVE_SCOPE_ACTIONABILITY_NON_ACTIONABLE_RUNTIME_WRAPPER,
            False,
        )
    return (
        COLLECTIVE_ALIGNMENT_MODE_KERNEL_ONLY,
        COLLECTIVE_SCOPE_ACTIONABILITY_ACTIONABLE,
        False,
    )


def _augment_total_row_with_collective_alignment(
    row: dict[str, Any],
    *,
    collective_scope_by_op: dict[str, dict[str, Any]],
    vllm_collective_totals: dict[str, dict[str, dict[str, float | int | None]]],
    frontier_collective_totals: dict[str, dict[str, dict[str, float | int | None]]],
    collective_scope_actionability: str,
    missing_frontier_wait_ops: list[str],
) -> None:
    op_name = str(row.get("op_name", ""))
    mapped_op = str(row.get("mapped_frontier_op", "") or "")
    audit_row = collective_scope_by_op.get(op_name)

    (
        alignment_mode,
        row_actionability,
        missing_frontier_wait,
    ) = _resolve_row_collective_scope_contract(
        op_name=op_name,
        audit_row=audit_row,
        collective_scope_actionability=collective_scope_actionability,
        missing_frontier_wait_ops=missing_frontier_wait_ops,
    )

    row["collective_scope_alignment_mode"] = (
        alignment_mode if _is_collective_scope_name(op_name) else None
    )
    row["collective_scope_actionability"] = row_actionability
    row["collective_scope_missing_frontier_wait"] = missing_frontier_wait
    row["collective_scope_domain_hint"] = (
        audit_row.get("collective_domain_hint") if audit_row is not None else None
    )
    row["collective_scope_recommended_action"] = (
        audit_row.get("recommended_comparator_action") if audit_row is not None else None
    )
    row["collective_component_semantics"] = (
        audit_row.get("collective_component_semantics")
        if audit_row is not None and _is_collective_scope_name(op_name)
        else None
    )
    row["collective_component_actionability"] = (
        audit_row.get("component_comparison_actionability")
        if audit_row is not None and _is_collective_scope_name(op_name)
        else None
    )

    vllm_components = vllm_collective_totals.get(op_name)
    frontier_components = frontier_collective_totals.get(mapped_op)
    row["vllm_kernel_only_total_ms"] = _collective_component_metric_value(
        vllm_components, "kernel_only", FULL_LAYER_CONTEXT
    )
    row["vllm_related_wait_total_ms"] = _collective_component_metric_value(
        vllm_components, "related_wait", FULL_LAYER_CONTEXT
    )
    row["vllm_wait_inclusive_total_ms"] = _collective_component_metric_value(
        vllm_components, "wait_inclusive", FULL_LAYER_CONTEXT
    )
    row["frontier_kernel_only_total_ms"] = _collective_component_metric_value(
        frontier_components, "kernel_only", FULL_LAYER_CONTEXT
    )
    row["frontier_related_wait_total_ms"] = _collective_component_metric_value(
        frontier_components, "related_wait", FULL_LAYER_CONTEXT
    )
    row["frontier_wait_inclusive_total_ms"] = _collective_component_metric_value(
        frontier_components, "wait_inclusive", FULL_LAYER_CONTEXT
    )

    if alignment_mode == COLLECTIVE_ALIGNMENT_MODE_RUNTIME_WRAPPER:
        row["relative_error_percent"] = None
        return

    if alignment_mode != COLLECTIVE_ALIGNMENT_MODE_WAIT_INCLUSIVE:
        return

    vllm_metric = row.get("vllm_wait_inclusive_total_ms")
    frontier_metric = row.get("frontier_wait_inclusive_total_ms")
    if vllm_metric is None or frontier_metric is None:
        return

    vllm_metric_f = float(vllm_metric)
    frontier_metric_f = float(frontier_metric)
    row["vllm_total_ms"] = vllm_metric_f
    row["frontier_total_ms"] = frontier_metric_f
    row["relative_error_percent"] = _safe_rel_error(vllm_metric_f, frontier_metric_f)


def _augment_context_row_with_collective_alignment(
    row: dict[str, Any],
    *,
    collective_scope_by_op: dict[str, dict[str, Any]],
    vllm_collective_metrics: dict[Any, dict[str, dict[str, float | int | None]]],
    frontier_collective_metrics: dict[Any, dict[str, dict[str, float | int | None]]],
    collective_scope_actionability: str,
    missing_frontier_wait_ops: list[str],
) -> None:
    op_name = str(row.get("op_name", ""))
    mapped_op = str(row.get("mapped_frontier_op", "") or "")
    audit_row = collective_scope_by_op.get(op_name)

    (
        alignment_mode,
        row_actionability,
        missing_frontier_wait,
    ) = _resolve_row_collective_scope_contract(
        op_name=op_name,
        audit_row=audit_row,
        collective_scope_actionability=collective_scope_actionability,
        missing_frontier_wait_ops=missing_frontier_wait_ops,
    )

    row["collective_scope_alignment_mode"] = (
        alignment_mode if _is_collective_scope_name(op_name) else None
    )
    row["collective_scope_actionability"] = row_actionability
    row["collective_scope_missing_frontier_wait"] = missing_frontier_wait
    row["collective_scope_domain_hint"] = (
        audit_row.get("collective_domain_hint") if audit_row is not None else None
    )
    row["collective_scope_recommended_action"] = (
        audit_row.get("recommended_comparator_action") if audit_row is not None else None
    )
    row["collective_component_semantics"] = (
        audit_row.get("collective_component_semantics")
        if audit_row is not None and _is_collective_scope_name(op_name)
        else None
    )
    row["collective_component_actionability"] = (
        audit_row.get("component_comparison_actionability")
        if audit_row is not None and _is_collective_scope_name(op_name)
        else None
    )

    context = str(row.get("context") or FULL_LAYER_CONTEXT)
    vllm_key: Any = op_name
    frontier_key: Any = mapped_op
    if "batch_size" in row and "total_tokens" in row:
        batch_size = int(row.get("batch_size", 0) or 0)
        total_tokens = int(row.get("total_tokens", 0) or 0)
        vllm_key = (op_name, batch_size, total_tokens)
        frontier_key = (mapped_op, batch_size, total_tokens)

    vllm_components = vllm_collective_metrics.get(vllm_key)
    frontier_components = frontier_collective_metrics.get(frontier_key)
    row["vllm_kernel_only_metric_ms"] = _collective_component_metric_value(
        vllm_components, "kernel_only", context
    )
    row["vllm_related_wait_metric_ms"] = _collective_component_metric_value(
        vllm_components, "related_wait", context
    )
    row["vllm_wait_inclusive_metric_ms"] = _collective_component_metric_value(
        vllm_components, "wait_inclusive", context
    )
    row["frontier_kernel_only_metric_ms"] = _collective_component_metric_value(
        frontier_components, "kernel_only", context
    )
    row["frontier_related_wait_metric_ms"] = _collective_component_metric_value(
        frontier_components, "related_wait", context
    )
    row["frontier_wait_inclusive_metric_ms"] = _collective_component_metric_value(
        frontier_components, "wait_inclusive", context
    )

    if alignment_mode == COLLECTIVE_ALIGNMENT_MODE_RUNTIME_WRAPPER:
        row["relative_error_percent"] = None
        if "abs_gap_metric_ms" in row:
            row["abs_gap_metric_ms"] = None
        return

    if alignment_mode != COLLECTIVE_ALIGNMENT_MODE_WAIT_INCLUSIVE:
        return

    vllm_metric = row.get("vllm_wait_inclusive_metric_ms")
    frontier_metric = row.get("frontier_wait_inclusive_metric_ms")
    if vllm_metric is None or frontier_metric is None:
        return

    vllm_metric_f = float(vllm_metric)
    frontier_metric_f = float(frontier_metric)
    row["vllm_metric_ms"] = vllm_metric_f
    row["frontier_metric_ms"] = frontier_metric_f
    row["relative_error_percent"] = _safe_rel_error(vllm_metric_f, frontier_metric_f)
    if "abs_gap_metric_ms" in row:
        row["abs_gap_metric_ms"] = abs(frontier_metric_f - vllm_metric_f)


def _new_metrics() -> dict[str, float]:
    return {
        "full_layer_total_ms": 0.0,
        "full_layer_invocations": 0.0,
        "single_layer_invocations": 0.0,
    }


def _accumulate_metrics(
    metrics: dict[str, float],
    duration_ms: float,
    layer_depth: int,
    *,
    full_layer_multiplier: int = 1,
) -> None:
    if layer_depth <= 0:
        raise ValueError(f"invalid layer_depth={layer_depth}; expected positive integer")
    if full_layer_multiplier <= 0:
        raise ValueError(
            f"invalid full_layer_multiplier={full_layer_multiplier}; expected positive integer"
        )
    metrics["full_layer_total_ms"] += duration_ms * float(full_layer_multiplier)
    metrics["full_layer_invocations"] += 1.0
    metrics["single_layer_invocations"] += float(layer_depth)


def _finalize_metrics(metrics: dict[str, float]) -> dict[str, float | int | None]:
    full_total_ms = float(metrics["full_layer_total_ms"])
    full_invocations = int(metrics["full_layer_invocations"])
    single_invocations = int(metrics["single_layer_invocations"])
    single_avg_ms = None
    if single_invocations > 0:
        single_avg_ms = full_total_ms / single_invocations
    avg_layer_depth = None
    if full_invocations > 0:
        avg_layer_depth = single_invocations / full_invocations
    return {
        "full_layer_total_ms": full_total_ms,
        "full_layer_invocations": full_invocations,
        "single_layer_invocations": single_invocations,
        "single_layer_avg_ms": single_avg_ms,
        "avg_layer_depth": avg_layer_depth,
    }


def _metrics_to_raw(metrics: dict[str, float | int | None]) -> dict[str, float]:
    return {
        "full_layer_total_ms": float(metrics.get("full_layer_total_ms") or 0.0),
        "full_layer_invocations": float(metrics.get("full_layer_invocations") or 0.0),
        "single_layer_invocations": float(metrics.get("single_layer_invocations") or 0.0),
    }


def _merge_finalized_metrics(
    metrics_list: list[dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    merged = _new_metrics()
    full_invocations: list[float] = []
    single_invocations: list[float] = []
    for metrics in metrics_list:
        raw = _metrics_to_raw(metrics)
        merged["full_layer_total_ms"] += raw["full_layer_total_ms"]
        if raw["full_layer_invocations"] > 0:
            full_invocations.append(raw["full_layer_invocations"])
        if raw["single_layer_invocations"] > 0:
            single_invocations.append(raw["single_layer_invocations"])
    if full_invocations:
        merged["full_layer_invocations"] = max(full_invocations)
    if single_invocations:
        merged["single_layer_invocations"] = max(single_invocations)
    return _finalize_metrics(merged)


def _infer_phase(record: dict[str, Any], *, batch_size: int, total_tokens: int) -> str:
    prefill_tokens = _to_int(record.get("batch_num_prefill_tokens"), default=0)
    decode_tokens = _to_int(record.get("batch_num_decode_tokens"), default=0)
    if prefill_tokens > 0 and decode_tokens <= 0:
        return PHASE_PREFILL
    if decode_tokens > 0 and prefill_tokens <= 0:
        return PHASE_DECODE
    if prefill_tokens > 0 and decode_tokens > 0:
        return PHASE_MIXED
    if batch_size > 0 and total_tokens > batch_size:
        return PHASE_PREFILL
    if batch_size > 0 and total_tokens == batch_size:
        return PHASE_DECODE
    return PHASE_UNKNOWN


def _normalize_vllm_bucket_signature(record: dict[str, Any]) -> tuple[int, int]:
    batch_size = _to_int(record.get("batch_size"), default=0)
    if batch_size <= 0:
        request_tokens = record.get("batch_request_num_tokens")
        if isinstance(request_tokens, list) and request_tokens:
            batch_size = len(request_tokens)

    total_tokens = _to_int(record.get("batch_num_tokens"), default=0)
    if total_tokens <= 0:
        request_tokens = record.get("batch_request_num_tokens")
        if isinstance(request_tokens, list) and request_tokens:
            total_tokens = sum(_to_int(item, default=0) for item in request_tokens)

    return batch_size, total_tokens


def _frontier_layer_depth(record: dict[str, Any]) -> int:
    layer_id = _to_int(record.get("layer_id"), default=-1)
    if layer_id >= 0:
        return 1

    meta = record.get("meta", {}) or {}
    if not isinstance(meta, dict):
        return 1

    num_layers = _to_int(meta.get("num_layers"), default=1)
    if num_layers <= 0:
        raise ValueError(f"invalid num_layers={num_layers} in Frontier record: {record}")
    return num_layers


def _normalize_frontier_bucket_signature(record: dict[str, Any]) -> tuple[int, int]:
    meta = record.get("meta", {}) or {}
    if not isinstance(meta, dict):
        return 0, 0

    batch_size = 0
    request_ids = meta.get("request_ids")
    if isinstance(request_ids, list) and request_ids:
        batch_size = len(request_ids)
    if batch_size <= 0:
        num_tokens = meta.get("num_tokens")
        if isinstance(num_tokens, list) and num_tokens:
            batch_size = len(num_tokens)
    if batch_size <= 0:
        batch_size = _to_int(meta.get("batch_size"), default=0)

    total_tokens = 0
    for key in ("effective_total_tokens_compute", "effective_total_tokens_rounded", "total_tokens"):
        if key in meta:
            total_tokens = _to_int(meta.get(key), default=0)
            if total_tokens > 0:
                break

    return batch_size, total_tokens


def _load_jsonl_records(path: Path, *, missing_message: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{missing_message}: {path}")

    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid json at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"jsonl row must be an object at {path}:{line_number}")
        records.append(payload)
    return records


def _parse_vllm_op_override_specs(specs: list[str]) -> list[dict[str, Any]]:
    parsed_specs: list[dict[str, Any]] = []
    seen_ops: set[str] = set()
    for raw_spec in specs:
        spec = str(raw_spec).strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                "invalid --vllm-op-log-override; expected ops=/path/to/probe_dir"
            )
        ops_raw, probe_dir_raw = spec.split("=", 1)
        op_names = tuple(
            op_name.strip() for op_name in ops_raw.split(",") if op_name.strip()
        )
        if not op_names:
            raise ValueError(f"override spec contains no op names: {raw_spec!r}")
        duplicated = seen_ops.intersection(op_names)
        if duplicated:
            raise ValueError(
                f"duplicate vLLM override assignment for ops: {sorted(duplicated)}"
            )
        probe_dir = Path(probe_dir_raw).expanduser().resolve()
        if not probe_dir.is_dir():
            raise FileNotFoundError(f"override probe dir not found: {probe_dir}")
        op_log = probe_dir / "vllm_cuda_ops.jsonl"
        batch_log = probe_dir / "vllm_batch_log.jsonl"
        if not op_log.is_file():
            raise FileNotFoundError(f"override op log not found: {op_log}")
        if not batch_log.is_file():
            raise FileNotFoundError(f"override batch log not found: {batch_log}")
        parsed_specs.append(
            {
                "op_names": op_names,
                "probe_dir": probe_dir,
                "op_log": op_log,
                "batch_log": batch_log,
            }
        )
        seen_ops.update(op_names)
    return parsed_specs


def _apply_vllm_op_overrides(
    logical_invocations: list[dict[str, Any]],
    override_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not override_specs:
        return list(logical_invocations)

    overridden_ops = {
        op_name
        for spec in override_specs
        for op_name in spec["op_names"]
    }
    merged = [
        row for row in logical_invocations if str(row.get("op_name")) not in overridden_ops
    ]

    for spec in override_specs:
        override_invocations = _collect_vllm_logical_invocations(
            spec["op_log"],
            spec["batch_log"],
        )
        selected_rows = [
            row
            for row in override_invocations
            if str(row.get("op_name")) in spec["op_names"]
        ]
        if not selected_rows:
            raise ValueError(
                "override probe emitted no requested ops: "
                f"ops={list(spec['op_names'])} probe_dir={spec['probe_dir']}"
            )
        missing_ops = {
            op_name
            for op_name in spec["op_names"]
            if not any(str(row.get("op_name")) == op_name for row in selected_rows)
        }
        if missing_ops:
            raise ValueError(
                "override probe is missing requested ops: "
                f"{sorted(missing_ops)} probe_dir={spec['probe_dir']}"
            )
        merged.extend(selected_rows)
    return merged


def _resolve_vllm_batch_log_time_window(
    path: Path | None,
) -> tuple[float, float] | None:
    if path is None:
        return None

    timestamps: list[float] = []
    for event in _load_jsonl_records(path, missing_message="vLLM batch log not found"):
        if event.get("batch_execution_time_ms") is None:
            continue
        timestamp = event.get("timestamp")
        if timestamp is None:
            continue
        timestamps.append(float(timestamp))

    if not timestamps:
        return None
    return min(timestamps), max(timestamps)


def _collect_vllm_logical_invocations(
    path: Path,
    batch_log_path: Path | None = None,
) -> list[dict[str, Any]]:
    logical_invocations: dict[tuple[object, ...], dict[str, object]] = {}
    time_window = _resolve_vllm_batch_log_time_window(batch_log_path)
    time_slack_s = 1e-3

    for line_number, event in enumerate(
        _load_jsonl_records(path, missing_message="vLLM op log not found"),
        start=1,
    ):
        op_name = event.get("op_name")
        cuda_time_ms = event.get("cuda_time_ms")
        if op_name is None or cuda_time_ms is None:
            continue

        if time_window is not None:
            timestamp = event.get("timestamp")
            if timestamp is not None:
                event_time = float(timestamp)
                window_start, window_end = time_window
                if event_time < (window_start - time_slack_s) or event_time > (window_end + time_slack_s):
                    continue

        duration_ms = float(cuda_time_ms)
        layer_depth = _to_int(event.get("count"), default=1)
        if layer_depth <= 0:
            raise ValueError(f"invalid vLLM op count at {path}:{line_number}: {layer_depth}")

        name = str(op_name)
        batch_size, total_tokens = _normalize_vllm_bucket_signature(event)
        batch_id = _to_int(event.get("batch_id"), default=-1)
        batch_num_prefill_tokens = _to_int(event.get("batch_num_prefill_tokens"), default=0)
        batch_num_decode_tokens = _to_int(event.get("batch_num_decode_tokens"), default=0)
        aggregation_mode = str(event.get("aggregation_mode") or "").strip() or "batch_sum_legacy"
        scope_seq = event.get("scope_seq")
        normalized_scope_seq = None
        if aggregation_mode == "per_scope":
            if scope_seq is None:
                raise ValueError(
                    f"missing scope_seq for per_scope vLLM op row at {path}:{line_number}"
                )
            normalized_scope_seq = _to_int(scope_seq, default=-1)
            if normalized_scope_seq < 0:
                raise ValueError(
                    f"invalid scope_seq for per_scope vLLM op row at {path}:{line_number}: {scope_seq}"
                )
        pp_rank = event.get("pp_rank")
        normalized_pp_rank = None
        if pp_rank is not None:
            normalized_pp_rank = _to_int(pp_rank, default=-1)
            if normalized_pp_rank < 0:
                raise ValueError(
                    f"invalid pp_rank for vLLM op row at {path}:{line_number}: {pp_rank}"
                )

        logical_batch_id = batch_id if batch_id >= 0 else -line_number
        logical_key = (
            name,
            logical_batch_id,
            batch_size,
            total_tokens,
            batch_num_prefill_tokens,
            batch_num_decode_tokens,
            normalized_scope_seq,
            normalized_pp_rank,
        )
        logical_invocation = logical_invocations.setdefault(
            logical_key,
            {
                "op_name": name,
                "batch_id": logical_batch_id,
                "batch_size": batch_size,
                "total_tokens": total_tokens,
                "batch_num_prefill_tokens": batch_num_prefill_tokens,
                "batch_num_decode_tokens": batch_num_decode_tokens,
                "aggregation_mode": aggregation_mode,
                "scope_seq": normalized_scope_seq,
                "pp_rank": normalized_pp_rank,
                "duration_ms_sum": 0.0,
                "layer_depth_sum": 0.0,
                "row_count": 0,
                "duration_ms_values": [],
            },
        )
        if str(logical_invocation["aggregation_mode"]) != aggregation_mode:
            raise ValueError(
                "inconsistent vLLM aggregation mode across duplicated rows: "
                f"path={path} logical_key={logical_key}"
            )
        logical_invocation["duration_ms_sum"] = float(logical_invocation["duration_ms_sum"]) + duration_ms
        logical_invocation["layer_depth_sum"] = float(logical_invocation["layer_depth_sum"]) + float(layer_depth)
        logical_invocation["row_count"] = int(logical_invocation["row_count"]) + 1
        logical_invocation["duration_ms_values"].append(duration_ms)

    finalized_invocations: list[dict[str, Any]] = []
    for logical_invocation in logical_invocations.values():
        row_count = int(logical_invocation["row_count"])
        if row_count <= 0:
            raise ValueError(f"invalid vLLM logical row_count={row_count} in {path}")

        duration_ms = float(logical_invocation["duration_ms_sum"]) / float(row_count)
        duration_ms_values = [
            float(value) for value in logical_invocation.get("duration_ms_values", [])
        ]
        if len(duration_ms_values) != row_count:
            raise ValueError(
                "inconsistent duplicated-rank duration capture in vLLM op log: "
                f"path={path} logical_invocation={logical_invocation}"
            )
        avg_layer_depth = float(logical_invocation["layer_depth_sum"]) / float(row_count)
        rounded_layer_depth = int(round(avg_layer_depth))
        if abs(avg_layer_depth - float(rounded_layer_depth)) > 1e-6:
            raise ValueError(
                "inconsistent duplicated vLLM op counts across TP ranks: "
                f"avg_layer_depth={avg_layer_depth} path={path} logical_invocation={logical_invocation}"
            )
        if rounded_layer_depth <= 0:
            raise ValueError(
                f"invalid aggregated vLLM op count={rounded_layer_depth} path={path}"
            )

        batch_size = int(logical_invocation["batch_size"])
        total_tokens = int(logical_invocation["total_tokens"])
        phase = _infer_phase(
            {
                "batch_num_prefill_tokens": int(logical_invocation["batch_num_prefill_tokens"]),
                "batch_num_decode_tokens": int(logical_invocation["batch_num_decode_tokens"]),
            },
            batch_size=batch_size,
            total_tokens=total_tokens,
        )
        finalized_invocations.append(
            {
                "op_name": str(logical_invocation["op_name"]),
                "batch_id": int(logical_invocation["batch_id"]),
                "batch_size": batch_size,
                "total_tokens": total_tokens,
                "batch_num_prefill_tokens": int(logical_invocation["batch_num_prefill_tokens"]),
                "batch_num_decode_tokens": int(logical_invocation["batch_num_decode_tokens"]),
                "aggregation_mode": str(logical_invocation["aggregation_mode"]),
                "scope_seq": logical_invocation.get("scope_seq"),
                "pp_rank": logical_invocation.get("pp_rank"),
                "duration_ms": duration_ms,
                "layer_depth": rounded_layer_depth,
                "phase": phase,
                "duplicated_rank_rows": row_count,
                "rank_time_ms_mean": duration_ms,
                "rank_time_ms_min": min(duration_ms_values),
                "rank_time_ms_max": max(duration_ms_values),
                "rank_time_ms_spread": max(duration_ms_values) - min(duration_ms_values),
            }
        )

    if not finalized_invocations:
        raise ValueError(f"no op rows found in vLLM log: {path}")
    return finalized_invocations


def _build_vllm_collective_scope_audit(
    logical_invocations: list[dict[str, Any]],
) -> dict[str, Any]:
    ops: list[dict[str, Any]] = []
    flagged_op_names: list[str] = []
    runtime_wrapper_op_names: list[str] = []

    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in logical_invocations:
        op_name = str(row.get("op_name", "")).strip()
        if not _is_collective_scope_name(op_name):
            continue
        by_op[op_name].append(row)

    for op_name in sorted(by_op):
        op_rows = by_op[op_name]
        duplicated_rows = [
            row for row in op_rows if int(row.get("duplicated_rank_rows", 1) or 1) > 1
        ]
        component_semantics = COLLECTIVE_COMPONENT_SEMANTICS_PER_SCOPE_VALID
        if any(
            _vllm_collective_component_semantics(row)
            == COLLECTIVE_COMPONENT_SEMANTICS_DEGRADED_BATCH_SUM
            for row in op_rows
        ):
            component_semantics = COLLECTIVE_COMPONENT_SEMANTICS_DEGRADED_BATCH_SUM
        component_actionability = COLLECTIVE_COMPONENT_ACTIONABILITY_ACTIONABLE
        if component_semantics == COLLECTIVE_COMPONENT_SEMANTICS_DEGRADED_BATCH_SUM:
            component_actionability = COLLECTIVE_COMPONENT_ACTIONABILITY_NON_ACTIONABLE
        min_values = [float(row.get("rank_time_ms_min", row["duration_ms"])) for row in duplicated_rows]
        max_values = [float(row.get("rank_time_ms_max", row["duration_ms"])) for row in duplicated_rows]
        mean_values = [float(row.get("rank_time_ms_mean", row["duration_ms"])) for row in duplicated_rows]
        spread_values = [
            float(row.get("rank_time_ms_spread", 0.0) or 0.0) for row in duplicated_rows
        ]
        domain_hint = _collective_domain_hint(op_name)

        if duplicated_rows:
            mean_min = float(statistics.mean(min_values))
            mean_max = float(statistics.mean(max_values))
            mean_mean = float(statistics.mean(mean_values))
            mean_spread = float(statistics.mean(spread_values))
            max_spread = float(max(spread_values))
            classification = "DUPLICATED_RANK_SCOPE_BALANCED"
            alignment_mode = COLLECTIVE_ALIGNMENT_MODE_KERNEL_ONLY
            actionability = COLLECTIVE_SCOPE_ACTIONABILITY_ACTIONABLE
            if (
                mean_spread > COLLECTIVE_WAIT_INCLUSIVE_SPREAD_THRESHOLD_MS
                or max_spread > COLLECTIVE_WAIT_INCLUSIVE_SPREAD_THRESHOLD_MS
            ):
                if domain_hint == "TP":
                    classification = "RUNTIME_WRAPPER_DUPLICATED_RANK_SCOPE"
                    alignment_mode = COLLECTIVE_ALIGNMENT_MODE_RUNTIME_WRAPPER
                    actionability = (
                        COLLECTIVE_SCOPE_ACTIONABILITY_NON_ACTIONABLE_RUNTIME_WRAPPER
                    )
                    runtime_wrapper_op_names.append(op_name)
                else:
                    classification = "WAIT_INCLUSIVE_DUPLICATED_RANK_SCOPE"
                    alignment_mode = COLLECTIVE_ALIGNMENT_MODE_WAIT_INCLUSIVE
                    flagged_op_names.append(op_name)
            mean_related_wait = max(0.0, mean_mean - mean_min)
            mean_wait_inclusive = mean_mean
        else:
            mean_min = None
            mean_max = None
            mean_mean = None
            mean_spread = None
            max_spread = None
            classification = "NO_DUPLICATED_RANK_SCOPE"
            alignment_mode = COLLECTIVE_ALIGNMENT_MODE_KERNEL_ONLY
            actionability = COLLECTIVE_SCOPE_ACTIONABILITY_ACTIONABLE
            mean_related_wait = None
            mean_wait_inclusive = None

        top_skew_invocations = sorted(
            (
                {
                    "batch_id": int(row.get("batch_id", -1)),
                    "batch_size": int(row.get("batch_size", 0)),
                    "total_tokens": int(row.get("total_tokens", 0)),
                    "phase": str(row.get("phase", PHASE_UNKNOWN)),
                    "duplicated_rank_rows": int(row.get("duplicated_rank_rows", 1)),
                    "rank_time_ms_mean": float(row.get("rank_time_ms_mean", row["duration_ms"])),
                    "rank_time_ms_min": float(row.get("rank_time_ms_min", row["duration_ms"])),
                    "rank_time_ms_max": float(row.get("rank_time_ms_max", row["duration_ms"])),
                    "spread_ms": float(row.get("rank_time_ms_spread", 0.0) or 0.0),
                }
                for row in duplicated_rows
            ),
            key=lambda item: float(item["spread_ms"]),
            reverse=True,
        )[:10]

        ops.append(
            {
                "op_name": op_name,
                "classification": classification,
                "num_logical_invocations": len(op_rows),
                "num_duplicated_rank_invocations": len(duplicated_rows),
                "mean_rank_time_ms": mean_mean,
                "mean_min_rank_time_ms": mean_min,
                "mean_max_rank_time_ms": mean_max,
                "mean_spread_ms": mean_spread,
                "max_spread_ms": max_spread,
                "collective_scope_alignment_mode": alignment_mode,
                "collective_component_semantics": component_semantics,
                "component_comparison_actionability": component_actionability,
                "collective_scope_actionability": actionability,
                "collective_domain_hint": domain_hint,
                "mean_kernel_only_rank_time_ms": mean_min,
                "mean_related_wait_ms": mean_related_wait,
                "mean_wait_inclusive_scope_ms": mean_wait_inclusive,
                "top_skew_invocations": top_skew_invocations,
                "recommended_comparator_action": (
                    "exclude_runtime_wrapper_scope_from_direct_predictor_comparison"
                    if classification == "RUNTIME_WRAPPER_DUPLICATED_RANK_SCOPE"
                    else (
                        "do_not_compare_kernel_wait_components"
                        if component_actionability
                        == COLLECTIVE_COMPONENT_ACTIONABILITY_NON_ACTIONABLE
                        else (
                            "do_not_interpret_as_pure_kernel_time"
                            if classification == "WAIT_INCLUSIVE_DUPLICATED_RANK_SCOPE"
                            else None
                        )
                    )
                ),
            }
        )

    status = "NO_SPECIAL_COLLECTIVE_SCOPE_DETECTED"
    if flagged_op_names and runtime_wrapper_op_names:
        status = "MIXED_SPECIAL_COLLECTIVE_SCOPE_DETECTED"
    elif flagged_op_names:
        status = "WAIT_INCLUSIVE_SCOPE_DETECTED"
    elif runtime_wrapper_op_names:
        status = "RUNTIME_WRAPPER_SCOPE_DETECTED"

    return {
        "status": status,
        "spread_threshold_ms": COLLECTIVE_WAIT_INCLUSIVE_SPREAD_THRESHOLD_MS,
        "num_collective_ops": len(ops),
        "num_flagged_ops": len(flagged_op_names),
        "flagged_op_names": flagged_op_names,
        "num_runtime_wrapper_ops": len(runtime_wrapper_op_names),
        "runtime_wrapper_op_names": runtime_wrapper_op_names,
        "ops": ops,
    }


def _summarize_vllm_ops_from_logical_invocations(
    logical_invocations: list[dict[str, Any]],
    *,
    request_level_decode_bucketing: bool = False,
    collective_bucket_mode: str = "raw",
) -> tuple[
    dict[str, dict[str, float | int | None]],
    dict[tuple[str, int, int], dict[str, float | int | None]],
    dict[tuple[str, int, int], str],
]:
    total_metrics_raw: dict[str, dict[str, float]] = {}
    bucket_metrics_raw: dict[tuple[str, int, int], dict[str, float]] = {}
    bucket_phase_counts: dict[tuple[str, int, int], dict[str, int]] = {}

    for logical_invocation in logical_invocations:
        name = str(logical_invocation["op_name"])
        duration_ms = float(logical_invocation["duration_ms"])
        layer_depth = int(logical_invocation["layer_depth"])
        batch_size = int(logical_invocation["batch_size"])
        total_tokens = int(logical_invocation["total_tokens"])

        op_metrics = total_metrics_raw.setdefault(name, _new_metrics())
        _accumulate_metrics(op_metrics, duration_ms, layer_depth)

        bucket_candidates: list[tuple[int, int, int]] = []
        if batch_size > 0 and total_tokens > 0:
            bucket_candidates.append((batch_size, total_tokens, 1))
            if (
                request_level_decode_bucketing
                and (
                    collective_bucket_mode == "request_level"
                    or not _is_collective_scope_name(name)
                )
                and str(logical_invocation.get("phase", PHASE_UNKNOWN)) == PHASE_DECODE
                and _to_int(logical_invocation.get("batch_num_prefill_tokens"), default=0) <= 0
                and _to_int(logical_invocation.get("batch_num_decode_tokens"), default=0) > 0
                and batch_size > 1
                and total_tokens % batch_size == 0
            ):
                per_request_tokens = total_tokens // batch_size
                if per_request_tokens > 0:
                    bucket_candidates = [(1, per_request_tokens, batch_size)]
        if not bucket_candidates:
            continue

        phase = str(logical_invocation.get("phase", PHASE_UNKNOWN))
        for bucket_batch_size, bucket_total_tokens, bucket_weight in bucket_candidates:
            bucket_key = (name, bucket_batch_size, bucket_total_tokens)
            bucket_metrics = bucket_metrics_raw.setdefault(bucket_key, _new_metrics())
            for _ in range(bucket_weight):
                _accumulate_metrics(bucket_metrics, duration_ms, layer_depth)
            phase_counter = bucket_phase_counts.setdefault(bucket_key, {})
            phase_counter[phase] = int(phase_counter.get(phase, 0)) + bucket_weight

    total_metrics = {
        op_name: _finalize_metrics(metrics) for op_name, metrics in total_metrics_raw.items()
    }
    bucket_metrics = {
        bucket_key: _finalize_metrics(metrics)
        for bucket_key, metrics in bucket_metrics_raw.items()
    }
    bucket_phase_map: dict[tuple[str, int, int], str] = {}
    for bucket_key, phase_counter in bucket_phase_counts.items():
        ordered = sorted(
            phase_counter.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )
        bucket_phase_map[bucket_key] = ordered[0][0]
    return total_metrics, bucket_metrics, bucket_phase_map


def _load_frontier_trace_events(path: Path) -> list[dict[str, Any]]:
    trace_events: list[dict[str, Any]] = []
    for event in _load_jsonl_records(path, missing_message="Frontier op trace file not found"):
        event_type = event.get("type")
        if event_type not in {"COMPUTE", "COMM"}:
            continue
        name = event.get("name")
        duration_ms = event.get("duration_ms")
        if name is None or duration_ms is None:
            continue
        trace_events.append(event)
    if not trace_events:
        raise ValueError(f"no compute/comm op rows found in Frontier trace: {path}")
    return trace_events


def _summarize_frontier_ops_from_trace_events(
    trace_events: list[dict[str, Any]],
    *,
    model_profile: str,
    request_level_decode_bucketing: bool = False,
) -> tuple[
    dict[str, dict[str, float | int | None]],
    dict[tuple[str, int, int], dict[str, float | int | None]],
]:
    total_metrics_raw: dict[str, dict[str, float]] = {}
    bucket_metrics_raw: dict[tuple[str, int, int], dict[str, float]] = {}
    frontier_batch_phase_map: dict[int, str] = {}
    if request_level_decode_bucketing:
        batch_events_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in trace_events:
            batch_id = _to_int(event.get("batch_id"), default=-1)
            if batch_id < 0:
                continue
            batch_events_by_id[batch_id].append(event)
        frontier_batch_phase_map = {
            batch_id: _infer_frontier_batch_phase(batch_events)
            for batch_id, batch_events in batch_events_by_id.items()
            if batch_events
        }

    for event in trace_events:
        op_name = str(event["name"])
        recorded_layer_depth = _frontier_layer_depth(event)
        full_layer_multiplier = _frontier_full_layer_multiplier(event)
        layer_depth = max(recorded_layer_depth, full_layer_multiplier)
        duration = float(event["duration_ms"])
        batch_id = _to_int(event.get("batch_id"), default=-1)
        batch_phase = frontier_batch_phase_map.get(batch_id, PHASE_UNKNOWN)

        op_metrics = total_metrics_raw.setdefault(op_name, _new_metrics())
        _accumulate_metrics(
            op_metrics,
            duration,
            layer_depth,
            full_layer_multiplier=full_layer_multiplier,
        )

        batch_size, total_tokens = _normalize_frontier_bucket_signature(event)
        if batch_size <= 0 or total_tokens <= 0:
            continue
        bucket_total_tokens = total_tokens
        if batch_phase == PHASE_DECODE:
            bucket_total_tokens = batch_size

        bucket_candidates: list[tuple[int, int, int]] = [
            (batch_size, bucket_total_tokens, 1)
        ]
        if request_level_decode_bucketing and batch_size > 1 and batch_phase == PHASE_DECODE:
            per_request_tokens = bucket_total_tokens // batch_size
            if per_request_tokens > 0:
                bucket_candidates = [(1, per_request_tokens, batch_size)]

        for bucket_batch_size, bucket_total_tokens, bucket_weight in bucket_candidates:
            bucket_key = (op_name, bucket_batch_size, bucket_total_tokens)
            bucket_metrics = bucket_metrics_raw.setdefault(bucket_key, _new_metrics())
            for _ in range(bucket_weight):
                _accumulate_metrics(
                    bucket_metrics,
                    duration,
                    layer_depth,
                    full_layer_multiplier=full_layer_multiplier,
                )

    total_metrics = {
        op_name: _finalize_metrics(metrics) for op_name, metrics in total_metrics_raw.items()
    }
    bucket_metrics = {
        bucket_key: _finalize_metrics(metrics)
        for bucket_key, metrics in bucket_metrics_raw.items()
    }

    if model_profile == "moe":
        for virtual_op, alternatives in VIRTUAL_FRONTIER_OP_COMPONENTS.items():
            if virtual_op in total_metrics:
                continue
            selected_components: list[str] = []
            for candidate_components in alternatives:
                if all(component in total_metrics for component in candidate_components):
                    selected_components = candidate_components
                    break
            if not selected_components:
                continue

            total_metrics[virtual_op] = _merge_finalized_metrics(
                [total_metrics[component] for component in selected_components]
            )

            component_bucket_sets: list[set[tuple[int, int]]] = []
            for component in selected_components:
                component_bucket_sets.append(
                    {
                        (batch_size, total_tokens)
                        for op_name, batch_size, total_tokens in bucket_metrics
                        if op_name == component
                    }
                )
            if not component_bucket_sets:
                continue
            shared_bucket_signatures = set.intersection(*component_bucket_sets)
            for batch_size, total_tokens in shared_bucket_signatures:
                bucket_rows = [
                    bucket_metrics[(component, batch_size, total_tokens)]
                    for component in selected_components
                ]
                bucket_metrics[(virtual_op, batch_size, total_tokens)] = (
                    _merge_finalized_metrics(bucket_rows)
                )
    return total_metrics, bucket_metrics


def _new_execution_metrics() -> dict[str, float]:
    return {
        "execution_time_ms_sum": 0.0,
        "invocations": 0.0,
    }


def _accumulate_execution_metrics(metrics: dict[str, float], execution_time_ms: float) -> None:
    metrics["execution_time_ms_sum"] += execution_time_ms
    metrics["invocations"] += 1.0


def _finalize_execution_metrics(metrics: dict[str, float]) -> dict[str, float | int | None]:
    total_execution_time_ms = float(metrics["execution_time_ms_sum"])
    invocations = int(metrics["invocations"])
    mean_execution_time_ms = None
    if invocations > 0:
        mean_execution_time_ms = total_execution_time_ms / float(invocations)
    return {
        "total_execution_time_ms": total_execution_time_ms,
        "invocations": invocations,
        "mean_execution_time_ms": mean_execution_time_ms,
    }


def _infer_frontier_batch_phase(batch_events: list[dict[str, Any]]) -> str:
    op_names = {str(event.get("name")) for event in batch_events if event.get("name") is not None}
    has_decode = "attn_decode" in op_names
    has_prefill = "attn_prefill" in op_names
    if has_decode and has_prefill:
        return PHASE_MIXED
    if has_decode:
        return PHASE_DECODE
    if has_prefill:
        return PHASE_PREFILL

    clusters = {str(event.get("cluster")) for event in batch_events if event.get("cluster")}
    if clusters == {"DECODE"}:
        return PHASE_DECODE
    if clusters == {"PREFILL"}:
        return PHASE_PREFILL

    batch_size, total_tokens = _normalize_frontier_bucket_signature(batch_events[0])
    return _infer_phase({}, batch_size=batch_size, total_tokens=total_tokens)


def _load_vllm_decode_batch_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "batch_records": [],
            "bucket_metrics": {},
            "num_decode_batches": 0,
        }

    grouped: dict[int, dict[str, Any]] = {}
    for line_number, event in enumerate(
        _load_jsonl_records(path, missing_message="vLLM batch log not found"),
        start=1,
    ):
        execution_time_ms = event.get("batch_execution_time_ms")
        if execution_time_ms is None:
            continue

        batch_size, total_tokens = _normalize_vllm_bucket_signature(event)
        batch_id = _to_int(event.get("batch_id"), default=-1)
        logical_batch_id = batch_id if batch_id >= 0 else -line_number
        phase = _infer_phase(event, batch_size=batch_size, total_tokens=total_tokens)
        request_ids = [str(item) for item in event.get("request_ids", [])] if isinstance(event.get("request_ids"), list) else []
        request_num_tokens = [
            _to_int(item, default=0) for item in event.get("request_num_tokens", [])
        ] if isinstance(event.get("request_num_tokens"), list) else []

        batch_record = grouped.setdefault(
            logical_batch_id,
            {
                "batch_id": logical_batch_id,
                "batch_size": batch_size,
                "total_tokens": total_tokens,
                "phase": phase,
                "request_ids": request_ids,
                "request_num_tokens": request_num_tokens,
                "execution_time_ms_sum": 0.0,
                "row_count": 0,
                "rows": [],
            },
        )
        if int(batch_record["batch_size"]) != batch_size or int(batch_record["total_tokens"]) != total_tokens:
            raise ValueError(
                f"inconsistent duplicated vLLM batch rows detected for batch_id={logical_batch_id}"
            )
        if str(batch_record["phase"]) != phase:
            raise ValueError(
                f"inconsistent duplicated vLLM batch phases detected for batch_id={logical_batch_id}"
            )
        if list(batch_record["request_ids"]) != request_ids or list(batch_record["request_num_tokens"]) != request_num_tokens:
            raise ValueError(
                f"inconsistent duplicated vLLM batch payload detected for batch_id={logical_batch_id}"
            )
        batch_record["execution_time_ms_sum"] = float(batch_record["execution_time_ms_sum"]) + float(execution_time_ms)
        batch_record["row_count"] = int(batch_record["row_count"]) + 1
        batch_record["rows"].append(
            {
                "batch_execution_time_ms": float(execution_time_ms),
                "timestamp": float(event["timestamp"]),
            }
        )

    decode_batch_records: list[dict[str, Any]] = []
    bucket_metrics_raw: dict[tuple[int, int], dict[str, float]] = {}
    for batch_record in grouped.values():
        row_count = int(batch_record["row_count"])
        if row_count <= 0:
            raise ValueError(f"invalid vLLM batch duplicate count for batch_id={batch_record['batch_id']}")
        duplicated_rows = sorted(
            list(batch_record["rows"]),
            key=lambda row: row["timestamp"],
        )
        if len(duplicated_rows) != row_count:
            raise ValueError(
                "inconsistent duplicated vLLM batch capture while reconstructing "
                f"logical batch timing for batch_id={batch_record['batch_id']}"
            )

        stage_groups: list[list[dict[str, float]]] = []
        for duplicated_row in duplicated_rows:
            timestamp = duplicated_row["timestamp"]
            if (
                not stage_groups
                or timestamp - stage_groups[-1][-1]["timestamp"]
                > VLLM_BATCH_DUPLICATE_STAGE_TIMESTAMP_SLACK_S
            ):
                stage_groups.append([duplicated_row])
                continue
            stage_groups[-1].append(duplicated_row)

        logical_execution_time_ms = sum(
            max(row["batch_execution_time_ms"] for row in stage_group)
            for stage_group in stage_groups
        )
        if str(batch_record["phase"]) != PHASE_DECODE:
            continue
        decode_batch_record = {
            "batch_id": int(batch_record["batch_id"]),
            "batch_size": int(batch_record["batch_size"]),
            "total_tokens": int(batch_record["total_tokens"]),
            "phase": str(batch_record["phase"]),
            "execution_time_ms": logical_execution_time_ms,
            "duplicated_rank_rows": row_count,
            "stage_group_count": len(stage_groups),
        }
        decode_batch_records.append(decode_batch_record)
        bucket_key = (
            int(batch_record["batch_size"]),
            int(batch_record["total_tokens"]),
        )
        bucket_metrics = bucket_metrics_raw.setdefault(bucket_key, _new_execution_metrics())
        _accumulate_execution_metrics(bucket_metrics, logical_execution_time_ms)

    bucket_metrics = {
        bucket_key: _finalize_execution_metrics(metrics)
        for bucket_key, metrics in bucket_metrics_raw.items()
    }
    return {
        "provided": True,
        "batch_records": decode_batch_records,
        "bucket_metrics": bucket_metrics,
        "num_decode_batches": len(decode_batch_records),
    }


def _load_frontier_decode_batch_metrics(trace_events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_events: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    unknown_phase_rows = 0
    for event in trace_events:
        batch_id = _to_int(event.get("batch_id"), default=-1)
        if batch_id < 0:
            unknown_phase_rows += 1
            continue
        cluster = str(event.get("cluster") or "UNKNOWN")
        grouped_events[(cluster, batch_id)].append(event)

    decode_batch_records: list[dict[str, Any]] = []
    bucket_metrics_raw: dict[tuple[int, int], dict[str, float]] = {}
    phase_row_counts: dict[str, int] = defaultdict(int)
    for (cluster, batch_id), batch_events in grouped_events.items():
        phase = _infer_frontier_batch_phase(batch_events)
        phase_row_counts[phase] += len(batch_events)
        if phase != PHASE_DECODE:
            continue
        batch_size, total_tokens = _normalize_frontier_bucket_signature(batch_events[0])
        if batch_size <= 0 or total_tokens <= 0:
            continue
        execution_time_ms = sum(float(event["duration_ms"]) for event in batch_events)
        decode_batch_records.append(
            {
                "cluster": cluster,
                "batch_id": batch_id,
                "batch_size": batch_size,
                "total_tokens": total_tokens,
                "phase": phase,
                "execution_time_ms": execution_time_ms,
                "op_row_count": len(batch_events),
            }
        )
        bucket_key = (batch_size, total_tokens)
        bucket_metrics = bucket_metrics_raw.setdefault(bucket_key, _new_execution_metrics())
        _accumulate_execution_metrics(bucket_metrics, execution_time_ms)

    bucket_metrics = {
        bucket_key: _finalize_execution_metrics(metrics)
        for bucket_key, metrics in bucket_metrics_raw.items()
    }
    return {
        "batch_records": decode_batch_records,
        "bucket_metrics": bucket_metrics,
        "num_decode_batches": len(decode_batch_records),
        "phase_row_counts": dict(phase_row_counts),
        "unknown_phase_rows": unknown_phase_rows,
    }


def _build_decode_batch_execution_summary(
    *,
    vllm_batch_metrics: dict[str, Any],
    frontier_batch_metrics: dict[str, Any],
    comparable: bool,
    threshold_percent: float,
) -> dict[str, Any]:
    if not bool(vllm_batch_metrics.get("provided", False)):
        return {
            "status": "MISSING_VLLM_BATCH_LOG",
            "comparable": False,
            "non_comparable_reason": "vLLM batch log not provided",
            "whole_run_comparable": comparable,
            "row_level_rows_available": False,
            "vllm_batch_log_provided": False,
            "vllm_num_decode_batches": 0,
            "frontier_num_decode_batches": int(frontier_batch_metrics.get("num_decode_batches", 0)),
            "num_compared_bucket_rows": 0,
            "mean_relative_error_percent": None,
            "max_relative_error_percent": None,
            "rows": [],
        }

    vllm_bucket_metrics = vllm_batch_metrics.get("bucket_metrics", {})
    frontier_bucket_metrics = frontier_batch_metrics.get("bucket_metrics", {})
    shared_bucket_keys = sorted(set(vllm_bucket_metrics) & set(frontier_bucket_metrics))
    rows: list[dict[str, Any]] = []
    relative_errors: list[float] = []
    for batch_size, total_tokens in shared_bucket_keys:
        vllm_payload = vllm_bucket_metrics[(batch_size, total_tokens)]
        frontier_payload = frontier_bucket_metrics[(batch_size, total_tokens)]
        vllm_mean_ms = vllm_payload.get("mean_execution_time_ms")
        frontier_mean_ms = frontier_payload.get("mean_execution_time_ms")
        relative_error = None
        if vllm_mean_ms is not None and frontier_mean_ms is not None:
            relative_error = _safe_rel_error(float(vllm_mean_ms), float(frontier_mean_ms))
            if relative_error is not None:
                relative_errors.append(relative_error)
        rows.append(
            {
                "batch_size": int(batch_size),
                "total_tokens": int(total_tokens),
                "is_pure_decode_bucket": True,
                "vllm_mean_execution_time_ms": vllm_mean_ms,
                "frontier_mean_execution_time_ms": frontier_mean_ms,
                "relative_error_percent": relative_error,
                "vllm_invocations": int(vllm_payload.get("invocations", 0)),
                "frontier_invocations": int(frontier_payload.get("invocations", 0)),
                "invocation_delta": int(frontier_payload.get("invocations", 0))
                - int(vllm_payload.get("invocations", 0)),
            }
        )

    mean_relative_error = statistics.mean(relative_errors) if relative_errors else None
    max_relative_error = max(relative_errors) if relative_errors else None
    non_comparable_reason: str | None = None
    if not rows:
        if int(vllm_batch_metrics.get("num_decode_batches", 0)) <= 0 and int(frontier_batch_metrics.get("num_decode_batches", 0)) <= 0:
            status = "NO_DECODE_BATCHES"
        elif not comparable:
            status = "NON_COMPARABLE"
            non_comparable_reason = "non-comparable due to path divergence"
        else:
            status = "UNAVAILABLE"
            non_comparable_reason = "no shared decode batch bucket signatures"
    elif not comparable:
        status = "NON_COMPARABLE"
        non_comparable_reason = "non-comparable due to path divergence"
    elif mean_relative_error is not None and max_relative_error is not None and (
        mean_relative_error > threshold_percent or max_relative_error > threshold_percent
    ):
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "status": status,
        "comparable": comparable and bool(rows),
        "non_comparable_reason": non_comparable_reason,
        "whole_run_comparable": comparable,
        "row_level_rows_available": bool(rows),
        "vllm_batch_log_provided": True,
        "threshold_percent": threshold_percent,
        "vllm_num_decode_batches": int(vllm_batch_metrics.get("num_decode_batches", 0)),
        "frontier_num_decode_batches": int(frontier_batch_metrics.get("num_decode_batches", 0)),
        "num_compared_bucket_rows": len(rows),
        "mean_relative_error_percent": mean_relative_error,
        "max_relative_error_percent": max_relative_error,
        "rows": rows,
    }


def _append_decode_batch_execution_section(
    *,
    lines: list[str],
    title: str,
    decode_batch_execution: dict[str, Any],
) -> None:
    lines.extend(
        [
            "",
            f"### {title}",
            "",
            "| Field | Value |",
            "|------|-------|",
            f"| Status | `{decode_batch_execution['status']}` |",
            f"| Comparable | `{decode_batch_execution['comparable']}` |",
            f"| vLLM decode batches | `{decode_batch_execution['vllm_num_decode_batches']}` |",
            f"| Frontier decode batches | `{decode_batch_execution['frontier_num_decode_batches']}` |",
            f"| Compared bucket rows | `{decode_batch_execution['num_compared_bucket_rows']}` |",
            f"| Mean relative error (%) | {_format_optional_float(decode_batch_execution['mean_relative_error_percent'])} |",
            f"| Max relative error (%) | {_format_optional_float(decode_batch_execution['max_relative_error_percent'])} |",
        ]
    )
    if decode_batch_execution.get("non_comparable_reason"):
        lines.append(
            f"- Decode batch evidence note: `{decode_batch_execution['non_comparable_reason']}`"
        )
    decode_rows = decode_batch_execution.get("rows", [])
    if decode_rows:
        lines.extend(
            [
                "",
                "| Batch size | Total tokens | vLLM mean exec (ms) | Frontier mean exec (ms) | Relative error (%) | vLLM inv | Frontier inv |",
                "|-----------:|-------------:|--------------------:|------------------------:|-------------------:|---------:|-------------:|",
            ]
        )
        for row in decode_rows:
            lines.append(
                "| {batch_size} | {total_tokens} | {vllm} | {frontier} | {rel} | {v_inv} | {f_inv} |".format(
                    batch_size=int(row["batch_size"]),
                    total_tokens=int(row["total_tokens"]),
                    vllm=_format_optional_float(row["vllm_mean_execution_time_ms"]),
                    frontier=_format_optional_float(row["frontier_mean_execution_time_ms"]),
                    rel=_format_optional_float(row["relative_error_percent"]),
                    v_inv=int(row["vllm_invocations"]),
                    f_inv=int(row["frontier_invocations"]),
                )
            )


def _build_decode_phase_observability(
    *,
    vllm_logical_invocations: list[dict[str, Any]],
    frontier_batch_metrics: dict[str, Any],
    bucket_context_rows: list[dict[str, Any]],
    decode_batch_execution_summary: dict[str, Any],
    vllm_batch_log_provided: bool,
) -> dict[str, Any]:
    vllm_phase_counts: dict[str, int] = defaultdict(int)
    for row in vllm_logical_invocations:
        vllm_phase_counts[str(row.get("phase", PHASE_UNKNOWN))] += 1

    frontier_phase_row_counts = frontier_batch_metrics.get("phase_row_counts", {})
    frontier_decode_op_rows = int(frontier_phase_row_counts.get(PHASE_DECODE, 0))
    frontier_prefill_op_rows = int(frontier_phase_row_counts.get(PHASE_PREFILL, 0))
    decode_bucket_context_rows = [
        row for row in bucket_context_rows if str(row.get("phase", PHASE_UNKNOWN)) == PHASE_DECODE
    ]
    decode_compared_rows = [
        row
        for row in decode_bucket_context_rows
        if row.get("relative_error_percent") is not None
    ]

    status = "NO_DECODE_PHASE_EVIDENCE"
    notes: list[str] = []
    vllm_decode_per_op_invocations = int(vllm_phase_counts.get(PHASE_DECODE, 0))
    if vllm_decode_per_op_invocations > 0 and decode_compared_rows:
        status = "DECODE_PER_OP_COMPARABLE"
    elif vllm_decode_per_op_invocations > 0:
        status = "DECODE_PER_OP_PRESENT"
    elif frontier_decode_op_rows > 0 and decode_batch_execution_summary.get("num_compared_bucket_rows", 0) > 0:
        status = "VLLM_DECODE_PER_OP_MISSING_BATCH_TIMING_AVAILABLE"
        notes.append(
            "vLLM raw CUDA-op log does not expose decode-phase per-op rows for this run; comparable decode evidence currently comes from batch execution timing instead of per-op rows."
        )
    elif frontier_decode_op_rows > 0:
        status = "VLLM_DECODE_PER_OP_MISSING"
        notes.append(
            "Frontier records decode-phase op rows, but vLLM raw CUDA-op log does not expose matching decode-phase rows for this run."
        )

    if not vllm_batch_log_provided:
        notes.append("Decode batch execution evidence is unavailable because --vllm-batch-log was not provided.")
    elif decode_batch_execution_summary.get("num_compared_bucket_rows", 0) > 0:
        notes.append(
            "Decode batch execution evidence compares vLLM batch_execution_time_ms against Frontier batch-summed decode-phase op traces by shared batch-size/token buckets."
        )

    return {
        "status": status,
        "vllm_batch_log_provided": vllm_batch_log_provided,
        "vllm_prefill_per_op_invocations": int(vllm_phase_counts.get(PHASE_PREFILL, 0)),
        "vllm_decode_per_op_invocations": vllm_decode_per_op_invocations,
        "vllm_mixed_per_op_invocations": int(vllm_phase_counts.get(PHASE_MIXED, 0)),
        "frontier_prefill_op_rows": frontier_prefill_op_rows,
        "frontier_decode_op_rows": frontier_decode_op_rows,
        "frontier_unknown_phase_rows": int(frontier_batch_metrics.get("unknown_phase_rows", 0)),
        "num_decode_bucket_context_rows": len(decode_bucket_context_rows),
        "num_decode_compared_bucket_rows": len(decode_compared_rows),
        "notes": notes,
    }


def _load_vllm_ops(
    path: Path,
    batch_log_path: Path | None = None,
) -> tuple[
    dict[str, dict[str, float | int | None]],
    dict[tuple[str, int, int], dict[str, float | int | None]],
    dict[tuple[str, int, int], str],
]:
    logical_invocations = _collect_vllm_logical_invocations(path, batch_log_path)
    return _summarize_vllm_ops_from_logical_invocations(logical_invocations)


def _load_frontier_ops(
    path: Path,
    *,
    model_profile: str,
) -> tuple[
    dict[str, dict[str, float | int | None]],
    dict[tuple[str, int, int], dict[str, float | int | None]],
]:
    trace_events = _load_frontier_trace_events(path)
    return _summarize_frontier_ops_from_trace_events(
        trace_events,
        model_profile=model_profile,
    )


def _resolve_frontier_op(
    vllm_op_name: str,
    frontier_totals: dict[str, dict[str, float | int | None]],
    *,
    op_name_map: dict[str, list[str]],
) -> str:
    candidates = op_name_map.get(vllm_op_name, [vllm_op_name])
    for candidate in candidates:
        if candidate in frontier_totals:
            return candidate
    return ""


def _load_mismatch_effective_count(path: Path | None) -> tuple[int, bool]:
    if path is None:
        return 0, False
    if not path.is_file():
        raise FileNotFoundError(f"schedule summary json not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"schedule summary json must be an object: {path}")

    raw_value = payload.get(
        "mismatch_effective_count",
        payload.get("num_mismatched_iterations", 0),
    )
    mismatch_effective_count = int(raw_value)
    if mismatch_effective_count < 0:
        raise ValueError(
            f"invalid mismatch_effective_count={mismatch_effective_count} in {path}"
        )
    return mismatch_effective_count, True


def _build_context_rows(
    vllm_totals: dict[str, dict[str, float | int | None]],
    frontier_totals: dict[str, dict[str, float | int | None]],
    mapped_frontier_op: dict[str, str],
    non_actionable_ops: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context_rows: list[dict[str, Any]] = []
    invocation_stats: list[dict[str, Any]] = []
    non_actionable_ops = non_actionable_ops or set()

    for op_name in sorted(vllm_totals):
        mapped_op = mapped_frontier_op.get(op_name, "")
        v_metrics = vllm_totals[op_name]
        f_metrics = frontier_totals.get(mapped_op, _finalize_metrics(_new_metrics()))

        v_full_total = float(v_metrics["full_layer_total_ms"])
        f_full_total = float(f_metrics["full_layer_total_ms"])
        v_full_inv = int(v_metrics["full_layer_invocations"])
        f_full_inv = int(f_metrics["full_layer_invocations"])
        v_single_inv = int(v_metrics["single_layer_invocations"])
        f_single_inv = int(f_metrics["single_layer_invocations"])

        if op_name in non_actionable_ops:
            full_rel_error = None
            v_single_avg = v_metrics["single_layer_avg_ms"]
            f_single_avg = f_metrics["single_layer_avg_ms"]
            single_rel_error = None
        elif mapped_op:
            full_rel_error = _safe_rel_error(v_full_total, f_full_total)
            v_single_avg = v_metrics["single_layer_avg_ms"]
            f_single_avg = f_metrics["single_layer_avg_ms"]
            if v_single_avg is None or f_single_avg is None:
                single_rel_error = None
            else:
                single_rel_error = _safe_rel_error(float(v_single_avg), float(f_single_avg))
        else:
            full_rel_error = None
            v_single_avg = None
            f_single_avg = None
            single_rel_error = None

        context_rows.append(
            {
                "op_name": op_name,
                "mapped_frontier_op": mapped_op,
                "context": FULL_LAYER_CONTEXT,
                "vllm_metric_ms": v_full_total,
                "frontier_metric_ms": f_full_total,
                "relative_error_percent": full_rel_error,
                "vllm_invocations": v_full_inv,
                "frontier_invocations": f_full_inv,
                "invocation_delta": f_full_inv - v_full_inv,
                "invocation_delta_percent": _safe_delta_percent(v_full_inv, f_full_inv),
                "vllm_avg_layer_depth": v_metrics["avg_layer_depth"],
                "frontier_avg_layer_depth": f_metrics["avg_layer_depth"],
                **_measurement_scope_actionability_fields(op_name, non_actionable_ops),
            }
        )
        context_rows.append(
            {
                "op_name": op_name,
                "mapped_frontier_op": mapped_op,
                "context": SINGLE_LAYER_CONTEXT,
                "vllm_metric_ms": v_single_avg,
                "frontier_metric_ms": f_single_avg,
                "relative_error_percent": single_rel_error,
                "vllm_invocations": v_single_inv,
                "frontier_invocations": f_single_inv,
                "invocation_delta": f_single_inv - v_single_inv,
                "invocation_delta_percent": _safe_delta_percent(v_single_inv, f_single_inv),
                "vllm_avg_layer_depth": v_metrics["avg_layer_depth"],
                "frontier_avg_layer_depth": f_metrics["avg_layer_depth"],
                **_measurement_scope_actionability_fields(op_name, non_actionable_ops),
            }
        )

        invocation_stats.append(
            {
                "op_name": op_name,
                "mapped_frontier_op": mapped_op,
                "vllm_full_layer_invocations": v_full_inv,
                "frontier_full_layer_invocations": f_full_inv,
                "full_layer_invocation_delta": f_full_inv - v_full_inv,
                "full_layer_invocation_delta_percent": _safe_delta_percent(v_full_inv, f_full_inv),
                "vllm_single_layer_invocations": v_single_inv,
                "frontier_single_layer_invocations": f_single_inv,
                "single_layer_invocation_delta": f_single_inv - v_single_inv,
                "single_layer_invocation_delta_percent": _safe_delta_percent(
                    v_single_inv, f_single_inv
                ),
                "vllm_avg_layer_depth": v_metrics["avg_layer_depth"],
                "frontier_avg_layer_depth": f_metrics["avg_layer_depth"],
                **_measurement_scope_actionability_fields(op_name, non_actionable_ops),
            }
        )

    return context_rows, invocation_stats


def _build_bucket_context_rows(
    vllm_buckets: dict[tuple[str, int, int], dict[str, float | int | None]],
    frontier_buckets: dict[tuple[str, int, int], dict[str, float | int | None]],
    mapped_frontier_op: dict[str, str],
    bucket_phase_map: dict[tuple[str, int, int], str],
    non_actionable_ops: set[str] | None = None,
) -> list[dict[str, Any]]:
    non_actionable_ops = non_actionable_ops or set()
    frontier_keys_by_op: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for op_name, batch_size, total_tokens in frontier_buckets:
        frontier_keys_by_op[op_name].add((batch_size, total_tokens))

    vllm_keys_by_op: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for op_name, batch_size, total_tokens in vllm_buckets:
        vllm_keys_by_op[op_name].add((batch_size, total_tokens))

    bucket_rows: list[dict[str, Any]] = []

    for op_name in sorted(mapped_frontier_op):
        mapped_op = mapped_frontier_op[op_name]
        if not mapped_op:
            continue

        all_bucket_keys = set(vllm_keys_by_op.get(op_name, set())) | set(
            frontier_keys_by_op.get(mapped_op, set())
        )

        for batch_size, total_tokens in sorted(all_bucket_keys):
            v_metrics = vllm_buckets.get(
                (op_name, batch_size, total_tokens), _finalize_metrics(_new_metrics())
            )
            f_metrics = frontier_buckets.get(
                (mapped_op, batch_size, total_tokens), _finalize_metrics(_new_metrics())
            )

            v_full_total = float(v_metrics["full_layer_total_ms"])
            f_full_total = float(f_metrics["full_layer_total_ms"])
            v_full_inv = int(v_metrics["full_layer_invocations"])
            f_full_inv = int(f_metrics["full_layer_invocations"])
            v_single_inv = int(v_metrics["single_layer_invocations"])
            f_single_inv = int(f_metrics["single_layer_invocations"])

            if op_name in non_actionable_ops:
                full_rel_error = None
            else:
                full_rel_error = _safe_rel_error(v_full_total, f_full_total)
            phase = bucket_phase_map.get((op_name, batch_size, total_tokens), PHASE_UNKNOWN)
            bucket_rows.append(
                {
                    "op_name": op_name,
                    "mapped_frontier_op": mapped_op,
                    "batch_size": int(batch_size),
                    "total_tokens": int(total_tokens),
                    "phase": phase,
                    "context": FULL_LAYER_CONTEXT,
                    "vllm_metric_ms": v_full_total,
                    "frontier_metric_ms": f_full_total,
                    "relative_error_percent": full_rel_error,
                    "abs_gap_metric_ms": abs(f_full_total - v_full_total),
                    "vllm_invocations": v_full_inv,
                    "frontier_invocations": f_full_inv,
                    "invocation_delta": f_full_inv - v_full_inv,
                    "invocation_delta_percent": _safe_delta_percent(v_full_inv, f_full_inv),
                    "vllm_avg_layer_depth": v_metrics["avg_layer_depth"],
                    "frontier_avg_layer_depth": f_metrics["avg_layer_depth"],
                    **_measurement_scope_actionability_fields(op_name, non_actionable_ops),
                }
            )

            v_single_avg = v_metrics["single_layer_avg_ms"]
            f_single_avg = f_metrics["single_layer_avg_ms"]
            if op_name in non_actionable_ops:
                single_rel_error = None
                abs_gap = abs(float(v_single_avg or 0.0) - float(f_single_avg or 0.0))
            elif v_single_avg is None or f_single_avg is None:
                single_rel_error = None
                abs_gap = abs(float(v_single_avg or 0.0) - float(f_single_avg or 0.0))
            else:
                single_rel_error = _safe_rel_error(float(v_single_avg), float(f_single_avg))
                abs_gap = abs(float(f_single_avg) - float(v_single_avg))

            bucket_rows.append(
                {
                    "op_name": op_name,
                    "mapped_frontier_op": mapped_op,
                    "batch_size": int(batch_size),
                    "total_tokens": int(total_tokens),
                    "phase": phase,
                    "context": SINGLE_LAYER_CONTEXT,
                    "vllm_metric_ms": v_single_avg,
                    "frontier_metric_ms": f_single_avg,
                    "relative_error_percent": single_rel_error,
                    "abs_gap_metric_ms": abs_gap,
                    "vllm_invocations": v_single_inv,
                    "frontier_invocations": f_single_inv,
                    "invocation_delta": f_single_inv - v_single_inv,
                    "invocation_delta_percent": _safe_delta_percent(v_single_inv, f_single_inv),
                    "vllm_avg_layer_depth": v_metrics["avg_layer_depth"],
                    "frontier_avg_layer_depth": f_metrics["avg_layer_depth"],
                    **_measurement_scope_actionability_fields(op_name, non_actionable_ops),
                }
            )

    return bucket_rows


def _build_trend_summary(bucket_rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable_rows = [
        row for row in bucket_rows if row["relative_error_percent"] is not None
    ]

    top_abs_gap = sorted(
        comparable_rows,
        key=lambda row: float(row["abs_gap_metric_ms"]),
        reverse=True,
    )
    top_rel_error = sorted(
        comparable_rows,
        key=lambda row: float(row["relative_error_percent"]),
        reverse=True,
    )

    batch_context_totals: dict[tuple[str, int], dict[str, float]] = {}
    for row in comparable_rows:
        context = str(row["context"])
        batch_size = int(row["batch_size"])
        key = (context, batch_size)
        item = batch_context_totals.setdefault(
            key,
            {
                "vllm_metric_sum_ms": 0.0,
                "frontier_metric_sum_ms": 0.0,
                "num_rows": 0.0,
            },
        )
        item["vllm_metric_sum_ms"] += float(row["vllm_metric_ms"])
        item["frontier_metric_sum_ms"] += float(row["frontier_metric_ms"])
        item["num_rows"] += 1.0

    batch_size_error_summary: list[dict[str, Any]] = []
    for (context, batch_size), item in sorted(batch_context_totals.items()):
        vllm_sum = float(item["vllm_metric_sum_ms"])
        frontier_sum = float(item["frontier_metric_sum_ms"])
        rel = _safe_rel_error(vllm_sum, frontier_sum)
        batch_size_error_summary.append(
            {
                "context": context,
                "batch_size": int(batch_size),
                "vllm_metric_sum_ms": vllm_sum,
                "frontier_metric_sum_ms": frontier_sum,
                "relative_error_percent": rel,
                "num_rows": int(item["num_rows"]),
            }
        )

    return {
        "top_bucket_errors_by_abs_gap": top_abs_gap,
        "top_bucket_errors_by_relative_error": top_rel_error,
        "batch_size_error_summary": batch_size_error_summary,
    }


def _build_prefill_only_summary(
    bucket_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prefill_rows = [
        row
        for row in bucket_rows
        if str(row.get("phase", PHASE_UNKNOWN)) == PHASE_PREFILL
        and str(row.get("context", "")) == FULL_LAYER_CONTEXT
    ]
    rel_errors = [
        float(row["relative_error_percent"])
        for row in prefill_rows
        if row.get("relative_error_percent") is not None
    ]
    if rel_errors:
        mean_rel_error = float(statistics.mean(rel_errors))
        max_rel_error = float(max(rel_errors))
    else:
        mean_rel_error = None
        max_rel_error = None
    return {
        "num_prefill_bucket_rows": len(prefill_rows),
        "num_compared_prefill_bucket_rows": len(rel_errors),
        "mean_relative_error_percent": mean_rel_error,
        "max_relative_error_percent": max_rel_error,
    }


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def _context_label(context: str) -> str:
    if context == FULL_LAYER_CONTEXT:
        return "full_layer_total"
    if context == SINGLE_LAYER_CONTEXT:
        return "single_layer_avg"
    return context


def _resolve_fused_add_norm_scope_enabled(
    *, model_profile: str, fused_add_norm_scope: str
) -> bool:
    if fused_add_norm_scope == "enabled":
        return True
    if fused_add_norm_scope == "disabled":
        return False
    if fused_add_norm_scope != "auto":
        raise ValueError(f"unexpected fused add norm scope policy: {fused_add_norm_scope}")
    return model_profile == "moe"


def _measurement_scope_actionability_fields(
    op_name: str, non_actionable_ops: set[str]
) -> dict[str, Any]:
    if op_name in non_actionable_ops:
        return {
            "measurement_scope_actionability": (
                MEASUREMENT_SCOPE_ACTIONABILITY_NON_ACTIONABLE_NESTED_SCOPE
            ),
            "measurement_scope_reason": FUSED_ADD_NORM_NESTED_SCOPE_REASON,
            "excluded_from_status": True,
        }
    return {
        "measurement_scope_actionability": MEASUREMENT_SCOPE_ACTIONABILITY_ACTIONABLE,
        "measurement_scope_reason": None,
        "excluded_from_status": False,
    }


def _build_markdown(summary: dict[str, Any], *, top_bucket_rows_in_md: int) -> str:
    comparable = bool(summary["comparable"])
    decode_observability = summary["decode_phase_observability"]
    decode_batch_execution = summary["decode_batch_execution_summary"]
    decode_batch_execution_clean = summary.get("decode_batch_execution_clean_summary")
    collective_scope_audit = summary.get("vllm_collective_scope_audit", {})
    frontier_collective_component_summary = summary.get(
        "frontier_collective_component_summary", {}
    )

    lines = [
        "## Per-Op Comparison (Online Alignment)",
        "",
        f"- Status: **{summary['status']}**",
        f"- Model profile: `{summary['model_profile']}`",
        f"- Comparable: `{comparable}`",
        f"- Threshold: `{summary['threshold_percent']:.2f}%`",
        f"- Compared ops: `{summary['num_compared_ops']}`",
        f"- Missing ops: `{summary['num_missing_ops']}`",
        f"- Bucket context rows: `{summary['num_bucket_context_rows']}`",
        f"- Decode observability: `{decode_observability['status']}`",
    ]
    if not comparable:
        lines.append(
            f"- Note: `{summary['non_comparable_reason']}` "
            f"(mismatch_effective_count={summary['mismatch_effective_count']})"
        )

    lines.extend(
        [
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean relative error (full-layer total, %) | {_format_optional_float(summary['mean_relative_error_percent'])} |",
            f"| Max relative error (full-layer total, %) | {_format_optional_float(summary['max_relative_error_percent'])} |",
            f"| Mean relative error (single-layer avg, %) | {_format_optional_float(summary['single_layer_mean_relative_error_percent'])} |",
            f"| Max relative error (single-layer avg, %) | {_format_optional_float(summary['single_layer_max_relative_error_percent'])} |",
            f"| Mean relative error (prefill-only bucket full-layer total, %) | {_format_optional_float(summary['prefill_only_summary']['mean_relative_error_percent'])} |",
            f"| Max relative error (prefill-only bucket full-layer total, %) | {_format_optional_float(summary['prefill_only_summary']['max_relative_error_percent'])} |",
        ]
    )

    lines.extend(
        [
            "",
            "### Decode-Phase Observability",
            "",
            "| Field | Value |",
            "|------|-------|",
            f"| Status | `{decode_observability['status']}` |",
            f"| vLLM batch log provided | `{decode_observability['vllm_batch_log_provided']}` |",
            f"| vLLM decode per-op invocations | `{decode_observability['vllm_decode_per_op_invocations']}` |",
            f"| vLLM prefill per-op invocations | `{decode_observability['vllm_prefill_per_op_invocations']}` |",
            f"| Frontier decode op rows | `{decode_observability['frontier_decode_op_rows']}` |",
            f"| Frontier prefill op rows | `{decode_observability['frontier_prefill_op_rows']}` |",
            f"| Decode bucket context rows | `{decode_observability['num_decode_bucket_context_rows']}` |",
            f"| Decode compared bucket rows | `{decode_observability['num_decode_compared_bucket_rows']}` |",
        ]
    )
    if decode_observability.get("notes"):
        lines.extend(["", "Notes:"])
        for note in decode_observability["notes"]:
            lines.append(f"- {note}")

    if collective_scope_audit:
        lines.extend(
            [
                "",
                "### vLLM Collective Scope Audit",
                "",
                "| Field | Value |",
                "|------|-------|",
                f"| Status | `{collective_scope_audit.get('status', 'UNKNOWN')}` |",
                f"| Collective ops | `{collective_scope_audit.get('num_collective_ops', 0)}` |",
                f"| Flagged ops | `{collective_scope_audit.get('num_flagged_ops', 0)}` |",
                f"| Spread threshold (ms) | `{_format_optional_float(collective_scope_audit.get('spread_threshold_ms'))}` |",
            ]
        )
        flagged = collective_scope_audit.get("flagged_op_names", [])
        if flagged:
            lines.append(f"| Flagged op names | `{','.join(str(item) for item in flagged)}` |")

        ops = collective_scope_audit.get("ops", [])
        if ops:
            lines.extend(
                [
                    "",
                    "| Op | Classification | Duplicated invocations | Mean rank ms | Mean min-rank ms | Mean max-rank ms | Mean spread ms | Max spread ms |",
                    "|----|----------------|-----------------------:|-------------:|-----------------:|-----------------:|---------------:|--------------:|",
                ]
            )
            for row in ops:
                lines.append(
                    "| {op} | {classification} | {dup} | {mean_ms} | {min_ms} | {max_ms} | {mean_spread} | {max_spread} |".format(
                        op=row["op_name"],
                        classification=row["classification"],
                        dup=int(row["num_duplicated_rank_invocations"]),
                        mean_ms=_format_optional_float(row["mean_rank_time_ms"]),
                        min_ms=_format_optional_float(row["mean_min_rank_time_ms"]),
                        max_ms=_format_optional_float(row["mean_max_rank_time_ms"]),
                        mean_spread=_format_optional_float(row["mean_spread_ms"]),
                        max_spread=_format_optional_float(row["max_spread_ms"]),
                    )
                )

    if frontier_collective_component_summary:
        lines.extend(
            [
                "",
                "### Collective Scope Alignment",
                "",
                "| Field | Value |",
                "|------|-------|",
                f"| Alignment mode | `{summary.get('collective_scope_alignment_mode')}` |",
                f"| Actionability | `{summary.get('collective_scope_actionability')}` |",
                f"| Missing Frontier wait ops | `{','.join(summary.get('collective_scope_alignment_missing_frontier_wait_ops', [])) or '(none)'}` |",
                f"| Frontier collective ops | `{frontier_collective_component_summary.get('num_collective_ops', 0)}` |",
            ]
        )
        ops = frontier_collective_component_summary.get("ops", [])
        if ops:
            lines.extend(
                [
                    "",
                    "| Op | Frontier kernel-only total (ms) | Frontier related-wait total (ms) | Frontier wait-inclusive total (ms) | Frontier wait rows |",
                    "|----|---------------------------------:|----------------------------------:|------------------------------------:|------------------:|",
                ]
            )
            for row in ops:
                lines.append(
                    "| {op} | {kernel} | {wait} | {inclusive} | {wait_rows} |".format(
                        op=row["op_name"],
                        kernel=_format_optional_float(row["kernel_only_total_ms"]),
                        wait=_format_optional_float(row["related_wait_total_ms"]),
                        inclusive=_format_optional_float(
                            row["wait_inclusive_total_ms"]
                        ),
                        wait_rows=int(row["num_wait_rows"]),
                    )
                )

    _append_decode_batch_execution_section(
        lines=lines,
        title="Decode Batch Execution Evidence (Instrumented Path)",
        decode_batch_execution=decode_batch_execution,
    )
    if decode_batch_execution_clean is not None:
        _append_decode_batch_execution_section(
            lines=lines,
            title="Decode Batch Execution Evidence (Clean Path)",
            decode_batch_execution=decode_batch_execution_clean,
        )

    lines.extend(
        [
            "",
            "### Full-Layer Totals (Backward-Compatible)",
            "",
            "| Op | vLLM total (ms) | Frontier total (ms) | Relative error (%) | Frontier mapped op |",
            "|----|-----------------:|--------------------:|-------------------:|--------------------|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| {op} | {vllm:.6f} | {frontier:.6f} | {rel} | {mapped} |".format(
                op=row["op_name"],
                vllm=float(row["vllm_total_ms"]),
                frontier=float(row["frontier_total_ms"]),
                rel=_format_optional_float(row["relative_error_percent"]),
                mapped=row["mapped_frontier_op"] or "(missing)",
            )
        )

    lines.extend(
        [
            "",
            "### Context-Level Error + Invocation Summary",
            "",
            "| Op | Context | Phase | vLLM metric (ms) | Frontier metric (ms) | Relative error (%) | vLLM invocations | Frontier invocations | Inv delta |",
            "|----|---------|-------|-----------------:|---------------------:|-------------------:|-----------------:|---------------------:|----------:|",
        ]
    )
    for row in summary["context_rows"]:
        lines.append(
            "| {op} | {context} | {phase} | {vllm} | {frontier} | {rel} | {v_inv} | {f_inv} | {delta} |".format(
                op=row["op_name"],
                context=_context_label(str(row["context"])),
                phase=str(row.get("phase", PHASE_UNKNOWN)),
                vllm=_format_optional_float(row["vllm_metric_ms"]),
                frontier=_format_optional_float(row["frontier_metric_ms"]),
                rel=_format_optional_float(row["relative_error_percent"]),
                v_inv=int(row["vllm_invocations"]),
                f_inv=int(row["frontier_invocations"]),
                delta=int(row["invocation_delta"]),
            )
        )

    lines.extend(
        [
            "",
            "### Invocation Statistics by Operation",
            "",
            "| Op | vLLM full-layer inv | Frontier full-layer inv | vLLM single-layer inv | Frontier single-layer inv |",
            "|----|--------------------:|------------------------:|----------------------:|--------------------------:|",
        ]
    )
    for row in summary["invocation_stats_by_op"]:
        lines.append(
            "| {op} | {v_full} | {f_full} | {v_single} | {f_single} |".format(
                op=row["op_name"],
                v_full=int(row["vllm_full_layer_invocations"]),
                f_full=int(row["frontier_full_layer_invocations"]),
                v_single=int(row["vllm_single_layer_invocations"]),
                f_single=int(row["frontier_single_layer_invocations"]),
            )
        )

    top_rows = summary["trend_summary"]["top_bucket_errors_by_abs_gap"][:top_bucket_rows_in_md]
    lines.extend(
        [
            "",
            f"### Bucket-Level Hotspots (Top {len(top_rows)} by abs gap)",
            "",
            "| Op | Phase | Batch size | Total tokens | Context | vLLM metric (ms) | Frontier metric (ms) | Relative error (%) | vLLM inv | Frontier inv |",
            "|----|-------|-----------:|-------------:|---------|-----------------:|---------------------:|-------------------:|---------:|-------------:|",
        ]
    )
    for row in top_rows:
        lines.append(
            "| {op} | {phase} | {batch_size} | {total_tokens} | {context} | {vllm} | {frontier} | {rel} | {v_inv} | {f_inv} |".format(
                op=row["op_name"],
                phase=str(row.get("phase", PHASE_UNKNOWN)),
                batch_size=int(row["batch_size"]),
                total_tokens=int(row["total_tokens"]),
                context=_context_label(str(row["context"])),
                vllm=_format_optional_float(row["vllm_metric_ms"]),
                frontier=_format_optional_float(row["frontier_metric_ms"]),
                rel=_format_optional_float(row["relative_error_percent"]),
                v_inv=int(row["vllm_invocations"]),
                f_inv=int(row["frontier_invocations"]),
            )
        )

    lines.extend(
        [
            "",
            "### Batch-Size Trend Summary",
            "",
            "| Context | Batch size | vLLM metric sum (ms) | Frontier metric sum (ms) | Relative error (%) | Rows |",
            "|---------|-----------:|---------------------:|-------------------------:|-------------------:|-----:|",
        ]
    )
    for row in summary["trend_summary"]["batch_size_error_summary"]:
        lines.append(
            "| {context} | {batch_size} | {vllm:.6f} | {frontier:.6f} | {rel} | {rows} |".format(
                context=_context_label(str(row["context"])),
                batch_size=int(row["batch_size"]),
                vllm=float(row["vllm_metric_sum_ms"]),
                frontier=float(row["frontier_metric_sum_ms"]),
                rel=_format_optional_float(row["relative_error_percent"]),
                rows=int(row["num_rows"]),
            )
        )

    lines.append("")
    lines.append(
        f"Note: Full bucket context rows are available in JSON (`bucket_context_rows`, count={summary['num_bucket_context_rows']})."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()

    if args.top_bucket_rows_in_md <= 0:
        raise ValueError("--top-bucket-rows-in-md must be positive")

    op_name_map = OP_NAME_MAP_BY_PROFILE[args.model_profile]
    fused_add_norm_scope_enabled = _resolve_fused_add_norm_scope_enabled(
        model_profile=args.model_profile,
        fused_add_norm_scope=args.fused_add_norm_scope,
    )
    non_actionable_ops = (
        set(FUSED_ADD_NORM_NESTED_OPS) if fused_add_norm_scope_enabled else set()
    )
    vllm_override_specs = _parse_vllm_op_override_specs(args.vllm_op_log_override)
    vllm_logical_invocations = _collect_vllm_logical_invocations(
        args.vllm_op_log,
        args.vllm_batch_log,
    )
    vllm_logical_invocations = _apply_vllm_op_overrides(
        vllm_logical_invocations,
        vllm_override_specs,
    )
    vllm_totals, vllm_bucket_metrics, bucket_phase_map = _summarize_vllm_ops_from_logical_invocations(
        vllm_logical_invocations,
        request_level_decode_bucketing=bool(args.vllm_clean_batch_log),
        collective_bucket_mode=args.collective_bucket_mode,
    )
    vllm_collective_totals, vllm_collective_bucket_metrics = (
        _build_vllm_collective_component_summary(
            vllm_logical_invocations,
            request_level_decode_bucketing=(
                bool(args.vllm_clean_batch_log)
                and args.collective_bucket_mode == "request_level"
            ),
        )
    )
    vllm_collective_scope_audit = _build_vllm_collective_scope_audit(
        vllm_logical_invocations
    )
    collective_scope_by_op = {
        str(row["op_name"]): row
        for row in vllm_collective_scope_audit.get("ops", [])
    }
    frontier_trace_events = _load_frontier_trace_events(args.frontier_op_traces)
    frontier_totals, frontier_bucket_metrics = _summarize_frontier_ops_from_trace_events(
        frontier_trace_events,
        model_profile=args.model_profile,
        request_level_decode_bucketing=bool(args.vllm_clean_batch_log),
    )
    (
        frontier_collective_component_summary,
        frontier_collective_totals,
        frontier_collective_bucket_metrics,
    ) = _build_frontier_collective_component_summary(frontier_trace_events)
    (
        collective_scope_alignment_mode,
        collective_scope_actionability,
        collective_scope_alignment_missing_frontier_wait_ops,
    ) = _resolve_collective_scope_alignment(
        vllm_collective_scope_audit=vllm_collective_scope_audit,
        frontier_collective_component_summary=frontier_collective_component_summary,
    )

    mismatch_effective_count, gate_enabled = _load_mismatch_effective_count(
        args.schedule_summary_json
    )
    comparable = mismatch_effective_count <= 0

    rows: list[dict[str, Any]] = []
    missing_ops = 0

    mapped_frontier_op: dict[str, str] = {}
    for op_name in sorted(vllm_totals):
        mapped_op = _resolve_frontier_op(
            op_name,
            frontier_totals,
            op_name_map=op_name_map,
        )
        if not mapped_op and op_name in non_actionable_ops:
            mapped_op = op_name
        mapped_frontier_op[op_name] = mapped_op

        vllm_ms = float(vllm_totals[op_name]["full_layer_total_ms"])
        frontier_ms = (
            float(
                frontier_totals.get(mapped_op, _finalize_metrics(_new_metrics()))[
                    "full_layer_total_ms"
                ]
            )
            if mapped_op
            else 0.0
        )

        relative_error: float | None = None
        if op_name in non_actionable_ops:
            relative_error = None
        elif mapped_op:
            relative_error = _safe_rel_error(vllm_ms, frontier_ms)
        else:
            missing_ops += 1

        row = {
            "op_name": op_name,
            "vllm_total_ms": vllm_ms,
            "frontier_total_ms": frontier_ms,
            "relative_error_percent": relative_error,
            "mapped_frontier_op": mapped_op,
            "vllm_collective_scope_classification": collective_scope_by_op.get(
                op_name, {}
            ).get("classification"),
            "vllm_collective_scope_mean_spread_ms": collective_scope_by_op.get(
                op_name, {}
            ).get("mean_spread_ms"),
            "vllm_collective_scope_max_spread_ms": collective_scope_by_op.get(
                op_name, {}
            ).get("max_spread_ms"),
            **_measurement_scope_actionability_fields(op_name, non_actionable_ops),
        }
        _augment_total_row_with_collective_alignment(
            row,
            collective_scope_by_op=collective_scope_by_op,
            vllm_collective_totals=vllm_collective_totals,
            frontier_collective_totals=frontier_collective_totals,
            collective_scope_actionability=collective_scope_actionability,
            missing_frontier_wait_ops=collective_scope_alignment_missing_frontier_wait_ops,
        )
        rows.append(row)

    if not rows:
        raise ValueError("no comparable op rows were produced")

    context_rows, invocation_stats_by_op = _build_context_rows(
        vllm_totals,
        frontier_totals,
        mapped_frontier_op,
        non_actionable_ops=non_actionable_ops,
    )
    bucket_context_rows = _build_bucket_context_rows(
        vllm_bucket_metrics,
        frontier_bucket_metrics,
        mapped_frontier_op,
        bucket_phase_map,
        non_actionable_ops=non_actionable_ops,
    )
    for collection in (context_rows, invocation_stats_by_op, bucket_context_rows):
        for row in collection:
            op_name = str(row.get("op_name", ""))
            collective_scope_row = collective_scope_by_op.get(op_name)
            if collective_scope_row is None:
                continue
            row["vllm_collective_scope_classification"] = collective_scope_row.get(
                "classification"
            )
            row["vllm_collective_scope_mean_spread_ms"] = collective_scope_row.get(
                "mean_spread_ms"
            )
            row["vllm_collective_scope_max_spread_ms"] = collective_scope_row.get(
                "max_spread_ms"
            )
    for row in context_rows:
        _augment_context_row_with_collective_alignment(
            row,
            collective_scope_by_op=collective_scope_by_op,
            vllm_collective_metrics=vllm_collective_totals,
            frontier_collective_metrics=frontier_collective_totals,
            collective_scope_actionability=collective_scope_actionability,
            missing_frontier_wait_ops=collective_scope_alignment_missing_frontier_wait_ops,
        )
    for row in bucket_context_rows:
        _augment_context_row_with_collective_alignment(
            row,
            collective_scope_by_op=collective_scope_by_op,
            vllm_collective_metrics=vllm_collective_bucket_metrics,
            frontier_collective_metrics=frontier_collective_bucket_metrics,
            collective_scope_actionability=collective_scope_actionability,
            missing_frontier_wait_ops=collective_scope_alignment_missing_frontier_wait_ops,
        )

    compared_rel_errors = [
        float(row["relative_error_percent"])
        for row in rows
        if row["relative_error_percent"] is not None
    ]
    trend_summary = _build_trend_summary(bucket_context_rows)
    prefill_only_summary = _build_prefill_only_summary(bucket_context_rows)

    vllm_decode_batch_metrics = _load_vllm_decode_batch_metrics(args.vllm_batch_log)
    frontier_decode_batch_metrics = _load_frontier_decode_batch_metrics(frontier_trace_events)
    decode_batch_execution_summary = _build_decode_batch_execution_summary(
        vllm_batch_metrics=vllm_decode_batch_metrics,
        frontier_batch_metrics=frontier_decode_batch_metrics,
        comparable=comparable,
        threshold_percent=args.threshold_percent,
    )
    decode_batch_execution_clean_summary: dict[str, Any] | None = None
    if args.vllm_clean_batch_log is not None:
        vllm_decode_batch_metrics_clean = _load_vllm_decode_batch_metrics(
            args.vllm_clean_batch_log
        )
        decode_batch_execution_clean_summary = _build_decode_batch_execution_summary(
            vllm_batch_metrics=vllm_decode_batch_metrics_clean,
            frontier_batch_metrics=frontier_decode_batch_metrics,
            comparable=comparable,
            threshold_percent=args.threshold_percent,
        )
    decode_phase_observability = _build_decode_phase_observability(
        vllm_logical_invocations=vllm_logical_invocations,
        frontier_batch_metrics=frontier_decode_batch_metrics,
        bucket_context_rows=bucket_context_rows,
        decode_batch_execution_summary=decode_batch_execution_summary,
        vllm_batch_log_provided=bool(args.vllm_batch_log),
    )

    full_context_rel_errors = [
        float(row["relative_error_percent"])
        for row in context_rows
        if row["context"] == FULL_LAYER_CONTEXT and row["relative_error_percent"] is not None
    ]
    single_context_rel_errors = [
        float(row["relative_error_percent"])
        for row in context_rows
        if row["context"] == SINGLE_LAYER_CONTEXT and row["relative_error_percent"] is not None
    ]

    mean_rel_error = statistics.mean(compared_rel_errors) if compared_rel_errors else None
    max_rel_error = max(compared_rel_errors) if compared_rel_errors else None
    single_mean_rel_error = (
        statistics.mean(single_context_rel_errors) if single_context_rel_errors else None
    )
    single_max_rel_error = max(single_context_rel_errors) if single_context_rel_errors else None

    if comparable:
        status = "PASS"
        if compared_rel_errors:
            if mean_rel_error is not None and mean_rel_error >= args.threshold_percent:
                status = "FAIL"
            if max_rel_error is not None and max_rel_error >= args.threshold_percent:
                status = "FAIL"
        else:
            status = "FAIL"
        non_comparable_reason = None
    else:
        status = "NON_COMPARABLE"
        non_comparable_reason = "non-comparable due to path divergence"

    summary: dict[str, Any] = {
        "status": status,
        "model_profile": args.model_profile,
        "fused_add_norm_scope_enabled": fused_add_norm_scope_enabled,
        "num_non_actionable_ops": len(
            [op_name for op_name in vllm_totals if op_name in non_actionable_ops]
        ),
        "non_actionable_op_names": sorted(
            op_name for op_name in vllm_totals if op_name in non_actionable_ops
        ),
        "comparable": comparable,
        "non_comparable_reason": non_comparable_reason,
        "path_divergence_gate_enabled": gate_enabled,
        "mismatch_effective_count": mismatch_effective_count,
        "threshold_percent": args.threshold_percent,
        "collective_bucket_mode": args.collective_bucket_mode,
        "num_ops": len(rows),
        "num_compared_ops": len(compared_rel_errors),
        "num_missing_ops": missing_ops,
        "mean_relative_error_percent": mean_rel_error,
        "max_relative_error_percent": max_rel_error,
        "single_layer_mean_relative_error_percent": single_mean_rel_error,
        "single_layer_max_relative_error_percent": single_max_rel_error,
        "prefill_only_mean_relative_error_percent": prefill_only_summary[
            "mean_relative_error_percent"
        ],
        "prefill_only_max_relative_error_percent": prefill_only_summary[
            "max_relative_error_percent"
        ],
        "rows": rows,
        "context_rows": context_rows,
        "invocation_stats_by_op": invocation_stats_by_op,
        "bucket_context_rows": bucket_context_rows,
        "num_bucket_context_rows": len(bucket_context_rows),
        "vllm_op_override_sources": [
            {
                "op_names": list(spec["op_names"]),
                "probe_dir": str(spec["probe_dir"]),
            }
            for spec in vllm_override_specs
        ],
        "trend_summary": trend_summary,
        "prefill_only_summary": prefill_only_summary,
        "decode_phase_observability": decode_phase_observability,
        "decode_batch_execution_summary": decode_batch_execution_summary,
        "vllm_collective_scope_audit": vllm_collective_scope_audit,
        "frontier_collective_component_summary": frontier_collective_component_summary,
        "collective_scope_alignment_mode": collective_scope_alignment_mode,
        "collective_scope_actionability": collective_scope_actionability,
        "collective_scope_alignment_missing_frontier_wait_ops": (
            collective_scope_alignment_missing_frontier_wait_ops
        ),
        "collective_scope_runtime_wrapper_op_names": list(
            vllm_collective_scope_audit.get("runtime_wrapper_op_names", [])
        ),
    }
    if decode_batch_execution_clean_summary is not None:
        summary["decode_batch_execution_clean_summary"] = (
            decode_batch_execution_clean_summary
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_md.write_text(
        _build_markdown(summary, top_bucket_rows_in_md=args.top_bucket_rows_in_md),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
