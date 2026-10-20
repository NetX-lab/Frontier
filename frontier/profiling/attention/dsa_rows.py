"""Turn raw DSA runner records into predictor-facing profiling rows.

The runner writes one JSON object per line:

* ``engine``: the engine's chunk budget, selected attention backend and
  implementation class, KV-cache dtype, layer inventory, and runtime identity;
* ``step``: one measured engine step with its scheduled requests and the
  scalar fields of the native sparse-MLA and indexer attention metadata;
* ``call``: the time of one scope (``attn_dsa_indexer`` or ``attn_dsa_mla``)
  of one layer in one measured step.

A row aggregates every call of one scope over the reduced model's layers of
one attention variant and over the steps of one case that share the same
scheduled requests and native metadata. Shared-index layers do not run the
indexer, so their rows carry no indexer timing. All times are milliseconds.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from frontier.attention.dsa_features import DSA_ROW_SCHEMA_VERSION
from frontier.config.precision_type import PrecisionType
from frontier.profiling.attention.dsa_plan import (
    DSA_SCOPES,
    ENGINE_POLICY_EAGER_CHUNKED,
    MLA_SCOPE,
    DsaProfilePlan,
    require_scope_timing,
)
from frontier.profiling.common.runtime_identity import (
    RuntimeIdentity,
    attach_identity_columns,
    canonical_json,
)

CUDA_EVENT = "CUDA_EVENT"
KERNEL_ONLY = "KERNEL_ONLY"
MEASUREMENT_TYPES = (CUDA_EVENT, KERNEL_ONLY)

# Native metadata fields every recorded step must carry.
REQUIRED_MLA_METADATA_FIELDS = (
    "num_actual_tokens",
    "num_decodes",
    "num_prefills",
    "num_decode_tokens",
)


@dataclass(frozen=True)
class StepRequest:
    num_scheduled_tokens: int
    num_computed_tokens: int
    prompt_len: int

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.prompt_len


@dataclass
class _RowAccumulator:
    template: dict[str, Any]
    times: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    kernel_counts: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    layers: set[int] = field(default_factory=set)
    steps: set[int] = field(default_factory=set)


def expected_scopes(plan: DsaProfilePlan, global_layer_id: int) -> tuple[str, ...]:
    layer = plan.layer(global_layer_id)
    return DSA_SCOPES if layer.indexer_executed else (MLA_SCOPE,)


def _requests(step: Mapping[str, Any]) -> tuple[StepRequest, ...]:
    requests = tuple(StepRequest(**request) for request in step["requests"])
    if not requests:
        raise ValueError(f"Step {step['step_index']} of {step['case_id']} has no requests")
    return requests


def _request_signature(requests: Sequence[StepRequest]) -> tuple:
    prefills = sorted(
        (r.num_scheduled_tokens, r.num_computed_tokens) for r in requests if r.is_prefill
    )
    decodes = sorted(
        (r.num_scheduled_tokens, r.num_computed_tokens) for r in requests if not r.is_prefill
    )
    return tuple(prefills), tuple(decodes)


def _metadata_signature(step: Mapping[str, Any]) -> str:
    mla = step["mla_metadata"]
    missing = [name for name in REQUIRED_MLA_METADATA_FIELDS if name not in mla]
    if missing:
        raise ValueError(
            f"Step {step['step_index']} of {step['case_id']} lacks native MLA "
            f"metadata fields {missing}"
        )
    return canonical_json({"mla": mla, "indexer": step["indexer_metadata"]})


def batch_load_columns(
    requests: Sequence[StepRequest],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the step's load columns and its context-size columns.

    Model-specific columns sit between the two groups in the DSA tables.
    """

    prefills = [r for r in requests if r.is_prefill]
    decodes = [r for r in requests if not r.is_prefill]
    for request in decodes:
        if request.num_scheduled_tokens != 1:
            raise ValueError(
                "A request past its prompt must schedule one decode token, got "
                f"{request.num_scheduled_tokens}"
            )
    decode_contexts = sorted(r.num_computed_tokens for r in decodes)
    load: dict[str, Any] = {
        "is_prefill": bool(prefills) and not decodes,
        "is_mixed_batch": bool(prefills) and bool(decodes),
        "batch_size": len(requests),
        "total_tokens": sum(r.num_scheduled_tokens for r in requests),
        "num_prefill_seqs": len(prefills),
        "num_decode_seqs": len(decodes),
        "total_prefill_tokens": sum(r.num_scheduled_tokens for r in prefills),
        "total_decode_tokens": len(decodes),
        "prefill_seq_lens": json.dumps([r.num_scheduled_tokens for r in prefills]),
        "prefill_kv_cache_sizes": json.dumps([r.num_computed_tokens for r in prefills]),
        "decode_kv_cache_sizes": json.dumps(decode_contexts),
        "max_key_len": max(r.num_computed_tokens + r.num_scheduled_tokens for r in requests),
    }
    context: dict[str, Any]
    if prefills and not decodes:
        context = {
            "prefill_chunk_size": sum(r.num_scheduled_tokens for r in prefills),
            "kv_cache_size": sum(r.num_computed_tokens for r in prefills) // len(prefills),
        }
    elif not prefills:
        context = {
            "prefill_chunk_size": 0,
            "kv_cache_size": sum(decode_contexts) // len(decode_contexts),
            "decode_kv_cache_size_min": decode_contexts[0],
            "decode_kv_cache_size_max": decode_contexts[-1],
            "decode_kv_cache_size_mean": statistics.fmean(decode_contexts),
        }
    else:
        raise ValueError("Measured DSA steps are pure prefill or pure decode")
    return load, context


def _batch_columns(
    requests: Sequence[StepRequest], index_topk: int
) -> dict[str, Any]:
    load, context = batch_load_columns(requests)
    return {
        **load,
        "max_selected_key_len": min(load["max_key_len"], index_topk),
        **context,
    }


def _model_columns(plan: DsaProfilePlan, engine: Mapping[str, Any]) -> dict[str, Any]:
    shape = plan.shape
    return {
        "model_name": shape.model_name,
        "model_arch": shape.model_arch,
        "model_architecture_profile": shape.model_architecture_profile,
        "attention_family": "dsa_attention",
        "attention_backend": engine["attention_backend"],
        "attention_impl": engine["attention_impl"],
        "profiling_precision": PrecisionType.from_string(shape.dtype).name,
        "n_embd": shape.hidden_size,
        "n_q_head": shape.num_attention_heads,
        "n_q_head_per_rank": shape.heads_per_rank,
        "num_tensor_parallel_workers": shape.target_tensor_parallel_size,
        "q_lora_rank": shape.q_lora_rank,
        "kv_lora_rank": shape.kv_lora_rank,
        "qk_nope_head_dim": shape.qk_nope_head_dim,
        "qk_rope_head_dim": shape.qk_rope_head_dim,
        "v_head_dim": shape.v_head_dim,
        "index_topk": shape.index_topk,
        "index_n_heads": shape.index_n_heads,
        "index_head_dim": shape.index_head_dim,
        "block_size": plan.limits.block_size,
        "kv_cache_dtype": engine["kv_cache_dtype"],
        "max_model_len": plan.limits.max_context_len,
        "max_num_batched_tokens": engine["chunk_budget"],
        "max_num_seqs": plan.limits.max_num_seqs,
    }


def _stat_columns(scope: str, values: Sequence[float]) -> dict[str, Any]:
    return {
        f"time_stats.{scope}.min": min(values),
        f"time_stats.{scope}.max": max(values),
        f"time_stats.{scope}.mean": statistics.fmean(values),
        f"time_stats.{scope}.median": statistics.median(values),
        f"time_stats.{scope}.std": statistics.pstdev(values),
        f"time_stats.{scope}.count": len(values),
    }


def scope_columns(
    scope: str, values: Sequence[float], kernel_counts: set[int] | None, row_label: Any
) -> dict[str, Any]:
    """Time statistics of one scope plus its kernel count when kernel-only."""

    columns = _stat_columns(scope, values)
    if kernel_counts:
        if len(kernel_counts) > 1:
            raise ValueError(
                f"Scope {scope} in row {row_label} launched different "
                f"kernel counts {sorted(kernel_counts)}; these calls measure "
                "different work"
            )
        columns[f"time_stats.{scope}.kernel_count"] = next(iter(kernel_counts))
    return columns


def index_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[tuple, Mapping[str, Any]],
    dict[tuple, list[Mapping[str, Any]]],
]:
    """Index raw records: engines by index; steps and calls by step key.

    A step key is ``(measurement, engine_index, step_index)``.
    """

    engines: dict[int, Mapping[str, Any]] = {}
    steps: dict[tuple, Mapping[str, Any]] = {}
    calls: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        kind = record["kind"]
        if kind == "engine":
            require_scope_timing(record.get("engine_policy", ENGINE_POLICY_EAGER_CHUNKED))
            if record["engine_index"] in engines:
                raise ValueError(f"Engine {record['engine_index']} recorded twice")
            engines[record["engine_index"]] = record
            continue
        key = (record["measurement"], record["engine_index"], record["step_index"])
        if kind == "step":
            if key in steps:
                raise ValueError(f"Step {key} recorded twice")
            steps[key] = record
        elif kind == "call":
            calls[key].append(record)
        else:
            raise ValueError(f"Unknown DSA record kind {kind!r}")

    orphan_calls = sorted(set(calls) - set(steps))
    if orphan_calls:
        raise ValueError(f"Calls without a step record, first {orphan_calls[0]}")
    for key, step in steps.items():
        if key[0] not in MEASUREMENT_TYPES:
            raise ValueError(f"Unknown measurement type {key[0]!r}")
        if key[1] not in engines:
            raise ValueError(f"Step {key} has no engine record")
    return engines, steps, calls


def _validate_step_calls(
    plan: DsaProfilePlan, step_key: tuple, calls: Sequence[Mapping[str, Any]]
) -> None:
    seen: dict[int, list[str]] = defaultdict(list)
    for call in calls:
        seen[call["global_layer_id"]].append(call["scope"])
    for layer in plan.shape.layers:
        observed = sorted(seen.pop(layer.global_layer_id, []))
        expected = sorted(expected_scopes(plan, layer.global_layer_id))
        if observed != expected:
            raise ValueError(
                f"Step {step_key}: layer {layer.global_layer_id} "
                f"({layer.attention_variant}) ran scopes {observed}, expected {expected}"
            )
    if seen:
        raise ValueError(f"Step {step_key}: calls for unknown layers {sorted(seen)}")


def build_dsa_rows(
    plan: DsaProfilePlan, records: Iterable[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Group raw records into rows, keyed by measurement type."""

    require_scope_timing(plan.limits.engine_policy)
    engines, steps, calls = index_records(records)
    groups: dict[tuple, _RowAccumulator] = {}
    for key, step in sorted(steps.items()):
        measurement, engine_index, step_index = key
        engine = engines[engine_index]
        step_calls = calls.get(key, [])
        _validate_step_calls(plan, key, step_calls)
        requests = _requests(step)
        metadata = _metadata_signature(step)
        signature = _request_signature(requests)
        for call in step_calls:
            layer = plan.layer(call["global_layer_id"])
            group_key = (
                measurement,
                engine_index,
                step["case_id"],
                layer.attention_variant,
                signature,
                metadata,
            )
            accumulator = groups.get(group_key)
            if accumulator is None:
                template = {
                    "measurement_type": measurement,
                    "case_id": step["case_id"],
                    "phase": step["phase"],
                    "attention_variant": layer.attention_variant,
                    "indexer_executed": layer.indexer_executed,
                    **_model_columns(plan, engine),
                    **_batch_columns(requests, plan.shape.index_topk),
                    **{
                        f"mla_metadata.{name}": value
                        for name, value in sorted(step["mla_metadata"].items())
                    },
                    **{
                        f"indexer_metadata.{name}": value
                        for name, value in sorted(step["indexer_metadata"].items())
                    },
                }
                accumulator = groups[group_key] = _RowAccumulator(template)
            accumulator.times[call["scope"]].append(float(call["time_ms"]))
            if measurement == KERNEL_ONLY:
                accumulator.kernel_counts[call["scope"]].add(int(call["kernel_count"]))
            accumulator.layers.add(layer.global_layer_id)
            accumulator.steps.add(step_index)

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in MEASUREMENT_TYPES}
    for group_key, accumulator in groups.items():
        row = dict(accumulator.template)
        layers = sorted(accumulator.layers)
        row["global_layer_ids"] = json.dumps(layers)
        row["num_layers_sampled"] = len(layers)
        row["index_producer_layers"] = json.dumps(
            sorted({plan.layer(layer).index_producer_layer for layer in layers})
        )
        row["num_steps_sampled"] = len(accumulator.steps)
        for scope in DSA_SCOPES:
            values = accumulator.times.get(scope)
            if not values:
                continue
            row.update(
                scope_columns(
                    scope, values, accumulator.kernel_counts.get(scope), group_key[:5]
                )
            )
        rows[group_key[0]].append(row)
    for measurement in MEASUREMENT_TYPES:
        rows[measurement].sort(
            key=lambda row: (
                row["case_id"],
                row["attention_variant"],
                row["prefill_kv_cache_sizes"],
                row["decode_kv_cache_sizes"],
            )
        )
    return rows


def write_rows_csv(
    rows: Sequence[Mapping[str, Any]], identity: RuntimeIdentity, path: str | Path
) -> Path:
    """Write rows with identity columns; column order follows first appearance."""

    if not rows:
        raise ValueError(f"No DSA rows to write to {path}")
    attached = attach_identity_columns(rows, identity)
    columns: list[str] = []
    for row in attached:
        for name in row:
            if name not in columns:
                columns.append(name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(attached)
    return path


def read_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


__all__ = [
    "CUDA_EVENT",
    "DSA_ROW_SCHEMA_VERSION",
    "KERNEL_ONLY",
    "MEASUREMENT_TYPES",
    "REQUIRED_MLA_METADATA_FIELDS",
    "StepRequest",
    "batch_load_columns",
    "build_dsa_rows",
    "expected_scopes",
    "index_records",
    "read_records",
    "scope_columns",
    "write_rows_csv",
]
