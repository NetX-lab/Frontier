"""Turn saved DeepSeek-V4 graph-replay records into profiling rows on CPU.

A row aggregates layer calls of one case, variant and tile-planner role that
share dispatch, scheduled requests, native variant metadata and observed
family scopes. The scope set comes from the calls, including captured CSA
paths that skip index scores. Dummy rows describe their native layout through
dispatch and metadata, without inventing request or key-length columns.

Each scope carries kernel-only cost statistics and the signed gaps before its
first kernel and between its kernels. Layer-window statistics retain A4's
host-ahead target, computed by the same timing helper. All times are ms.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from frontier.attention.families import DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT
from frontier.profiling.attention.dsa_plan import ENGINE_POLICY_GRAPH_REPLAY, PHASE_DUMMY
from frontier.profiling.attention.dsa_rows import (
    KERNEL_ONLY, _RowAccumulator, _request_signature, _requests, _stat_columns,
    request_load_columns, scope_columns,
)
from frontier.profiling.attention.dsv4_attention_plan import (
    CSA_VARIANT, Dsv4ProfilePlan, WARMUP_ONLY_OPERATORS,
    require_engine_layer_inventory, require_realized_kv_cache_groups,
)
from frontier.profiling.attention.dsv4_attention_rows import (
    _key_length_columns, _model_columns, _variant_metadata,
)
from frontier.profiling.common.runtime_identity import canonical_json
from frontier.profiling.kernel_gap.dsv4_replay_gaps import (
    ACTIVITY, CASE_ID, CG_MODE, CUDAGRAPH_DISPATCH_KEY, ENGINE_CAPTURE, ENGINE_INDEX,
    ENGINE_MODE, GLOBAL_LAYER_ID, KERNEL, NUM_TOKENS, SCOPE, STEP_INDEX,
    LayerCall, ReplayRecords, activity_gaps, attention_gaps_by_layer,
    build_layer_calls, layer_call_timing, validate_identities,
)

DSV4_REPLAY_OP_NAME = "attention_graph_replay"
DSV4_REPLAY_ROW_SCHEMA_VERSION = "dsv4_attention_replay.v1"
GAP_STATS_PREFIX = "gap_stats"
LAYER_WINDOW_STATS_PREFIX = "layer_window_stats"
NUM_REQUESTS = "num_reqs"
MEASUREMENT = "measurement"
TIME_MS, KERNEL_COUNT = "time_ms", "kernel_count"
INDEX_SCORES_SCOPE = "dsv4_index_scores"


@dataclass
class _ReplayRowAccumulator(_RowAccumulator):
    gaps: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    windows: list[float] = field(default_factory=list)
    label: str = ""


def _call_label(call: LayerCall) -> str:
    return (f"Replay {call.source}, case {call.step[CASE_ID]}, step {call.step[STEP_INDEX]}, "
            f"layer {call.global_layer_id}")


def _validated_calls(plan: Dsv4ProfilePlan, record: ReplayRecords) -> tuple[LayerCall, ...]:
    """Validate the recorded workload before deriving activity dependencies."""
    for row in (*record.steps, *record.calls):
        label = f"Replay {record.source}, case {row[CASE_ID]}, step {row[STEP_INDEX]}"
        if GLOBAL_LAYER_ID in row:
            label += f", layer {row[GLOBAL_LAYER_ID]}"
        if row.get(MEASUREMENT) != KERNEL_ONLY:
            raise ValueError(f"{label}: requires measurement={KERNEL_ONLY}, got {row.get(MEASUREMENT)!r}")
    calls = build_layer_calls((record,))
    by_step = defaultdict(dict)
    for call in calls:
        label = _call_label(call)
        if not call.calls:
            raise ValueError(f"{label}: missing layer call")
        scopes = [row[SCOPE] for row in call.calls]
        family = {operator.name for operator in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT[call.variant]}
        for scope in scopes:
            if scope in WARMUP_ONLY_OPERATORS:
                raise ValueError(f"{label}: warmup-only scope {scope!r}")
            if scope not in family:
                raise ValueError(f"{label}: non-family scope {scope!r} for {call.variant}")
        if len(set(scopes)) != len(scopes):
            raise ValueError(f"{label}: repeated call scope")
        counts = Counter(row[SCOPE] for row in call.intervals if row[ACTIVITY] == KERNEL)
        for row in call.calls:
            if counts[row[SCOPE]] != row[KERNEL_COUNT]:
                raise ValueError(f"{label}: scope {row[SCOPE]} kernel_count={row[KERNEL_COUNT]} "
                                 f"differs from owned kernels {counts[row[SCOPE]]}")
        if set(counts) - set(scopes):
            raise ValueError(f"{label}: owned kernel scopes lack call records {sorted(set(counts) - set(scopes))}")
        by_step[(call.step[CASE_ID], call.step[STEP_INDEX])][call.global_layer_id] = call
    for step in record.steps:
        if step[ENGINE_INDEX] != record.engine[ENGINE_INDEX]:
            raise ValueError(f"Replay {record.source}, case {step[CASE_ID]}, step {step[STEP_INDEX]}: "
                             "step engine_index differs from its engine")
        present = by_step[(step[CASE_ID], step[STEP_INDEX])]
        for layer in plan.shape.layers:
            if layer.global_layer_id not in present:
                raise ValueError(f"Replay {record.source}, case {step[CASE_ID]}, step {step[STEP_INDEX]}, "
                                 f"layer {layer.global_layer_id}: missing layer call")
    return calls


def build_dsv4_replay_rows(
    plan: Dsv4ProfilePlan, records: Sequence[ReplayRecords],
) -> list[dict[str, Any]]:
    """Aggregate valid held replay calls, retaining per-scope signed gaps."""
    sources = ", ".join(record.source for record in records)
    if plan.limits.engine_policy != ENGINE_POLICY_GRAPH_REPLAY:
        raise ValueError(f"Replay {sources}: requires plan engine_policy=graph_replay, "
                         f"got {plan.limits.engine_policy!r}")
    seen_engines, routing_identities = set(), set()
    for record in records:
        engine = record.engine
        index = engine[ENGINE_INDEX]
        if index in seen_engines:
            raise ValueError(f"Replay {record.source}: engine index {index} recorded twice")
        seen_engines.add(index)
        try:
            mode = engine[ENGINE_CAPTURE][ENGINE_MODE]
            if mode != plan.limits.cudagraph_mode:
                raise ValueError(f"engine cudagraph_mode={mode!r} differs from plan {plan.limits.cudagraph_mode!r}")
            require_engine_layer_inventory(plan, engine)
            require_realized_kv_cache_groups(plan, engine)
        except (ValueError, KeyError) as error:
            raise ValueError(f"Replay {record.source}: {error}") from error
        routing_identities.add(canonical_json(engine.get("moe_routing")))
        if len(routing_identities) > 1:
            raise ValueError(f"Replay {record.source}: raw engines disagree on constructed routing identity")

    validate_identities(records)
    groups: dict[tuple, _ReplayRowAccumulator] = {}
    for record in records:
        try:
            calls = _validated_calls(plan, record)
            by_layer = attention_gaps_by_layer(activity_gaps((record,)))
        except ValueError as error:
            raise ValueError(f"Replay {record.source}: {error}") from error
        for call in calls:
            label = _call_label(call)
            timing = layer_call_timing(by_layer.get(
                (call.source, call.step[CASE_ID], call.step[STEP_INDEX], call.global_layer_id), (),
            ))
            if timing.exclusion_reason is not None:
                raise ValueError(f"{label}: {timing.exclusion_reason}")
            layer = plan.layer(call.global_layer_id)
            step, engine, variant = call.step, call.engine, call.variant
            dispatch = step[CUDAGRAPH_DISPATCH_KEY]
            scopes = frozenset(row[SCOPE] for row in call.calls)
            dummy = step["phase"] == PHASE_DUMMY
            try:
                requests = () if dummy else _requests(step)
                metadata = _variant_metadata(step, variant)
                load = {} if dummy else request_load_columns(requests)
                model = _model_columns(plan, engine)
            except ValueError as error:
                raise ValueError(f"{label}: {error}") from error
            index_scores = variant == CSA_VARIANT and INDEX_SCORES_SCOPE in scopes
            key = (engine[ENGINE_INDEX], step[CASE_ID], variant, layer.plans_tile_schedule,
                   dispatch[CG_MODE], dispatch[NUM_TOKENS], dispatch[NUM_REQUESTS],
                   _request_signature(requests), canonical_json(metadata), scopes)
            accumulator = groups.get(key)
            if accumulator is None:
                template = {
                    "measurement_type": KERNEL_ONLY,
                    "case_id": step[CASE_ID], "phase": step["phase"],
                    "attention_variant": variant, "compress_ratio": max(1, layer.compress_ratio),
                    "plans_tile_schedule": layer.plans_tile_schedule,
                    "engine_cudagraph_mode": engine[ENGINE_CAPTURE][ENGINE_MODE],
                    "graph_runtime_mode": dispatch[CG_MODE],
                    "padded_rows": dispatch[NUM_TOKENS],
                    "dispatch_num_requests": dispatch[NUM_REQUESTS] if dispatch[NUM_REQUESTS] is not None else "",
                    "indexer_short_context": call.short_context,
                    "index_scores_executed": index_scores,
                    **model, **load,
                    **({} if dummy else _key_length_columns(plan, layer, load["max_key_len"], index_scores)),
                    **{f"{source}_metadata.{name}": value for source, fields in metadata.items()
                       for name, value in sorted(fields.items())},
                }
                accumulator = groups[key] = _ReplayRowAccumulator(template, label=label)
            for row in call.calls:
                scope = row[SCOPE]
                accumulator.times[scope].append(float(row[TIME_MS]))
                accumulator.kernel_counts[scope].add(int(row[KERNEL_COUNT]))
                accumulator.gaps[scope].append(timing.scope_gaps_ms.get(scope, 0.0))
            accumulator.windows.append(timing.window_ms)
            accumulator.layers.add(layer.global_layer_id)
            accumulator.steps.add(step[STEP_INDEX])

    rows = []
    for key, accumulator in sorted(groups.items(), key=lambda item: (
        item[0][1], item[0][2], not item[0][3], min(item[1].steps),
    )):
        row = dict(accumulator.template)
        layers = sorted(accumulator.layers)
        row.update(global_layer_ids=json.dumps(layers), num_layers_sampled=len(layers),
                   num_steps_sampled=len(accumulator.steps))
        scopes = [operator.name for operator in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT[key[2]]
                  if operator.name in accumulator.times]
        row["operator_scopes"] = canonical_json({scope: [scope] for scope in scopes})
        for scope in scopes:
            row.update(scope_columns(scope, accumulator.times[scope],
                                     accumulator.kernel_counts[scope], accumulator.label))
            row.update(_stat_columns(scope, accumulator.gaps[scope], prefix=GAP_STATS_PREFIX))
        row.update(_stat_columns("", accumulator.windows, prefix=LAYER_WINDOW_STATS_PREFIX))
        rows.append(row)
    return rows


__all__ = ["DSV4_REPLAY_OP_NAME", "DSV4_REPLAY_ROW_SCHEMA_VERSION", "build_dsv4_replay_rows"]
