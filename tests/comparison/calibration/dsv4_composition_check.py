"""Check V4 attention composition against the same saved replay layer calls (A4).

Plans restore the tile-planner role absent from the raw engine inventory. Gap
means use all engine modes together, while errors are grouped by every runtime,
planner and workload coordinate. Only calls with a defined, host-ahead gap for
every owned activity count. The target is the joined kernel window plus its
first kernel's signed gap. The gate substitutes this call's own signed
kernel gaps into scope costs and gates acceptance; the class means remain diagnostic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase, segment_class
from frontier.profiling.attention import dsv4_attention_main
from frontier.profiling.attention.dsv4_attention_plan import (
    require_engine_layer_inventory, require_realized_kv_cache_groups,
)
from frontier.profiling.kernel_gap.dsv4_replay_gaps import (
    CASE_ID, CG_MODE, ENGINE_CAPTURE, ENGINE_MODE, ENGINE_INDEX, NUM_TOKENS, SCOPE,
    STEP_INDEX, TARGET_TP, THRESHOLD, US_TO_MS, CUDAGRAPH_DISPATCH_KEY,
    LayerCall, ReplayRecords, activity_gaps, build_layer_calls, read_replay_records,
    reduce_dsv4_gaps, gap_statistics, attention_gaps_by_layer, layer_call_timing,
)
from dsv4_graph_bridge_check import BridgeConfig, graph_mode_flags, load_config

MEDIAN_ERROR_LIMIT = 0.05
P90_ERROR_LIMIT = 0.10
REPORT_FILE = "composition_report.json"
PASS, FAIL = "PASS", "FAIL"
GROUP_PASS, GROUP_FAIL = "pass", "fail"
NO_HOST_AHEAD_CALL, MISSING_GAP_VALUE = "no_host_ahead_call", "missing_gap_value"
TIME_MS, KERNEL_COUNT = "time_ms", "kernel_count"


_HOST_AHEAD_FRACTION = 'host_ahead_fraction'
_SCOPE_GAPS = 'scope_gaps'
_GAP_SUM_MS = 'gap_sum_ms'
_RATIO_TO_SCOPE_COST = 'ratio_to_scope_cost'

@dataclass(frozen=True)
class CompositionGroup:
    engine_mode: str
    step_mode: str
    target_tensor_parallel_size: int
    variant: str
    plans_tile_schedule: bool
    case_id: str
    rows_at_or_below_threshold: bool
    indexer_short_context: bool


def planner_roles(config: BridgeConfig, records: Sequence[ReplayRecords]) -> Mapping[tuple[str, int, int], bool]:
    """Rebuild each mode's plan and require the exact ordered layer inventory.

    The engine module stores at least ratio 1, while the plan keeps the native
    ``compress_ratios`` entry, which is 0 for uncompressed layers.
    """
    roles = {}
    by_run = {(record.engine[ENGINE_CAPTURE][ENGINE_MODE], record.engine[ENGINE_INDEX]): record for record in records}
    for run in config.runs:
        mode = run.mode
        plan = dsv4_attention_main.build_plan(dsv4_attention_main.build_parser().parse_args(
            ["plan", *config.plan_args, *graph_mode_flags(mode, config.cudagraph_capture_sizes)]
        ))
        record = by_run[(mode, run.engine_index)]
        try:
            require_engine_layer_inventory(plan, record.engine)
        except ValueError as error:
            raise ValueError(f"A4 {mode} engine {record.source}: {error}") from error
        require_realized_kv_cache_groups(plan, record.engine)
        roles.update({(mode, run.engine_index, layer.global_layer_id): layer.plans_tile_schedule for layer in plan.shape.layers})
    return roles


def _group(call: LayerCall, roles) -> CompositionGroup:
    mode = call.engine[ENGINE_CAPTURE][ENGINE_MODE]
    dispatch = call.step[CUDAGRAPH_DISPATCH_KEY]
    return CompositionGroup(mode, dispatch[CG_MODE], call.engine[TARGET_TP], call.variant,
                            roles[(mode, call.engine[ENGINE_INDEX], call.global_layer_id)], call.step[CASE_ID],
                            dispatch[NUM_TOKENS] <= call.engine[THRESHOLD], call.short_context)


def _errors(signed_errors):
    if not signed_errors:
        return {"median_error": None, "p90_error": None, "signed_median_error": None}
    return {"median_error": float(np.median(np.abs(signed_errors))),
            "p90_error": float(np.percentile(np.abs(signed_errors), 90, method="linear")),
            "signed_median_error": float(np.median(signed_errors))}


def _status(count, errors):
    if count == 0:
        return NO_HOST_AHEAD_CALL
    return (GROUP_PASS if errors["median_error"] <= MEDIAN_ERROR_LIMIT
            and errors["p90_error"] <= P90_ERROR_LIMIT else GROUP_FAIL)


def _percentiles(values):
    stats = gap_statistics(values)
    return {key: stats[key] for key in ("median", "p10", "p90")}


def check_composition(records: Sequence[ReplayRecords], roles: Mapping[tuple[str, int, int], bool]) -> dict[str, Any]:
    """Evaluate layer calls without pooling different composition coordinates."""
    reduction = reduce_dsv4_gaps(records)
    gap_ms = {segment: value * US_TO_MS for segment, value in reduction.values_us.items()}
    by_layer = attention_gaps_by_layer(activity_gaps(records))
    grouped = defaultdict(list)
    for call in build_layer_calls(records):
        grouped[_group(call, roles)].append(call)
    groups = []
    for group, calls in grouped.items():
        exclusions, missing, errors, diagnostic_errors = Counter(), set(), [], []
        host_ahead_count = 0
        scope_gaps, scope_ratios = defaultdict(list), defaultdict(list)
        for call in calls:
            scope_ms = {row[SCOPE]: row[TIME_MS] for row in call.calls}
            counts = {row[SCOPE]: row[KERNEL_COUNT] for row in call.calls}
            required = {segment_class(group.step_mode, name) for name in scope_ms}
            missing.update(required - gap_ms.keys())
            owned = by_layer.get((call.source, call.step[CASE_ID], call.step[STEP_INDEX], call.global_layer_id), ())
            timing = layer_call_timing(owned)
            if timing.exclusion_reason is not None:
                exclusions[timing.exclusion_reason] += 1
                continue
            target_ms = timing.window_ms
            host_ahead_count += 1
            options = dict(variant=call.variant, graph_runtime_mode=group.step_mode,
                           rows=call.step[CUDAGRAPH_DISPATCH_KEY][NUM_TOKENS],
                           multi_stream_gemm_token_threshold=call.engine[THRESHOLD],
                           indexer_short_context=call.short_context)
            signed_gaps = timing.scope_gaps_ms
            composed = compose_attention_phase(
                {name: value + signed_gaps.get(name, 0.0) for name, value in scope_ms.items()},
                counts, dict.fromkeys(required, 0.0), **options,
            )
            errors.append((composed.elapsed_ms - target_ms) / target_ms)
            for name, cost in scope_ms.items():
                scope_gaps[name].append(signed_gaps.get(name, 0.0))
                if cost != 0:
                    scope_ratios[name].append(signed_gaps.get(name, 0.0) / cost)
            if not required - gap_ms.keys():
                diagnostic = compose_attention_phase(scope_ms, counts, gap_ms, **options)
                diagnostic_errors.append((diagnostic.elapsed_ms - target_ms) / target_ms)
        summary, diagnostic = _errors(errors), _errors(diagnostic_errors)
        status = _status(host_ahead_count, summary)
        diagnostic["status"] = (NO_HOST_AHEAD_CALL if host_ahead_count == 0
                                else MISSING_GAP_VALUE if missing
                                else _status(host_ahead_count, diagnostic))
        diagnostic["missing_gap_values"] = sorted(missing)
        groups.append({**asdict(group), "call_count": len(calls), "host_ahead_count": host_ahead_count,
                       _HOST_AHEAD_FRACTION: host_ahead_count / len(calls),
                       "exclusion_counts": dict(sorted(exclusions.items())),
                       _SCOPE_GAPS: {name: {_GAP_SUM_MS: _percentiles(scope_gaps[name]),
                                             _RATIO_TO_SCOPE_COST: _percentiles(scope_ratios[name])}
                                      for name in sorted(scope_gaps)},
                       **summary, "diagnostic": diagnostic, "status": status})
    return {"verdict": PASS if groups and all(group["status"] == GROUP_PASS for group in groups) else FAIL,
            "limits": {"median_error": MEDIAN_ERROR_LIMIT, "p90_error": P90_ERROR_LIMIT},
            "gap_values_ms": gap_ms, "gap_value_reasons": list(reduction.reasons), "groups": groups}


def report_composition(config: BridgeConfig) -> int:
    try:
        records = read_replay_records([config.output_root / run.mode / "raw" / f"dsv4_raw_e{run.engine_index}.jsonl" for run in config.runs])
        report = check_composition(records, planner_roles(config, records))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"verdict": FAIL, "limits": {"median_error": MEDIAN_ERROR_LIMIT, "p90_error": P90_ERROR_LIMIT},
                  "gap_values_ms": {}, "groups": [], "reasons": [str(error)]}
    config.output_root.mkdir(parents=True, exist_ok=True)
    path = config.output_root / REPORT_FILE
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    counts = Counter(group["status"] for group in report["groups"])
    print(f"[dsv4_composition] {report['verdict']}: {len(report['groups'])} groups, {dict(counts)}")
    for reason in report.get("reasons", []):
        print(f"[dsv4_composition] {reason}")
    return int(report["verdict"] != PASS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    return report_composition(load_config(args.config))


__all__ = ["CompositionGroup", "MEDIAN_ERROR_LIMIT", "P90_ERROR_LIMIT", "check_composition",
           "main", "planner_roles", "report_composition"]

if __name__ == "__main__":
    raise SystemExit(main())
