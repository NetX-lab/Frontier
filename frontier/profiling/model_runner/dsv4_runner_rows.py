"""Step ownership census and runner operator rows, using replay records on CPU.

Schema dsv4_model_runner.v1 uses the DSA time_stats.<operator> convention (ms).
Kernel and memcpy samples remain separate; it is a producer artifact for the
forthcoming V4 operator loader, not an execution-time consumer activation.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from frontier.profiling.attention.dsa_rows import _stat_columns
from frontier.profiling.attention.dsv4_step_scopes import (
    INPUT_SCOPE, LOGITS_SCOPE, SAMPLING_SCOPE, WRITE_SCOPE, ZERO_SCOPE, is_step_scope,
)
from frontier.profiling.attention.graph_replay_trace import _Range, _RangeIndex

ROW_SCHEMA = "dsv4_model_runner.v1"
MODEL_FORWARD_PREFIX = "frontier_v4_model_forward|"
ACTIVITY_CLASSES = ("attention_scope", "step_scope", "model_forward", "unowned")
# Replay rows price input preparation and sampling. Direct path-B rows price
# zeroing and writes; replay writes validate them and replay zeroing and the
# LM head (priced by the standalone operators) are diagnostics.
COST, VALIDATION, DIAGNOSTIC = "cost", "validation", "diagnostic"
REPLAY_ROW_ROLES = {ZERO_SCOPE: DIAGNOSTIC, WRITE_SCOPE: VALIDATION, INPUT_SCOPE: COST,
                    SAMPLING_SCOPE: COST, LOGITS_SCOPE: DIAGNOSTIC}


def forward_correlations(events):
    """Match forward membership by CPU launch correlation, never kernel name."""
    ranges = defaultdict(list)
    for event in events:
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith(MODEL_FORWARD_PREFIX):
            start = float(event["ts"])
            ranges[(event.get("pid"), event.get("tid"))].append(_Range(start, start + float(event["dur"]), True))
    if not ranges:
        return None
    indices = {thread: _RangeIndex(rows, False) for thread, rows in ranges.items()}
    return {event.get("args", {}).get("correlation") for event in events
            if event.get("args", {}).get("correlation") is not None
            and event.get("cat") in ("cuda_runtime", "cuda_driver")
            and (index := indices.get((event.get("pid"), event.get("tid")))) is not None
            and index.at(float(event["ts"])) is not None}


def classify_interval(row):
    if is_step_scope(row):
        return "step_scope"
    if row.get("scope") is not None:
        return "attention_scope"
    if row.get("launch_kind") == "graph" or row.get("model_forward", False):
        return "model_forward"
    return "unowned"


def case_kind(step):
    phase = step["phase"]
    if phase in ("dummy", "prefill"):
        return phase
    if phase == "first_tail":
        return "multi_tail" if "multi_tail" in step["case_id"] else "first_tail"
    return step["cudagraph_dispatch"]["cg_mode"]


def ownership_report(steps, intervals):
    """Classify every kernel/copy/memset and summarize unowned names and grids."""
    by_step = {(s["case_id"], s["step_index"]): s for s in steps}
    counts = {key: Counter() for key in by_step}
    grouped, summaries = defaultdict(Counter), defaultdict(Counter)
    for row in intervals:
        if row["activity"] == "hold":
            continue
        key = row["case_id"], row["step_index"]
        classification = classify_interval(row)
        counts[key][classification] += 1
        kind = case_kind(by_step[key])
        summaries[kind][classification] += 1
        if classification == "unowned":
            grouped[kind][(row["activity"], row["name"], tuple(row.get("grid") or ()))] += 1
    return {"steps": [{"case_id": case, "step_index": index,
                       "counts": {name: values[name] for name in ACTIVITY_CLASSES}}
                      for (case, index), values in sorted(counts.items())],
            "case_kinds": {kind: {"counts": {name: values[name] for name in ACTIVITY_CLASSES},
                                 "unowned": [{"activity": activity, "name": name, "grid": list(grid) or None, "count": count}
                                             for (activity, name, grid), count in sorted(grouped[kind].items())]}
                           for kind, values in sorted(summaries.items())}}


def timing_row(operator, role, coordinate, method, layout, kernel_ms, memcpy_ms, event_ms=None):
    row = {"schema_version": ROW_SCHEMA, "operator": operator, "row_role": role, "profile_method": method,
           **coordinate, "layout": json.dumps(layout, sort_keys=True), "repeat_count": len(kernel_ms),
           **_stat_columns(operator, event_ms if event_ms is not None else kernel_ms),
           **_stat_columns(operator, kernel_ms, prefix="kernel_stats"),
           **_stat_columns(operator, memcpy_ms, prefix="memcpy_stats")}
    return row


def replay_operator_rows(engine, steps, calls, intervals):
    by_step, grouped, activity = {}, defaultdict(list), defaultdict(lambda: defaultdict(float))
    for step in steps:
        by_step[(step["case_id"], step["step_index"])] = step
    for row in intervals:
        if is_step_scope(row):
            activity[(row["case_id"], row["step_index"], row["scope"])][row["activity"]] += (row["end_us"] - row["start_us"]) / 1000
    for row in calls:
        if is_step_scope(row):
            grouped[(row["case_id"], row["scope"])].append(row)
    result = []
    for (case, scope), rows in sorted(grouped.items()):
        samples = {row["step_index"]: row for row in rows}
        # A call row is already the sum of all disjoint invocations in a step.
        first = by_step[(case, min(samples))]
        kernel = [samples[i]["time_ms"] for i in sorted(samples)]
        copies = [activity[(case, i, scope)]["memcpy"] for i in sorted(samples)]
        result.append(timing_row(scope, REPLAY_ROW_ROLES[scope],
                                 {"case_id": case, "case_kind": case_kind(first),
                                  "engine_index": engine["engine_index"]},
                                 "record_function", engine["runner_layout"], kernel, copies))
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("Runner profiling produced no rows")
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_replay_report(raw_path):
    engine, steps, calls, intervals = None, [], [], []
    with Path(raw_path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["kind"] == "engine":
                engine = row
            else:
                {"step": steps, "call": calls, "interval": intervals}[row["kind"]].append(row)
    report = ownership_report(steps, intervals)
    Path(raw_path).with_suffix(".ownership.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = replay_operator_rows(engine, steps, calls, intervals)
    write_csv(Path(raw_path).with_suffix(".runner.csv"), rows)
