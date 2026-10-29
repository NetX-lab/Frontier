#!/usr/bin/env python3
"""A11 diagnostic comparison of an A10 observer record with native counters/R1.

This never gates a replay or changes a cost. Counter totals are lane observations,
weighted by successful_forwards, not globally unique DP forwards. Native dummy
rows whose phase is null cannot be assigned to formal/warmup serving. Native
multi-request first-tail counters without context/generation counts cannot be
split exactly. Both are reported explicitly, without inferring missing inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import v4_runtime_census as native


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def counter_record(role, row):
    evidence = row["dispatch_evidence"]
    dispatch = evidence["dispatch"]
    classifier = evidence.get("classifier") or {}
    inputs = classifier.get("inputs") or {}
    dummy = row["kind"] == "dummy"
    requests = row.get("scheduled_num_requests")
    if requests is None:
        requests = inputs.get("num_reqs")
    if dummy:
        kind = "dummy"
    elif row.get("step") is not None:
        kind = native.step_class(role, {"real": True, "step": row["step"]})
    elif role == "prefill":
        kind = "prefill"
    elif inputs.get("has_prefill") is False:
        kind = "decode"
    elif inputs.get("has_prefill") is True:
        kind = "first_tail_alone" if requests == 1 else "first_tail_unresolved"
    else:
        kind = "real_without_classifier"
    dp = dispatch["inputs"]["dp_rank"]
    widths = dispatch["num_tokens_across_dp"]
    mode = dispatch["descriptor"]["cg_mode"]
    # Synchronization enforces a shared final mode. Desired peer modes remain
    # separately visible; do not label a desired mode as a peer's final mode.
    desired = evidence["dp_post_all_reduce"][1]
    return {"role": role, "dp": dp, "real": not dummy, "step_class": kind,
            "decode_batch_size": requests if kind == "decode" else None,
            "mode": mode, "phase": row.get("phase"),
            "count": row.get("successful_forwards") or 0,
            "synchronized_width": widths[dp], "num_tokens_across_dp": widths,
            "peer_modes": [mode for index in range(len(widths)) if index != dp],
            "desired_peer_modes": [native_mode(value) for index, value in enumerate(desired) if index != dp],
            "classifier": {**inputs, "uniform_token_count": classifier.get("uniform_token_count")},
            "local_tokens": evidence["dp_post_all_reduce"][0][dp],
            "local_requests": requests, "descriptor": dispatch["descriptor"]}


def native_mode(value):
    return ("NONE", "PIECEWISE", "FULL")[value]


def read_counters(artifacts, serving_phase="formal"):
    selected, excluded, unresolved, sources = [], [], [], []
    phase_totals = Counter()
    for path in sorted(Path(artifacts).glob("nodes/*/serving/rank0.json")):
        data = json.loads(path.read_text())
        role = data["role"].lower()
        sources.append({"path": str(path), "sha256": digest(path), "role": role})
        for row in data["counters"]["rows"]:
            count = row.get("successful_forwards") or 0
            if type(count) is not int or count < 0:
                raise ValueError(f"{path}: invalid successful_forwards {count!r}")
            phase_totals[(role, row.get("phase"), row["kind"])] += count
            if row["kind"] not in ("serving", "dummy"):
                excluded.append({"role": role, "kind": row["kind"], "phase": row.get("phase"),
                                 "count": count, "source": str(path)})
                continue
            record = {**counter_record(role, row), "source": str(path)}
            if row.get("phase") == serving_phase:
                selected.append(record)
            elif row.get("phase") is None:
                unresolved.append(record)
            else:
                excluded.append(record)
    if not sources:
        raise ValueError(f"no native nodes/*/serving/rank0.json under {artifacts}")
    return {"records": selected, "unassigned_phase_records": unresolved, "excluded_records": excluded,
            "sources": sources, "phase_counts": [
                {"role": role, "phase": phase, "kind": kind, "count": count}
                for (role, phase, kind), count in sorted(phase_totals.items(), key=lambda item: str(item[0]))],
            "serving_phase": serving_phase}


def class_key(record):
    return (record["step_class"], record.get("decode_batch_size") if record["step_class"] == "decode" else None)


def census_counts(records, dimension):
    result = Counter()
    for row in records:
        base = (row["role"], row["dp"], *class_key(row))
        if dimension == "forwards":
            key = (*base, "real" if row["real"] else "dummy", row["mode"])
        elif dimension == "widths":
            key = (*base, row["synchronized_width"])
        elif dimension == "dummy_peers":
            if row["real"]:
                continue
            key = (row["role"], row["dp"], tuple(row.get("peer_modes", ())))
        else:
            raise ValueError(dimension)
        result[key] += row.get("count", 1)
    return result


def count_comparison(frontier, reference):
    return [{"key": list(key), "frontier": frontier[key], "native": reference[key],
             "difference": frontier[key] - reference[key]}
            for key in sorted(frontier.keys() | reference.keys(), key=str)]


def distribution(values):
    values = [value for value in values if value is not None]
    return {"n": len(values), "median": native.median(values), "p90": native.percentile(values, .9)}


def error(frontier, reference):
    if frontier is None or reference is None:
        return {"signed_error": None, "absolute_error": None, "relative_error": None}
    return {"signed_error": frontier - reference, "absolute_error": abs(frontier - reference),
            "relative_error": None if reference == 0 else abs(frontier - reference) / abs(reference)}


def time_stat(frontier, reference):
    f, n = distribution(frontier), distribution(reference)
    return {"frontier": f, "native": n,
            "errors": {quantile: error(f[quantile], n[quantile]) for quantile in ("median", "p90")}}


def runtime_records(artifacts):
    rows, sources = [], []
    for role, dp, path in native.lane_traces(Path(artifacts) / "torch_traces"):
        result = native.forward_census(role, path)
        sources.append({"path": str(path), "sha256": digest(path), "role": role, "dp": dp,
                        "forwards": len(result["forwards"])})
        for row in result["forwards"]:
            step = row.get("step") or {}
            # Device union is R1's segment metric; whole forward uses span.
            rows.append({"role": role, "dp": dp, "phase": "profiler_window", "real": row["real"],
                         "step_class": row["class"], "mode": row["mode"],
                         "decode_batch_size": step.get("generation_requests") if row["class"] == "decode" else None,
                         "composed_ms": row.get("device", {}).get("span_us", None),
                         "segment_ms": {name: row["segments"][name]["device_union_us"] / 1000
                                        for name in native.SEGMENTS}})
            if rows[-1]["composed_ms"] is not None:
                rows[-1]["composed_ms"] /= 1000
    return rows, sources


def compare_times(frontier, reference, per_phase=False):
    groups = [defaultdict(list), defaultdict(list)]
    for records, group in zip((frontier, reference), groups):
        for row in records:
            # Phases are not known in R1. Keep profiler_window explicit, with a
            # corresponding whole serving replay distribution beside it.
            key = (row["role"], "all_classes") if per_phase else (row["role"], *class_key(row))
            group[key].append(row)
    result = []
    for key in sorted(groups[0].keys() | groups[1].keys(), key=str):
        f, n = groups[0][key], groups[1][key]
        fields = {"composed_ms": time_stat([r.get("composed_ms") for r in f], [r.get("composed_ms") for r in n])}
        for field in sorted(set(native.SEGMENTS) | {name for r in f for name in r.get("segment_ms", {})}):
            fields[field] = time_stat([r.get("segment_ms", {}).get(field) for r in f],
                                      [r.get("segment_ms", {}).get(field) for r in n])
        result.append({"key": list(key), "frontier_phase": "serving", "native_phase": "profiler_window",
                       "frontier_forwards": len(f), "native_window_forwards": len(n), "time_ms": fields})
    return result


def collective_shares(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["role"], tuple(row["curve_key"]))].append(row)
    result = []
    for (role, key), parts in sorted(groups.items()):
        calls = sum(r["call_count"] for r in parts)
        duration = sum(r["predicted_total_ms"] for r in parts)
        shares = {}
        for label, exact in (("exact", True), ("interpolated", False)):
            selected = [r for r in parts if r["exact"] == exact]
            count = sum(r["call_count"] for r in selected)
            time = sum(r["predicted_total_ms"] for r in selected)
            shares[label] = {"calls": count, "call_share": count / calls if calls else None,
                             "predicted_ms": time, "time_share": time / duration if duration else None}
        result.append({"role": role, "curve_key": list(key), "calls": calls,
                       "predicted_total_ms": duration, **shares})
    return result


NOTES = [
    "Diagnostic only: differences never gate a run or feed a cost.",
    "Native counts use rank0.json from every node of both roles, weighted by successful_forwards (not successful_replays).",
    "Real serving phase is formal; warmup is excluded. Capture/profile kinds are excluded regardless of phase.",
    "Null-phase serving/dummy rows are unassigned, separately counted, and excluded from formal comparison. Native dummy formal counts are unavailable when their phase is null.",
    "Multi-request first-tail counters lack context/generation counts and use first_tail_unresolved; R1 classes are exact inside the profiler window.",
    "R1 has no phase field: time compares the whole Frontier serving replay with native profiler_window, not a phase-matched sample. Every native statistic carries its actual window count.",
    "Times are milliseconds: whole native forward is R1 device span; native segment is R1 device union. Segment unions need not sum to span.",
    "Frontier composed time is the predicted stage duration including DP/EP coordination; scope attribution uses TP rank zero and composed attention. Unmatched coordination/overlap is a signed own column.",
    "Pre-DP scopes are own columns: native R1 assigns a following forward's pre-DP work to the previous post segment. No cross-forward reassignment is guessed.",
    "Native zero denominators have undefined relative error (JSON null). Native host durations are profiler-inflated upper bounds; host time is not a residual cost.",
]


def compare_run(run_root, artifacts, serving_phase="formal"):
    run_root, artifacts = Path(run_root), Path(artifacts)
    receipt = json.loads((run_root / "receipt.json").read_text())
    if receipt["status"] != "FINISHED":
        raise ValueError(f"A11 needs a finished replay: {run_root} status={receipt['status']}")
    forwards = run_root / "forwards.jsonl.gz"
    collectives = run_root / "collectives.json"
    for name, path in (("forwards", forwards), ("collectives", collectives)):
        if digest(path) != receipt["a11"][name]["sha256"]:
            raise ValueError(f"{path}: observer receipt SHA-256 differs")
    with gzip.open(forwards, "rt") as stream:
        frontier = [json.loads(line) for line in stream]
    if len(frontier) != receipt["a11"]["forwards"]["rows"]:
        raise ValueError("observer row count differs from receipt")
    counters = read_counters(artifacts, serving_phase)
    runtime, sources = runtime_records(artifacts)
    census = {dimension: count_comparison(census_counts(frontier, dimension),
                                        census_counts(counters["records"], dimension))
              for dimension in ("forwards", "widths", "dummy_peers")}
    unassigned = {dimension: [dict(row, frontier=None, difference=None) for row in
                  count_comparison(Counter(), census_counts(counters["unassigned_phase_records"], dimension))]
                  for dimension in census}
    # A null-phase native dummy is missing formal evidence, not zero formal
    # forwards. Preserve that distinction even on Frontier-only dummy keys.
    for dimension, rows in census.items():
        unknown_roles = {r["role"] for r in counters["unassigned_phase_records"] if not r["real"]}
        for row in rows:
            dummy = dimension == "dummy_peers" or row["key"][2] == "dummy"
            if dummy and row["key"][0] in unknown_roles and row["native"] == 0:
                row.update(native=None, difference=None)
    return {"schema": "a11-replay-census-v1", "case": receipt["case"], "run_root": str(run_root),
            "native_artifacts": str(artifacts), "notes": NOTES,
            "scope_mapping": receipt["a11"]["scope_mapping"],
            "observer_forwards": len(frontier), "native_counters": counters,
            "native_runtime_sources": sources, "census": census, "unassigned_phase_census": unassigned,
            "time_by_step_class": compare_times(frontier, runtime), "time_by_phase": compare_times(frontier, runtime, True),
            "collectives": collective_shares(json.loads(collectives.read_text())["rows"])}


def summary(result):
    lines = [f"A11 {result['case']}: Frontier forwards={result['observer_forwards']}", *result["notes"],
             "Scope mapping: " + json.dumps(result["scope_mapping"], sort_keys=True)]
    for label, census in (("formal census", result["census"]), ("unassigned phase census (not compared)", result["unassigned_phase_census"])):
        lines.append(label)
        for dimension, rows in census.items():
            lines.append(f"  {dimension}: key | Frontier native difference")
            lines.extend(f"    {row['key']} | {row['frontier']} {row['native']} {row['difference']}" for row in rows)
    lines.append("Native weighted counts by phase and kind:")
    lines.extend("  " + json.dumps(row) for row in result["native_counters"]["phase_counts"])
    for table in ("time_by_step_class", "time_by_phase"):
        lines.append(table + ": milliseconds median/p90, n beside each native statistic")
        for row in result[table]:
            for field, stat in row["time_ms"].items():
                lines.append(f"  {row['key']} {field}: Frontier n={stat['frontier']['n']} "
                             f"{stat['frontier']['median']}/{stat['frontier']['p90']}; native window n={stat['native']['n']} "
                             f"{stat['native']['median']}/{stat['native']['p90']}; errors={json.dumps(stat['errors'])}")
    lines.append("Collective exact/interpolated call and predicted-time shares:")
    lines.extend("  " + json.dumps(row) for row in result["collectives"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--native-artifacts", required=True, type=Path)
    parser.add_argument("--serving-phase", default="formal")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_run(args.run_root, args.native_artifacts, args.serving_phase)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    args.summary.write_text(summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
