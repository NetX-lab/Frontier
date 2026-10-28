#!/usr/bin/env python3
"""Native DeepSeek-V4 collective kernels and G5h benchmark comparison (CPU only).

Native durations include peer waits: their minima best approximate the kernel's
own time. Kernel equality is independent of the reported duration verdicts.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from frontier.profiling.collectives.vllm_dsv4_forward_collectives import (
    CALL_NAMES, DeepseekV4Shape, dsv4_plan,
)
from frontier.profiling.collectives.vllm_forward_collectives import ServingParallelism, group_members
from v4_runtime_census import DEVICE_CATEGORIES, forward_kernel_events, lane_traces, percentile

# Only prefixes observed in the native P2 traces belong in this table. Review
# unclassified names before expanding it; the search expression never classifies.
COLLECTIVE_PREFIXES = {
    "ncclDevKernel_": "nccl",
    "void vllm::cross_device_reduce_": "custom_all_reduce",
    "void vllm::mnnvl_lamport_": "mnnvl",
    "void (anonymous namespace)::multimem_": "symm_mem",
}
REVIEW_PATTERN = re.compile(
    r"nccl|lamport|cross_device|multimem|symm|all_?reduce|all_?gather|reduce_?scatter|one_shot|two_shot",
    re.IGNORECASE,
)
ROW_COUNT_KERNEL = "_fused_q_kv_rmsnorm_kernel"
GRAPH_MODES = ("FULL", "PIECEWISE")
DIMENSIONS = 3
KernelKey = tuple[str, tuple[int, ...], tuple[int, ...]]
SegmentKey = tuple[str, str, tuple[int, ...], tuple[int, ...]]


def collective_family(name: str) -> str | None:
    return next((family for prefix, family in COLLECTIVE_PREFIXES.items() if name.startswith(prefix)), None)


def review_names(events) -> Counter:
    return Counter(event["name"] for event in events
                   if collective_family(event["name"]) is None and REVIEW_PATTERN.search(event["name"]))


def kernel_key(event: dict, source: str) -> KernelKey:
    dimensions = []
    for field in ("grid", "block"):
        value = event.get("args", {}).get(field)
        if (not isinstance(value, (list, tuple)) or len(value) != DIMENSIONS
                or any(type(item) is not int for item in value)):
            raise ValueError(f"{source}: kernel {event['name']!r} requires {field} with three ints")
        dimensions.append(tuple(value))
    return event["name"], *dimensions


def duration_stats(values: list[float], native: bool = False) -> dict:
    result = {"min": min(values), "median": statistics.median(values), "max": max(values)}
    if native:
        result.update(p10=percentile(values, 0.1), p90=percentile(values, 0.9))
    return result


def key_row(key: KernelKey | SegmentKey, **fields) -> dict:
    segment, identity = (key[0], key[1:]) if len(key) == 4 else (None, key)
    name, grid, block = identity
    row = {"name": name, "family": collective_family(name), "grid": list(grid), "block": list(block), **fields}
    if segment is not None:
        row["segment"] = segment
    return row


def row_key(row: dict, segment: bool = False):
    identity = row["name"], tuple(row["grid"]), tuple(row["block"])
    return (row["segment"], *identity) if segment else identity


def multiset_rows(counts: Counter) -> list[dict]:
    return [key_row(key, count=count) for key, count in sorted(counts.items())]


def rows_multiset(rows: list[dict], segment: bool = False) -> Counter:
    counts = Counter()
    for row in rows:
        counts[row_key(row, segment)] += row["count"]
    return counts


def forward_reference(forward: dict) -> dict:
    return {key: forward[key] for key in ("lane", "index", "trace_path")}


def _read_lane(role: str, lane: int, path: Path, review: Counter) -> dict:
    records = []
    for forward in forward_kernel_events(role, path):
        record = forward.record
        source = f"{path}: forward {record['index']}"
        rows = next((kernel_key(event, source)[1][0]
                     for event in forward.kernels_by_segment["dispatch"] if event["name"] == ROW_COUNT_KERNEL), None)
        if rows is None:
            raise ValueError(f"{source}: missing row-count kernel {ROW_COUNT_KERNEL}")
        if rows <= 0:
            raise ValueError(f"{source}: nonpositive own rows {rows}")
        durations = defaultdict(list)
        for segment, events in forward.kernels_by_segment.items():
            review.update(review_names(events))
            for event in events:
                if collective_family(event["name"]) is None:
                    continue
                if segment == "prepare":
                    raise ValueError(f"{source}: collective in prepare: {event['name']}; requires lead review")
                durations[(segment, *kernel_key(event, source))].append(event["dur"])
        step = record.get("step")
        requests = None if step is None else step["context_requests"] + step["generation_requests"]
        scheduled = None if step is None else step["context_tokens"] + step["generation_tokens"]
        if record["real"] and step is None:
            raise ValueError(f"{source}: real forward has no step range")
        if record["real"] and (scheduled > rows or (record["mode"] == "NONE" and scheduled != rows)):
            raise ValueError(f"{source}: own rows {rows} disagree with scheduled tokens {scheduled}, mode {record['mode']}")
        records.append({
            "lane": lane, "trace_path": str(path), "index": record["index"], "mode": record["mode"],
            "step_class": record["class"], "real": record["real"], "step": step,
            "requests": requests, "scheduled_tokens": scheduled, "own_rows": rows,
            "segment_bounds_us": {name: [start, None if stop == float("inf") else stop]
                                  for name, (start, stop) in forward.segment_bounds.items()},
            "dispatch_start_us": forward.segment_bounds["dispatch"][0],
            "kernels": [key_row(key, count=len(values), durations_us=values)
                        for key, values in sorted(durations.items())],
        })
    print("LANE", role, f"dp{lane}", path.name, "forwards", len(records), flush=True)
    return {"role": role, "lane": lane, "trace_path": str(path), "forwards": records}


def pair_lanes(lanes: list[dict]) -> dict:
    """Pair all lanes to lane 0 by nearest same-mode host dispatch start.

    The acceptance radius is half the smaller adjacent-forward gap on the
    anchor lane (all modes). Ties or reusing a candidate contradict uniqueness
    and require review. Incomplete tuples are excluded, including unused peers.
    """
    lanes = sorted(lanes, key=lambda lane: lane["lane"])
    if [lane["lane"] for lane in lanes] != list(range(len(lanes))):
        raise ValueError("DP lane ids must be contiguous from zero")
    anchor = lanes[0]["forwards"]
    times = [forward["dispatch_start_us"] for forward in anchor]
    by_mode = []
    for lane in lanes:
        modes = defaultdict(list)
        for forward in lane["forwards"]:
            modes[forward["mode"]].append(forward)
        by_mode.append({mode: (entries, [entry["dispatch_start_us"] for entry in entries])
                        for mode, entries in modes.items()})
    used = [set() for _ in lanes]
    accepted, differences, rejected = [], [], []
    for index, forward in enumerate(anchor):
        gaps = [abs(times[neighbor] - times[index]) for neighbor in (index - 1, index + 1)
                if 0 <= neighbor < len(times)]
        # An isolated forward supplies no cadence: only exact simultaneity can
        # be established without inventing a time threshold.
        radius = min(gaps) / 2 if gaps else 0.0
        peers, deltas, failure = [forward], [], None
        for lane_index in range(1, len(lanes)):
            entries, starts = by_mode[lane_index].get(forward["mode"], ([], []))
            position = bisect.bisect_left(starts, times[index])
            candidates = [entries[n] for n in (position - 1, position) if 0 <= n < len(entries)]
            if not candidates:
                failure = {"reason": "no_same_mode", "missing_lane": lane_index}
                break
            distances = [abs(peer["dispatch_start_us"] - times[index]) for peer in candidates]
            if len(distances) == 2 and distances[0] == distances[1] and distances[0] < radius:
                raise ValueError(f"{forward_reference(forward)}: ambiguous nearest pairing in lane {lane_index}")
            peer = candidates[distances.index(min(distances))]
            delta = min(distances)
            if delta >= radius and not (delta == radius == 0):
                failure = {"reason": "beyond_half_gap", "missing_lane": lane_index,
                           "candidate": forward_reference(peer), "difference_us": delta, "radius_us": radius}
                break
            if peer["index"] in used[lane_index]:
                raise ValueError(f"{forward_reference(forward)}: pairing reuses {forward_reference(peer)}")
            peers.append(peer)
            deltas.append(delta)
        if failure is not None:
            rejected.append({"anchor": forward_reference(forward), **failure})
            continue
        vector = [peer["own_rows"] for peer in peers]
        for lane_index, peer in enumerate(peers):
            used[lane_index].add(peer["index"])
            peer["lane_vector"] = vector
            peer["paired_forwards"] = [forward_reference(p) for p in peers]
        accepted.append([forward_reference(peer) for peer in peers])
        differences.extend(deltas)
    for lane, indices in zip(lanes, used):
        for forward in lane["forwards"]:
            forward.setdefault("lane_vector", None)
    return {
        "role": lanes[0]["role"], "anchor_lane": lanes[0]["lane"], "accepted_tuples": len(accepted),
        "accepted_pairs": len(differences), "start_difference_us": {
            "median": statistics.median(differences) if differences else None,
            "max": max(differences) if differences else None},
        "pairs": accepted, "rejected_anchors": rejected,
        "left_out_per_lane": {str(lane["lane"]): len(lane["forwards"]) - len(indices)
                              for lane, indices in zip(lanes, used)},
    }


def _validate_graph_rows(lanes: list[dict]) -> None:
    buckets = sorted({f["own_rows"] for lane in lanes for f in lane["forwards"] if f["mode"] in GRAPH_MODES})
    lookup = {(f["lane"], f["index"]): f for lane in lanes for f in lane["forwards"]}
    for forward in lookup.values():
        if not forward["real"]:
            continue
        scheduled = forward["scheduled_tokens"]
        if forward["mode"] in GRAPH_MODES:
            peers = [lookup[(ref["lane"], ref["index"])] for ref in forward.get("paired_forwards", [])]
            if peers:
                if len(set(forward["lane_vector"])) != 1:
                    raise ValueError(f"{forward_reference(forward)}: nonuniform graph lane rows; requires lead review")
                scheduled = max(peer["scheduled_tokens"] or 0 for peer in peers)
            expected = next(bucket for bucket in buckets if bucket >= scheduled)
        else:
            expected = scheduled
        if expected != forward["own_rows"]:
            raise ValueError(f"{forward_reference(forward)}: expected {expected} rows after observed graph padding, "
                             f"got {forward['own_rows']}; observed buckets {buckets}")
        forward["row_check"] = {"expected_rows": expected, "padding_rows": expected - forward["scheduled_tokens"],
                                "basis": "observed_graph_buckets" if forward["mode"] in GRAPH_MODES else "scheduled_tokens"}


def _native_groups(lanes: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for lane in lanes:
        for forward in lane["forwards"]:
            if forward["lane_vector"] is not None:
                key = (lane["role"], forward["mode"], forward["real"], forward["own_rows"],
                       tuple(forward["lane_vector"]), forward["requests"] if forward["real"] else None)
                groups[key].append(forward)
    result = []
    for identity, forwards in sorted(groups.items()):
        variants = defaultdict(list)
        durations = defaultdict(list)
        per_forward = []
        for forward in forwards:
            counts = rows_multiset(forward["kernels"], segment=True)
            per_forward.append(counts)
            variants[tuple(sorted(counts.items()))].append(forward_reference(forward))
            for row in forward["kernels"]:
                durations[row_key(row, segment=True)].extend(row["durations_us"])
        role, mode, real, rows, vector, requests = identity
        result.append({
            "id": len(result), "key": {"role": role, "mode": mode, "real": real, "own_rows": rows,
                                       "lane_vector": list(vector), "requests": requests},
            "forward_count": len(forwards), "step_classes": dict(Counter(f["step_class"] for f in forwards)),
            "distinct_multisets": len(variants),
            "multisets": [{"forward_count": len(refs), "kernels": multiset_rows(Counter(dict(signature))),
                           "forwards": refs} for signature, refs in sorted(variants.items())],
            "kernel_stats": [key_row(key, counts_per_forward=[{"count": count, "forwards": n}
                                                              for count, n in sorted(Counter(c[key] for c in per_forward).items())],
                                     duration_us=duration_stats(values, native=True))
                             for key, values in sorted(durations.items())],
        })
    return result


def native_table(trace_dir: Path) -> dict:
    lanes, review = [], Counter()
    tp_ranks = set()
    for role, lane, path in lane_traces(trace_dir):
        lanes.append(_read_lane(role, lane, path, review))
        tp_ranks.update(int(match[1]) for file in path.parent.glob("*.pt.trace.json.gz")
                        if (match := re.search(r"_pp0_tp(\d+)_", file.name)))
    if tp_ranks != set(range(max(tp_ranks) + 1)):
        raise ValueError(f"{trace_dir}: noncontiguous TP trace ranks {sorted(tp_ranks)}")
    roles = defaultdict(list)
    for lane in lanes:
        roles[lane["role"]].append(lane)
    pairing = []
    for role_lanes in roles.values():
        pairing.append(pair_lanes(role_lanes))
        _validate_graph_rows(role_lanes)
    return {"schema_version": 1, "trace_dir": str(trace_dir), "tensor_parallel_size": len(tp_ranks),
            "classification": COLLECTIVE_PREFIXES, "duration_unit": "us",
            "duration_note": "Native durations include peer waits; minima are the closest estimate of kernel time.",
            "pairing_note": "accepted_pairs are anchor-to-peer edges in complete lane tuples; differences are absolute host times.",
            "review_list": [{"name": name, "count": count} for name, count in sorted(review.items())],
            "pairing": pairing, "lanes": lanes, "groups": _native_groups(lanes)}


def benchmark_case(path: Path, calls_per_repeat: int) -> dict:
    """Normalize a single untimed run to kernels per call; retain auxiliary work."""
    events = json.loads(path.read_text())["traceEvents"]
    kernels, auxiliary, durations = Counter(), Counter(), defaultdict(list)
    auxiliary_durations = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in DEVICE_CATEGORIES:
            continue
        if event["cat"] == "kernel" and collective_family(event["name"]) is not None:
            key = kernel_key(event, str(path))
            kernels[key] += 1
            durations[key].append(event["dur"])
        else:
            # Auxiliary events need no launch shape (memcpy/memset have none).
            key = event["cat"], event["name"]
            auxiliary[key] += 1
            auxiliary_durations[key].append(event["dur"])
    errors = [f"{path}: {count} occurrences of {key} not divisible by calls_per_repeat={calls_per_repeat}"
              for key, count in sorted(kernels.items()) if count % calls_per_repeat]
    return {
        "trace_path": str(path), "errors": errors,
        "kernels": [key_row(key, count=count // calls_per_repeat, total_count=count,
                            duration_us=duration_stats(durations[key]))
                    for key, count in sorted(kernels.items()) if count % calls_per_repeat == 0],
        "auxiliary": [{"category": key[0], "name": key[1], "total_count": count,
                       "count_per_call": count / calls_per_repeat,
                       "duration_us": duration_stats(auxiliary_durations[key])}
                      for key, count in sorted(auxiliary.items())],
        "review_list": [{"name": name, "count": count} for name, count in sorted(review_names(
            event for event in events if event.get("ph") == "X" and event.get("cat") == "kernel").items())],
        "durations": dict(durations),
    }


@dataclass
class Benchmark:
    directory: Path
    stage: str
    parallel: ServingParallelism
    model: DeepseekV4Shape
    calls_per_repeat: int
    planned: dict
    records: dict
    lane_ranks: list[int]
    cache: dict

    @classmethod
    def read(cls, directory: Path) -> "Benchmark":
        document = json.loads((directory / "plan.json").read_text())
        config = document["config"]
        parallel = ServingParallelism.from_engine(config["engine"])
        model_path = Path(config["model_config"])
        if not model_path.is_absolute():
            model_path = directory / model_path
        model = DeepseekV4Shape.from_hf_config(json.loads(model_path.read_text()))
        planned = {(scenario.name, call.name): call for scenario, call in dsv4_plan(model, parallel, config["grid"])}
        recorded_groups = document["runtime"]["group_ranks"]
        for group in ("tp", "ep", "dp"):
            expected = group_members(parallel, group, 0)
            if recorded_groups[group] != expected:
                raise ValueError(f"{directory}: recorded {group} group {recorded_groups[group]} differs from runner layout {expected}")
        records = {}
        for line in (directory / "kernels.jsonl").read_text().splitlines():
            record = json.loads(line)
            key = record["scenario"], record["call"]
            if key in records:
                raise ValueError(f"{directory}: duplicate kernel evidence for {key}")
            records[key] = {rank["rank"]: rank for rank in record["per_rank"] if rank is not None}
        per_repeat = int(config.get("settings", config)["calls_per_repeat"])
        if per_repeat <= 0:
            raise ValueError(f"{directory}: calls_per_repeat must be positive")
        return cls(directory, config["stage"], parallel, model, per_repeat, planned, records,
                   recorded_groups["dp"], {})

    def case(self, scenario: str, call: str, lane: int) -> tuple[int, dict]:
        # TP groups are interchangeable for TP reuse across DP sizes. The
        # modulo selects an existing measured lane, never invents a rank layout.
        rank = self.lane_ranks[lane % len(self.lane_ranks)]
        return rank, self.rank_case(scenario, call, rank)

    def rank_case(self, scenario: str, call: str, rank: int) -> dict:
        key = scenario, call, rank
        if key not in self.cache:
            evidence = self.records.get((scenario, call), {}).get(rank)
            path = self.directory / "traces" / scenario / call / f"rank{rank}.json"
            try:
                if evidence is None or evidence.get("trace") is None:
                    raise ValueError(f"{self.directory}: missing kernel evidence for {key}")
                self.cache[key] = benchmark_case(path, self.calls_per_repeat)
            except (OSError, ValueError, KeyError) as error:
                self.cache[key] = {"errors": [str(error)], "kernels": [], "auxiliary": [],
                                   "durations": {}, "review_list": [], "trace_path": str(path)}
        return self.cache[key]


def _scenario(key: dict, call: str, dp: int) -> str:
    if call == "logits_all_gather":
        return f"logits_b{key['requests']}"
    if key["mode"] in GRAPH_MODES:
        return f"graph_t{key['own_rows']}"
    vector = key["lane_vector"] if call.startswith("moe_") else [key["own_rows"]] * dp
    return "eager_t" + "_".join(map(str, vector))


def _expected(group: dict, lane: int, benchmarks: list[Benchmark], native_tp: int) -> dict:
    key = group["key"]
    counts, durations = Counter(), defaultdict(list)
    selected, missing, errors = [], [], []
    calls = CALL_NAMES if key["real"] else CALL_NAMES[:-1]
    for call_name in calls:
        options = []
        for benchmark in benchmarks:
            if benchmark.parallel.tensor_parallel_size != native_tp:
                continue
            if call_name.startswith("moe_") and benchmark.parallel.data_parallel_size != len(key["lane_vector"]):
                continue
            scenario = _scenario(key, call_name, benchmark.parallel.data_parallel_size)
            options.append({"scenario": scenario, "call": call_name, "benchmark": str(benchmark.directory)})
            call = benchmark.planned.get((scenario, call_name))
            if call is None:
                continue
            rank, case = benchmark.case(scenario, call_name, lane)
            selected.append({"scenario": scenario, "call": call_name, "rank": rank,
                             "calls_per_forward": call.calls_per_forward, "benchmark": str(benchmark.directory),
                             "segment": "post" if call_name == "logits_all_gather" else "dispatch"})
            errors.extend(case["errors"])
            for identity, count in rows_multiset(case["kernels"]).items():
                counts[identity] += call.calls_per_forward * count
            for identity, values in case["durations"].items():
                durations[identity].extend(values * call.calls_per_forward)
            break
        else:
            missing.append({"scenario": _scenario(key, call_name, len(key["lane_vector"])),
                            "call": call_name, "searched": options})
    return {"lane": lane, "kernels": multiset_rows(counts), "selected_calls": selected,
            "missing": missing, "errors": errors,
            "durations": {identity: duration_stats(values) for identity, values in durations.items()}}


def _differences(native: Counter, expected: Counter) -> list[dict]:
    return [key_row(key, native_count=native[key], expected_count=expected[key],
                    native_minus_expected=native[key] - expected[key])
            for key in sorted(native.keys() | expected.keys()) if native[key] != expected[key]]


def _compare_group(group: dict, benchmarks: list[Benchmark], native_tp: int) -> dict:
    lanes = sorted({ref["lane"] for variant in group["multisets"] for ref in variant["forwards"]})
    expected = {lane: _expected(group, lane, benchmarks, native_tp) for lane in lanes}
    missing = [dict(lane=lane, **item) for lane, item in expected.items() for item in item["missing"]]
    errors = sorted({error for item in expected.values() for error in item["errors"]})
    result = {"id": group["id"], "key": group["key"], "forward_count": group["forward_count"],
              "native_multisets": group["multisets"],
              "expected_by_lane": [{k: value for k, value in item.items() if k != "durations"}
                                   for item in expected.values()],
              "missing": missing, "errors": errors, "differences": [], "durations": []}
    if missing or errors:
        result.update(status="error" if errors else "unmatched", kernels=None)
        return result
    for ordinal, variant in enumerate(group["multisets"]):
        native = rows_multiset(variant["kernels"])
        for lane in sorted({ref["lane"] for ref in variant["forwards"]}):
            differences = _differences(native, rows_multiset(expected[lane]["kernels"]))
            if differences:
                result["differences"].append({"native_multiset": ordinal, "lane": lane, "kernels": differences})
    # Segment is ignored only for equality and pooled duration bounds, retained
    # in native multisets and in each selected benchmark call's provenance.
    native_bounds = {}
    for row in group["kernel_stats"]:
        identity, stats = row_key(row), row["duration_us"]
        if identity not in native_bounds:
            native_bounds[identity] = dict(stats)
        else:
            # Exact pooling is needed if an identity occurs in dispatch and post.
            native_bounds[identity] = None
    for identity, stats in list(native_bounds.items()):
        if stats is None:
            values = [duration for lane in group["duration_forwards"]
                      for row in lane["kernels"] if row_key(row) == identity for duration in row["durations_us"]]
            native_bounds[identity] = duration_stats(values, native=True)
    for lane, item in expected.items():
        for identity in sorted(native_bounds.keys() & item["durations"].keys()):
            native, measured = native_bounds[identity], item["durations"][identity]
            verdict = ("below" if measured["median"] < native["min"] else
                       "above" if measured["median"] > native["p90"] else "within")
            result["durations"].append(key_row(identity, lane=lane, native_us=native,
                                              benchmark_us=measured, verdict=verdict))
    result.update(status="matched", kernels="different" if result["differences"] else "equal")
    return result


def compare_tables(native: dict, directories: list[Path]) -> dict:
    benchmarks = [Benchmark.read(directory) for directory in directories]
    if len({benchmark.model for benchmark in benchmarks}) != 1:
        raise ValueError("Benchmark directories have different V4 model shapes")
    forwards = {(lane["role"], f["lane"], f["index"]): f for lane in native["lanes"] for f in lane["forwards"]}
    groups = []
    for group in native["groups"]:
        duration_forwards = [forwards[(group["key"]["role"], ref["lane"], ref["index"])]
                             for variant in group["multisets"] for ref in variant["forwards"]]
        groups.append(_compare_group({**group, "duration_forwards": duration_forwards}, benchmarks,
                                     native["tensor_parallel_size"]))
    cases = []
    review = Counter()
    # Record every planned case/rank's auxiliary work, including cases not
    # selected for these native groups. Read and discard one trace at a time.
    for benchmark in benchmarks:
        for scenario, call in benchmark.planned:
            for rank in group_members(benchmark.parallel, "ep", 0):
                case = benchmark.rank_case(scenario, call, rank)
                review.update({row["name"]: row["count"] for row in case["review_list"]})
                cases.append({"benchmark": str(benchmark.directory), "scenario": scenario, "call": call, "rank": rank,
                              **{field: case[field] for field in ("trace_path", "kernels", "auxiliary", "errors")}})
    return {"schema_version": 1, "native_trace_dir": native["trace_dir"], "duration_unit": "us",
            "benchmarks": [{"directory": str(b.directory), "stage": b.stage,
                            "tensor_parallel_size": b.parallel.tensor_parallel_size,
                            "data_parallel_size": b.parallel.data_parallel_size,
                            "calls_per_repeat": b.calls_per_repeat} for b in benchmarks],
            "selection_note": "First directory planning the call wins; EP requires matching DP; TP lanes reuse measured lanes modulo source DP using recorded DP-group ranks.",
            "duration_note": native["duration_note"], "groups": groups,
            "verdict_counts": dict(Counter(g["kernels"] if g["status"] == "matched" else g["status"] for g in groups)),
            "cases": cases, "native_review_list": native["review_list"],
            "benchmark_review_list": [{"name": name, "count": count} for name, count in sorted(review.items())]}


def _group_label(group: dict) -> str:
    key = group["key"]
    return (f"group {group['id']} {key['role']} {key['mode']} real={key['real']} own_rows={key['own_rows']} "
            f"lanes={tuple(key['lane_vector'])} requests={key['requests']} forwards={group['forward_count']}")


def _kernel_text(row: dict) -> str:
    return (f"{row['count']:4d} {row['family'] or 'auxiliary':17s} {row['name'][:60]:60s} "
            f"grid={tuple(row['grid'])} block={tuple(row['block'])}")


def _review_text(label: str, rows: list[dict]) -> list[str]:
    return [label, *[f"  {row['count']} {row['name']}" for row in rows]] if rows else [label + " empty"]


def native_summary(table: dict) -> str:
    lines = [table["trace_dir"], table["duration_note"], table["pairing_note"]]
    for pairing in table["pairing"]:
        lines.append(f"PAIRING {pairing['role']} accepted_tuples={pairing['accepted_tuples']} "
                     f"accepted_pairs={pairing['accepted_pairs']} left_out={pairing['left_out_per_lane']} "
                     f"start_difference_us={pairing['start_difference_us']}")
    for group in table["groups"]:
        lines.extend(["", _group_label(group) + f" distinct_multisets={group['distinct_multisets']}"])
        for ordinal, variant in enumerate(group["multisets"]):
            lines.append(f"  multiset {ordinal}, forwards={variant['forward_count']}")
            for row in variant["kernels"]:
                lines.append(f"    {row['segment']:8s} {_kernel_text(row)}")
        for row in group["kernel_stats"]:
            lines.append(f"  duration_us {row['segment']} {row['name'][:60]} grid={row['grid']} block={row['block']} "
                         f"counts_per_forward={row['counts_per_forward']} {row['duration_us']}")
    lines.extend(["", *_review_text("Native review list:", table["review_list"])])
    return "\n".join(lines) + "\n"


def compare_summary(table: dict) -> str:
    lines = [table["native_trace_dir"], table["duration_note"], table["selection_note"]]
    for group in table["groups"]:
        lines.extend(["", _group_label(group)])
        lines.append(f"VERDICT group {group['id']}: kernels={group['kernels'] or group['status']} "
                     f"durations={dict(Counter(row['verdict'] for row in group['durations']))}")
        for expected in group["expected_by_lane"]:
            lane = expected["lane"]
            variants = [(ordinal, variant) for ordinal, variant in enumerate(group["native_multisets"])
                        if any(ref["lane"] == lane for ref in variant["forwards"])]
            for ordinal, variant in variants:
                lines.append(f"  lane {lane}, native multiset {ordinal} (forwards={variant['forward_count']}): "
                             "native [segment] | benchmark expected")
                native_rows = variant["kernels"]
                expected_rows = expected["kernels"]
                for index in range(max(len(native_rows), len(expected_rows))):
                    left = (f"[{native_rows[index]['segment']}] " + _kernel_text(native_rows[index])
                            if index < len(native_rows) else "-")
                    right = _kernel_text(expected_rows[index]) if index < len(expected_rows) else "-"
                    lines.append(f"    {left} | {right}")
        for row in group["missing"]:
            lines.append(f"  UNMATCHED lane={row['lane']} ({row['scenario']}, {row['call']}) searched={row['searched']}")
        lines.extend(f"  ERROR {error}" for error in group["errors"])
        for difference in group["differences"]:
            lines.append(f"  DIFFERENT lane={difference['lane']} multiset={difference['native_multiset']}")
            lines.extend(f"    {row['name'][:60]} grid={row['grid']} block={row['block']} "
                         f"native={row['native_count']} expected={row['expected_count']} "
                         f"native_minus_expected={row['native_minus_expected']}" for row in difference["kernels"])
        for row in group["durations"]:
            lines.append(f"  DURATION {row['verdict']} lane={row['lane']} {row['name'][:60]} "
                         f"grid={row['grid']} block={row['block']} native_us={row['native_us']} benchmark_us={row['benchmark_us']}")
    lines.extend(["", f"Verdict counts: {table['verdict_counts']}", "Auxiliary device work (reported, not compared):"])
    for case in table["cases"]:
        if case["auxiliary"] or case["errors"]:
            lines.append(f"  {case['benchmark']} {case['scenario']} {case['call']} rank={case['rank']}")
            lines.extend(f"    ERROR {error}" for error in case["errors"])
            lines.extend(f"    {row['count_per_call']:g}/call {row['category']} {row['name']} "
                         f"duration_us={row['duration_us']}" for row in case["auxiliary"])
    lines.extend(["", *_review_text("Native review list:", table["native_review_list"]),
                  *_review_text("Benchmark review list:", table["benchmark_review_list"])])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    native = commands.add_parser("native")
    native.add_argument("--trace-dir", required=True, type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("--native", required=True, type=Path)
    compare.add_argument("--benchmark", action="append", required=True, type=Path)
    for command in (native, compare):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "native":
        table = native_table(args.trace_dir)
        summary = native_summary(table)
    else:
        table = compare_tables(json.loads(args.native.read_text()), args.benchmark)
        summary = compare_summary(table)
    args.output.write_text(json.dumps(table, indent=1) + "\n")
    args.summary.write_text(summary)
    print(f"WROTE {args.output} {args.summary}", flush=True)
    return int(args.command == "compare" and any(case["errors"] for case in table["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
