"""Describe copied Step communication evidence without enforcing a runtime gate."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from frontier.profiling.collectives.step5_dispatch_probe import (
    compare_dispatch, compare_placement, expected_topology,
)
from frontier.profiling.collectives.vllm_forward_collectives import (
    DTYPE_BYTES, ModelShape, ServingParallelism, call_bytes, forward_collectives,
    group_members, scenarios_from_grid,
)


def iter_jsonl(path: Path):
    """Read one record at a time, including detached gzip-only copies."""
    if not path.exists() and path.suffix != ".gz":
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_stats(values: list[float]) -> dict:
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else None
    return {
        "repeat_count": len(values),
        "median_us": statistics.median(values),
        "mean_us": mean,
        "sample_std_us": deviation,
        "sample_cv": deviation / mean if deviation is not None and mean else None,
        "min_us": min(values),
        "max_us": max(values),
        "range_over_median": (max(values) - min(values)) / statistics.median(values),
    }


def queue_held_analysis(record: dict, timing_domain: str, threshold: float) -> dict:
    """Filter repeat positions by the intersection of participating-rank flags."""
    ranks = record["participating_ranks"]
    timings = record["per_rank_call_us"]
    count = len(timings[ranks[0]])
    critical = [max(timings[rank][j] for rank in ranks) for j in range(count)]
    launch = record["per_rank_launch_call_us"]
    result = timing_stats(critical)
    result.update({
        "critical_path_repeats_us": critical,
        "launch_critical_path_median_us": statistics.median(
            max(launch[rank][j] for rank in ranks) for j in range(count)
        ) if timing_domain == "device" else None,
        "rank_repeat_count": len(ranks) * count,
    })
    if timing_domain != "device":
        result.update({"queue_gate": "NOT_APPLICABLE", "retained_repeat_indices": []})
        return result
    flags = record["per_rank_queue_held"]
    if any(flags[rank] is None for rank in ranks):
        result.update({
            "queue_gate": "UNVERIFIED", "retained_repeat_indices": [],
            "all_rank_held_count": 0, "all_rank_held_fraction": None,
            "per_rank_held_count": 0, "per_rank_mean_held_fraction": None,
            "filtered_median_us": None, "absolute_fractional_median_delta": None,
        })
        return result
    retained = [j for j in range(count) if all(flags[rank][j] for rank in ranks)]
    rank_held = sum(sum(flags[rank]) for rank in ranks)
    filtered = statistics.median(critical[j] for j in retained) if retained else None
    delta = abs(filtered - result["median_us"]) / result["median_us"] if filtered is not None else None
    result.update({
        "retained_repeat_indices": retained,
        "all_rank_held_count": len(retained),
        "all_rank_held_fraction": len(retained) / count,
        "per_rank_held_count": rank_held,
        "per_rank_mean_held_fraction": rank_held / (len(ranks) * count),
        "filtered_median_us": filtered,
        "absolute_fractional_median_delta": delta,
        "queue_gate": "UNVERIFIED" if delta is None else "PASS" if delta < threshold else "FAIL",
    })
    return result


def payload_bytes(record: dict) -> dict:
    """Return tensor endpoint bytes, not algorithm-dependent network traffic."""
    row_bytes = sum(DTYPE_BYTES[dtype] * columns for dtype, columns in zip(record["dtypes"], record["columns"]))
    groups = {rank: group for group in record["group_ranks"] for rank in group}
    rows = record["shard_rows"]
    inputs, outputs = {}, {}
    for rank in record["participating_ranks"]:
        local = rows[rank] * row_bytes
        grouped = sum(rows[peer] for peer in groups[rank]) * row_bytes
        inputs[rank] = grouped if record["op"] == "reduce_scatterv" else local
        outputs[rank] = grouped if record["op"] in ("all_gather", "all_gatherv") else local
    return {
        "row_bytes": row_bytes,
        "per_rank_input_bytes": inputs,
        "per_rank_output_bytes": outputs,
        "sum_rank_input_bytes": sum(inputs.values()),
        "sum_rank_output_bytes": sum(outputs.values()),
        "payload_maxima_match": max(inputs.values()) == record["max_input_bytes"] and max(outputs.values()) == record["max_output_bytes"],
    }


def label_noise(rows: list[dict]) -> list[dict]:
    result = []
    groups = defaultdict(list)
    for row in rows:
        if row["op"] == "all_reduce" and row["group"] == "tp":
            groups[row["pass"], row["scenario"]].append(row)
    for (pass_id, scenario), labels in groups.items():
        medians = [row["median_us"] for row in labels]
        deviations = [row["sample_std_us"] for row in labels if row["sample_std_us"] is not None]
        cvs = [row["sample_cv"] for row in labels if row["sample_cv"] is not None]
        result.append({
            "pass": pass_id, "scenario": scenario,
            "labels": [row["call"] for row in labels],
            "label_count": len(labels),
            "calls_per_forward_sum": sum(row["calls_per_forward"] for row in labels),
            "same_payload_and_dispatch": len({json.dumps([row["per_rank_input_bytes"], row["per_rank_output_bytes"], row["communicator"], row["kernels"]], sort_keys=True) for row in labels}) == 1,
            "median_of_label_medians_us": statistics.median(medians),
            "minimum_label_median_us": min(medians),
            "maximum_label_median_us": max(medians),
            "label_median_sample_cv": statistics.stdev(medians) / statistics.mean(medians) if len(medians) > 1 else None,
            "label_median_range_over_median": (max(medians) - min(medians)) / statistics.median(medians),
            "median_within_label_sample_cv": statistics.median(cvs) if cvs else None,
            "pooled_within_label_sample_cv": math.sqrt(statistics.mean(d ** 2 for d in deviations)) / statistics.mean(row["mean_us"] for row in labels) if deviations else None,
            "minimum_within_label_sample_cv": min(cvs) if cvs else None,
            "maximum_within_label_sample_cv": max(cvs) if cvs else None,
        })
    return result


def comparison_rows(rows: list[dict]) -> list[dict]:
    index = {(row["pass"], row["scenario"], row["call"]): row for row in rows}
    scenarios = {row["scenario"] for row in rows}
    comparisons = []
    for candidate in sorted(scenarios):
        for prefix, replacement, name in (
            ("graph_one_rail_b", "graph_decode_b", "rails_1_vs_all"),
            ("eager_one_rail_t", "eager_uniform_t", "rails_1_vs_all"),
            ("graph_idle_b", "graph_decode_b", "graph_idle_vs_uniform"),
            ("eager_idle_t", "eager_uniform_t", "eager_idle_vs_uniform"),
            ("eager_uneven_t", "eager_uniform_t", "eager_uneven_vs_uniform"),
            ("eager_uniform_t", "graph_decode_b", "eager_vs_graph_same_physical_tokens"),
        ):
            if candidate.startswith(prefix):
                size = candidate[len(prefix):].split("_")[0]
                baseline = replacement + size
                if baseline in scenarios:
                    comparisons.append((name, baseline, candidate))
    result = []
    for name, baseline, candidate in comparisons:
        for (pass_id, scenario, call), row in index.items():
            if scenario != baseline:
                continue
            other = index.get((pass_id, candidate, call))
            if other is None:
                continue
            comparison = name
            if name == "rails_1_vs_all":
                rail_counts = [r["concurrent_dp_rails"] for r in rows if r["pass"] == pass_id and r["scenario"] == baseline and r["group"] == "dp"]
                comparison = f"rails_1_vs_{max(rail_counts)}"
            filtered_delta = None
            if row.get("filtered_median_us") is not None and other.get("filtered_median_us") is not None:
                filtered_delta = other["filtered_median_us"] / row["filtered_median_us"] - 1
            result.append({
                "pass": pass_id, "comparison": comparison, "call": call, "baseline_scenario": baseline,
                "candidate_scenario": candidate, "baseline_median_us": row["median_us"],
                "candidate_median_us": other["median_us"],
                "fractional_change": other["median_us"] / row["median_us"] - 1,
                "filtered_fractional_change": filtered_delta,
                "baseline_queue_gate": row["queue_gate"], "candidate_queue_gate": other["queue_gate"],
                "baseline_max_input_bytes": row["max_input_bytes"], "candidate_max_input_bytes": other["max_input_bytes"],
                "baseline_max_output_bytes": row["max_output_bytes"], "candidate_max_output_bytes": other["max_output_bytes"],
                "same_endpoint_payload": row["per_rank_input_bytes"] == other["per_rank_input_bytes"] and row["per_rank_output_bytes"] == other["per_rank_output_bytes"],
            })
    return result


def native_rows(probe: dict) -> tuple[list[dict], list[dict]]:
    layers, collectives = [], []
    for rank in probe["ranks"]:
        for layer in rank["layers"]:
            base = {
                "rank": rank["rank"], "rank_status": rank["status"], "layer_id": layer["layer_id"],
                "host": rank["host"], "local_rank": rank["local_rank"],
                "quant_method": layer["quant_method"], "oracle_backend": layer["oracle_backend"],
                "experts_class": layer["experts_class"], "executed_experts_classes": layer["executed_experts_classes"],
                "prepare_finalize_class": layer["prepare_finalize_class"],
                "comparison": layer["comparison"],
            }
            layers.append(base)
            for number, collective in enumerate(layer["collectives"]):
                scales = [tensor for tensor in collective["inputs"] if tensor["dtype"] == "float32" and tensor["shape"][-1] == 32]
                collectives.append({
                    **base, "collective_index": number, **collective,
                    "input_bytes": sum(tensor["nbytes"] for tensor in collective["inputs"]),
                    "output_bytes": sum(tensor["nbytes"] for tensor in collective["outputs"]),
                    "activation_scale_tensors": scales,
                })
    return layers, collectives


def native_evidence(probe: dict, gpus_per_node: int) -> dict:
    """Recheck copied schemas with the native probe's existing CPU comparator."""
    topology = expected_topology(probe["parallelism"]["tp"], probe["parallelism"]["dp"], gpus_per_node)
    expected = {r["rank"]: r for r in topology["ranks"]}
    cases = {case["layer_id"]: case for case in probe["cases"]}
    layers, collectives = native_rows(probe)
    layer_index = {(r["rank"], r["layer_id"]): r for r in layers}
    descriptor_matches = 0
    descriptor_count = 0
    for rank in probe["ranks"]:
        groups = expected[rank["rank"]]["group_ranks"]
        for layer in rank["layers"]:
            case = cases[layer["layer_id"]]
            hidden = case["expected_columns_dtypes"][0][0]
            rows = probe["tokens_per_dp_rank"][rank["rank"] // probe["parallelism"]["tp"]]
            comparison = compare_dispatch(case, layer["collectives"], rows=rows,
                                          sizes=probe["tokens_per_dp_rank"],
                                          dp_ranks=groups["dp"], tp_ranks=groups["tp"], hidden=hidden)
            ar_schema_matches = all(
                [(t["shape"], t["dtype"]) for t in call[endpoint]] == [([rows, hidden], "bfloat16")]
                for call in layer["collectives"] if call["op"] == "all_reduce"
                for endpoint in ("inputs", "outputs"))
            item = layer_index[rank["rank"], layer["layer_id"]]
            item.update(recomputed_comparison=comparison, tp_all_reduce_schema_matches=ar_schema_matches,
                        shape_match=comparison["status"] == "match" and ar_schema_matches,
                        observer_errors=layer.get("observer_errors", []),
                        diagnostic_errors=layer.get("diagnostic_errors", []))
    for call in collectives:
        valid = True
        for tensor in call["inputs"] + call["outputs"]:
            descriptor_count += 1
            match = (tensor["element_size"] == DTYPE_BYTES[tensor["dtype"]]
                     and tensor["numel"] == math.prod(tensor["shape"])
                     and tensor["nbytes"] == tensor["numel"] * tensor["element_size"]
                     and len(tensor["stride"]) == len(tensor["shape"]))
            descriptor_matches += match
            valid &= match
        call["tensor_descriptors_match"] = valid
    actual_layers = [(r["rank"], r["layer_id"]) for r in layers]
    expected_layers = {(rank, layer) for rank in expected for layer in cases}
    return {
        "layers": layers, "collectives": collectives,
        "expected_rank_layer_records": len(expected_layers),
        "rank_layer_keys_complete": len(actual_layers) == len(expected_layers) and set(actual_layers) == expected_layers,
        "rank_layer_source_matches": sum(r["comparison"]["status"] == "match" for r in layers),
        "rank_layer_shape_matches": sum(r["shape_match"] for r in layers),
        "expected_collective_records": len(expected_layers) * 4,
        "tensor_descriptors": descriptor_count, "tensor_descriptor_matches": descriptor_matches,
        "placement": compare_placement(topology, probe["ranks"]),
        "rank_group_matches": sum(rank["group_ranks"] == expected[rank["rank"]]["group_ranks"] for rank in probe["ranks"]),
        "placements": [{"rank": r["rank"], "host": r["host"], "local_rank": r["local_rank"],
                        "group_ranks": r["group_ranks"]} for r in probe["ranks"]],
    }


def pass_spreads(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario"], row["call"]].append(row)
    return [{"scenario": scenario, "call": call,
             "passes": [r["pass"] for r in sorted(items, key=lambda r: r["pass"])],
             "pass_count": len(items), "timing_domain": items[0]["timing_domain"],
             "pass_medians_us": [r["median_us"] for r in sorted(items, key=lambda r: r["pass"])],
             **{key: value for key, value in timing_stats([r["median_us"] for r in items]).items()
                if key != "repeat_count"}}
            for (scenario, call), items in groups.items()]


def queue_counts(rows: list[dict]) -> dict:
    device = [row for row in rows if row["timing_domain"] == "device"]
    repeats = sum(row["repeat_count"] for row in device)
    rank_repeats = sum(row["rank_repeat_count"] for row in device)
    held = sum(row.get("all_rank_held_count", 0) for row in device)
    rank_held = sum(row.get("per_rank_held_count", 0) for row in device)
    return {
        "device_records": len(device), "host_records": len(rows) - len(device),
        "queue_gate_verdicts": dict(Counter(row["queue_gate"] for row in device)),
        "device_repeat_positions": repeats, "all_rank_held_repeat_positions": held,
        "all_rank_held_fraction": held / repeats if repeats else None,
        "device_rank_repeat_observations": rank_repeats, "per_rank_held_observations": rank_held,
        "per_rank_mean_held_fraction": rank_held / rank_repeats if rank_repeats else None,
        "device_world_rank_repeat_positions": sum(row["world_rank_repeat_count"] for row in device),
        "device_nonparticipating_rank_repeat_positions": sum(row["nonparticipating_rank_repeat_count"] for row in device),
        "device_useful_idle_participating_rank_repeat_observations": sum(row["useful_idle_rank_repeat_count"] for row in device),
        "retained_count_histogram": dict(Counter(row.get("all_rank_held_count", 0) for row in device)),
    }


def analyze_result(directory: Path, threshold: float) -> dict:
    kernels = read_jsonl(directory / "collectives/kernels.jsonl")
    kernel_index = {(row["scenario"], row["call"]): row for row in kernels}
    with (directory / "collectives/summary.csv").open() as handle:
        summaries = list(csv.DictReader(handle))
    summary_index = {(int(row["pass"]), row["scenario"], row["call"]): row for row in summaries}
    plan = json.loads((directory / "collectives/plan.json").read_text())
    config = plan["config"]
    parallel = ServingParallelism.from_engine(config["engine"])
    scenarios = list(scenarios_from_grid(parallel, config["grid"]))
    manifest = json.loads((directory / "manifest.json").read_text())
    model_path = Path(config["model_config"])
    model = ModelShape.from_hf_config(json.loads(model_path.read_text()))
    planned_calls = {(scenario.name, call.name): (scenario, call)
                     for scenario in scenarios for call in forward_collectives(model, parallel, scenario)}
    expected_cases = set(planned_calls)
    expected_keys = {(p, *key) for p in range(config["passes"]) for key in expected_cases}
    keys = []
    rows = []
    for record in iter_jsonl(directory / "collectives/measurements.jsonl"):
        keys.append((record["pass"], record["scenario"], record["call"]))
        summary = summary_index[record["pass"], record["scenario"], record["call"]]
        kernel = kernel_index[record["scenario"], record["call"]]
        row = {key: value for key, value in record.items() if not key.startswith("per_rank_")}
        row.update(kernel)
        row["timing_domain"] = summary["timing_domain"]
        row.update(queue_held_analysis(record, row["timing_domain"], threshold))
        row.update(payload_bytes(record))
        scenario, call = planned_calls[record["scenario"], record["call"]]
        rails = ((0,) if call.op == "cpu_all_reduce" else scenario.active_dp_rails
                 if scenario.active_dp_rails is not None else tuple(range(parallel.tensor_parallel_size)))
        ranks = [rank for rank in range(parallel.world_size)
                 if (call.group != "dp" or rank % parallel.tensor_parallel_size in rails)
                 and any(call_bytes(call, parallel, rank))]
        groups = sorted({tuple(group_members(parallel, call.group, rank)) for rank in ranks})
        identity = {"op": call.op, "group": call.group, "calls_per_forward": call.calls_per_forward,
                    "shard_rows": list(call.shard_rows),
                    "group_size": len(groups[0]), "cuda_graph": scenario.cuda_graph and call.captured,
                    "tokens_per_dp_rank": list(scenario.tokens_per_dp_rank),
                    "useful_tokens_per_dp_rank": list(scenario.useful_tokens_per_dp_rank or scenario.tokens_per_dp_rank),
                    "padded_tokens_per_dp_rank": list(scenario.tokens_per_dp_rank),
                    "participating_ranks": ranks, "group_ranks": [list(g) for g in groups],
                    "dtypes": list(call.dtypes), "columns": list(call.columns),
                    "concurrent_dp_rails": len(groups) if call.group == "dp" else None,
                    "max_input_bytes": max(call_bytes(call, parallel, rank)[0] for rank in ranks),
                    "max_output_bytes": max(call_bytes(call, parallel, rank)[1] for rank in ranks)}
        row["identity_differences"] = {key: {"expected": value, "actual": record[key]}
                                       for key, value in identity.items() if record[key] != value}
        row["calls_per_repeat"] = config["calls_per_repeat"]
        row["world_rank_repeat_count"] = parallel.world_size * row["repeat_count"]
        row["nonparticipating_ranks"] = sorted(set(range(parallel.world_size)) - set(record["participating_ranks"]))
        row["nonparticipating_rank_repeat_count"] = len(row["nonparticipating_ranks"]) * row["repeat_count"]
        useful = record["useful_tokens_per_dp_rank"]
        row["useful_idle_participating_ranks"] = [rank for rank in record["participating_ranks"] if useful[rank // parallel.tensor_parallel_size] == 0]
        row["useful_idle_rank_repeat_count"] = len(row["useful_idle_participating_ranks"]) * row["repeat_count"]
        row["repeat_schema_matches"] = all(
            len(record[field]) == parallel.world_size
            and all(record[field][rank] is not None and len(record[field][rank]) == config["repeats"]
                    for rank in record["participating_ranks"])
            for field in ("per_rank_call_us", "per_rank_launch_call_us")
            if field == "per_rank_call_us" or row["timing_domain"] == "device")
        row["queue_flag_schema_matches"] = row["timing_domain"] != "device" or (
            len(record["per_rank_queue_held"]) == parallel.world_size
            and all(record["per_rank_queue_held"][rank] is not None
                    and len(record["per_rank_queue_held"][rank]) == config["repeats"]
                    and all(isinstance(flag, bool) for flag in record["per_rank_queue_held"][rank])
                    for rank in record["participating_ranks"]))
        row["positive_finite_repeats"] = all(math.isfinite(t) and t > 0
            for rank in record["participating_ranks"] for t in record["per_rank_call_us"][rank])
        row["source_median_matches"] = math.isclose(row["median_us"], float(summary["median_us"]), rel_tol=1e-12)
        row["source_queue_fraction"] = float(summary["queue_held_fraction"]) if summary["queue_held_fraction"] else None
        row["source_queue_fraction_matches"] = (row["source_queue_fraction"] is not None
            and row.get("per_rank_mean_held_fraction") is not None
            and math.isclose(row["source_queue_fraction"], row["per_rank_mean_held_fraction"], rel_tol=1e-12))
        rows.append(row)
    device = [row for row in rows if row["timing_domain"] == "device"]
    failures = [row for row in device if row["queue_gate"] != "PASS"]
    observed_rank_fractions = [row["per_rank_mean_held_fraction"] for row in device if row["per_rank_mean_held_fraction"] is not None]
    probe = json.loads((directory / "native_dispatch/dispatch_probe.json").read_text())
    native = native_evidence(probe, config["gpus_per_node"])
    layers, native_collectives = native.pop("layers"), native.pop("collectives")
    native_rank_file_matches = sum(rank == json.loads((directory / f"native_dispatch/rank_{rank['rank']}.json").read_text()) for rank in probe["ranks"])
    native_events = {rank["rank"]: dict(Counter(row["event"] for row in read_jsonl(directory / f"native_dispatch/rank_{rank['rank']}.jsonl"))) for rank in probe["ranks"]}
    measured_shifts = [row for row in device if row["absolute_fractional_median_delta"] is not None]
    worst = max(measured_shifts, key=lambda row: row["absolute_fractional_median_delta"]) if measured_shifts else None
    counts = {
        "measurement_records": len(rows), "measurement_unique_keys": len(set(keys)),
        "kernel_records": len(kernels), "kernel_unique_keys": len(kernel_index),
        "summary_records": len(summaries), "summary_unique_keys": len(summary_index),
        "planned_cases": plan["cases"], "expected_measurement_keys": len(expected_keys),
        "observed_passes": sorted({row["pass"] for row in rows}),
        "measurement_records_by_pass": dict(Counter(row["pass"] for row in rows)),
        **queue_counts(rows),
        "mean_record_per_rank_held_fraction": statistics.mean(observed_rank_fractions) if observed_rank_fractions else None,
        "missing_device_flag_records": sum(row["all_rank_held_fraction"] is None for row in device),
        "source_median_matches": sum(row["source_median_matches"] for row in rows),
        "source_queue_fraction_matches": sum(row["source_queue_fraction_matches"] for row in device),
        "payload_maxima_match": sum(row["payload_maxima_match"] for row in rows),
        "repeat_counts": sorted({row["repeat_count"] for row in rows}),
        "native_rank_records": len(probe["ranks"]),
        "native_rank_layer_records": len(layers), "native_collective_records": len(native_collectives),
        "native_completed_collectives": sum(row["completed"] for row in native_collectives),
        "native_rank_file_matches": native_rank_file_matches,
        "native_rank_layer_shape_matches": native["rank_layer_shape_matches"],
    }
    checks = {
        "measurement_keys": len(keys) == len(expected_keys) and set(keys) == expected_keys,
        "kernel_keys": len(kernels) == len(expected_cases) == plan["cases"] and set(kernel_index) == expected_cases,
        "summary_keys": len(summaries) == len(expected_keys) and set(summary_index) == expected_keys,
        "repeat_schema": all(row["repeat_schema_matches"] for row in rows),
        "queue_flag_schema": all(row["queue_flag_schema_matches"] for row in device),
        "positive_finite_repeats": all(row["positive_finite_repeats"] for row in rows),
        "source_medians": counts["source_median_matches"] == len(rows),
        "source_queue_fractions": counts["source_queue_fraction_matches"] == len(device),
        "payload_maxima": counts["payload_maxima_match"] == len(rows),
        "planned_payload_groups_rails": all(not row["identity_differences"] for row in rows),
        "model_config_hash": sha256(model_path) == manifest["job"]["source_hashes"][str(model_path.relative_to(manifest["job"]["repository"]))],
        "native_rank_layer_keys": native["rank_layer_keys_complete"],
        "native_shapes": native["rank_layer_shape_matches"] == native["expected_rank_layer_records"],
        "native_source_comparisons": probe["status"] == "match" and all(
            rank["status"] == "match" and all(layer["comparison"]["status"] == "match"
                                            for layer in rank["layers"])
            for rank in probe["ranks"]),
        "native_observer_diagnostics": not probe.get("diagnostic_errors") and all(
            not rank.get("diagnostic_errors") and all(not layer.get("observer_errors")
                and not layer.get("diagnostic_errors") for layer in rank["layers"])
            for rank in probe["ranks"]),
        "native_collective_count": len(native_collectives) == native["expected_collective_records"],
        "native_completed_collectives": all(row["completed"] for row in native_collectives),
        "native_tensor_descriptors": native["tensor_descriptor_matches"] == native["tensor_descriptors"],
        "native_rank_files": native_rank_file_matches == parallel.world_size,
        "native_placement": native["placement"]["status"] == "match",
        "native_groups": native["rank_group_matches"] == parallel.world_size,
    }
    stage = manifest["job"]["run_id"].split("_")[0].upper()
    return {
        "task_id": f"T12-{stage}-STATS", "input_directory": str(directory),
        "run_id": manifest["job"]["run_id"], "stage": stage,
        "model_config_input": {"path": str(model_path), "sha256": sha256(model_path)},
        "environment": {"python_executable": sys.executable, "python_version": sys.version, "platform": platform.platform()},
        "threshold": threshold, "threshold_comparison": "strictly less than",
        "quality_gate_is_advisory": True,
        "queue_gate": "FAIL" if failures else "PASS",
        "schema_checks": checks, "schema_status": "PASS" if all(checks.values()) else "UNVERIFIED",
        "plan": plan,
        "worker_manifest": manifest,
        "worker_result": json.loads((directory / "result.json").read_text()),
        "counts": counts,
        "counts_by_pass": [{"pass": p, **queue_counts([r for r in rows if r["pass"] == p])}
                           for p in counts["observed_passes"]],
        "native_evidence": native,
        "native_probe_scope": {key: value for key, value in probe.items() if key not in ("ranks", "topology")},
        "native_events_by_rank": native_events,
        "worst_queue_shift": {key: worst[key] for key in ("pass", "scenario", "call", "median_us", "filtered_median_us", "absolute_fractional_median_delta")} if worst else None,
        "queue_failures": failures, "timings": rows, "label_noise": label_noise(rows),
        "pass_spreads": pass_spreads(rows),
        "comparisons": comparison_rows(rows), "native_layers": layers, "native_collectives": native_collectives,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def markdown_report(result: dict) -> str:
    counts = result["counts"]
    rows = result["timings"]
    config = result["plan"]["config"]
    native = result["native_evidence"]
    by_key = {(row["pass"], row["scenario"], row["call"]): row for row in rows}
    lines = [
        f"# {result['task_id']}: local {result['stage']} communication analysis", "", "## Modification History", "",
        "| Date | Change |", "| --- | --- |",
        f"| {date.today().isoformat()} | Recomputed copied evidence by pass/scenario/call; advisory verdict only. |", "",
        f"Run `{result['run_id']}`. Schema: **{result['schema_status']}**. Queue-held check: **{result['queue_gate']}**, {counts['queue_gate_verdicts']}; {counts['host_records']} host/Gloo records are not applicable.",
        f"Full median = median_j(max_participating_rank(t_rank,j)); keep j only when every participating rank held j. PASS means abs(filtered-full)/full < {100 * result['threshold']:g}%.",
        f"All-rank held: {counts['all_rank_held_repeat_positions']}/{counts['device_repeat_positions']} = {100 * counts['all_rank_held_fraction']:.4f}%; rank-average held: {counts['per_rank_held_observations']}/{counts['device_rank_repeat_observations']} = {100 * counts['per_rank_mean_held_fraction']:.4f}% (different denominator).",
        f"Measurement keys: {counts['measurement_unique_keys']}/{counts['expected_measurement_keys']}; kernel keys: {counts['kernel_unique_keys']}/{counts['planned_cases']}; summary keys: {counts['summary_unique_keys']}/{counts['expected_measurement_keys']}. Recomputed source medians: {counts['source_median_matches']}/{counts['measurement_records']}; source queue fractions: {counts['source_queue_fraction_matches']}/{counts['device_records']}; payload maxima: {counts['payload_maxima_match']}/{counts['measurement_records']}.",
        f"Passes: {counts['observed_passes']}; rows by pass: {counts['measurement_records_by_pass']}. Repeats per cell: {counts['repeat_counts']} (planned {config['repeats']}); each repeat averages {config['calls_per_repeat']} calls, not independent call samples. Passes are not pooled into a larger within-cell sample.",
        f"Retained-repeat histogram: {counts['retained_count_histogram']}; missing device flag records: {counts['missing_device_flag_records']}.",
        f"Device participating-rank denominator includes {counts['device_useful_idle_participating_rank_repeat_observations']} logically idle rank/repeat observations with dummy or padded rows. It excludes {counts['device_nonparticipating_rank_repeat_positions']} nonparticipating rank/repeat positions (zero-row logits or inactive rails). All world-rank positions: {counts['device_world_rank_repeat_positions']}; participating plus nonparticipating equals this total. Counts sum distinct cells and passes for accounting, not statistical independence.",
        "", "## Schema checks", "", "| Check | Result |", "| --- | --- |",
    ]
    for name, passed in result["schema_checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'UNVERIFIED'} |")
    lines.extend(["", "## Per-pass queue accounting", "",
        "| Pass | Device/host cells | Held intersections / repeats | Held ranks / rank observations | Verdicts |",
        "| --- | ---: | ---: | ---: | --- |"])
    for item in result["counts_by_pass"]:
        lines.append(f"| {item['pass']} | {item['device_records']}/{item['host_records']} | {item['all_rank_held_repeat_positions']}/{item['device_repeat_positions']} | {item['per_rank_held_observations']}/{item['device_rank_repeat_observations']} | {item['queue_gate_verdicts']} |")
    lines.extend([
        "", "## Timings (microseconds, full unfiltered medians)", "",
        "AR is the median across same-payload label medians with their min–max; it is not an additional measurement. timings.csv retains every label's critical-rank repeat averages, mean, sample CV, min/max, queue fractions, held indices, launch median, rank byte maps and endpoint sums. No outliers are removed.", "",
        "| Pass | Scenario; physical tokens/DP | TP AR [label range] | FP8 AGv | BF16 AGv | RSv | Logits AG | Host sync |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for item in sorted(result["label_noise"], key=lambda r: (r["pass"], r["scenario"])):
        p, scenario = item["pass"], item["scenario"]
        record = by_key[p, scenario, item["labels"][0]]
        values = [by_key[p, scenario, call]["median_us"] for call in ("moe_dispatch_all_gatherv_fp8", "moe_dispatch_all_gatherv_bf16", "moe_combine_reduce_scatterv", "logits_all_gather", "dp_batch_sync_all_reduce")]
        lines.append(f"| {p} | {scenario}; {'/'.join(map(str, record['tokens_per_dp_rank']))} | {item['median_of_label_medians_us']:.2f} [{item['minimum_label_median_us']:.2f}–{item['maximum_label_median_us']:.2f}] | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    lines.extend(["", "## Queue-held failures", "", "| Pass | Scenario / call | Kept / repeats | Full us | Held-only us | Absolute shift |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in sorted(result["queue_failures"], key=lambda row: row.get("absolute_fractional_median_delta") or 0, reverse=True):
        filtered = f"{row['filtered_median_us']:.6f}" if row['filtered_median_us'] is not None else "unverified"
        shift = f"{100 * row['absolute_fractional_median_delta']:.4f}%" if row['absolute_fractional_median_delta'] is not None else "undefined"
        lines.append(f"| {row['pass']} | {row['scenario']} / {row['call']} | {row['all_rank_held_count']}/{row['repeat_count']} | {row['median_us']:.6f} | {filtered} | {shift} |")
    lines.extend(["", "## Repeat and pass distributions", "",
        "CV is sample standard deviation / mean of actual critical-rank repeat averages, separately per cell and pass. Pass spread = (max pass median - min pass median) / median of pass medians; no new threshold is imposed. Label-median CV describes isolated labels, not a causal test or confidence bound.", "",
        "| Pass | Device cells | Repeat CV min / median / max | Worst CV cell |",
        "| --- | ---: | ---: | --- |"])
    for p in counts["observed_passes"]:
        items = [r for r in rows if r["pass"] == p and r["timing_domain"] == "device" and r["sample_cv"] is not None]
        cvs = [r["sample_cv"] for r in items]
        if items:
            worst = max(items, key=lambda r: r["sample_cv"])
            lines.append(f"| {p} | {len(items)} | {100 * min(cvs):.4f}% / {100 * statistics.median(cvs):.4f}% / {100 * max(cvs):.4f}% | {worst['scenario']} / {worst['call']} |")
    for domain in ("device", "host"):
        spreads = [r for r in result["pass_spreads"] if r["timing_domain"] == domain]
        if spreads:
            worst = max(spreads, key=lambda r: r["range_over_median"])
            values = [r["range_over_median"] for r in spreads]
            lines.append(f"\n{domain} pass spreads ({len(spreads)} cells), min/median/max: {100 * min(values):.4f}% / {100 * statistics.median(values):.4f}% / {100 * max(values):.4f}%; worst `{worst['scenario']} / {worst['call']}`: {worst['pass_medians_us']} us.")
    lines.extend(["", f"Label counts per pass/scenario: {sorted({r['label_count'] for r in result['label_noise']})}; configured call-count sums: {sorted({r['calls_per_forward_sum'] for r in result['label_noise']})}. They are isolated measurements, not a measured native forward chain. forward_totals.csv weights isolated medians, not native timing. Native dispatch has zero timing rows.", "",
        "label_noise.csv preserves every pass/scenario; pass_spreads.csv preserves every scenario/call. These batches do not establish independent-run variance or a sub-1% noise floor. No environmental cause is inferred.", "",
        "## Supported comparisons", "",
        f"comparisons.csv contains {len(result['comparisons'])} within-pass paired call rows, full and held-only ratios, both queue verdicts, byte maxima and endpoint-payload equality. Only existing pairs are compared; missing sizes are not substituted. Rail, idle and uneven comparisons are descriptive, not causal scaling or speedup qualifications. Equal physical token counts do not guarantee equal logits payload or dispatch and do not alone isolate graph benefit.", "",
        "## Payloads and observed dispatch", "",
        "Tensor endpoint bytes are not network wire traffic. timings.csv preserves participating-rank byte maps, sums and maxima. Logically idle lanes with positive dummy/padded rows count; zero-row logits and inactive rail ranks do not. kernel_dispatch.csv preserves observed names by pass/scenario/call, but profiler observations were collected per scenario/call, not independently per timing pass. Enabled communicators do not prove execution; aggregated kernel lists do not attribute kernels to ranks.", "",
        "## Native layer identity and completeness", "",
        f"{counts['native_rank_records']} rank files; {counts['native_rank_file_matches']} exact matches to the aggregate. {counts['native_rank_layer_records']}/{native['expected_rank_layer_records']} rank-layer keys; {native['rank_layer_source_matches']} source comparison matches; {native['rank_layer_shape_matches']} independently recomputed shape matches. {counts['native_completed_collectives']}/{counts['native_collective_records']} completed collectives (expected {native['expected_collective_records']}); {native['tensor_descriptor_matches']}/{native['tensor_descriptors']} tensor descriptors agree with dtype bytes, shape product, numel, nbytes and stride dimensionality.",
        f"Native parallelism: {result['native_probe_scope']['parallelism']}; physical token split: {result['native_probe_scope']['tokens_per_dp_rank']}; recomputed placement: {native['placement']['status']}; rank-group matches: {native['rank_group_matches']}/{counts['native_rank_records']}.", "",
        "| Host | Global ranks | Local ranks |", "| --- | --- | --- |"])
    for host in sorted({r["host"] for r in native["placements"]}):
        ranks = [r for r in native["placements"] if r["host"] == host]
        lines.append(f"| {host} | {'/'.join(str(r['rank']) for r in ranks)} | {'/'.join(str(r['local_rank']) for r in ranks)} |")
    lines.extend(["", "native_layers.csv and native_collectives.csv preserve rank/backend, activation-scale shape/stride, schemas and bytes. They describe native MoE modules with dummy weights, not full-model/model-runner verification, numerical parity, timed native kernels or held-out accuracy. Platform/transport/source receipts are outside this statistical verdict. This report makes no cost-admission decision.", "",
        "Input SHA-256 values and unchanged-byte checks are in analysis.json, including compressed evidence. The external model-config path/hash used to reconstruct the expected call plan is recorded separately. Raw gzip evidence is streamed, never expanded on disk.", "",
        "Analysis is CPU-only and advisory. A quality rejection is logged in files and does not return a nonzero exit status or control a workload.", "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Local copied communication artifact directory")
    parser.add_argument("--output-dir", type=Path, help="Defaults to DIRECTORY/analysis")
    parser.add_argument("--queue-shift-threshold", type=float, default=0.01)
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    output = args.output_dir or directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [path for path in directory.rglob("*") if path.is_file() and not path.is_relative_to(output.resolve())]
    before = {str(path.relative_to(directory)): sha256(path) for path in inputs}
    result = analyze_result(directory, args.queue_shift_threshold)
    result["input_sha256"] = before
    result["input_bytes_unchanged"] = all(sha256(path) == before[str(path.relative_to(directory))] for path in inputs)
    result["command"] = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    for name in ("timings", "label_noise", "comparisons", "native_layers", "native_collectives"):
        write_csv(output / f"{name}.csv", result[name])
    write_csv(output / "queue_failures.csv", result["queue_failures"])
    write_csv(output / "pass_spreads.csv", result.get("pass_spreads", []))
    write_csv(output / "counts_by_pass.csv", result.get("counts_by_pass", []))
    write_csv(output / "kernel_dispatch.csv", [{key: row[key] for key in ("pass", "scenario", "call", "op", "group", "cuda_graph", "tokens_per_dp_rank", "participating_ranks", "max_input_bytes", "max_output_bytes", "communicator", "kernels")} for row in result["timings"]])
    (output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(markdown_report(result))
    print(json.dumps({"queue_gate": result["queue_gate"], "advisory_only": True, "counts": result["counts"], "report": str(output / "report.md")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
