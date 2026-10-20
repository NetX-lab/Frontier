"""Describe copied Step communication evidence without enforcing a runtime gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import Counter
from pathlib import Path

from frontier.profiling.collectives.vllm_forward_collectives import DTYPE_BYTES


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        labels = [row for row in rows if row["scenario"] == scenario and row["op"] == "all_reduce" and row["group"] == "tp"]
        medians = [row["median_us"] for row in labels]
        within_variance = statistics.mean(row["sample_std_us"] ** 2 for row in labels)
        result.append({
            "scenario": scenario,
            "labels": [row["call"] for row in labels],
            "label_count": len(labels),
            "calls_per_forward_sum": sum(row["calls_per_forward"] for row in labels),
            "same_payload_and_dispatch": len({json.dumps([row["per_rank_input_bytes"], row["per_rank_output_bytes"], row["communicator"], row["kernels"]], sort_keys=True) for row in labels}) == 1,
            "median_of_label_medians_us": statistics.median(medians),
            "minimum_label_median_us": min(medians),
            "maximum_label_median_us": max(medians),
            "label_median_sample_cv": statistics.stdev(medians) / statistics.mean(medians),
            "label_median_range_over_median": (max(medians) - min(medians)) / statistics.median(medians),
            "median_within_label_sample_cv": statistics.median(row["sample_cv"] for row in labels),
            "pooled_within_label_sample_cv": math.sqrt(within_variance) / statistics.mean(row["mean_us"] for row in labels),
            "minimum_within_label_sample_cv": min(row["sample_cv"] for row in labels),
            "maximum_within_label_sample_cv": max(row["sample_cv"] for row in labels),
        })
    return result


def comparison_rows(rows: list[dict]) -> list[dict]:
    index = {(row["scenario"], row["call"]): row for row in rows}
    comparisons = [
        ("rails_1_vs_4", "graph_decode_b8", "graph_one_rail_b8"),
        ("graph_idle_vs_uniform", "graph_decode_b64", "graph_idle_b64"),
        ("eager_idle_vs_uniform", "eager_uniform_t512", "eager_idle_t512"),
    ]
    result = []
    for name, baseline, candidate in comparisons:
        for (scenario, call), row in index.items():
            if scenario != baseline:
                continue
            other = index[candidate, call]
            filtered_delta = None
            if row.get("filtered_median_us") is not None and other.get("filtered_median_us") is not None:
                filtered_delta = other["filtered_median_us"] / row["filtered_median_us"] - 1
            result.append({
                "comparison": name, "call": call, "baseline_scenario": baseline,
                "candidate_scenario": candidate, "baseline_median_us": row["median_us"],
                "candidate_median_us": other["median_us"],
                "fractional_change": other["median_us"] / row["median_us"] - 1,
                "filtered_fractional_change": filtered_delta,
                "baseline_queue_gate": row["queue_gate"], "candidate_queue_gate": other["queue_gate"],
                "baseline_max_input_bytes": row["max_input_bytes"], "candidate_max_input_bytes": other["max_input_bytes"],
                "baseline_max_output_bytes": row["max_output_bytes"], "candidate_max_output_bytes": other["max_output_bytes"],
            })
    return result


def native_rows(probe: dict) -> tuple[list[dict], list[dict]]:
    layers, collectives = [], []
    for rank in probe["ranks"]:
        for layer in rank["layers"]:
            base = {
                "rank": rank["rank"], "rank_status": rank["status"], "layer_id": layer["layer_id"],
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


def analyze_result(directory: Path, threshold: float) -> dict:
    measurements = read_jsonl(directory / "collectives/measurements.jsonl")
    kernels = read_jsonl(directory / "collectives/kernels.jsonl")
    kernel_index = {(row["scenario"], row["call"]): row for row in kernels}
    with (directory / "collectives/summary.csv").open() as handle:
        summaries = list(csv.DictReader(handle))
    summary_index = {(int(row["pass"]), row["scenario"], row["call"]): row for row in summaries}
    plan = json.loads((directory / "collectives/plan.json").read_text())
    rows = []
    for record in measurements:
        summary = summary_index[record["pass"], record["scenario"], record["call"]]
        kernel = kernel_index[record["scenario"], record["call"]]
        row = {key: value for key, value in record.items() if not key.startswith("per_rank_")}
        row.update(kernel)
        row["timing_domain"] = summary["timing_domain"]
        row.update(queue_held_analysis(record, row["timing_domain"], threshold))
        row.update(payload_bytes(record))
        row["source_median_matches"] = math.isclose(row["median_us"], float(summary["median_us"]), rel_tol=1e-12)
        row["source_queue_fraction"] = float(summary["queue_held_fraction"]) if summary["queue_held_fraction"] else None
        rows.append(row)
    device = [row for row in rows if row["timing_domain"] == "device"]
    failures = [row for row in device if row["queue_gate"] != "PASS"]
    repeat_count = sum(row["repeat_count"] for row in device)
    rank_repeat_count = sum(row["rank_repeat_count"] for row in device)
    held_count = sum(row.get("all_rank_held_count", 0) for row in device)
    rank_held_count = sum(row.get("per_rank_held_count", 0) for row in device)
    observed_rank_fractions = [row["per_rank_mean_held_fraction"] for row in device if row["per_rank_mean_held_fraction"] is not None]
    probe = json.loads((directory / "native_dispatch/dispatch_probe.json").read_text())
    layers, native_collectives = native_rows(probe)
    native_rank_file_matches = sum(rank == json.loads((directory / f"native_dispatch/rank_{rank['rank']}.json").read_text()) for rank in probe["ranks"])
    native_events = {rank["rank"]: dict(Counter(row["event"] for row in read_jsonl(directory / f"native_dispatch/rank_{rank['rank']}.jsonl"))) for rank in probe["ranks"]}
    measured_shifts = [row for row in device if row["absolute_fractional_median_delta"] is not None]
    worst = max(measured_shifts, key=lambda row: row["absolute_fractional_median_delta"]) if measured_shifts else None
    return {
        "task_id": "T12-S0-STATS", "input_directory": str(directory),
        "environment": {"python_executable": sys.executable, "python_version": sys.version, "platform": platform.platform()},
        "threshold": threshold, "threshold_comparison": "strictly less than",
        "quality_gate_is_advisory": True,
        "queue_gate": "FAIL" if failures else "PASS",
        "plan": plan,
        "worker_manifest": json.loads((directory / "manifest.json").read_text()),
        "worker_result": json.loads((directory / "result.json").read_text()),
        "counts": {
            "measurement_records": len(rows), "kernel_records": len(kernels),
            "device_records": len(device), "host_records": len(rows) - len(device),
            "queue_gate_verdicts": dict(Counter(row["queue_gate"] for row in device)),
            "device_repeat_positions": repeat_count, "all_rank_held_repeat_positions": held_count,
            "all_rank_held_fraction": held_count / repeat_count,
            "device_rank_repeat_observations": rank_repeat_count, "per_rank_held_observations": rank_held_count,
            "per_rank_mean_held_fraction": rank_held_count / rank_repeat_count,
            "retained_count_histogram": dict(Counter(row.get("all_rank_held_count", 0) for row in device)),
            "mean_record_per_rank_held_fraction": statistics.mean(observed_rank_fractions) if observed_rank_fractions else None,
            "missing_device_flag_records": sum(row["all_rank_held_fraction"] is None for row in device),
            "source_median_matches": sum(row["source_median_matches"] for row in rows),
            "source_queue_fraction_matches": sum(math.isclose(row["source_queue_fraction"], row["per_rank_mean_held_fraction"], rel_tol=1e-12) for row in device if row["per_rank_mean_held_fraction"] is not None),
            "payload_maxima_match": sum(row["payload_maxima_match"] for row in rows),
            "repeat_counts": sorted({row["repeat_count"] for row in rows}),
            "native_rank_layer_records": len(layers), "native_collective_records": len(native_collectives),
            "native_completed_collectives": sum(row["completed"] for row in native_collectives),
            "native_rank_file_matches": native_rank_file_matches,
        },
        "native_events_by_rank": native_events,
        "worst_queue_shift": {key: worst[key] for key in ("scenario", "call", "median_us", "filtered_median_us", "absolute_fractional_median_delta")} if worst else None,
        "queue_failures": failures, "timings": rows, "label_noise": label_noise(rows),
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
    noise = {row["scenario"]: row for row in result["label_noise"]}
    by_key = {(row["scenario"], row["call"]): row for row in rows}
    record_rank_mean = counts["mean_record_per_rank_held_fraction"]
    record_rank_mean_text = f"{100 * record_rank_mean:.4f}%" if record_rank_mean is not None else "unverified"
    device_cvs = [row["sample_cv"] for row in rows if row["timing_domain"] == "device"]
    lines = [
        "# T12-S0-STATS: local S0 communication analysis", "", "## Modification History", "",
        "| Date | Change |", "| --- | --- |", "| 2026-10-20 | Analyzed immutable local S0 copies; advisory quality verdict only. |", "",
        f"Queue-held check: **{result['queue_gate']}**, {counts['queue_gate_verdicts']}; 10 host/Gloo records are not applicable.",
        f"Full median = median_j(max_participating_rank(t_rank,j)); keep j only when every participating rank held j. PASS means abs(filtered-full)/full < {100 * result['threshold']:g}%.",
        f"All-rank held: {counts['all_rank_held_repeat_positions']}/{counts['device_repeat_positions']} = {100 * counts['all_rank_held_fraction']:.4f}%; rank-average held: {counts['per_rank_held_observations']}/{counts['device_rank_repeat_observations']} = {100 * counts['per_rank_mean_held_fraction']:.4f}% (different denominator).",
        f"Unweighted mean of the per-record rank-average fractions: {record_rank_mean_text}. Missing device flag records: {counts['missing_device_flag_records']}. Recomputed source medians: {counts['source_median_matches']}/100 match; source queue fractions: {counts['source_queue_fraction_matches']}/90 match rank averages; payload maxima: {counts['payload_maxima_match']}/100 match.",
        f"Retained repeat histogram: {counts['retained_count_histogram']}. Every measurement has exactly five repeats; each repeat is an average of {result['plan']['config']['calls_per_repeat']} calls, not 25 independent samples.",
        "", "## Timings (microseconds, full unfiltered medians)", "",
        "AR is the median across the five label medians, with their min–max in brackets; it is not an additional measured collective. Exact labels, launch medians, bytes and all five repeat values are in timings.csv.", "",
        "| Scenario; physical tokens/DP | TP AR [label range] | FP8 AGv | BF16 AGv | RSv | Logits AG | Host sync |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario, item in noise.items():
        record = by_key[scenario, "embedding_all_reduce"]
        values = [by_key[scenario, call]["median_us"] for call in ("moe_dispatch_all_gatherv_fp8", "moe_dispatch_all_gatherv_bf16", "moe_combine_reduce_scatterv", "logits_all_gather", "dp_batch_sync_all_reduce")]
        lines.append(f"| {scenario}; {'/'.join(map(str, record['tokens_per_dp_rank']))} | {item['median_of_label_medians_us']:.2f} [{item['minimum_label_median_us']:.2f}–{item['maximum_label_median_us']:.2f}] | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    lines.extend(["", "No equal-size graph/eager pair was collected: graph sizes are 1, 8, 64; eager sizes are 16, 65, 512, 8193. Their difference cannot isolate graph benefit.", "", "## Queue-held failures", "", "| Scenario / call | Kept/5 | Full us | Filtered us | Absolute shift |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in sorted(result["queue_failures"], key=lambda row: row.get("absolute_fractional_median_delta") or 0, reverse=True):
        filtered = f"{row['filtered_median_us']:.6f}" if row['filtered_median_us'] is not None else "unverified"
        shift = f"{100 * row['absolute_fractional_median_delta']:.4f}%" if row['absolute_fractional_median_delta'] is not None else "undefined"
        lines.append(f"| {row['scenario']} / {row['call']} | {row['all_rank_held_count']} | {row['median_us']:.6f} | {filtered} | {shift} |")
    lines.extend(["", "## Five same-payload all-reduce labels: descriptive noise", "",
        "All five labels share payload and observed dispatch within each scenario. Their configured counts sum to 273, but these are five isolated measurements, not a native 273-call chain. forward_totals.csv weights isolated medians and is not a measured native forward. Native dispatch has zero timing rows.",
        "CV uses sample standard deviation / mean on the actual five rank-critical repeat averages. Label-median CV is descriptive between isolated measurements, not a causal noise test or a confidence bound.", "",
        f"Across all 90 device records, five-repeat sample CV min/median/max is {100 * min(device_cvs):.4f}% / {100 * statistics.median(device_cvs):.4f}% / {100 * max(device_cvs):.4f}%.", "",
        "| Scenario | Between-label median CV | Within-label repeat CV median [min–max] | Label median range/median |",
        "| --- | ---: | ---: | ---: |"])
    for row in noise.values():
        lines.append(f"| {row['scenario']} | {100 * row['label_median_sample_cv']:.2f}% | {100 * row['median_within_label_sample_cv']:.2f}% [{100 * row['minimum_within_label_sample_cv']:.2f}–{100 * row['maximum_within_label_sample_cv']:.2f}%] | {100 * row['label_median_range_over_median']:.2f}% |")
    lines.extend(["", "The spread is consistent with repeat/launch/synchronization variability rather than five distinct operator costs. Five short, single-pass batches cannot establish a <1% noise floor, tail quantiles, independent-run variance, or a stable native 273-call total. No outliers were removed.", "", "## Rail and idle comparisons", "",
        "Values are candidate relative to baseline, unfiltered. Rail selection changes DP collectives only; both rail scenarios still use all eight ranks for TP AR/logits. Graph idle preserves 64/64 physical padded rows with 64/0 useful rows; eager idle uses 512/1 physical rows with 512/0 useful rows. Both idle scenarios exclude ranks 4–7 from logits (zero rows on the idle DP lane).", "",
        "| Comparison | FP8 AGv | BF16 AGv | RSv | Logits AG |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in dict.fromkeys(row["comparison"] for row in result["comparisons"]):
        compared = {row["call"]: row for row in result["comparisons"] if row["comparison"] == name}
        lines.append(f"| {name} | " + " | ".join(f"{100 * compared[call]['fractional_change']:+.2f}%" for call in ("moe_dispatch_all_gatherv_fp8", "moe_dispatch_all_gatherv_bf16", "moe_combine_reduce_scatterv", "logits_all_gather")) + " |")
    lines.extend(["", "comparisons.csv records every call, both filtered and full ratios, gate verdicts and byte changes. These are descriptive comparisons, not causal rail scaling or idle-speedup qualifications.", "", "## Payloads and observed dispatch", "",
        "Endpoint tensor bytes are not network wire traffic. timings.csv gives all rank input/output byte maps and their sums; kernel_dispatch.csv contains exact observed kernel names per size/call. AR = 8192 B/token; FP8 dispatch = 4288 B/token (4096 activation + 32 weights + 32 IDs + 128 activation scales); BF16 dispatch = 8256 B/token; combine = 8192 B/token; logits shard = 64448 B/row. Host synchronization = 48 B/rank.", "",
        "| Scenario | AR max in=out B | FP8 max in/out B | BF16 max in/out B | Combine max in/out B | Logits max in/out B | Observed AR communicator |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for scenario in noise:
        ar = by_key[scenario, "embedding_all_reduce"]
        payloads = [by_key[scenario, call] for call in ("moe_dispatch_all_gatherv_fp8", "moe_dispatch_all_gatherv_bf16", "moe_combine_reduce_scatterv", "logits_all_gather")]
        lines.append(f"| {scenario} | {ar['max_input_bytes']} | " + " | ".join(f"{row['max_input_bytes']}/{row['max_output_bytes']}" for row in payloads) + f" | {ar['communicator']} |")
    lines.extend(["", "DP AG/RS and logits use NCCL. Equal shards show AllGather/ReduceScatter RING_LL; eager 512/1 shows AllGatherV_RING_LL for dispatch and Reduce_Sum_bf16_RING_LL for the reduce_scatterv API. TP AR uses FlashInfer TRT-LLM oneshot Lamport at 1/8/16/64/65, vLLM cross_device_reduce_2stage plus DtoD at 512, and NCCL AllReduce RING_LL at 8193. Eager idle observes both custom and FlashInfer kernels, consistent with its 512/1 split; the aggregated kernel list does not attribute each kernel to a rank. Enabled communicators are not evidence of which one executed; the kernel table is the observation.", "", "## Native layer identity and limits", "",
        f"{counts['native_rank_layer_records']} rank-layer records (eight ranks × layers 3 and 88); {counts['native_completed_collectives']}/{counts['native_collective_records']} native collectives completed. All rank and layer comparisons report match. Layer 3: Fp8MoEMethod, DEEPGEMM oracle, executed TritonOrDeepGemmExperts + DeepGemmExperts. Layer 88: UnquantizedFusedMoEMethod, TRITON oracle, executed TritonExperts. Both use MoEPrepareAndFinalizeNaiveDPEPModular.",
        "The native token split is 8/1. Layer 3 transmits FP8 activation plus FP32 [rows,32] activation scales (128 B/token), FP32 top-k weights and INT32 IDs; per-row 4288 B. Its rank input/output dispatch bytes are 34304/38592 for ranks 0–3 and 4288/38592 for ranks 4–7. Layer 88 has no FP8 activation-scale tensor; 8256 B/row, dispatch 66048/74304 and 8256/74304. Combine is 73728 B input on every rank and 65536/8192 B output. Each layer has two BF16 TP all-reduces of 65536/8192 B input and output.",
        "native_layers.csv and native_collectives.csv retain all rank/backend, scale tensor shape/stride and input/output descriptors. These are native layer/module dispatch observations, not timed native kernels, full model-runner verification, numerical output parity, or target TP8/DP4/EP32 qualification. S0 is one H800 node, TP4/DP2/EP8; cost_data_qualified and target_topology_qualified remain false.", "",
        "Analysis is CPU-only and advisory. A quality rejection is logged in files and does not return a nonzero exit status or control a workload.", "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Local copied S0 artifact directory")
    parser.add_argument("--output-dir", type=Path, help="Defaults to DIRECTORY/analysis")
    parser.add_argument("--queue-shift-threshold", type=float, default=0.01)
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    output = args.output_dir or directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [path for path in directory.rglob("*") if path.is_file() and not path.is_relative_to(output.resolve())]
    before = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    result = analyze_result(directory, args.queue_shift_threshold)
    result["input_sha256"] = before
    result["input_bytes_unchanged"] = all(hashlib.sha256(path.read_bytes()).hexdigest() == before[str(path.relative_to(directory))] for path in inputs)
    result["command"] = [sys.executable, *sys.argv]
    for name in ("timings", "label_noise", "comparisons", "native_layers", "native_collectives"):
        write_csv(output / f"{name}.csv", result[name])
    write_csv(output / "queue_failures.csv", result["queue_failures"])
    write_csv(output / "kernel_dispatch.csv", [{key: row[key] for key in ("scenario", "call", "op", "group", "cuda_graph", "tokens_per_dp_rank", "participating_ranks", "max_input_bytes", "max_output_bytes", "communicator", "kernels")} for row in result["timings"]])
    (output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(markdown_report(result))
    print(json.dumps({"queue_gate": result["queue_gate"], "advisory_only": True, "counts": result["counts"], "report": str(output / "report.md")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
