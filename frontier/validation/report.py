"""Summarize unprofiled TP timing samples and their repeatability diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from frontier.validation.batch_record import read_batches, validate_rank_cohorts


def summarize_output_checks(checks, *, fixed_inputs=False):
    if not checks or any(row["output_tokens"] <= 0 for row in checks):
        raise ValueError("Output diagnostics require nonempty token counts")
    return {
        "tp_output_agreement": all(row["tp_output_agreement"] for row in checks),
        "repetitions_with_changed_outputs": sum(row["different_from_first_repeat"] > 0 for row in checks),
        "max_changed_output_fraction": max(row["different_from_first_repeat"] / row["output_tokens"]
                                           for row in checks),
        "fixed_decode_inputs": fixed_inputs,
        "note": "Finite prefill logits and cross-rank output agreement are checked. Generated output "
                "equality is a diagnostic, not numerical-accuracy validation against a reference model. "
                "With fixed inputs, output differences do not feed back into later decode inputs.",
    }


def summarize_profile_perturbation(records, *, tensor_parallel_size: int,
                                   max_absolute_change_pct: float = 5.0) -> dict:
    """Compare traced forwards to unprofiled repeats of the *same* step/shape.

    This is a rejection diagnostic, not a bias correction. Never compare an
    early traced decode to a later steady-state median or subtract the observed
    perturbation from individual operators.
    """
    if not math.isfinite(max_absolute_change_pct) or max_absolute_change_pct < 0:
        raise ValueError("Profile perturbation threshold must be finite and nonnegative")
    cohorts = validate_rank_cohorts(records, tensor_parallel_size=tensor_parallel_size)
    baseline, profiled = defaultdict(list), []
    for cohort in cohorts.values():
        r = cohort[0]
        if any(item.forward_gpu_ms is None for item in cohort):
            raise ValueError("Profile quality check requires every rank's forward GPU timing")
        key = (r.phase, r.step, len(r.request_ids), r.query_lens, r.context_lens,
               r.prefill_mask, r.graph_mode, r.capture_size, r.decode_input_sha256)
        elapsed = max(item.forward_gpu_ms for item in cohort)
        if r.profiled:
            profiled.append((r, key, elapsed))
        else:
            baseline[key].append(elapsed)
    rows = []
    for record, key, elapsed in profiled:
        values = baseline.get(key)
        if not values:
            raise ValueError(f"No matching unprofiled step/shape for {record.batch_id}")
        median = statistics.median(values)
        change = 100 * (elapsed / median - 1)
        rows.append({"batch_id": record.batch_id, "phase": record.phase,
                     "step": record.step, "baseline_repetitions": len(values),
                     "fixed_decode_inputs": record.decode_input_sha256 is not None,
                     "baseline_forward_gpu_median_ms": median,
                     "baseline_forward_gpu_cv_pct": 100 * statistics.pstdev(values) / statistics.mean(values),
                     "profiled_forward_gpu_ms": elapsed,
                     "signed_change_pct": change,
                     "passes_latency_check": abs(change) <= max_absolute_change_pct})
    return {"threshold_absolute_change_pct": max_absolute_change_pct,
            "rank_aggregation": "max per-rank forward GPU duration",
            "batches": rows, "has_profiled_samples": bool(rows),
            "all_pass_latency_check": bool(rows) and all(r["passes_latency_check"] for r in rows),
            "note": "Reject materially perturbed samples for full-model calibration. Passing is "
                    "not proof of unbiased per-operator timing. No correction factor is applied; "
                    "routing differences and run-to-run noise can also contribute."}


def summarize_capture(path: Path, *, skip_decode_steps: int = 4) -> dict:
    if skip_decode_steps < 0:
        raise ValueError("skip_decode_steps must be nonnegative")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Capture must be complete before summarizing")
    records = read_batches(path / "batches.jsonl")
    cohorts = validate_rank_cohorts(records,
                                   tensor_parallel_size=manifest["topology"]["tp"])
    groups = defaultdict(lambda: defaultdict(list))
    for cohort in cohorts.values():
        record = cohort[0]
        if record.profiled or (record.phase == "decode" and record.step <= skip_decode_steps):
            continue
        # The static runner records step 0 as prefill and step 1 as the first
        # decode with context=input_len. Preserve each input length in a sweep.
        input_len = (max(record.query_lens) if record.phase == "prefill"
                     else max(record.context_lens) - record.step + 1)
        key = (record.phase, len(record.request_ids), input_len, record.capture_size)
        if any(r.forward_gpu_ms is None or r.step_wall_ms is None for r in cohort):
            raise ValueError("Timing summary requires observed GPU and wall times on every rank")
        groups[key][record.repetition].append((max(r.forward_gpu_ms for r in cohort),
                                               max(r.step_wall_ms for r in cohort)))
    if not groups:
        raise ValueError("No timing samples remain after warmup-step exclusion")
    rows = []
    for (phase, bs, length, capture), repetitions in sorted(groups.items()):
        gpu = [statistics.median(v[0] for v in values) for values in repetitions.values()]
        wall = [statistics.median(v[1] for v in values) for values in repetitions.values()]
        rows.append({"phase": phase, "batch_size": bs, "input_len": length,
                     "capture_size": capture, "repetitions": len(repetitions),
                     "samples": sum(len(v) for v in repetitions.values()),
                     "forward_gpu_median_ms": statistics.median(gpu),
                     "forward_gpu_cv_pct": 100 * statistics.pstdev(gpu) / statistics.mean(gpu),
                     "step_wall_median_ms": statistics.median(wall),
                     "step_wall_cv_pct": 100 * statistics.pstdev(wall) / statistics.mean(wall)})
    checks = json.loads((path / "output-checks-rank0.json").read_text())
    operator_quality = None
    if (path / "operator-batches.jsonl").exists():
        operator_quality = summarize_profile_perturbation(
            [r for r in records if not r.profiled] + read_batches(path / "operator-batches.jsonl"),
            tensor_parallel_size=manifest["topology"]["tp"])
    graph_quality = None
    graph_outputs = None
    routing_quality = None
    if (path / "routing-batches.jsonl").exists():
        routing_quality = summarize_profile_perturbation(
            [r for r in records if not r.profiled] + read_batches(path / "routing-batches.jsonl"),
            tensor_parallel_size=manifest["topology"]["tp"])
    if (path / "graph-batches.jsonl").exists():
        graph_quality = summarize_profile_perturbation(
            [r for r in records if not r.profiled] + read_batches(path / "graph-batches.jsonl"),
            tensor_parallel_size=manifest["topology"]["tp"])
        graph_checks = json.loads((path / "graph-output-checks-rank0.json").read_text())
        graph_outputs = summarize_output_checks(graph_checks,
            fixed_inputs=manifest["options"].get("freeze_decode_inputs", False))
        baseline_digests = defaultdict(set)
        for row in checks:
            baseline_digests[(row["batch_size"], row["input_len"])].add(row["output_sha256"])
        graph_outputs["repetitions_matching_any_baseline_output_digest"] = sum(
            row["output_sha256"] in baseline_digests[(row["batch_size"], row["input_len"])]
            for row in graph_checks)
    return {"scope": "static TP batches; no request-serving metrics",
            "excluded_initial_decode_steps": skip_decode_steps,
            "timings": rows,
            "profile_perturbation": summarize_profile_perturbation(
                records, tensor_parallel_size=manifest["topology"]["tp"]),
            "operator_event_perturbation": operator_quality,
            "graph_event_perturbation": graph_quality,
            "routing_perturbation": routing_quality,
            "graph_output_repeatability": graph_outputs,
            "output_repeatability": summarize_output_checks(checks,
                fixed_inputs=manifest.get("options", {}).get("freeze_decode_inputs", False))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--skip-decode-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_capture(args.capture_dir, skip_decode_steps=args.skip_decode_steps)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
