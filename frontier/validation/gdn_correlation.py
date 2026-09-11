"""Calibrate GDN on selected batch shapes and validate disjoint held-out shapes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path

import pandas as pd

from frontier.gdn.predictor import GDNBatchFeatures, ProfiledGDNPredictor
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.gdn.sglang_trace import build_sglang_profile_row_from_samples
from frontier.types import MeasurementType
from frontier.validation.batch_record import read_batches
from frontier.validation.trace import extract_batch_kernels, summarize_batch_kernels


def correlate(capture_dir: Path, *, model: str, device: str,
              calibration_sizes: set[int], validation_sizes: set[int]) -> tuple:
    if not calibration_sizes or not validation_sizes or calibration_sizes & validation_sizes:
        raise ValueError("Calibration and validation batch sizes must be nonempty and disjoint")
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Capture must be complete before correlation")
    config = ModelConfig.from_model_name(model)
    if Path(manifest["model_path"]).name != Path(model).name:
        raise ValueError("Model does not match captured checkpoint")
    tp = manifest["topology"]["tp"]
    records = {(r.batch_id, r.rank): r for r in read_batches(capture_dir / "batches.jsonl")}
    summaries, training_rows, validation = [], [], []
    training_samples = {}
    signature = json.dumps(manifest["versions"], sort_keys=True)
    seen_shapes = set()
    for path in sorted(capture_dir.glob("*-rank0.trace.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            kernels_by_batch = extract_batch_kernels(json.load(stream))
        for batch_id, kernels in kernels_by_batch.items():
            record = records[(batch_id, 0)]
            size = len(record.request_ids)
            if not record.profiled:
                raise ValueError("Kernel traces must identify diagnostic/profiled passes")
            summary = summarize_batch_kernels(kernels, phase=record.phase,
                                              gdn_layers=config.get_num_gdn_layers())
            summaries.append({"batch_id": batch_id, "rank": 0, **summary})
            if size not in calibration_sizes | validation_sizes:
                continue
            seen_shapes.add((size, record.phase))
            if size in validation_sizes:
                validation.append((record, summary))
                continue
            if len(set(record.query_lens)) != 1:
                raise ValueError("Initial GDN calibration requires uniform query lengths")
            if record.phase == "prefill" and any(record.context_lens):
                raise ValueError("Initial GDN calibration requires cold prefill")
            physical_size = record.capture_size or size
            # Decode context length is not a GDN predictor feature. Pool repeated
            # layer samples before computing statistics rather than writing
            # conflicting rows for the same exact predictor feature vector.
            key = (record.phase, physical_size,
                   record.query_lens[0] if record.phase == "prefill" else 1)
            if key not in training_samples:
                training_samples[key] = (record, str(path),
                                         {name: [] for name in summary["gdn_samples_ms"]})
            for name, values in summary["gdn_samples_ms"].items():
                training_samples[key][2][name].extend(values)
    expected = {(size, phase) for size in calibration_sizes | validation_sizes
                for phase in ("prefill", "decode")}
    if missing := expected - seen_shapes:
        raise ValueError(f"Missing complete profiled batches: {sorted(missing)}")
    for (phase, physical_size, _), (record, path, samples) in training_samples.items():
        training_rows.append(build_sglang_profile_row_from_samples(
            samples, shape={"phase": phase, "batch_size": physical_size,
                            "input_len": record.query_lens[0] if phase == "prefill"
                            else max(record.context_lens)},
            trace_path=path, model_config=config, device=device,
            tensor_parallel_size=tp, runtime_stack_signature=signature,
        ))
    frame = pd.json_normalize(training_rows)
    predictor = ProfiledGDNPredictor(frame, model_config=config, device=device,
                                    tensor_parallel_size=tp,
                                    measurement_type=MeasurementType.KERNEL_ONLY)
    errors = []
    for record, summary in validation:
        features = GDNBatchFeatures.from_batch(record.to_frontier_batch(is_moe=config.is_moe))
        # Reject graph-padding leakage: logical held-out sizes must also produce
        # physical feature vectors absent from the training set.
        if any(features.exact_key() == (
            int(row["batch_size"]), int(row["batch_num_tokens"]),
            int(row["batch_num_prefill_tokens"]), int(row["batch_num_decode_tokens"]),
            int(row["max_query_len"]), bool(row["has_initial_state"])
        ) for row in training_rows):
            raise ValueError("Held-out shape maps to an already calibrated physical graph shape")
        predictions = predictor.predict_operator_times(features)
        predicted = sum(predictions.values()) * config.get_num_gdn_layers()
        observed = summary["gdn_kernel_sum_ms"]
        errors.append({"batch_id": record.batch_id, "phase": record.phase,
                       "batch_size": len(record.request_ids),
                       "observed_gdn_ms": observed, "predicted_gdn_ms": predicted,
                       "absolute_error_pct": 100 * abs(predicted / observed - 1)})
    report = {
        "scope": "rank-0 GDN kernel sums only; excludes attention, MoE, collectives and serving overhead",
        "measurement_note": "Calibration and validation both use profiled kernel sums. Their error "
                            "does not establish unprofiled decode latency accuracy. Check matching-step "
                            "profile perturbation with frontier.validation.report before broader use.",
        "measurement_type": "KERNEL_ONLY", "tensor_parallel_size": tp,
        "calibration_sizes": sorted(calibration_sizes),
        "validation_sizes": sorted(validation_sizes),
        "by_phase": {phase: {
            "batches": len(rows),
            "observed_median_ms": statistics.median(r["observed_gdn_ms"] for r in rows),
            "predicted_median_ms": statistics.median(r["predicted_gdn_ms"] for r in rows),
            "mape_pct": statistics.mean(r["absolute_error_pct"] for r in rows),
        } for phase in ("prefill", "decode")
          if (rows := [r for r in errors if r["phase"] == phase])},
    }
    return frame, summaries, errors, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="mi355x")
    parser.add_argument("--calibration-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--validation-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame, summaries, rows, report = correlate(
        args.capture_dir, model=args.model, device=args.device,
        calibration_sizes=set(args.calibration_sizes), validation_sizes=set(args.validation_sizes))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / "gdn_kernel_only.csv", index=False)
    (args.output_dir / "kernel-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output_dir / "correlation.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
