"""Export local Step collective measurements as unqualified candidate sidecars.

Scalar API aliases describe endpoint byte semantics only. Neither this exporter
nor its CSVs change a predictor, admit training data, or control a workload.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

from analyze_result import payload_bytes, queue_held_analysis, timing_stats, write_csv


API_OPERATIONS = {
    "all_reduce": "allreduce",
    "device_group_all_reduce": "allreduce",
    "all_gather": "allgather",
    "all_gatherv": "allgather",
    "reduce_scatter": "reduce_scatter",
    "reduce_scatterv": "reduce_scatter",
    "cpu_all_reduce": None,
}
REPEAT_SEMANTICS = (
    "Each repeat is a critical-rank batch-average per call; repeats, labels and "
    "passes are descriptive observations, not independent native forward calls."
)
CELL_FIELDS = (
    "scenario", "op", "group", "group_size", "cuda_graph", "timing_domain",
    "backend", "kernels", "dtypes", "columns", "row_bytes", "shard_rows",
    "group_ranks", "participating_ranks", "active_dp_rails", "concurrent_dp_rails",
    "tokens_per_dp_rank", "useful_tokens_per_dp_rank", "padded_tokens_per_dp_rank",
    "per_rank_input_bytes", "per_rank_output_bytes", "world_size", "devices_per_node",
    "calls_per_repeat", "expected_repeat_count", "expected_pass_count",
)


def measurement_path(directory: Path) -> Path:
    path = directory / "collectives/measurements.jsonl"
    return path if path.exists() else path.with_suffix(".jsonl.gz")


def iter_records(path: Path):
    """Stream JSONL without retaining the world-rank timing matrices."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def milliseconds(stats: dict) -> dict:
    result = {}
    for key, value in stats.items():
        if key.endswith("_us"):
            key = key[:-3] + "_ms"
            if isinstance(value, list):
                value = [v / 1000 for v in value]
            elif value is not None:
                value /= 1000
        result[key] = value
    return result


def describe_ms(values: list[float]) -> dict:
    # timing_stats performs unit-neutral arithmetic; these inputs are already ms.
    return {key[:-3] + "_ms" if key.endswith("_us") else key: value
            for key, value in timing_stats(values).items()}


def api_arguments(row: dict) -> dict:
    operation = API_OPERATIONS[row["op"]]
    inputs, outputs = row["per_rank_input_bytes"], row["per_rank_output_bytes"]
    useful = row["useful_tokens_per_dp_rank"]
    uniform = (len(set(inputs.values())) == len(set(outputs.values())) == 1
               and len(set(row["tokens_per_dp_rank"])) == 1
               and len(set(useful)) == 1 and min(useful) > 0
               and row["group_size_matches"])
    representable = operation is not None and uniform
    return {
        "api_operation": operation,
        "comm_domain": row["group"].upper() if operation is not None else None,
        "num_devices": row["group_size"],
        "data_size_bytes": next(iter(inputs.values())) if representable else None,
        "scalar_api_representable": representable,
        "scalar_api_limit": ("host/Gloo has no production collective API" if operation is None
                             else "uniform endpoint payload only; graph/backend/rails are not API dimensions"
                             if representable else "uneven/idle rank or group vectors require the sidecar"),
    }


def observation(record: dict, summary: dict, kernel: dict, config: dict,
                queue_shift_threshold: float) -> dict:
    row = {key: value for key, value in record.items() if not key.startswith("per_rank_")}
    # Fixed-size reduce_scatter has the same full-input endpoint as uniform RSv.
    endpoint_record = ({**record, "op": "reduce_scatterv"}
                       if record["op"] == "reduce_scatter" else record)
    row.update(payload_bytes(endpoint_record))
    row.update(backend=kernel["communicator"], communicator=kernel["communicator"],
               kernels=kernel["kernels"], timing_domain=summary["timing_domain"])
    row.update(milliseconds(queue_held_analysis(record, row["timing_domain"], queue_shift_threshold)))
    tp = config["engine"]["tensor_parallel_size"]
    row.update(
        active_dp_rails=sorted({group[0] % tp for group in row["group_ranks"]})
        if row["group"] == "dp" else [],
        world_size=tp * config["engine"]["data_parallel_size"],
        devices_per_node=config["gpus_per_node"],
        calls_per_repeat=config["calls_per_repeat"],
        expected_repeat_count=config["repeats"], expected_pass_count=config["passes"],
        group_size_matches=all(len(group) == row["group_size"] for group in row["group_ranks"]),
        time_unit="ms", repeat_semantics=REPEAT_SEMANTICS,
        candidate_only=True, cost_data_qualified=False, training_selected=False,
        production_consumable=False, native_forward_timing_observed=False,
    )
    row["group_payload_vectors"] = [
        {"ranks": group, "input_bytes": [row["per_rank_input_bytes"][rank] for rank in group],
         "output_bytes": [row["per_rank_output_bytes"][rank] for rank in group]}
        for group in row["group_ranks"]
    ]
    row.update(api_arguments(row))
    row["source_median_matches"] = math.isclose(row["median_ms"], float(summary["median_us"]) / 1000,
                                               rel_tol=1e-12)
    row["repeat_count_matches"] = row["repeat_count"] == config["repeats"]
    row["positive_finite_repeats"] = all(math.isfinite(v) and v > 0
                                         for v in row["critical_path_repeats_ms"])
    row["filtered_median_shift_ms"] = (row["filtered_median_ms"] - row["median_ms"]
                                       if row.get("filtered_median_ms") is not None else None)
    checks = ("payload_maxima_match", "source_median_matches", "repeat_count_matches",
              "group_size_matches", "positive_finite_repeats")
    row["quality_fail_reasons"] = [key for key in checks if not row[key]]
    if row["queue_gate"] == "FAIL":
        row["quality_fail_reasons"].append("queue_gate")
    row["observation_quality"] = "FAIL" if row["quality_fail_reasons"] else row["queue_gate"]
    return row


def cell_key(row: dict) -> str:
    context = {key: row[key] for key in CELL_FIELDS}
    # Only same-context TP all-reduce labels may share a descriptive curve.
    if row["op"] != "all_reduce" or row["group"] != "tp":
        context["call"] = row["call"]
    return json.dumps(context, sort_keys=True)


def pass_statistics(rows: list[dict]) -> dict:
    repeats = [value for row in rows for value in row["critical_path_repeats_ms"]]
    return {
        **describe_ms(repeats), "observation_count": len(rows),
        "queue_gate_counts": dict(Counter(row["queue_gate"] for row in rows)),
        "quality_fail_count": sum(row["observation_quality"] == "FAIL" for row in rows),
        "quality_unverified_count": sum(row["observation_quality"] == "UNVERIFIED" for row in rows),
    }


def aggregate_curves(observations: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for row in observations:
        cells[cell_key(row)].append(row)
    curves = []
    for key, rows in sorted(cells.items()):
        curve = json.loads(key)
        curve.pop("call", None)
        by_pass = defaultdict(list)
        counts = {}
        for row in rows:
            by_pass[row["pass"]].append(row)
            if row["call"] in counts and counts[row["call"]] != row["calls_per_forward"]:
                raise ValueError(f"Conflicting calls_per_forward for {row['scenario']}/{row['call']}")
            counts[row["call"]] = row["calls_per_forward"]
        per_pass = [{"pass": pass_id, **pass_statistics(items),
                     "labels": [{key: row[key] for key in (
                         "call", "calls_per_forward", "median_ms", "sample_cv", "min_ms", "max_ms",
                         "range_over_median", "queue_gate", "observation_quality")}
                                for row in sorted(items, key=lambda row: row["call"])]}
                    for pass_id, items in sorted(by_pass.items())]
        curve.update({key: rows[0][key] for key in (
            "api_operation", "comm_domain", "num_devices", "data_size_bytes", "scalar_api_representable",
            "scalar_api_limit", "time_unit", "repeat_semantics", "candidate_only", "cost_data_qualified",
            "training_selected", "production_consumable", "native_forward_timing_observed",
            "group_payload_vectors", "max_input_bytes", "max_output_bytes",
        )})
        curve.update(pass_statistics(rows))
        curve.update(
            call_labels=sorted(counts), calls_per_forward_by_label=dict(sorted(counts.items())),
            calls_per_forward=sum(counts.values()), label_count=len(counts),
            pass_count=len(by_pass), per_pass=per_pass,
            passes_complete=all(
                sorted(row["pass"] for row in rows if row["call"] == label)
                == list(range(curve["expected_pass_count"]))
                for label in counts),
            per_label_pass_counts={label: len({row["pass"] for row in rows if row["call"] == label})
                                   for label in sorted(counts)},
            pass_median_statistics=describe_ms([item["median_ms"] for item in per_pass]),
            pooled_statistics_semantics="unfiltered pooled critical-rank batch-averages; descriptive only",
            pooled_repeats_include_failed_observations=curve["quality_fail_count"] > 0,
            outliers_removed=0,
        )
        curve["observation_quality"] = ("FAIL" if curve["quality_fail_count"] else "UNVERIFIED"
                                        if curve["quality_unverified_count"] else
                                        "NOT_APPLICABLE" if curve["timing_domain"] == "host" else "PASS")
        curves.append(curve)
    return curves


def export_costs(directory: Path, output: Path, environment_note: str = "",
                  queue_shift_threshold: float = 0.01) -> dict:
    directory, output = directory.resolve(), output.resolve()
    if output == directory:
        raise ValueError("Cost output must not overwrite the source worker manifest")
    plan_path = directory / "collectives/plan.json"
    config = json.loads(plan_path.read_text())["config"]
    kernels_path = directory / "collectives/kernels.jsonl"
    kernels = {(row["scenario"], row["call"]): row for row in iter_records(kernels_path)}
    summary_path = directory / "collectives/summary.csv"
    with summary_path.open() as handle:
        summaries = {(int(row["pass"]), row["scenario"], row["call"]): row
                     for row in csv.DictReader(handle)}
    measurements = measurement_path(directory)
    observations = [observation(record, summaries[record["pass"], record["scenario"], record["call"]],
                                kernels[record["scenario"], record["call"]], config, queue_shift_threshold)
                    for record in iter_records(measurements)]
    curves = aggregate_curves(observations)
    input_paths = [plan_path, measurements, kernels_path, summary_path]
    worker_manifest = directory / "manifest.json"
    worker = json.loads(worker_manifest.read_text())
    input_paths.append(worker_manifest)
    manifest = {
        "task_id": "T12-COST-EXPORT", "input_directory": str(directory), "output_directory": str(output),
        "candidate_only": True, "cost_data_qualified": False, "training_selected": False,
        "production_consumable": False, "advisory_only": True,
        "environment": {"python_executable": sys.executable, "python_version": sys.version,
                        "platform": platform.platform()},
        "environment_note": environment_note,
        "environment_note_provenance": "parent/user reported; not independently verified; causal effect unknown",
        "worker_identity": {"job": worker["job"], "torch": worker.get("torch"),
                            "vllm": worker.get("vllm"), "python": worker.get("python")},
        "input_sha256": {str(path.relative_to(directory)): sha256(path) for path in input_paths},
        "exporter_sha256": sha256(Path(__file__)),
        "analysis_source_sha256": sha256(Path(sys.modules[queue_held_analysis.__module__].__file__)),
        "queue_shift_threshold": queue_shift_threshold, "queue_threshold_comparison": "strictly less than",
        "counts": {"observations": len(observations), "curves": len(curves),
                   "observation_quality": dict(Counter(row["observation_quality"] for row in observations)),
                   "queue_gates": dict(Counter(row["queue_gate"] for row in observations)),
                   "curve_quality": dict(Counter(row["observation_quality"] for row in curves)),
                   "scalar_api_representable_observations": sum(row["scalar_api_representable"] for row in observations)},
        "limits": [
            "Endpoint tensor bytes are not algorithm-dependent wire traffic; time_unit is ms.",
            REPEAT_SEMANTICS,
            "TP label multiplicities count unique labels once per forward, never once per pass; no native forward timing is inferred.",
            "No observations or outliers are removed; failed rows remain marked and are never selected for training.",
            "Scalar API representability is only operation/byte/cardinality mapping, not admission; graph, observed backend, concurrent rails and vectors are not production API dimensions.",
            "Host/Gloo has no production API mapping and is never pooled with device costs.",
            "Kernel/backend labels come from scenario/call kernels.jsonl, not independent dispatch attribution for every timed pass, repeat or rank.",
            "Legacy Vidur cannot consume graph/backend/rail dimensions; devices_per_node == world_size filtering excludes a real 8-GPU/node world-size-32 run.",
            "Legacy Vidur lookup uses max(1, data_size_bytes // 2); its precomputed keys stop at 100000, so this sidecar is not a compatible cost-loader replacement.",
            "Passes share the source job and startup; pooled CV and min/max are descriptive, not confidence bounds or independent-run variance.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "observations.csv", observations)
    write_csv(output / "curves.csv", curves)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Local copied run directory; no platform access")
    parser.add_argument("--output-dir", type=Path, help="Defaults to DIRECTORY/costs")
    parser.add_argument("--environment-note", default="", help="Parent-reported context, not a causal attribution")
    args = parser.parse_args(argv)
    manifest = export_costs(args.directory, args.output_dir or args.directory / "costs", args.environment_note)
    print(json.dumps({key: manifest[key] for key in ("task_id", "counts", "output_directory", "advisory_only")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
