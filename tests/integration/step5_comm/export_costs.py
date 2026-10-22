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
    "reduce_scatterv_into_output": "reduce_scatter",
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


def fit_affine(cells, *, positive_slope, supported_range, calibration_receipt):
    """NNLS on equally weighted cell medians; preserve signed observations."""
    import numpy as np
    from scipy.optimize import nnls
    fitted = [cell for cell in cells if cell["split"] == "calibration"]
    heldout = [cell for cell in cells if cell["split"] == "qualification"]
    if any(cell["quality"] != "PASS" for cell in cells):
        raise ValueError("Unqualified cells must be excluded explicitly before fitting")
    if any(not __import__("math").isfinite(c["duration_ms"]) or
           not __import__("math").isfinite(c["operand_bytes"]) for c in cells):
        raise ValueError("Calibration and qualification observations must remain finite")
    xs = np.asarray([cell["operand_bytes"] for cell in fitted],dtype=float)
    if len(set(xs)) < 3 or not heldout:
        raise ValueError("Affine fit requires three distinct anchors and independent qualification")
    scale = float(max(xs))
    matrix = np.column_stack((np.ones(len(xs)),xs/scale))
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != 2:
        raise ValueError("Affine design matrix is rank deficient")
    ys = np.asarray([cell["duration_ms"] for cell in fitted],dtype=float)
    if not np.all(np.isfinite(ys)) or not np.all(np.isfinite(xs)) or min(xs) <= 0:
        raise ValueError("Affine observations must be finite with positive operand bytes")
    coefficients,_ = nnls(matrix,ys)
    a,b = float(coefficients[0]),float(coefficients[1]/scale)
    if positive_slope and b <= 0:
        raise ValueError("Measured copy/add/TP service needs a positive identified slope")
    if not min(xs) <= supported_range[0] <= supported_range[1] <= max(xs):
        raise ValueError("Supported range may not extrapolate beyond frozen fit anchors")
    errors = [dict(cell, predicted_ms=a+b*cell["operand_bytes"],
        signed_error_ms=a+b*cell["operand_bytes"]-cell["duration_ms"],
        relative_error=None if cell["duration_ms"] == 0 else
            (a+b*cell["operand_bytes"]-cell["duration_ms"])/cell["duration_ms"]) for cell in heldout]
    if any(not supported_range[0] <= c["operand_bytes"] <= supported_range[1] for c in heldout):
        raise ValueError("Qualification input is outside the frozen supported range")
    return dict(a_ms=a,b_ms_per_byte=b,range_bytes=list(supported_range),design_matrix_rank=rank,
        design_matrix_scale_bytes=scale,distinct_sizes=len(set(xs)),time_unit="ms",
        signed_residuals_ms=(ys-matrix@coefficients).tolist(),raw_cells=cells,
        calibration_receipt=calibration_receipt,qualification=dict(status="UNVERIFIED",independent=True,
        observations=errors),candidate_only=True)


def export_finite_parameters(directory, output):
    """Export measured local fits and a fixed-simulator effective remainder candidate."""
    from dataclasses import replace
    from frontier.cc_backend.collective_plan import CollectivePlan, CollectiveTensor
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"frontier/cc_backend/backends/collective-sim/python"))
    from collective_sim_core.schema import Scenario
    from collective_sim_core.runner import run_htsim_runner
    from collective_sim_core.predictor import completed_plan_network_ms
    from collective_sim_core.intra_server_model import plan_memory_service_ms
    directory,output = Path(directory),Path(output)
    source = json.loads((directory/"collectives/plan.json").read_text())
    config = source["config"]
    if source["evidence_kind"] == "native_measurement":
        from validate_result import validate
        report=validate(json.loads((directory/"job.json").read_text()),directory)
        if report["status"] != "PASS":
            raise ValueError("Finite export requires complete selected producer evidence")
    frozen_path = Path(config["frozen_model_config"])
    frozen = json.loads(frozen_path.read_text())
    if config["frozen_model_config_sha256"] != sha256(frozen_path):
        raise ValueError("Frozen simulator configuration receipt differs from file bytes")
    scenario = Scenario.from_dict(frozen["scenario"])
    measurements = list(iter_records(measurement_path(directory)))
    kernel_rows = list(iter_records(directory/"collectives/kernels.jsonl"))
    kernels = {(r["scenario"],r["call"]):r for r in kernel_rows}
    if len(kernels) != len(kernel_rows):
        raise ValueError("Duplicate native kernel evidence")
    local = list(iter_records(directory/"collectives/local_work.jsonl"))
    shared = source["runtime"]["native_facts"]["shared"]
    evidence_kind = source["evidence_kind"]
    if evidence_kind not in ("native_measurement","cpu_fixture"):
        raise ValueError("Finite exporter requires an explicit evidence kind")
    anchors = config["grid"]["scenario_splits"]
    models,counts,excluded = [],{},[]
    local_models = {}
    layouts = {(r["dtype"],r["columns"],r["cuda_graph"],r["kind"]) for r in local}
    for dtype,columns,captured,kind in layouts:
        rank_fits = {}
        for rank in range(16):
            cells = []
            for r in local:
                if (r["dtype"],r["columns"],r["cuda_graph"],r["kind"]) != (dtype,columns,captured,kind):
                    continue
                if r["row_stride"] != columns or r["scope"] != "independent_preallocated_operand":
                    raise ValueError("Local probe layout/scope does not match the model")
                quality = all(r["per_rank_queue_held"][rank])
                if not quality:
                    excluded.append(dict(kind="local",name=r["name"],rank=rank,reason="queue_not_held"));continue
                from frontier.profiling.collectives.vllm_forward_collectives import DTYPE_BYTES
                cells.append(dict(operand_bytes=r["rows"]*r["columns"]*DTYPE_BYTES[r["dtype"]],
                    duration_ms=float(__import__("statistics").median(r["per_rank_call_us"][rank]))/1000,
                    split=r["split"],quality="PASS",cell_id=r["name"],source=dict(name=r["name"],pass_id=r["pass"])))
            cells = cell_medians(cells)
            fit = fit_affine(cells,positive_slope=True,supported_range=(min(c["operand_bytes"] for c in cells),
                max(c["operand_bytes"] for c in cells)),calibration_receipt=str(directory/"collectives/local_work.jsonl"))
            rank_fits[str(rank)] = fit
        local_models[dtype,columns,captured,kind] = rank_fits
    grouped = defaultdict(list)
    for record in measurements:
        tp_phase = {"shared_expert_down_proj_all_reduce":"shared_tp","routed_expert_all_reduce":"routed_tp"}.get(record["call"])
        if record["op"] not in ("all_gatherv","reduce_scatterv_into_output") and tp_phase is None:
            continue
        if queue_held_analysis(record,"device",.01)["queue_gate"] != "PASS":
            excluded.append(dict(kind="native",scenario=record["scenario"],call=record["call"],reason="queue_quality"));continue
        names = ("activation","weights","ids","scales") if record["op"] == "all_gatherv" else ("output",)
        context = record["native_context"]
        if context["shared"] != shared or record["evidence_kind"] != evidence_kind:
            raise ValueError("Measured native context differs from the bound producer source")
        if evidence_kind == "native_measurement" and not context.get("execution_policy"):
            raise ValueError("Measured model context requires actual execution policy receipts")
        from frontier.profiling.collectives.vllm_forward_collectives import DTYPE_BYTES
        tensors = tuple(CollectiveTensor(n,d,DTYPE_BYTES[d],c,c) for n,d,c in
                        zip(names,record["dtypes"],record["columns"]))
        phase = tp_phase or ("dispatch" if record["op"] == "all_gatherv" else "combine")
        plan = CollectivePlan(phase,tuple(record["tokens_per_dp_rank"]),tensors,
            tuple(tuple(g) for g in record["group_ranks"]),"full" if record["cuda_graph"] else "none",
            json.dumps(context,sort_keys=True),json.dumps(record["buffer_receipt"],sort_keys=True))
        from frontier.cc_backend.collective_plan import collective_model_context
        model_context=collective_model_context(context)
        grouped[phase,plan.graph_mode,plan.sizes[0]==plan.sizes[1],tuple(tensors),json.dumps(model_context,sort_keys=True)].append((record,plan))
        evidence = kernels[record["scenario"],record["call"]]
        if plan.cache_key() in counts and counts[plan.cache_key()]["kernel_counts"] != evidence["per_rank_kernel_counts"]:
            raise ValueError("Identical native plan has inconsistent paired kernel counts")
        counts[plan.cache_key()] = dict(evidence,plan_key=plan.cache_key(),context=context,
            kernel_counts=evidence["per_rank_kernel_counts"],source_receipt=source["runtime"]["native_facts"],
            evidence_kind=evidence_kind)
    for (phase,mode,uniform,tensors,context_json),rows in grouped.items():
        context = json.loads(context_json)
        copy,reduce = {},{}
        if phase in ("dispatch","combine"):
            for rank in range(16):
                copy[str(rank)] = {t.name:local_models[t.dtype,t.columns,mode=="full","copy"][str(rank)] for t in tensors}
                if phase == "combine":
                    t=tensors[0];reduce[str(rank)]=local_models[t.dtype,t.columns,mode=="full","add"][str(rank)]
        cells = []
        for record,plan in rows:
            parameters=dict(context=record["native_context"],model_identity=str(frozen_path),evidence_kind=evidence_kind,copy=copy,reduce=reduce)
            sc=replace(scenario,collective=replace(scenario.collective,plan=plan.to_dict(),local_parameters=parameters))
            native=float(__import__("statistics").median(queue_held_analysis(record,"device",.01)["critical_path_repeats_us"]))/1000
            if phase in ("shared_tp","routed_tp"):
                import statistics
                for node,node_rows in enumerate(plan.sizes):
                    samples = record["per_rank_call_us"][node*8:(node+1)*8]
                    node_ms=statistics.median(max(values) for values in zip(*samples))/1000
                    cells.append(dict(operand_bytes=node_rows*tensors[0].columns*tensors[0].element_size,
                        duration_ms=node_ms,split=anchors[record["scenario"]],quality="PASS",
                        cell_id=(record["scenario"],node),source=dict(record,node=node)))
                continue
            network=run_htsim_runner(repo_root=Path(frozen["collective_sim_root"]),spec=sc.to_runner_spec(),
                out_dir=str(output/record["scenario"]/record["call"]))
            network_ms=completed_plan_network_ms(network)
            local_ms = plan_memory_service_ms(plan.to_dict(),copy,reduce)
            cells.append(dict(operand_bytes=sum(plan.sizes)*sum(t.columns*t.element_size for t in tensors),
                duration_ms=native-network_ms-max(local_ms),split=anchors[record["scenario"]],
                quality="PASS",cell_id=record["scenario"],source=dict(scenario=record["scenario"],pass_id=record["pass"],native_ms=native,
                network_ms=network_ms,local_ms=local_ms,native_record=record)))
        cells=cell_medians(cells)
        fit=fit_affine(cells,positive_slope=phase in ("shared_tp","routed_tp"),supported_range=(min(c["operand_bytes"] for c in cells),
            max(c["operand_bytes"] for c in cells)),calibration_receipt=str(directory))
        parameters=dict(context=context,model_identity=str(frozen_path),evidence_kind=evidence_kind)
        if phase in ("shared_tp","routed_tp"):
            nvlink=dict(frozen["scenario"]["intra_server"])
            nvlink["nvlink_efficiency"]=1.75*1000/(fit["b_ms_per_byte"]*nvlink["nvlink_one_way_bw_GBps"]*1e9)
            nvlink["nvlink_allreduce_launch_overhead_us"]=fit["a_ms"]*1000/14-nvlink["nvlink_latency_us"]
            if not 0 < nvlink["nvlink_efficiency"] <= 1 or nvlink["nvlink_allreduce_launch_overhead_us"] < 0:
                raise ValueError("Measured TP8 fit gives infeasible effective bandwidth/device overhead")
            parameters["tp_nvlink"]={**nvlink,"fit":fit}
        else:
            parameters.update(copy=copy,reduce=reduce,protocol=fit)
            for error in fit["qualification"]["observations"]:
                src=error["source"];prediction=error["predicted_ms"]+src["network_ms"]+max(src["local_ms"])
                native=error["duration_ms"]+src["network_ms"]+max(src["local_ms"])
                error.update(full_plan_predicted_ms=prediction,full_plan_native_ms=native,
                    full_plan_signed_error_ms=prediction-native,full_plan_relative_error=(prediction-native)/native)
        models.append(dict(context=context,phase=phase,graph_mode=mode,uniform=uniform,tensors=[vars(t) for t in tensors],
            parameters=parameters))
    artifact=dict(schema_version="step5.collective_local.v1",device_type=frozen["device_type"],
        evidence_kind=evidence_kind,frozen_scenario=frozen["scenario"],models=models,counts=counts,excluded=excluded,
        qualification=dict(status="UNVERIFIED",independent=True,observations=[],parent_receipt=None),candidate_only=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/"collective_local.json").write_text(json.dumps(artifact,indent=2)+"\n")
    return artifact


def cell_medians(cells):
    """Equal weight per cell, retaining all passes/raw medians."""
    import statistics
    groups=defaultdict(list)
    for cell in cells: groups[cell.get("cell_id",cell["operand_bytes"]),cell["split"]].append(cell)
    return [dict(group[0],duration_ms=statistics.median(c["duration_ms"] for c in group),raw_passes=group)
             for group in groups.values()]


def qualify_finite_parameters(candidate_path, receipt_path, output_path):
    """Select frozen measured parameters only through a separate parent receipt."""
    candidate_path,receipt_path,output_path=map(Path,(candidate_path,receipt_path,output_path))
    artifact=json.loads(candidate_path.read_text());receipt=json.loads(receipt_path.read_text())
    if (artifact["evidence_kind"] != "native_measurement" or not artifact["candidate_only"] or
            receipt["schema_version"] != "step5.collective_qualification.v1" or
            receipt["candidate_sha256"] != sha256(candidate_path) or receipt["status"] != "PASS" or
            not receipt["independent"] or not receipt["observations"]):
        raise ValueError("Candidate selection needs independent native parent qualification")
    checks=receipt["model_checks"]
    if (len(checks) != len(artifact["models"]) or {row["model_index"] for row in checks} != set(range(len(artifact["models"]))) or
            any(row["status"] != "PASS" or not row["independent"] or not row["observations"] for row in checks)):
        raise ValueError("Parent qualification must cover every frozen model exactly once")
    def select(value):
        if isinstance(value,dict):
            if "a_ms" in value and "b_ms_per_byte" in value:
                if not value["qualification"]["observations"] or value["design_matrix_rank"] != 2:
                    raise ValueError("Every selected fit needs held-out evidence and identified parameters")
                value["qualification"].update(status="PASS",parent_receipt=str(receipt_path))
                value["candidate_only"]=False
            for nested in value.values(): select(nested)
        elif isinstance(value,list):
            for nested in value: select(nested)
    select(artifact)
    artifact.update(candidate_only=False,qualification={**receipt,"parent_receipt":str(receipt_path)})
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(artifact,indent=2)+"\n")
    return artifact




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Local copied run directory; no platform access")
    parser.add_argument("--output-dir", type=Path, help="Defaults to DIRECTORY/costs")
    parser.add_argument("--environment-note", default="", help="Parent-reported context, not a causal attribution")
    parser.add_argument("--finite-model",action="store_true",help="Export a candidate from frozen native/local model evidence")
    parser.add_argument("--qualification-receipt",type=Path,help="Separate parent receipt selecting DIRECTORY/collective_local.json")
    args = parser.parse_args(argv)
    if args.finite_model:
        if args.qualification_receipt:
            artifact=qualify_finite_parameters(args.directory/"collective_local.json",args.qualification_receipt,
                (args.output_dir or args.directory/"qualified")/"collective_local.json")
            print(json.dumps({"candidate_only":artifact["candidate_only"],"models":len(artifact["models"])}))
            return 0
        artifact=export_finite_parameters(args.directory,args.output_dir or args.directory/"costs")
        print(json.dumps({"candidate_only":True,"models":len(artifact["models"])}))
        return 0
    manifest = export_costs(args.directory, args.output_dir or args.directory / "costs", args.environment_note)
    print(json.dumps({key: manifest[key] for key in ("task_id", "counts", "output_directory", "advisory_only")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
