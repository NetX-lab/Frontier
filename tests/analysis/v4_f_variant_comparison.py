"""Verify retained F1-F4 comparison evidence without rerunning any gate.

Usage: v4_f_variant_comparison.py --task-root TASK --stage-root STAGE --output DIR
The output directory must be new. Incomplete F2 runs are rejected; F1b coverage
failures and the rejected F3 fit remain explicit unavailable metric results.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METRICS = {"ttft_ms", "tpot_ms", "request_e2e_time_ms", "token_throughput_tps"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def entries(task, stage):
    records = []
    for command in read(task / "h3o_operator_costs/codex_resume_20261031/commands.json"):
        _, _, case, run, output = command["command"]
        label = Path(run).name
        variant = "B29" if label.endswith("b29") else label.split("_")[0].upper()
        records.append((variant, case, Path(run), Path(output)))
    root = task / "h3o_operator_costs/f_comparisons_20261101"
    records.append(("F1A", "p2_64", stage / "f_variants/runs/f1a_p2_64_codex_20261031a", root / "f1a_p2_64"))
    for case in ("p2_32", "p2_64"):
        records.append(("F2", case, stage / f"f_variants/runs/f2_{case}_k18_codex_20261101", root / f"f2_{case}"))
    return records


def verify_reading(variant, case, run, output, baseline):
    receipt = read(run / "receipt.json")
    require(receipt["status"] == "FINISHED", f"{variant}/{case}: run is incomplete")
    require(receipt["source_dirty_paths"] == [], f"{variant}/{case}: dirty source")
    reading = read(output / "reading.json")
    require(reading["formal_request_count"] == 128, f"{variant}/{case}: request population")
    require(reading["relative_error_limit"] == 0.1, f"{variant}/{case}: changed threshold")
    require(len(reading["repetitions"]) == 3, f"{variant}/{case}: missing reference")
    require(reading["routing_status"] == "MATCH", f"{variant}/{case}: routing")
    for name in ("request_id_gaps", "token_count_mismatches", "client_metric_gaps", "blocked_status"):
        require(not reading[name], f"{variant}/{case}: {name}={reading[name]}")
    rows_path = output / "rows/frontier/a10.csv"
    with rows_path.open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len({r["request_id"] for r in rows}) == 128, f"{variant}/{case}: duplicate/missing rows")
    require(all(int(r["request_num_prefill_tokens"]) == 6144 and int(r["request_num_decode_tokens"]) == 1024 for r in rows), f"{variant}/{case}: tokens")
    references = {p.name: digest(p) for p in (output / "rows/vllm").glob("*.csv")}
    require(len(references) == 3, f"{variant}/{case}: reference files")
    base_receipt, base_reading, base_references = baseline
    require(references == base_references, f"{variant}/{case}: reference bytes changed")
    require(receipt["trace"] == base_receipt["trace"], f"{variant}/{case}: trace changed")
    compute = {Path(t["path"]).name: t["sha256"] for t in receipt["profiling_tables"] if "/compute/" in t["path"]}
    base_compute = {Path(t["path"]).name: t["sha256"] for t in base_receipt["profiling_tables"] if "/compute/" in t["path"]}
    require(compute == base_compute and len(compute) == 6, f"{variant}/{case}: compute inputs")
    require(not any(receipt["a1"]["enable_dummy_mode"].values()), f"{variant}/{case}: dummy enabled")
    require(not any(receipt["a1"]["step_cost_table_bound_by_predictor"].values()), f"{variant}/{case}: step table bound")
    require(receipt["a11"]["stale_stage_end_events"] == 0, f"{variant}/{case}: stale events")
    metrics = []
    groups = {"pooled": reading["metrics"], **{k: v["metrics"] for k, v in reading["repetitions"].items()}}
    base_groups = {"pooled": base_reading["metrics"], **{k: v["metrics"] for k, v in base_reading["repetitions"].items()}}
    for repeat, values in groups.items():
        require({v["metric"] for v in values} == METRICS and len(values) == 4, f"{variant}/{case}: metric set")
        base_values = {v["metric"]: v for v in base_groups[repeat]}
        for value in values:
            native, predicted = value["vllm"], value["frontier"]
            require(math.isfinite(native) and math.isfinite(predicted) and native > 0, f"{variant}/{case}: nonpositive reference or nonfinite metric")
            absolute = abs(predicted - native)
            relative = absolute / abs(native)
            require(native == base_values[value["metric"]]["vllm"], f"{variant}/{case}: native value changed")
            require(math.isclose(relative, value["relative_error"], rel_tol=1e-12, abs_tol=1e-12), f"{variant}/{case}: error mismatch")
            require(value["status"] == ("PASS" if relative <= 0.1 else "FAIL"), f"{variant}/{case}: verdict mismatch")
            require(value["sample_count"] == value["eligible_count"] == 128 and value["excluded_count"] == 0, f"{variant}/{case}: excluded requests")
            metrics.append(dict(variant=variant, case=case, repeat=repeat, metric=value["metric"],
                                native=native, predicted=predicted, delta=predicted-native,
                                absolute_error=absolute, relative_error=relative, status=value["status"],
                                b29_prediction=base_values[value["metric"]]["frontier"]))
    expected = "FAIL" if any(m["status"] == "FAIL" for m in metrics) else "PASS"
    require(reading["verdict"] == expected, f"{variant}/{case}: overall verdict")
    config = read(Path(receipt["metrics_dir"]) / "config.json")
    if variant == "F4":
        # The retained B28/F4 generation predates the B29 delay fields.
        ingress = config.get("request_ingress_delay_ms", 0.0)
        handoff = config["kv_cache_transfer_config"].get("request_handoff_delay_ms", 0.0)
        require(ingress == handoff == 0.0, f"F4/{case}: expected B28 zero delays")
    else:
        ingress = config["request_ingress_delay_ms"]
        handoff = config["kv_cache_transfer_config"]["request_handoff_delay_ms"]
    result = dict(variant=variant, case=case, run=str(run), reading=str(output / "reading.json"),
                  source_commit=receipt["source_commit"], source_dirty_paths=receipt["source_dirty_paths"],
                  reading_sha256=digest(output / "reading.json"), rows_sha256=digest(rows_path),
                  reference_sha256=references, compute_sha256=compute, verdict=expected,
                  init_wall_s=receipt["init_wall_s"], run_wall_s=receipt["run_wall_s"],
                  K=config["cluster_config"]["replica_scheduler_config"]["max_forwards_in_flight"],
                  ingress_ms=ingress, handoff_ms=handoff,
                  cc_backend=config["cluster_config"]["cc_backend_config"]["name"],
                  kv_admission=receipt["kv_admission"], a11=receipt["a11"], metrics=metrics)
    return result


def baselines(records):
    return {case: (read(run / "receipt.json"), read(output / "reading.json"),
                   {p.name: digest(p) for p in (output / "rows/vllm").glob("*.csv")})
            for variant, case, run, output in records if variant == "B29"}


def verify_f2(task, result):
    root = task / "h3o_operator_costs"
    case = result["case"]
    command = read(root / f"b30b/f2_replays_20261101/{case}_command.json")
    watch = read(root / "f_comparisons_20261101/watcher.json")["cases"][f"f2_{case}"]
    require(command.get("exit_code") == 0, f"F2/{case}: no successful launcher terminal")
    require(watch["status"] == "READING_COMPLETE" and watch["exit_code"] == 0, f"F2/{case}: gate execution")
    require(result["K"] == 18 and result["ingress_ms"] == 10.3 and result["handoff_ms"] == 11.4, f"F2/{case}: effective fields")
    require(result["cc_backend"] == "measured", f"F2/{case}: backend")
    comparison = read(root / f"b30b/f2_replays_20261101/{case}_input_comparison.json")
    require(comparison["table_hashes_equal"] and comparison["trace_equal"], f"F2/{case}: inputs differ")
    behavior = [d for d in comparison["differences"] if not d["path"].endswith(("/cache_dir", "/output_dir"))]
    require(behavior == [{"path": "/cluster_config/replica_scheduler_config/max_forwards_in_flight", "baseline": 19, "candidate": 18}], f"F2/{case}: extra behavior changes")
    result["launcher"] = command
    result["gate_execution"] = watch


def unavailable(task):
    root = task / "h3o_operator_costs/b30b"
    failed = []
    for case in ("p2_32", "p2_64"):
        path = root / f"f1b_replays_20261101/{case}_terminal_receipt.json"
        receipt = read(path)
        require(receipt["status"] == "FAILED", f"F1B/{case}: missing failure")
        failed.append(dict(variant="F1B", case=case, status="COVERAGE_MISS", metrics=None,
                           reason=receipt["error"], receipt=str(path), sha256=digest(path)))
    path = root / "f3_nixl02_20261101/fit_diagnosis.json"
    diagnosis = read(path)
    require(diagnosis["classification"] == "DIAGNOSTIC_MODEL_REJECTED", "F3: missing rejection")
    failed.append(dict(variant="F3", case="p2_32/p2_64", status="INVALID_FIT_NO_REPLAY", metrics=None,
                       fit=diagnosis["unconstrained_fit"], source=str(path), sha256=digest(path)))
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a new directory")
    records = entries(args.task_root, args.stage_root)
    baseline = baselines(records)
    results = []
    for variant, case, run, output in records:
        result = verify_reading(variant, case, run, output, baseline[case])
        if variant == "F2":
            verify_f2(args.task_root, result)
        results.append(result)
    failed = unavailable(args.task_root)
    rows = [row for result in results for row in result["metrics"]]
    require(len(results) == 8 and len(rows) == 128, "Incomplete comparison matrix")
    args.output.mkdir(parents=True, exist_ok=False)
    document = dict(status="PASS_EVIDENCE_CHECKS", classification="DIAGNOSTIC_COMPARISON", formal_qualification=False,
                    results=results, unavailable=failed,
                    limits="No variant promotion. F4 uses B28 zero delays. A6, memory and NIXL qualifications remain open.")
    (args.output / "comparison.json").write_text(json.dumps(document, indent=2) + "\n")
    with (args.output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(document["status"], f"readings={len(results)} metrics={len(rows)} unavailable={len(failed)}")


if __name__ == "__main__":
    main()
