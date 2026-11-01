"""Collect retained B29/pass1/pass2 P2 evidence; never run a replay or gate.

Usage: v4_attention_pass_comparison.py --task-root TASK --output NEW_JSON
Each new replay needs its terminal *_command.json, including its nested gate receipt.
Numeric FAIL results are retained; this comparison cannot qualify formal inputs.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from v4_f_variant_comparison import METRICS, digest, read, require, verify_reading

CASES = ("p2_32", "p2_64")
GENERATIONS = ("B29", "pass1_all", "pass2_all")
ATTENTION = "attention_graph_replay_kernel_only.csv"


def table_hashes(tables):
    hashes = {Path(t["path"]).name: t["sha256"] for t in tables}
    require(len(hashes) == len(tables) == 7 and ATTENTION in hashes, "Profiling table set changed")
    return hashes


def metric_rows(generation, case, reading, baseline):
    require(reading["formal_request_count"] == 128 and reading["relative_error_limit"] == 0.1, "Population/limit changed")
    require(all(reading[k] == baseline[k] for k in ("gate", "ttft_semantic", "cell_id")), "Metric semantics changed")
    require(len(reading["repetitions"]) == 3 and reading["repetitions"].keys() == baseline["repetitions"].keys(), "Native repeats changed")
    require(reading["routing_status"] == "MATCH", "Routing mismatch")
    for name in ("request_id_gaps", "token_count_mismatches", "client_metric_gaps", "blocked_status"):
        require(not reading[name], f"{generation}/{case}: {name}")
    groups = {"pooled": reading, **reading["repetitions"]}
    bases = {"pooled": baseline, **baseline["repetitions"]}
    predictions, rows = {}, []
    for repeat, group in groups.items():
        require(len(group["metrics"]) == 4 and {v["metric"] for v in group["metrics"]} == METRICS, "Metric set changed")
        native_values = {v["metric"]: v["vllm"] for v in bases[repeat]["metrics"]}
        statuses = []
        for value in group["metrics"]:
            metric, native, predicted = value["metric"], value["vllm"], value["frontier"]
            require(math.isfinite(native) and math.isfinite(predicted) and native > 0 and predicted > 0, "Invalid metric")
            require(native == native_values[metric], "Native value changed")
            require(predicted == predictions.setdefault(metric, predicted), "Prediction changed across native repeats")
            delta, absolute = predicted - native, abs(predicted - native)
            relative, status = absolute / native, "PASS" if absolute / native <= 0.1 else "FAIL"
            require(math.isclose(delta, value["absolute_error"], rel_tol=1e-12, abs_tol=1e-12), "Signed source error mismatch")
            require(math.isclose(relative, value["relative_error"], rel_tol=1e-12, abs_tol=1e-12) and value["status"] == status, "Error/verdict mismatch")
            require(value["sample_count"] == value["eligible_count"] == 128 and value["excluded_count"] == 0, "Excluded requests")
            rows.append(dict(generation=generation, case=case, repeat=repeat, metric=metric,
                             unit="tokens/s" if metric == "token_throughput_tps" else "ms", native=native,
                             predicted=predicted, signed_delta=delta, absolute_error=absolute,
                             source_absolute_error=value["absolute_error"], relative_error=relative, status=status))
            statuses.append(status)
        if repeat != "pooled":
            require(group["verdict"] == ("FAIL" if "FAIL" in statuses else "PASS"), "Repeat verdict mismatch")
    require(reading["verdict"] == ("FAIL" if any(r["status"] == "FAIL" for r in rows) else "PASS"), "Overall verdict mismatch")
    return rows


def summaries(rows):
    spreads, variability = [], []
    for case in CASES:
        for metric in sorted(METRICS):
            selected = [r for r in rows if r["case"] == case and r["metric"] == metric]
            pooled = {r["generation"]: r for r in selected if r["repeat"] == "pooled"}
            values = {g: pooled[g]["predicted"] for g in GENERATIONS}
            spread = max(values.values()) - min(values.values())
            b29, native = pooled["B29"]["predicted"], pooled["B29"]["native"]
            spreads.append(dict(case=case, metric=metric, unit=pooled["B29"]["unit"], predictions=values,
                                minimum=min(values.values()), maximum=max(values.values()), spread=spread,
                                b29_prediction=b29, native_pooled=native,
                                spread_over_b29_prediction=spread / b29, spread_over_native_pooled=spread / native))
            repeats = {r["repeat"]: r["native"] for r in selected if r["generation"] == "B29" and r["repeat"] != "pooled"}
            require(len(repeats) == 3, "Missing native variability values")
            numbers = list(repeats.values())
            sd = statistics.stdev(numbers)
            variability.append(dict(case=case, metric=metric, unit=pooled["B29"]["unit"], values=repeats,
                                    minimum=min(numbers), maximum=max(numbers), range=max(numbers) - min(numbers),
                                    sample_sd=sd, sample_cv=sd / statistics.mean(numbers), pooled=native))
    return spreads, variability


def collect(task):
    root = task / "h3o_operator_costs/a6_pass_replays_20261101"
    sources = [root / name for name in ("baseline.json", "preflight.json", "tables_manifest.json", "rows_manifest.json")]
    baseline, preflight, tables, _ = map(read, sources)
    require(preflight["status"] == "PASS" and tables["independent_pass_complete"], "Pass integrity incomplete")
    bindings = {(b["pass"], b["case"]): b for b in preflight["bindings"]}
    require(set(bindings) == {(g, c) for g in GENERATIONS[1:] for c in CASES}, "Expected four pass bindings")
    terminals = {}
    for generation, case in bindings:
        command_path = root / f"{generation}_{case}_command.json"
        command = read(command_path)
        require(command.get("status") == "FINISHED" and command.get("exit_code") == 0 and command.get("finished_utc"), f"{generation}/{case}: run command incomplete")
        gate = command["gate"]
        require(gate.get("status") == "FINISHED" and gate.get("exit_code") == 0 and gate.get("finished_utc"), f"{generation}/{case}: once-run gate incomplete")
        require(gate["command"][2:] == [case, command["run_root"], command["gate_root"]], "Gate command binding changed")
        terminals[generation, case] = command, gate
        sources.append(command_path)
    commands_path = task / "h3o_operator_costs/codex_resume_20261031/commands.json"
    comparison_path = task / "h3o_operator_costs/f_comparisons_20261101/complete/comparison.json"
    sources.extend((commands_path, comparison_path))
    comparison = read(comparison_path)
    require(comparison["status"] == "PASS_EVIDENCE_CHECKS", "B29 comparison incomplete")
    results, rows = [], []
    for case in CASES:
        base = baseline["runs"][case]
        base_run = Path(base["receipt"]).parent
        require(digest(Path(base["receipt"])) == base["receipt_sha256"], "B29 receipt changed")
        gate = next(c for c in read(commands_path) if c["command"][2:4] == [case, str(base_run)])
        require(gate["exit_code"] == 0 and gate["finished_utc"], "B29 gate incomplete")
        base_output = Path(gate["command"][4])
        base_receipt, base_reading = read(Path(base["receipt"])), read(base_output / "reading.json")
        references = {p.name: digest(p) for p in (base_output / "rows/vllm").glob("*.csv")}
        verified = verify_reading("B29", case, base_run, base_output, (base_receipt, base_reading, references))
        retained = next(r for r in comparison["results"] if r["variant"] == "B29" and r["case"] == case)
        require(all(verified[k] == retained[k] for k in ("reading", "reading_sha256", "rows_sha256", "reference_sha256", "K", "ingress_ms", "handoff_ms", "cc_backend")), "Retained B29 changed")
        base_config = read(Path(base_receipt["metrics_dir"]) / "config.json")
        clean = baseline["clean_references"][case]
        require(digest(Path(clean["run_check"])) == clean["sha256"], "Clean reference binding changed")
        for generation in GENERATIONS:
            command, gate_record = (None, gate) if generation == "B29" else terminals[generation, case]
            run, output = (base_run, base_output) if command is None else (Path(command["run_root"]), Path(command["gate_root"]))
            receipt_path, reading_path = run / "receipt.json", output / "reading.json"
            receipt = read(receipt_path)
            require(receipt["status"] == "FINISHED" and receipt["source_dirty_paths"] == [], "Run incomplete/dirty")
            hashes = table_hashes(receipt["profiling_tables"])
            expected_hashes = table_hashes(base["profiling_tables"])
            expected_config = json.loads(json.dumps(base_config).replace(str(base_run), str(run)))
            if command is not None:
                binding = bindings[generation, case]
                require(binding["status"] == "PASS_ONLY_ATTENTION_INPUT_CHANGED" and binding["root"] == str(run), "Preflight binding changed")
                expected_hashes[ATTENTION] = tables["tables"][generation]["sha256"]
                require(hashes == table_hashes(binding["profiling_tables"]) and binding["table"] == tables["tables"][generation]["path"], "Pass table binding changed")
                expected_config["cluster_config"]["execution_time_predictor_config"]["atten_kernel_only_input_file"] = binding["table"]
                require(receipt["source_commit"] == command["source_commit"] and command["command"][5:7] == [case, str(run)] and command["command"][-2:] == ["--field", f"{binding['field']}={binding['table']}"], "Source/command changed")
            require(hashes == expected_hashes, "Non-attention inputs changed")
            require(all(digest(Path(t["path"])) == t["sha256"] for t in receipt["profiling_tables"]), "Table bytes changed")
            require(receipt["trace"] == base["trace"] and digest(Path(base["trace"]["path"])) == base["trace"]["sha256"] and digest(Path(receipt["input"])) == receipt["input_sha256"] == base["input_sha256"], "Workload changed")
            require(read(Path(receipt["metrics_dir"]) / "config.json") == expected_config, "Non-attention configuration changed")
            require(not any(receipt["a1"]["enable_dummy_mode"].values()) and not any(receipt["a1"]["step_cost_table_bound_by_predictor"].values()) and receipt["a11"]["stale_stage_end_events"] == 0, "Invalid replay path")
            require({p.name: digest(p) for p in (output / "rows/vllm").glob("*.csv")} == references, "Native CSV bytes changed")
            with (output / "rows/frontier/a10.csv").open() as stream:
                request_rows = list(csv.DictReader(stream))
            require(len(request_rows) == len({r["request_id"] for r in request_rows}) == 128 and all(int(r["request_num_prefill_tokens"]) == 6144 and int(r["request_num_decode_tokens"]) == 1024 for r in request_rows), "Request rows changed")
            reading = read(reading_path)
            if command is not None:
                require(digest(receipt_path) == command["receipt_sha256"] and digest(reading_path) == gate_record["reading_sha256"] and reading["verdict"] == gate_record["verdict"], "Terminal evidence changed")
            rows.extend(metric_rows(generation, case, reading, base_reading))
            results.append(dict(generation=generation, case=case, run=str(run), reading=str(reading_path),
                                receipt_sha256=digest(receipt_path), reading_sha256=digest(reading_path),
                                rows_sha256=digest(output / "rows/frontier/a10.csv"), reference_sha256=references,
                                table_sha256=hashes, source_commit=receipt["source_commit"], command=command, gate_execution=gate_record, verdict=reading["verdict"]))
            sources.extend((receipt_path, reading_path, Path(receipt["metrics_dir"]) / "config.json"))
    require(len(results) == 6 and len(rows) == 96, "Incomplete comparison matrix")
    spread, variability = summaries(rows)
    return dict(status="PASS_EVIDENCE_CHECKS", classification="DIAGNOSTIC_ATTENTION_PASS_COMPARISON", formal_qualification=False,
                results=results, metrics=rows, cross_generation_spread=spread, native_variability=variability,
                methods="Spread=max-min over B29/pass1/pass2; SD/CV describe three native repeats, not a confidence interval.",
                sources=[dict(path=str(p), sha256=digest(p)) for p in [*sources, Path(__file__), Path(__file__).with_name("v4_f_variant_comparison.py")]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a fresh JSON path")
    document = collect(args.task_root)
    with args.output.open("x") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print("PASS_EVIDENCE_CHECKS readings=6 metrics=96 spreads=8 native_variability=8")


if __name__ == "__main__":
    main()
