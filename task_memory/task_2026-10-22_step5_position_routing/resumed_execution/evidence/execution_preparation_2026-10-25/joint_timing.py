"""Measure two newly priced D29 waves with full production reporting and flush.

This single-task driver has no warmup, profiling loop, shortened output, or retry.
Execute it only after the two-call proposal in joint_bindings.json is approved.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def check_bindings(bindings, directory):
    source = Path(bindings["source_root"])
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", bindings["source_commit"], "--", ".",
         ":(exclude)task_memory"], cwd=source, text=True,
    ).strip()
    require(not changed, f"Production source changed: {changed}")
    checkout = source / "frontier/cc_backend/backends/collective-sim"
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout,
            text=True).strip() == bindings["collective_checkout"], "Wrong collective checkout")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout,
            text=True).strip(), "Dirty collective checkout")
    for relative, expected in bindings["source_hashes"].items():
        require(sha256(source / relative) == expected, f"Source hash: {relative}")
    for filename, expected in bindings["input_hashes"].items():
        require(sha256(filename) == expected, f"Input hash: {filename}")
    for filename, expected in bindings["scripts"].items():
        require(sha256(directory / filename) == expected, f"Driver hash: {filename}")
    fixture = Path(bindings["fixture_root"])
    for relative, expected in bindings["fixture_metadata"].items():
        stat = (fixture / relative).stat()
        require([stat.st_size, stat.st_mtime_ns] == expected, f"Fixture metadata: {relative}")
    original = ast.parse(Path(bindings["original_setup"]).read_bytes())
    original_wave = next(node for node in original.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "phases" for target in node.targets))
    current_wave = ast.parse((directory / "joint_wave.py").read_bytes()).body[0]
    require(ast.dump(current_wave, include_attributes=False) ==
            ast.dump(original_wave, include_attributes=False), "Wave expression changed")
    return {"source_commit": bindings["source_commit"],
            "records_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=source, text=True).strip(),
            "source_paths": len(bindings["source_hashes"]),
            "fixture_metadata_paths": len(bindings["fixture_metadata"])}


def read_references(bindings):
    references = bindings["references"]
    for name in ("pricing", "ledger"):
        require(sha256(references[name]["path"]) == references[name]["sha256"],
                f"Reference hash: {name}")
    pricing = Path(references["pricing"]["path"]).read_bytes()
    ledger = json.loads(Path(references["ledger"]["path"]).read_bytes())
    operator_spec = references["operators"]
    with Path(operator_spec["path"]).open("rb") as stream:
        for _ in range(operator_spec["skip_header_lines"]):
            require(bool(stream.readline()), "Missing reference header")
        lines = [stream.readline() for _ in range(operator_spec["record_count"])]
    require(all(lines), "Incomplete reference operator segment")
    segment = b"".join(lines)
    require(len(segment) == operator_spec["bytes"] and
            hashlib.sha256(segment).hexdigest() == operator_spec["sha256"],
            "Operator reference segment changed")
    operators = [json.loads(line) for line in lines]
    require(len(ledger) == 16 and len(operators) == 208, "Wrong reference cardinality")
    require([row["ep_id"] for row in ledger] == list(range(16)), "Reference lane order")
    return pricing, ledger, operators


def pricing_payload(phases):
    rows = []
    for record in phases.lane_records:
        layer = record.execution_time.layer_execution_times[0]
        moe = layer.moe_or_mlp_time_component.operator_times
        communication = layer.communication_time_component.operator_times
        rows.append({"ep": record.ep_id, "moe": moe.op_times, "comm": communication.op_times,
            "plans": {name: plan.cache_key() for name, plan in communication.plans.items()},
            "proofs": moe.physical_frame["operator_costs"]})
    require(len(rows) == 16, "Missing priced lane")
    require(sum(len(row["moe"]) + len(row["comm"]) for row in rows) == 208,
            "Missing priced operator")
    require(sum(len(row["proofs"]) for row in rows) == 144, "Missing compute proof")
    return json.dumps(rows, sort_keys=True).encode()


def lane_identities(phases):
    return [{"ep_id": record.ep_id, "batch_id": record.batch.id,
             "request_ids": [str(request.id) for request in record.batch.requests],
             "num_tokens": list(record.batch.num_tokens),
             "source_batch_ids": list(record.batch.source_batch_ids)}
            for record in phases.lane_records]


def expected_output(identities, ledger, operators):
    require([row["ep_id"] for row in identities] == list(range(16)), "Actual lane order")
    batch_offset = identities[0]["batch_id"] - ledger[0]["batch_id"]
    request_offset = int(identities[0]["request_ids"][0]) - int(ledger[0]["request_ids"][0])
    expected_ledger = copy.deepcopy(ledger)
    for identity, reference, expected in zip(identities, ledger, expected_ledger):
        require(identity["batch_id"] == reference["batch_id"] + batch_offset,
                "Unexpected EP batch allocation")
        require(identity["request_ids"] ==
                [str(int(value) + request_offset) for value in reference["request_ids"]],
                "Unexpected synthetic destination request allocation")
        require(identity["source_batch_ids"] == reference["source_batch_ids"],
                "Captured source batch identity changed")
        require(len(identity["num_tokens"]) == len(reference["request_ids"]),
                "Destination token-count cardinality changed")
        expected["batch_id"] = identity["batch_id"]
        expected["request_ids"] = identity["request_ids"]
    expected_operators = copy.deepcopy(operators)
    for row in expected_operators:
        identity = identities[row["meta"]["ep_id"]]
        row["batch_id"] = identity["batch_id"]
        row["meta"]["request_ids"] = identity["request_ids"]
    return expected_ledger, expected_operators


def validate_output(directory, identities, ledger, operators):
    expected_ledger, expected_operators = expected_output(identities, ledger, operators)
    ledger_path = directory / "frontier_ep_wave_lane_ledger.jsonl"
    with ledger_path.open("rb") as stream:
        for index, expected in enumerate(expected_ledger):
            line = stream.readline()
            require(line == (json.dumps(expected, sort_keys=True) + "\n").encode(),
                    f"Full sorted ledger row {index} differs")
        require(stream.readline() == b"", "Extra ledger rows")
    trace_path = directory / "op_traces.jsonl"
    with trace_path.open("rb") as stream:
        header = json.loads(stream.readline())
        require(set(header) == {"meta"} and set(header["meta"]) ==
                {"timestamp", "simulation_info", "version"}, "Trace header schema differs")
        require(header["meta"]["version"] == "1.0" and
                math.isfinite(header["meta"]["timestamp"]), "Invalid trace header")
        for index, expected in enumerate(expected_operators):
            require(stream.readline() == (json.dumps(expected) + "\n").encode(),
                    f"Full operator row {index} differs")
        require(stream.readline() == b"", "Extra operator rows")
    return {"status": "PASS_COMPLETE_OUTPUT_PARITY", "ledger_rows": 16,
            "operator_rows": 208, "compute_proofs": 144,
            "ledger_bytes": ledger_path.stat().st_size, "ledger_sha256": sha256(ledger_path),
            "trace_bytes": trace_path.stat().st_size, "trace_sha256": sha256(trace_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    command = [sys.executable, *sys.argv]
    directory = args.bindings.resolve().parent
    bindings = json.loads(args.bindings.read_bytes())
    source = Path(bindings["source_root"])
    require(bindings["proposal"]["labels"] == ["cold", "warm"], "Exactly two calls are proposed")
    root = args.output_root.resolve()
    require(root.is_relative_to(Path("/data/ycfeng/tmp")), "Use the approved local scratch root")
    require(sys.version.split()[0] == bindings["python_version"], "Wrong Python version")
    require(Path(sys.executable).absolute() == Path(bindings["python"]), "Wrong interpreter")
    require(sys.dont_write_bytecode, "Run Python with -B")
    for name, value in bindings["math_thread_env"].items():
        require(os.environ.get(name) == value, f"Wrong math-thread setting: {name}")
    source_before = check_bindings(bindings, directory)
    references = read_references(bindings)
    root.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "IN_PROGRESS", "command": command, "cwd": str(Path.cwd()),
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "source": source_before, "python": sys.version,
               "configured_memory_max": "30G", "pricing_attempts": 0,
               "samples": [], "limitations": bindings["limitations"]}
    write_json(root / "execution.json", receipt)
    try:
        working = root / "config_cwd"
        model_path = Path("data/config/models/stepfun-ai__Step-5-Preview.json")
        (working / model_path).parent.mkdir(parents=True)
        (working / model_path).write_bytes((source / model_path).read_bytes())
        os.chdir(working)
        sys.path.insert(0, str(source))
        setup = directory / "joint_setup.py"
        namespace = {"__name__": "__main__", "__file__": str(setup)}
        sys.argv = [str(setup), "--output", str(root / "setup-output")]
        setup_started = time.perf_counter()
        exec(compile(setup.read_bytes(), str(setup), "exec"), namespace)
        from frontier.cc_backend.collective_profile import CollectiveProfileInputs
        manager = namespace["manager"]
        manager._collective_profile_inputs = CollectiveProfileInputs()
        namespace["predictor"]._cc_backend.bind_collective_profile_inputs(
            manager.collective_profile_inputs)
        receipt["setup_s"] = time.perf_counter() - setup_started
        wave_path = directory / "joint_wave.py"
        wave_code = compile(wave_path.read_bytes(), str(wave_path), "exec")
        for label in bindings["proposal"]["labels"]:
            sample_root = root / label
            sample_config = copy.copy(namespace["config"])
            sample_config.metrics_config = copy.copy(sample_config.metrics_config)
            sample_config.metrics_config.output_dir = str(sample_root)
            config = sample_config.metrics_config
            require(config.enable_op_level_tracing and config.write_metrics and
                    config.store_frontier_stage_batch_ledger, "Required full output disabled")
            require(not config.store_operation_metrics and not config.enable_per_layer_expansion,
                    "Captured diagnostic output configuration changed")
            trace = namespace["TraceStore"](str(sample_root), enabled=config.enable_op_level_tracing)
            trace.initialize({"evidence": "D29_CURRENT_JOINT_TIMING", "label": label,
                              "source_commit": bindings["source_commit"]})
            store = namespace["MetricsStore"](sample_config, namespace["clusters"], trace_store=trace)
            require(trace._buffer_size_limit == 5000, "Production trace threshold changed")
            receipt["pricing_attempts"] += 1
            write_json(root / "execution.json", receipt)
            started = time.perf_counter()
            exec(wave_code, namespace)
            priced = time.perf_counter()
            phases = namespace["phases"]
            timing = namespace["calculate_ep_wave_timing"](
                start_time_s=namespace["captured"]["meta"]["wave_start_time_s"], phases=phases,
                barrier_logger=namespace["noop"], wave_logger=namespace["noop"],
                cluster_type=namespace["role"], batch_id=namespace["batch"].id,
                layer_id=namespace["layer_id"], participant_ep_ids=namespace["workload"].participant_ep_ids,
                trace_identity=None)
            plan = namespace["EPWavePlan"](None, namespace["workload"], phases, timing, None)
            planned = time.perf_counter()
            store.on_ep_wave_schedule(plan, time=namespace["captured"]["meta"]["wave_start_time_s"],
                replica_id=namespace["batch"].replica_id, stage_id=0, cluster_type=namespace["role"])
            emitted = time.perf_counter()
            trace.close()
            store._frontier_ep_wave_lane_ledger_rows.close()
            finished = time.perf_counter()
            sample = {"label": label, "pricing_s": priced - started,
                "plan_and_timing_s": planned - priced, "emission_with_ledger_flush_s": emitted - planned,
                "final_flush_and_close_s": finished - emitted, "joint_s": finished - started,
                "trace_buffer_limit": trace._buffer_size_limit, "trace_buffer_after_close": len(trace._buffer),
                "ledger_count": len(store._frontier_ep_wave_lane_ledger_rows),
                "lane_identities": lane_identities(phases), "validation_in_timed_interval": False}
            receipt["samples"].append(sample)
            write_json(root / "execution.json", receipt)
            require(not trace._buffer and sample["ledger_count"] == 16, "Incomplete final flush")
            payload = pricing_payload(phases)
            (sample_root / "pricing_payload.json").write_bytes(payload)
            require(payload == references[0], "Complete pricing payload differs")
            sample["validation"] = validate_output(sample_root, sample["lane_identities"],
                                                   references[1], references[2])
            write_json(root / "execution.json", receipt)
        os.chdir(source)
        require(check_bindings(bindings, directory) == source_before, "Binding changed during run")
        require(receipt["pricing_attempts"] == 2, "Unexpected pricing call count")
        require(receipt["samples"][0]["lane_identities"] != receipt["samples"][1]["lane_identities"],
                "Warm reporting reused the cold plan")
        receipt["status"] = "PASS_TWO_COHESIVE_CPU_FIXTURE_WAVES"
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["error"] = f"{type(error).__name__}: {error}"
        (root / "failure.log").write_text(traceback.format_exc())
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        write_json(root / "execution.json", receipt)
    print(json.dumps({"status": receipt["status"], "pricing_attempts": receipt["pricing_attempts"],
                      "joint_s": [sample["joint_s"] for sample in receipt["samples"]]}))


if __name__ == "__main__":
    main()
