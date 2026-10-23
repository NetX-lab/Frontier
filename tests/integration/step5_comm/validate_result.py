"""Advisory communication artifact checks; never control a running workload."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from frontier.profiling.collectives.vllm_forward_collectives import (
    ModelShape, ServingParallelism, call_bytes, forward_collectives, group_members,
    scenarios_from_grid,
)


def validate(job: dict, root: Path) -> dict:
    from analyze_result import queue_held_analysis, read_jsonl

    config = json.loads(Path(job["benchmark"]).read_text())
    parallel = ServingParallelism.from_engine(config["engine"])
    model = ModelShape.from_hf_config(json.loads(Path(config["model_config"]).read_text()))
    plan = {(s.name, c.name): (s, c) for s in scenarios_from_grid(parallel, config["grid"])
            for c in forward_collectives(model, parallel, s,
                                        native_call_names=config["grid"].get("native_call_names"))}
    records = read_jsonl(root / "collectives/measurements.jsonl")
    kernels = read_jsonl(root / "collectives/kernels.jsonl")
    summaries = list(csv.DictReader((root / "collectives/summary.csv").open()))
    summary_medians = {(int(r["pass"]), r["scenario"], r["call"]): float(r["median_us"]) for r in summaries}
    expected_keys = {(p, s, c) for p in range(config["passes"]) for s, c in plan}
    actual_keys = [(r["pass"], r["scenario"], r["call"]) for r in records]
    checks = {"measurement_rows": len(records) == job["expected"]["measurements"],
              "measurement_keys": len(actual_keys) == len(expected_keys) and set(actual_keys) == expected_keys,
              "kernel_rows": len(kernels) == job["expected"]["kernel_rows"],
              "kernel_keys": {(r["scenario"], r["call"]) for r in kernels} == set(plan),
              "summary_rows": len(summaries) == len(records),
              "summary_keys": {(int(r["pass"]), r["scenario"], r["call"]) for r in summaries} == expected_keys,
              "known_communicators": all(r["communicator"] != "unknown" for r in kernels)}
    checks["prepared_counts"] = (job["expected"]["measurements"] == len(expected_keys) and
                                 job["expected"]["kernel_rows"] == len(plan))
    checks["kernel_keys"] &= len(kernels) == len(plan)
    checks["summary_keys"] &= len(summaries) == len(expected_keys)
    identity_errors, queue_failures = [], []
    checks["summary_medians"] = True
    medians = defaultdict(list)
    for record in records:
        key = record["scenario"], record["call"]
        if key not in plan:
            identity_errors.append({"scenario":key[0], "call":key[1], "field":"unselected_call"})
            continue
        scenario, call = plan[key]
        rails = ((0,) if call.op == "cpu_all_reduce" else
                 scenario.active_dp_rails if scenario.active_dp_rails is not None else
                 tuple(range(parallel.tensor_parallel_size)))
        ranks = [r for r in range(parallel.world_size)
                 if (call.group != "dp" or r % parallel.tensor_parallel_size in rails)
                 and any(call_bytes(call, parallel, r))]
        groups = sorted({tuple(group_members(parallel, call.group, r)) for r in ranks})
        expected = {"participating_ranks": ranks, "group_ranks": [list(g) for g in groups],
                    "op":call.op,"group":call.group,"cuda_graph":call.captured,
                    "tokens_per_dp_rank": list(scenario.tokens_per_dp_rank),
                    "useful_tokens_per_dp_rank": list(scenario.useful_tokens_per_dp_rank or scenario.tokens_per_dp_rank),
                    "padded_tokens_per_dp_rank": list(scenario.tokens_per_dp_rank),
                    "concurrent_dp_rails": len(groups) if call.group == "dp" else None,
                    "dtypes": list(call.dtypes), "columns": list(call.columns),
                    "max_input_bytes": max(call_bytes(call, parallel, r)[0] for r in ranks),
                    "max_output_bytes": max(call_bytes(call, parallel, r)[1] for r in ranks)}
        for field, value in expected.items():
            if record[field] != value:
                identity_errors.append({"pass": record["pass"], "scenario": key[0],
                                        "call": key[1], "field": field})
        domain = "host" if call.op == "cpu_all_reduce" else "device"
        timing = queue_held_analysis(record, domain, 0.01)
        summary_key = record["pass"], *key
        checks["summary_medians"] &= (summary_key in summary_medians and math.isclose(
            timing["median_us"], summary_medians[summary_key], rel_tol=1e-12))
        if domain == "device" and timing["queue_gate"] != "PASS":
            queue_failures.append({"pass": record["pass"], "scenario": key[0],
                                   "call": key[1], **timing})
        medians[key].append(timing["median_us"])
    checks["payload_groups_rails"] = not identity_errors
    checks["queue_held"] = not queue_failures
    checks["positive_finite_times"] = all(math.isfinite(t) and t > 0 for ts in medians.values() for t in ts)
    probe = json.loads((root / "native_dispatch/dispatch_probe.json").read_text())
    ranks = probe["ranks"]
    checks["native_rank_count"] = len(ranks) == parallel.world_size and {r["rank"] for r in ranks} == set(range(parallel.world_size))
    selected = "profile_manifest" in job
    keys = ({(key,layer) for key in job["native_probe"]["moe_input_ids"] for layer in job["native_probe"]["layers"]}
            if selected else {(None,layer) for layer in job["native_probe"]["layers"]})
    checks["native_match"] = probe["status"] == "match" and all(
        r["status"] == "match" and len(r["layers"]) == len(keys) and
        {(layer.get("moe_input_id"),layer["layer_id"]) for layer in r["layers"]} == keys and
        all(layer["comparison"]["status"] == "match" and
            all(c["completed"] for c in layer["collectives"]) for layer in r["layers"])
        for r in ranks)
    if selected:
        native=json.loads((root/"collectives/plan.json").read_text())
        shared=native["runtime"]["native_facts"]["shared"]
        binding=json.loads(Path(job["profile_manifest"]).read_text())["native_binding"]
        binding_keys=("native_runtime_adapter","native_reference","profile_generation","installed_package",
                      "model_config_revision","model_config_sha256","installed_source_hashes")
        checks["native_binding_identity"] = all(shared[k] == binding[k] for k in binding_keys)
        checks["native_measurement_identity"] = native["evidence_kind"] == "native_measurement" and all(
            r["evidence_kind"] == "native_measurement" and r["native_context"]["shared"] == shared for r in records)
        checks["native_identity_final"] = json.loads((root/"collectives/native_identity.json").read_text())["status"] == "PASS"
        checks["native_selected_source"] = (probe["verification_scope"] == "eager_native_moe_module" and
            not probe["full_model_verified"] and not probe["model_runner_verified"] and all(
            layer["source_frame"]["source_lane_id"] == r["rank"] // parallel.tensor_parallel_size and
            layer["native_facts"]["shared"] == shared and
             layer["native_facts"]["callables"] for r in ranks for layer in r["layers"]))
        checks["native_actual_routing"]=all(layer.get("routing_comparison",{}).get("status") == "match" and
            not layer["routing_comparison"]["differences"] and len(layer["routing_comparison"]["observations"]) == 1 and
            layer["routing_comparison"]["observations"][0]["physical_rows"] == layer["source_frame"]["physical_rows"] and
            layer["routing_comparison"]["observations"][0]["live_rows"] == sum(layer["source_frame"]["query_lens"]) and
            layer["routing_comparison"]["observations"][0]["live_assignments"] == sum(layer["source_frame"]["query_lens"])*model.num_experts_per_tok
            for r in ranks for layer in r["layers"])
        from dataclasses import asdict
        from frontier.profiling.moe.inputs import read_routing_profile_inputs
        inputs={i.moe_input_id:i for i in read_routing_profile_inputs(job["native_probe"]["routing_input_file"])}
        checks["native_source_frames"] = all(layer["moe_input_id"] in inputs and layer["source_frame"] ==
            json.loads(json.dumps(asdict(inputs[layer["moe_input_id"]].source_frames[r["rank"]//parallel.tensor_parallel_size])))
            for r in ranks for layer in r["layers"])
        checks["native_actual_counts"] = all(type(k["profiled_calls"]) is int and
            k["profiled_calls"] == config["calls_per_repeat"] and
            len(k["per_rank_kernel_counts"]) == parallel.world_size and all(type(n) is int and n > 0 and
            len(e) == n*k["profiled_calls"] for n,e in zip(k["per_rank_kernel_counts"],k["per_rank_events"])) and
            len(k["per_rank_events"]) == parallel.world_size for k in kernels)
        reached={tuple((t["dtype"],t["shape"][1]) for t in call["inputs"]) for r in ranks for layer in r["layers"]
            for call in layer["collectives"] if call.get("op") == "all_gatherv"}
        checks["native_reached_dispatch"] = all(tuple(zip(r["dtypes"],r["columns"])) in reached
            for r in records if r["op"] == "all_gatherv")
        runtime=native["runtime"].get("per_rank_startup_runtime",{})
        from frontier.cc_backend.collective_plan import collective_runtime_identity
        rank_probes={r["rank"]:r for r in ranks}
        checks["native_execution_policy"]=(set(runtime) == {str(rank) for rank in range(parallel.world_size)} and all(
            set(record["native_context"].get("execution_policy",{})) == {str(rank) for rank in record["participating_ranks"]} and
            all(policy.get("runtime") and collective_runtime_identity(policy["runtime"]) ==
                collective_runtime_identity(runtime[rank]) and policy.get("branches") and not policy.get("diagnostic_errors")
                for rank,policy in record["native_context"].get("execution_policy",{}).items()) for record in records))
        checks["native_probe_policy"]=all(any(
            call.get("execution_policy") == policy and call["op"] == record["op"] and
            call["group_ranks"] == group_members(parallel,record["group"],int(rank)) and
            tuple((t["dtype"],t["shape"][1]) for t in call["inputs"]) == tuple(zip(record["dtypes"],record["columns"]))
            for layer in rank_probes[int(rank)]["layers"] for call in layer["collectives"])
            for record in records
            for rank,policy in record["native_context"].get("execution_policy",{}).items())
    if "local_work" in config["grid"]:
        local=read_jsonl(root/"collectives/local_work.jsonl")
        local_keys={(p,row["name"]) for p in range(config["passes"]) for row in config["grid"]["local_work"]}
        checks["local_evidence_keys"] = len(local) == job["expected"]["local_work_rows"] == len(local_keys) and {
            (row["pass"],row["name"]) for row in local} == local_keys
        definitions={row["name"]:row for row in config["grid"]["local_work"]}
        checks["local_layout_queue"] = all(row["name"] in definitions and all(row[k] == v for k,v in
            definitions[row["name"]].items()) and row["scope"] == "independent_preallocated_operand" and
            row["row_stride"] == row["columns"] and len(row["per_rank_call_us"]) == parallel.world_size and
            len(row["per_rank_queue_held"]) == parallel.world_size and
            all(len(held) == len(times) == config["repeats"] and all(held) and
                all(math.isfinite(t) and t > 0 for t in times) for held,times in
                zip(row["per_rank_queue_held"],row["per_rank_call_us"])) for row in local)
    checks["native_groups"] = all(r["group_ranks"] == {
        group: group_members(parallel, group, r["rank"]) for group in ("tp", "dp", "ep")}
        for r in ranks)
    node_hosts = defaultdict(set)
    for r in ranks:
        node_hosts[r["rank"] // job["gpus_per_node"]].add(r["host"])
    checks["native_placement"] = (len(node_hosts) == job["resources"]["replica"] and
                                  all(len(h) == 1 for h in node_hosts.values()) and
                                  len(set().union(*node_hosts.values())) == len(node_hosts))
    outputs = ([root] if job["resources"]["replica"] == 1 else
               [root / f"node_{i}" for i in range(job["resources"]["replica"])])
    for name in ("source_integrity", "transport_snapshot", "manifest"):
        checks[name] = all((path / f"{name}.json").is_file() and
                           json.loads((path / f"{name}.json").read_text())["status"] == "PASS"
                           for path in outputs)
    if selected:
        checks["worker_native_binding"] = all(json.loads((path/"native_binding.json").read_text())["status"] == "PASS" for path in outputs)
    for phase in ("collective", "native_probe"):
        checks[phase + "_exits"] = all((path / f"{phase}_exit.json").is_file() and
                                      json.loads((path / f"{phase}_exit.json").read_text())["returncode"] == 0
                                      for path in outputs)
    if job["resources"]["replica"] > 1:
        checks["actual_rdma_transport"] = json.loads((root / "transport.json").read_text())["status"] == "PASS"
    spreads = {"/".join(key): (max(values) - min(values)) / statistics.median(values)
               for key, values in medians.items() if all(t > 0 for t in values)}
    return {"status": "PASS" if all(checks.values()) else "UNVERIFIED", "advisory_only": True,
            "cost_data_qualified": False, "checks": checks, "identity_errors": identity_errors,
            "queue_failures": queue_failures, "pass_median_spread": spreads,
            "limitations": ["Spread is reported, not a new acceptance threshold.",
                            "Transport, every container exit and source pins require their own receipts.",
                            "No full-model, independent held-out or predictor qualification."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(json.loads(args.config.read_text()), args.root)
    except Exception as error:
        report = {"status": "UNVERIFIED", "advisory_only": True, "error": repr(error)}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
