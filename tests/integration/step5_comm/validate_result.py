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
            for c in forward_collectives(model, parallel, s)}
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
    identity_errors, queue_failures = [], []
    checks["summary_medians"] = True
    medians = defaultdict(list)
    for record in records:
        key = record["scenario"], record["call"]
        scenario, call = plan[key]
        rails = ((0,) if call.op == "cpu_all_reduce" else
                 scenario.active_dp_rails if scenario.active_dp_rails is not None else
                 tuple(range(parallel.tensor_parallel_size)))
        ranks = [r for r in range(parallel.world_size)
                 if (call.group != "dp" or r % parallel.tensor_parallel_size in rails)
                 and any(call_bytes(call, parallel, r))]
        groups = sorted({tuple(group_members(parallel, call.group, r)) for r in ranks})
        expected = {"participating_ranks": ranks, "group_ranks": [list(g) for g in groups],
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
        checks["summary_medians"] &= math.isclose(
            timing["median_us"], summary_medians[(record["pass"], *key)], rel_tol=1e-12)
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
    checks["native_match"] = probe["status"] == "match" and all(
        r["status"] == "match" and len(r["layers"]) == 2 and
        {layer["layer_id"] for layer in r["layers"]} == set(job["native_probe"]["layers"]) and
        all(layer["comparison"]["status"] == "match" and
            all(c["completed"] for c in layer["collectives"]) for layer in r["layers"])
        for r in ranks)
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
