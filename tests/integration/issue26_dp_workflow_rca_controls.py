"""Separate DP selection-state equivalence from request-order sensitivity."""

import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from frontier.scheduler.request_load import RequestLoad
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer


def source_selector(path):
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "DPLBAsyncMPClient")
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "get_core_engine_for_request")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), method], type_ignores=[])
    namespace = {"sys": sys}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method.name]


def replay(selector, counts, request_ids):
    owner = SimpleNamespace(lb_engines=copy.deepcopy(counts), eng_start_index=0,
                            core_engines=list(range(len(counts))), client_count=1,
                            reqs_in_flight={})
    selections = []
    for request_id in request_ids:
        before = copy.deepcopy(owner.lb_engines)
        request = SimpleNamespace(request_id=request_id, data_parallel_rank=None)
        lane = selector(owner, request)
        selections.append({"request_id": request_id, "lane": lane,
                           "counts_before": before})
    return selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "observation", "mapping", "comparison", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    observation = json.loads(args.observation.read_text())
    assert observation["status"] == "PASS"
    comparison = json.loads(args.comparison.read_text())
    with args.mapping.open() as stream:
        mapping = {row["frontier_request_id"]: row["client_request_id"]
                   for row in csv.DictReader(stream)}
    select = source_selector(args.source)
    records = []
    for row in observation["routes"]:
        request_id = mapping[row["request_id"]]
        reference = replay(select, row["counts_before"], [request_id])[0]
        router = VllmDPLoadBalancer(len(row["counts_before"]))
        router.frontend_counts = [RequestLoad(*load) for load in row["counts_before"]]
        actual = router.select(0.0)
        assert actual == reference["lane"] == row["lane"]
        records.append({**reference, "time_s": row["time_s"], "frontier_lane": actual})

    indexed = {row["request_id"]: row for row in comparison["comparisons"]}
    first = next(i for i, row in enumerate(records) if not indexed[row["request_id"]]["same_dp"])
    pair = records[first:first + 2]
    assert len(pair) == 2
    request_ids = [row["request_id"] for row in pair]
    original = replay(select, pair[0]["counts_before"], request_ids)
    reversed_order = replay(select, pair[0]["counts_before"], request_ids[::-1])
    assert [row["lane"] for row in original] == [row["lane"] for row in pair]
    restored = all(row["lane"] == indexed[row["request_id"]]["vllm"]["dp"]
                   for row in reversed_order)
    swap = dict(zip(request_ids, request_ids[::-1]))
    member_controls = []
    for row in comparison["comparisons"]:
        members = sorted([swap.get(rid, rid), count]
                         for rid, count in row["frontier"]["members"])
        member_controls.append({"request_id": row["request_id"],
                                "original_match": row["same_members"],
                                "match_after_label_swap": members == row["vllm"]["members"]})
    result = {
        "status": "PASS", "source": str(args.source),
        "scope": "Exact source-method controls on measured Frontier states; diagnostic only.",
        "same_state_selections": records,
        "first_pair_counterfactual": {
            "original": original, "reversed_request_order": reversed_order,
            "reversed_order_matches_observed_vllm_owners": restored,
            "operator_durations_changed": False,
            "report_events_between_routes": [row for row in observation["events"]
                if row["event"] == "report" and pair[0]["time_s"] <= row["time_s"] <= pair[1]["time_s"]],
        },
        "membership_label_controls": member_controls,
        "limits": "The reversed order is a counterfactual, not an observed vLLM routing order. "
                  "It establishes sufficiency, not attribution of the isolated-run mismatch. "
                  "Membership relabeling does not rerun admission or align KV progress. "
                  "No contribution of execution-time error to later load snapshots is quantified.",
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "same_state_routes": len(records),
                      "first_pair_owner_match_after_order_swap": restored}))


if __name__ == "__main__":
    main()
