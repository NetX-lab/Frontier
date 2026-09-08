"""Compare bounded formal operator scopes and inspect correlated CUDA kernels."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def kernel_summary(path):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    device_events = [event for event in events if event.get("cat") in
                     {"kernel", "gpu_memcpy", "gpu_memset"} and event.get("dur", 0) > 0]
    assert device_events, path
    runtimes = {event.get("args", {}).get("correlation"): event for event in events
                if event.get("cat") in {"cuda_runtime", "cuda_driver"}}
    scopes = [event for event in events if event.get("cat") == "user_annotation"
              and event.get("name", "").startswith("frontier_") and event.get("dur", 0) > 0]
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in device_events)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    kernel_rows = []
    by_scope = defaultdict(float)
    for event in device_events:
        runtime = runtimes.get(event.get("args", {}).get("correlation"))
        parents = []
        if runtime:
            parents = [scope for scope in scopes
                       if scope.get("pid") == runtime.get("pid")
                       and scope.get("tid") == runtime.get("tid")
                       and scope["ts"] <= runtime["ts"]
                       and runtime["ts"] + runtime.get("dur", 0) <= scope["ts"] + scope["dur"]]
        # Attribute nested all-reduce separately; fused add belongs to layernorm.
        parents.sort(key=lambda scope: scope["dur"])
        owners = [scope for scope in parents if scope["name"] != "frontier_add"]
        owner = owners[0]["name"].removeprefix("frontier_") if owners else "unscoped"
        by_scope[owner] += event["dur"] / 1000
        kernel_rows.append({"name": event["name"], "category": event["cat"],
                            "duration_ms": event["dur"] / 1000, "timestamp_us": event["ts"],
                            "stream": event.get("args", {}).get("stream"),
                            "device": event.get("args", {}).get("device"),
                            "scope": owner, "parents": [scope["name"] for scope in parents],
                            "launch_timestamp_us": runtime["ts"] if runtime else None})
    envelope = (max(end for _, end in intervals) - intervals[0][0]) / 1000
    active = sum(end - start for start, end in merged) / 1000
    idle_rows = []
    for previous, following in zip(merged, merged[1:]):
        next_event = next(event for event in device_events if event["ts"] == following[0])
        launch = runtimes.get(next_event.get("args", {}).get("correlation"))
        gap_us = following[0] - previous[1]
        not_launched_us = (min(max(launch["ts"] - previous[1], 0), gap_us)
                           if launch else None)
        idle_rows.append({"start_us": previous[1], "end_us": following[0],
                          "duration_ms": gap_us / 1000, "next_device_event": next_event["name"],
                          "next_launch_not_started_ms": not_launched_us / 1000 if launch else None})
    return {"trace_path": str(path), "device_event_count": len(device_events),
            "device_time_sum_ms": sum(event["dur"] for event in device_events) / 1000,
            "device_active_union_ms": active, "device_envelope_ms": envelope,
            "device_idle_within_envelope_ms": envelope - active,
            "scope_device_time_ms": dict(by_scope), "idle_intervals": idle_rows,
            "next_launch_not_started_ms": sum(row["next_launch_not_started_ms"] or 0
                                              for row in idle_rows),
            "idle_interpretation": "Device idle between recorded activities. A delayed next launch establishes absent command submission, not CPU compute or a removable overhead; synchronization, thread scheduling and waits remain possible causes."}, kernel_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frontier = json.loads(args.frontier.read_text())
    inventory, op_rows = [], []
    for batch_path in sorted(args.run.glob("server.batch.dp*.jsonl")):
        batches = read_rows(batch_path)
        selected = [row for row in batches if row.get("op_profile_selected")]
        assert len(selected) == 3, (batch_path, len(selected))
        assert [row["op_profile_selected_index"] for row in selected] == [0, 1, 2]
        ops_path = batch_path.with_name(batch_path.name.replace("server.batch", "server.ops"))
        ops = read_rows(ops_path)
        assert {row["batch_id"] for row in ops} == {row["batch_id"] for row in selected}
        for batch in selected:
            batch_ops = [row for row in ops if row["batch_id"] == batch["batch_id"]]
            inventory.append(dict(batch, op_rows=len(batch_ops), op_source=str(ops_path)))
            for row in batch_ops:
                op_rows.append(dict(row, request_ids=batch["request_ids"],
                                    request_num_tokens=batch["request_num_tokens"], source=str(ops_path)))
    assert len(inventory) == 24, len(inventory)
    first = next(row for row in inventory if row["dp_rank"] == row["tp_rank"] == 0
                 and row["op_profile_selected_index"] == 0)
    assert first["request_ids"] == ["cmpl-pf4096_dc1024:0-0"]
    assert first["request_num_tokens"] == [4096]
    rank_totals = []
    for batch in inventory:
        if batch["request_ids"] != first["request_ids"] or batch["request_num_tokens"] != [4096]:
            continue
        rank_ops = [row for row in op_rows if row["dp_rank"] == batch["dp_rank"]
                    and row["tp_rank"] == batch["tp_rank"] and row["batch_id"] == batch["batch_id"]]
        rank_values = defaultdict(float)
        for row in rank_ops:
            rank_values[row["op_name"]] += row["cuda_time_ms"]
        rank_totals.append({"dp_rank": batch["dp_rank"], "tp_rank": batch["tp_rank"],
                            "batch_id": batch["batch_id"],
                            "batch_cuda_event_ms": batch["batch_execution_time_ms"],
                            "inclusive_op_totals_ms": dict(rank_values)})
    write_json(args.output / "first_batch_rank_totals.json", rank_totals)
    selected_ops = [row for row in op_rows if row["dp_rank"] == row["tp_rank"] == 0
                    and row["batch_id"] == first["batch_id"]]
    totals = defaultdict(float)
    counts = defaultdict(int)
    inclusive_totals = defaultdict(float)
    for row in selected_ops:
        name = row["op_name"]
        inclusive_totals[name] += row["cuda_time_ms"]
        if name in {"expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine"} and row["scope_seq"] % 2:
            continue
        if name == "attn_post_proj":
            name = "attn_post_proj_with_tp_allreduce"
        if name == "moe_gating":
            name = "moe_gating_linear" if row["scope_seq"] % 2 == 0 else "moe_gating_routing_topk"
        if name == "tensor_parallel_allreduce" and row["scope_seq"] == 0:
            name = "embedding_tensor_parallel_allreduce"
        totals[name] += row["cuda_time_ms"]
        counts[name] += 1
    assert counts["attn_post_proj_with_tp_allreduce"] == 48
    assert counts["embedding_tensor_parallel_allreduce"] == 1
    assert counts["expert_parallel_alltoall_dispatch"] == counts["expert_parallel_alltoall_combine"] == 48
    predicted = dict(frontier["op_totals_ms"])
    predicted["attn_post_proj_with_tp_allreduce"] = (predicted.pop("attn_post_proj")
                                                   + predicted.pop("attn_tp_allreduce"))
    ep = json.loads((args.frontier.parent / "frontier_ep_rows.json").read_text())
    for phase in ("dispatch", "combine"):
        predicted["expert_parallel_alltoall_" + phase] = sum(
            float(row[phase + "_ms"]) for row in ep if row["ep_id"] == "0")
    comparisons = []
    for name in sorted(set(totals) | set(predicted)):
        actual, prediction = totals.get(name), predicted.get(name)
        excluded = name in set(frontier["excluded_from_sum"])
        gap = prediction - actual if actual is not None and prediction is not None and not excluded else None
        comparisons.append({"op_name": name, "frontier_ms": prediction, "vllm_ms": actual,
                            "gap_ms": gap, "absolute_gap_ms": abs(gap) if gap is not None else None,
                            "relative_gap": abs(gap) / actual if gap is not None and actual > 0 else None,
                            "scope_count": counts[name], "timing_mode": selected_ops[0]["timing_mode"],
                            "status": ("NESTED_OR_ALIAS_EXCLUDED" if excluded else
                                       "MISSING_SIDE" if gap is None else "DIAGNOSTIC_ONLY")})
    with (args.output / "op_gap_table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    write_json(args.output / "batch_inventory.json", inventory)
    write_json(args.output / "first_batch.json", first)
    write_json(args.output / "op_gap_table.json", comparisons)
    (args.output / "op_trace_normalized.jsonl").write_text("".join(json.dumps(row) + "\n" for row in op_rows))
    pid_ranks = {}
    with (args.run / "server.log").open() as stream:
        for line in stream:
            match = re.search(r"Worker_DP(\d+)_TP(\d+)_EP\d+.*?pid=(\d+)", line)
            if match:
                dp, tp, pid = map(int, match.groups())
                pid_ranks[pid] = (dp, tp)
    kernel_results = []
    for path in sorted((args.run / "frontier_profiler_traces").glob("*.json")):
        match = re.fullmatch(r"frontier_batch_(\d+)_(\d+)_(\d+)\.json", path.name)
        batch_id, pid, _ = map(int, match.groups())
        assert pid in pid_ranks, (path, "missing worker identity")
        dp, tp = pid_ranks[pid]
        if (dp, tp, batch_id) != (0, 0, first["batch_id"]):
            continue
        summary, rows = kernel_summary(path)
        summary.update(dp_rank=dp, tp_rank=tp, batch_id=batch_id,
                       batch_cuda_event_ms=first["batch_execution_time_ms"])
        kernel_results.append(summary)
        (args.output / "first_batch_cuda_kernels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(args.output / "kernel_summary.json", kernel_results)
    summary = {"status": "INSUFFICIENT_EVIDENCE", "analysis_state": "INCOMPLETE",
               "first_formal_local_batch_match": True, "profiled_rank_batches": len(inventory),
               "first_batch_cuda_event_ms": first["batch_execution_time_ms"],
               "exclusive_scope_sum_ms": sum(value for name, value in totals.items() if name != "add"),
               "unattributed_batch_interval_ms": first["batch_execution_time_ms"] - sum(
                   value for name, value in totals.items() if name != "add"),
               "first_batch_op_totals_ms": dict(totals), "inclusive_scope_totals_ms": dict(inclusive_totals), "kernel_summaries": kernel_results,
               "limits": "First local batch only: global dummy token, collective algorithm and profiling families differ. Three fully aligned logical batches and native KV coverage are not established. attn_post_proj contains an unselected attn_post_proj_tp_allreduce child and is compared with Frontier projection+TP; duplicate nested EP scopes use only outer even sequence; nested add rows are not additive; moe_gating is split by two ordered scopes per layer."}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
