"""Audit partial profiler coverage and retain complete first-batch kernel evidence."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from issue26_first_batch_op_rca_vllm import kernel_summary, read_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    identities = {}
    for line in (args.run / "server.log").open():
        match = re.search(r"Worker_DP(\d+)_TP(\d+)_EP\d+.*?pid=(\d+)", line)
        if match:
            dp, tp, pid = map(int, match.groups())
            identities[pid] = (dp, tp)
    first_batches = {}
    for path in args.run.glob("server.batch.dp0.*.jsonl"):
        first = next(row for row in read_rows(path) if row.get("op_profile_selected_index") == 0)
        assert first["request_ids"] == ["cmpl-pf4096_dc1024:0-0"]
        assert first["request_num_tokens"] == [4096]
        first_batches[(first["dp_rank"], first["tp_rank"])] = first
    summaries, coverage, families = [], [], []
    for path in sorted((args.run / "frontier_profiler_traces").glob("*.json")):
        match = re.fullmatch(r"frontier_batch_(\d+)_(\d+)_(\d+)\.json", path.name)
        batch, pid, _ = map(int, match.groups())
        dp, tp = identities[pid]
        if dp != 0:
            continue
        events = json.loads(path.read_text())["traceEvents"]
        launches = [event for event in events if event.get("cat") in {"cuda_runtime", "cuda_driver"}
                    and "LaunchKernel" in event.get("name", "")]
        correlations = {event.get("args", {}).get("correlation")
                        for event in events if event.get("cat") == "kernel"}
        missing = [event for event in launches
                   if event.get("args", {}).get("correlation") not in correlations]
        coverage.append(dict(batch_id=batch, dp_rank=dp, tp_rank=tp, pid=pid, trace=str(path),
                             launches=len(launches), missing_device_correlations=missing))
        first = first_batches[(dp, tp)]
        if batch != first["batch_id"]:
            continue
        assert not missing, (path, "Incomplete first-batch device activity")
        summary, rows = kernel_summary(path)
        summary.update(dp_rank=dp, tp_rank=tp, pid=pid, batch_id=batch,
                       batch_cuda_event_ms=first["batch_execution_time_ms"],
                       kernel_launches=len(launches), missing_device_correlations=0)
        write_json(args.output / f"first_dp{dp}_tp{tp}_summary.json", summary)
        (args.output / f"first_dp{dp}_tp{tp}_kernels.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        grouped = defaultdict(lambda: {"count": 0, "duration_ms": 0.0})
        for row in rows:
            key = (row["scope"], row["name"])
            grouped[key]["count"] += 1
            grouped[key]["duration_ms"] += row["duration_ms"]
        families.append(dict(dp_rank=dp, tp_rank=tp, batch_id=batch, kernels=[
            dict(scope=scope, name=name, **value) for (scope, name), value in grouped.items()]))
        summaries.append({key: value for key, value in summary.items() if key != "idle_intervals"})
    assert {row["tp_rank"] for row in summaries} == set(range(4))
    write_json(args.output / "first_batch_rank_summary.json", summaries)
    write_json(args.output / "all_dp0_collector_coverage.json", coverage)
    write_json(args.output / "first_batch_kernel_families.json", families)
    print(json.dumps({"first_batch_coverage": "PASS", "full_run": "NOT_ESTABLISHED",
                      "coverage": [{"batch": row["batch_id"], "tp": row["tp_rank"],
                                    "missing": len(row["missing_device_correlations"])}
                                   for row in coverage]}))


if __name__ == "__main__":
    main()
