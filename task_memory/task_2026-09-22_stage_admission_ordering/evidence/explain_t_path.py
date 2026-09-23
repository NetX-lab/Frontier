"""Explain every path-T difference of the P3 comparison from the stage ledger.

For each T case whose metrics differ, check that every (stage, lane) runs the
same ordered batches with the same component durations before and after, so
the difference is start times only; report the §4.5 metric as absolute and as
a fraction of stage busy time.

Usage: python explain_t_path.py <matrix root> <after set> <compare json> <output json>

The before set is always ``base``.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

root, after_set = Path(sys.argv[1]), sys.argv[2]
compare_path, output = Path(sys.argv[3]), Path(sys.argv[4])


def lane_rows(set_name, case_id):
    ledger = next((root / set_name / case_id / "metrics").rglob("frontier_stage_batch_ledger.jsonl"))
    rows = defaultdict(list)
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if row["execution_scope"] == "ATTN_DP_LANE":
            rows[(row["stage_id"], row["replica_local_id"])].append(row)
    for key in rows:
        rows[key].sort(key=lambda row: row["stage_start_ts"])
    return rows


def signature(row):
    return (tuple(row["request_ids"]), round(row["stage_end_ts"] - row["stage_start_ts"], 9),
            json.dumps(row["execution_time"], sort_keys=True))


def fraction(metric):
    return {stage: round(m["multi_lane_busy_time"] / m["busy_time"], 4) for stage, m in metric.items()}


report = []
for row in json.load(open(compare_path)):
    if row["path"] != "T" or row["verdict"] == "PASS":
        continue
    before, after = lane_rows("base", row["case_id"]), lane_rows(after_set, row["case_id"])
    same_work = before.keys() == after.keys() and all(
        [signature(r) for r in before[key]] == [signature(r) for r in after[key]] for key in before
    )
    first_start = {side: {lane: rows[(0, lane)][0]["stage_start_ts"] for (stage, lane) in rows if stage == 0}
                   for side, rows in (("before", before), ("after", after))}
    report.append({
        "case_id": row["case_id"], "verdict": row["verdict"],
        "same_batches_and_component_durations": same_work,
        "differing_files": row.get("differing_files"),
        "witness_increase": row.get("witness_increase"),
        "multi_lane_busy_time": {side: {s: m["multi_lane_busy_time"] for s, m in row[f"lane_metric_{side}"].items()}
                                 for side in ("before", "after")},
        "co_execution_fraction": {side: fraction(row[f"lane_metric_{side}"]) for side in ("before", "after")},
        "peak_lanes": {side: {s: m["peak_lanes"] for s, m in row[f"lane_metric_{side}"].items()}
                       for side in ("before", "after")},
        "first_stage0_start": first_start,
    })
output.write_text(json.dumps(report, indent=1, sort_keys=True))
for item in report:
    print(item["case_id"], item["verdict"], "same_work=", item["same_batches_and_component_durations"],
          "frac", item["co_execution_fraction"]["before"].get("MONOLITHIC/0/0"), "->",
          item["co_execution_fraction"]["after"].get("MONOLITHIC/0/0"),
          "peak", item["peak_lanes"]["before"].get("MONOLITHIC/0/0"), "->",
          item["peak_lanes"]["after"].get("MONOLITHIC/0/0"),
          "starts", item["first_stage0_start"]["before"], "->", item["first_stage0_start"]["after"])
