#!/usr/bin/env python3
"""Compare DP-lane stage admission between vLLM traces and Frontier ledgers.

Reads the vLLM ground truth written by ``run_vllm_worker.sh`` and the G7 cases
of ``tests.e2e.stage_admission_matrix`` for a set run before the change and one
run after it, computes the per-forward lane metrics of the stage-admission plan
(§4.7, M1–M5) on both sides with the same definitions, and writes the
workflow-gap table, summary and status for the calibration case.

A vLLM forward row is one ``pp_boundary`` record: its lane is the DP rank the
driver pinned its requests to, and its stage is ``pp_rank``.  Stage-0
intervals end at ``send_start_ts``, taken after the post-forward synchronize;
last-stage intervals end at the record's wall-clock ``timestamp`` converted to
the monotonic clock with the offset the driver sampled around the round.  A
Frontier forward row is one ``ATTN_DP_LANE`` ledger row.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from tests.e2e.stage_admission_matrix import (
    ADMISSION_DEADLOCK,
    ATTN_DP_LANE,
    SUCCESS,
    interval_overlap,
    matrix_root,
    read_ledger,
)

MODELS = ("moe", "dense")
BURSTS = (8, 16)
CO_START_BOUND = 0.5
CO_EXECUTION_BOUND = 0.10
# Dense DP ranks meet once per forward, in the DP metadata all-reduce that
# vLLM runs after ``forward_start_ts``.  A rank's recorded interval therefore
# includes its wait for the other rank (start offsets) as well as its own
# duration variation (end offsets); the dummy predictor models neither, so
# dense co-execution is reported, not gated (plan §4.7 V5, D-9).  MoE ranks
# stay aligned by EP collectives.
CO_EXECUTION_GATED = {"moe": True, "dense": False}
INFORMATIONAL = "INFORMATIONAL"
HOLDS, LOST = "HOLDS", "LOST"
FRONTIER_OWNER = "frontier/scheduler/replica_stage_scheduler/stage_execution_context.py"


def _forward(lane: int, stage: int, start: float, end: float, indices) -> dict:
    return {"lane": lane, "stage": stage, "start": start, "end": end,
            "indices": tuple(sorted(indices))}


def vllm_forwards(scenario_dir: Path) -> dict[tuple[int, int], dict]:
    """Formal vLLM forwards keyed by ``(burst size, round)``."""
    requests = {row["request_id"]: row for row in map(json.loads, (scenario_dir / "requests.jsonl").read_text().splitlines())}
    summary = json.loads((scenario_dir / "summary.json").read_text())
    offsets = {
        entry["label"]: (entry["wall_minus_monotonic_before"] + entry["wall_minus_monotonic_after"]) / 2
        for entry in summary["rounds"]
    }
    runs: dict[tuple[int, int], dict] = {}
    for line in (scenario_dir / "pp_boundary.jsonl").read_text().splitlines():
        record = json.loads(line)
        members = [requests[request_id] for request_id in record["request_ids"]]
        labels = {member["burst"] for member in members}
        ranks = {member["rank"] for member in members}
        if len(labels) != 1 or len(ranks) != 1:
            raise ValueError(f"forward mixes bursts or ranks: {record['request_ids']}")
        label = labels.pop()
        if label == "warmup":
            continue
        if record["is_last_rank"]:
            end = record["timestamp"] - offsets[label]
        else:
            end = record["send_start_ts"]
        member = members[0]
        run = runs.setdefault(
            (int(label.split("-")[0][1:]), member["round"]),
            {"forwards": [], "requests": [row for row in requests.values() if row["burst"] == label]},
        )
        run["forwards"].append(_forward(ranks.pop(), record["pp_rank"], record["forward_start_ts"], end,
                                        (m["index"] for m in members)))
    for run in runs.values():
        rows = run.pop("requests")
        run["submitted"] = len(rows)
        run["completed"] = sum(1 for row in rows if row["num_output_tokens"] == 1)
    return runs


def vllm_placement(scenario_dir: Path) -> dict:
    """Check from engine iterations that each formal request ran on its pinned rank."""
    pinned = {row["request_id"]: row["rank"] for row in map(json.loads, (scenario_dir / "requests.jsonl").read_text().splitlines())}
    scheduled_by = defaultdict(set)
    for path in sorted((scenario_dir / "dp_placement").glob("*.jsonl")):
        for record in map(json.loads, path.read_text().splitlines()):
            if record["kind"] == "engine_iteration":
                for request_id in record["scheduled_new_req_ids"]:
                    scheduled_by[request_id].add(record["engine"])
    misplaced = sorted(rid for rid, rank in pinned.items() if scheduled_by.get(rid, {rank}) != {rank})
    unseen = sorted(rid for rid in pinned if rid not in scheduled_by)
    return {"requests": len(pinned), "misplaced": misplaced, "unseen": unseen,
            "ok": not misplaced and not unseen}


def frontier_run(set_dir: Path, case_id: str) -> dict:
    case_dir = set_dir / case_id
    run = json.loads((case_dir / "run.json").read_text())
    case = json.loads((case_dir / "case.json").read_text())
    result = {"outcome": run["outcome"], "submitted": case["num_requests"], "forwards": []}
    if run["outcome"] != SUCCESS:
        result["completed"] = None
        return result
    result["completed"] = case["num_requests"]
    for row in read_ledger(case_dir / "metrics"):
        if row["execution_scope"] != ATTN_DP_LANE:
            continue
        result["forwards"].append(_forward(row["replica_local_id"], row["stage_id"], row["stage_start_ts"],
                                           row["stage_end_ts"], (int(rid) for rid in row["request_ids"])))
    lanes_match_index = all(index % 2 == forward["lane"] for forward in result["forwards"]
                            for index in forward["indices"])
    result["placement_ok"] = lanes_match_index
    return result


def lane_metrics(forwards: list[dict]) -> dict:
    """M2–M5 of plan §4.7 from one run's forwards."""
    by_lane_stage = defaultdict(list)
    for forward in forwards:
        by_lane_stage[(forward["lane"], forward["stage"])].append(forward)
    sequences = {
        f"lane{lane}/stage{stage}": [list(f["indices"]) for f in sorted(rows, key=lambda f: f["start"])]
        for (lane, stage), rows in sorted(by_lane_stage.items())
    }
    metrics = {"M2_sequences": sequences}
    for stage in sorted({forward["stage"] for forward in forwards}):
        lane0 = sorted(by_lane_stage[(0, stage)], key=lambda f: f["start"])
        lane1 = sorted(by_lane_stage[(1, stage)], key=lambda f: f["start"])
        pairing = []
        for forward in lane0:
            overlaps = [(min(forward["end"], other["end"]) - max(forward["start"], other["start"]), other)
                        for other in lane1]
            overlap, partner = max(overlaps, key=lambda item: item[0], default=(0.0, None))
            pairing.append([list(forward["indices"]), list(partner["indices"]) if partner and overlap > 0 else None])
        durations = [f["end"] - f["start"] for f in lane0 + lane1]
        skew = abs(lane0[0]["start"] - lane1[0]["start"]) / statistics.median(durations) if lane0 and lane1 else None
        overlap = interval_overlap([(f["start"], f["end"], f["lane"]) for f in lane0 + lane1])
        metrics[f"stage{stage}"] = {
            "M3_pairing": pairing,
            "M4_co_start": skew,
            "M5_co_execution": overlap["multi_lane_busy_time"] / overlap["busy_time"] if overlap["busy_time"] else None,
            "median_forward_duration": statistics.median(durations) if durations else None,
            "self_overlap": overlap["self_overlap"],
        }
    return metrics


def _row(check, model, burst, round_index, metric, groundtruth, after, base, status, note=""):
    return {"check": check, "model": model, "burst": burst, "round": round_index, "metric": metric,
            "groundtruth": json.dumps(groundtruth), "frontier_after": json.dumps(after),
            "frontier_base": json.dumps(base), "status": status,
            "frontier_owner": FRONTIER_OWNER if status == "MISMATCH" else "", "note": note}


def compare(vllm_run: Path, frontier_root: Path, before: str, after: str) -> tuple[list[dict], dict]:
    rows, details = [], {"placement": {}, "runs": {}}
    for model in MODELS:
        scenario_dir = vllm_run / "runs" / model
        vllm_runs = vllm_forwards(scenario_dir)
        details["placement"][model] = vllm_placement(scenario_dir)
        for burst in BURSTS:
            case_id = f"G7-{model}-dp2-pp2-n{burst}"
            base = frontier_run(frontier_root / before, case_id)
            new = frontier_run(frontier_root / after, case_id)
            base_metrics = lane_metrics(base["forwards"]) if base["outcome"] == SUCCESS else None
            new_metrics = lane_metrics(new["forwards"]) if new["outcome"] == SUCCESS else None
            rounds = sorted(r for (b, r) in vllm_runs if b == burst)
            vllm_metrics = {r: lane_metrics(vllm_runs[(burst, r)]["forwards"]) for r in rounds}
            details["runs"][case_id] = {"frontier_base": base_metrics, "frontier_after": new_metrics,
                                        "frontier_base_outcome": base["outcome"],
                                        "frontier_after_outcome": new["outcome"],
                                        "frontier_after_placement_ok": new.get("placement_ok"),
                                        "vllm": vllm_metrics}
            base_m4 = base_metrics["stage0"]["M4_co_start"] if base_metrics else None
            # Negative controls: the base rule deadlocks MoE and starts dense
            # lanes one forward apart.  They describe the base, not vLLM.
            if model == "moe":
                rows.append(_row("N1", model, burst, "base", "base outcome", None, new["outcome"],
                                 base["outcome"], HOLDS if base["outcome"] == ADMISSION_DEADLOCK else LOST,
                                 note=f"expected {ADMISSION_DEADLOCK}"))
            else:
                rows.append(_row("N4", model, burst, "base", "M4 stage-0 co-start", None,
                                 new_metrics["stage0"]["M4_co_start"] if new_metrics else None, base_m4,
                                 HOLDS if base_m4 is not None and base_m4 >= CO_START_BOUND else LOST,
                                 note=f"expected >= {CO_START_BOUND}"))
            for r in rounds:
                run = vllm_runs[(burst, r)]
                completed = run["completed"] == run["submitted"] == burst
                status = "MATCH" if completed and new["outcome"] == SUCCESS else "MISMATCH"
                rows.append(_row("V1", model, burst, r, "M1 completion",
                                 f"{run['completed']}/{run['submitted']}", new["outcome"], base["outcome"], status))
                gt = vllm_metrics[r]
                after_m2 = new_metrics["M2_sequences"] if new_metrics else None
                rows.append(_row("V2", model, burst, r, "M2 lane sequences", gt["M2_sequences"], after_m2,
                                 base_metrics["M2_sequences"] if base_metrics else None,
                                 "MATCH" if after_m2 == gt["M2_sequences"] else "MISMATCH"))
                after_m3 = new_metrics["stage0"]["M3_pairing"] if new_metrics else None
                rows.append(_row("V3", model, burst, r, "M3 stage-0 pairing", gt["stage0"]["M3_pairing"], after_m3,
                                 base_metrics["stage0"]["M3_pairing"] if base_metrics else None,
                                 "MATCH" if after_m3 == gt["stage0"]["M3_pairing"] else "MISMATCH"))
                after_m4 = new_metrics["stage0"]["M4_co_start"] if new_metrics else None
                m4_ok = (gt["stage0"]["M4_co_start"] < CO_START_BOUND and after_m4 is not None
                         and after_m4 < CO_START_BOUND)
                rows.append(_row("V4", model, burst, r, "M4 stage-0 co-start", gt["stage0"]["M4_co_start"],
                                 after_m4, base_m4, "MATCH" if m4_ok else "MISMATCH"))
            gt_m5 = statistics.mean(vllm_metrics[r]["stage0"]["M5_co_execution"] for r in rounds)
            after_m5 = new_metrics["stage0"]["M5_co_execution"] if new_metrics else None
            base_m5 = base_metrics["stage0"]["M5_co_execution"] if base_metrics else None
            if CO_EXECUTION_GATED[model]:
                m5_ok = after_m5 is not None and abs(after_m5 - gt_m5) <= CO_EXECUTION_BOUND
                m5_status = "MATCH" if m5_ok else "MISMATCH"
            else:
                m5_status = INFORMATIONAL
            rows.append(_row("V5", model, burst, "mean", "M5 stage-0 co-execution", gt_m5, after_m5, base_m5,
                             m5_status))
    return rows, details


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--vllm-run", type=Path, required=True, help="evidence directory of one worker run")
    parser.add_argument("--before", default="base")
    parser.add_argument("--after", default="after")
    parser.add_argument("--output", type=Path, required=True, help="case analysis directory")
    args = parser.parse_args(argv)

    rows, details = compare(args.vllm_run, matrix_root(), args.before, args.after)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "workflow_gap_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "lane_metrics.json").write_text(json.dumps(details, indent=1, sort_keys=True))
    mismatches = [row for row in rows if row["status"] == "MISMATCH"]
    placement_ok = all(p["ok"] for p in details["placement"].values())
    placement_unseen = sum(len(p["unseen"]) for p in details["placement"].values())
    controls = [row for row in rows if row["status"] in (HOLDS, LOST)]
    status = {
        "analysis_state": "COMPLETE",
        "status": "PASS" if not mismatches and placement_ok else "FAIL",
        "correction_state": "not_applicable",
        "rows": len(rows),
        "mismatches": len(mismatches),
        "vllm_placement_ok": placement_ok,
        "vllm_placement_unseen_requests": placement_unseen,
        "negative_control_holds": all(row["status"] == HOLDS for row in controls),
        "next_action": ("record C7 in the test report" if not mismatches and placement_ok
                        else "report each MISMATCH row with its cause before P4; adjust nothing"),
    }
    (args.output / "workflow_gap_status.json").write_text(json.dumps(status, indent=1))
    for row in rows:
        print(f"{row['check']} {row['model']:<5} n{row['burst']:<3} r{row['round']!s:<5} {row['status']:<9} "
              f"gt={row['groundtruth'][:40]} after={row['frontier_after'][:40]} base={row['frontier_base'][:40]}")
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
