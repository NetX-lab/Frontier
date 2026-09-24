"""Step 9 P1(b): score report-key candidates against forward membership.

Reads the probe files written by ``probe_boundaries.py``. For every report the
opt-in policy would emit under the settled D9-1 rule (an admission is
published on its own only while ``running_after < stages``; otherwise it is
folded into the lane's next completion report), it derives the report's
*target*: the stage-0 forward group of the iteration the report describes.

* admission published on its own: the group the admitted batch ran in;
* completion with a folded admission: the folded batch's group;
* completion without one (``completion_only``): the lane launches no forward
  of its own here (the reference runs a dummy forward), so the target is the
  lane's next Frontier forward: the first group after the lane's newest batch
  in which the lane takes part after the boundary, with a real or an idle
  batch.

Targets are Frontier's own forward grouping, read after the fact. Each
candidate key is read at the boundary and scored in two ways:

1. Pairwise order against the target, split into peer splits (equal target,
   different key: invariant I1), merges (different targets, equal key:
   I5 is the one-lane case) and inversions.
2. Replay: the reports, keyed by the candidate, drive a fresh
   ``VllmDPLoadBalancer`` and the frontend-visible counts are sampled every
   millisecond; the score is the number of milliseconds in which they differ
   from the replay keyed by the target.

Candidates:

* ``A``: ``ForwardSyncState._next_step_id_by_replica`` (the plan's first);
* ``lane_counter``: the lane's number of earlier reports;
* ``next_group``: the stage-0 context's next forward-group id;
* ``predicted_group``: the group a batch admitted now would join, from
  stage-0 state and the lane's admitted batches that have not started stage 0
  (for a completion with a folded admission, the value captured at that
  admission);
* ``group_anchored``: the same group from state a policy can hold:
  ``max(open-or-next stage-0 group, lane's last admitted key + 1)``, where the
  first term is the bound group while it is unsealed, else the next group id.
  An admission stores its key as the lane's last admitted key; a completion
  with a folded admission reports the key held for it; a completion without
  one reads the formula and stores nothing;
* ``every_report_advances``: the same formula, but a completion without a
  folded admission also stores its key (a per-iteration increment like the
  reference's ``step_counter``).

Usage: python analyze_keys.py <probe dir> <output json>
"""
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

from frontier.scheduler.request_load import RequestLoad
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer

CANDIDATES = ("A", "lane_counter", "next_group", "predicted_group", "group_anchored",
              "every_report_advances")


def stage0_membership(records):
    group_of_batch = {}
    rooms_by_lane = defaultdict(list)
    for record in records:
        if record["kind"] == "stage_start" and record["stage"] == 0:
            group_of_batch[record["batch"]] = record["group"]
        if record["kind"] == "room" and record["stage"] == 0 and record["group"] is not None:
            rooms_by_lane[record["lane"]].append((record["seq"], record["group"]))
    start_seq = {record["batch"]: record["seq"] for record in records
                 if record["kind"] == "stage_start" and record["stage"] == 0}
    return group_of_batch, rooms_by_lane, start_seq


def open_or_next_group(record):
    if record["stage0_bound_group"] is not None and not record["stage0_sealed"]:
        return record["stage0_bound_group"]
    return record["stage0_next_group"]


def predicted_group(record, admitted_by_lane, start_seq):
    """Group a batch admitted at this boundary would join, from stage-0 state."""
    not_started_ahead = sum(
        1 for batch, _ in admitted_by_lane[record["lane"]]
        if batch != record["batch"] and start_seq.get(batch, float("inf")) > record["seq"]
    )
    open_group = (record["stage0_bound_group"] is not None
                  and not record["stage0_sealed"] and not record["lane_stage0_busy"])
    first = record["stage0_bound_group"] if open_group else record["stage0_next_group"]
    return first + not_started_ahead


def reports(result):
    records = result["records"]
    group_of_batch, rooms_by_lane, start_seq = stage0_membership(records)
    admitted_by_lane = defaultdict(list)
    folded = {}
    newest_group = defaultdict(lambda: -1)
    last_admitted_key = defaultdict(lambda: -1)
    last_report_key = defaultdict(lambda: -1)
    count_by_lane = defaultdict(int)
    rows = []
    for record in records:
        if record["kind"] not in ("admit", "complete"):
            continue
        lane = record["lane"]
        anchored = max(open_or_next_group(record), last_admitted_key[lane] + 1)
        advanced = max(open_or_next_group(record), last_report_key[lane] + 1)
        if record["kind"] == "admit":
            admitted_by_lane[lane].append((record["batch"], record["seq"]))
            key = predicted_group(record, admitted_by_lane, start_seq)
            last_admitted_key[lane] = anchored
            last_report_key[lane] = advanced
            newest_group[lane] = max(newest_group[lane], group_of_batch.get(record["batch"], -1))
            if record["running_after"] < record["stages"]:
                row = {"kind": "admission", "target": group_of_batch.get(record["batch"]),
                       "predicted_group": key, "group_anchored": anchored,
                       "every_report_advances": advanced}
            else:
                folded[lane] = (record["batch"], key, anchored, advanced)
                continue
        else:
            if lane in folded:
                batch, key, held, held_advanced = folded.pop(lane)
                row = {"kind": "completion_folded", "target": group_of_batch.get(batch),
                       "predicted_group": key, "group_anchored": held,
                       "every_report_advances": held_advanced, "folded_batch": batch}
            else:
                later = [group for seq, group in rooms_by_lane.get(lane, [])
                         if seq > record["seq"] and group > newest_group[lane]]
                last_report_key[lane] = advanced
                row = {"kind": "completion_only", "target": min(later) if later else None,
                       "predicted_group": predicted_group(record, admitted_by_lane, start_seq),
                       "group_anchored": anchored, "every_report_advances": advanced}
        row.update(seq=record["seq"], time=record["time"], lane=lane, batch=record["batch"],
                   load=record["load"], A=record["replica_forward_id"],
                   lane_counter=count_by_lane[lane], next_group=record["stage0_next_group"])
        count_by_lane[lane] += 1
        rows.append(row)
    return rows


def pairwise(rows, name, kinds=("admission", "completion_folded", "completion_only")):
    scored = [row for row in rows if row["target"] is not None and row["kind"] in kinds]
    splits, merges, inversions, merge_same_lane = [], [], [], 0
    for i, first in enumerate(scored):
        for second in scored[i + 1:]:
            target = (first["target"] > second["target"]) - (first["target"] < second["target"])
            key = (first[name] > second[name]) - (first[name] < second[name])
            if target == key:
                continue
            pair = (first["seq"], second["seq"])
            if target == 0:
                splits.append(pair)
            elif key == 0:
                merges.append(pair)
                merge_same_lane += first["lane"] == second["lane"]
            else:
                inversions.append(pair)
    return {"peer_splits": len(splits), "merges": len(merges), "merges_same_lane": merge_same_lane,
            "inversions": len(inversions), "examples": {"peer_splits": splits[:3], "merges": merges[:3],
                                                        "inversions": inversions[:3]}}


def frontend_timeline(rows, name, num_lanes, end_ms):
    balancer = VllmDPLoadBalancer(num_lanes)
    timeline = []
    pending = [row for row in rows if row[name] is not None]
    index = 0
    for now_ms in range(end_ms + 1):
        while index < len(pending) and int(pending[index]["time"] * 1000) <= now_ms:
            row = pending[index]
            balancer.report(row["time"], row["lane"], row[name], RequestLoad(*row["load"]))
            index += 1
        view = copy.deepcopy(balancer)
        view._advance(now_ms / 1000)
        timeline.append(tuple(view.frontend_counts))
    return timeline


def analyze(path: Path) -> dict:
    result = json.loads(path.read_text())
    rows = reports(result)
    num_lanes = result["case"]["attn_dp"]
    end_ms = int(max(record["time"] for record in result["records"]) * 1000) + 200
    reference = frontend_timeline(
        [dict(row, target_key=row["target"]) for row in rows if row["target"] is not None],
        "target_key", num_lanes, end_ms)
    scored_rows = [row for row in rows if row["target"] is not None]
    summary = {
        "shape": result["shape"], "completed": result["completed"], "requests": result["requests"],
        "reports": {kind: sum(row["kind"] == kind for row in rows)
                    for kind in ("admission", "completion_folded", "completion_only")},
        "unscored_reports": len(rows) - len(scored_rows),
        "candidates": {},
    }
    for name in CANDIDATES:
        timeline = frontend_timeline(scored_rows, name, num_lanes, end_ms)
        forward_rows = pairwise(rows, name, kinds=("admission", "completion_folded"))
        summary["candidates"][name] = {
            **pairwise(rows, name),
            "forward_reports": {key: forward_rows[key] for key in ("peer_splits", "merges", "inversions")},
            "equal_to_target": sum(row[name] == row["target"] for row in scored_rows),
            "frontend_mismatch_ms": sum(a != b for a, b in zip(timeline, reference)),
        }
    summary["rows"] = rows
    return summary


if __name__ == "__main__":
    probe_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
    summaries = [analyze(path) for path in sorted(probe_dir.glob("*/*.json"))]
    output.write_text(json.dumps(summaries, indent=1))
    print(f"{'shape':22} {'adm/fold/only':>13} {'candidate':15} {'=target':>7} "
          f"{'split/merge/inv':>15} {'forward rows':>12} {'fe_ms':>6}")
    for summary in summaries:
        counts = "/".join(str(value) for value in summary["reports"].values())
        for name, score in summary["candidates"].items():
            scored = len(summary["rows"]) - summary["unscored_reports"]
            forward = "/".join(str(value) for value in score["forward_reports"].values())
            print(f"{summary['shape']:22} {counts:>13} {name:15} {score['equal_to_target']:>3}/{scored:<3} "
                  f"{score['peer_splits']:>5}/{score['merges']}/{score['inversions']:<5} {forward:>12} "
                  f"{score['frontend_mismatch_ms']:>6}")
