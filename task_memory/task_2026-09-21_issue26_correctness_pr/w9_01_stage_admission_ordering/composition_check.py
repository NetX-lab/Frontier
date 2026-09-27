"""Evaluate K1-K3 of the W9-01 composition check (plan §18.14).

K1: every case of the composed set succeeds and conserves requests and tokens.
K2 (EXPLAIN rows only; the harness ``compare`` settles the rest): every
    (cluster, replica, stage, lane) runs the same ordered batches with the same
    component durations in both sets, so only start times differ.
K3: every Poisson cell with ``attn_dp > 1`` places batches on all lanes of each
    MONOLITHIC and PREFILL stage; the reference set is reported beside it.

Usage: python composition_check.py <before set> <composed set> <reference set>
       <compare json> <output json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from tests.e2e.stage_admission_matrix import (
    ATTN_DP_LANE, SUCCESS, _conservation, build_cases, ledger_lane_metric, matrix_root, read_ledger,
)

GROUPS = ("G3b", "G9", "G10")
LANE_CLUSTERS = ("MONOLITHIC", "PREFILL")

before_set, composed_set, reference_set = sys.argv[1:4]
compare_rows = json.loads(Path(sys.argv[4]).read_text())
output = Path(sys.argv[5])
root = matrix_root()
cases = [case for case in build_cases() if case.group in GROUPS]


def metrics_dir(set_name, case_id):
    return root / set_name / case_id / "metrics"


def outcome(set_name, case_id):
    return json.loads((root / set_name / case_id / "run.json").read_text())["outcome"]


def lane_batches(set_name, case_id):
    rows = defaultdict(list)
    for row in read_ledger(metrics_dir(set_name, case_id)):
        if row["execution_scope"] == ATTN_DP_LANE:
            key = (row["cluster_type"], row["replica_id"], row["stage_id"], row["replica_local_id"])
            rows[key].append(row)
    return {
        str(key): [(tuple(row["request_ids"]), round(row["stage_end_ts"] - row["stage_start_ts"], 9),
                    json.dumps(row["execution_time"], sort_keys=True))
                   for row in sorted(stage_rows, key=lambda row: row["stage_start_ts"])]
        for key, stage_rows in rows.items()
    }


k1 = {}
for case in cases:
    state = outcome(composed_set, case.case_id)
    k1[case.case_id] = {"outcome": state}
    if state == SUCCESS:
        k1[case.case_id]["conservation"] = _conservation(case, metrics_dir(composed_set, case.case_id))
k1_pass = all(row["outcome"] == SUCCESS and row["conservation"]["ok"] for row in k1.values())

def first_divergence(case_id):
    """First differing row of the two time-ordered ledgers.

    Passes when it is the same batch on the same stage and lane, admitted
    earlier in the composed set: later batch differences on that lane then
    follow from online arrivals meeting an earlier lane schedule.
    """
    def ordered(set_name):
        rows = [row for row in read_ledger(metrics_dir(set_name, case_id))
                if row["execution_scope"] == ATTN_DP_LANE]
        rows.sort(key=lambda row: (row["stage_start_ts"], row["stage_id"], row["replica_local_id"]))
        return [((row["cluster_type"], row["replica_id"], row["stage_id"], row["replica_local_id"],
                  tuple(row["request_ids"]), round(row["stage_end_ts"] - row["stage_start_ts"], 9),
                  json.dumps(row["execution_time"], sort_keys=True)), row["stage_start_ts"])
                for row in rows]

    before, after = ordered(before_set), ordered(composed_set)
    index = next(i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1])
    first_after = after[index]
    held = next((start for work, start in before[index:] if work == first_after[0]), None)
    return {
        "position": index,
        "batch": list(first_after[0][:5]),
        "start_composed": first_after[1],
        "start_before": held,
        "earlier_admission_of_same_batch": held is not None and first_after[1] < held,
    }


k2 = {}
for row in compare_rows:
    if row["verdict"] != "EXPLAIN":
        continue
    before = lane_batches(before_set, row["case_id"])
    after = lane_batches(composed_set, row["case_id"])
    entry = {
        "same_batches_and_component_durations": before == after,
        "identical_lanes": sorted(key for key in before if before[key] == after.get(key)),
    }
    if before != after:
        entry["first_divergence"] = first_divergence(row["case_id"])
    entry["pass"] = (entry["same_batches_and_component_durations"]
                     or entry["first_divergence"]["earlier_admission_of_same_batch"])
    k2[row["case_id"]] = entry
k2_pass = all(row["pass"] for row in k2.values())


def stage_lanes(set_name, case_id):
    return {stage: metric["lanes"] for stage, metric in ledger_lane_metric(metrics_dir(set_name, case_id)).items()}


k3 = {}
for case in cases:
    if case.arrival != "poisson" or case.attn_dp == 1:
        continue
    composed = stage_lanes(composed_set, case.case_id)
    reference = (stage_lanes(reference_set, case.case_id)
                 if outcome(reference_set, case.case_id) == SUCCESS else None)
    covered = all(lanes == list(range(case.attn_dp))
                  for stage, lanes in composed.items() if stage.split("/")[0] in LANE_CLUSTERS)
    k3[case.case_id] = {"all_lanes": covered, "composed": composed, "reference": reference}
k3_pass = all(row["all_lanes"] for row in k3.values())

report = {
    "sets": {"before": before_set, "composed": composed_set, "reference": reference_set},
    "K1": {"pass": k1_pass, "cases": k1},
    "K2_explain": {"pass": k2_pass, "cases": k2},
    "K3": {"pass": k3_pass, "cases": k3},
}
output.write_text(json.dumps(report, indent=1, sort_keys=True))
print("K1", k1_pass, len(k1), "K2 explain", k2_pass, len(k2), "K3", k3_pass, len(k3))
for case_id, row in k3.items():
    lane_view = lambda stages: ({stage: lanes for stage, lanes in stages.items()
                                 if stage.split("/")[0] in LANE_CLUSTERS} if stages else stages)
    print(case_id, "composed", lane_view(row["composed"]), "reference", lane_view(row["reference"]))
