"""Split vLLM stage-0 non-overlap into start and end offsets of paired forwards.

Pairs are the i-th lane-0 and i-th lane-1 stage-0 forwards of a round; the
output states whether this equals the M3 pairing of ``compare_lanes``.  For an
overlapping pair, union minus overlap equals |start difference| + |end
difference|, so the non-overlap splits into a start part and an end part.  A
disjoint pair has no such split; it is counted and its non-overlap is
reported whole.

Two counterfactual co-execution fractions are reported:

* ``M5_equal_durations``: every lane-1 forward keeps its start and takes the
  duration of its lane-0 partner, as with the dummy predictor's equal
  durations.
* ``M5_barrier_aligned``: both forwards of a pair start at the later of the
  two starts and keep their own ends.  vLLM 0.10.2 without CUDA graphs runs
  the per-forward DP metadata all-reduce inside ``set_forward_context``, after
  ``forward_start_ts``, so neither rank computes before the later one arrives.
  The traces carry no timestamp after that exchange, so this value is derived
  from where the all-reduce sits, not measured.  It is ``None`` when a round
  has a disjoint pair.

Usage: python decompose_co_execution.py <vllm run dir> <model> [<model> ...]
"""
import json
import statistics
import sys
from pathlib import Path

from tests.comparison.stage_admission_pp.compare_lanes import lane_metrics, vllm_forwards
from tests.e2e.stage_admission_matrix import interval_overlap

run_dir = Path(sys.argv[1])
summary = {}
for model in sys.argv[2:]:
    runs = vllm_forwards(run_dir / "runs" / model)
    for (burst, round_index), run in sorted(runs.items()):
        stage0 = [f for f in run["forwards"] if f["stage"] == 0]
        lanes = {lane: sorted((f for f in stage0 if f["lane"] == lane), key=lambda f: f["start"]) for lane in (0, 1)}
        pairs = list(zip(lanes[0], lanes[1]))
        start_part = end_part = disjoint_part = 0.0
        disjoint = 0
        equal_duration, barrier_aligned = [], []
        for first, second in pairs:
            later_start = max(first["start"], second["start"])
            if min(first["end"], second["end"]) > later_start:
                start_part += abs(first["start"] - second["start"])
                end_part += abs(first["end"] - second["end"])
            else:
                disjoint += 1
                disjoint_part += (first["end"] - first["start"]) + (second["end"] - second["start"])
            equal_duration.append((first["start"], first["end"], 0))
            equal_duration.append((second["start"], second["start"] + first["end"] - first["start"], 1))
            barrier_aligned.append((later_start, first["end"], 0))
            barrier_aligned.append((later_start, second["end"], 1))
        observed = interval_overlap([(f["start"], f["end"], f["lane"]) for f in stage0])
        equal = interval_overlap(equal_duration)
        aligned = interval_overlap(barrier_aligned)
        m3 = lane_metrics(run["forwards"])["stage0"]["M3_pairing"]
        durations = [f["end"] - f["start"] for f in stage0]
        summary[f"{model}/n{burst}/r{round_index}"] = {
            "pairs": len(pairs),
            "unpaired_forwards": abs(len(lanes[0]) - len(lanes[1])),
            "pairing_equals_M3": m3 == [[list(a["indices"]), list(b["indices"])] for a, b in pairs],
            "disjoint_pairs": disjoint,
            "M5_observed": round(observed["multi_lane_busy_time"] / observed["busy_time"], 4),
            "M5_equal_durations": round(equal["multi_lane_busy_time"] / equal["busy_time"], 4),
            "M5_barrier_aligned": (round(aligned["multi_lane_busy_time"] / aligned["busy_time"], 4)
                                   if not disjoint else None),
            "non_overlap_ms_from_start_offsets": round(1e3 * start_part, 3),
            "non_overlap_ms_from_end_offsets": round(1e3 * end_part, 3),
            "non_overlap_ms_of_disjoint_pairs": round(1e3 * disjoint_part, 3),
            "stage0_duration_ms_median": round(1e3 * statistics.median(durations), 3),
            "stage0_duration_ms_cv": round(statistics.pstdev(durations) / statistics.mean(durations), 3),
        }
print(json.dumps(summary, indent=1))
