"""Split vLLM stage-0 non-overlap into start and end offsets of paired forwards.

For each M3 pair (lane-0 forward, lane-1 forward) of each round, the union
minus the overlap equals |start difference| + |end difference|.  The start
part is where admission could act; the end part comes from per-rank forward
duration variation.  Also reports M5 with every lane-1 end aligned to the
lane-0 duration, i.e. the co-execution vLLM would show with the equal
per-forward durations of Frontier's dummy predictor.

Usage: python decompose_co_execution.py <vllm run dir> <model> [<model> ...]
"""
import json
import statistics
import sys
from pathlib import Path

from tests.comparison.stage_admission_pp.compare_lanes import vllm_forwards
from tests.e2e.stage_admission_matrix import interval_overlap

run_dir = Path(sys.argv[1])
summary = {}
for model in sys.argv[2:]:
    runs = vllm_forwards(run_dir / "runs" / model)
    for (burst, round_index), run in sorted(runs.items()):
        stage0 = [f for f in run["forwards"] if f["stage"] == 0]
        lanes = {lane: sorted((f for f in stage0 if f["lane"] == lane), key=lambda f: f["start"]) for lane in (0, 1)}
        start_part = end_part = 0.0
        equal_duration = []
        for first, second in zip(lanes[0], lanes[1]):
            start_part += abs(first["start"] - second["start"])
            end_part += abs(first["end"] - second["end"])
            equal_duration.append((first["start"], first["end"], 0))
            equal_duration.append((second["start"], second["start"] + first["end"] - first["start"], 1))
        observed = interval_overlap([(f["start"], f["end"], f["lane"]) for f in stage0])
        aligned = interval_overlap(equal_duration)
        durations = [f["end"] - f["start"] for f in stage0]
        summary[f"{model}/n{burst}/r{round_index}"] = {
            "pairs": min(len(lanes[0]), len(lanes[1])),
            "M5_observed": round(observed["multi_lane_busy_time"] / observed["busy_time"], 4),
            "M5_equal_durations": round(aligned["multi_lane_busy_time"] / aligned["busy_time"], 4),
            "non_overlap_ms_from_start_offsets": round(1e3 * start_part, 3),
            "non_overlap_ms_from_end_offsets": round(1e3 * end_part, 3),
            "stage0_duration_ms_median": round(1e3 * statistics.median(durations), 3),
            "stage0_duration_ms_cv": round(statistics.pstdev(durations) / statistics.mean(durations), 3),
        }
print(json.dumps(summary, indent=1))
