"""Native D step durations from the lifecycle runs' scheduler records: the k=1 step against decode-only steps.

Usage: native_k1_step.py <task directory>

On each D engine core, step s starts when its schedule has returned and the previous step's
update has been recorded (the GPU runs the queued steps back to back), and ends at the update
of step s. The k=1 step of a formal request is the step whose schedule promotes it ("new",
6143 computed + 1 scheduled token); it carries decodes when that schedule also has cached
(running) requests. Decode-only steps are formal-window steps whose schedule promotes no
request and has cached requests. The profiled step-cost tables price the same steps; this
output is native and unprofiled. Times are in ms.
"""
import statistics
import sys
from pathlib import Path

RUNS = (
    ("H800l", "h4b_lifecycle_measurement/inputs/h800l"),
    ("P2-32l", "h200_mainline/diagnostic/inputs/p2_32l"),
    ("P2-64l", "h200_mainline/diagnostic/inputs/p2_64l"),
)
FORMAL = "-formal-"


def step_durations(rows):
    """Per step number: (duration, promoted request ids, cached request count)."""
    schedules = {row["step"]: row for row in rows if row["event"] == "schedule"}
    updates = {row["step"]: row for row in rows if row["event"] == "update"}
    steps = {}
    for step, update in updates.items():
        schedule = schedules.get(step)
        if schedule is None or step - 1 not in updates:
            continue
        start = max(schedule["mono"] + schedule["dur_ms"] / 1e3, updates[step - 1]["mono"])
        promoted = [request_id for request_id, _, _ in schedule.get("new", ())]
        steps[step] = ((update["mono"] - start) * 1e3, promoted, schedule.get("cached", 0))
    return steps


def describe(values):
    return (f"n {len(values)}  median {statistics.median(values):.2f}  mean {statistics.fmean(values):.2f}"
            if values else "n 0")


def main() -> None:
    root = Path(sys.argv[1])
    sys.path.insert(0, str(root / "h4b_lifecycle_measurement"))
    import analyze_lifecycle

    for name, relative in RUNS:
        problems = []
        cores = analyze_lifecycle.load_cores(root / relative, problems)
        k1_mixed, k1_alone, decode_only = [], [], []
        for (role, _), core in sorted(cores.items()):
            if role != "decode":
                continue
            steps = step_durations(core["rows"])
            formal = [step for step, (_, promoted, _) in steps.items() if any(FORMAL in r for r in promoted)]
            window = range(min(formal), max(formal) + 1)
            for step, (duration, promoted, cached) in steps.items():
                if promoted and all(FORMAL in r for r in promoted):
                    (k1_mixed if cached else k1_alone).append(duration)
                elif not promoted and cached and step in window:
                    decode_only.append(duration)
        print(f"== {name}: problems {problems}")
        print(f"  k=1 step with decodes: {describe(k1_mixed)}")
        print(f"  k=1 step alone:        {describe(k1_alone)}")
        print(f"  decode-only step:      {describe(decode_only)}")


if __name__ == "__main__":
    main()
