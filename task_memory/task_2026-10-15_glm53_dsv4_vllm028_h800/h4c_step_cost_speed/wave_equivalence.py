"""H4c check 1: every table-priced EP wave equals the full lane-prediction path (scratch).

Runs the H4a run5 input for a bounded wall time. Each wave that takes the table-priced
branch is recomputed with lane timings requested, which forces the full path, and the
participants, the six phase tuples and the barrier timing are compared bit for bit.
"""
import json
import signal
import sys
import time
from pathlib import Path

REPO = Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028")
sys.path.insert(0, str(REPO / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic"))
import h4a4_driver  # noqa: E402
from frontier.utils.random import set_seeds  # noqa: E402
import frontier.scheduler.cluster_scheduler.base_cluster_scheduler as bcs  # noqa: E402

root, wall_s = Path(sys.argv[1]), int(sys.argv[2])
original = bcs.prepare_moe_wave_from_inputs
summary = {"table_priced_waves": 0, "full_path_waves": 0, "mismatches": [], "by_cluster": {}}


def bits(values):
    return [float(v).hex() for v in values]


def checked(**kwargs):
    plan = original(**kwargs)
    if plan.layer_workload is not None:
        summary["full_path_waves"] += 1
        return plan
    summary["table_priced_waves"] += 1
    name = kwargs["cluster_type"].name
    summary["by_cluster"][name] = summary["by_cluster"].get(name, 0) + 1
    full = original(**{**kwargs, "capture_lane_timings": True})
    same = (
        plan.participant_ep_ids == tuple(full.layer_workload.participant_ep_ids) == full.participant_ep_ids
        and all(bits(a) == bits(b) for a, b in zip(plan.phase_times[:6], full.phase_times[:6]))
        and bits(plan.timing) == bits(full.timing)
    )
    if not same and len(summary["mismatches"]) < 20:
        summary["mismatches"].append({
            "cluster": name, "layer": kwargs["layer_id"], "time": kwargs["time"],
            "shortcut": [list(map(bits, plan.phase_times[:6])), bits(plan.timing)],
            "full": [list(map(bits, full.phase_times[:6])), bits(full.timing)],
        })
    return plan


bcs.prepare_moe_wave_from_inputs = checked
config, _ = h4a4_driver.build_config(root)
set_seeds(config.seed)
from frontier.simulator import Simulator  # noqa: E402

simulator = Simulator(config)


class Stop(Exception):
    pass


def stop(*_):
    raise Stop


signal.signal(signal.SIGALRM, stop)
signal.alarm(wall_s)
started = time.monotonic()
try:
    simulator.run()
    summary["finished"] = True
except Stop:
    summary["finished"] = False
summary["wall_s"] = time.monotonic() - started
summary["simulated_s"] = getattr(simulator, "_time", None)
summary["verdict"] = "PASS" if summary["table_priced_waves"] and not summary["mismatches"] else "FAIL"
(root / "wave_equivalence.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}), len(summary["mismatches"]))
