"""C6: Step 9 boundary probe shapes on a branch without the PR 35 batch-end seam.

Uses probe_main.build_config unchanged and reports completion only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_main import build_config  # noqa: E402

SHAPES = {
    "moe_dp2_pp1": dict(is_moe=True, attn_dp=2, moe_ep=2, stages=1),
    "moe_dp2_pp2": dict(is_moe=True, attn_dp=2, moe_ep=2, stages=2),
    "moe_dp2_pp3": dict(is_moe=True, attn_dp=2, moe_ep=2, stages=3),
    "dense_dp1_pp2": dict(is_moe=False, attn_dp=1, moe_ep=1, stages=2),
}

root, label = Path(sys.argv[1]), sys.argv[2]
case_root = root / label
case_root.mkdir(parents=True, exist_ok=True)
from frontier.simulator import Simulator  # noqa: E402

result = {"shape": SHAPES[label]}
try:
    simulator = Simulator(build_config(case_root, **SHAPES[label]))
    simulator.run()
    requests = list(simulator._all_requests)
    result.update(completed=sum(1 for r in requests if r.completed), requests=len(requests))
except Exception as exc:
    result.update(error=repr(exc)[:800])
(case_root / "result.json").write_text(json.dumps(result, indent=1))
print(label, json.dumps(result))
