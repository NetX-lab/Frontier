"""Does a short arrival trace reproduce W9-04 through the DP placement test's config?"""
import importlib.util, sys
from pathlib import Path
import pytest
tree, policy_name, root, n = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
import json
VARIANT = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
spec = importlib.util.spec_from_file_location("dp_rt", tree + "/tests/integration/test_vllm_dp_placement_runtime.py")
dp_rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp_rt)
from frontier.config import RoundRobinClusterSchedulerConfig, VllmLoadBalancingClusterSchedulerConfig
from frontier.simulator import Simulator
ROWS = [
    (0.005100301436374005, 7, 3), (0.006708421756748833, 21, 6), (0.013376385120789248, 53, 14),
    (0.02451282561854615, 11, 4), (0.027253056415528606, 7, 3), (0.028486639895327726, 41, 11),
    (0.028621111913729264, 19, 6), (0.033868571919842126, 43, 12),
]
ROWS = [tuple(r) for r in VARIANT] if VARIANT else ROWS
root.mkdir(parents=True, exist_ok=True)
trace = root / "trace.csv"
trace.write_text("arrived_at,num_prefill_tokens,num_decode_tokens\n" + "".join(f"{a!r},{p},{d}\n" for a, p, d in ROWS[:n]))
policy = {"rr": RoundRobinClusterSchedulerConfig, "vlb": VllmLoadBalancingClusterSchedulerConfig}[policy_name]
with pytest.MonkeyPatch.context() as patch:
    cfg = dp_rt._config(root, patch, is_moe=True, attn_dp=4, moe_ep=4, policy=policy, trace=str(trace))
    sim = Simulator(cfg)
    try:
        sim.run(); status = "drained"
    except RuntimeError as e:
        status = "stuck" if "non-empty scheduler state" in str(e) else f"error:{e}"
    reqs = list(sim._all_requests)
    print("RESULT", policy_name, n, status, sum(r.completed for r in reqs), len(reqs))
