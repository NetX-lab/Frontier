"""Run one C2 workload under round-robin against any tree (no policy import)."""
import sys
from pathlib import Path
tree, case, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in getattr(type(f), "__module__", "").lower()]
sys.path.insert(0, tree)
import frontier.simulator as probe
assert probe.__file__.startswith(tree + "/"), probe.__file__
import frontier.config as fc
fc.VllmLoadBalancingClusterSchedulerConfig = fc.RoundRobinClusterSchedulerConfig
sys.path.insert(1, str(Path(__file__).parent))
import c2_pp1_policy_matrix as m
from frontier.simulator import Simulator
root.mkdir(parents=True, exist_ok=True)
sim = Simulator(m._config(root, case))
sim.run()
reqs = list(sim._all_requests)
print("RR_RESULT", sum(r.completed for r in reqs), len(reqs))
