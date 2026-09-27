"""Print the requests of the stalled C2 case, and where the run stops."""
import sys
from pathlib import Path
tree = sys.argv[1]
sys.path.insert(0, tree)
sys.path.insert(1, tree + "/task_memory/task_2026-09-21_issue26_correctness_pr/step9_p5")
import c2_pp1_policy_matrix as m
import frontier.config as fc
if sys.argv[2] == "round_robin":
    fc.VllmLoadBalancingClusterSchedulerConfig = fc.RoundRobinClusterSchedulerConfig
from frontier.simulator import Simulator
cfg = m._config(Path(sys.argv[3]), "moe_dp4__poisson_uniform_n24_q200")
seed = int(sys.argv[4])
cfg.request_generator_config.interval_generator_config.seed = seed
cfg.request_generator_config.length_generator_config.seed = seed
cfg.request_generator_config.seed = seed
sim = Simulator(cfg)
reqs = sorted(sim._all_requests, key=lambda r: r.arrived_at) if hasattr(sim, "_all_requests") else None
try:
    sim.run(); status = "drained"
except RuntimeError as e:
    status = "stuck" if "non-empty scheduler state" in str(e) else f"error:{e}"
reqs = sorted(sim._all_requests, key=lambda r: r.arrived_at)
print(status, sum(r.completed for r in reqs), len(reqs))
for r in reqs[:8]:
    print(repr(float(r.arrived_at)), r.num_prefill_tokens, r.num_decode_tokens)
