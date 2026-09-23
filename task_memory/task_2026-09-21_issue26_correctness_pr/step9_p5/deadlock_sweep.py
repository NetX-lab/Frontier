"""Reachability sweep: does a MoE multi-lane Poisson run drain, per tree and cluster scheduler?"""
import sys, json, subprocess, os
from pathlib import Path
HERE = Path(__file__).parent
PY = "/data/ycfeng/envs/frontier-py310/bin/python"

def child(tree, scheduler, shape, qps, seed, root):
    sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in getattr(type(f), "__module__", "").lower()]
    sys.path.insert(0, tree)
    import frontier.simulator as probe
    assert probe.__file__.startswith(tree + "/")
    sys.path.insert(1, str(HERE))
    import c2_pp1_policy_matrix as m
    import frontier.config as fc
    fc.VllmLoadBalancingClusterSchedulerConfig = getattr(fc, scheduler)
    m.WORKLOADS["sweep"] = ("online", 24, "uniform", qps, 128)
    from frontier.simulator import Simulator
    cfg = m._config(Path(root), f"{shape}__sweep")
    cfg.request_generator_config.interval_generator_config.seed = seed
    cfg.request_generator_config.length_generator_config.seed = seed
    cfg.request_generator_config.seed = seed
    sim = Simulator(cfg)
    try:
        sim.run(); status = "drained"
    except RuntimeError as e:
        status = "stuck" if "non-empty scheduler state" in str(e) else f"error:{str(e)[:60]}"
    reqs = list(sim._all_requests)
    print("SWEEP", json.dumps([status, sum(r.completed for r in reqs), len(reqs)]))

if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
        sys.exit(0)
    tree, scheduler, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    rows = []
    for shape in ("moe_dp2", "moe_dp4"):
        for qps in (50.0, 100.0, 200.0, 400.0):
            for seed in (42, 7, 123):
                root = out / f"{shape}_q{int(qps)}_s{seed}"
                r = subprocess.run([PY, __file__, "child", tree, scheduler, shape, str(qps), str(seed), str(root)],
                                   cwd=tree, env={**os.environ, "PYTHONPATH": tree}, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
                line = [l for l in r.stdout.splitlines() if l.startswith("SWEEP")]
                result = json.loads(line[0][6:]) if line else [f"crash rc={r.returncode}", None, None]
                rows.append([shape, qps, seed] + result)
                print(shape, qps, seed, result, flush=True)
    (out / "sweep.json").write_text(json.dumps(rows))
