"""W9-05: request conservation under KV pressure, per tree.

    python kv_pressure_sweep.py <tree> <out_dir>

Runs the C2 PP=1 configuration (step9_p5/c2_pp1_policy_matrix.py) over shapes,
KV block counts, arrival rates and seeds. Each cell records how many requests
complete, whether each completed request produced all its decode tokens, and
how many preemptions hit a request before and after its prefill completed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "/data/ycfeng/envs/frontier-py310/bin/python"
C2_DIR = "/data/ycfeng/tmp/issue26-correctness-pr/step9_p5"
SHAPES = ("dense_dp1", "moe_dp2", "moe_dp4")
NUM_BLOCKS = (10, 12, 16, 24)
QPS = (50.0, 200.0)
SEEDS = (42, 7, 123)


def child(tree: str, shape: str, num_blocks: int, qps: float, seed: int, root: str) -> None:
    sys.meta_path[:] = [
        f for f in sys.meta_path if "editable" not in getattr(type(f), "__module__", "").lower()
    ]
    sys.path.insert(0, tree)
    import frontier.simulator as probe

    assert probe.__file__.startswith(tree + "/"), probe.__file__
    sys.path.insert(1, C2_DIR)
    import c2_pp1_policy_matrix as c2
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
        VLLMv1EngineReplicaScheduler,
    )
    from frontier.simulator import Simulator

    phases = {"prefill": 0, "decode": 0}
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request

    def observed(self, victim, preempted_requests):
        phases["decode" if victim.is_prefill_complete else "prefill"] += 1
        return preempt_request(self, victim, preempted_requests)

    VLLMv1EngineReplicaScheduler._preempt_request = observed
    c2.WORKLOADS["kv"] = ("online", 24, "uniform", qps, num_blocks)
    config = c2._config(Path(root), f"{shape}__kv")
    config.request_generator_config.interval_generator_config.seed = seed
    config.request_generator_config.length_generator_config.seed = seed
    config.request_generator_config.seed = seed
    simulator = Simulator(config)
    try:
        simulator.run()
        status = "drained"
    except RuntimeError as error:
        status = f"error:{str(error)[:80]}"
    requests = list(simulator._all_requests)
    print("CELL", json.dumps(dict(
        status=status,
        completed=sum(r.completed for r in requests),
        total=len(requests),
        short_output=sum(
            r.completed and r.num_processed_decode_tokens != r.num_decode_tokens for r in requests
        ),
        prefill_preemptions=phases["prefill"],
        decode_preemptions=phases["decode"],
    )))


def run_cell(tree: str, out: Path, cell: tuple) -> dict:
    shape, num_blocks, qps, seed = cell
    root = out / f"{shape}_b{num_blocks}_q{int(qps)}_s{seed}"
    result = subprocess.run(
        [PY, __file__, "child", tree, shape, str(num_blocks), str(qps), str(seed), str(root)],
        cwd=tree, env={**os.environ, "PYTHONPATH": tree, "OMP_NUM_THREADS": "1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900,
    )
    line = [l for l in result.stdout.splitlines() if l.startswith("CELL ")]
    record = json.loads(line[0][5:]) if line else dict(status=f"crash rc={result.returncode}")
    return dict(shape=shape, num_blocks=num_blocks, qps=qps, seed=seed, **record)


def main(tree: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cells = [(s, b, q, seed) for s in SHAPES for b in NUM_BLOCKS for q in QPS for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(lambda cell: run_cell(tree, out, cell), cells))
    (out / "kv_pressure_sweep.json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
    else:
        main(sys.argv[1], Path(sys.argv[2]))
