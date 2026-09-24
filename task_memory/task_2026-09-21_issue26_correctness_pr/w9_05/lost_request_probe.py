"""W9-05: run one C2 tight-KV case and report where each unfinished request ends.

    python lost_request_probe.py <tree> <case> <case_root>

Uses the C2 PP=1 policy-matrix configuration (step9_p5/c2_pp1_policy_matrix.py).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

C2 = "/data/ycfeng/tmp/issue26-correctness-pr/step9_p5/c2_pp1_policy_matrix.py"


def main(tree: str, case: str, case_root: Path) -> None:
    spec = importlib.util.spec_from_file_location("c2", C2)
    c2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2)
    c2._isolate(tree)
    from frontier.simulator import Simulator

    case_root.mkdir(parents=True, exist_ok=True)
    simulator = Simulator(c2._config(case_root, case))
    simulator.run()
    lost = [r for r in simulator._all_requests if not r.completed]
    schedulers = []
    for cluster_scheduler in simulator._global_scheduler._cluster_schedulers.values():
        schedulers.extend(cluster_scheduler._replica_schedulers.values())
    report = []
    for request in lost:
        where = []
        for sched in schedulers:
            for name in ("_request_queue", "_preempted_requests", "_running_requests",
                         "_waiting_requests"):
                queue = getattr(sched, name, None)
                if queue is not None and request in queue:
                    where.append(f"{sched.replica_id}/{getattr(sched, '_replica_local_id', None)}:{name}")
            if request.id in getattr(sched, "_allocation_map", {}):
                where.append(f"alloc={sched._allocation_map[request.id]}")
        report.append(dict(
            id=request.id,
            prefill=request.num_prefill_tokens,
            decode=request.num_decode_tokens,
            processed=request.num_processed_tokens,
            prefill_complete=request.is_prefill_complete,
            preempted=getattr(request, "_preempted", None),
            num_restarts=getattr(request, "num_restarts", None),
            where=where,
        ))
    print(json.dumps(dict(total=len(simulator._all_requests), lost=report), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
