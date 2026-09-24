"""W9-05: record where watched requests sit around every scheduler call.

    python membership_trace.py <tree> <case> <case_root> <request_id>...

Wraps the vllm_v1 replica scheduler entry points and prints each call after
which a watched request changed queue membership or allocation.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

C2 = "/data/ycfeng/tmp/issue26-correctness-pr/step9_p5/c2_pp1_policy_matrix.py"
QUEUES = ("_request_queue", "_preempted_requests", "_running_requests")
METHODS = ("add_request", "_get_next_batch", "on_batch_end", "_preempt_request",
           "_set_waiting_queues_from_ordered_requests", "_free_request_resources",
           "_schedule_running_requests", "_schedule_waiting_requests")


def main(tree: str, case: str, case_root: Path, watched: set[int]) -> None:
    spec = importlib.util.spec_from_file_location("c2", C2)
    c2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2)
    c2._isolate(tree)
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
        VLLMv1EngineReplicaScheduler,
    )
    from frontier.simulator import Simulator

    def state(sched):
        out = {}
        for rid in watched:
            places = [n for n in QUEUES if any(r.id == rid for r in getattr(sched, n))]
            alloc = sched._allocation_map.get(rid)
            out[rid] = (tuple(places), alloc)
        return out

    depth = [0]

    def wrap(name):
        original = getattr(VLLMv1EngineReplicaScheduler, name)

        @functools.wraps(original)
        def traced(self, *args, **kwargs):
            before = state(self)
            depth[0] += 1
            try:
                result = original(self, *args, **kwargs)
            finally:
                depth[0] -= 1
            after = state(self)
            changed = {rid: (before[rid], after[rid]) for rid in watched if before[rid] != after[rid]}
            detail = ""
            if name == "_get_next_batch" and result is not None:
                detail = f" batch={[(r.id, n) for r, n in zip(result.requests, result.num_tokens)]}"
            if name == "on_batch_end":
                batch = args[0]
                detail = f" batch={[(r.id, n, r.num_processed_tokens, r.completed) for r, n in zip(batch.requests, batch.num_tokens)]}"
            if name == "_preempt_request":
                detail = f" victim={args[0].id}"
            if changed or (name in ("_preempt_request",) ) or (detail and any(str(w) in detail for w in watched)):
                print(f"{'  ' * depth[0]}t={self._current_schedule_time} {name}{detail} "
                      f"changes={changed}", flush=True)
            return result

        setattr(VLLMv1EngineReplicaScheduler, name, traced)

    for method in METHODS:
        wrap(method)
    case_root.mkdir(parents=True, exist_ok=True)
    Simulator(c2._config(case_root, case)).run()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]), {int(a) for a in sys.argv[4:]})
