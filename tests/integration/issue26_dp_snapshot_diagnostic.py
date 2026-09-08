"""Run a prepared Frontier command and verify post-step DP load feedback."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch

from frontier.scheduler.cluster_scheduler.vllm_load_balancing_cluster_scheduler import (
    VllmLoadBalancingClusterScheduler,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = json.loads(args.command.read_text())
    assert command[1:3] == ["-m", "frontier.main"]
    original = VllmLoadBalancingClusterScheduler.on_replica_batch_end
    calls = Counter()
    changes = []
    final_loads = {}

    def observe(scheduler, time, replica_id, lane, batch):
        before = scheduler._load_balancer.engine_counts[lane]
        original(scheduler, time, replica_id, lane, batch)
        load = scheduler.get_replica_scheduler(replica_id, lane).get_request_load()
        assert scheduler._load_balancer.engine_counts[lane] == load
        calls[lane] += 1
        final_loads[lane] = list(load)
        if before != load:
            changes.append({
                "time_s": time, "lane": lane, "load": list(load),
                "batch_id": batch.id, "request_ids": [r.id for r in batch.requests],
            })

    result = {"status": "FAIL", "command_file": str(args.command),
              "scope": "Dummy operator timing; real current-case scheduler and DES callbacks."}
    try:
        from frontier.main import main as run_frontier
        sys.argv = [command[2], *command[3:]]
        with patch.object(VllmLoadBalancingClusterScheduler, "on_replica_batch_end", observe):
            run_frontier()
        assert set(calls) == {0, 1}, calls
        assert all(load == [0, 0] for load in final_loads.values()), final_loads
        result["status"] = "PASS"
    finally:
        result.update(callbacks_by_lane=dict(calls), load_changes=changes,
                      final_engine_loads=final_loads)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
