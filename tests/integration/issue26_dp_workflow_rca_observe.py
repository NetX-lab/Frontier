"""Observe a bounded prefix of real Frontier DP routing and load feedback."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch

from frontier.scheduler.cluster_scheduler.vllm_load_balancing_cluster_scheduler import (
    VllmLoadBalancingClusterScheduler,
)
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
from frontier.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler


class ObservationComplete(Exception):
    """Stop after the requested routing prefix without claiming E2E completion."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routes", type=int, required=True)
    args = parser.parse_args()
    command = json.loads(args.command.read_text())
    assert command[1:3] == ["-m", "frontier.main"]
    assert args.routes > 0
    args.output.mkdir(parents=True, exist_ok=False)
    command[command.index("--metrics_config_output_dir") + 1] = str(args.output / "metrics")
    events = []
    routes = []
    select = VllmDPLoadBalancer.select
    report = VllmDPLoadBalancer.report
    schedule = VllmLoadBalancingClusterScheduler.schedule_at
    admit = BaseReplicaScheduler.on_schedule
    model_tokens = Counter()

    def observe_admit(scheduler, time=0.0):
        batches = admit(scheduler, time)
        for batch in batches:
            members = [{"request_id": str(request.id), "tokens": tokens,
                        "prior_scheduled_tokens": model_tokens[str(request.id)]}
                       for request, tokens in zip(batch.requests, batch.num_tokens)]
            events.append({"event": "admission", "time_s": time,
                           "lane": scheduler._replica_local_id,
                           "batch_id": batch.id, "members": members})
            model_tokens.update({row["request_id"]: row["tokens"] for row in members})
        return batches

    def observe_select(router, time):
        router._advance(time)
        before = [list(load) for load in router.frontend_counts]
        lane = select(router, time)
        record = {"event": "route", "time_s": time, "lane": lane,
                  "counts_before": before,
                  "engine_counts": [list(load) for load in router.engine_counts],
                  "last_publish_ms": router.last_publish_ms}
        events.append(record)
        routes.append(record)
        return lane

    def observe_report(router, time, lane, step, load):
        before = router.engine_counts[lane]
        report(router, time, lane, step, load)
        if before != load:
            events.append({"event": "report", "time_s": time, "lane": lane,
                           "step": step, "load": list(load)})

    def observe_schedule(scheduler, time):
        start = len(routes)
        mapping = schedule(scheduler, time)
        assert len(mapping) == len(routes) - start
        for record, (_, lane, request) in zip(routes[start:], mapping):
            assert record["lane"] == lane
            record["request_id"] = str(request.id)
        if len(routes) >= args.routes:
            raise ObservationComplete
        return mapping

    result = {"status": "FAIL", "command": command,
              "scope": "Bounded diagnostic prefix; current-task caches retained; no E2E metric."}
    try:
        from frontier.main import main as run_frontier
        sys.argv = [command[2], *command[3:]]
        with (patch.object(VllmDPLoadBalancer, "select", observe_select),
              patch.object(VllmDPLoadBalancer, "report", observe_report),
              patch.object(VllmLoadBalancingClusterScheduler, "schedule_at", observe_schedule),
              patch.object(BaseReplicaScheduler, "on_schedule", observe_admit)):
            run_frontier()
    except ObservationComplete:
        result["status"] = "PASS"
    finally:
        result.update(events=events, routes=routes)
        (args.output / "observation.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "PASS", "Simulation ended before the requested routing prefix"
    print(json.dumps({"status": result["status"], "routes": len(routes)}))


if __name__ == "__main__":
    main()
