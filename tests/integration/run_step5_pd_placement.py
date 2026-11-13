"""CPU-only Step P32/D32 placement evidence using the real simulator.

The observer records, but never supplies, loads or choices. Its independent
source-rule calculation compares the published frontend view with the latest
coordinator reports; the latter is a counterfactual, never a placement input.
Native IPC timing, tie/update parity and the D prompt tail remain unverified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tests.integration.run_glm53_structural_pd import read_jsonl, require, source_identity
from tests.integration.run_step5_structural_pd import command, validate_config

REPO = Path(__file__).resolve().parents[2]
ARRIVALS = (0, .01, .02, .03, .49, .51, .53, .55, .57, .59, .61, .63, .65, .67, 1.01, 1.03, 2.48)
PROMPTS = (529, 16, 16, 529, 16, 16, 16, 16, 529, 16, 16, 16, 16, 16, 16, 16, 16)


def source_choice(counts, inflight, start, clients=1):
    """Image core_client.py:1653-1691, independent of Frontier's score method."""
    scores = [max(clients * n, w + r) + w * 6.0 * max(0, kv - .5)
              for (w, r, kv), n in zip(counts, inflight)]
    order = [(start + i) % len(scores) for i in range(len(scores))]
    return min(order, key=lambda i: scores[i]), scores


def child(mode: str, root: Path):
    from frontier.main import main as frontier_main
    from frontier.scheduler.cluster_scheduler.vllm_load_balancing_cluster_scheduler import (
        VllmLoadBalancingClusterScheduler as Policy,
    )
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
        VLLMv1EngineReplicaScheduler as Lane,
    )
    from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer as Balancer

    trace = []
    schedulers = []
    owners = {}
    original_init = Policy.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._load_balancer.role = self._cluster_type.name
        self._load_balancer.endpoint = self._serving_replica_id
        schedulers.append(self)

    Policy.__init__ = init
    original_select = Balancer.select

    def select(self, at, request_id=None):
        self._advance(at)
        counts = lambda loads: [(load.waiting, load.running, load.kv_cache_usage) for load in loads]
        observed = counts(self.frontend_counts)
        current = counts(self.engine_counts)
        expected, scores = source_choice(observed, self.engine_inflight, self.eng_start_index)
        counterfactual, _ = source_choice(current, self.engine_inflight, self.eng_start_index)
        actual = original_select(self, at, request_id)
        require(actual == expected, "placement differs from the image source rule")
        owners[(self.role, request_id)] = (self.endpoint, actual)
        trace.append(dict(phase="placement", time=at, role=self.role, request_id=request_id,
                          endpoint=self.endpoint, lane=actual, expected=expected, actual=actual,
                          current_report_choice=counterfactual, observed=observed, current=current,
                          scores=scores, tie_count=scores.count(min(scores)),
                          published_ms=self.last_publish_ms))
        return actual

    Balancer.select = select

    def record(phase, role, endpoint, lane, request, at=None):
        require(owners[(role, request.id)] == (endpoint, lane),
                f"{phase} changed request {request.id}'s {role} identity")
        trace.append(dict(phase=phase, role=role, endpoint=endpoint, lane=lane,
                          request_id=request.id, time=at))

    def wrap_lane(name, phase):
        original = getattr(Lane, name)

        def wrapped(self, request, *args, **kwargs):
            result = original(self, request, *args, **kwargs)
            record(phase if result is not None or name != "reserve_kv_transfer_target"
                   else "capacity_wait", self._cluster_type.name,
                   self._replica_id, self._replica_local_id, request)
            return result

        setattr(Lane, name, wrapped)

    wrap_lane("reserve_kv_transfer_target", "reservation")
    wrap_lane("validate_kv_transfer_target_completion", "read_complete")
    wrap_lane("mark_kv_transfer_target_ready", "ready")
    wrap_lane("consume_kv_transfer_target_ready", "admission")

    def wrap_batch(name, phase):
        original = getattr(Policy, name)

        def wrapped(self, at, endpoint, lane, batch):
            result = original(self, at, endpoint, lane, batch)
            if batch is not None:
                for request in batch.requests:
                    record(phase, self._cluster_type.name, endpoint, lane, request, at)
                    if phase == "step_end" and request.completed:
                        record("completion", self._cluster_type.name, endpoint, lane, request, at)
            return result

        setattr(Policy, name, wrapped)

    wrap_batch("on_replica_batch_scheduled", "step_start")
    wrap_batch("on_replica_batch_end", "step_end")
    argv = command(mode, "placement", root)
    argv += ["--cluster_scheduler_config_type", "vllm_load_balancing",
             "--vllm_load_balancing_cluster_scheduler_config_policy_version", "v0.28.0"]
    (root / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    sys.argv = ["frontier.main", *argv[3:]]
    frontier_main()
    for scheduler in schedulers:
        require(not scheduler._load_balancer.request_owners, "unfinished role-local inflight")
        require(scheduler._load_balancer.engine_inflight == [0] * 4, "leaked inflight count")
    with (root / "placement_trace.jsonl").open("w") as stream:
        for row in trace:
            stream.write(json.dumps(row) + "\n")


def validate(root, mode):
    [config] = (root / "metrics").rglob("config.json")
    validate_config(json.loads(config.read_text()), mode)
    metadata = json.loads((config.parent / "system_metrics.json").read_text())["simulation_metadata"]
    require(metadata["completed_requests"] == len(PROMPTS), "incomplete requests")
    rows = read_jsonl(root / "placement_trace.jsonl")
    placements = [row for row in rows if row["phase"] == "placement"]
    require(len(placements) == 2 * len(PROMPTS), "duplicate or missing role placement")
    for request_id in range(len(PROMPTS)):
        for phase in ("reservation", "read_complete", "ready", "admission", "completion"):
            require(sum(row["phase"] == phase and row["request_id"] == request_id
                        for row in rows) == 1, f"request {request_id}: {phase} is not unique")
    transfers = read_jsonl(config.parent / "frontier_transfer_ledger.jsonl")
    owners = {(row["role"], row["request_id"]): row for row in placements}
    for row in transfers:
        rid = row["request_ids"][0]
        for role, prefix in (("PREFILL", "source"), ("DECODE", "target")):
            selected = owners[(role, rid)]
            require((row[f"{prefix}_replica_id"], row[f"{prefix}_replica_local_id"])
                    == (selected["endpoint"], selected["lane"]), "transfer reassigned ownership")
        require(owners[("DECODE", rid)]["time"] <= row["start_ts_s"], "D placement after READ")
    stale_changes = [row for row in placements if row["current_report_choice"] != row["actual"]]
    if mode == "online":
        require(stale_changes, "online arrivals did not distinguish delayed from current reports")
    return dict(mode=mode, completed=len(PROMPTS), placements=len(placements),
                expected_matches=sum(row["expected"] == row["actual"] for row in placements),
                ties=sum(row["tie_count"] > 1 for row in placements),
                stale_changes=stale_changes,
                lanes={role: [owners[(role, i)]["lane"] for i in range(len(PROMPTS))]
                       for role in ("PREFILL", "DECODE")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("offline", "online"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args.mode, args.output_root)
        return
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    with (root / "requests.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arrived_at", "num_prefill_tokens", "num_decode_tokens"))
        for arrival, prompt in zip(ARRIVALS, PROMPTS):
            writer.writerow((arrival if args.mode == "online" else 0, prompt, 3))
    before = source_identity(Path(__file__))
    started = time.monotonic()
    env = {**os.environ, "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (root / "run.log").open("w") as stream:
        process = subprocess.Popen([sys.executable, __file__, "--child", "--mode", args.mode,
                                    "--output-root", str(root)], cwd=REPO, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(process.pid, 0)
    require(os.waitstatus_to_exitcode(status) == 0, f"simulation failed: {root / 'run.log'}")
    require(source_identity(Path(__file__)) == before, "source changed during run")
    result = validate(root, args.mode)
    result.update(wall_s=time.monotonic() - started, peak_rss_mib=usage.ru_maxrss / 1024,
                  source_identity=before)
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
