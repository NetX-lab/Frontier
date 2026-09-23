"""Step 9 P1(b): Frontier report boundaries and forward membership under PP.

No source change. One shape per process (``IS_MOE`` is process-global). The
probe builds the shape with the stage-admission matrix fixture, wraps four
existing seams and writes every record with a global sequence number:

* ``admit``: ``VLLMv1EngineReplicaScheduler._get_next_batch`` returned a
  batch (the admission; ``_running_requests`` has already grown).
* ``complete``: ``BaseClusterScheduler.on_replica_batch_end``, the existing
  report boundary.
* ``stage_start``: ``ReplicaStageScheduler.pop_batch_if_not_busy`` returned a
  batch; for a MoE lane its ``_forward_cohort_provisional_id`` is then the
  stage's shared forward-group id (``StageExecutionContext.bind_forward_group``).
* ``room``: ``ForwardSyncState.resolve_step`` placed a lane batch, real or
  idle, in a layer room.

At ``admit`` and ``complete`` it reads three candidate keys:

* ``replica_forward_id``: ``ForwardSyncState._next_step_id_by_replica``
  (candidate A, the plan's first candidate);
* ``stage0_next_group``: the stage-0 context's next forward-group id;
* ``stage0_bound_group``: the stage-0 forward group currently bound, if any;

and the stage-0 state a derived key would need: whether the bound group is
sealed, whether this lane's stage 0 is busy, and its stage-0 queue length.
A ``room`` record also carries the batch's forward-group id, so an idle lane's
participation can be placed in a group.

Usage: python probe_boundaries.py <shape> <output dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.e2e.stage_admission_matrix import PREFILL_DECODE, Case, build_config

SHAPES = {
    "moe_dp2_pp1_burst": dict(is_moe=True, attn_dp=2, stages=1),
    "moe_dp2_pp2_burst": dict(is_moe=True, attn_dp=2, stages=2),
    "moe_dp2_pp3_burst": dict(is_moe=True, attn_dp=2, stages=3),
    "moe_dp2_pp2_staggered": dict(is_moe=True, attn_dp=2, stages=2, staggered=True),
    "moe_dp2_pp3_staggered": dict(is_moe=True, attn_dp=2, stages=3, staggered=True),
    "moe_dp1_pp2_burst": dict(is_moe=True, attn_dp=1, stages=2),
    "moe_dp1_pp3_burst": dict(is_moe=True, attn_dp=1, stages=3),
}
NUM_REQUESTS = 6
BURST_QPS = 1e6
STAGGERED_QPS = 20.0


def build_case(label: str) -> Case:
    shape = dict(SHAPES[label])
    staggered = shape.pop("staggered", False)
    return Case(
        case_id=f"P1b-{label}", group="P1b", num_requests=NUM_REQUESTS,
        prefill_tokens=PREFILL_DECODE[0], decode_tokens=PREFILL_DECODE[1],
        arrival="poisson", qps=STAGGERED_QPS if staggered else BURST_QPS,
        simulation_mode="online" if staggered else "offline", **shape,
    )


def run(label: str, output: Path) -> dict:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import BaseClusterScheduler
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
        VLLMv1EngineReplicaScheduler,
    )
    from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import (
        ReplicaStageScheduler,
    )
    from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
    from frontier.simulator import Simulator

    case = build_case(label)
    records: list[dict] = []
    held: dict = {}

    def emit(kind: str, **fields) -> None:
        records.append({"seq": len(records), "kind": kind, "time": held["sim"]._time, **fields})

    def keys(cluster_scheduler, replica_id: int, lane_scheduler) -> dict:
        context = cluster_scheduler._stage_execution_contexts[(replica_id, 0)]
        stage0 = lane_scheduler.get_replica_stage_scheduler(0)
        return {
            "replica_forward_id": int(
                cluster_scheduler._forward_sync_state._next_step_id_by_replica.get(replica_id, 0)
            ),
            "stage0_next_group": context._next_forward_group_id,
            "stage0_bound_group": context._forward_group_id,
            "stage0_sealed": context._forward_group_sealed,
            "lane_stage0_busy": stage0._is_busy,
            "lane_stage0_queue": len(stage0._batch_queue),
        }

    original_next_batch = VLLMv1EngineReplicaScheduler._get_next_batch
    original_batch_end = BaseClusterScheduler.on_replica_batch_end
    original_pop = ReplicaStageScheduler.pop_batch_if_not_busy
    original_resolve = ForwardSyncState.resolve_step

    def observed_next_batch(self, is_micro_batch=False):
        batch = original_next_batch(self, is_micro_batch=is_micro_batch)
        if batch is not None:
            emit("admit", lane=self._replica_local_id, batch=batch.id,
                 running_after=self._num_running_batches + 1, stages=self._num_stages,
                 load=list(self.get_request_load()),
                 **keys(self._cluster_scheduler, self._replica_id, self))
        return batch

    def observed_batch_end(self, time, replica_id, replica_local_id, batch):
        result = original_batch_end(self, time, replica_id, replica_local_id, batch)
        lane = self.get_replica_scheduler(replica_id, replica_local_id)
        emit("complete", lane=replica_local_id, batch=batch.id,
             running_after=lane.num_running_batches, load=list(lane.get_request_load()),
             **keys(self, replica_id, lane))
        return result

    def observed_pop(self):
        batch = original_pop(self)
        if batch is not None:
            emit("stage_start", lane=self._replica_local_id, stage=self._stage_id,
                 batch=batch.id, group=getattr(batch, "_forward_cohort_provisional_id", None))
        return batch

    def observed_resolve(self, **kwargs):
        step = original_resolve(self, **kwargs)
        batch = kwargs["batch"]
        emit("room", lane=kwargs["lane_id"], stage=kwargs["stage_id"], layer=kwargs["layer_id"],
             sync_stage=kwargs["sync_stage"], batch=batch.id, idle=bool(batch.is_idle), step=step,
             group=getattr(batch, "_forward_cohort_provisional_id", None))
        return step

    VLLMv1EngineReplicaScheduler._get_next_batch = observed_next_batch
    BaseClusterScheduler.on_replica_batch_end = observed_batch_end
    ReplicaStageScheduler.pop_batch_if_not_busy = observed_pop
    ForwardSyncState.resolve_step = observed_resolve
    try:
        config = build_config(case, output / "metrics", output / "cache")
        simulator = Simulator(config)
        held["sim"] = simulator
        simulator.run()
        requests = list(simulator._all_requests)
    finally:
        VLLMv1EngineReplicaScheduler._get_next_batch = original_next_batch
        BaseClusterScheduler.on_replica_batch_end = original_batch_end
        ReplicaStageScheduler.pop_batch_if_not_busy = original_pop
        ForwardSyncState.resolve_step = original_resolve
    return {
        "shape": label,
        "case": {field: getattr(case, field) for field in (
            "is_moe", "attn_dp", "stages", "num_requests", "prefill_tokens",
            "decode_tokens", "arrival", "qps", "simulation_mode", "cc_backend")},
        "completed": sum(1 for request in requests if request.completed),
        "requests": len(requests),
        "records": records,
    }


if __name__ == "__main__":
    label, output = sys.argv[1], Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    result = run(label, output)
    (output / f"{label}.json").write_text(json.dumps(result, indent=1))
    print(label, f"{result['completed']}/{result['requests']} completed,",
          len(result["records"]), "records")
