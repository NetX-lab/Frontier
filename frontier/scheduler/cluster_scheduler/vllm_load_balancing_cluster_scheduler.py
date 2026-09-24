"""Route one vLLM serving Replica through its internal DP load balancer."""

from typing import List, Tuple

from frontier.entities import Batch, Request
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
from frontier.types import ClusterType, ReplicaSchedulerType


class VllmLoadBalancingClusterScheduler(BaseClusterScheduler):
    """Model single-frontend vLLM V1 DP routing with delayed load snapshots.

    Supported scope, and nothing wider: one co-location Replica, one modeled
    frontend, the `vllm_v1` replica scheduler, and a report key whose ordering
    actually holds -- a MoE model, whose attention-DP lanes share each stage-0
    forward, or a single lane, where the question does not arise. Everything
    else is rejected in the constructor.

    No placement or timing equivalence with a real vLLM deployment is claimed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self._cluster_type is not ClusterType.MONOLITHIC
            or self._num_replicas != 1
            or self._replica_scheduler_type is not ReplicaSchedulerType.VLLM_V1
        ):
            raise ValueError(
                "vllm_load_balancing supports one co-location Replica with "
                "vllm_v1, got "
                f"cluster_type={self._cluster_type.name}, "
                f"num_replicas={self._num_replicas}, "
                f"replica_scheduler={self._replica_scheduler_type}"
            )
        # The load report is keyed by the Replica's stage-0 forward group. A
        # MoE Replica binds one group for every lane of one forward, so peer
        # lanes share the key. A dense Replica has no per-forward collective
        # across its attention-DP lanes and binds no group, so two lanes would
        # order their reports only by their own admission counts. One lane is
        # safe either way.
        if not self._config.replica_config.model_config.is_moe:
            if self._replica_dp_size != 1:
                raise ValueError(
                    "vllm_load_balancing orders reports by the stage-0 forward "
                    "that attention-DP lanes share; dense attention-DP lanes "
                    "share no stage-0 forward, so use attn_dp=1, got "
                    f"attn_dp={self._replica_dp_size}"
                )
        self._serving_replica_id = next(iter(self._cluster.replicas))
        self._num_pipeline_stages = self._config.replica_config.num_pipeline_stages
        self._load_balancer = VllmDPLoadBalancer(self._replica_dp_size)
        self._last_admitted_key = [-1] * self._replica_dp_size
        self._held_key: list[int | None] = [None] * self._replica_dp_size

    def schedule_at(self, time: float) -> List[Tuple[int, int, Request]]:
        """Route every queued request at this simulation time.

        Selection depends on when it happens, because the frontend's view of
        engine load is a snapshot published on a timer. The time therefore
        arrives as an argument rather than through retained state.
        """

        self.sort_requests()
        mapping = [
            (self._serving_replica_id, self._load_balancer.select(time), request)
            for request in self._request_queue
        ]
        self._request_queue.clear()
        return mapping

    def schedule(self) -> List[Tuple[int, int, Request]]:
        """Refuse to route without a time, rather than reuse a stale one."""

        raise RuntimeError(
            "vllm_load_balancing selects from a time-dependent load snapshot; "
            "route through schedule_at(time), which ClusterScheduleEvent calls"
        )

    def on_replica_batch_scheduled(
        self,
        time: float,
        replica_id: int,
        replica_local_id: int | None,
        batch: Batch,
    ) -> None:
        """Report the lane's post-admission load while its pipeline has room.

        A vLLM engine whose batch queue still has room after scheduling
        publishes at once; otherwise it waits for its oldest batch and
        publishes after applying that output. That filled-pipeline admission
        is therefore reported with the lane's next completion, under the key
        taken here.
        """

        lane_id = self._lane_index(replica_local_id)
        key = self._next_report_key(replica_id, lane_id)
        self._last_admitted_key[lane_id] = key
        lane = self.get_replica_scheduler(replica_id, lane_id)
        if lane.num_running_batches < self._num_pipeline_stages:
            self._load_balancer.report(time, lane_id, key, lane.get_request_load())
        else:
            self._held_key[lane_id] = key

    def on_replica_batch_end(
        self,
        time: float,
        replica_id: int,
        replica_local_id: int | None,
        batch: Batch | None,
    ) -> None:
        """Report the lane's post-step load under the iteration that applied it.

        That iteration is the one whose admission filled the pipeline, when
        there is one. Otherwise the lane launches no forward here, and the key
        is that of the lane's next forward.
        """

        lane_id = self._lane_index(replica_local_id)
        key = self._held_key[lane_id]
        if key is None:
            key = self._next_report_key(replica_id, lane_id)
        else:
            self._held_key[lane_id] = None
        lane = self.get_replica_scheduler(replica_id, lane_id)
        self._load_balancer.report(time, lane_id, key, lane.get_request_load())

    def _next_report_key(self, replica_id: int, lane_id: int) -> int:
        """Key of the stage-0 forward that the lane's next batch joins.

        vLLM keys a report by the engine's step counter. Each step launches
        one forward, real or dummy, that pairs with the peers' forward of the
        same index, so the key names a forward shared by the peer engines. The
        Replica's stage-0 forward group is that shared forward here. A lane's
        batches admitted before they start stage 0 take consecutive groups,
        which the `+ 1` term provides. Only comparisons of keys are used, as the
        reference coordinator uses its `(wave, step)` pair; Frontier has no wave
        reset, so the group id collapses that pair to one scalar.
        """

        context = self.get_stage_execution_context(replica_id, 0)
        return max(
            context.joinable_forward_group_id,
            self._last_admitted_key[lane_id] + 1,
        )

    @staticmethod
    def _lane_index(replica_local_id: int | None) -> int:
        if type(replica_local_id) is not int:
            raise ValueError(
                "vllm_load_balancing reports load per attention-DP lane and "
                "needs an exact lane index, got "
                f"replica_local_id={replica_local_id!r}"
            )
        return replica_local_id
