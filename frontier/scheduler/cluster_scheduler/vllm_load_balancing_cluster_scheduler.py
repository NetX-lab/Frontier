"""Route one vLLM serving Replica through its internal DP load balancer."""

from typing import List, Tuple

from frontier.entities import Batch, Request
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
from frontier.types import ClusterType, ReplicaSchedulerType


class VllmLoadBalancingClusterScheduler(BaseClusterScheduler):
    """Model single-frontend vLLM V1 DP routing with delayed load snapshots.

    Supported scope, and nothing wider: one co-location Replica, one modeled
    frontend, the `vllm_v1` replica scheduler, one pipeline stage, and a report
    key whose ordering actually holds -- a MoE model, whose attention-DP lanes
    share one forward step identity, or a single lane, where the question does
    not arise. Everything else is rejected in the constructor.

    No placement or timing equivalence with a real vLLM deployment is claimed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self._cluster_type is not ClusterType.MONOLITHIC
            or self._num_replicas != 1
            or self._config.replica_config.num_pipeline_stages != 1
            or self._replica_scheduler_type is not ReplicaSchedulerType.VLLM_V1
        ):
            raise ValueError(
                "vllm_load_balancing supports one co-location Replica with "
                "vllm_v1 and PP1, got "
                f"cluster_type={self._cluster_type.name}, "
                f"num_replicas={self._num_replicas}, "
                f"num_pipeline_stages="
                f"{self._config.replica_config.num_pipeline_stages}, "
                f"replica_scheduler={self._replica_scheduler_type}"
            )
        # The load report is ordered by the forward step identity. A MoE
        # Replica resolves one shared identity for every lane of one forward, so
        # the key is monotonic per Replica. A dense Replica has no per-forward
        # collective across its attention-DP lanes, so each lane keeps its own
        # creation counter and two lanes report interleaved keys against one
        # shared ordering scalar. One lane is safe either way.
        if not self._config.replica_config.model_config.is_moe:
            if self._replica_dp_size != 1:
                raise ValueError(
                    "vllm_load_balancing requires a forward step identity that "
                    "is monotonic per Replica; a dense model provides one only "
                    "at attn_dp=1, got "
                    f"attn_dp={self._replica_dp_size}"
                )
        self._serving_replica_id = next(iter(self._cluster.replicas))
        self._load_balancer = VllmDPLoadBalancer(self._replica_dp_size)

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

    def on_replica_batch_end(
        self,
        time: float,
        replica_id: int,
        replica_local_id: int | None,
        batch: Batch,
    ) -> None:
        """Report the lane's post-step load, keyed by the forward identity.

        The key's numeric value is not a vLLM step counter: it advances once per
        layer, so consecutive forwards are roughly `num_layers` apart. Only its
        ordering and equality are used, which is all the reference coordinator
        uses its `(wave, step)` pair for. Frontier has no wave reset, so a
        Replica-scoped monotonic counter collapses that pair to one scalar.
        """

        if type(replica_local_id) is not int:
            raise ValueError(
                "vllm_load_balancing reports load per attention-DP lane and "
                "needs an exact lane index, got "
                f"replica_local_id={replica_local_id!r}"
            )
        lane = self.get_replica_scheduler(replica_id, replica_local_id)
        self._load_balancer.report(
            time,
            replica_local_id,
            ForwardSyncState.get_step_id(batch),
            lane.get_request_load(),
        )
