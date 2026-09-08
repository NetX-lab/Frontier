"""Route one vLLM serving Replica through its internal DP load balancer."""

from frontier.entities import Batch, Request
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
from frontier.types import ClusterType, ReplicaSchedulerType


class VllmLoadBalancingClusterScheduler(BaseClusterScheduler):
    """Model single-frontend vLLM V1 DP routing with delayed load snapshots."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self._cluster_type != ClusterType.MONOLITHIC
            or self._num_replicas != 1
            or self._config.replica_config.num_pipeline_stages != 1
            or self._replica_scheduler_type != ReplicaSchedulerType.VLLM_V1
        ):
            raise ValueError(
                "vllm_load_balancing supports one co-location Replica with "
                "vllm_v1 and PP1"
            )
        self._serving_replica_id = next(iter(self._cluster.replicas))
        self._load_balancer = VllmDPLoadBalancer(self._replica_dp_size)
        self._routing_time = 0.0

    def schedule_at(self, time: float) -> list[tuple[int, int, Request]]:
        self._routing_time = time
        return self.schedule()

    def schedule(self) -> list[tuple[int, int, Request]]:
        self.sort_requests()
        mapping = [
            (
                self._serving_replica_id,
                self._load_balancer.select(self._routing_time),
                request,
            )
            for request in self._request_queue
        ]
        self._request_queue.clear()
        return mapping

    def on_replica_batch_end(
        self, time: float, replica_id: int, replica_local_id: int, batch: Batch
    ) -> None:
        lane = self.get_replica_scheduler(replica_id, replica_local_id)
        self._load_balancer.report(
            time,
            replica_local_id,
            ForwardSyncState.get_step_id(batch),
            lane.get_request_load(),
        )
