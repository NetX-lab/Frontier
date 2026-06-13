from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import threading
import queue
from collections import defaultdict

from frontier.entities import Cluster, Replica, Request
from frontier.config import BaseRequestGeneratorConfig
from frontier.scheduler.cluster_scheduler import ClusterSchedulerRegistry
from frontier.types import ClusterType
from frontier.execution_time_predictor import BaseExecutionTimePredictor
from frontier.logger import init_logger

if TYPE_CHECKING:
    from frontier.kv_cache_transfer import BaseKVCacheTransferPredictor
    from frontier.m2n_transfer import BaseM2NTransferPredictor
    from frontier.events import BaseEvent

logger = init_logger(__name__)


class BaseGlobalScheduler(ABC):
    def __init__(
        self,
        clusters: Dict[ClusterType, Cluster],
        request_generator_config: BaseRequestGeneratorConfig,
        predictors: Dict[ClusterType, BaseExecutionTimePredictor] = None,
        kv_cache_transfer_predictor: Optional["BaseKVCacheTransferPredictor"] = None,
        m2n_transfer_predictor: Optional["BaseM2NTransferPredictor"] = None,
        enable_parallel_mode: bool = False,
        max_inter_cluster_queue_size: int = 1000,
    ):
        self._clusters = clusters
        self._cluster_schedulers = {}
        self._kv_cache_transfer_predictor = kv_cache_transfer_predictor
        self._m2n_transfer_predictor = m2n_transfer_predictor
        self._enable_parallel_mode = enable_parallel_mode

        assert predictors is not None, "Predictors are required for scheduler initialization"

        for cluster_type, cluster in clusters.items():
            # Get the appropriate predictor for this cluster type
            predictor = predictors.get(cluster_type)

            self._cluster_schedulers[cluster_type] = ClusterSchedulerRegistry.get(
                cluster._config.cluster_scheduler_config.get_type(),
                config=cluster._config,
                cluster=cluster,
                request_generator_config=request_generator_config,
                predictor=predictor,
                kv_cache_transfer_predictor=kv_cache_transfer_predictor,
                m2n_transfer_predictor=m2n_transfer_predictor,
                available_clusters=set(clusters.keys()),  # Pass available cluster types
            )
        self._request_queue = []  # List[Tuple[Request, ClusterType]]
        self._cluster_logical_times: Dict[ClusterType, float] = {
            cluster_type: 0.0 for cluster_type in clusters.keys()
        }
        self._cluster_logical_time_lock = threading.Lock()

        # Parallel mode inter-cluster communication
        if self._enable_parallel_mode:
            self._init_parallel_communication(max_inter_cluster_queue_size)

    def _init_parallel_communication(self, max_queue_size: int):
        """Initialize parallel mode communication infrastructure."""
        self._max_inter_cluster_queue_size = max_queue_size
        self._parallel_coordination_lock = threading.Lock()

        # Thread-safe message queue for inter-cluster events
        self._inter_cluster_queue = queue.Queue(maxsize=max_queue_size)

        # Per-cluster event buffers for efficient retrieval
        self._cluster_event_buffers: Dict[ClusterType, List["BaseEvent"]] = defaultdict(list)
        self._buffer_lock = threading.Lock()

        # Statistics
        self._events_sent = 0
        self._events_delivered = 0
        self._queue_full_count = 0

        logger.info(f"GlobalScheduler parallel communication initialized with max queue size: {max_queue_size}")

    def add_request(self, request: Request, cluster_type: ClusterType) -> None:
        self._request_queue.append((request, cluster_type))

    def initialize_periodic_scheduling(self, start_time: float = 0.0) -> List["BaseEvent"]:
        """
        Initialize periodic scheduling for all clusters that have it enabled.

        Args:
            start_time: Time to start the first periodic scheduling events

        Returns:
            List of initial PeriodicScheduleEvent objects for clusters with periodic scheduling enabled
        """
        periodic_events = []

        for cluster_type, cluster_scheduler in self._cluster_schedulers.items():
            events = cluster_scheduler.initialize_periodic_scheduling(start_time)
            periodic_events.extend(events)

        if periodic_events:
            logger.info(f"Initialized periodic scheduling for {len(periodic_events)} clusters")

        return periodic_events

    def schedule(self) -> Dict[ClusterType, List[Tuple[int, Request]]]:
        """
        Schedules requests for each cluster and returns a mapping of cluster type to replica schedule events.
        """
        request_mapping = {}
        for request, cluster_type in self._request_queue:
            if cluster_type not in request_mapping:
                request_mapping[cluster_type] = []
            request_mapping[cluster_type].append(request)

        return request_mapping

    # def get_replica(self, replica_id: int) -> Replica:
    #     for cluster_scheduler in self._cluster_schedulers.values():
    #         if replica_id in cluster_scheduler.replicas:
    #             return cluster_scheduler.get_replica(replica_id)
    #     raise ValueError(f"Replica with id {replica_id} not found.")

    # def get_replica_scheduler(self, replica_id: int):
    #     for cluster_scheduler in self._cluster_schedulers.values():
    #         if replica_id in cluster_scheduler.replicas:
    #             return cluster_scheduler.get_replica_scheduler(replica_id)
    #     raise ValueError(f"Replica scheduler for replica id {replica_id} not found.")

    def get_cluster_scheduler(self, cluster_type: ClusterType):
        # Each cluster has a unique scheduler.
        return self._cluster_schedulers[cluster_type]

    def clear_queues(self):
        self._request_queue = []

    @property
    def is_empty(self) -> bool:
        return all(
            cluster_scheduler.is_empty()
            for cluster_scheduler in self._cluster_schedulers.values()
        )

    def update_cluster_logical_time(
        self, cluster_type: ClusterType, logical_time: float
    ) -> None:
        with self._cluster_logical_time_lock:
            self._cluster_logical_times[cluster_type] = max(
                self._cluster_logical_times.get(cluster_type, 0.0), logical_time
            )

    def get_cluster_logical_time(self, cluster_type: ClusterType) -> float:
        with self._cluster_logical_time_lock:
            return float(self._cluster_logical_times.get(cluster_type, 0.0))

    # TODO: remove this; we don't need to handle moe ready and collective schedule in global scheduler
    # def on_moe_ready(self, time: float, replica_id: int, stage_id: int, batch):
    #     for cluster_scheduler in self._cluster_schedulers.values():
    #         if replica_id in cluster_scheduler.replicas:
    #             return cluster_scheduler.on_moe_ready(
    #                 time, replica_id, stage_id, batch
    #             )
    #     raise ValueError(f"Replica with id {replica_id} not found in any cluster.")

    # def on_moe_collective_schedule(
    #     self, time: float, stage_id: int, batch_global_id: int, metrics_store
    # ):
    #     # This assumes that a collective operation for a given batch_global_id
    #     # only happens within one cluster.
    #     for cluster_scheduler in self._cluster_schedulers.values():
    #         if batch_global_id in cluster_scheduler.moe_waiting_room[stage_id]:
    #             return cluster_scheduler.on_moe_collective_schedule(
    #                 time, stage_id, batch_global_id, metrics_store
    #             )
    #     raise ValueError(f"Batch with global id {batch_global_id} not found.")

    def get_cluster_scheduler(self, cluster_type: ClusterType):
        """
        Get the cluster scheduler for a specific cluster type.

        Args:
            cluster_type: Type of the cluster

        Returns:
            Cluster scheduler instance for the specified cluster type
        """
        if cluster_type not in self._cluster_schedulers:
            raise ValueError(f"Cluster scheduler for {cluster_type} not found.")
        return self._cluster_schedulers[cluster_type]

    # Parallel mode inter-cluster communication methods
    def route_event_to_cluster(self, event: "BaseEvent", target_cluster: ClusterType):
        """
        Route an event to a target cluster in parallel mode.

        Args:
            event: Event to be routed
            target_cluster: Target cluster type
        """
        if not self._enable_parallel_mode:
            logger.warning("route_event_to_cluster called but parallel mode is not enabled")
            return

        with self._parallel_coordination_lock:
            try:
                # Try to put the event in the inter-cluster queue
                message = (event, target_cluster)
                self._inter_cluster_queue.put(message, block=False)
                self._events_sent += 1

                logger.debug(f"Event {event.__class__.__name__} routed to {target_cluster.name}")

            except queue.Full:
                # Queue is full: do NOT drop events. Retry with a short blocking put; if still full,
                # fall back to placing the event directly into the target cluster's buffer.
                self._queue_full_count += 1
                try:
                    # Block briefly to alleviate transient bursts
                    self._inter_cluster_queue.put(message, block=True, timeout=0.1)
                    self._events_sent += 1
                    logger.debug(
                        f"Inter-cluster queue was full; succeeded after retry for {event.__class__.__name__} → {target_cluster.name}"
                    )
                except queue.Full:
                    # Fallback: place directly into buffer to guarantee delivery
                    with self._buffer_lock:
                        self._cluster_event_buffers[target_cluster].append(event)
                    self._events_sent += 1
                    logger.info(
                        f"Inter-cluster queue still full after retry. Buffered event {event.__class__.__name__} directly to {target_cluster.name}. "
                        f"Queue full count: {self._queue_full_count}"
                    )

    def get_events_for_cluster(self, cluster_type: ClusterType) -> List["BaseEvent"]:
        """
        Get all pending events for a specific cluster in parallel mode.

        Args:
            cluster_type: Cluster type to get events for

        Returns:
            List of events destined for the specified cluster
        """
        if not self._enable_parallel_mode:
            return []

        # First, process any new messages from the queue
        self._process_inter_cluster_messages()

        # Then return events from the cluster's buffer
        with self._buffer_lock:
            events = self._cluster_event_buffers[cluster_type].copy()
            self._cluster_event_buffers[cluster_type].clear()

            if events:
                self._events_delivered += len(events)
                logger.debug(f"Delivered {len(events)} events to {cluster_type.name}")

            return events

    def _process_inter_cluster_messages(self):
        """
        Process messages from the inter-cluster queue and distribute them to cluster buffers.
        """
        messages_processed = 0

        # Process all available messages in the queue
        while True:
            try:
                # Get message from queue (non-blocking)
                event, target_cluster = self._inter_cluster_queue.get(block=False)

                # Add to the target cluster's buffer
                with self._buffer_lock:
                    self._cluster_event_buffers[target_cluster].append(event)

                messages_processed += 1

                # Mark task as done
                self._inter_cluster_queue.task_done()

            except queue.Empty:
                # No more messages to process
                break

        if messages_processed > 0:
            logger.debug(f"Processed {messages_processed} inter-cluster messages")

    def get_inter_cluster_communication_stats(self) -> dict:
        """Get statistics about inter-cluster communication."""
        if not self._enable_parallel_mode:
            return {}

        buffer_sizes = {}
        with self._buffer_lock:
            buffer_sizes = {cluster_type.name: len(events)
                           for cluster_type, events in self._cluster_event_buffers.items()}

        return {
            "events_sent": self._events_sent,
            "events_delivered": self._events_delivered,
            "queue_size": self._inter_cluster_queue.qsize(),
            "queue_full_count": self._queue_full_count,
            "buffer_sizes": buffer_sizes,
            "total_buffered_events": sum(len(events) for events in self._cluster_event_buffers.values()),
        }
