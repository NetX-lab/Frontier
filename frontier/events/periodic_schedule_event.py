import time
from typing import List

from frontier.events.base_event import BaseEvent
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import BaseClusterScheduler
from frontier.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from frontier.types import ClusterType, EventType

logger = init_logger(__name__)


class PeriodicScheduleEvent(BaseEvent):
    """
    Event for periodic scheduling of clusters.
    
    This event is used to implement interval-based scheduling for specific cluster types
    (e.g., DECODE_ATTN) instead of immediate event-driven scheduling. It helps reduce
    scheduling overhead in large-scale scenarios by batching requests that arrive
    within a scheduling interval.
    """
    
    def __init__(self, time: float, cluster_type: ClusterType, scheduling_interval_ms: float):
        super().__init__(time, EventType.PERIODIC_SCHEDULE)
        self._cluster_type = cluster_type
        self._scheduling_interval_ms = scheduling_interval_ms

    def __repr__(self):
        return f"PeriodicScheduleEvent(time={self.time}, cluster_type={self._cluster_type}, interval={self._scheduling_interval_ms}ms)"

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        self._dp_replica_set = set()
        cluster_scheduler: BaseClusterScheduler = scheduler.get_cluster_scheduler(self._cluster_type)

        # Log periodic scheduling details
        from_prefill_queue_size = len(cluster_scheduler._request_queue)
        if from_prefill_queue_size > 0:
            logger.info(f"Periodic scheduling triggered at {self.time:.3f}s: "
                    f"{self._cluster_type.name} cluster with {from_prefill_queue_size} requests in queue "
                    f"(interval={self._scheduling_interval_ms}ms)")
        # from F to A
        from_f_queue_size = len(cluster_scheduler._af_batch_queue)
        if from_f_queue_size > 0:
            logger.info(f"Periodic scheduling triggered at {self.time:.3f}s: "
                    f"{self._cluster_type.name} cluster with {from_f_queue_size} batches in A→F queue "
                    f"(interval={self._scheduling_interval_ms}ms)")

        # Add frequency control to prevent simulator overload
        # This sleep controls the actual execution frequency, not simulation time
        self._apply_frequency_control()

        # Check termination conditions before proceeding
        if self._should_terminate_periodic_scheduling(scheduler, metrics_store):
            logger.info(f"Terminating periodic scheduling for {self._cluster_type.name} cluster "
                       f"at {self.time:.3f}s (termination condition met)")
            return []  # No more events

        # Only schedule if there are requests in the queue
        if from_prefill_queue_size == 0 and from_f_queue_size == 0:
            return self._create_next_periodic_event_if_needed(scheduler)

        # Perform cluster scheduling
        self._request_mapping = cluster_scheduler.schedule()

        # # Log scheduling results
        # if self._request_mapping:
        #     # Handle both individual request scheduling and batch-level scheduling
        #     request_ids = [
        #         request.id
        #         for _, _, request in self._request_mapping
        #         if request is not None
        #     ]
        #     batch_count = len(
        #         [
        #             request
        #             for _, _, request in self._request_mapping
        #             if request is None
        #         ]
        #     )

        #     log_parts = []
        #     if request_ids:
        #         log_parts.append(f"{len(request_ids)} individual requests")
        #     if batch_count > 0:
        #         log_parts.append(f"{batch_count} batch-level assignments")

        #     if log_parts:
        #         logger.info(
        #             f"📋 Periodic scheduling completed: {', '.join(log_parts)} "
        #             f"in {self._cluster_type.name} cluster"
        #             + (f", request_ids={request_ids}" if request_ids else "")
        #         )
        # else:
        #     logger.info(
        #         f"📋 Periodic scheduling completed: no requests to schedule in {self._cluster_type.name} cluster"
        #     )

        # Distribute requests to replica schedulers
        for replica_id, dp_id, request in self._request_mapping:
            self._dp_replica_set.add((replica_id, dp_id))
            # Only add individual requests to replica scheduler
            # Batch-level assignments (request=None) are already handled by the cluster scheduler
            if request is not None:
                cluster_scheduler.get_dp_replica_scheduler(
                    replica_id, dp_id
                ).add_request(request)

        # Create replica schedule events for affected replicas
        replica_events = [
            ReplicaScheduleEvent(self.time, replica_id, self._cluster_type, dp_id)
            for replica_id, dp_id in self._dp_replica_set
        ]

        # Create next periodic event with proper termination checks
        next_periodic_events = self._create_next_periodic_event_if_needed(scheduler)

        # Return both replica events and next periodic event (if any)
        return replica_events + next_periodic_events

    def _should_terminate_periodic_scheduling(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore = None
    ) -> bool:
        """
        Check if periodic scheduling should be terminated.

        This method implements multiple termination conditions to prevent
        infinite event generation and resource waste.

        Args:
            scheduler: Global scheduler instance
            metrics_store: Metrics store (unused but kept for interface consistency)

        Returns:
            True if periodic scheduling should stop, False otherwise
        """
        # Note: metrics_store parameter is unused but kept for interface consistency
        _ = metrics_store
        # Check 1: Simulation time limit
        # Use a reasonable default maximum simulation time
        # In practice, this should be configurable or derived from simulation config
        MAX_SIMULATION_TIME = 3600.0  # 1 hour in simulation time

        if self.time >= MAX_SIMULATION_TIME:
            return True

        # Check 2: Next event time would exceed reasonable bounds
        next_event_time = self.time + self._scheduling_interval_ms / 1000.0
        if next_event_time >= MAX_SIMULATION_TIME:
            return True

        # Check 3: Global scheduler termination signal
        # Check if the global scheduler has a termination flag
        if hasattr(scheduler, '_terminate') and getattr(scheduler, '_terminate', False):
            return True

        # Check 4: Cluster-specific termination conditions
        cluster_scheduler = scheduler.get_cluster_scheduler(self._cluster_type)

        # Check if cluster has been idle for too long (no requests for extended period)
        # This prevents infinite empty scheduling cycles
        queue_size = len(cluster_scheduler._request_queue)
        if queue_size == 0:
            # Implement adaptive idle detection
            idle_threshold_cycles = max(10, int(60.0 / (self._scheduling_interval_ms / 1000.0)))  # At least 60 seconds of idle time
            current_cycle = int(self.time / (self._scheduling_interval_ms / 1000.0))

            # Heuristic: if we've had many empty cycles and we're past initial startup,
            # consider terminating to avoid resource waste
            if current_cycle > idle_threshold_cycles and self.time > 5.0:  # After 5 seconds of simulation
                return True

        # Check 5: System-wide activity check
        # If all clusters are idle and no new requests are expected, terminate
        try:
            all_clusters_idle = True
            for _, cluster_sched in scheduler._cluster_schedulers.items():
                if len(cluster_sched._request_queue) > 0:
                    all_clusters_idle = False
                    break

            # In disaggregated mode, be more conservative about termination
            # Check if we're in disaggregated mode by looking at cluster types
            is_disaggregated = any(
                cluster_type.name in ['PREFILL', 'DECODE_ATTN', 'DECODE_FFN']
                for cluster_type in scheduler._cluster_schedulers.keys()
            )

            if is_disaggregated:
                # In disaggregated mode, use a much longer idle threshold to account for
                # KV cache transfers that may arrive later
                idle_threshold = 60.0  # 60 seconds
            else:
                # In monolithic mode, use the original threshold
                idle_threshold = 10.0  # 10 seconds

            # If all clusters are idle for a significant time, consider termination
            if all_clusters_idle and self.time > idle_threshold:
                return True

        except (AttributeError, KeyError):
            # If we can't access other cluster schedulers, continue with local checks only
            pass

        return False

    def _apply_frequency_control(self) -> None:
        """
        Apply frequency control to prevent simulator overload.

        This method introduces a small real-time delay to control the actual
        execution frequency of periodic events, preventing CPU overload.
        This delay is independent of simulation time progression.

        """

        # if self._scheduling_interval_ms < 1.0:
        #     # Very frequent scheduling: tiny delay to prevent tight loops
        #     sleep_time_ms = 0.01  # 0.01ms (10 microseconds) real-time delay
        # elif self._scheduling_interval_ms < 5.0:
        #     # Frequent scheduling: minimal proportional delay
        #     sleep_time_ms = 0.001 * self._scheduling_interval_ms  # 0.1% of interval
        # elif self._scheduling_interval_ms < 50.0:
        #     # Normal scheduling: very small fixed delay
        #     sleep_time_ms = 0.05  # 0.05ms real-time delay
        # else:
        #     # Infrequent scheduling: minimal delay
        #     sleep_time_ms = 0.01  # 0.01ms real-time delay

        # TODO: currently use a fixed value here.
        sleep_time_ms = 0.001
        # Only apply if delay is meaningful (> 0.001ms)
        # if sleep_time_ms > 0.001:
        time.sleep(sleep_time_ms / 1000.0)

        # Log frequency control application (debug level)
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug(f"Applied frequency control: {sleep_time_ms:.3f}ms real-time delay "
                        f"for {self._scheduling_interval_ms}ms simulation interval")

    def _create_next_periodic_event_if_needed(self, scheduler: BaseGlobalScheduler) -> List[BaseEvent]:
        """
        Create the next periodic scheduling event if termination conditions are not met.

        Returns:
            List containing the next PeriodicScheduleEvent, or empty list if should terminate
        """
        # Check termination conditions
        if self._should_terminate_periodic_scheduling(scheduler, None):
            return []

        # Calculate next event time
        next_event_time = self.time + self._scheduling_interval_ms / 1000.0

        # Validate next event time
        if next_event_time <= self.time:
            # This should never happen, but guard against it
            from frontier.logger import get_cluster_logger
            logger = get_cluster_logger(__name__, self._cluster_type.name)
            logger.error(f"Invalid next event time calculation: {next_event_time} <= {self.time}")
            return []

        # Create next periodic event
        next_periodic_event = PeriodicScheduleEvent(
            next_event_time,
            self._cluster_type,
            self._scheduling_interval_ms
        )

        return [next_periodic_event]

    def get_target_cluster(self) -> ClusterType:
        """Return the target cluster for this periodic scheduling event."""
        return self._cluster_type

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "cluster_type": str(self._cluster_type),
            "scheduling_interval_ms": self._scheduling_interval_ms,
            "replica_dp_set": list(getattr(self, '_dp_replica_set', [])),
            "request_mapping": [
                (replica_id, dp_id, request.id)
                for replica_id, dp_id, request in getattr(self, '_request_mapping', [])
            ],
        }
