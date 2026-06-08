"""
ClusterSimulator for per-cluster event processing in parallel mode.

This module implements the ClusterSimulator class that enables parallel processing
of events within individual clusters in Vidur's disaggregated architecture.
"""

import heapq
import threading
import time
from typing import Callable, List, Optional, TYPE_CHECKING

from frontier.logger import get_cluster_logger
from frontier.types import ClusterType
from frontier.utils.cluster_event_logger import ClusterEventLogger

if TYPE_CHECKING:
    from frontier.events import BaseEvent
    from frontier.metrics import MetricsStore
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import BaseClusterScheduler
    from frontier.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

logger = None  # Deprecated: instance-level cluster-tagged logger will be used


class ClusterSimulator:
    """
    Per-cluster event processor running in separate thread.
    
    This class manages event processing for a single cluster in parallel mode,
    maintaining its own event queue and processing events independently while
    coordinating with other clusters through inter-cluster communication.
    """
    
    def __init__(
        self,
        cluster_type: ClusterType,
        cluster_scheduler: "BaseClusterScheduler",
        global_scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
        enable_event_logging: bool = False,
        event_log_dir: str = "logs/cluster_events",
        event_log_level: str = "INFO",
        profiler=None,
        can_process_event_time: Optional[Callable[[ClusterType, float], bool]] = None,
    ):
        """
        Initialize the cluster simulator.

        Args:
            cluster_type: Type of the cluster (PREFILL, DECODE_ATTN, etc.)
            cluster_scheduler: Scheduler for this specific cluster
            global_scheduler: Global scheduler for routing events and inter-cluster communication
            metrics_store: Metrics store for recording performance data
            enable_event_logging: Whether to enable detailed event logging
            event_log_dir: Directory for event log files
            event_log_level: Log level for event logging (DEBUG, INFO, WARNING, ERROR)
            profiler: Performance profiler instance (optional)
        """
        self._cluster_type = cluster_type
        self._cluster_scheduler = cluster_scheduler
        self._global_scheduler = global_scheduler
        self._metrics_store = metrics_store
        self._profiler = profiler
        self._can_process_event_time = can_process_event_time
        self._parallel_coordination_lock = getattr(
            global_scheduler, "_parallel_coordination_lock", None
        )
        
        # Per-cluster event queue
        self._event_queue = []
        self._queue_lock = threading.Lock()
        
        # Thread management
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._local_time = 0.0
        self._fatal_error: Optional[Exception] = None  # Store fatal errors for main thread

        # Statistics
        self._events_processed = 0
        self._last_event_time = 0.0
        self._is_processing_event = False
        self._current_event_name: Optional[str] = None
        self._current_event_time: Optional[float] = None

        # Instance-scoped cluster-tagged logger
        self._logger = get_cluster_logger(__name__, cluster_type.name)

        # Event logging (conditional based on configuration)
        self._enable_event_logging = enable_event_logging
        self._event_logger = ClusterEventLogger(
            cluster_type,
            log_dir=event_log_dir,
            enabled=enable_event_logging,
            log_level=event_log_level
        )

        self._logger.info(f"ClusterSimulator initialized for {cluster_type.name}")
        if enable_event_logging:
            self._logger.info(f"Event logging ENABLED for {cluster_type.name}")
        else:
            self._logger.info(f"Event logging DISABLED for {cluster_type.name}")

    def start(self):
        """Start the cluster simulator in a separate thread."""
        if self._running:
            self._logger.warning(f"ClusterSimulator for {self._cluster_type.name} is already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name=f"ClusterSimulator-{self._cluster_type.name}",
            daemon=True
        )
        self._thread.start()
        self._logger.info(f"ClusterSimulator started for {self._cluster_type.name}")

    def stop(self):
        """Stop the cluster simulator."""
        if not self._running:
            return
        
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        # Write event processing summary to log file
        stats = self._event_logger.write_summary()

        self._logger.info(f"ClusterSimulator stopped for {self._cluster_type.name}. "
                          f"Processed {self._events_processed} events.")
        self._logger.info(f"Event log summary: {stats['total_events']} events, "
                          f"{stats['total_errors']} errors, {stats['events_per_second']:.2f} events/sec")

    def add_event(self, event: "BaseEvent"):
        """
        Add an event to this cluster's event queue.
        
        Args:
            event: Event to be processed by this cluster
        """
        with self._queue_lock:
            heapq.heappush(self._event_queue, (event._priority_number, event))

        self._logger.debug(f"Event {event.__class__.__name__} added to {self._cluster_type.name} queue")

    def add_events(self, events: List["BaseEvent"]):
        """
        Add multiple events to this cluster's event queue.

        Args:
            events: List of events to be processed by this cluster
        """
        with self._queue_lock:
            for event in events:
                heapq.heappush(self._event_queue, (event._priority_number, event))

        if events:
            self._logger.debug(f"{len(events)} events added to {self._cluster_type.name} queue")

    def has_events(self) -> bool:
        """Check if this cluster has pending events."""
        with self._queue_lock:
            return len(self._event_queue) > 0
    
    def get_queue_size(self) -> int:
        """Get the current size of the event queue."""
        with self._queue_lock:
            return len(self._event_queue)
    
    def get_local_time(self) -> float:
        """Get the current local time of this cluster."""
        return self._local_time

    def is_processing_event(self) -> bool:
        """Return whether the cluster thread is currently inside an event handler."""
        return self._is_processing_event

    def get_runtime_state(self) -> dict:
        """Return lightweight runtime state for deadlock diagnostics."""
        return {
            "cluster_type": self._cluster_type.name,
            "queue_size": self.get_queue_size(),
            "local_time": self._local_time,
            "is_running": self._running,
            "is_processing_event": self._is_processing_event,
            "current_event_name": self._current_event_name,
            "current_event_time": self._current_event_time,
            "next_event_time": self.peek_next_event_time(),
        }
    
    def _run_event_loop(self):
        """
        Main event processing loop running in separate thread.

        This method continuously processes events from the cluster's event queue,
        handles inter-cluster communication, and maintains local time.

        NOTE: Exception handling has been removed to allow errors to propagate
        immediately during development. Any exception will terminate this thread
        and should be caught by the main monitoring loop.
        """
        self._logger.info(f"Event loop started for {self._cluster_type.name}")

        try:
            while self._running:
                event = self._claim_next_event()
                if event is None:
                    # No events to process, sleep briefly
                    time.sleep(0.001)  # 1ms sleep
                    continue

                # Update local time
                self._set_local_time(event.time)
                self._is_processing_event = True
                self._current_event_name = event.__class__.__name__
                self._current_event_time = event.time
                try:
                    # Process the event with or without detailed logging based on configuration
                    if self._enable_event_logging:
                        new_events = self._handle_event_with_logging(event)
                    else:
                        new_events = self._handle_event(event)

                    # Route new events to appropriate clusters
                    self._route_events(new_events)

                    # Update statistics
                    self._events_processed += 1
                    self._last_event_time = event.time
                finally:
                    self._current_event_name = None
                    self._is_processing_event = False
                    self._current_event_time = None

        except Exception as e:
            # Fatal error: log and store for main thread to detect
            self._logger.error(f"FATAL ERROR in {self._cluster_type.name} event loop: {type(e).__name__}: {e}", exc_info=True)

            # Store exception for main thread
            self._fatal_error = e
            self._running = False

            # Log to event logger if available
            if self._enable_event_logging and 'event' in locals():
                self._event_logger.log_event_error(
                    event.__class__.__name__,
                    str(getattr(event, '_id', 'unknown')),
                    str(e)
                )

            # Re-raise to ensure thread terminates with exception
            raise

        finally:
            self._logger.info(f"Event loop ended for {self._cluster_type.name}")

    def _get_next_event(self) -> Optional["BaseEvent"]:
        """Get the next event from the queue."""
        with self._queue_lock:
            if not self._event_queue:
                return None
            _, event = heapq.heappop(self._event_queue)
            return event

    def peek_next_event_time(self) -> Optional[float]:
        """Peek the earliest local event time without removing the event."""
        with self._queue_lock:
            if not self._event_queue:
                return None
            return float(self._event_queue[0][0][0])

    def _claim_next_event(self) -> Optional["BaseEvent"]:
        """
        Claim the next event if no peer cluster can still emit an earlier event.

        In parallel PD online mode, target clusters can otherwise pop a future local
        event before another cluster finishes a smaller-time event that routes an
        earlier inter-cluster message. The coordination lock plus frontier-time gate
        makes the local pop observe a stable snapshot of peer frontiers.
        """
        coordination_lock = self._parallel_coordination_lock
        if coordination_lock is None:
            self._process_incoming_events()
            return self._get_next_event()

        with coordination_lock:
            self._process_incoming_events()
            next_event_time = self.peek_next_event_time()
            if next_event_time is None:
                return None

            if (
                self._can_process_event_time is not None
                and not self._can_process_event_time(self._cluster_type, next_event_time)
            ):
                return None

            self._process_incoming_events()
            refreshed_next_event_time = self.peek_next_event_time()
            if refreshed_next_event_time is None:
                return None
            if refreshed_next_event_time < next_event_time:
                next_event_time = refreshed_next_event_time
                if (
                    self._can_process_event_time is not None
                    and not self._can_process_event_time(
                        self._cluster_type, next_event_time
                    )
                ):
                    return None

            return self._get_next_event()

    def _set_local_time(self, new_time: float):
        """Update the local time of this cluster."""
        self._local_time = max(self._local_time, new_time)
        self._global_scheduler.update_cluster_logical_time(
            self._cluster_type, self._local_time
        )
    
    def _handle_event_with_logging(self, event: "BaseEvent") -> List["BaseEvent"]:
        """
        Handle an event with detailed logging.

        Args:
            event: Event to be processed

        Returns:
            List of new events generated by processing this event

        NOTE: Exception handling has been removed to allow errors to propagate
        immediately during development. Event logging will still record the error
        before the exception propagates.
        """
        event_type = event.__class__.__name__
        event_id = str(getattr(event, '_id', f"{event_type}_{self._events_processed}"))

        # Extract event details for logging
        event_details = self._extract_event_details(event)

        # Log event start
        self._event_logger.log_event_start(event_type, event_id, event_details)

        start_time = time.perf_counter()
        try:
            # Use existing event handling mechanism
            new_events = event.handle_event(self._global_scheduler, self._metrics_store)

            # Calculate processing duration
            duration = time.perf_counter() - start_time
            duration_ms = duration * 1000

            # Log event completion
            # Include the same correlation keys as START for robust parsing.
            completion_details = dict(event_details)
            completion_details.update({
                'new_events_generated': len(new_events) if new_events else 0,
                'cluster_time': self._local_time,
                'event_time': event.time,
            })
            self._event_logger.log_event_complete(event_type, event_id, duration_ms, completion_details)

            # Profile event handling if profiler is available
            if self._profiler and self._profiler.enabled:
                self._profiler.record_event_processing(event_type, duration)
                self._profiler.record_cluster_operation(
                    self._cluster_type.name,
                    f"event_{event_type}",
                    duration
                )

            return new_events if new_events else []

        except Exception as e:
            # Log event error before re-raising
            duration = time.perf_counter() - start_time
            duration_ms = duration * 1000
            self._event_logger.log_event_error(event_type, event_id, str(e), event_details)

            # Log to cluster logger as well
            if self._event_logger.is_enabled():
                self._event_logger.get_logger().error(
                    f"Error handling event {event_type} in {self._cluster_type.name}: {e}",
                    exc_info=True
                )
            else:
                self._logger.error(
                    f"Error handling event {event_type} in {self._cluster_type.name}: {e}",
                    exc_info=True
                )

            # Re-raise to propagate the exception
            raise

    def _handle_event(self, event: "BaseEvent") -> List["BaseEvent"]:
        """
        Handle an event using existing event processing logic (without detailed logging).

        Args:
            event: Event to be processed

        Returns:
            List of new events generated by processing this event

        NOTE: Exception handling has been removed to allow errors to propagate
        immediately during development. Any exception will be caught by the
        outer _run_event_loop() and terminate the thread.
        """
        # Use existing event handling mechanism
        # No try-except: let exceptions propagate naturally

        # Profile event handling if profiler is available
        if self._profiler and self._profiler.enabled:
            event_type_name = event.__class__.__name__
            start_time = time.perf_counter()
            new_events = event.handle_event(self._global_scheduler, self._metrics_store)
            duration = time.perf_counter() - start_time
            self._profiler.record_event_processing(event_type_name, duration)
            self._profiler.record_cluster_operation(
                self._cluster_type.name,
                f"event_{event_type_name}",
                duration
            )
        else:
            new_events = event.handle_event(self._global_scheduler, self._metrics_store)

        return new_events if new_events else []

    def _extract_event_details(self, event: "BaseEvent") -> dict:
        """Extract relevant details from an event for logging."""
        details = {
            'event_time': event.time,
            'cluster': self._cluster_type.name
        }

        # Helper: best-effort get batch object
        batch = None
        if hasattr(event, '_batch'):
            batch = getattr(event, '_batch', None)
        elif hasattr(event, 'batch'):
            batch = getattr(event, 'batch', None)
        elif hasattr(event, '_transfer_info'):
            transfer_info = getattr(event, '_transfer_info', None)
            batch = getattr(transfer_info, 'batch', None) if transfer_info is not None else None

        # Helper: best-effort get transfer_info object
        transfer_info = getattr(event, '_transfer_info', None)

        # Extract batch information if available
        if batch is not None:
            # Required Phase 2 correlation keys (best-effort; keep backward compatible)
            try:
                details['batch_id'] = getattr(batch, 'id', getattr(batch, '_id', 'unknown'))
            except Exception:
                details['batch_id'] = 'unknown'

            try:
                details['batch_global_id'] = getattr(batch, 'global_id', 'unknown')
            except Exception:
                details['batch_global_id'] = 'unknown'

            # request_ids
            try:
                details['request_ids'] = getattr(batch, 'request_ids', [req.id for req in batch.requests])
            except Exception:
                pass

            # layer_id: prefer explicit event/transfer info; fallback to batch AF layer count
            layer_id = None
            if hasattr(event, '_layer_id'):
                layer_id = getattr(event, '_layer_id', None)
            if layer_id is None and transfer_info is not None:
                layer_id = getattr(transfer_info, 'layer_id', None)
            if layer_id is None:
                try:
                    layer_id = getattr(batch, 'af_inflight_layer_count')
                except Exception:
                    layer_id = None
            if layer_id is not None:
                details['layer_id'] = layer_id

            # Per-request decode/layer state (pre-event snapshot, consistent with START log semantics)
            try:
                if getattr(batch, 'requests', None):
                    request_decode_steps = [
                        getattr(req, 'current_decode_token_index', 'unknown')
                        for req in batch.requests
                    ]
                    request_layer_ids = [
                        getattr(req, 'completed_layer_count', 'unknown')
                        for req in batch.requests
                    ]
                    details['request_decode_steps'] = request_decode_steps
                    details['request_layer_ids'] = request_layer_ids
                    details['decode_step'] = request_decode_steps[0]
            except Exception:
                pass

            # Replica/DP IDs: prefer event lane; fallback to decode-attn original lane on batch
            try:
                if hasattr(event, '_replica_id'):
                    details['replica_id'] = getattr(event, '_replica_id')
                elif hasattr(batch, 'decode_attn_original_replica_id') and batch.decode_attn_original_replica_id is not None:
                    details['replica_id'] = batch.decode_attn_original_replica_id
            except Exception:
                pass

            try:
                if hasattr(event, '_dp_id'):
                    details['dp_id'] = getattr(event, '_dp_id')
                elif hasattr(batch, 'decode_attn_original_dp_id') and batch.decode_attn_original_dp_id is not None:
                    details['dp_id'] = batch.decode_attn_original_dp_id
            except Exception:
                pass

            # Optional batch stats
            details.update({
                'batch_size': getattr(batch, 'size', 'unknown'),
                'num_tokens': getattr(batch, 'total_num_tokens', 'unknown'),
            })

            # Log batch info separately for better tracking
            self._event_logger.log_batch_info(
                details['batch_id'],
                details['batch_size'] if details['batch_size'] != 'unknown' else 0,
                details['num_tokens'] if details['num_tokens'] != 'unknown' else 0,
                details.get('replica_id', -1) if details.get('replica_id', 'unknown') != 'unknown' else -1
            )

        # M2N transfer direction
        if transfer_info is not None:
            try:
                details['is_attn_to_ffn'] = bool(getattr(transfer_info, 'is_attn_to_ffn'))
            except Exception:
                pass
            try:
                if hasattr(transfer_info, 'activation_size_bytes'):
                    details['activation_size_bytes'] = int(transfer_info.activation_size_bytes)
            except Exception:
                pass
            try:
                if hasattr(transfer_info, 'kv_cache_size_bytes'):
                    details['kv_cache_size_bytes'] = int(transfer_info.kv_cache_size_bytes)
            except Exception:
                pass
            # Prefer decode-attn original mapping for multi-replica disambiguation
            try:
                b = getattr(transfer_info, 'batch', None)
                if b is not None:
                    if getattr(b, 'decode_attn_original_replica_id', None) is not None:
                        details['replica_id'] = b.decode_attn_original_replica_id
                    if getattr(b, 'decode_attn_original_dp_id', None) is not None:
                        details['dp_id'] = b.decode_attn_original_dp_id
            except Exception:
                pass
            # Fallback to transfer_info source ids
            try:
                if 'replica_id' not in details and hasattr(transfer_info, 'source_replica_id'):
                    details['replica_id'] = transfer_info.source_replica_id
                if 'dp_id' not in details and hasattr(transfer_info, 'source_dp_id'):
                    details['dp_id'] = transfer_info.source_dp_id
            except Exception:
                pass

        # Best-effort direction for legacy transfer START events (no transfer_info yet).
        if 'is_attn_to_ffn' not in details and hasattr(event, '_source_cluster_type'):
            try:
                details['is_attn_to_ffn'] = bool(getattr(event, '_source_cluster_type').name == "DECODE_ATTN")
            except Exception:
                pass
        # Transfer size fields (best-effort)
        try:
            if hasattr(event, '_kv_cache_size_bytes'):
                details['kv_cache_size_bytes'] = int(getattr(event, '_kv_cache_size_bytes'))
        except Exception:
            pass
        try:
            if hasattr(event, '_activation_size_bytes'):
                details['activation_size_bytes'] = int(getattr(event, '_activation_size_bytes'))
        except Exception:
            pass

        # Extract replica information if available
        if hasattr(event, 'replica_id'):
            details['replica_id'] = event.replica_id

        # Extract pipeline stage information if available
        if hasattr(event, 'pipeline_stage'):
            details['pipeline_stage'] = event.pipeline_stage

        return details
    
    def _route_events(self, events: List["BaseEvent"]):
        """
        Route events to appropriate clusters.

        Args:
            events: List of events to be routed
        """
        for event in events:
            target_cluster = self._determine_target_cluster(event)

            if target_cluster == self._cluster_type:
                # Event belongs to this cluster
                self.add_event(event)
            else:
                # Event belongs to another cluster - route via GlobalScheduler
                self._global_scheduler.route_event_to_cluster(event, target_cluster)
    
    def _determine_target_cluster(self, event: "BaseEvent") -> ClusterType:
        """
        Determine which cluster should process this event.
        
        Args:
            event: Event to be routed
            
        Returns:
            Target cluster type for this event
        """
        # Use the event's get_target_cluster method if available
        if hasattr(event, 'get_target_cluster'):
            return event.get_target_cluster()
        
        # Fallback to cluster type attribute if available
        if hasattr(event, '_cluster_type'):
            return event._cluster_type
        
        # Default fallback - route to current cluster
        self._logger.warning(f"Could not determine target cluster for {event.__class__.__name__}, "
                             f"routing to current cluster {self._cluster_type.name}")
        return self._cluster_type

    def _process_incoming_events(self):
        """Process events sent from other clusters via GlobalScheduler."""
        incoming_events = self._global_scheduler.get_events_for_cluster(self._cluster_type)
        if incoming_events:
            self.add_events(incoming_events)
            self._logger.debug(f"Received {len(incoming_events)} events from other clusters "
                               f"for {self._cluster_type.name}")

    def get_statistics(self) -> dict:
        """Get statistics about this cluster simulator."""
        return {
            "cluster_type": self._cluster_type.name,
            "events_processed": self._events_processed,
            "queue_size": self.get_queue_size(),
            "local_time": self._local_time,
            "last_event_time": self._last_event_time,
            "is_running": self._running,
        }
