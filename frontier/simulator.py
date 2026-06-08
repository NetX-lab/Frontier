import atexit
import heapq
import json
import os
import threading
import time
from typing import Dict, List, Optional

from frontier.config import (
    DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR,
    SimulationConfig,
    get_quantization_manager,
    global_vars,
)
from frontier.entities import Cluster
from frontier.events import (
    BaseEvent,
    RequestArrivalEvent,
    ReplicaScheduleEvent,
    GlobalScheduleEvent,
)
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.metrics.trace_store import TraceStore
from frontier.request_generator import RequestGeneratorRegistry
from frontier.scheduler.global_scheduler.base_global_scheduler import (
    BaseGlobalScheduler,
)
from frontier.types import ClusterType
from frontier.execution_time_predictor import ExecutionTimePredictorRegistry
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.utils.cluster_event_logger import ClusterEventLogger
from frontier.utils.performance_profiler import PerformanceProfiler


logger = init_logger(__name__)


class Simulator:
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config
        if self._config.is_disaggregated_mode():
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

        self._time = 0
        self._terminate = False
        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue = []
        self._sequential_event_loggers = None
        self._sequential_log_cluster_type = None

        self._event_trace = []
        self._event_chrome_trace = []
        self._all_requests = []
        self._requests_by_session_id = {}
        self._checkpoint_export_completed = False
        self._checkpoint_expected_session_ids = (
            self._load_checkpoint_expected_session_ids()
        )

        self._clusters: Dict[ClusterType, Cluster] = {}

        cluster_configs = self._config.get_clusters()
        model_configs = {
            cluster_config.replica_config.model_config.get_name(): cluster_config.replica_config.model_config
            for cluster_config in cluster_configs.values()
        }
        if len(model_configs) != 1:
            raise ValueError(
                "All clusters must share the same model config for quantization setup. "
                f"Found: {sorted(model_configs.keys())}"
            )
        model_config = next(iter(model_configs.values()))

        self._quantization_manager = get_quantization_manager()
        self._quantization_manager.configure_from_model_config(model_config)
        self._quantization_manager.print_config_summary()
        global_vars.set_quantization_manager(self._quantization_manager)

        # Initialize performance profiler
        self._profiler = PerformanceProfiler(
            enabled=config.enable_performance_profiling
        )
        if config.enable_performance_profiling:
            logger.info("Performance profiling enabled")

        # Create cluster event logs directory only if event logging is enabled
        if self._config.enable_cluster_event_logging:
            self._cluster_logs_dir = self._config.cluster_event_log_dir
            os.makedirs(self._cluster_logs_dir, exist_ok=True)
            logger.info(
                f"Created cluster event logs directory: {self._cluster_logs_dir}"
            )
        else:
            self._cluster_logs_dir = None
            logger.info(f"Cluster event logging disabled - no log directory created")

        logger.info("Simulation mode: %s", self._config.simulation_mode)
        # Co-location (monolithic) mode is now supported
        if not self._config.is_disaggregated_mode():
            logger.info("Running in co-location (monolithic) mode")

        for cluster_type, cluster_config in cluster_configs.items():
            self._clusters[cluster_type] = Cluster(
                cluster_config,
                self._config.metrics_config,
                self._config.request_generator_config,
            )

        # Initialize TraceStore if enabled
        self._trace_store = None
        if self._config.metrics_config.enable_op_level_tracing:
            self._trace_store = TraceStore(
                output_dir=self._config.metrics_config.output_dir,
                enabled=True,
                filename=self._config.metrics_config.trace_output_file,
            )
            # Initialize with simulation metadata
            self._trace_store.initialize(
                {
                    "sys_arch": self._config.sys_arch,
                    "simulation_mode": self._config.simulation_mode,
                    "clusters": [c.name for c in self._clusters.keys()],
                }
            )
            logger.info("Op-level tracing enabled")

        self._metric_store = MetricsStore(
            simulation_config=self._config,
            cluster_configs=cluster_configs,
            trace_store=self._trace_store,
        )
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )
        self._request_generator.configure_thinking_mode(
            enable_thinking_mode=self._config.enable_thinking_mode,
            thinking_depth=self._config.thinking_depth,
            tool_call_latency=self._config.tool_call_latency,
            thinking_round_prefill_tokens=self._config.thinking_round_prefill_tokens,
            thinking_round_decode_tokens=self._config.thinking_round_decode_tokens,
        )

        # Initialize shared execution time prediction model manager and predictors for each cluster
        self._predictors = {}
        if self._config.is_disaggregated_mode():
            # Create shared model manager that trains prediction models once for all clusters
            self._execution_time_prediction_model_manager = (
                ExecutionTimePredictionModelManager(
                    cluster_configs, self._config.metrics_config
                )
            )

            # Create individual predictors for each cluster
            for cluster_type, cluster_config in cluster_configs.items():
                # Get CC Backend from the cluster for communication predictions
                cluster = self._clusters[cluster_type]
                cc_backend = cluster.cc_backend

                self._predictors[cluster_type] = ExecutionTimePredictorRegistry.get(
                    cluster_config.execution_time_predictor_config.get_type(),
                    predictor_config=cluster_config.execution_time_predictor_config,
                    replica_config=cluster_config.replica_config,
                    replica_scheduler_config=cluster_config.replica_scheduler_config,
                    metrics_config=self._config.metrics_config,
                    cluster_config=self._config.cluster_config,
                    model_manager=self._execution_time_prediction_model_manager,
                    cluster_type=cluster_type,
                    training_file_paths=self._execution_time_prediction_model_manager.get_training_file_paths(
                        cluster_type
                    ),
                    actual_replica_ids=list(
                        self._clusters[cluster_type].replicas.keys()
                    ),
                    cc_backend=cc_backend,
                )
        else:
            # For monolithic mode, create single predictor without model manager
            cluster_config = cluster_configs[ClusterType.MONOLITHIC]
            # Get CC Backend from the monolithic cluster
            cluster = self._clusters[ClusterType.MONOLITHIC]
            cc_backend = cluster.cc_backend

            self._predictors[ClusterType.MONOLITHIC] = (
                ExecutionTimePredictorRegistry.get(
                    cluster_config.execution_time_predictor_config.get_type(),
                    predictor_config=cluster_config.execution_time_predictor_config,
                    replica_config=cluster_config.replica_config,
                    replica_scheduler_config=cluster_config.replica_scheduler_config,
                    metrics_config=self._config.metrics_config,
                    cluster_config=self._config.cluster_config,
                    model_manager=None,
                    cluster_type=ClusterType.MONOLITHIC,
                    cc_backend=cc_backend,
                )
            )

        # In disaggregated mode, global scheduler gets all clusters with their predictors.
        # In monolithic mode, it gets a dict with one cluster and predictor.
        # Enable parallel mode in GlobalScheduler if configured
        enable_parallel = (
            self._config.enable_parallel_clusters
            and self._config.is_disaggregated_mode()
            and len(self._clusters) > 1
        )

        self._global_scheduler = BaseGlobalScheduler(
            self._clusters,
            self._config.request_generator_config,
            predictors=self._predictors,
            enable_parallel_mode=enable_parallel,
            max_inter_cluster_queue_size=self._config.max_inter_cluster_queue_size,
        )
        logger.info(
            f"Simulator: Finish basic initialization. Starting simulation with {len(self._clusters)} clusters"
        )

        # Log cluster configuration details
        for cluster_type, cluster in self._clusters.items():
            logger.info(
                f"Cluster {cluster_type.name}: {cluster._config.num_replicas} replicas, "
                f"device={cluster._config.replica_config.device}"
            )

        # Log parallel mode status
        if enable_parallel:
            logger.info("Parallel cluster processing mode enabled")
        else:
            logger.info("Sequential processing mode enabled")

        # Initialize simulation mode (parallel or sequential)
        self._init_simulation_mode()

    @property
    def scheduler(self) -> BaseGlobalScheduler:
        return self._global_scheduler

    @property
    def metric_store(self) -> MetricsStore:
        return self._metric_store

    def _init_simulation_mode(self):
        """Initialize simulation mode based on configuration."""
        if (
            self._config.enable_parallel_clusters
            and self._config.is_disaggregated_mode()
            and len(self._clusters) > 1
        ):
            self._init_parallel_mode()
        else:
            self._init_sequential_mode()

    def _init_parallel_mode(self):
        """Disaggregated parallel cluster mode is not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _can_parallel_cluster_process_event(
        self,
        cluster_type: ClusterType,
        event_time: float,
    ) -> bool:
        """
        Conservatively gate parallel event claims against peer cluster frontiers.

        Without this guard, a cluster can pop a future local event while another
        cluster is still processing or queueing a smaller-time event that will route
        an earlier inter-cluster message. That breaks the sequential DES ordering and
        makes parallel online PD numerically unstable.
        """
        for other_cluster_type, other_cluster_simulator in self._cluster_simulators.items():
            if other_cluster_type == cluster_type:
                continue

            state = other_cluster_simulator.get_runtime_state()
            current_event_time = state.get("current_event_time")
            if (
                state.get("is_processing_event")
                and current_event_time is not None
                and float(current_event_time) < event_time
            ):
                return False

            next_event_time = state.get("next_event_time")
            if next_event_time is not None and float(next_event_time) < event_time:
                return False

        return True

    def _init_sequential_mode(self):
        """Initialize sequential processing mode (existing behavior)."""
        logger.info("Initializing sequential processing mode")

        # Use existing sequential processing
        self._event_queue = []
        self._parallel_mode = False

        # Initialize per-cluster event loggers for sequential mode if enabled.
        if self._config.enable_cluster_event_logging:
            self._sequential_event_loggers = {}
            for cluster_type in self._clusters.keys():
                self._sequential_event_loggers[cluster_type] = ClusterEventLogger(
                    cluster_type=cluster_type,
                    log_dir=self._config.cluster_event_log_dir,
                    enabled=True,
                    log_level=self._config.cluster_event_log_level,
                )
            if len(self._sequential_event_loggers) == 1:
                self._sequential_log_cluster_type = next(
                    iter(self._sequential_event_loggers.keys())
                )
            logger.info("Sequential cluster event logging ENABLED")
        else:
            self._sequential_event_loggers = None
            self._sequential_log_cluster_type = None
            logger.info("Sequential cluster event logging DISABLED")

        # Initialize events for sequential mode
        self._init_event_queue()

        logger.info("Sequential mode initialized")

    def _init_parallel_events(self):
        """Initialize events for parallel mode."""
        requests = self._request_generator.generate()
        logger.info(f"Generated {len(requests)} requests for parallel processing")
        # Register total requests for completion-based termination.
        # Registration errors must propagate so request completion accounting
        # cannot silently drift from the generated workload.
        self._metric_store.register_total_requests(len(requests))

        # Initialize periodic scheduling events first
        periodic_events = self._global_scheduler.initialize_periodic_scheduling(
            start_time=0.0
        )
        for event in periodic_events:
            target_cluster = event.get_target_cluster()
            if target_cluster in self._cluster_simulators:
                self._cluster_simulators[target_cluster].add_event(event)
                logger.info(
                    f"Added periodic scheduling event to {target_cluster.name} cluster"
                )
            else:
                logger.warning(
                    f"Target cluster {target_cluster} not found for periodic scheduling event"
                )

        if (
            self._config.simulation_mode == "offline"
            and self._config.is_disaggregated_mode()
            and not getattr(
                self._config, "offline_use_generated_request_arrivals", False
            )
        ):
            # In offline mode, all requests are added to the scheduler at time 0.
            for request in requests:
                request.set_arrived_at(0)
                request.on_arrival(0, ClusterType.PREFILL)
                # Record request arrival metrics (histogram data: num_tokens, prefill_tokens, decode_tokens, etc.)
                # This must be called to ensure metrics are recorded correctly
                self._metric_store.on_request_arrival(0, request, ClusterType.PREFILL)
                self._global_scheduler.add_request(
                    request=request, cluster_type=ClusterType.PREFILL
                )

            # Create initial global schedule event and route it to prefill cluster
            initial_event = GlobalScheduleEvent(0)
            target_cluster = initial_event.get_target_cluster()

            if target_cluster in self._cluster_simulators:
                self._cluster_simulators[target_cluster].add_event(initial_event)
            else:
                logger.warning(
                    f"Target cluster {target_cluster} not found for initial event"
                )
        else:
            # Handle online mode for parallel processing
            for request in requests:
                cluster_type = (
                    ClusterType.PREFILL
                    if self._config.is_disaggregated_mode()
                    else ClusterType.MONOLITHIC
                )
                arrival_event = RequestArrivalEvent(
                    request.arrived_at, request, cluster_type
                )

                if cluster_type in self._cluster_simulators:
                    self._cluster_simulators[cluster_type].add_event(arrival_event)
                else:
                    logger.warning(
                        f"Cluster {cluster_type} not found for request arrival event"
                    )

    def run(self) -> None:
        """Run the simulation in either parallel or sequential mode."""
        if self._parallel_mode:
            self._run_parallel()
        else:
            self._run_sequential()

    def _run_parallel(self) -> None:
        """Run simulation in parallel mode with multiple cluster threads."""
        logger.info("Starting parallel simulation")

        try:
            # Start all cluster simulators
            for cluster_type, cluster_simulator in self._cluster_simulators.items():
                cluster_simulator.start()
                logger.info(f"Started cluster simulator for {cluster_type.name}")

            # Monitor simulation progress
            self._monitor_parallel_simulation()

        finally:
            # Stop all cluster simulators and collect event statistics (if enabled)
            if self._config.enable_cluster_event_logging:
                self._print_cluster_event_statistics()

            for cluster_type, cluster_simulator in self._cluster_simulators.items():
                cluster_simulator.stop()
                logger.info(f"Stopped cluster simulator for {cluster_type.name}")

        logger.info(f"Parallel simulation ended at: {self._time}s. Writing output...")

        # Close trace store if it exists
        if self._trace_store:
            self._trace_store.close()

        self._write_output()

    def _print_cluster_event_statistics(self):
        """Print detailed event processing statistics for all clusters."""
        if not self._config.enable_cluster_event_logging:
            logger.info(
                "Cluster event logging disabled - no detailed statistics available"
            )
            return

        logger.info("=" * 80)
        logger.info("CLUSTER EVENT PROCESSING STATISTICS")
        logger.info("=" * 80)

        total_events = 0
        total_errors = 0

        for cluster_type, cluster_simulator in self._cluster_simulators.items():
            if hasattr(cluster_simulator, "_event_logger"):
                stats = cluster_simulator._event_logger.get_statistics()
                total_events += stats["total_events"]
                total_errors += stats["total_errors"]

                logger.info(f"{cluster_type.name} Cluster:")
                logger.info(f"   Events Processed: {stats['total_events']:,}")
                logger.info(f"   Errors: {stats['total_errors']}")
                logger.info(f"   Processing Time: {stats['total_time_seconds']:.3f}s")
                logger.info(f"   Events/Second: {stats['events_per_second']:.2f}")
                logger.info(f"   Error Rate: {stats['error_rate']:.4f}")
                logger.info(f"   Log File: {stats['log_file']}")

                # Print top event types
                if stats["event_type_stats"]:
                    logger.info(f"   Top Event Types:")
                    sorted_events = sorted(
                        stats["event_type_stats"].items(),
                        key=lambda x: x[1]["count"],
                        reverse=True,
                    )[:5]  # Top 5 event types

                    for event_type, type_stats in sorted_events:
                        logger.info(
                            f"     {event_type}: {type_stats['count']} events, {type_stats['errors']} errors"
                        )

                logger.info("-" * 60)

        logger.info(f"TOTAL ACROSS ALL CLUSTERS:")
        logger.info(f"   Total Events: {total_events:,}")
        logger.info(f"   Total Errors: {total_errors}")
        logger.info(
            f"   Overall Error Rate: {total_errors / total_events:.4f}"
            if total_events > 0
            else "   Overall Error Rate: 0.0000"
        )
        logger.info("=" * 80)

    def _build_sequential_scheduler_state_report(self) -> str:
        """Build a structured report for non-empty sequential scheduler exits."""
        if not hasattr(self._global_scheduler, "_cluster_schedulers"):
            raise RuntimeError(
                "Global scheduler missing _cluster_schedulers for sequential diagnostics"
            )

        cluster_states = []
        for cluster_type, cluster_scheduler in sorted(
            self._global_scheduler._cluster_schedulers.items(),
            key=lambda item: item[0].name,
        ):
            if not hasattr(cluster_scheduler, "get_debug_state"):
                raise RuntimeError(
                    f"Cluster scheduler {cluster_type.name} missing get_debug_state()"
                )
            cluster_states.append(
                {
                    "cluster_key": cluster_type.name,
                    "state": cluster_scheduler.get_debug_state(),
                }
            )

        payload = {
            "message": "Sequential simulation ended with non-empty scheduler state",
            "simulation_time": self._time,
            "terminate": self._terminate,
            "event_queue_length": len(self._event_queue),
            "global_scheduler_is_empty": self._global_scheduler.is_empty,
            "clusters": cluster_states,
        }
        return (
            "Sequential simulation ended with non-empty scheduler state:\n"
            + json.dumps(payload, indent=2, sort_keys=True, default=str)
        )

    def _try_promote_terminal_pdaf_scheduler_work(self) -> bool:
        """Promote terminal PD-AF DECODE_FFN groups when no event can fill them.

        This is a fail-fast recovery for sequential mode only: it should run
        after the event queue becomes empty, so missing lanes are proven absent
        for the current terminal group rather than merely delayed.
        """
        from frontier.events.cluster_schedule_event import ClusterScheduleEvent
        from frontier.logger import get_cluster_logger
        from frontier.types import ClusterType

        cluster_schedulers = getattr(self._global_scheduler, "_cluster_schedulers", {})
        decode_ffn_scheduler = cluster_schedulers.get(ClusterType.DECODE_FFN)
        if decode_ffn_scheduler is None:
            return False
        waiting_groups = getattr(decode_ffn_scheduler, "_m2n_waiting_by_layer", None)
        ready_groups = getattr(decode_ffn_scheduler, "_m2n_ready_groups", None)
        if not waiting_groups and not ready_groups:
            return False
        promote = getattr(
            decode_ffn_scheduler,
            "_promote_incomplete_m2n_groups_with_idle_lanes",
            None,
        )
        if promote is None:
            raise RuntimeError(
                "DECODE_FFN scheduler cannot promote terminal incomplete M2N groups"
            )

        logger_instance = get_cluster_logger(__name__, ClusterType.DECODE_FFN.name)
        promoted_count = promote(logger_instance) if waiting_groups else 0
        ready_group_count = len(getattr(decode_ffn_scheduler, "_m2n_ready_groups", []))
        if promoted_count <= 0 and ready_group_count <= 0:
            return False

        logger.info(
            "Scheduling terminal PD-AF DECODE_FFN work: "
            f"promoted_count={promoted_count}, "
            f"ready_group_count={ready_group_count}, time={self._time}"
        )
        for _ in range(ready_group_count):
            self._add_event(ClusterScheduleEvent(self._time, ClusterType.DECODE_FFN))
        return True

    def _run_sequential(self) -> None:
        """Run simulation in sequential mode (existing behavior)."""
        import time as time_module

        event_processing_start = time_module.perf_counter()

        while not self._terminate:
            while self._event_queue and not self._terminate:
                _, event = heapq.heappop(self._event_queue)
                self._set_time(event._time)

                # Profile event handling
                event_type_name = event.__class__.__name__
                event_logger = None
                event_details = None
                log_cluster_type = None
                if self._sequential_event_loggers:
                    if self._sequential_log_cluster_type is not None:
                        log_cluster_type = self._sequential_log_cluster_type
                    else:
                        log_cluster_type = event.get_target_cluster()
                        if log_cluster_type not in self._sequential_event_loggers:
                            raise RuntimeError(
                                f"Missing event logger for cluster {log_cluster_type}"
                            )

                    event_logger = self._sequential_event_loggers[log_cluster_type]
                    event_details = self._extract_sequential_event_details(
                        event, log_cluster_type
                    )
                    event_logger.log_event_start(
                        event_type_name,
                        str(event.id),
                        details=event_details,
                    )

                event_start = time_module.perf_counter()
                try:
                    new_events = event.handle_event(
                        self._global_scheduler, self._metric_store
                    )
                except Exception as exc:
                    if event_logger is not None:
                        event_logger.log_event_error(
                            event_type_name,
                            str(event.id),
                            f"{type(exc).__name__}: {exc}",
                            details=event_details,
                        )
                    raise
                event_duration = time_module.perf_counter() - event_start
                self._profiler.record_event_processing(event_type_name, event_duration)
                if event_logger is not None:
                    complete_details = dict(event_details or {})
                    complete_details.update(event.to_dict())
                    complete_details["target_cluster"] = (
                        log_cluster_type.name if log_cluster_type is not None else "unknown"
                    )
                    complete_details["new_events_generated"] = (
                        len(new_events) if new_events else 0
                    )
                    complete_details["cluster_time"] = self._time
                    complete_details["event_time"] = event.time
                    event_logger.log_event_complete(
                        event_type_name,
                        str(event.id),
                        event_duration * 1000.0,
                        details=complete_details,
                    )
                    self._emit_monolithic_layer_completion_markers(
                        event,
                        event_logger,
                        complete_details,
                    )
                    self._emit_monolithic_prefill_completion_markers(
                        event,
                        event_logger,
                        complete_details,
                    )

                self._add_events(new_events)

                if self._config.metrics_config.write_json_trace:
                    self._event_trace.append(event.to_dict())

                if self._config.metrics_config.enable_chrome_trace:
                    chrome_trace = event.to_chrome_trace()
                    if chrome_trace:
                        self._event_chrome_trace.append(chrome_trace)

                self._maybe_export_sequential_checkpoint()


            if self._terminate:
                break
            if self._try_promote_terminal_pdaf_scheduler_work():
                continue
            break

        event_processing_duration = time_module.perf_counter() - event_processing_start
        self._profiler.record_phase("event_processing_loop", event_processing_duration)

        if (
            self._is_sequential_checkpoint_observer_enabled()
            and not self._checkpoint_export_completed
        ):
            raise RuntimeError(
                "Sequential checkpoint observer was enabled but never exported a checkpoint."
            )

        global_scheduler_is_empty = self._global_scheduler.is_empty
        if not global_scheduler_is_empty and not self._terminate:
            report = self._build_sequential_scheduler_state_report()
            logger.error(report)
            if self._sequential_event_loggers:
                for logger_instance in self._sequential_event_loggers.values():
                    logger_instance.write_summary()
            if self._trace_store:
                self._trace_store.close()
            raise RuntimeError(report)

        if self._sequential_event_loggers:
            for logger_instance in self._sequential_event_loggers.values():
                logger_instance.write_summary()

        logger.info(f"Sequential simulation ended at: {self._time}s. Writing output...")

        # Close trace store if it exists
        if self._trace_store:
            self._trace_store.close()

        self._write_output()

    def _extract_sequential_event_details(
        self,
        event: BaseEvent,
        cluster_type: ClusterType,
    ) -> Dict[str, object]:
        """Extract enriched event details for sequential-mode cluster logs."""
        details: Dict[str, object] = {
            "event_time": event.time,
            "cluster": cluster_type.name,
            "event_type": str(event.event_type),
            "event_id": event.id,
            "target_cluster": cluster_type.name,
        }

        transfer_info = getattr(event, "_transfer_info", None)

        batch = None
        if hasattr(event, "_batch"):
            batch = getattr(event, "_batch", None)
        elif hasattr(event, "batch"):
            batch = getattr(event, "batch", None)
        elif transfer_info is not None:
            batch = getattr(transfer_info, "batch", None)

        if batch is not None:
            details["batch_id"] = getattr(batch, "id", getattr(batch, "_id", "unknown"))
            details["batch_global_id"] = getattr(batch, "global_id", "unknown")

            try:
                details["request_ids"] = getattr(
                    batch,
                    "request_ids",
                    [req.id for req in batch.requests],
                )
            except Exception:
                pass

            try:
                if getattr(batch, "requests", None):
                    request_decode_steps = [
                        getattr(req, "current_decode_token_index", "unknown")
                        for req in batch.requests
                    ]
                    request_layer_ids = [
                        getattr(req, "completed_layer_count", "unknown")
                        for req in batch.requests
                    ]
                    details["request_decode_steps"] = request_decode_steps
                    details["request_layer_ids"] = request_layer_ids
                    details["decode_step"] = request_decode_steps[0]
            except Exception:
                pass

            layer_id = None
            if hasattr(event, "_layer_id"):
                layer_id = getattr(event, "_layer_id", None)
            if layer_id is None and transfer_info is not None:
                layer_id = getattr(transfer_info, "layer_id", None)
            if layer_id is None:
                layer_id = getattr(batch, "af_inflight_layer_count", None)
            if layer_id is None:
                try:
                    if getattr(batch, "requests", None):
                        layer_id = getattr(batch.requests[0], "completed_layer_count", None)
                except Exception:
                    layer_id = None
            if layer_id is not None:
                details["layer_id"] = layer_id

            if hasattr(event, "_replica_id"):
                details["replica_id"] = getattr(event, "_replica_id")
            elif (
                hasattr(batch, "decode_attn_original_replica_id")
                and batch.decode_attn_original_replica_id is not None
            ):
                details["replica_id"] = batch.decode_attn_original_replica_id

            if hasattr(event, "_dp_id"):
                details["dp_id"] = getattr(event, "_dp_id")
            elif (
                hasattr(batch, "decode_attn_original_dp_id")
                and batch.decode_attn_original_dp_id is not None
            ):
                details["dp_id"] = batch.decode_attn_original_dp_id

            details["batch_size"] = getattr(batch, "size", "unknown")
            details["num_tokens"] = getattr(batch, "total_num_tokens", "unknown")

        batch_stage = getattr(event, "_batch_stage", None)
        if batch_stage is not None:
            details["batch_stage_id"] = getattr(
                batch_stage, "id", getattr(batch_stage, "_id", "unknown")
            )
            try:
                details["batch_stage_start_time"] = batch_stage.scheduled_at
            except Exception:
                pass
            try:
                details["batch_stage_execution_time"] = batch_stage.execution_time
            except Exception:
                pass
            details["batch_stage_end_time"] = event.time

            if cluster_type == ClusterType.MONOLITHIC:
                predictor = self._predictors.get(ClusterType.MONOLITHIC)
                num_layers = getattr(predictor, "_num_layers_per_pipeline_stage", None)
                if isinstance(num_layers, int) and num_layers > 0:
                    details["num_layers_per_stage"] = num_layers

        if transfer_info is not None:
            if hasattr(transfer_info, "is_attn_to_ffn"):
                details["is_attn_to_ffn"] = bool(
                    getattr(transfer_info, "is_attn_to_ffn")
                )
            if hasattr(transfer_info, "activation_size_bytes"):
                details["activation_size_bytes"] = int(
                    transfer_info.activation_size_bytes
                )
            if hasattr(transfer_info, "kv_cache_size_bytes"):
                details["kv_cache_size_bytes"] = int(transfer_info.kv_cache_size_bytes)

        if hasattr(event, "_kv_cache_size_bytes"):
            details["kv_cache_size_bytes"] = int(getattr(event, "_kv_cache_size_bytes"))
        if hasattr(event, "_activation_size_bytes"):
            details["activation_size_bytes"] = int(getattr(event, "_activation_size_bytes"))

        if hasattr(event, "replica_id"):
            details["replica_id"] = event.replica_id
        if hasattr(event, "pipeline_stage"):
            details["pipeline_stage"] = event.pipeline_stage

        return details

    def _emit_monolithic_layer_completion_markers(
        self,
        event: BaseEvent,
        event_logger: ClusterEventLogger,
        details: Dict[str, object],
    ) -> None:
        """Emit inferred per-layer completion markers for monolithic aggregated stages."""
        if details.get("cluster") != ClusterType.MONOLITHIC.name:
            return
        if event.__class__.__name__ != "BatchStageEndEvent":
            return

        num_layers = details.get("num_layers_per_stage")
        stage_start_time = details.get("batch_stage_start_time")
        if not isinstance(num_layers, int) or num_layers <= 0:
            return
        if not isinstance(stage_start_time, (int, float)):
            return

        stage_end_time = event.time
        stage_duration = stage_end_time - float(stage_start_time)
        if stage_duration <= 0:
            return

        layer_duration = stage_duration / num_layers
        request_ids = details.get("request_ids", [])

        for layer_id in range(num_layers):
            layer_completion_time = float(stage_start_time) + layer_duration * (layer_id + 1)
            marker_details = {
                "event_time": layer_completion_time,
                "cluster": ClusterType.MONOLITHIC.name,
                "batch_id": details.get("batch_id", "unknown"),
                "batch_global_id": details.get("batch_global_id", "unknown"),
                "request_ids": request_ids,
                "request_decode_steps": details.get("request_decode_steps", []),
                "request_layer_ids": details.get("request_layer_ids", []),
                "layer_id": layer_id,
                "decode_step": details.get("decode_step", "unknown"),
                "replica_id": details.get("replica_id", "unknown"),
                "dp_id": details.get("dp_id", "unknown"),
                "is_estimated_layer_marker": True,
                "source_event_id": details.get("event_id", event.id),
            }
            event_logger.log_event_complete(
                "MonolithicLayerCompletionEvent",
                f"{event.id}:layer:{layer_id}",
                0.0,
                details=marker_details,
            )

    def _emit_monolithic_prefill_completion_markers(
        self,
        event: BaseEvent,
        event_logger: ClusterEventLogger,
        details: Dict[str, object],
    ) -> None:
        """Emit exact prefill-completion markers for monolithic mode."""
        if details.get("cluster") != ClusterType.MONOLITHIC.name:
            return
        if event.__class__.__name__ != "GlobalBatchEndEvent":
            return

        batch = getattr(event, "_batch", None)
        if batch is None or not hasattr(batch, "requests"):
            return

        request_ids: List[int] = []
        request_decode_steps: List[int] = []
        request_layer_ids: List[int] = []

        for request in batch.requests:
            prefill_completed_at = getattr(request, "_prefill_completed_at", 0.0)
            if not isinstance(prefill_completed_at, (int, float)):
                continue
            if abs(float(prefill_completed_at) - float(event.time)) > 1e-12:
                continue

            request_ids.append(int(request.id))
            request_decode_steps.append(int(getattr(request, "current_decode_token_index", 0)))
            request_layer_ids.append(int(getattr(request, "completed_layer_count", 0)))

        if not request_ids:
            return

        marker_details = {
            "event_time": float(event.time),
            "cluster": ClusterType.MONOLITHIC.name,
            "batch_id": details.get("batch_id", "unknown"),
            "batch_global_id": details.get("batch_global_id", "unknown"),
            "request_ids": request_ids,
            "request_decode_steps": request_decode_steps,
            "request_layer_ids": request_layer_ids,
            "replica_id": details.get("replica_id", "unknown"),
            "dp_id": details.get("dp_id", "unknown"),
            "is_prefill_boundary_marker": True,
            "source_event_id": details.get("event_id", event.id),
        }
        event_logger.log_event_complete(
            "MonolithicPrefillCompletionEvent",
            f"{event.id}:prefill_boundary",
            0.0,
            details=marker_details,
        )

    def _collect_parallel_quiescence_state(self) -> dict:
        """Collect lightweight parallel runtime state for deadlock detection."""
        cluster_states = {}
        all_cluster_queues_empty = True
        any_cluster_processing = False

        for cluster_type, cluster_simulator in self._cluster_simulators.items():
            if hasattr(cluster_simulator, "get_runtime_state"):
                state = cluster_simulator.get_runtime_state()
            else:
                queue_size = (
                    cluster_simulator.get_queue_size()
                    if hasattr(cluster_simulator, "get_queue_size")
                    else (1 if cluster_simulator.has_events() else 0)
                )
                state = {
                    "cluster_type": cluster_type.name,
                    "queue_size": queue_size,
                    "local_time": cluster_simulator.get_local_time(),
                    "is_running": getattr(cluster_simulator, "_running", False),
                    "is_processing_event": (
                        cluster_simulator.is_processing_event()
                        if hasattr(cluster_simulator, "is_processing_event")
                        else False
                    ),
                    "current_event_name": None,
                }

            cluster_states[cluster_type] = state
            all_cluster_queues_empty = (
                all_cluster_queues_empty and int(state.get("queue_size", 0)) == 0
            )
            any_cluster_processing = any_cluster_processing or bool(
                state.get("is_processing_event", False)
            )

        inter_cluster_stats = {}
        if (
            getattr(self._global_scheduler, "_enable_parallel_mode", False)
            and hasattr(self._global_scheduler, "get_inter_cluster_communication_stats")
        ):
            inter_cluster_stats = (
                self._global_scheduler.get_inter_cluster_communication_stats()
            )

        inter_cluster_idle = (
            not inter_cluster_stats
            or (
                int(inter_cluster_stats.get("queue_size", 0)) == 0
                and int(inter_cluster_stats.get("total_buffered_events", 0)) == 0
            )
        )

        return {
            "cluster_states": cluster_states,
            "all_cluster_queues_empty": all_cluster_queues_empty,
            "any_cluster_processing": any_cluster_processing,
            "inter_cluster_stats": inter_cluster_stats,
            "inter_cluster_idle": inter_cluster_idle,
        }

    def _build_parallel_quiescent_deadlock_message(
        self,
        *,
        completed_requests: int,
        total_requests: int,
        quiescence_state: dict,
    ) -> str:
        """Build a fail-fast error for drained parallel simulations."""
        scheduler_states = {}
        for cluster_type in self._cluster_simulators.keys():
            scheduler_state = {"scheduler_empty": "unknown"}
            try:
                cluster_scheduler = self._global_scheduler.get_cluster_scheduler(
                    cluster_type
                )
                scheduler_state["scheduler_empty"] = cluster_scheduler.is_empty()
            except Exception as exc:
                scheduler_state["scheduler_empty_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            scheduler_states[cluster_type.name] = scheduler_state

        cluster_states = {
            cluster_type.name: dict(state)
            for cluster_type, state in quiescence_state["cluster_states"].items()
        }
        return (
            "Parallel simulation drained all cluster event queues and inter-cluster "
            "buffers while requests remain incomplete. "
            f"completed_requests={completed_requests}/{total_requests}, "
            f"cluster_states={cluster_states}, "
            f"scheduler_states={scheduler_states}, "
            f"inter_cluster_stats={quiescence_state['inter_cluster_stats']}"
        )

    def _monitor_parallel_simulation(self):
        """Monitor the progress of parallel simulation."""
        check_interval = (
            self._config.cluster_sync_interval_ms / 500
        )  # 1000.0  # Convert to seconds
        quiescent_iterations = 0

        while not self._terminate:
            # PRIORITY 1: Check for fatal errors in any cluster thread
            for cluster_type, cluster_simulator in self._cluster_simulators.items():
                if cluster_simulator._fatal_error is not None:
                    logger.error(f"FATAL ERROR detected in {cluster_type.name} cluster")
                    logger.error(
                        f"Error: {type(cluster_simulator._fatal_error).__name__}: {cluster_simulator._fatal_error}"
                    )
                    logger.error(
                        "Terminating simulation due to fatal error in cluster thread"
                    )
                    self._terminate = True
                    # Re-raise the exception to propagate to main thread
                    raise cluster_simulator._fatal_error

            # Check if any cluster thread has died unexpectedly
            for cluster_type, cluster_simulator in self._cluster_simulators.items():
                if (
                    cluster_simulator._thread is not None
                    and not cluster_simulator._thread.is_alive()
                ):
                    if cluster_simulator._running:
                        # Thread died but _running is still True - unexpected termination
                        logger.error(
                            f"Cluster thread {cluster_type.name} terminated unexpectedly"
                        )
                        self._terminate = True
                        raise RuntimeError(
                            f"Cluster thread {cluster_type.name} terminated unexpectedly"
                        )

            # Check each cluster's local time and activity
            any_cluster_active = False
            cluster_times = []

            for cluster_simulator in self._cluster_simulators.values():
                if cluster_simulator.has_events():
                    any_cluster_active = True
                cluster_times.append(cluster_simulator.get_local_time())

            # Update global time to the furthest progressed cluster.
            # Using min(cluster_times) can pin global time at 0 when one cluster is idle,
            # which breaks time-limit checks in parallel mode.
            if cluster_times:
                self._time = max(cluster_times)

            # Primary termination condition: all requests completed
            total_requests = self._metric_store.get_total_requests()
            completed_requests = self._metric_store.get_completed_requests()
            all_requests_completed = self._metric_store.all_requests_completed()
            # logger.info(f"[COMPLETION_CHECK] {completed_requests}/{total_requests} requests completed, any_cluster_active={any_cluster_active}")

            if all_requests_completed:
                logger.info(
                    f"All requests completed ({completed_requests}/{total_requests}); terminating simulation"
                )
                break

            # Secondary guard: time limit
            if self._time > self._time_limit:
                logger.info(
                    f"Time limit reached: {self._time_limit}s, terminating simulation"
                )
                self._terminate = True
                break

            quiescence_state = self._collect_parallel_quiescence_state()
            is_quiescent = (
                quiescence_state["all_cluster_queues_empty"]
                and not quiescence_state["any_cluster_processing"]
                and quiescence_state["inter_cluster_idle"]
            )
            if is_quiescent:
                quiescent_iterations += 1
                if quiescent_iterations >= 3:
                    message = self._build_parallel_quiescent_deadlock_message(
                        completed_requests=completed_requests,
                        total_requests=total_requests,
                        quiescence_state=quiescence_state,
                    )
                    logger.error(message)
                    self._terminate = True
                    raise RuntimeError(message)
            else:
                quiescent_iterations = 0

            # Sleep briefly before next check
            time.sleep(check_interval)

        # Wait for all clusters to finish processing
        self._wait_for_clusters_completion()

        # Close trace store if it exists (for parallel termination case)
        if self._trace_store:
            self._trace_store.close()

    def _wait_for_clusters_completion(self):
        """Wait for all cluster simulators to complete processing."""
        logger.info("Waiting for all clusters to complete processing...")

        # Give clusters a moment to process any final events
        time.sleep(0.1)

        # Log final statistics
        if hasattr(self, "_cluster_simulators"):
            for cluster_type, cluster_simulator in self._cluster_simulators.items():
                stats = cluster_simulator.get_statistics()
                logger.info(f"Cluster {cluster_type.name} final stats: {stats}")

        # Log inter-cluster communication stats from GlobalScheduler
        if self._global_scheduler._enable_parallel_mode:
            comm_stats = self._global_scheduler.get_inter_cluster_communication_stats()
            logger.info(f"Inter-cluster communication stats: {comm_stats}")

    def _write_output(self) -> None:
        logger.info("Starting metrics output...")
        logger.info(f"Output directory: {self._config.metrics_config.output_dir}")
        logger.info(f"write_metrics={self._config.metrics_config.write_metrics}")
        logger.info(f"store_plots={self._config.metrics_config.store_plots}")
        logger.info(
            f"enable_chrome_trace={self._config.metrics_config.enable_chrome_trace}"
        )

        # Only call plot() if metrics writing is enabled
        # Note: MetricsStore.plot() is decorated with @if_write_metrics which checks write_metrics
        # The store_plots flag is checked internally by individual plotting methods
        if self._config.metrics_config.write_metrics:
            try:
                with self._profiler.profile("metrics_output"):
                    if self._config.metrics_config.store_plots:
                        logger.info("Writing metrics (CSV + plots)...")
                    else:
                        logger.info("Writing metrics (CSV only, plots disabled)...")
                    self._metric_store.plot()
                    logger.info("Metrics output completed")
            except Exception as e:
                logger.error(f"Error during metrics output: {e}", exc_info=True)
                raise
        else:
            logger.info("Skipping metrics output (write_metrics=False)")

        if self._config.metrics_config.write_json_trace:
            try:
                with self._profiler.profile("write_json_trace"):
                    self._write_event_trace()
                logger.info("Json event trace written")
            except Exception as e:
                logger.error(f"Error writing JSON trace: {e}", exc_info=True)
                raise

        # Close TraceStore if it exists
        if self._trace_store:
            try:
                self._trace_store.close()
            except Exception as e:
                logger.error(f"Error closing TraceStore: {e}", exc_info=True)
                raise

        if self._config.metrics_config.enable_chrome_trace:
            try:
                with self._profiler.profile("write_chrome_trace"):
                    self._write_chrome_trace()
                logger.info("Chrome event trace written")
            except Exception as e:
                logger.error(f"Error writing Chrome trace: {e}", exc_info=True)
                raise

        # Close TraceStore if it exists
        if self._trace_store:
            try:
                self._trace_store.close()
                logger.info("Op-level trace store closed and flushed")
            except Exception as e:
                logger.error(f"Error closing trace store: {e}", exc_info=True)
                raise

        # Write performance profiling results
        if self._config.enable_performance_profiling:
            try:
                self._profiler.print_summary()
                output_path = os.path.join(
                    self._config.metrics_config.output_dir,
                    self._config.performance_profiling_output_file,
                )
                self._profiler.save_to_file(output_path)
                logger.info(f"Performance profiling results saved to {output_path}")
            except Exception as e:
                logger.error(
                    f"Error writing performance profiling results: {e}", exc_info=True
                )
                raise

        mismatches = self._quantization_manager.get_precision_mismatch_summary()
        if mismatches:
            logger.warning(
                "Precision mismatch summary (%d): %s",
                len(mismatches),
                [mismatch.get_warning_message() for mismatch in mismatches],
            )

        logger.info("Metrics output completed")

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, (event._priority_number, event))

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        with self._profiler.profile("request_generation"):
            requests = self._request_generator.generate()
        logger.info(f"Generated {len(requests)} requests")
        self._all_requests = list(requests)
        self._requests_by_session_id = {
            int(request.session_id): request
            for request in self._all_requests
            if request.session_id is not None
        }
        # Register total requests for completion-based termination.
        # Registration errors must propagate so request completion accounting
        # cannot silently drift from the generated workload.
        self._metric_store.register_total_requests(len(requests))

        # Initialize periodic scheduling events first
        with self._profiler.profile("initialize_periodic_scheduling"):
            periodic_events = self._global_scheduler.initialize_periodic_scheduling(
                start_time=0.0
            )
        for event in periodic_events:
            self._add_event(event)
            logger.info(
                f"Added periodic scheduling event for {event.get_target_cluster().name} cluster"
            )

        if (
            self._config.simulation_mode == "offline"
            and self._config.is_disaggregated_mode()
            and not getattr(
                self._config, "offline_use_generated_request_arrivals", False
            )
        ):
            # In offline mode, all requests are added to the scheduler at time 0.
            for request in requests:
                request.set_arrived_at(0)
                request.on_arrival(0, ClusterType.PREFILL)
                # Record request arrival metrics (histogram data: num_tokens, prefill_tokens, decode_tokens, etc.)
                # This must be called after on_arrival() so metrics collection does not
                # mutate request arrival bookkeeping.
                self._metric_store.on_request_arrival(0, request, ClusterType.PREFILL)
                self._global_scheduler.add_request(
                    request=request, cluster_type=ClusterType.PREFILL
                )

            # Kick off the scheduling process.
            self._add_event(GlobalScheduleEvent(0))
        elif (
            self._config.simulation_mode == "offline"
            and not self._config.is_disaggregated_mode()
            and not getattr(
                self._config, "offline_use_generated_request_arrivals", False
            )
        ):
            # Offline mode for monolithic (co-location) mode
            # All requests are added to the scheduler at time 0, similar to disaggregated mode
            for request in requests:
                request.set_arrived_at(0)
                request.on_arrival(0, ClusterType.MONOLITHIC)
                # Record request arrival metrics
                self._metric_store.on_request_arrival(
                    0, request, ClusterType.MONOLITHIC
                )
                self._global_scheduler.add_request(
                    request=request, cluster_type=ClusterType.MONOLITHIC
                )

            # Kick off the scheduling process.
            self._add_event(GlobalScheduleEvent(0))
        else:
            for request in requests:
                # In disaggregated mode, a request first arrives at the prefill cluster
                # In monolithic mode, it arrives at the monolithic cluster
                cluster_type = (
                    ClusterType.PREFILL
                    if self._config.is_disaggregated_mode()
                    else ClusterType.MONOLITHIC
                )
                self._add_event(
                    RequestArrivalEvent(request.arrived_at, request, cluster_type)
                )

    def _set_time(self, time: float) -> None:
        self._time = time

    def _is_sequential_checkpoint_observer_enabled(self) -> bool:
        return bool(
            getattr(self._config, "enable_sequential_checkpoint_observer", False)
        )

    def _load_checkpoint_expected_session_ids(self) -> Optional[set[int]]:
        if not self._is_sequential_checkpoint_observer_enabled():
            return None
        expected_session_ids_file = getattr(
            self._config, "sequential_checkpoint_expected_session_ids_file", None
        )
        if not expected_session_ids_file:
            return None

        with open(expected_session_ids_file, "r", encoding="utf-8") as handle:
            payload = handle.read().strip()
        if not payload:
            return set()
        if payload.startswith("["):
            return {int(value) for value in json.loads(payload)}
        return {int(line.strip()) for line in payload.splitlines() if line.strip()}

    @staticmethod
    def _get_next_event_time_from_queue_entry(queue_entry: tuple) -> float:
        if len(queue_entry) < 2:
            raise ValueError("Sequential event queue entries must contain an event object.")
        event = queue_entry[1]
        event_time = getattr(event, "time", None)
        if event_time is not None:
            return float(event_time)

        priority = queue_entry[0]
        if isinstance(priority, tuple) and priority:
            return float(priority[0])
        return float(priority)

    def _maybe_export_sequential_checkpoint(self) -> bool:
        if (
            not self._is_sequential_checkpoint_observer_enabled()
            or self._checkpoint_export_completed
        ):
            return False

        if (
            self._event_queue
            and self._get_next_event_time_from_queue_entry(self._event_queue[0])
            <= self._time
        ):
            return False

        expected_survivor_count = int(
            getattr(self._config, "sequential_checkpoint_expected_survivor_count", 0)
        )
        incomplete_requests = [
            request
            for request in self._all_requests
            if not getattr(request, "completed", False)
        ]
        if len(incomplete_requests) != expected_survivor_count:
            return False

        survivor_session_ids = set()
        for request in incomplete_requests:
            session_id = getattr(request, "session_id", None)
            if session_id is None:
                raise ValueError(
                    "Sequential checkpoint export requires session_id on every survivor."
                )
            survivor_session_ids.add(int(session_id))
            if getattr(request, "pending_thinking_requeue", False):
                raise ValueError(
                    "Sequential checkpoint export requires survivors with no pending requeue state."
                )

        if (
            self._checkpoint_expected_session_ids is not None
            and survivor_session_ids != self._checkpoint_expected_session_ids
        ):
            raise ValueError(
                "Sequential checkpoint survivor set mismatch: "
                f"expected={sorted(self._checkpoint_expected_session_ids)}, "
                f"actual={sorted(survivor_session_ids)}"
            )

        raw_snapshot_path = getattr(
            self._config, "sequential_checkpoint_raw_snapshot_path", None
        )
        if not raw_snapshot_path:
            raise ValueError(
                "Sequential checkpoint export requires sequential_checkpoint_raw_snapshot_path."
            )
        os.makedirs(os.path.dirname(raw_snapshot_path), exist_ok=True)
        sorted_requests = sorted(
            incomplete_requests, key=lambda request: int(getattr(request, "session_id"))
        )
        with open(raw_snapshot_path, "w", encoding="utf-8") as handle:
            for request in sorted_requests:
                handle.write(
                    json.dumps(
                        {
                            "checkpoint_time": float(self._time),
                            "request": request.to_dict(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        logger.info(
            "Exported sequential checkpoint snapshot at t=%s with %s survivors.",
            self._time,
            len(sorted_requests),
        )
        self._checkpoint_export_completed = True
        self._terminate = True
        return True
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._terminate = True

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f)
