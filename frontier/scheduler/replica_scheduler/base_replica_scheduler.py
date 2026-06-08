from abc import ABC, abstractmethod
from typing import Any, Dict, List

from frontier.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
from frontier.errors import FrontierMemoryOOMError
from frontier.entities import Batch, Replica, Request
from frontier.execution_time_predictor import BaseExecutionTimePredictor
from frontier.logger import get_cluster_logger
from frontier.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.spec_decode import (
    get_planned_draft_tokens,
    is_spec_decode_enabled,
    method_uses_lookahead_slots,
)
from frontier.types import ClusterType, ReplicaSchedulerType



class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        predictor: BaseExecutionTimePredictor,
        cluster_type: ClusterType = None,
        dp_id: int = None,
        af_pipeline_num_micro_batch: int = -1,
        cluster_scheduler = None
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._replica = replica
        self._replica_is_moe = replica.is_moe
        self._num_stages = replica.num_pipeline_stages
        self._predictor = predictor
        self._cluster_type = cluster_type
        self._dp_id = dp_id
        self._af_pipeline_num_micro_batch = af_pipeline_num_micro_batch
        self._cluster_scheduler = cluster_scheduler

        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        memory_planner = MemoryPlanner(
            replica_config=self._replica_config,
            replica=replica,
            cluster_type=self._cluster_type,
        )

        num_blocks_mode = getattr(self._config, "num_blocks_mode", "memory_planner")
        if num_blocks_mode not in {
            "memory_planner",
            "memory_planner_profiled",
            "explicit",
        }:
            raise ValueError(
                "num_blocks_mode must be 'memory_planner', "
                "'memory_planner_profiled', or 'explicit', "
                f"got={num_blocks_mode!r}"
            )

        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        planner_gpu_utilization = getattr(
            self._config, "gpu_memory_utilization", None
        )
        planner_overhead_bytes = int(
            getattr(self._config, "non_kv_cache_overhead_bytes", 0)
        )
        runtime_weights_memory_source = str(
            getattr(self._config, "runtime_weights_memory_source", "param_counter")
        )
        use_analytical_param_memory = bool(
            getattr(
                self._config,
                "use_analytical_param_memory",
                False,
            )
        )

        runtime_overhead_enabled = bool(
            getattr(
                self._config,
                "enable_runtime_non_kv_cache_overhead_profiling",
                False,
            )
        )
        if runtime_overhead_enabled:
            if num_blocks_mode != "memory_planner_profiled":
                raise ValueError(
                    "enable_runtime_non_kv_cache_overhead_profiling requires "
                    "num_blocks_mode=memory_planner_profiled, "
                    f"got={num_blocks_mode!r}"
                )

            from frontier.profiling.non_kv_cache_overhead.runtime_estimator import (
                estimate_non_kv_cache_profile,
            )
            from frontier.profiling.non_kv_cache_overhead.nccl_buffer_estimator import (
                NCCLBufferEstimationConfig,
            )

            cluster_type_for_profile = self._cluster_type or ClusterType.MONOLITHIC
            profile_max_num_batched_tokens = int(
                getattr(
                    self._config,
                    "max_tokens_in_batch",
                    self._request_generator_config.max_tokens,
                )
            )
            input_weights_memory_bytes = int(
                memory_planner.get_parameter_memory_per_device_bytes()
            )

            nccl_buffer_config = NCCLBufferEstimationConfig(
                nccl_comm_base_overhead_bytes=int(
                    getattr(
                        self._config,
                        "nccl_buffer_comm_base_overhead_bytes",
                        100 * 1024 * 1024,
                    )
                ),
                nccl_per_peer_overhead_bytes=int(
                    getattr(
                        self._config,
                        "nccl_buffer_per_peer_overhead_bytes",
                        15 * 1024 * 1024,
                    )
                ),
                custom_ar_enabled=bool(
                    getattr(
                        self._config,
                        "nccl_buffer_custom_ar_enabled",
                        False,
                    )
                ),
                vllm_worker_base_extra_bytes=int(
                    getattr(
                        self._config,
                        "nccl_buffer_vllm_worker_base_extra_bytes",
                        0,
                    )
                ),
                pp_final_stage_extra_bytes=int(
                    getattr(
                        self._config,
                        "nccl_buffer_pp_final_stage_extra_bytes",
                        0,
                    )
                ),
                dp_communicator_extra_bytes=int(
                    getattr(
                        self._config,
                        "nccl_buffer_dp_communicator_extra_bytes",
                        0,
                    )
                ),
                ep_all2all_extra_bytes=int(
                    getattr(
                        self._config,
                        "nccl_buffer_ep_all2all_extra_bytes",
                        0,
                    )
                ),
            )

            profile_result = estimate_non_kv_cache_profile(
                replica_config=self._replica_config,
                cluster_type=cluster_type_for_profile,
                max_num_batched_tokens=profile_max_num_batched_tokens,
                weights_memory_bytes=input_weights_memory_bytes,
                weights_memory_source=runtime_weights_memory_source,
                nccl_buffer_config=nccl_buffer_config,
            )

            planner_overhead_bytes = int(profile_result.overhead_bytes)
            measured_weights_memory_bytes = int(profile_result.measured_weights_memory_bytes)
            if not use_analytical_param_memory:
                adjusted_overhead = (
                    int(profile_result.overhead_bytes)
                    + measured_weights_memory_bytes
                    - int(profile_result.input_weights_memory_bytes)
                )
                if adjusted_overhead < 0:
                    raise RuntimeError(
                        "Adjusted planner overhead became negative after applying "
                        "runtime-measured weights, "
                        f"adjusted_overhead={adjusted_overhead}, "
                        f"profile_result={profile_result}"
                    )
                planner_overhead_bytes = int(adjusted_overhead)

            self._config.non_kv_cache_overhead_bytes = int(planner_overhead_bytes)

            logger.info(
                "[RUNTIME_NON_KV_PROFILE_APPLIED] cluster_type=%s, dp_id=%s, "
                "profile_max_num_batched_tokens=%s, weights_memory_source=%s, "
                "input_weights_memory_bytes=%s, measured_weights_memory_bytes=%s, "
                "use_analytical_param_memory=%s, non_kv_cache_overhead_bytes=%s",
                cluster_type_for_profile.name,
                self._dp_id,
                profile_max_num_batched_tokens,
                runtime_weights_memory_source,
                input_weights_memory_bytes,
                measured_weights_memory_bytes,
                use_analytical_param_memory,
                planner_overhead_bytes,
            )

        planner_overhead_for_memory_estimation = (
            planner_overhead_bytes
            if num_blocks_mode == "memory_planner_profiled"
            else 0
        )

        if num_blocks_mode == "explicit":
            if int(self._config.num_blocks) <= 0:
                raise ValueError(
                    "num_blocks must be > 0 when num_blocks_mode='explicit'"
                )
        elif not self._config.num_blocks:
            if hasattr(self._config, "gpu_memory_utilization"):
                self._config.num_blocks = memory_planner.get_num_blocks(
                    block_size=self._config.block_size,
                    gpu_memory_utilization=planner_gpu_utilization,
                    non_kv_cache_overhead_bytes=planner_overhead_for_memory_estimation,
                )
            else:
                self._config.num_blocks = (
                    self._max_blocks_per_sequence
                    * memory_planner.get_max_request_slots(
                        planner_gpu_utilization,
                        planner_overhead_for_memory_estimation,
                    )
                )

        scheduler_type = (
            self._config.get_type() if hasattr(self._config, "get_type") else None
        )
        if scheduler_type in {
            ReplicaSchedulerType.VLLM_V1,
            ReplicaSchedulerType.SGLANG,
            ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE,
            ReplicaSchedulerType.SJ2Q_PENALTY_ONLY,
            ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER,
        }:
            watermark_blocks = int(
                self._config.watermark_blocks_fraction * self._config.num_blocks
            )
            minimum_chunk_blocks = 1
            minimum_required_total_blocks = watermark_blocks + minimum_chunk_blocks
            if int(self._config.num_blocks) < minimum_required_total_blocks:
                raise FrontierMemoryOOMError(
                    "Derived KV cache budget cannot admit even a single block-sized chunk under the current watermark policy.",
                    reason="insufficient_initial_block_budget",
                    details={
                        "cluster_type": (
                            self._cluster_type.name
                            if self._cluster_type is not None
                            else "MONOLITHIC"
                        ),
                        "total_blocks": int(self._config.num_blocks),
                        "watermark_blocks": watermark_blocks,
                        "minimum_chunk_blocks": minimum_chunk_blocks,
                        "minimum_required_total_blocks": minimum_required_total_blocks,
                        "block_size": int(self._config.block_size),
                    },
                )

        if scheduler_type in {
            ReplicaSchedulerType.VLLM_V1,
            ReplicaSchedulerType.SGLANG,
            ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE,
            ReplicaSchedulerType.SJ2Q_PENALTY_ONLY,
            ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER,
        }:
            # `MemoryPlanner.get_max_batch_size()` is a static worst-case estimator
            # based on max_request_tokens, and is too conservative for vLLM-style
            # schedulers. Runtime admission is governed by token budget and
            # block-level checks instead.
            self._max_batch_size = int(self._config.batch_size_cap)
            max_request_slots = None
        else:
            self._max_batch_size = min(
                memory_planner.get_max_batch_size(
                    planner_gpu_utilization,
                    planner_overhead_for_memory_estimation,
                ),
                self._config.batch_size_cap,
            )
            max_request_slots = memory_planner.get_max_request_slots(
                planner_gpu_utilization,
                planner_overhead_for_memory_estimation,
            )

        # Activation memory tracking for DECODE_FFN (no KV cache).
        self._activation_bytes_allocated = 0
        self._activation_memory_capacity_bytes = None
        if self._cluster_type == ClusterType.DECODE_FFN:
            total_memory_bytes = int(self._replica.total_memory_gb * 1024**3)
            available_memory_bytes = int(
                total_memory_bytes * (1 - self._replica.memory_margin_fraction)
            )
            parameter_memory_bytes = int(
                memory_planner.get_parameter_memory_per_device_bytes()
            )
            self._activation_memory_capacity_bytes = max(
                0, available_memory_bytes - parameter_memory_bytes
            )
            logger = get_cluster_logger(
                __name__, self._cluster_type.name if self._cluster_type else None
            )
            logger.info(
                f"[ACTIVATION_MEMORY_CAPACITY] total_memory_bytes={total_memory_bytes}, "
                f"available_memory_bytes={available_memory_bytes}, "
                f"parameter_memory_bytes={parameter_memory_bytes}, "
                f"activation_capacity_bytes={self._activation_memory_capacity_bytes}"
            )

        logger.info(
            f"[MEMORY_STATE] total_blocks={int(self._config.num_blocks)}, "
            f"max_blocks_per_sequence={int(self._max_blocks_per_sequence)}, "
            f"max_request_slots={int(max_request_slots) if max_request_slots is not None else 'n/a'}, "
            f"max_batch_size={int(self._max_batch_size)}"
        )
        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}
        self._batch_creation_counter = 0
        self._decode_sync_batch_creation_counter = 0
        self._num_running_batches = 0

        # Current schedule time for preemption tracking
        # This is set by on_schedule() and used by _preempt_request()
        self._current_schedule_time: float = 0.0

        # for decode-ffn cluster, we need to add the batch to a processing queue
        if self._cluster_type == ClusterType.DECODE_FFN:
            self._m2n_immediate_batch_queue = []

        # for decode-attn cluster, we need a queue for batches returning from decode-ffn
        if self._cluster_type == ClusterType.DECODE_ATTN:
            self._af_immediate_batch_queue = []

        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == self._num_stages - 1,
                replica.is_moe,
                self._predictor,
                self._cluster_type,
                self._dp_id,
            )
            for stage_id in range(self._num_stages)
        }

    def _allocate_decode_sync_global_id(self) -> int:
        lane_decode_sync_counter = int(
            getattr(self, "_decode_sync_batch_creation_counter", 0) or 0
        )
        self._decode_sync_batch_creation_counter = lane_decode_sync_counter + 1

        cluster_scheduler = getattr(self, "_cluster_scheduler", None)
        if (
            cluster_scheduler is not None
            and hasattr(cluster_scheduler, "make_decode_sync_global_id")
        ):
            return int(
                cluster_scheduler.make_decode_sync_global_id(
                    self._replica_id,
                    self._dp_id,
                    lane_decode_sync_counter,
                )
            )

        return lane_decode_sync_counter

    def _should_assign_decode_sync_global_id(self, batch: Batch) -> bool:
        return (
            self._cluster_type == ClusterType.MONOLITHIC
            and bool(getattr(self, "_replica_is_moe", False))
            and batch.num_prefill_tokens == 0
            and batch.num_decode_tokens > 0
        )

    def _create_batch(self, requests: List[Request], num_tokens: List[int]) -> Batch:
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name if self._cluster_type else None)

        batch = Batch(
            self._replica_id,
            requests,
            num_tokens,
            is_moe=self._replica_is_moe,
        )
        batch.set_global_id(self._batch_creation_counter)
        self._batch_creation_counter += 1
        if self._should_assign_decode_sync_global_id(batch):
            batch.decode_sync_global_id = self._allocate_decode_sync_global_id()

        # DEBUG: Log batch creation details
        request_ids = [req.id for req in requests]
        decode_sync_global_id = getattr(batch, "decode_sync_global_id", None)
        debug_msg = (
            f"[BATCH_CREATE] batch_id={batch.id}, global_id={batch.global_id}, "
            f"decode_sync_global_id={decode_sync_global_id}, "
            f"replica_id={self._replica_id}, dp_id={self._dp_id}, "
            f"requests={request_ids}, num_requests={len(requests)}, "
            f"cluster={self._cluster_type.name if self._cluster_type else 'None'}"
        )
        logger.debug(debug_msg)

        if self._cluster_type == ClusterType.DECODE_ATTN:
            batch.decode_attn_original_replica_id = self._replica_id
            batch.decode_attn_original_dp_id = self._dp_id

        return batch

    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    def peek_waiting_requests(self) -> List[Request]:
        return list(self._request_queue)

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:
        if self._cluster_type == ClusterType.DECODE_FFN:
            if not self._activation_memory_capacity_bytes:
                return 0
            return (
                self._activation_bytes_allocated * 100
            ) / self._activation_memory_capacity_bytes
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    @property
    def num_running_batches(self) -> int:
        return self._num_running_batches

    def decrement_num_running_batches(self) -> None:
        self._num_running_batches -= 1

    @staticmethod
    def _debug_request_id(request: Request) -> int:
        if not hasattr(request, "id"):
            raise TypeError(f"Expected Request-like object with id, got {type(request)}")
        return int(request.id)

    @classmethod
    def _debug_request_collection_state(cls, requests: Any) -> Dict[str, Any]:
        if requests is None:
            return {"status": "not_applicable"}
        if isinstance(requests, dict):
            request_values = list(requests.values())
        else:
            request_values = list(requests)
        return {
            "count": len(request_values),
            "request_ids": [
                cls._debug_request_id(request) for request in request_values
            ],
            "requests": [
                {
                    "id": cls._debug_request_id(request),
                    "arrived_at": getattr(request, "arrived_at", None),
                    "num_prefill_tokens": getattr(
                        request, "num_prefill_tokens", None
                    ),
                    "num_decode_tokens": getattr(request, "num_decode_tokens", None),
                    "num_processed_tokens": getattr(
                        request, "num_processed_tokens", None
                    ),
                    "current_decode_token_index": getattr(
                        request, "current_decode_token_index", None
                    ),
                    "completed_layer_count": getattr(
                        request, "completed_layer_count", None
                    ),
                    "af_roundtrip_inflight": getattr(
                        request, "af_roundtrip_inflight", None
                    ),
                    "completed": getattr(request, "completed", None),
                }
                for request in request_values
            ],
        }

    @staticmethod
    def _debug_batch_id(batch: Batch) -> int:
        if not hasattr(batch, "id"):
            raise TypeError(f"Expected Batch-like object with id, got {type(batch)}")
        return int(batch.id)

    @classmethod
    def _debug_batch_collection_state(cls, batches: Any) -> Dict[str, Any]:
        if batches is None:
            return {"status": "not_applicable"}
        batch_values = list(batches)
        return {
            "count": len(batch_values),
            "batch_ids": [cls._debug_batch_id(batch) for batch in batch_values],
            "batch_global_ids": [
                getattr(batch, "global_id", None) for batch in batch_values
            ],
            "request_ids": [
                list(getattr(batch, "request_ids", [])) for batch in batch_values
            ],
            "batches": [
                {
                    "id": cls._debug_batch_id(batch),
                    "global_id": getattr(batch, "global_id", None),
                    "replica_id": getattr(batch, "replica_id", None),
                    "afd_stage_idx": getattr(batch, "afd_stage_idx", None),
                    "total_num_tokens": getattr(batch, "total_num_tokens", None),
                    "request_ids": list(getattr(batch, "request_ids", [])),
                    "is_idle": getattr(batch, "is_idle", None),
                }
                for batch in batch_values
            ],
        }

    @staticmethod
    def _debug_allocation_map_state(allocation_map: Dict[Any, Any]) -> Dict[str, Any]:
        if allocation_map is None:
            raise RuntimeError("_allocation_map is required for replica diagnostics")
        sorted_request_ids = sorted(int(request_id) for request_id in allocation_map)
        return {
            "count": len(allocation_map),
            "request_ids": sorted_request_ids,
            "allocated_blocks_by_request_id": {
                str(int(request_id)): allocation_map[request_id]
                for request_id in sorted(allocation_map.keys())
            },
        }

    def get_debug_state(self) -> Dict[str, Any]:
        """Return fail-fast diagnostic state for this replica scheduler."""
        required_attrs = [
            "_request_queue",
            "_allocation_map",
            "_num_running_batches",
            "_replica_stage_schedulers",
        ]
        for attr_name in required_attrs:
            if not hasattr(self, attr_name):
                raise RuntimeError(
                    f"Replica scheduler missing required debug field {attr_name}"
                )

        stage_states = {}
        for stage_id, stage_scheduler in sorted(
            self._replica_stage_schedulers.items(), key=lambda item: item[0]
        ):
            if not hasattr(stage_scheduler, "get_debug_state"):
                raise RuntimeError(
                    f"Stage scheduler {stage_id} missing get_debug_state()"
                )
            stage_states[str(stage_id)] = stage_scheduler.get_debug_state()

        return {
            "scheduler_class": self.__class__.__name__,
            "cluster_type": (
                self._cluster_type.name if self._cluster_type is not None else None
            ),
            "replica_id": self._replica_id,
            "dp_id": self._dp_id,
            "request_queue": self._debug_request_collection_state(
                self._request_queue
            ),
            "waiting_requests": self._debug_request_collection_state(
                getattr(self, "_waiting_requests", None)
            ),
            "running_requests": self._debug_request_collection_state(
                getattr(self, "_running_requests", None)
            ),
            "allocation_map": self._debug_allocation_map_state(
                self._allocation_map
            ),
            "num_running_batches": int(self._num_running_batches),
            "af_immediate_batch_queue": self._debug_batch_collection_state(
                getattr(self, "_af_immediate_batch_queue", None)
            ),
            "m2n_immediate_batch_queue": self._debug_batch_collection_state(
                getattr(self, "_m2n_immediate_batch_queue", None)
            ),
            "activation_bytes_allocated": getattr(
                self, "_activation_bytes_allocated", None
            ),
            "activation_memory_capacity_bytes": getattr(
                self, "_activation_memory_capacity_bytes", None
            ),
            "stage_schedulers": stage_states,
        }

    def is_empty(self) -> bool:
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        stages_empty = all(
            stage_scheduler.is_empty()
            for stage_scheduler in self._replica_stage_schedulers.values()
        )
        af_len = len(self._af_immediate_batch_queue) if hasattr(self, '_af_immediate_batch_queue') else 0
        logger.info(
            f"[RS-IDLE-CHECK][replica={self._replica_id}][dp={self._dp_id}] "
            f"num_pending_requests={self.num_pending_requests}, allocated_blocks={len(self._allocation_map)}, "
            f"num_running_batches={self._num_running_batches}, stages_empty={stages_empty}, af_immediate_len={af_len}"
        )
        # If AF immediate queue has pending batches, the replica is not idle
        if af_len > 0:
            return False
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and self._num_running_batches == 0
            and stages_empty
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        """
        Calculate the number of tokens to process for a request in the next batch.

        For prefill phase: returns the REMAINING prefill tokens (not total).
        For decode phase: returns 1 (one token per decode step).

        This method correctly handles partial prefill completion when token budget
        limits how many tokens can be processed in a single batch.

        Args:
            request: The request to calculate tokens for

        Returns:
            Number of tokens to process in the next batch
        """
        assert not request.completed

        if request.is_prefill_complete:
            if request.spec_decode_enabled:
                remaining_decode_tokens = request.remaining_decode_tokens
                spec_config = getattr(
                    getattr(self, "_replica_config", None),
                    "speculative_decoding_config",
                    None,
                )
                has_scheduled_draft_trace = (
                    getattr(spec_config, "_scheduled_draft_tokens_trace", None)
                    is not None
                    or getattr(
                        spec_config,
                        "_per_request_scheduled_draft_tokens_trace",
                        None,
                    )
                    is not None
                )
                if has_scheduled_draft_trace:
                    planned_draft_tokens = request.spec_next_planned_draft_tokens
                else:
                    max_planned = max(remaining_decode_tokens - 1, 0)
                    planned_draft_tokens = min(
                        request.spec_next_planned_draft_tokens, max_planned
                    )
                if planned_draft_tokens < 0:
                    raise ValueError(
                        f"planned_draft_tokens must be >= 0, got={planned_draft_tokens}"
                    )
                return 1 + planned_draft_tokens
            return 1

        # ISSUE-011 FIX: Return remaining prefill tokens, not total prefill tokens.
        # When token budget limits prefill to partial completion, the request may
        # have already processed some prefill tokens. We need to return only the
        # remaining tokens to avoid token count overflow.
        remaining_prefill_tokens = request.num_prefill_tokens - request.num_processed_tokens
        return remaining_prefill_tokens

    def _initialize_request_spec_decode_state(self, request: Request) -> None:
        """Initialize per-request speculative decode runtime state.

        This helper keeps state initialization consistent across scheduler
        subclasses that customize queue routing in add_request().
        """
        spec_config = getattr(
            self._replica_config, "speculative_decoding_config", None
        )
        if is_spec_decode_enabled(spec_config):
            request.initialize_spec_decode_state(
                enabled=True,
                method=spec_config.method,
                num_speculative_tokens=spec_config.num_speculative_tokens,
                method_uses_lookahead_slots=method_uses_lookahead_slots(
                    spec_config.method
                ),
            )
            request.set_spec_next_planned_draft_tokens(
                get_planned_draft_tokens(
                    spec_config,
                    request.remaining_decode_tokens,
                    iteration_index=request.spec_total_iterations,
                    request_id=str(request.id),
                )
            )
            return
        request.initialize_spec_decode_state(enabled=False)

    def add_request(self, request: Request) -> None:
        self._initialize_request_spec_decode_state(request)
        self._request_queue.append(request)

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks

    def allocate_batch(self, batch: Batch) -> None:
        for request in batch.requests:
            self.allocate(request.id, self._max_blocks_per_sequence)

    def free(self, *request_ids: List[int]) -> None:
        for request_id in request_ids:
            # Check if request_id exists in allocation_map before freeing
            # This handles cases where a request was never allocated (e.g., skipped in scheduling)
            if request_id not in self._allocation_map:
                logger = get_cluster_logger(__name__, self._cluster_type.name if self._cluster_type else None)
                logger.debug(
                    f"[BaseReplicaScheduler] Attempted to free request {request_id} "
                    f"but it was not in allocation_map (may have been skipped or already freed)"
                )
                continue
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0

    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    def on_cluster_stage_end(self, _batch: Batch) -> None:
        """Hook for cluster-internal stage completion.

        Default implementation is a no-op. Concrete replica schedulers may override
        for per-cluster accounting or diagnostics (but MUST NOT modify the global
        running-batch semantics defined by Solution A).
        """
        del _batch  # explicitly mark unused in default implementation
        return

    @abstractmethod
    def _get_next_batch(self, is_micro_batch: bool) -> Batch:
        pass

    def on_schedule(self, time: float = 0.0) -> List[Batch]:
        # Store current schedule time for preemption tracking
        self._current_schedule_time = time
        
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        scheduled_batches = []

        # DEBUG: queue state at entry
        try:
            af_len = len(self._af_immediate_batch_queue) if hasattr(self, '_af_immediate_batch_queue') else 0
            af_ids = [b.id for b in getattr(self, '_af_immediate_batch_queue', [])]
        except Exception:
            af_len, af_ids = 0, []
        logger.info(f"[ON_SCHEDULE][{self._cluster_type.name}][replica={self._replica_id}][dp={self._dp_id}] af_immediate_len={af_len}, af_batch_ids={af_ids}, num_running={self._num_running_batches}, af_pipeline_cap={self._af_pipeline_num_micro_batch}")

        # Different batching logic based on cluster type
        if self._cluster_type == ClusterType.DECODE:
            # For unified DECODE cluster (PD-disaggregation mode):
            # CRITICAL FIX: Each DP replica scheduler operates independently
            # The batch capacity should be based on pipeline stages, NOT DP size
            # Each DP replica (dp_id) has its own replica scheduler instance
            # and can run batches independently up to num_pipeline_stages
            #
            # This is different from DECODE_ATTN which uses micro-batch pipeline
            # and different from the DP synchronization which happens at MoE layers
            scheduled_request_ids = set()
            self._continuation_request_ids = scheduled_request_ids
            while self._num_running_batches < self._num_stages:
                batch = self._get_next_batch(is_micro_batch=False)
                if not batch:
                    break
                for req in batch.requests:
                    scheduled_request_ids.add(req.id)
                self._continuation_request_ids = scheduled_request_ids
                scheduled_batches.append(batch)
                self._num_running_batches += 1

                # DEBUG: Enhanced logging for batch scheduling
                request_ids = [req.id for req in batch.requests] if batch.requests else []
                debug_msg = (
                    f"[BATCH_SCHEDULE][DECODE] batch_id={batch.id}, global_id={batch.global_id}, "
                    f"replica_id={self._replica_id}, dp_id={self._dp_id}, "
                    f"requests={request_ids}, num_running={self._num_running_batches}/{self._num_stages}"
                )
                logger.debug(debug_msg)
            self._continuation_request_ids = set()
        elif self._cluster_type == ClusterType.DECODE_ATTN:
            # Track ALL scheduled request IDs in this on_schedule() cycle
            # This prevents the same request from being scheduled into multiple batches
            # within the same scheduling cycle (both Priority 1 and Priority 2)
            scheduled_request_ids = set()
            
            # Priority 1: drain AF-immediate (F→A returned) inflight micro-batches
            if hasattr(self, '_af_immediate_batch_queue') and self._af_immediate_batch_queue:
                logger.info(
                    f"[DECODE_ATTN][Replica {self._replica_id}][DP {self._dp_id}] Draining {len(self._af_immediate_batch_queue)} AF-immediate inflight micro-batches"
                )
                while self._af_immediate_batch_queue:
                    micro_batch = self._af_immediate_batch_queue.pop(0)
                    # Collect request IDs from this inflight batch
                    for req in micro_batch.requests:
                        scheduled_request_ids.add(req.id)
                    # Inflight micro-batches already occupy a pipeline slot; do NOT increment _num_running_batches
                    scheduled_batches.append(micro_batch)
                    logger.info(
                        f"[DECODE_ATTN][Replica {self._replica_id}][DP {self._dp_id}] Scheduled inflight micro-batch {getattr(micro_batch,'id','?')} (no slot increment)"
                    )
            
            # Store scheduled request IDs for use by _schedule_decode_attn_only()
            # This is updated after each Priority 2 iteration to prevent duplicate scheduling
            self._continuation_request_ids = scheduled_request_ids
            if scheduled_request_ids:
                logger.debug(
                    f"[DECODE_ATTN][Replica {self._replica_id}][DP {self._dp_id}] "
                    f"Excluding {len(scheduled_request_ids)} requests from Priority 2: {scheduled_request_ids}"
                )

            # Priority 2: top up pipeline with NEW micro-batches up to capacity
            # note: in pd-af, if we have 3 stage batches, we should quickly trigger
            # them into pipeline (e.g.,finish mb_0's 1st layer then trigger mb_1's 1st layer)
            # and, in that case, we only have 3 micro_batches in total (batching based on vllm v1's scheduler output)

            # Note: the key point here is we use while to make sure we have enough micro-batches in pipeline
            # we add all mb needed to stage queue berfore the trigger after one mb is finished
            while self._num_running_batches < self._af_pipeline_num_micro_batch:
                # todo: make sure mb is created based on vllm v1's scheduler output and af_inflight_layer_count
                # if we can schedule 24 reqs (get from scheduler output), each mb will be included by 24/8=3 (af_pipeline_num_micro_batch)
                micro_batch = self._get_next_batch(is_micro_batch=True)
                if not micro_batch:
                    break
                
                # CRITICAL FIX: Track newly scheduled requests to prevent duplicate scheduling
                # in subsequent iterations of this Priority 2 loop
                for req in micro_batch.requests:
                    scheduled_request_ids.add(req.id)
                # Update the set for next iteration
                self._continuation_request_ids = scheduled_request_ids
                
                scheduled_batches.append(micro_batch)
                self._num_running_batches += 1
                logger.info(
                    f"[DECODE_ATTN][Replica {self._replica_id}][DP {self._dp_id}] Scheduled NEW micro-batch {getattr(micro_batch,'id','?')} (num_running={self._num_running_batches}/{self._af_pipeline_num_micro_batch})"
                )
            
            # Clear the temporary attribute
            self._continuation_request_ids = set()

        elif self._cluster_type == ClusterType.DECODE_FFN:
            scheduled_batches.extend(self._m2n_immediate_batch_queue)
            self._m2n_immediate_batch_queue.clear()

            self._num_running_batches += len(scheduled_batches)
        else:
            # For prefill and monolithic clusters: use original logic based on pipeline stages
            while self._num_running_batches < self._num_stages:
                batch = self._get_next_batch(is_micro_batch=False)
                if not batch:
                    break
                scheduled_batches.append(batch)
                self._num_running_batches += 1
                if (
                    hasattr(self, "_has_monolithic_pp_mtp_output_wait")
                    and self._has_monolithic_pp_mtp_output_wait()
                ):
                    break
            if scheduled_batches and hasattr(
                self, "_clear_monolithic_pp_mtp_output_wait"
            ):
                self._clear_monolithic_pp_mtp_output_wait()

        return scheduled_batches

    def _create_m2n_transfer_events(self, batch: Batch, layer_id: int = None) -> List:
        """Disaggregated M2N transfer events are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def should_decrement_running_batches_on_layer_end(self, batch: Batch) -> bool:
        """
        Determine if _num_running_batches should be decremented when a batch completes a layer.

        For decode clusters, this depends on whether the batch has completed its final layer
        in the current pipeline stage (a->f or f->a).

        Args:
            batch: The batch that completed a layer

        Returns:
            bool: True if _num_running_batches should be decremented

        TODO: Implement proper logic based on pipeline stage tracking
        """
        if self._cluster_type in [ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN]:
            # TODO: Implement logic to track pipeline stages and determine final layer
            # For now, return True (temporary behavior)
            return True
        else:
            # For prefill/monolithic clusters, always decrement on batch end
            return True

    def add_batch_to_m2n_queue(self, batch):
        """
        Add a batch to the M2N immediate batch queue for DECODE_FFN cluster processing.

        This method is used by the cluster scheduler to add EP-distributed batches
        to the replica scheduler's processing queue.

        Args:
            batch: The batch portion assigned to this EP replica
        """
        if self._cluster_type == ClusterType.DECODE_FFN:
            self._m2n_immediate_batch_queue.append(batch)
            activation_bytes = getattr(batch, "activation_bytes", None)
            if activation_bytes is not None:
                self.add_activation_memory_bytes(int(activation_bytes))
        else:
            # For non-DECODE_FFN clusters, this method should not be called
            raise ValueError(f"add_batch_to_m2n_queue called on non-DECODE_FFN cluster: {self._cluster_type}")

    def add_activation_memory_bytes(self, activation_bytes: int) -> None:
        if self._cluster_type != ClusterType.DECODE_FFN:
            raise ValueError(
                f"add_activation_memory_bytes called on non-DECODE_FFN cluster: {self._cluster_type}"
            )
        if activation_bytes < 0:
            raise ValueError(
                f"activation_bytes must be non-negative, got {activation_bytes}"
            )
        self._activation_bytes_allocated += activation_bytes

    def release_activation_memory_bytes(self, activation_bytes: int) -> None:
        if self._cluster_type != ClusterType.DECODE_FFN:
            raise ValueError(
                f"release_activation_memory_bytes called on non-DECODE_FFN cluster: {self._cluster_type}"
            )
        if activation_bytes < 0:
            raise ValueError(
                f"activation_bytes must be non-negative, got {activation_bytes}"
            )
        self._activation_bytes_allocated -= activation_bytes
        if self._activation_bytes_allocated < 0:
            raise ValueError(
                f"activation_bytes_allocated below zero: {self._activation_bytes_allocated}"
            )

    def add_batch_to_immediate_queue(self, batch):
        """
        Add a batch to the immediate batch queue for DECODE_ATTN cluster processing.

        This method is used by the cluster scheduler to add batches returning from
        decode-ffn cluster directly to the replica scheduler's processing queue,
        preserving batch integrity and original replica/DP assignment.

        Args:
            batch: The batch returning from decode-ffn cluster
        """
        if self._cluster_type == ClusterType.DECODE_ATTN:
            if hasattr(self, '_af_immediate_batch_queue'):
                self._af_immediate_batch_queue.append(batch)
            else:
                # Initialize queue if not present (fallback)
                self._af_immediate_batch_queue = [batch]
        else:
            # For non-DECODE_ATTN clusters, this method should not be called
            raise ValueError(f"add_batch_to_immediate_queue called on non-DECODE_ATTN cluster: {self._cluster_type}")
