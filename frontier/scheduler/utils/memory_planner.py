from typing import Optional

from frontier.attention.memory import get_attention_runtime_kv_layout
from frontier.attention.model_binding import bind_attention_family
from frontier.config import ReplicaConfig
from frontier.errors import FrontierMemoryOOMError
from frontier.entities.replica import Replica
from frontier.spec_decode import method_uses_lookahead_slots
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


class MemoryPlanner:
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica: Replica,
        cluster_type: ClusterType = None,
        max_num_seqs: int | None = None,
    ) -> None:
        self._replica_config = replica_config
        self._replica = replica
        self._cluster_type = cluster_type
        if max_num_seqs is not None and int(max_num_seqs) <= 0:
            raise ValueError(f"max_num_seqs must be positive, got={max_num_seqs!r}")
        self._max_num_seqs = None if max_num_seqs is None else int(max_num_seqs)

        self._param_counter = ParamCounter(
            replica_config=replica_config,
            cluster_type=self._cluster_type,
        )

    def _get_effective_gpu_memory_utilization(
        self, gpu_memory_utilization: Optional[float]
    ) -> float:
        if gpu_memory_utilization is not None:
            if gpu_memory_utilization <= 0 or gpu_memory_utilization > 1.0:
                raise ValueError(
                    "gpu_memory_utilization must be in (0, 1], "
                    f"got={gpu_memory_utilization!r}"
                )
            return gpu_memory_utilization

        return 1 - self._replica.memory_margin_fraction

    def _get_requested_memory_bytes(
        self, gpu_memory_utilization: Optional[float]
    ) -> int:
        utilization = self._get_effective_gpu_memory_utilization(
            gpu_memory_utilization
        )
        return int(self._replica.total_memory_gb * 1024**3 * utilization)

    @staticmethod
    def _validate_non_kv_cache_overhead_bytes(
        non_kv_cache_overhead_bytes: int,
    ) -> int:
        if non_kv_cache_overhead_bytes < 0:
            raise ValueError(
                "non_kv_cache_overhead_bytes must be >= 0, "
                f"got={non_kv_cache_overhead_bytes!r}"
            )
        return int(non_kv_cache_overhead_bytes)

    def _raise_memory_oom(
        self,
        *,
        reason: str,
        message: str,
        details: dict[str, object],
    ) -> None:
        cluster_name = (
            self._cluster_type.name if self._cluster_type is not None else "MONOLITHIC"
        )
        error_details = {
            "cluster_type": cluster_name,
            "model_name": self._replica_config.model_config.get_name(),
            **details,
        }
        raise FrontierMemoryOOMError(
            message,
            reason=reason,
            details=error_details,
        )

    def _get_kv_cache_memory_per_layer_per_request(self) -> int:
        # DECODE_FFN does not allocate KV cache blocks.
        if self._cluster_type == ClusterType.DECODE_FFN:
            return 0

        return (
            2  # 2 bytes per fp16/bf16 element
            * self._get_kv_cache_elements_per_token_per_worker()
            * self._replica.max_request_tokens
        )

    def _model_has_gdn_layers(self) -> bool:
        getter = getattr(self._replica_config.model_config, "get_num_gdn_layers", None)
        return callable(getter) and int(getter()) > 0

    def _get_max_local_sequence_mixer_layer_count(self, *, gdn: bool) -> int:
        if self._cluster_type == ClusterType.DECODE_FFN:
            return 0
        model_config = self._replica_config.model_config
        if not self._model_has_gdn_layers():
            return 0 if gdn else int(self._replica.num_layers)
        layers_per_stage = int(self._replica.num_layers_per_pipeline_stage)
        num_pipeline_stages = int(self._replica.num_pipeline_stages)
        is_gdn_layer = model_config.is_gdn_layer
        counts = []
        for stage_id in range(num_pipeline_stages):
            first_layer_id = stage_id * layers_per_stage
            gdn_count = sum(
                bool(is_gdn_layer(layer_id))
                for layer_id in range(first_layer_id, first_layer_id + layers_per_stage)
            )
            counts.append(gdn_count if gdn else layers_per_stage - gdn_count)
        return max(counts, default=0)

    def _get_num_full_attention_layers_per_device(self) -> int:
        return self._get_max_local_sequence_mixer_layer_count(gdn=False)

    def _get_num_gdn_layers_per_device(self) -> int:
        return self._get_max_local_sequence_mixer_layer_count(gdn=True)

    def _get_gdn_state_memory_per_device_per_request(self) -> int:
        if self._cluster_type == ClusterType.DECODE_FFN:
            return 0
        getter = getattr(self._replica_config.model_config, "get_gdn_config", None)
        gdn_config = getter() if callable(getter) else None
        if gdn_config is None:
            return 0
        state_layout = gdn_config.get_state_layout(
            tensor_parallel_size=self._replica_config.attn_tensor_parallel_size,
        )
        return int(state_layout.total_bytes) * self._get_num_gdn_layers_per_device()

    def get_gdn_state_memory_per_device_per_request_bytes(self) -> int:
        return self._get_gdn_state_memory_per_device_per_request()

    def _get_gdn_state_reservation_bytes(self) -> int:
        state_per_request = self._get_gdn_state_memory_per_device_per_request()
        if state_per_request == 0:
            return 0
        max_num_seqs = getattr(self, "_max_num_seqs", None)
        if max_num_seqs is None:
            raise ValueError(
                "MemoryPlanner requires max_num_seqs to reserve fixed GDN state"
            )
        return state_per_request * int(max_num_seqs)

    def _get_kv_cache_elements_per_token_per_worker(self) -> int:
        if self._cluster_type == ClusterType.DECODE_FFN:
            return 0

        family = bind_attention_family(self._replica_config.model_config).family
        layout = get_attention_runtime_kv_layout(
            family,
            runtime_num_kv_heads_per_worker=(
                self._replica.kv_heads_per_tensor_parallel_worker
            ),
            runtime_head_size=self._replica.attention_head_dim,
        )
        return layout.elements_per_token_per_worker

    def _get_kv_cache_memory_per_layer_per_block(self, block_size: int) -> int:
        # DECODE_FFN does not allocate KV cache blocks.
        if self._cluster_type == ClusterType.DECODE_FFN:
            return 0

        return (
            2  # 2 bytes per fp16/bf16 element
            * block_size
            * self._get_kv_cache_elements_per_token_per_worker()
        )

    def _get_parameter_memory_per_device(self) -> int:
        return self._param_counter.get_parameter_memory_per_device_bytes()

    def get_parameter_memory_per_device_bytes(self) -> int:
        """Return parameter memory per device in bytes."""
        return self._get_parameter_memory_per_device()

    def _get_kv_cache_memory_per_device_per_request(self) -> int:
        # DECODE_FFN does not allocate KV cache blocks.
        if self._cluster_type == ClusterType.DECODE_FFN:
            return 0

        dense_kv_bytes = (
            self._get_kv_cache_memory_per_layer_per_request()
            * self._get_num_full_attention_layers_per_device()
        )
        return dense_kv_bytes + self._get_gdn_state_memory_per_device_per_request()

    def get_num_blocks(
        self,
        *,
        block_size: int,
        gpu_memory_utilization: Optional[float] = None,
        non_kv_cache_overhead_bytes: int = 0,
    ) -> int:
        """Estimate vLLM-style KV cache block count for one device.

        vLLM computes:
          requested_memory = total_memory * gpu_memory_utilization
          available_kv_cache_memory = requested_memory - non_kv_cache_memory
          num_blocks = floor(available_kv_cache_memory / page_size / num_layers)

        Frontier approximates non_kv_cache_memory with parameter memory
        plus an optional calibrated overhead term.
        """
        page_size = self._get_kv_cache_memory_per_layer_per_block(block_size)
        num_full_attention_layers = self._get_num_full_attention_layers_per_device()
        gdn_state_reservation_bytes = self._get_gdn_state_reservation_bytes()
        if page_size == 0 and gdn_state_reservation_bytes == 0:
            return 0

        requested_memory = self._get_requested_memory_bytes(gpu_memory_utilization)
        parameter_memory_per_device = self._get_parameter_memory_per_device()
        overhead_bytes = self._validate_non_kv_cache_overhead_bytes(
            non_kv_cache_overhead_bytes
        )

        if parameter_memory_per_device >= requested_memory:
            self._raise_memory_oom(
                reason="parameter_memory_exceeds_requested_budget",
                message=(
                    "Model parameter shard does not fit inside the requested GPU "
                    "memory budget."
                ),
                details={
                    "requested_memory_bytes": requested_memory,
                    "parameter_memory_per_device_bytes": parameter_memory_per_device,
                    "non_kv_cache_overhead_bytes": overhead_bytes,
                    "block_size": int(block_size),
                },
            )

        available_kv_cache_memory = (
            requested_memory
            - parameter_memory_per_device
            - overhead_bytes
            - gdn_state_reservation_bytes
        )
        if available_kv_cache_memory <= 0:
            self._raise_memory_oom(
                reason="insufficient_kv_cache_budget",
                message="No KV cache budget remains after subtracting weights and non-KV overhead.",
                details={
                    "requested_memory_bytes": requested_memory,
                    "parameter_memory_per_device_bytes": parameter_memory_per_device,
                    "non_kv_cache_overhead_bytes": overhead_bytes,
                    "gdn_state_reservation_bytes": gdn_state_reservation_bytes,
                    "available_kv_cache_memory_bytes": available_kv_cache_memory,
                    "block_size": int(block_size),
                    "page_size_bytes": int(page_size),
                    "num_layers": int(self._replica.num_layers),
                },
            )

        if page_size == 0 or num_full_attention_layers == 0:
            return 0

        num_blocks = int(
            available_kv_cache_memory // page_size // num_full_attention_layers
        )
        num_blocks = max(num_blocks, 0)

        if num_blocks <= 0:
            self._raise_memory_oom(
                reason="insufficient_kv_cache_budget",
                message="Not enough KV cache budget to allocate even one block.",
                details={
                    "requested_memory_bytes": requested_memory,
                    "parameter_memory_per_device_bytes": parameter_memory_per_device,
                    "available_kv_cache_memory_bytes": available_kv_cache_memory,
                    "non_kv_cache_overhead_bytes": overhead_bytes,
                    "page_size_bytes": int(page_size),
                    "num_full_attention_layers": num_full_attention_layers,
                    "num_gdn_layers": self._get_num_gdn_layers_per_device(),
                    "gdn_state_reservation_bytes": gdn_state_reservation_bytes,
                },
            )

        return num_blocks

    def get_max_batch_size(
        self,
        gpu_memory_utilization: Optional[float] = None,
        non_kv_cache_overhead_bytes: int = 0,
    ) -> int:
        """Return a legacy worst-case request-count estimate.

        Warning:
            This uses `replica.max_request_tokens` as a static per-request KV
            footprint proxy, so it is intentionally conservative and should not
            be used as the runtime admission bound for `vllm_v1` or `sglang`
            schedulers. Those schedulers admit work using token budget and
            block-level checks at runtime.
        """
        requested_memory = self._get_requested_memory_bytes(gpu_memory_utilization)
        parameter_memory_per_device = self._get_parameter_memory_per_device()
        overhead_bytes = self._validate_non_kv_cache_overhead_bytes(
            non_kv_cache_overhead_bytes
        )
        kv_cache_memory_per_device_per_request = (
            self._get_kv_cache_memory_per_device_per_request()
        )

        memory_for_kv_cache = (
            requested_memory - parameter_memory_per_device - overhead_bytes
        )

        # For clusters that do not use KV cache (e.g., DECODE_FFN),
        # the batch size is bounded by an activation-memory proxy.
        if kv_cache_memory_per_device_per_request == 0:
            if self._cluster_type == ClusterType.DECODE_FFN:
                estimated_memory_per_request = max(
                    parameter_memory_per_device // 100,
                    1024 * 1024,
                )
                number_of_requests = max(
                    1, memory_for_kv_cache // estimated_memory_per_request
                )
            else:
                raise NotImplementedError(
                    f"Unsupported cluster_type: {self._cluster_type}"
                )
        else:
            number_of_requests = (
                memory_for_kv_cache // kv_cache_memory_per_device_per_request
            )

        if number_of_requests <= 0:
            self._raise_memory_oom(
                reason="insufficient_request_budget",
                message="Not enough memory to store even a single worst-case request.",
                details={
                    "requested_memory_bytes": requested_memory,
                    "parameter_memory_per_device_bytes": parameter_memory_per_device,
                    "memory_for_kv_cache_bytes": memory_for_kv_cache,
                    "kv_cache_per_request_bytes": kv_cache_memory_per_device_per_request,
                    "gdn_state_per_request_bytes": (
                        self._get_gdn_state_memory_per_device_per_request()
                    ),
                    "non_kv_cache_overhead_bytes": overhead_bytes,
                },
            )

        return number_of_requests

    def get_max_request_slots(
        self,
        gpu_memory_utilization: Optional[float] = None,
        non_kv_cache_overhead_bytes: int = 0,
    ) -> int:
        return (
            self.get_max_batch_size(
                gpu_memory_utilization,
                non_kv_cache_overhead_bytes,
            )
            * self._replica.num_pipeline_stages
        )

    @staticmethod
    def estimate_speculative_lookahead_tokens(
        method: str, planned_draft_tokens: int
    ) -> int:
        planned = int(planned_draft_tokens)
        if planned < 0:
            raise ValueError(
                f"planned_draft_tokens must be >= 0, got={planned_draft_tokens}"
            )
        if not method_uses_lookahead_slots(method):
            return 0
        return planned
