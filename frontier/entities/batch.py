from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from frontier.entities.base_entity import BaseEntity
from frontier.entities.request import Request
from frontier.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class AFDStageMetadata:
    """Minimal AFD metadata for simulation.

    Simulator only needs values that affect compute time and communication volume.
    No tensor-level trimming is required.

    padded_total_tokens: attention-side padded total (stage + DP + CUDA Graph)
    ffn_compute_total_tokens: decode-ffn compute total after batch-level CUDA Graph padding

    Aligned with StepFun-vLLM's AFDMetadata but simplified for simulation purposes.
    See: vllm/forward_context.py::AFDMetadata
    """

    num_stages: int
    original_total_tokens: int
    padded_total_tokens: int
    ffn_compute_total_tokens: int

    @property
    def num_pad_tokens(self) -> int:
        """Number of padding tokens added."""
        return self.padded_total_tokens - self.original_total_tokens

    @staticmethod
    def _pad_total_tokens_to_capture_size(
        total_tokens: int, cudagraph_capture_sizes: List[int]
    ) -> int:
        """Pad total tokens to the nearest CUDA Graph capture size."""
        sorted_sizes = sorted(cudagraph_capture_sizes)
        for size in sorted_sizes:
            if total_tokens <= size:
                return size
        return total_tokens

    @classmethod
    def compute_stage_token_lens(
        cls,
        num_reqs: int,
        num_tokens_per_req: List[int],
        num_stages: int,
    ) -> List[int]:
        """Compute per-stage original token lengths following StepFun partitioning.

        StepFun-vLLM partitioning logic (from gpu_model_runner.py::_prepare_inputs):
        - If num_reqs >= num_stages: divide requests evenly across stages
        - If num_reqs < num_stages: each request gets its own stage

        Args:
            num_reqs: Number of requests in batch
            num_tokens_per_req: Token count for each request
            num_stages: Target number of stages

        Returns:
            List of token counts per stage (length may be < num_stages if num_reqs < num_stages)
        """
        # Step 1: Compute stage request boundaries (same as StepFun)
        if num_reqs >= num_stages:
            num_reqs_per_stage = num_reqs // num_stages
            stage_reqs_start_loc = [num_reqs_per_stage * i for i in range(num_stages + 1)]
            stage_reqs_start_loc[-1] = num_reqs
        else:
            # Each request is its own stage
            stage_reqs_start_loc = list(range(num_reqs + 1))

        # Step 2: Compute cumulative sum of tokens
        cumsum = [0]
        for n in num_tokens_per_req:
            cumsum.append(cumsum[-1] + n)

        # Step 3: Compute per-stage token lengths
        stage_tokens_lens: List[int] = []
        for i in range(len(stage_reqs_start_loc) - 1):
            start_req = stage_reqs_start_loc[i]
            end_req = stage_reqs_start_loc[i + 1]
            stage_tokens_lens.append(cumsum[end_req] - cumsum[start_req])

        return stage_tokens_lens

    @classmethod
    def apply_padding(
        cls,
        stage_tokens_lens: List[int],
        num_stages: int,
        dp_stage_max_tokens: Optional[List[int]] = None,
        use_cuda_graph: bool = False,
        cudagraph_capture_sizes: Optional[List[int]] = None,
    ) -> List[int]:
        """Apply three-layer padding following StepFun order.

        StepFun-vLLM padding order (from gpu_model_runner.py::get_afd_padding):
        1. Stage count padding: pad to reach num_stages (add dummy stages with 1 token)
        2. DP padding: take per-stage max across DP ranks
        3. CUDA Graph padding: pad each stage to nearest capture size

        Args:
            stage_tokens_lens: Per-stage token counts before padding
            num_stages: Target number of stages
            dp_stage_max_tokens: Optional per-stage max tokens across DP ranks (pre-computed)
            use_cuda_graph: Whether CUDA Graph is enabled
            cudagraph_capture_sizes: List of valid capture sizes (ascending order)

        Returns:
            Padded per-stage token counts
        """
        padded_lens = list(stage_tokens_lens)

        # Layer 1: Stage count padding (add dummy stages with 1 token each)
        # Aligned with StepFun: vllm/v1/worker/gpu_model_runner.py lines 680-683
        while len(padded_lens) < num_stages:
            padded_lens.append(1)

        # Layer 2: DP padding (take per-stage max across DP ranks)
        # Aligned with StepFun: vllm/v1/worker/gpu_model_runner.py lines 685-691
        if dp_stage_max_tokens is not None:
            if len(dp_stage_max_tokens) != len(padded_lens):
                raise ValueError(
                    f"dp_stage_max_tokens length ({len(dp_stage_max_tokens)}) "
                    f"must match padded stage count ({len(padded_lens)})"
                )
            padded_lens = [max(p, m) for p, m in zip(padded_lens, dp_stage_max_tokens)]

        # Layer 3: CUDA Graph padding (pad to nearest capture size)
        # Aligned with StepFun: vllm/v1/worker/gpu_model_runner.py lines 693-701
        if use_cuda_graph and cudagraph_capture_sizes:
            sorted_sizes = sorted(cudagraph_capture_sizes)

            def pad_to_capture_size(n: int) -> int:
                for s in sorted_sizes:
                    if n <= s:
                        return s
                # Larger than max capture size: leave as-is, will run eager
                return n

            padded_lens = [pad_to_capture_size(n) for n in padded_lens]

        return padded_lens

    @classmethod
    def from_batch_params(
        cls,
        num_reqs: int,
        num_tokens_per_req: List[int],
        num_stages: int,
        dp_stage_max_tokens: Optional[List[int]] = None,
        use_cuda_graph: bool = False,
        cudagraph_capture_sizes: Optional[List[int]] = None,
        ffn_use_cuda_graph: bool = False,
        ffn_cudagraph_capture_sizes: Optional[List[int]] = None,
    ) -> "AFDStageMetadata":
        """Create AFDStageMetadata from batch parameters.

        This is the main factory method that computes padded token totals
        following StepFun's three-layer padding order.

        Args:
            num_reqs: Number of requests in batch
            num_tokens_per_req: Token count for each request
            num_stages: Target number of stages (from af_pipeline_num_micro_batch)
            dp_stage_max_tokens: Pre-computed per-stage max across DP ranks
            use_cuda_graph: Whether CUDA Graph is enabled
            cudagraph_capture_sizes: List of valid CUDA Graph capture sizes

        Returns:
            AFDStageMetadata with original and padded token counts

        Raises:
            ValueError: If num_stages <= 0
        """
        if num_stages <= 0:
            raise ValueError(
                f"num_stages must be positive, got {num_stages}. "
                f"AFD stage count should be set via af_pipeline_num_micro_batch."
            )

        stage_tokens_lens = cls.compute_stage_token_lens(
            num_reqs, num_tokens_per_req, num_stages
        )
        padded_tokens_lens = cls.apply_padding(
            stage_tokens_lens=stage_tokens_lens,
            num_stages=num_stages,
            dp_stage_max_tokens=dp_stage_max_tokens,
            use_cuda_graph=use_cuda_graph,
            cudagraph_capture_sizes=cudagraph_capture_sizes,
        )

        ffn_compute_total_tokens = sum(padded_tokens_lens)
        if ffn_use_cuda_graph and ffn_cudagraph_capture_sizes:
            ffn_compute_total_tokens = cls._pad_total_tokens_to_capture_size(
                ffn_compute_total_tokens, ffn_cudagraph_capture_sizes
            )

        return cls(
            num_stages=num_stages,
            original_total_tokens=sum(stage_tokens_lens),
            padded_total_tokens=sum(padded_tokens_lens),
            ffn_compute_total_tokens=ffn_compute_total_tokens,
        )

    def with_dp_padding(
        self,
        dp_stage_max_tokens: List[int],
        ffn_cudagraph_capture_sizes: Optional[List[int]] = None,
    ) -> "AFDStageMetadata":
        """Return a new AFDStageMetadata with DP padding (Layer 2) applied.

        Called by BaseClusterScheduler._apply_dp_padding_on_promotion() after
        all DP lanes arrive at the (layer_id, afd_stage_idx) barrier. Each
        stage's token count is replaced by the max across all DP ranks for
        that stage, ensuring all ranks send/recv identical activation sizes.

        Since AFDStageMetadata is frozen, this creates a new instance.

        Reference: StepFun-vLLM gpu_model_runner.py:1240-1244

        Args:
            dp_stage_max_tokens: Per-stage max tokens across all DP ranks
            ffn_cudagraph_capture_sizes: Optional FFN CUDA Graph capture sizes

        Returns:
            New AFDStageMetadata with DP-padded values
        """
        if len(dp_stage_max_tokens) != self.num_stages:
            raise ValueError(
                f"dp_stage_max_tokens length ({len(dp_stage_max_tokens)}) "
                f"must match num_stages ({self.num_stages})"
            )

        padded_total = sum(dp_stage_max_tokens)
        ffn_total = padded_total
        if ffn_cudagraph_capture_sizes:
            ffn_total = self._pad_total_tokens_to_capture_size(
                ffn_total, ffn_cudagraph_capture_sizes
            )

        return AFDStageMetadata(
            num_stages=self.num_stages,
            original_total_tokens=self.original_total_tokens,
            padded_total_tokens=padded_total,
            ffn_compute_total_tokens=ffn_total,
        )


@dataclass(frozen=True)
class DecodeCudaGraphMetadata:
    """VLLM V1 decode CUDA Graph metadata for monolithic/PD decode runtime."""

    config_mode: str
    runtime_mode: str
    capture_hit: bool
    is_mixed_batch: bool
    original_total_tokens: int
    padded_total_tokens: int
    original_decode_batch_size: int
    padded_decode_batch_size: int

    def get_effective_total_tokens_for_compute(self) -> int:
        if self.runtime_mode in {"FULL", "PIECEWISE"}:
            return self.padded_total_tokens
        return self.original_total_tokens

    def get_effective_decode_batch_size_for_attention(self) -> int:
        if self.runtime_mode == "FULL":
            return self.padded_decode_batch_size
        return self.original_decode_batch_size


@dataclass
class SpecDecodeBatchMetadata:
    method: str
    planned_draft_tokens_per_request: List[int]
    verify_tokens_per_request: List[int]
    accepted_draft_tokens_per_request: List[int]
    rejected_draft_tokens_per_request: List[int]
    committed_tokens_per_request: List[int]
    uses_lookahead_slots: bool
    terminal_overshoot_planned_draft_tokens_per_request: Optional[List[List[int]]] = None
    terminal_overshoot_verify_tokens_per_request: Optional[List[List[int]]] = None
    terminal_overshoot_accepted_draft_tokens_per_request: Optional[List[List[int]]] = None
    terminal_overshoot_rejected_draft_tokens_per_request: Optional[List[List[int]]] = None
    terminal_overshoot_raw_committed_tokens_per_request: Optional[List[List[int]]] = None

    def validate(self, num_requests: int) -> None:
        vectors = [
            self.planned_draft_tokens_per_request,
            self.verify_tokens_per_request,
            self.accepted_draft_tokens_per_request,
            self.rejected_draft_tokens_per_request,
            self.committed_tokens_per_request,
        ]
        names = [
            "planned_draft_tokens_per_request",
            "verify_tokens_per_request",
            "accepted_draft_tokens_per_request",
            "rejected_draft_tokens_per_request",
            "committed_tokens_per_request",
        ]
        for name, values in zip(names, vectors):
            if len(values) != num_requests:
                raise ValueError(
                    f"{name} length mismatch: expected={num_requests}, got={len(values)}"
                )
            for value in values:
                if int(value) < 0:
                    raise ValueError(f"{name} entries must be >= 0, got={value!r}")

        nested_vectors = [
            self.terminal_overshoot_planned_draft_tokens_per_request,
            self.terminal_overshoot_verify_tokens_per_request,
            self.terminal_overshoot_accepted_draft_tokens_per_request,
            self.terminal_overshoot_rejected_draft_tokens_per_request,
            self.terminal_overshoot_raw_committed_tokens_per_request,
        ]
        nested_names = [
            "terminal_overshoot_planned_draft_tokens_per_request",
            "terminal_overshoot_verify_tokens_per_request",
            "terminal_overshoot_accepted_draft_tokens_per_request",
            "terminal_overshoot_rejected_draft_tokens_per_request",
            "terminal_overshoot_raw_committed_tokens_per_request",
        ]
        for name, values in zip(nested_names, nested_vectors):
            if values is None:
                continue
            if len(values) != num_requests:
                raise ValueError(
                    f"{name} length mismatch: expected={num_requests}, got={len(values)}"
                )
            for row in values:
                for value in row:
                    if int(value) < 0:
                        raise ValueError(f"{name} entries must be >= 0, got={value!r}")
        if all(values is not None for values in nested_vectors):
            for request_index in range(num_requests):
                expected_len = len(nested_vectors[0][request_index])
                for name, values in zip(nested_names[1:], nested_vectors[1:]):
                    if len(values[request_index]) != expected_len:
                        raise ValueError(
                            "terminal overshoot metadata row length mismatch: "
                            f"request_index={request_index}, "
                            f"expected={expected_len}, {name}="
                            f"{len(values[request_index])}"
                        )


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


class Batch(BaseEntity):
    def __init__(
        self,
        replica_id: int,
        requests: List[Request],
        num_tokens: List[int],
        is_idle: bool = False,
        is_moe: bool = None,
    ) -> None:
        if is_moe is None:
            raise ValueError("Batch.is_moe must be explicitly set")
        self._id = Batch.generate_id()

        # PD-AF Disaggregation support
        # preserve the original replica ID and DP ID for batches in decode-attn cluster
        self.decode_attn_original_replica_id: Optional[int] = None
        self.decode_attn_original_dp_id: Optional[int] = None

        # In the DECODE_FFN cluster, an original batch is decomposed into multiple EP sub-batches:
        # Therefore, we need a global batch ID to track all the sub-batches
        # # used for EP batch synchronization (in decode-ffn)
        # e.g.,
        # original batch：batch.id = 6
        # EP sub-batches：
        # - ep_batch_0: batch.id = 8, batch.global_id = 6
        # - ep_batch_1: batch.id = 9, batch.global_id = 6
        self._global_id = -1

        self._replica_id = replica_id
        self._requests = requests
        self._num_tokens: List[int] = num_tokens
        self._total_num_tokens: int = sum(num_tokens)
        self._num_prefill_tokens = sum(
            [
                (t if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )

        # TODO: why this is needed?
        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False
        self._completed = False
        self._schedule_epoch = 0

        # Time attribute for timing information preservation
        self._time = None

        # KV cache transfer support
        self._requires_decode_processing = False

        # af common layer count (update in decode-ffn; )
        self._af_common_layer_count = -1

        # num routing tokens (used for moe clusters)
        self._num_routing_tokens = -1

        # Idle batch flag for DP synchronization
        # Idle batches are created when num_requests < attn_dp_size to ensure
        # all DP replicas can participate in MoE synchronization.
        # Idle batches skip attention computation but participate in MoE sync.
        self._is_idle = is_idle
        self._is_moe = is_moe
        self.decode_cuda_graph_metadata: Optional[DecodeCudaGraphMetadata] = None
        self._request_runtime_epochs = [
            int(getattr(request, "runtime_epoch", 0)) for request in self._requests
        ]
        self._request_execution_signatures = [
            self._get_request_execution_signature(request) for request in self._requests
        ]
        self._thinking_round_start_times = [
            self._get_thinking_round_start_time(request) for request in self._requests
        ]
        self._request_mutation_signatures = [
            self._get_request_mutation_signature(request) for request in self._requests
        ]

        # AFD stage metadata for StepFun-vLLM alignment
        # Contains padded token counts for compute/communication prediction
        # Set by scheduler when num_stages > 1 in DECODE_ATTN cluster
        self.afd_stage_metadata: Optional[AFDStageMetadata] = None
        # AFD stage index for StepFun-vLLM alignment (micro-batch stage id)
        self.afd_stage_idx: Optional[int] = None
        self.spec_decode_metadata: Optional[SpecDecodeBatchMetadata] = None
        self._spec_terminal_completion_delay_s_by_request: Dict[int, float] = {}

    @staticmethod
    def _get_request_execution_signature(request: Request) -> tuple[int, int, int]:
        """Capture the request execution epoch for stale-batch detection.

        A batch stores direct references to mutable Request objects. In parallel mode,
        a later round or restart can advance the shared Request state before an older
        batch-end event is processed on another cluster thread. In that case, the old
        batch must not mutate the newer Request state.
        """
        return (
            request.current_thinking_round_index,
            request.num_restarts,
            request.execution_epoch,
        )

    @staticmethod
    def _get_request_mutation_signature(request: Request) -> tuple[int, int, int, int]:
        """Capture decode-step progress for stale batch-end mutation checks.

        The decode-token cursor advances once per committed decode iteration and
        cleanly distinguishes stale same-round decode callbacks without being
        perturbed by prefix-cache hits or chunked-prefill bookkeeping.
        """
        execution_signature = Batch._get_request_execution_signature(request)
        return (*execution_signature, request.current_decode_token_index)

    def add_spec_terminal_completion_delay(
        self,
        request_ids: Iterable[int],
        delay_s: float,
    ) -> None:
        """Accumulate terminal speculative batch service for request metrics."""
        delay_value = float(delay_s)
        if delay_value < 0.0:
            raise ValueError(
                f"spec terminal completion delay must be >= 0, got={delay_value}"
            )
        if delay_value == 0.0:
            return
        for request_id in request_ids:
            request_id_int = int(request_id)
            self._spec_terminal_completion_delay_s_by_request[request_id_int] = (
                self._spec_terminal_completion_delay_s_by_request.get(
                    request_id_int,
                    0.0,
                )
                + delay_value
            )

    def get_spec_terminal_completion_delay(self, request: Request) -> float:
        """Return accumulated terminal speculative delay for a request."""
        return float(
            self._spec_terminal_completion_delay_s_by_request.get(
                int(request.id),
                0.0,
            )
        )

    @staticmethod
    def _get_thinking_round_start_time(request: Request) -> Optional[float]:
        if (
            request.is_thinking_mode_enabled
            and request.thinking_home_cluster_type is not None
        ):
            return request.get_cluster_arrival_time(request.thinking_home_cluster_type)
        return None

    def _request_execution_matches_snapshot(self, index: int) -> bool:
        request = self._requests[index]
        expected_signature = self._request_execution_signatures[index]
        current_signature = self._get_request_execution_signature(request)
        return current_signature == expected_signature

    @staticmethod
    def _thinking_round_start_is_in_future(
        thinking_round_start_time: Optional[float],
        event_time: float,
    ) -> bool:
        return (
            thinking_round_start_time is not None
            and thinking_round_start_time > event_time + 1e-9
        )

    @property
    def current_execution_requests(self) -> List[Request]:
        current_requests: List[Request] = []
        seen_request_ids: set[int] = set()
        for index, request in enumerate(self._requests):
            if request.id in seen_request_ids:
                continue
            seen_request_ids.add(request.id)
            if self._request_execution_matches_snapshot(index):
                current_requests.append(request)
        return current_requests

    @property
    def request_execution_signatures(self) -> List[tuple[int, int, int]]:
        return list(self._request_execution_signatures)

    @property
    def request_mutation_signatures(self) -> List[tuple[int, int, int, int]]:
        return list(self._request_mutation_signatures)

    @property
    def request_runtime_epochs(self) -> List[int]:
        return list(self._request_runtime_epochs)

    @property
    def thinking_round_start_times(self) -> List[Optional[float]]:
        return list(self._thinking_round_start_times)

    def set_global_id(self, global_id: int):
        self._global_id = global_id

    # TODO: IS REDUNDANT (EQUAL TO _total_num_tokens)?
    @property
    def num_routing_tokens(self) -> int:
        if self.all_requests_ongoing_decoding:  
            # decoding stage
            self._num_routing_tokens = len(self._requests)
        else:
            # prefill stage
            self._num_routing_tokens = self.num_prefill_tokens
            # for req in self._requests:
            #     self._num_routing_tokens += req.num_prefill_tokens
        return self._num_routing_tokens


    @property
    def af_inflight_layer_count(self) -> int:
        # used for micro batch only
        assert len(self._requests) > 0, "Batch has no requests"
        
        # ISSUE-009 ROOT FIX: Use first non-completed request's layer count
        # to be consistent with _get_current_layer_id_from_batch() in cluster_batch_end_event.py.
        # When a request completes all decode tokens but the batch continues A↔F ping-pong
        # for other requests, the completed request's layer count may be stale (not incremented
        # due to ISSUE-006 fix that skips completed requests in mb_on_step_layer_count_increment).
        
        # Collect non-completed requests for layer-consistent validation
        non_completed_requests = [r for r in self._requests if not r.completed]
        
        if non_completed_requests:
            # Verify layer-consistent grouping among non-completed requests only
            if len(non_completed_requests) > 1:
                layer_counts = [r._completed_layer_count for r in non_completed_requests]
                if layer_counts[0] != layer_counts[1]:
                    # Provide detailed error information for debugging
                    request_details = [
                        f"req={r.id}|layer={r._completed_layer_count}|token_idx={r._current_decode_token_index}|completed={r.completed}"
                        for r in self._requests
                    ]
                    raise AssertionError(
                        f"Layer count mismatch in batch {self._id}: "
                        f"non-completed requests have inconsistent _completed_layer_count values. "
                        f"Details: [{', '.join(request_details)}]. "
                        f"This indicates a bug in batch formation - requests at different layers "
                        f"should not be grouped together."
                    )
            
            # Use first non-completed request's layer count
            self._af_common_layer_count = non_completed_requests[0]._completed_layer_count
            return self._af_common_layer_count
        
        # All requests completed - return first request's layer count
        self._af_common_layer_count = self._requests[0]._completed_layer_count
        return self._af_common_layer_count

    @property
    def global_id(self) -> int:
        return self._global_id

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def creation_time(self) -> float:
        return self._creation_time

    @property
    def time(self) -> float:
        """Get the time attribute for timing information preservation."""
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        """Set the time attribute for timing information preservation."""
        self._time = value

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self.total_num_tokens - self.num_prefill_tokens

    def get_decode_cuda_graph_runtime_mode(self) -> str:
        if self.decode_cuda_graph_metadata is None:
            return "NONE"
        return self.decode_cuda_graph_metadata.runtime_mode

    @property
    def is_pure_decode_batch(self) -> bool:
        return self.num_prefill_tokens == 0 and self.num_decode_tokens > 0

    def get_effective_total_tokens_for_compute(
        self, cluster_type: "ClusterType" = None
    ) -> int:
        """Get effective total tokens for compute prediction.

        For AFD-enabled batches:
        - DECODE_ATTN uses attention-side padded_total_tokens
        - DECODE_FFN uses batch-level CUDA Graph padding (ffn_compute_total_tokens)

        Args:
            cluster_type: The cluster type for context-aware token selection

        Returns:
            Effective token count for compute prediction
        """
        # Import here to avoid circular imports
        from frontier.types import ClusterType

        if cluster_type in (ClusterType.MONOLITHIC, ClusterType.DECODE):
            if self.decode_cuda_graph_metadata is not None:
                return (
                    self.decode_cuda_graph_metadata.get_effective_total_tokens_for_compute()
                )

        if self.afd_stage_metadata is not None:
            if cluster_type == ClusterType.DECODE_ATTN:
                return self.afd_stage_metadata.padded_total_tokens
            if cluster_type == ClusterType.DECODE_FFN:
                return self.afd_stage_metadata.ffn_compute_total_tokens

        spec_metadata = getattr(self, "spec_decode_metadata", None)
        if spec_metadata is not None:
            method = str(getattr(spec_metadata, "method", "")).strip()
            from frontier.spec_decode.mtp_registry import is_target_embedded_mtp_method

            if is_target_embedded_mtp_method(method):
                # vLLM target-embedded MTP target forward consumes the
                # scheduler-visible tokens plus scheduled draft tokens.
                return self._total_num_tokens + sum(
                    int(value)
                    for value in spec_metadata.planned_draft_tokens_per_request
                )

        return self._total_num_tokens

    def get_effective_total_tokens_for_transfer(
        self, cluster_type: "ClusterType" = None
    ) -> int:
        """Get effective total tokens for transfer size prediction.

        For AFD-enabled batches in DECODE_ATTN/DECODE_FFN clusters, returns
        attention-side padded_total_tokens. FFN batch-level CUDA Graph padding
        does not affect transfer size.
        """
        # Import here to avoid circular imports
        from frontier.types import ClusterType

        if self.afd_stage_metadata is not None:
            if cluster_type in (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
                return self.afd_stage_metadata.padded_total_tokens

        spec_metadata = getattr(self, "spec_decode_metadata", None)
        if spec_metadata is not None and bool(spec_metadata.uses_lookahead_slots):
            method = str(getattr(spec_metadata, "method", "")).strip()
            from frontier.spec_decode.mtp_registry import is_target_embedded_mtp_method

            if is_target_embedded_mtp_method(method):
                # Target-embedded MTP PP traces carry the scheduled verification
                # payload. Adding reserved draft slots here double-counts the
                # draft payload and breaks PP overhead lookup parity.
                return self._total_num_tokens
            lookahead_tokens = 0
            for request in self.requests:
                speculative_tokens = getattr(
                    request, "spec_num_speculative_tokens", None
                )
                if speculative_tokens is None:
                    raise ValueError(
                        "Spec-decode lookahead transfer sizing requires "
                        "request.spec_num_speculative_tokens."
                    )
                speculative_tokens = int(speculative_tokens)
                if speculative_tokens < 0:
                    raise ValueError(
                        "request.spec_num_speculative_tokens must be >= 0, "
                        f"got {speculative_tokens}."
                    )
                lookahead_tokens += speculative_tokens
            return self._total_num_tokens + lookahead_tokens

        return self._total_num_tokens

    def get_effective_total_tokens(self, cluster_type: "ClusterType" = None) -> int:
        """Backward-compatible token selection for transfer prediction."""
        return self.get_effective_total_tokens_for_transfer(cluster_type)

    def get_effective_total_tokens_rounded(self, cluster_type: "ClusterType" = None) -> int:
        """Get compute-effective tokens for predictor/cache lookup.

        NOTE: The method name is kept for backward compatibility. For vLLM V1
        eager-path parity, this helper no longer applies additional fixed
        multiple-of-8 rounding. CUDA Graph related padding is represented by
        AFDStageMetadata and consumed through get_effective_total_tokens_for_compute().
        """
        return self.get_effective_total_tokens_for_compute(cluster_type)

    def get_effective_decode_batch_size_for_attention(self) -> int:
        if self.decode_cuda_graph_metadata is not None:
            return (
                self.decode_cuda_graph_metadata.get_effective_decode_batch_size_for_attention()
            )

        return sum(
            1
            for request in self.requests
            if getattr(
                request,
                "is_prefill_complete",
                getattr(request, "_is_prefill_complete", False),
            )
        )

    @property
    def is_moe(self) -> bool:
        return self._is_moe

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def requires_decode_processing(self) -> bool:
        """Check if this batch requires decode processing after prefill."""
        return self._requires_decode_processing

    def set_requires_decode_processing(self, requires_decode: bool = True) -> None:
        """Set whether this batch requires decode processing."""
        self._requires_decode_processing = requires_decode

    @property
    def is_idle(self) -> bool:
        """
        Check if this is an idle batch.

        Idle batches are created when num_requests < attn_dp_size to ensure
        all DP replicas can participate in MoE synchronization.
        Idle batches skip attention computation but participate in MoE sync.
        """
        return self._is_idle

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def schedule_epoch(self) -> int:
        return self._schedule_epoch

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def all_requests_completed(self) -> bool:
        return all([request.completed for request in self._requests])

    # not include first to second decode token processing
    @property
    def all_requests_started_decode_and_not_completed(self) -> bool:
        return all([request.has_started_decode for request in self._requests]) and not self.all_requests_completed
    
    # include first to second decode token processing
    @property
    def all_requests_ongoing_decoding(self) -> bool:
        return all([request.ongoing_decoding for request in self._requests])
        
    @property
    def all_requests_early_decoding_on_first_layer(self) -> bool:
        return all([request.early_decoding_on_first_layer for request in self._requests])

    def mb_on_step_layer_count_increment(self, num_layers_completed: int = 1) -> None:
        # Debug: log per-request layer counters before increment
        try:
            from frontier.logger import init_logger
            _logger = init_logger(__name__)
            before = [
                f"req={r.id}|completed_layers={getattr(r,'completed_layer_count',None)}|completed={getattr(r,'completed',None)}"
                for r in self._requests
            ]
            _logger.debug(f"[MB-LAYER][BEFORE] mb={self.id} {', '.join(before)}; step={num_layers_completed}")
        except Exception:
            pass
        
        # ISSUE-006 FIX: Skip completed requests to prevent layer_id overflow.
        # When a request completes all decode tokens but the batch continues A↔F ping-pong
        # for other requests, we must not increment the completed request's layer count.
        skipped_count = 0
        for request in self._requests:
            if request.completed:
                skipped_count += 1
                continue
            request.mb_on_step_layer_count_increment()
        
        # Debug: log after increment
        try:
            after = [
                f"req={r.id}|completed_layers={getattr(r,'completed_layer_count',None)}|completed={getattr(r,'completed',None)}"
                for r in self._requests
            ]
            _logger.debug(f"[MB-LAYER][AFTER] mb={self.id} {', '.join(after)}; skipped={skipped_count}")
        except Exception:
            pass

    def on_schedule(
        self,
        time: float,
        cluster_type: "ClusterType" = None,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True
        self._schedule_epoch += 1
        self._request_runtime_epochs = [
            int(getattr(request, "runtime_epoch", 0)) for request in self._requests
        ]
        self._request_execution_signatures = [
            self._get_request_execution_signature(request) for request in self._requests
        ]
        self._thinking_round_start_times = [
            self._get_thinking_round_start_time(request) for request in self._requests
        ]
        self._request_mutation_signatures = [
            self._get_request_mutation_signature(request) for request in self._requests
        ]

        for request in self._requests:
            request.on_batch_schedule(time, cluster_type)

    def on_batch_end(
        self,
        time: float,
        cluster_type: "ClusterType" = None,
        request_execution_signatures: Optional[List[tuple[int, int, int]]] = None,
        request_mutation_signatures: Optional[List[tuple[int, int, int, int]]] = None,
        thinking_round_start_times: Optional[List[Optional[float]]] = None,
    ):
        self._completed = True
        self._completed_at = time

        # When requests are replicated across DP replicas for MoE synchronization,
        # the same request object may appear multiple times in the batch.
        # We need to ensure each unique request's on_batch_end() is called only once.
        if self.spec_decode_metadata is not None:
            self.spec_decode_metadata.validate(len(self._requests))

        expected_execution_signatures = (
            self._request_execution_signatures
            if request_execution_signatures is None
            else request_execution_signatures
        )
        expected_mutation_signatures = (
            self._request_mutation_signatures
            if request_mutation_signatures is None
            else request_mutation_signatures
        )
        expected_round_start_times = (
            self._thinking_round_start_times
            if thinking_round_start_times is None
            else thinking_round_start_times
        )
        seen_request_ids = set()
        for idx, (request, num_tokens) in enumerate(zip(self._requests, self._num_tokens)):
            if request.id not in seen_request_ids:
                expected_execution_signature = expected_execution_signatures[idx]
                current_execution_signature = self._get_request_execution_signature(
                    request
                )
                if current_execution_signature != expected_execution_signature:
                    logger.warning(
                        "[STALE-BATCH-END] Skipping request mutation for batch=%s req=%s "
                        "expected_signature=%s current_signature=%s",
                        self._id,
                        request.id,
                        expected_execution_signature,
                        current_execution_signature,
                    )
                    seen_request_ids.add(request.id)
                    continue
                expected_mutation_signature = expected_mutation_signatures[idx]
                current_mutation_signature = self._get_request_mutation_signature(
                    request
                )
                if current_mutation_signature != expected_mutation_signature:
                    logger.warning(
                        "[STALE-BATCH-END-PROGRESS] Skipping request mutation for "
                        "batch=%s req=%s expected_signature=%s current_signature=%s",
                        self._id,
                        request.id,
                        expected_mutation_signature,
                        current_mutation_signature,
                    )
                    seen_request_ids.add(request.id)
                    continue
                expected_round_start = expected_round_start_times[idx]
                if self._thinking_round_start_is_in_future(
                    expected_round_start,
                    time,
                ):
                    logger.warning(
                        "[STALE-BATCH-END-FUTURE-ROUND-START] Skipping request mutation "
                        "for batch=%s req=%s expected_round_start=%s event_time=%s",
                        self._id,
                        request.id,
                        expected_round_start,
                        time,
                    )
                    seen_request_ids.add(request.id)
                    continue
                effective_tokens = num_tokens
                if self.spec_decode_metadata is not None:
                    effective_tokens = self.spec_decode_metadata.committed_tokens_per_request[idx]
                request_completion_time = time
                processed_tokens = int(getattr(request, "num_processed_tokens", 0))
                total_tokens = int(getattr(request, "total_tokens", processed_tokens))
                will_complete = processed_tokens + int(effective_tokens) >= total_tokens
                terminal_delay = self.get_spec_terminal_completion_delay(request)
                if terminal_delay > 0.0:
                    already_post_first = (
                        float(
                            getattr(
                                request,
                                "first_decode_token_completed_at",
                                0.0,
                            )
                        )
                        > 0.0
                    )
                    if already_post_first and not will_complete:
                        request.add_spec_post_first_service_delay(terminal_delay)
                if (
                    will_complete
                    and request.spec_post_first_service_delay > 0.0
                ):
                    request_completion_time = (
                        time + request.spec_post_first_service_delay
                    )
                request.on_batch_end(
                    request_completion_time,
                    effective_tokens,
                    cluster_type,
                )
                seen_request_ids.add(request.id)

    def on_cluster_stage_end(self, time: float, cluster_type: "ClusterType" = None) -> None:
        """Lightweight hook for cluster-internal stage completion.

        Per-stage execution/model times are already recorded via BatchStage callbacks.
        This method exists to keep a clear semantic separation and future extensibility.
        """
        # Currently no per-request updates are required here.
        return

    @property
    def preempted_requests(self) -> List[Request]:
        return [
            request
            for request in self.current_execution_requests
            if request.preempted
        ]

    @property
    def completed_requests(self) -> List[Request]:
        return [
            request
            for request in self.current_execution_requests
            if request.completed
        ]

    def to_dict(self) -> dict:
        result = {
            "id": self._id,
            "global_id": self._global_id,
            "size": self.size,
            "replica_id": self._replica_id,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "num_decode_tokens": self.num_decode_tokens,
            "is_moe": self._is_moe,
        }
        # Include AFD metadata if present
        if self.afd_stage_metadata is not None:
            result["afd_stage_metadata"] = {
                "num_stages": self.afd_stage_metadata.num_stages,
                "original_total_tokens": self.afd_stage_metadata.original_total_tokens,
                "padded_total_tokens": self.afd_stage_metadata.padded_total_tokens,
                "ffn_compute_total_tokens": self.afd_stage_metadata.ffn_compute_total_tokens,
            }
        if self.spec_decode_metadata is not None:
            result["spec_decode_metadata"] = {
                "method": self.spec_decode_metadata.method,
                "planned_draft_tokens_per_request": self.spec_decode_metadata.planned_draft_tokens_per_request,
                "verify_tokens_per_request": self.spec_decode_metadata.verify_tokens_per_request,
                "accepted_draft_tokens_per_request": self.spec_decode_metadata.accepted_draft_tokens_per_request,
                "rejected_draft_tokens_per_request": self.spec_decode_metadata.rejected_draft_tokens_per_request,
                "committed_tokens_per_request": self.spec_decode_metadata.committed_tokens_per_request,
                "uses_lookahead_slots": self.spec_decode_metadata.uses_lookahead_slots,
            }
        return result

    def __str__(self) -> str:
        # Header information
        lines = []
        lines.append(f"batch id = {self._id}")
        lines.append(
            f"decode_attn_original_replica_id={self.decode_attn_original_replica_id}, "
            f"decode_attn_original_dp_id={self.decode_attn_original_dp_id}"
        )
        lines.append(f"num req = {self.size}, {self.request_ids}")
        lines.append("------------")

        # Per-request details
        for request in self._requests:
            lines.append(f"req_id: {request.id}")
            lines.append(f"num_prefill_tokens={request.num_prefill_tokens}")
            lines.append(f"num_decode_tokens={request.num_decode_tokens}")
            lines.append(f"num_processed_tokens={request.num_processed_tokens}")
            lines.append(f"current_decode_token_index={request.current_decode_token_index}")
            lines.append(f"completed_layer_count={request.completed_layer_count}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()



class EPBatchGroup(Batch):
    """
    EPGroupingBatch is a logical construct used in decode-ffn clusters to represent
    a collection of requests that are grouped together for processing by a specific
    Expert Parallel (EP) replica. It encapsulates the necessary information for
    efficient routing and processing within the MoE architecture.

    Key Attributes:
    - replica_id: The ID of the EP replica responsible for processing this batch.
    - ep_id: The ID of the Expert Parallel (EP) replica to which this batch is assigned.
    - time: The timestamp associated with this batch group.
    - source_batch_ids: List of source batch IDs that contribute to this group.
    - per_expert_tokens: Dictionary mapping expert IDs to the number of tokens assigned to them.
    """

    def __init__(
        self,
        requests: List[Request],
        num_tokens: List[int],
        replica_id: int,
        ep_id: int,
        time: float,
        source_batch_ids: List[int],
        per_expert_tokens: Dict[int, int],
        cluster_type: "ClusterType",
        is_moe: bool,
    ) -> None:
        super().__init__(replica_id, requests, num_tokens, is_moe=is_moe)
        self._ep_id = ep_id
        self._time = time
        self._source_batch_ids = source_batch_ids
        self._per_expert_tokens = per_expert_tokens
        self._cluster_type = cluster_type
        # Store the actual execution time (expert computation time) for metrics recording
        # This is set by ReplicaStageScheduleEvent after predict_and_create_stage()
        self._execution_time: float = 0.0
        self.activation_bytes: int = 0
    
    # to avoid per-request state mutation
    def on_schedule(
        self,
        time: float,
        cluster_type: "ClusterType" = None,
    ) -> None:
        """Override scheduling for EPBatchGroup to avoid per-request state mutation.

        In DECODE_FFN (intermediate) stage, we must not update per-request global state.
        Therefore, record only batch-level scheduling time without invoking
        Request.on_batch_schedule() for contained requests.
        """
        assert time >= 0, "Invalid scheduling time for EPBatchGroup"
        self._scheduled_at = time
        self._scheduled = True

    @property
    def ep_id(self) -> int:
        return self._ep_id

    @property
    def time(self) -> float:
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        self._time = value

    @property
    def source_batch_ids(self) -> List[int]:
        return self._source_batch_ids

    @property
    def per_expert_tokens(self) -> Dict[int, int]:
        return self._per_expert_tokens
    
    @property
    def cluster_type(self) -> "ClusterType":
        return self._cluster_type

    @property
    def execution_time(self) -> float:
        """Get the actual execution time (expert computation time) for this EP batch."""
        return self._execution_time

    @execution_time.setter
    def execution_time(self, value: float) -> None:
        """Set the actual execution time (expert computation time) for this EP batch."""
        self._execution_time = value
