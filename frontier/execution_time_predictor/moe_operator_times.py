"""Predicted time of each MoE operator in a layer.

Gating, routing top-k, shuffling, the grouped expert GEMM and the
expert-parallel collective, plus the token-count resolution each of them
needs and the model selection that decides which trained estimator answers.
"""

from frontier.config import get_quantization_manager
from frontier.entities import Batch, ExecutionTime
from frontier.entities.time_components import MoETime
from frontier.execution_time_predictor.moe_predictor_helpers import (
    _MOE_GATING_OPERATOR_NAMES,
)
from frontier.logger import init_logger
from frontier.moe_ep_workload import EPLaneWorkload, resolve_ep_lane_workload
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
    get_moe_gating_prediction_model_name,
    should_use_prefill_hot_moe_gating_context,
)
from frontier.operators.families import (
    MOE_FAMILY,
    get_comm_operator,
    is_moe_operator_ep_agnostic,
    resolve_moe_operator_tp_key,
)
from frontier.types import ClusterType
from typing import Dict, Mapping, Optional


logger = init_logger(__name__)


class MoeOperatorTimes:
    """Per-operator MoE time prediction and its inputs."""

    @staticmethod

    def _get_dummy_shared_domain_moe_scope_time(
        execution_time: ExecutionTime,
    ) -> float:
        """Return the fixed per-operator MoE scope used by shared-domain decode.

        The generic dummy ``ExecutionTime`` keeps the deprecated aggregate
        ``moe_gating_time`` contract by splitting that baseline across the two
        structured gating fields.  The shared-domain decode contract models
        each named gating operator as one fixed structural slot, matching its
        historical five-operator scope.  Resolve that compatibility at this
        boundary from the MoE family registry; all other operators retain the
        descriptor-aware structured timing, including zero-lane routed work.
        """

        moe_time = execution_time.moe_or_mlp_time_component
        if not isinstance(moe_time, MoETime):
            raise ValueError(
                "shared-domain dummy timing requires a MoE execution component"
            )
        operator_times = moe_time.operator_times
        if operator_times is None:
            raise ValueError(
                "shared-domain dummy timing requires structured MoE operator times"
            )

        scope_time = 0.0
        for operator_name, operator_time in operator_times.op_times.items():
            if operator_name in _MOE_GATING_OPERATOR_NAMES:
                # ``moe_gating_time`` is the one fixed dummy baseline for each
                # named gating operator; the structured fields store its
                # compatibility split as 0.5 * baseline each.
                scope_time += float(moe_time.moe_gating_time)
            else:
                scope_time += float(operator_time)
        return scope_time

    def predict_monolithic_decode_shared_domain_lane_moe_times_ms(
        self,
        batch: Batch,
        layer_id: int,
    ) -> Dict[int, float]:
        """Estimate per-EP-lane pre-collective MoE time for monolithic pure decode.

        Returns per-lane post-attention MoE compute in milliseconds. The result is
        used by the MONOLITHIC decode sync path to model shared-domain readiness skew
        before `expert_parallel_allreduce`.
        """
        if self._enable_dummy_mode:
            lane_workloads = self._resolve_shared_domain_lane_workloads(
                batch,
                cluster_type=ClusterType.MONOLITHIC,
                layer_id=layer_id,
            )
            lane_times_ms: Dict[int, float] = {}
            for lane_workload in lane_workloads:
                # This helper returns only the per-layer MoE scope.  The dummy
                # execution seam needs a stage value for its complete object,
                # but no stage-boundary term is included in the component total.
                execution_time = self._get_dummy_execution_time(
                    batch,
                    pipeline_stage=0,
                    include_attention=False,
                    lane_workload=lane_workload,
                )
                lane_times_ms[lane_workload.ep_id] = (
                    self._get_dummy_shared_domain_moe_scope_time(execution_time)
                )
            return lane_times_ms

        lane_workloads = self._resolve_shared_domain_lane_workloads(
            batch,
            cluster_type=ClusterType.MONOLITHIC,
            layer_id=layer_id,
        )

        post_attention_layernorm_time = self._get_mlp_norm_layer_act_execution_time(batch)
        gating_linear_time = self._get_gating_linear_time(batch)
        gating_routing_topk_time = self._get_gating_routing_topk_time(batch)
        share_expert_total_time = 0.0
        if self._model_config.supports_share_expert():
            share_expert_total_time = (
                self._get_share_expert_up_proj_execution_time(batch)
                + self._get_share_expert_down_proj_execution_time(batch)
                + self._get_share_expert_act_execution_time(batch)
            )

        lane_times_ms: Dict[int, float] = {}
        for lane_workload in lane_workloads:
            lane_id = lane_workload.ep_id
            shuffling_time = self._get_moe_shuffling_time(
                batch,
                moe_tokens_input=lane_workload,
            )
            grouped_gemm_time = self._get_grouped_gemm_time(
                lane_workload,
                batch=batch,
            )

            lane_times_ms[lane_id] = (
                post_attention_layernorm_time
                + gating_linear_time
                + gating_routing_topk_time
                + shuffling_time
                + grouped_gemm_time
                + share_expert_total_time
            )

        return lane_times_ms

    @staticmethod

    def _get_moe_op_tp_key(
        op_name: str,
        moe_tp_size: int,
        cluster_type: ClusterType | None = None,
    ) -> int:
        try:
            return resolve_moe_operator_tp_key(
                op_name,
                moe_tp_size=moe_tp_size,
                cluster_type=cluster_type,
                family=MOE_FAMILY,
            )
        except ValueError as exc:
            if str(exc).startswith("Unsupported MoE op:"):
                raise ValueError(
                    f"Unsupported MoE op for TP mapping: {op_name}"
                ) from exc
            raise

    @staticmethod

    def _is_moe_op_ep_agnostic(op_name: str) -> bool:
        try:
            return is_moe_operator_ep_agnostic(op_name, family=MOE_FAMILY)
        except ValueError as exc:
            if str(exc).startswith("Unsupported MoE op:"):
                raise ValueError(
                    f"Unsupported MoE op for EP mapping: {op_name}"
                ) from exc
            raise

    def _select_moe_gating_prediction_model_name(
        self,
        base_model_name: str,
        batch: Batch,
    ) -> str:
        requested_context = DEFAULT_MOE_GATING_RUNTIME_CONTEXT
        if should_use_prefill_hot_moe_gating_context(
            model_config=self._model_config,
            batch=batch,
        ):
            requested_context = PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT
        candidate_model_name = get_moe_gating_prediction_model_name(
            base_model_name,
            requested_context=requested_context,
        )
        if candidate_model_name in self._predictions:
            return candidate_model_name
        return base_model_name

    def _use_expert_parallel_alltoall_path(self, batch: Batch) -> bool:
        moe_ep_size = int(getattr(self, "_moe_ep_size", 1))
        if moe_ep_size <= 1:
            return False
        # EP is replica-local and is independent of the retired attention-DP
        # lane concept.  A full batch on any MoE serving role therefore uses
        # the EP communication/accounting path whenever EP>1.
        return True

    def _predict_expert_parallel_phase_operator_times(
        self,
        batch: Batch,
        *,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> dict[str, float]:
        """Predict exact dispatch and combine collectives for one MoE layer."""

        if self._moe_ep_size <= 1:
            return {
                "expert_parallel_alltoall_dispatch": 0.0,
                "expert_parallel_alltoall_combine": 0.0,
            }
        if not self._use_expert_parallel_alltoall_path(batch):
            raise ValueError(
                "Canonical MoE EP execution requires named all-to-all dispatch "
                "and combine phases"
            )
        return {
            op_name: self._predict_comm_operator(
                get_comm_operator(op_name),
                batch,
                lane_workload=lane_workload,
            )
            for op_name in (
                "expert_parallel_alltoall_dispatch",
                "expert_parallel_alltoall_combine",
            )
        }

    def _get_effective_moe_total_tokens(self, batch: Batch) -> int:
        effective_tokens = int(
            batch.get_effective_total_tokens_rounded(self._cluster_type)
        )
        if effective_tokens < 0:
            raise ValueError(
                f"effective MoE tokens must be non-negative, got {effective_tokens}"
            )
        return effective_tokens

    def _get_moe_pre_routing_token_count(self, batch: Optional[Batch]) -> int:
        """Return the source-batch width used by pre-routing MoE models.

        A physical EP lane carries only an assignment subset, so its routed
        count cannot identify the source width.  Callers that need the
        one-feature profiling domain must provide the source batch explicitly.
        """

        if batch is None:
            raise ValueError(
                "MoE pre-routing token lookup requires the source batch; "
                "an EPLaneWorkload cannot supply that width"
            )
        return self._get_effective_moe_total_tokens(batch)

    def _get_local_ep_routed_tokens(
        self,
        batch: Batch,
        *,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> int:
        source = batch if lane_workload is None else lane_workload
        resolved_lane_workload = resolve_ep_lane_workload(source, required=True)
        assert resolved_lane_workload is not None
        return resolved_lane_workload.routed_token_count

    def _get_moe_tokens_input(
        self, batch: Batch, layer_id: int = 0
    ) -> EPLaneWorkload | int:
        """
        Unified entry point to get MoE tokens input for grouped GEMM prediction.

        EP lane batches carry the canonical physical descriptor.  A regular
        non-lane batch may use the scalar pre-routing token path for legacy
        one-feature models; load-aware models require an explicit descriptor.

        Args:
            batch: The batch being processed
            layer_id: The layer ID for which to get token allocation (default 0)

        Returns:
            - In load-imbalance mode: ``EPLaneWorkload``
            - In single-token-count profiling mode: pre-routing token count

        Raises:
            ValueError: If the selected routing mode is not supported by the active predictor
        """
        lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is not None:
            if lane_workload.router_topk != int(self._router_topk):
                raise ValueError(
                    "EPLaneWorkload router_topk does not match predictor topology: "
                    f"descriptor={lane_workload.router_topk}, predictor={self._router_topk}"
                )
            return lane_workload

        load_aware = any(
            isinstance(prediction, dict)
            and prediction.get("_on_demand_prediction", False)
            for prediction in (
                getattr(self, "_predictions", {}).get("moe_shuffling"),
                getattr(self, "_predictions", {}).get("moe_grouped_gemm"),
            )
        )
        if load_aware:
            cluster_type = getattr(self, "_cluster_type", None)
            if not isinstance(cluster_type, ClusterType):
                raise ValueError(
                    "load-aware MoE prediction requires an initialized cluster_type"
                )
            workload = self._materialize_layer_ep_workload(
                batch=batch,
                cluster_type=cluster_type,
                layer_id=layer_id,
            )
            if len(workload.participant_ep_ids) != int(self._moe_ep_size):
                raise ValueError(
                    "materialized EP lane count does not match predictor topology: "
                    f"descriptors={len(workload.participant_ep_ids)}, "
                    f"predictor={self._moe_ep_size}"
                )
            if int(self._moe_ep_size) != 1:
                raise ValueError(
                    "load-aware regular-batch prediction requires an explicit "
                    "physical EP lane for EP>1"
                )
            return workload.lane(0)
        return self._get_effective_moe_total_tokens(batch)

    def _get_gating_time(self, batch: Batch) -> float:
        """
        Get total MoE gating network execution time (linear + routing_topk).

        The gating network determines which experts each token should be routed to.
        Prediction is based on num_tokens feature from profiling data.

        Returns:
            Total gating time (sum of linear and routing_topk times)
        """
        return self._get_gating_linear_time(batch) + self._get_gating_routing_topk_time(
            batch
        )

    def _get_gating_linear_time(self, batch: Batch) -> float:
        """
        Get MoE gating linear layer execution time.

        The gating linear layer computes logits from hidden states (hidden_dim -> num_experts).
        """
        if not self._supports_operation("moe_gating_linear"):
            raise NotImplementedError(
                "MoE gating linear is not supported for cluster type"
            )
        model_name = self._select_moe_gating_prediction_model_name(
            "moe_gating_linear",
            batch,
        )
        if model_name not in self._predictions:
            raise NotImplementedError(
                "MoE gating linear is not supported for cluster type"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._get_prediction_for_features(
            model_name,
            {"num_tokens": effective_tokens},
            feature_names=("num_tokens",),
        )

    def _get_gating_routing_topk_time(self, batch: Batch) -> float:
        """
        Get MoE gating routing topk execution time.

        The routing topk operation selects top-K experts and applies softmax normalization.
        """
        if not self._supports_operation("moe_gating_routing_topk"):
            raise NotImplementedError(
                "MoE gating routing topk is not supported for cluster type"
            )
        model_name = self._select_moe_gating_prediction_model_name(
            "moe_gating_routing_topk",
            batch,
        )
        if model_name not in self._predictions:
            raise NotImplementedError(
                "MoE gating routing topk is not supported for cluster type"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._get_prediction_for_features(
            model_name,
            {"num_tokens": effective_tokens},
            feature_names=("num_tokens",),
        )

    def _resolve_shuffling_per_expert_tokens(
        self,
        batch: Batch,
        moe_tokens_input: Optional[EPLaneWorkload] = None,
    ) -> EPLaneWorkload:
        source = batch if moe_tokens_input is None else moe_tokens_input
        lane_workload = resolve_ep_lane_workload(source, required=True)
        assert lane_workload is not None
        return lane_workload

    def _get_moe_shuffling_time(
        self,
        batch: Batch,
        moe_tokens_input: Optional[EPLaneWorkload] = None,
    ) -> float:
        """
        Get MoE token shuffling execution time using trained prediction model.

        Shuffling involves dispatching tokens to assigned experts. When the model is
        trained with load-imbalance features, use on-demand prediction driven by
        per-expert allocation; otherwise use the legacy num_tokens lookup table.
        """
        if not self._supports_operation("moe_shuffling"):
            raise NotImplementedError("MoE shuffling is not supported for cluster type")
        if "moe_shuffling" not in self._predictions:
            raise NotImplementedError("MoE shuffling is not supported for cluster type")
        if moe_tokens_input is not None and not isinstance(
            moe_tokens_input, EPLaneWorkload
        ):
            raise TypeError(
                "MoE shuffling requires an EPLaneWorkload descriptor when an "
                "explicit workload is supplied"
            )

        prediction_cache = self._predictions["moe_shuffling"]
        if isinstance(prediction_cache, dict) and prediction_cache.get(
            "_on_demand_prediction", False
        ):
            lane_workload = self._resolve_shuffling_per_expert_tokens(
                batch,
                moe_tokens_input=moe_tokens_input,
            )
            if lane_workload.routed_token_count == 0:
                raw_time = 0.0
            else:
                features = self._build_moe_load_imbalance_features(
                    lane_workload,
                    batch=batch,
                )
                raw_time = self._get_on_demand_prediction(
                    "moe_shuffling", features
                )
        else:
            lane_workload = resolve_ep_lane_workload(batch, required=False)
            if moe_tokens_input is not None:
                lane_workload = resolve_ep_lane_workload(
                    moe_tokens_input,
                    required=True,
                )
            if lane_workload is not None:
                if lane_workload.routed_token_count == 0:
                    return 0.0
                effective_tokens = self._get_moe_pre_routing_token_count(batch)
            else:
                effective_tokens = batch.get_effective_total_tokens_rounded(
                    self._cluster_type
                )
            raw_time = self._get_prediction_for_features(
                "moe_shuffling",
                {"num_tokens": effective_tokens},
                feature_names=("num_tokens",),
            )

        return raw_time

    def _get_expert_parallel_communication_time(
        self,
        batch: Batch,
        *,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> float:
        """
        Get expert parallel communication time.

        Shared-domain MoE execution (monolithic / prefill / decode) uses
        expert-parallel all-reduce when EP is enabled without all-to-all routing.
        Post-routing EP batches (e.g. DECODE_FFN) and flattened multi-DP MoE
        paths keep the all-to-all communication model.
        """
        if self._moe_ep_size <= 1:
            return 0.0

        uses_alltoall = self._use_expert_parallel_alltoall_path(batch)
        resolved_lane_workload = None
        if uses_alltoall:
            resolved_lane_workload = resolve_ep_lane_workload(
                batch if lane_workload is None else lane_workload,
                required=True,
            )
            assert resolved_lane_workload is not None

        if self._cc_backend is not None:
            quant_manager = get_quantization_manager()

            if uses_alltoall:
                routed_tokens = self._get_local_ep_routed_tokens(
                    batch,
                    lane_workload=resolved_lane_workload,
                )
                data_size_bytes = self._model_config.embedding_dim * 2 * routed_tokens
                data_size_bytes = quant_manager.adjust_tensor_size(
                    "expert_parallel_communication", data_size_bytes, self._cluster_type
                )
                result = self._cc_backend.predict_all_to_all(
                    data_size_bytes=data_size_bytes,
                    num_devices=self._moe_ep_size,
                    cluster_type=self._cluster_type,
                    comm_domain="EP",
                )
                logger.debug(
                    f"_get_expert_parallel_communication_time: using EP all-to-all, "
                    f"data_size={data_size_bytes}, num_devices={self._moe_ep_size}, "
                    f"result={result:.6f} ms"
                )
                return result

            effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
            data_size_bytes = self._model_config.embedding_dim * 2 * effective_tokens
            data_size_bytes = quant_manager.adjust_tensor_size(
                "allreduce", data_size_bytes, self._cluster_type
            )
            result = self._cc_backend.predict_allreduce(
                data_size_bytes=data_size_bytes,
                num_devices=self._moe_ep_size,
                cluster_type=self._cluster_type,
                comm_domain="EP",
            )
            result = self._strip_collective_sim_allreduce_launch_overhead_if_needed(
                batch=batch,
                predicted_ms=result,
                num_devices=self._moe_ep_size,
                comm_domain="EP",
            )
            logger.debug(
                f"_get_expert_parallel_communication_time: using EP all-reduce, "
                f"data_size={data_size_bytes}, num_devices={self._moe_ep_size}, "
                f"result={result:.6f} ms"
            )
            return result

        if self._enable_dummy_mode:
            logger.debug(
                f"_get_expert_parallel_communication_time: CC Backend not available, "
                f"using dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        raise RuntimeError(
            f"CC Backend is required for expert parallel communication prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def _get_grouped_gemm_time(
        self,
        num_tokens_or_allocation,
        batch: Optional[Batch] = None,
    ) -> float:
        """
        Calculate grouped GEMM time using trained prediction model.

        Args:
            num_tokens_or_allocation: An ``EPLaneWorkload`` for EP-aware
                                    prediction, or an integer for the legacy
                                    one-feature non-lane path.

        Returns:
            Total grouped GEMM execution time
        """
        if not self._supports_operation("moe_grouped_gemm"):
            raise NotImplementedError(
                "MoE grouped_gemm is not supported for cluster type"
            )

        if "moe_grouped_gemm" not in self._predictions:
            raise NotImplementedError(
                "MoE grouped_gemm is not supported for cluster type"
            )

        prediction_cache = self._predictions["moe_grouped_gemm"]

        if isinstance(num_tokens_or_allocation, Mapping):
            raise TypeError(
                "MoE grouped_gemm requires an EPLaneWorkload descriptor; raw "
                "expert-token maps are not a predictor workload contract"
            )
        lane_workload = (
            resolve_ep_lane_workload(num_tokens_or_allocation, required=True)
            if isinstance(num_tokens_or_allocation, EPLaneWorkload)
            else None
        )

        # Check if this model uses on-demand prediction (trained with load imbalance features)
        if isinstance(prediction_cache, dict) and prediction_cache.get(
            "_on_demand_prediction"
        ):
            # On-demand prediction mode: model was trained with load imbalance features.
            # We must provide the full feature set computed from per-expert token distribution.
            if lane_workload is None:
                raise ValueError(
                    "moe_grouped_gemm is in load-imbalance (on-demand) mode and "
                    "requires an EPLaneWorkload descriptor"
                )

            if lane_workload.routed_token_count == 0:
                return 0.0

            features = self._build_moe_load_imbalance_features(
                lane_workload,
                batch=batch,
            )
            return self._get_on_demand_prediction("moe_grouped_gemm", features)

        # Standard cache lookup mode (trained with num_tokens only)
        if lane_workload is not None:
            if lane_workload.routed_token_count == 0:
                return 0.0
            source_num_tokens = self._get_moe_pre_routing_token_count(batch)
            raw_time = self._get_prediction_for_features(
                "moe_grouped_gemm",
                {"num_tokens": source_num_tokens},
                feature_names=("num_tokens",),
            )
            return raw_time

        # Backward compatibility: single number of tokens
        num_tokens = num_tokens_or_allocation
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, (int, float)):
            raise TypeError(
                "MoE grouped_gemm requires an EPLaneWorkload descriptor or a "
                "numeric token count"
            )
        if num_tokens <= 0:
            return 0.0
        raw_time = self._get_prediction_for_features(
            "moe_grouped_gemm",
            {"num_tokens": num_tokens},
            feature_names=("num_tokens",),
        )
        return raw_time

    @staticmethod

    def _resolve_moe_execution_inputs(
        *,
        moe_tokens_input: object,
        lane_workload: Optional[EPLaneWorkload],
        include_moe: bool,
    ) -> tuple[object, Optional[EPLaneWorkload]]:
        """Resolve one canonical MoE input and its optional physical lane.

        ``moe_tokens_input`` is retained for the legacy scalar one-feature
        lookup, while ``lane_workload`` carries the physical routed domain.
        A physical call must use one descriptor for both roles; allowing a
        scalar or a second descriptor alongside it would let communication and
        routed compute describe different workloads.
        """

        if isinstance(moe_tokens_input, Mapping):
            raise TypeError(
                "moe_tokens_input cannot be a raw expert-token map; provide an "
                "EPLaneWorkload descriptor"
            )

        explicit_lane = (
            resolve_ep_lane_workload(lane_workload, required=True)
            if lane_workload is not None
            else None
        )
        input_lane = (
            resolve_ep_lane_workload(moe_tokens_input, required=True)
            if isinstance(moe_tokens_input, EPLaneWorkload)
            else None
        )

        if explicit_lane is not None:
            if input_lane is not None:
                if input_lane != explicit_lane:
                    raise ValueError(
                        "moe_tokens_input and lane_workload must refer to the "
                        "same EPLaneWorkload descriptor"
                    )
                return explicit_lane, explicit_lane
            if moe_tokens_input is not None:
                raise TypeError(
                    "cannot combine a scalar moe_tokens_input with an "
                    "explicit lane_workload"
                )
            return explicit_lane, explicit_lane

        if input_lane is not None:
            return input_lane, input_lane

        if include_moe and moe_tokens_input is None:
            raise ValueError(
                "moe_tokens_input is required when include_moe=True. "
                "Provide a scalar token count or an EPLaneWorkload descriptor."
            )
        return moe_tokens_input, None
