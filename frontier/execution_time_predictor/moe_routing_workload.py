"""Expert-load distribution and the per-lane workload it produces.

Routing decides how many of a batch's tokens each expert receives.  These
methods draw that distribution, turn it into the per-lane token counts an
expert-parallel domain sees, and admit the aggregate a lane may process.
"""

import json

from frontier.config import ReplicaConfig
from frontier.entities import Batch
from frontier.execution_time_predictor.moe_predictor_helpers import (
    _normalize_routing_details_for_trace,
)
from frontier.logger import init_logger
from frontier.moe_ep_workload import (
    EPLaneWorkload,
    LayerEPWorkload,
    build_contiguous_expert_ownership,
    generate_moe_routing_ratios,
    materialize_layer_ep_workload,
    resolve_ep_lane_workload,
    resolve_routing_details,
)
from frontier.moe_routing_runtime import resolve_moe_gating_routing_runtime_path
from frontier.types import ClusterType
from typing import Dict, List, Mapping, Optional


logger = init_logger(__name__)


class MoeRoutingWorkload:
    """Expert-load distribution and per-lane routed workload."""

    @staticmethod

    def _emit_routing_details_snapshot(
        cluster_type: ClusterType,
        routing_details: Mapping[int, Mapping[int, Mapping[int, float]]],
    ) -> None:
        """Emit the exact predictor-owned routing map for external validation."""

        normalized = _normalize_routing_details_for_trace(routing_details)
        payload = {
            "schema_version": 1,
            "cluster": cluster_type.name,
            "routing_details": normalized,
        }
        logger.info(
            "[ROUTING-SNAPSHOT] %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def _get_requested_moe_gating_routing_runtime_path(self) -> str:
        return resolve_moe_gating_routing_runtime_path(
            getattr(self, "_moe_routing_distribution_type", "balanced")
        )

    @staticmethod

    def _get_ep_lane_routed_token_count(
        batch: Batch,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> Optional[int]:
        """Return an EP lane's routed-token count, or ``None`` for full batches.

        Dummy mode still models the same five-phase EP contract as the
        profiling-backed path.  The lane-local routed compute therefore has
        to depend on the materialized expert map even when the other dummy
        components use fixed structural timings.
        """

        if lane_workload is None:
            lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is None:
            return None
        return lane_workload.routed_token_count

    def _admit_routed_ep_aggregate(
        self,
        batch: Batch,
        *,
        routed_moe: bool,
        ep_size: Optional[int] = None,
        router_topk: Optional[int] = None,
        lane_workload: Optional[EPLaneWorkload] = None,
        conservation_context: str = "routed MoE admission",
    ) -> Optional[EPLaneWorkload]:
        """Admit a concrete routed MoE call at the public predictor boundary.

        Concrete predictors own the semantic classification of a call.  Once
        that classification is routed MoE, an EP>1 call must identify one
        physical lane before any mode-specific timing or lookup work begins.
        The helper also validates the descriptor against the active predictor
        topology and the source/lane token ledger. Workload construction
        remains owned by the scheduler/materializer path.
        """
        if type(routed_moe) is not bool:
            raise ValueError("routed_moe must be a bool")
        if not routed_moe:
            return None

        if ep_size is None:
            configured_ep_size = getattr(self, "_moe_ep_size", None)
            if configured_ep_size is None:
                replica_config = getattr(self, "_replica_config", None)
                configured_ep_size = getattr(
                    replica_config,
                    "moe_expert_parallel_size",
                    None,
                )
        else:
            configured_ep_size = ep_size
        if type(configured_ep_size) is not int or configured_ep_size < 1:
            raise ValueError(
                "routed MoE admission requires a positive integer EP size, got "
                f"{configured_ep_size!r}"
            )

        batch_lane_workload = resolve_ep_lane_workload(batch, required=False)
        explicit_lane_workload = (
            resolve_ep_lane_workload(lane_workload, required=True)
            if lane_workload is not None
            else None
        )
        if (
            batch_lane_workload is not None
            and explicit_lane_workload is not None
            and batch_lane_workload != explicit_lane_workload
        ):
            raise ValueError(
                "batch and lane_workload must refer to the same "
                "EPLaneWorkload descriptor"
            )
        resolved_lane_workload = explicit_lane_workload or batch_lane_workload
        if configured_ep_size > 1 and resolved_lane_workload is None:
            raise ValueError(
                "Routed MoE prediction with EP>1 requires an EPLaneWorkload "
                "descriptor"
            )
        if resolved_lane_workload is None:
            return None

        configured_router_topk = (
            getattr(self, "_router_topk", None)
            if router_topk is None
            else router_topk
        )
        if configured_router_topk is not None:
            if (
                type(configured_router_topk) is not int
                or configured_router_topk < 1
            ):
                raise ValueError(
                    "routed MoE admission requires a positive integer router top-k, "
                    f"got {configured_router_topk!r}"
                )
            if resolved_lane_workload.router_topk != configured_router_topk:
                raise ValueError(
                    "lane_workload router_topk does not match predictor topology: "
                    f"descriptor={resolved_lane_workload.router_topk}, "
                    f"predictor={configured_router_topk}"
                )
        else:
            configured_router_topk = resolved_lane_workload.router_topk

        if (
            resolved_lane_workload.moe_expert_parallel_size
            != configured_ep_size
        ):
            raise ValueError(
                "lane_workload EP size does not match predictor topology: "
                f"descriptor={resolved_lane_workload.moe_expert_parallel_size}, "
                f"predictor={configured_ep_size}"
            )

        # A descriptor attached to an EP lane entity already contains routed
        # assignments, so its count must match that entity's physical width.
        # An explicit descriptor paired with an ordinary source batch represents
        # one assignment subset of the aggregate.  The aggregate materializer,
        # rather than this predictor boundary, owns its conservation ledger.
        source_total_num_tokens = getattr(batch, "total_num_tokens", None)
        if source_total_num_tokens is not None:
            if (
                type(source_total_num_tokens) is not int
                or source_total_num_tokens < 0
            ):
                raise ValueError(
                    "routed MoE admission requires batch.total_num_tokens to be "
                    "a non-negative integer, got "
                    f"{source_total_num_tokens!r}"
                )
            if batch_lane_workload is not None:
                expected_routed_token_count = source_total_num_tokens
                if (
                    resolved_lane_workload.routed_token_count
                    != expected_routed_token_count
                ):
                    raise ValueError(
                        f"Token conservation violated in {conservation_context}: "
                        f"allocated {resolved_lane_workload.routed_token_count}, "
                        f"expected {expected_routed_token_count}"
                    )

        return resolved_lane_workload

    def _init_global_routing_allocations(self) -> Dict[int, Dict[int, float]]:
        """Pre-compute global expert allocation ratios for shared-domain EP sync.

        Monolithic decode with EP enabled needs a global view across all experts to
        derive per-lane post-MoE arrival skew before the shared-domain all-reduce.
        """
        total_experts = self._replica_config.total_expert_num
        if self._cluster_type == ClusterType.DECODE_ATTN or not self._model_config.is_moe:
            return {}
        num_layers = self._model_config.num_layers

        if type(total_experts) is not int or total_experts <= 0:
            raise ValueError(
                "total_expert_num must be an exact positive int for routing; "
                f"got {total_experts!r}"
            )
        if type(self._moe_ep_size) is not int or self._moe_ep_size <= 0:
            raise ValueError(
                "moe_expert_parallel_size must be an exact positive int for routing; "
                f"got {self._moe_ep_size!r}"
            )
        if total_experts % self._moe_ep_size != 0:
            raise ValueError(
                "total_expert_num must be divisible by moe_expert_parallel_size; "
                f"got total_expert_num={total_experts}, "
                f"moe_expert_parallel_size={self._moe_ep_size}"
            )

        distribution_type = self._moe_routing_distribution_type
        allocations: Dict[int, Dict[int, float]] = {}
        for layer_id in range(num_layers):
            allocations[layer_id] = generate_moe_routing_ratios(
                total_expert_num=total_experts,
                distribution_type=distribution_type,
                seed=self._moe_routing_seed,
                layer_id=layer_id,
            )

        return allocations

    def _build_shared_routing_details(
        self,
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        """Expose one immutable-shape routing source for monolithic schedulers.

        ``_global_routing_allocations`` is generated once per model layer and is
        intentionally replica-independent.  The scheduler, however, performs an
        exact ``(replica_id, global_layer_id)`` lookup.  Materialize that lookup
        shape here without generating a second distribution or assigning tokens
        to requests.  Integer token accounting and EP ownership splitting remain
        the responsibility of the shared per-layer materializer.

        The current monolithic predictor is constructed from ``ReplicaConfig``
        rather than ``ClusterConfig``.  The canonical cluster capacity is
        bound to ``ReplicaConfig.cluster_num_replicas`` before this method is
        called.  When the simulator supplies ``_actual_replica_ids``, those
        process-global IDs are used as the outer map keys; otherwise local
        ``range(replica_count)`` keys support standalone predictor construction.
        A missing capacity is an invalid topology, not a condition to infer
        from an attention-DP field.
        """
        replica_count = self._replica_config.cluster_num_replicas
        if type(replica_count) is not int or replica_count <= 0:
            raise ValueError(
                "A positive cluster replica count is required to build shared "
                f"routing details; got {replica_count!r}"
            )

        actual_replica_ids = self._actual_replica_ids
        if actual_replica_ids is None:
            replica_ids = list(range(replica_count))
        else:
            if not isinstance(actual_replica_ids, (list, tuple)):
                raise ValueError(
                    "actual_replica_ids must be a list or tuple when provided"
                )
            replica_ids = list(actual_replica_ids)
            if len(replica_ids) != replica_count:
                raise ValueError(
                    "actual_replica_ids length must match cluster replica count; "
                    f"got {len(replica_ids)} for {replica_count} replicas"
                )
            if any(
                type(replica_id) is not int or replica_id < 0
                for replica_id in replica_ids
            ):
                raise ValueError(
                    "actual_replica_ids must contain exact non-negative integers"
                )
            if len(set(replica_ids)) != len(replica_ids):
                raise ValueError("actual_replica_ids must be unique")

        return {
            replica_id: {
                layer_id: dict(expert_ratios)
                for layer_id, expert_ratios in self._global_routing_allocations.items()
            }
            for replica_id in replica_ids
        }

    def _get_routing_details_for_cluster(self, cluster_type: ClusterType):
        """Return the exact pre-generated routing map for one serving role."""
        attribute_by_cluster = {
            ClusterType.MONOLITHIC: "_monolithic_routing_details",
            ClusterType.PREFILL: "_prefill_routing_details",
            ClusterType.DECODE: "_decode_routing_details",
            ClusterType.DECODE_FFN: "_decode_ffn_routing_details",
        }
        attribute_name = attribute_by_cluster.get(cluster_type)
        if attribute_name is None:
            raise ValueError(
                f"MoE routing materialization does not support cluster_type={cluster_type}"
            )
        routing_details = getattr(self, attribute_name, None)
        if routing_details is None:
            raise ValueError(
                f"Missing pre-generated routing details for cluster_type={cluster_type}"
            )
        return routing_details

    def _get_cluster_replica_config(self, cluster_type: ClusterType) -> ReplicaConfig:
        """Return the serving replica; disaggregated predictors override by role."""
        return self._replica_config

    def _materialize_layer_ep_workload(
        self, batch: Batch, cluster_type: ClusterType, layer_id: int
    ) -> LayerEPWorkload:
        """Materialize one exact Replica-local EP workload for a MoE layer."""
        # Routing tables are built once by the constructor and have no runtime
        # mutation API. Topology and exact replica/layer/token identity are in
        # the key; the frozen workload can be shared across repeated EP waves.
        workload_cache = self._layer_workload_cache
        cache_capacity = self._layer_workload_cache_capacity
        cluster_replica_config = self._get_cluster_replica_config(cluster_type)
        routing_details = self._get_routing_details_for_cluster(cluster_type)
        target_replica_id = int(batch.replica_id)
        global_layer_id = int(layer_id)
        routing_token_count = int(batch.total_num_tokens)
        router_topk = int(cluster_replica_config.router_topk)
        total_expert_num = int(cluster_replica_config.total_expert_num)
        moe_ep_size = int(cluster_replica_config.moe_expert_parallel_size)
        cache_key = (
            cluster_type,
            target_replica_id,
            global_layer_id,
            routing_token_count,
            router_topk,
            total_expert_num,
            moe_ep_size,
        )
        cached_workload = workload_cache.get(cache_key)
        if cached_workload is not None:
            workload_cache.move_to_end(cache_key)
            return cached_workload
        workload = materialize_layer_ep_workload(
            routing_ratios=resolve_routing_details(
                routing_details,
                target_replica_id=target_replica_id,
                global_layer_id=global_layer_id,
            ),
            target_replica_id=target_replica_id,
            global_layer_id=global_layer_id,
            routing_token_count=routing_token_count,
            router_topk=router_topk,
            total_expert_num=total_expert_num,
            moe_expert_parallel_size=moe_ep_size,
            expert_to_ep=build_contiguous_expert_ownership(
                total_expert_num,
                moe_ep_size,
            ),
        )
        workload_cache[cache_key] = workload
        workload_cache.move_to_end(cache_key)
        while len(workload_cache) > cache_capacity:
            workload_cache.popitem(last=False)
        return workload

    def _resolve_layer_lane_workload(
        self,
        batch: Batch,
        *,
        cluster_type: ClusterType,
        layer_id: int,
    ) -> EPLaneWorkload:
        """Resolve one physical lane descriptor at a predictor boundary.

        Scheduler-created lane entities already carry the descriptor.  A
        regular batch may be materialized into one lane only for EP=1; an EP>1
        aggregate must be expanded by the scheduler's lane wave first so the
        predictor never guesses which local expert domain a global map denotes.
        """

        lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is not None:
            return lane_workload

        layer_workload = self._materialize_layer_ep_workload(
            batch=batch,
            cluster_type=cluster_type,
            layer_id=layer_id,
        )
        participant_ep_ids = tuple(layer_workload.participant_ep_ids)
        if len(participant_ep_ids) != int(self._moe_ep_size):
            raise ValueError(
                "materialized EP lane count does not match predictor topology: "
                f"descriptors={len(participant_ep_ids)}, predictor={self._moe_ep_size}"
            )
        if len(participant_ep_ids) != 1:
            raise ValueError(
                "regular aggregate MoE prediction requires a physical EP lane "
                "for EP>1; scheduler lane materialization is required"
            )
        return layer_workload.lane(participant_ep_ids[0])

    def _resolve_shared_domain_lane_workloads(
        self,
        batch: Batch,
        *,
        cluster_type: ClusterType,
        layer_id: int,
    ) -> tuple[EPLaneWorkload, ...]:
        """Resolve every physical lane for a shared-domain MoE timing probe."""

        lane_count = int(self._moe_ep_size)
        if lane_count <= 0:
            raise ValueError(
                "shared-domain MoE lane resolution requires a positive EP size, "
                f"got {lane_count}"
            )

        lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is not None:
            if lane_workload.moe_expert_parallel_size != lane_count:
                raise ValueError(
                    "batch lane workload EP size does not match predictor: "
                    f"descriptor={lane_workload.moe_expert_parallel_size}, "
                    f"predictor={lane_count}"
                )
            lane_workloads = (lane_workload,)
        else:
            workload = self._materialize_layer_ep_workload(
                batch=batch,
                cluster_type=cluster_type,
                layer_id=layer_id,
            )
            lane_workloads = tuple(
                workload.lane(ep_id) for ep_id in workload.participant_ep_ids
            )

        if len(lane_workloads) != lane_count:
            raise ValueError(
                "materialized EP lane count does not match predictor topology: "
                f"descriptors={len(lane_workloads)}, predictor={lane_count}"
            )
        return lane_workloads

    def _build_moe_load_imbalance_features(
        self,
        lane_workload: EPLaneWorkload,
        *,
        batch: Optional[Batch] = None,
    ) -> Dict[str, float]:
        lane_workload = resolve_ep_lane_workload(lane_workload, required=True)
        assert lane_workload is not None

        from frontier.moe_load_imbalance import MoELoadImbalanceInput

        expert_token_counts = [int(v) for v in lane_workload.local_token_counts]

        total_routed_tokens = int(sum(expert_token_counts))
        if lane_workload.router_topk <= 0:
            raise ValueError(f"Invalid router_topk={lane_workload.router_topk}")

        source_num_tokens = self._get_moe_pre_routing_token_count(batch)

        load_input = MoELoadImbalanceInput(
            num_tokens=source_num_tokens,
            num_experts_per_device=lane_workload.local_expert_width,
            hidden_dim=int(self._model_config.embedding_dim),
            expert_hidden_dim=int(self._model_config.mlp_hidden_dim),
            router_topk=int(lane_workload.router_topk),
            expert_token_counts=expert_token_counts,
            load_distribution="runtime",
        )
        features = load_input.to_features_dict()
        features.pop("load_distribution", None)
        features.pop("seed", None)
        missing_features = [
            name
            for name in self.MOE_LOAD_IMBALANCE_FEATURES
            if name not in features
        ]
        if missing_features:
            raise ValueError(
                "MoE load-imbalance feature construction is missing canonical "
                f"features: {missing_features}"
            )
        unexpected_features = sorted(
            set(features) - set(self.MOE_LOAD_IMBALANCE_FEATURES)
        )
        if unexpected_features:
            raise ValueError(
                "MoE load-imbalance feature construction produced unexpected "
                f"features: {unexpected_features}"
            )
        return {
            name: features[name]
            for name in self.MOE_LOAD_IMBALANCE_FEATURES
        }

    def _simulate_routing_per_layer(
        self, batches: List[Batch], stage_id: int
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        """
        Simulate routing for each layer in the stage.
        Returns: {layer_id: {replica_id: {moe_component: time_value}}}
        """
        del stage_id
        cluster_type = getattr(self, "_cluster_type", None)
        if not isinstance(cluster_type, ClusterType):
            raise ValueError(
                "layer routing prediction requires an initialized cluster_type"
            )

        # Routing materialization is stage-local and follows the canonical
        # aggregate-to-lane seam.  Predictor consumers receive only physical
        # lane descriptors, even when this legacy helper returns one result per
        # source replica.
        num_layers = self._num_layers_per_pipeline_stage
        layer_routing_results = {}

        for layer_id in range(num_layers):
            layer_routing_results[layer_id] = {}

            for batch in batches:
                replica_id = int(batch.replica_id)
                layer_workload = self._materialize_layer_ep_workload(
                    batch=batch,
                    cluster_type=cluster_type,
                    layer_id=layer_id,
                )
                lane_workloads = tuple(
                    layer_workload.lane(ep_id)
                    for ep_id in layer_workload.participant_ep_ids
                )
                if not lane_workloads:
                    raise ValueError(
                        "layer routing materialization produced no EP lanes: "
                        f"replica_id={replica_id}, layer_id={layer_id}"
                    )
                grouped_gemm_time = max(
                    self._get_grouped_gemm_time(lane_workload, batch=batch)
                    for lane_workload in lane_workloads
                )
                shuffling_time = max(
                    self._get_moe_shuffling_time(
                        batch,
                        moe_tokens_input=lane_workload,
                    )
                    for lane_workload in lane_workloads
                )
                communication_time = max(
                    self._get_expert_parallel_communication_time(
                        batch,
                        lane_workload=lane_workload,
                    )
                    for lane_workload in lane_workloads
                )
                layer_routing_results[layer_id][replica_id] = {
                    "moe_grouped_gemm_time": grouped_gemm_time,
                    "expert_parallel_communication_time": communication_time,
                    "moe_gating_time": self._get_gating_time(batch),
                    "moe_shuffling_time": shuffling_time,
                }

        return layer_routing_results
