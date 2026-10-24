"""Expert-load distribution and the per-lane workload it produces.

Routing decides how many of a batch's tokens each expert receives.  These
methods draw that distribution, turn it into the per-lane token counts an
expert-parallel domain sees, and admit the aggregate a lane may process.
"""

import json
import math

from frontier.config import ReplicaConfig
from frontier.entities import Batch
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
from frontier.moe_load_imbalance import (
    MOE_GROUPED_GEMM_PADDED_FEATURES,
    MOE_LOAD_IMBALANCE_FEATURES,
    MoELoadImbalanceInput,
    block_size_m_for_tokens,
    moe_grouped_gemm_schema,
    num_tokens_post_padded,
)
from frontier.moe_routing_runtime import resolve_moe_gating_routing_runtime_path
from frontier.moe_routing_policy import WaveRoutingSource, resolve_routing_count_policy
from frontier.types import ClusterType
from typing import Dict, List, Mapping, Optional


logger = init_logger(__name__)


def validate_position_moe_source_row(row):
    """Compare retained producer census with the one position-count owner."""
    import pandas as pd
    from frontier.profiling.moe.inputs import routing_profile_cell
    from frontier.moe_routing_policy import position_cyclic_expert_counts, TokenInterval, source_lane_index_for_ep
    scope = row.moe_frame_scope
    cell = routing_profile_cell(dict(routing_input=json.loads(row.routing_input),
        moe_frame_scope=scope,source_lane_id=int(row.moe_source_lane_id),
        ep_id=None if pd.isna(row.moe_ep_rank) else int(row.moe_ep_rank)))
    frames, local = cell.routing_input.source_frames, cell.routing_input.source_frames[cell.source_lane_id]
    live = local.live_rows if scope == "source_local" else sum(frame.live_rows for frame in frames)
    physical = local.physical_rows if scope == "source_local" else cell.routing_input.gathered_physical_rows
    if (row.num_tokens != physical or row.live_routing_token_count != live
            or row.gating_runtime_context != local.gating_runtime_context
            or row.moe_input_id != cell.routing_input.moe_input_id):
        raise ValueError("MoE measured physical row differs from its supplied source descriptor")
    if scope == "dp_gathered":
        ep = int(row.expert_parallel_size)
        if cell.source_lane_id != source_lane_index_for_ep(cell.ep_id,attention_tp_size=ep//len(frames),
                source_lane_count=len(frames),participant_ep_ids=range(ep)):
            raise ValueError("MoE measured source/EP association differs from its descriptor")
        counts = position_cyclic_expert_counts((TokenInterval(c,q) for frame in frames
            for c,q in zip(frame.context_lens,frame.query_lens)),int(row.router_topk),int(row.num_experts))
        width = int(row.num_experts)//ep
        expected = [counts[i] for i in range(cell.ep_id*width,(cell.ep_id+1)*width)]
        if json.loads(row.expert_token_counts) != expected or row.total_routed_tokens != sum(expected):
            raise ValueError("MoE measured expert counts differ from the supplied position source")


def _normalize_routing_details_for_trace(
    routing_details: Mapping[int, Mapping[int, Mapping[int, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return a strict JSON-safe copy of runtime routing details.

    The matrix checker compares this emitted object with an independently
    materialized sidecar.  The trace must therefore contain the actual
    predictor-owned map, not a derived token allocation or a digest.
    """

    if not isinstance(routing_details, Mapping) or not routing_details:
        raise ValueError("routing_details trace payload must be a non-empty mapping")
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for replica_id, per_layer in routing_details.items():
        if type(replica_id) is not int or replica_id < 0:
            raise ValueError(
                "routing_details trace replica IDs must be non-negative integers"
            )
        if not isinstance(per_layer, Mapping) or not per_layer:
            raise ValueError(
                f"routing_details trace replica {replica_id} has no layer map"
            )
        normalized_layers: dict[str, dict[str, float]] = {}
        for layer_id, per_expert in per_layer.items():
            if type(layer_id) is not int or layer_id < 0:
                raise ValueError(
                    "routing_details trace layer IDs must be non-negative integers"
                )
            if not isinstance(per_expert, Mapping) or not per_expert:
                raise ValueError(
                    f"routing_details trace layer {layer_id} has no expert map"
                )
            normalized_experts: dict[str, float] = {}
            for expert_id, ratio in per_expert.items():
                if type(expert_id) is not int or expert_id < 0:
                    raise ValueError(
                        "routing_details trace expert IDs must be non-negative integers"
                    )
                value = float(ratio)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "routing_details trace ratios must be finite and non-negative"
                    )
                normalized_experts[str(expert_id)] = value
            ratio_sum = sum(normalized_experts.values())
            if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "routing_details trace ratios must sum to one "
                    f"for replica={replica_id} layer={layer_id}, got {ratio_sum}"
                )
            normalized_layers[str(layer_id)] = normalized_experts
        normalized[str(replica_id)] = normalized_layers
    return normalized


class MoeRoutingWorkload:
    """Expert-load distribution and per-lane routed workload."""

    def _step5_routed_layer_group(self, layer_id):
        """Select from complete group facts for this predictor's fixed model."""
        groups = getattr(self, "_step5_routed_layer_groups", None)
        if groups is None:
            from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
            groups = resolve_routed_layer_groups(self._model_config)
            self._step5_routed_layer_groups = groups
        return next(group for group in groups if layer_id in group.layer_ids)

    def _step5_model_points(self, model):
        """Index post-training rows; serving replaces tuples/estimators, not row dicts."""
        from frontier.profiling.moe.inputs import MoESourceFrame
        rows, identity = model._frontier_moe_physical_rows, model._frontier_measurement_identity
        retained = getattr(self, "_step5_physical_points", {})
        previous = retained.get(model)
        if previous is not None and previous[0] is rows and previous[1] == identity:
            return previous[2]
        points, frames = {}, {}
        for row in rows:
            text = row["routing_input"]
            if text not in frames:
                frames[text] = tuple(MoESourceFrame(**frame) for frame in json.loads(text)["source_frames"])
            key = (row["moe_ep_rank"], row["total_routed_tokens"], frames[text])
            points.setdefault(key, {})[row["native_context"]] = None
        retained[model] = (rows, identity, points)
        self._step5_physical_points = retained
        return points

    def _step5_collective_points(self):
        """Prepare structural receipts from this run's fixed collective input."""
        from frontier.profiling.common.runtime_identity import canonical_json
        retained = getattr(self, "_step5_collective_receipts", None)
        if retained is None:
            artifact, shapes = json.loads(self._cc_backend.collective_profile_bytes().decode("utf-8")), {}
            for key in artifact["counts"]:
                value = json.loads(key)
                shape = (value["phase"], tuple(value["sizes"]), value["graph_mode"], canonical_json(value["tensors"]))
                shapes.setdefault(shape, []).append((key, value))
            retained = (artifact, shapes, {}, {})
        else:
            retained = (*retained[:2], retained[2].copy(), retained[3].copy())
        return retained

    def _step5_plan_candidates(self, retained, phase, sizes, mode, tensors, context):
        from frontier.profiling.common.runtime_identity import canonical_json
        artifact, shapes, domains, plans = retained
        shape = (phase, sizes, mode, canonical_json(tensors))
        shared = canonical_json(context["native_facts"]["shared"])
        domain = (shape, shared)
        if domain not in domains:
            points = {}
            for key, value in shapes.get(shape, ()):
                measured = artifact["counts"][key]["context"]
                if measured["shared"] != context["native_facts"]["shared"] or not measured.get("execution_policy"):
                    continue
                buffer = json.loads(value["buffer_json"])
                points.setdefault(canonical_json(buffer.get("compute_context")), []).append((key, value, buffer))
            domains[domain] = points
        return artifact, plans, domains[domain].get(canonical_json(context), ())

    def _step5_position_queries(self, lane):
        """Recover actual point facts and complete measured plans, never numerical defaults."""
        from frontier.profiling.moe.inputs import MoESourceFrame
        from frontier.cc_backend.collective_plan import CollectivePlan, CollectiveTensor
        source = lane.routing_source
        group = self._step5_routed_layer_group(self._step5_layer_metadata.attention_spec.global_layer_id)
        family = self._measurement_family_name(self._active_measurement_type)
        models = self._model_manager._models_view_for_family(family, self._cluster_type,
            model_name="moe_grouped_gemm")
        frames = tuple(MoESourceFrame(s.lane_id,tuple(i.length for i in s.intervals),
            tuple(i.start for i in s.intervals),s.physical_rows,s.gating_runtime_context) for s in source.source_lanes)
        contexts = {}
        for model in models.values():
            if getattr(model, "_frontier_cache_model_name", None) != "moe_grouped_gemm":
                continue
            binding = self._model_manager._native_measurement_binding(
                self._model_manager._measurement_identity(model))
            if binding["selected_native_metadata"]["moe_layer_ids"] == group.row_metadata()["moe_layer_ids"]:
                contexts.update(self._step5_model_points(model).get((lane.ep_id, lane.routed_token_count, frames), {}))
        if len(contexts) != 1: raise ValueError("Position expert query lacks a unique actual source/native point receipt")
        context = json.loads(next(iter(contexts)))
        features = self._build_grouped_gemm_features(lane,native_context=context)
        local = source.source_lanes[lane.ep_id//self._replica_config.attn_tensor_parallel_size]
        from frontier.operators.families import get_moe_profiling_operators
        queries = {op.name:dict(num_tokens=local.physical_rows,live_routing_token_count=local.live_rows)
                   if op.ep_agnostic else features for op in get_moe_profiling_operators(context)}
        retained = self._step5_collective_points()
        plans = {}
        phases = {"step5_moe_dispatch":"dispatch","step5_moe_combine":"combine",
            "moe_tensor_parallel_allreduce":"routed_tp"}
        if self._model_config.supports_share_expert(): phases["share_expert_tensor_parallel_allreduce"] = "shared_tp"
        mode = "full" if local.graph_mode == "FULL" else "none"
        for name,phase in phases.items():
            matches = {}
            tensors = context["collective_tensors"]["dispatch" if phase == "dispatch" else "output"]
            artifact, prepared, candidates = self._step5_plan_candidates(retained,phase,
                tuple(s.physical_rows for s in source.source_lanes),mode,tensors,context)
            for key,value,buffer in candidates:
                receipt = artifact["counts"][key]
                if (value["phase"] != phase or tuple(value["sizes"]) != tuple(s.physical_rows for s in source.source_lanes)
                        or value["graph_mode"] != mode or value["tensors"] != tensors): continue
                measured = receipt["context"]
                if measured["shared"] != context["native_facts"]["shared"] or not measured.get("execution_policy"): continue
                from frontier.profiling.common.runtime_identity import canonical_json
                if canonical_json(buffer.get("compute_context")) != canonical_json(context): continue
                if receipt["buffer_receipt"] != buffer: raise ValueError("Measured collective buffer/count key mismatch")
                if key not in prepared:
                    plan = CollectivePlan(phase,tuple(value["sizes"]),tuple(CollectiveTensor(**t) for t in tensors),
                        tuple(tuple(g) for g in value["groups"]),mode,value["context_json"],value["buffer_json"])
                    decoded, rebuilt = json.loads(plan.context_json), plan.cache_key()
                    if decoded != measured or rebuilt != key: raise ValueError("Measured collective context/count key mismatch")
                    prepared[key] = (plan, decoded, rebuilt)
                plan, decoded, rebuilt = prepared[key]
                if decoded != measured or rebuilt != key: raise ValueError("Measured collective context/count key mismatch")
                matches[key] = plan
            if len(matches) != 1: raise ValueError("Position collective lacks unique measured policy/layout/buffer coverage")
            plans[name] = next(iter(matches.values()))
        self._step5_collective_receipts = retained
        return context,queries,plans

    def _emit_routing_details_snapshot(
        self,
        cluster_type: ClusterType,
        routing_details: Mapping[int, Mapping[int, Mapping[int, float]]],
    ) -> None:
        """Emit the exact predictor-owned routing map for external validation."""

        policy = resolve_routing_count_policy(getattr(self, "_moe_routing_distribution_type", "balanced"))
        if policy.requires_source:
            logger.info("[ROUTING-POLICY-SNAPSHOT] %s", json.dumps({
                "policy_id": self._moe_routing_distribution_type, "router_topk": self._replica_config.router_topk,
                "total_expert_num": self._replica_config.total_expert_num,
                "interval_contract": "(logical_position * top_k + slot) % num_experts",
            }, sort_keys=True))
            return
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
        if resolve_routing_count_policy(self._moe_routing_distribution_type).requires_source:
            return {}
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
        self, batch: Batch, cluster_type: ClusterType, layer_id: int, routing_source: WaveRoutingSource | None = None,
    ) -> LayerEPWorkload:
        """Materialize one exact Replica-local EP workload for a MoE layer."""
        # Routing tables are built once by the constructor and have no runtime
        # mutation API. Topology and exact replica/layer/token identity are in
        # the key; the frozen workload can be shared across repeated EP waves.
        policy_id = getattr(self, "_moe_routing_distribution_type", "balanced")
        policy = resolve_routing_count_policy(policy_id)
        if policy.requires_source and (
            not isinstance(routing_source, WaveRoutingSource) or routing_source.policy_id != policy_id
        ):
            raise ValueError("source-based prediction requires an explicitly matching routing_source")
        workload_cache = self._layer_workload_cache
        cache_capacity = self._layer_workload_cache_capacity
        cluster_replica_config = self._get_cluster_replica_config(cluster_type)
        routing_details = None if policy.requires_source else self._get_routing_details_for_cluster(cluster_type)
        target_replica_id = int(batch.replica_id)
        global_layer_id = int(layer_id)
        # Physical rows include padding even when source assignments are masked.
        routing_token_count = int(batch.get_effective_total_tokens_for_compute(cluster_type))
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
        if policy.requires_source:
            cache_key += (policy_id, routing_source)
        cached_workload = workload_cache.get(cache_key)
        if cached_workload is not None:
            workload_cache.move_to_end(cache_key)
            return cached_workload
        workload = materialize_layer_ep_workload(
            routing_ratios=None if policy.requires_source else resolve_routing_details(
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
            policy_id=policy_id,
            routing_source=routing_source,
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
            for name in MOE_LOAD_IMBALANCE_FEATURES
            if name not in features
        ]
        if missing_features:
            raise ValueError(
                "MoE load-imbalance feature construction is missing canonical "
                f"features: {missing_features}"
            )
        unexpected_features = sorted(
            set(features) - set(MOE_LOAD_IMBALANCE_FEATURES)
        )
        if unexpected_features:
            raise ValueError(
                "MoE load-imbalance feature construction produced unexpected "
                f"features: {unexpected_features}"
            )
        return {
            name: features[name]
            for name in MOE_LOAD_IMBALANCE_FEATURES
        }

    def _build_grouped_gemm_features(
        self,
        lane_workload: EPLaneWorkload,
        *,
        batch: Optional[Batch] = None,
        native_context=None,
    ) -> Dict[str, float]:
        """Features of the on-demand grouped-GEMM model for one EP lane.

        A model trained on a table with moe_align_block_size's padded count
        (decision T33-C2-F) takes the lane's routed token count and the padded
        count. Each rank pads every expert of the EP domain (decision T43-PAD)
        with the BLOCK_SIZE_M that vLLM's kernel config selects at the step's
        pre-routing token count, from the ranges the profiler recorded. Other
        models take the load-imbalance features.
        """
        if lane_workload.routing_source is not None:
            if native_context is None:
                raise ValueError("Position expert features require the actual selected native context")
            from frontier.execution_time_predictor.step5_cost import step5_native_expert_features
            model, _ = self._step5_ffn_model("moe_grouped_gemm", context=native_context)
            schema = moe_grouped_gemm_schema(model._frontier_feature_names, native_context=native_context)
            features = step5_native_expert_features(native_context,
                physical_rows=lane_workload.routing_source.gathered_physical_rows,
                live_assignments=lane_workload.routed_token_count)
            if tuple(features) != schema.features:
                raise ValueError("Native expert features differ from the selected per-layer model schema")
            return features
        model_info = self._predictions["moe_grouped_gemm"]
        schema = moe_grouped_gemm_schema(model_info["_feature_names"])
        if schema is None:
            return self._build_moe_load_imbalance_features(lane_workload, batch=batch)
        block_size_m = block_size_m_for_tokens(
            model_info["_model"]._frontier_block_size_m_ranges,
            self._get_moe_pre_routing_token_count(batch),
        )
        return {
            "total_routed_tokens": lane_workload.routed_token_count,
            "num_tokens_post_padded": num_tokens_post_padded(
                lane_workload.global_token_counts, block_size_m
            ),
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
