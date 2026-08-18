#!/usr/bin/env python3
"""Real-data, non-dummy E2E matrix for the MoE EP rank-staggering contract.

The harness deliberately invokes the checked-in architecture example wrappers.  It
does not change simulator semantics, invent profile rows, or convert a failed run
to a pass.  Generation is deterministic so the same manifest can be replayed on a
read-only baseline worktree.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from frontier.spec_decode.mtp_registry import (
    get_target_embedded_mtp_linear_ops,
    get_target_embedded_mtp_same_tp_linear_ops,
)
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    MOE_GATING_RUNTIME_CONTEXT_COLUMN,
)
from frontier.moe_routing_runtime import resolve_moe_gating_routing_runtime_path
from frontier.operators.families import (
    MOE_FAMILY,
    is_moe_operator_ep_agnostic,
    resolve_moe_operator_tp_key,
)
from frontier.types import ClusterType


ARCHITECTURE_CASE_COUNTS = {
    # The vLLM reference currently covers co-location and PDD.  PD-AF is kept
    # as a smaller Frontier-only structural sample until a vLLM PD-AF runtime
    # exists for a meaningful numerical comparison.
    "co-location": 50,
    "pd-disaggregation": 50,
    "pd-af-disaggregation": 10,
}
OPTIMIZATION_ARCHITECTURE_CASE_COUNTS = {
    "co-location": 91,
    "pd-disaggregation": 91,
    "pd-af-disaggregation": 18,
}
MODEL_ORDER = ("dense", "moe", "mixed")
PD_AF_VARIANT_INDICES = (0, 1, 2, 3, 5, 5, 6, 9, 10, 11)
MODEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "dense": {
        "model_name": "llama2_7b_dense_example",
        "total_experts": 1,
        "router_topk": 1,
    },
    "moe": {
        "model_name": "Phi-tiny-MoE-instruct",
        "device": "h800",
        "total_experts": 16,
        "router_topk": 2,
    },
    "moe_standard": {
        "model_name": "qwen3-a3b-30b-moe",
        "device": "a800",
        "total_experts": 128,
        "router_topk": 8,
    },
    "mixed": {
        "model_name": "step-moe-noquant-small",
        "device": "h800",
        "total_experts": 24,
        "router_topk": 3,
    },
}
OPTIMIZATION_MTP_MODEL_SPEC: Mapping[str, Any] = {
    "model_name": "qwen3-next-80b-a3b-instruct-reduced-l2",
    "device": "h800",
    "total_experts": 512,
    "router_topk": 10,
}
ROUTING_DISTRIBUTIONS = ("balanced", "random", "skewed", "zipf")
WORKLOADS: Mapping[str, tuple[int, int, int]] = {
    "prefill-heavy": (8, 1, 1),
    "decode-heavy": (2, 4, 1),
    "mixed": (4, 4, 2),
    "zero-routed": (2, 2, 1),
}
VARIANTS = (
    ("balanced", "prefill-heavy"),
    ("random", "prefill-heavy"),
    ("skewed", "prefill-heavy"),
    ("zipf", "prefill-heavy"),
    ("balanced", "decode-heavy"),
    ("random", "decode-heavy"),
    ("skewed", "decode-heavy"),
    ("zipf", "decode-heavy"),
    ("balanced", "mixed"),
    ("random", "mixed"),
    ("skewed", "zero-routed"),
    ("zipf", "zero-routed"),
)
REQUIRED_PROFILE_METADATA_COLUMNS = (
    "profiling_precision",
    "model_arch",
    "model_architecture_profile",
    "quant_signature",
    "measurement_type",
)
TARGET_EMBEDDED_MTP_COLUMNS = tuple(
    f"time_stats.{op_name}.median"
    for op_name in get_target_embedded_mtp_linear_ops()
)
TARGET_EMBEDDED_MTP_SAME_TP_COLUMNS = tuple(
    f"time_stats.{op_name}.median"
    for op_name in get_target_embedded_mtp_same_tp_linear_ops()
)


_EP_WORKLOAD_LINE_RE = re.compile(
    r"\[EP-WORKLOAD\]\[(?P<cluster>[^\]]+)\]\s+"
    r"batch_id=(?P<batch_id>-?\d+),\s+"
    r"layer_id=(?P<layer_id>-?\d+),\s+"
    r"ep_id=(?P<ep_id>-?\d+),\s+"
    r"moe_ep_size=(?P<moe_ep_size>\d+),\s+"
    r"per_expert_tokens=(?P<per_expert_tokens>\{.*\}),\s+"
    r"lane_compute_ms=(?P<lane_compute_ms>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"lane_comm_ms=(?P<lane_comm_ms>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
_EP_BARRIER_LINE_RE = re.compile(
    r"\[EP-BARRIER\]\[(?P<cluster>[^\]]+)\]\s+"
    r"batch_id=(?P<batch_id>-?\d+),\s+"
    r"layer_id=(?P<layer_id>-?\d+),\s+"
    r"phase=(?P<phase>dispatch|combine),\s+"
    r"expected_ep_ids=(?P<expected_ep_ids>\[[^\]]*\]),\s+"
    r"arrived_ep_ids=(?P<arrived_ep_ids>\[[^\]]*\]),\s+"
    r"max_lane_time_ms=(?P<max_lane_time_ms>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"barrier_time_ms=(?P<barrier_time_ms>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"barrier_end_time_s=(?P<barrier_end_time_s>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*$"
)
_EP_CONSERVATION_LINE_RE = re.compile(
    r"\[EP-CONSERVATION\]\[(?P<cluster>[^\]]+)\]\s+"
    r"batch_id=(?P<batch_id>-?\d+),\s+"
    r"layer_id=(?P<layer_id>-?\d+),\s+"
    r"routing_token_count=(?P<routing_token_count>\d+),\s+"
    r"router_topk=(?P<router_topk>\d+),\s+"
    r"total_routed_assignments=(?P<total_routed_assignments>\d+),\s+"
    r"per_ep_routed_tokens=(?P<per_ep_routed_tokens>\{.*\})\s*$"
)


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    baseline_case_id: str
    architecture: str
    model_kind: str
    model_name: str
    device: str
    routing_distribution: str
    seed: int
    workload_kind: str
    prefill_tokens: int
    decode_tokens: int
    num_requests: int
    ep_size: int
    moe_tensor_parallel_size: int
    total_experts: int
    router_topk: int
    pipeline_stages: int
    replica_count: int
    prefill_replicas: int
    decode_replicas: int
    decode_attn_replicas: int
    decode_ffn_replicas: int
    attn_tensor_parallel_size: int
    prefill_attn_tensor_parallel_size: int
    decode_attn_tensor_parallel_size: int
    prefill_moe_tensor_parallel_size: int
    decode_moe_tensor_parallel_size: int
    prefill_moe_expert_parallel_size: int
    decode_moe_expert_parallel_size: int
    total_cards: int
    num_layers: int
    moe_layer_ids: tuple[int, ...]
    simulation_mode: str = "offline"
    enable_chunked_prefill: bool = False
    decode_cuda_graph_mode: str = "none"
    use_cuda_graph: bool = False
    enable_prefix_caching: bool = False
    enable_mtp: bool = False
    request_source: str = "synthetic"
    optimization_stratum: str = "ordinary"
    pair_id: str | None = None
    comparison_group_id: str | None = None
    pair_role: str = "standalone"

    @property
    def is_moe(self) -> bool:
        return self.model_kind != "dense"

    @property
    def expects_zero_routed_lane(self) -> bool:
        return self.is_moe and self.workload_kind == "zero-routed" and self.ep_size > 1


@dataclass(frozen=True)
class _MoeProfileRequirement:
    """One runtime MoE profile contract for a concrete cluster role."""

    cluster_type: ClusterType
    moe_tensor_parallel_size: int
    moe_expert_parallel_size: int
    routing_runtime_path: str


def _model_layer_shape(model_name: str) -> tuple[int, tuple[int, ...]]:
    # Imported lazily so manifest generation remains usable from a minimal shell
    # while still failing explicitly when the simulator environment is absent.
    from frontier.config.model_config import BaseModelConfig

    config = BaseModelConfig.create_from_name(model_name)
    return int(config.num_layers), tuple(int(x) for x in config.get_moe_layer_ids())


def _case_cards(
    architecture: str,
    replica_count: int,
    ep_size: int,
    moe_tp: int,
    pipeline_stages: int,
) -> tuple[int, dict[str, int]]:
    if architecture == "co-location":
        attn_tp = moe_tp * ep_size
        effective_replicas = min(
            replica_count, max(1, 32 // (attn_tp * pipeline_stages))
        )
        return effective_replicas * attn_tp * pipeline_stages, {
            "attn_tp": attn_tp,
            "prefill_attn_tp": attn_tp,
            "decode_attn_tp": attn_tp,
            "prefill_replicas": effective_replicas,
            "decode_replicas": effective_replicas,
            "decode_attn_replicas": effective_replicas,
            "decode_ffn_replicas": effective_replicas,
        }
    if architecture == "pd-disaggregation":
        attn_tp = moe_tp * ep_size
        effective_replicas = min(
            replica_count, max(1, 32 // (2 * attn_tp * pipeline_stages))
        )
        return 2 * effective_replicas * attn_tp * pipeline_stages, {
            "attn_tp": attn_tp,
            "prefill_attn_tp": attn_tp,
            "decode_attn_tp": attn_tp,
            "prefill_replicas": effective_replicas,
            "decode_replicas": effective_replicas,
            "decode_attn_replicas": effective_replicas,
            "decode_ffn_replicas": effective_replicas,
        }
    if architecture == "pd-af-disaggregation":
        # Decode-attention is an independent role domain.  Keep it at TP=1 in
        # this matrix so the FFN EP capacity is the dimension under test.
        prefill_attn_tp = moe_tp * ep_size
        decode_attn_tp = 1
        role_replicas = min(
            replica_count,
            max(
                1,
                32
                // (
                    prefill_attn_tp * pipeline_stages
                    + decode_attn_tp
                    + moe_tp * ep_size * pipeline_stages
                ),
            ),
        )
        cards = role_replicas * (
            prefill_attn_tp * pipeline_stages
            + decode_attn_tp
            + moe_tp * ep_size * pipeline_stages
        )
        return cards, {
            "attn_tp": prefill_attn_tp,
            "prefill_attn_tp": prefill_attn_tp,
            "decode_attn_tp": decode_attn_tp,
            "prefill_replicas": role_replicas,
            "decode_replicas": role_replicas,
            "decode_attn_replicas": role_replicas,
            "decode_ffn_replicas": role_replicas,
        }
    raise ValueError(f"unsupported architecture: {architecture}")


def validate_case_parallel_semantics(case: MatrixCase) -> None:
    """Apply the same shared-domain mapping contract used by Frontier/vLLM."""

    from frontier.config.parallel_semantics import (
        FrontierParallelismMapping,
        validate_frontier_shared_parallel_domains,
    )

    if case.architecture in {"co-location", "pd-disaggregation"}:
        mapping = FrontierParallelismMapping(
            cluster_num_replicas=case.replica_count,
            attn_tensor_parallel_size=case.attn_tensor_parallel_size,
            attn_dp=1,
            moe_tensor_parallel_size=case.moe_tensor_parallel_size,
            moe_expert_parallel_size=case.ep_size,
        )
        validate_frontier_shared_parallel_domains(mapping)
        return

    # PD-AF PREFILL is a shared full-model domain and follows the same
    # invariant.  DECODE_ATTN and DECODE_FFN are independent domains; do not
    # compare their capacities across roles.  The FFN local world must still
    # be a positive TP×EP product.
    prefill_mapping = FrontierParallelismMapping(
        cluster_num_replicas=case.prefill_replicas,
        attn_tensor_parallel_size=case.prefill_attn_tensor_parallel_size,
        attn_dp=1,
        moe_tensor_parallel_size=case.prefill_moe_tensor_parallel_size,
        moe_expert_parallel_size=case.prefill_moe_expert_parallel_size,
    )
    validate_frontier_shared_parallel_domains(prefill_mapping)
    if case.decode_moe_tensor_parallel_size * case.decode_moe_expert_parallel_size <= 0:
        raise ValueError(f"invalid PD-AF DECODE_FFN local world for {case.case_id}")


def validate_optimization_case(case: MatrixCase) -> None:
    """Reject unsupported optimization combinations without rewriting them."""

    if case.device != "h800":
        raise ValueError(f"optimization matrix requires H800, got {case.device}")
    if case.prefill_tokens < 2:
        raise ValueError(
            f"prefill_tokens must be >1, got {case.prefill_tokens}"
        )
    if case.workload_kind == "zero-routed":
        if (
            not case.is_moe
            or case.ep_size <= 1
            or case.prefill_tokens * case.router_topk >= case.ep_size
        ):
            raise ValueError(
                "zero-routed workload cannot guarantee a zero-routed EP lane: "
                f"prefill_tokens={case.prefill_tokens}, "
                f"router_topk={case.router_topk}, ep_size={case.ep_size}"
            )
    if case.total_cards not in {8, 32}:
        raise ValueError(
            f"optimization matrix requires 8 or 32 cards, got {case.total_cards}"
        )
    if case.simulation_mode not in {"offline", "online"}:
        raise ValueError(f"unsupported simulation mode: {case.simulation_mode}")
    if case.simulation_mode == "online" and case.num_requests < 2:
        raise ValueError(
            "online optimization cases require at least two requests "
            "to emit inter-arrival evidence"
        )
    if case.decode_cuda_graph_mode not in {
        "none",
        "full_decode_only",
        "piecewise",
    }:
        raise ValueError(
            f"unsupported decode CUDA Graph mode: {case.decode_cuda_graph_mode}"
        )
    if case.enable_prefix_caching and case.enable_mtp:
        raise ValueError("Prefix Caching and MTP cannot be enabled together")

    if case.architecture == "pd-af-disaggregation":
        if case.pipeline_stages != 1:
            raise ValueError(
                "PD-AF optimization cases require pipeline stages=1, "
                f"got {case.pipeline_stages}"
            )
        if case.enable_prefix_caching or case.optimization_stratum == "prefix":
            raise ValueError("PD-AF does not support Prefix Caching")
        if case.enable_mtp or case.optimization_stratum == "mtp":
            raise ValueError("PD-AF does not support MTP")
        if case.decode_cuda_graph_mode != "none":
            raise ValueError("PD-AF does not support decode CUDA Graph modes")
    elif case.use_cuda_graph:
        raise ValueError("global CUDA Graph is PD-AF-only")

    if case.use_cuda_graph and case.decode_cuda_graph_mode != "none":
        raise ValueError(
            "global and decode CUDA Graph modes cannot be enabled together"
        )

    if case.optimization_stratum == "prefix":
        if case.enable_mtp:
            raise ValueError("Prefix Caching controls cannot enable MTP")
        if case.request_source != "prefix-trace":
            raise ValueError("Prefix Caching requires the repeated-prefix trace")
        if case.model_kind == "dense":
            raise ValueError("Prefix Caching matrix rows require a MoE-bearing model")
    elif case.enable_prefix_caching:
        raise ValueError("Prefix Caching is only valid in the prefix stratum")

    if case.optimization_stratum == "mtp":
        if case.decode_cuda_graph_mode != "none":
            raise ValueError("MTP cannot be combined with decode CUDA Graph")
        if case.routing_distribution != "random":
            raise ValueError("MTP matrix rows require random routing")
        if case.model_name != str(OPTIMIZATION_MTP_MODEL_SPEC["model_name"]):
            raise ValueError("MTP matrix rows require the Qwen3-Next target model")
        if case.enable_prefix_caching:
            raise ValueError("MTP controls cannot enable Prefix Caching")
    elif case.enable_mtp:
        raise ValueError("MTP is only valid in the MTP stratum")


def calculate_case_cards(case: MatrixCase) -> int:
    """Recompute physical cards from role-local topology."""

    if case.architecture == "co-location":
        return (
            case.replica_count
            * case.attn_tensor_parallel_size
            * case.pipeline_stages
        )
    if case.architecture == "pd-disaggregation":
        return (
            case.prefill_replicas
            * case.prefill_attn_tensor_parallel_size
            * case.pipeline_stages
            + case.decode_replicas
            * case.decode_attn_tensor_parallel_size
            * case.pipeline_stages
        )
    if case.architecture == "pd-af-disaggregation":
        decode_ffn_world = (
            case.decode_moe_tensor_parallel_size
            * case.decode_moe_expert_parallel_size
        )
        return (
            case.prefill_replicas
            * case.prefill_attn_tensor_parallel_size
            * case.pipeline_stages
            + case.decode_attn_replicas
            * case.decode_attn_tensor_parallel_size
            + case.decode_ffn_replicas
            * decode_ffn_world
            * case.pipeline_stages
        )
    raise ValueError(f"unsupported architecture: {case.architecture}")


def _undeclared_pair_differences(
    reference: MatrixCase,
    candidate: MatrixCase,
    *,
    declared_fields: set[str],
) -> list[str]:
    ignored_fields = {
        "case_id",
        "baseline_case_id",
        "pair_id",
        "comparison_group_id",
        "pair_role",
    }
    reference_payload = asdict(reference)
    candidate_payload = asdict(candidate)
    return sorted(
        field_name
        for field_name in reference_payload
        if field_name not in ignored_fields
        and field_name not in declared_fields
        and reference_payload[field_name] != candidate_payload[field_name]
    )


def validate_optimization_pairs(cases: Sequence[MatrixCase]) -> None:
    """Require paired controls to differ only in their declared optimization."""

    pair_groups: dict[str, list[MatrixCase]] = {}
    comparison_groups: dict[str, list[MatrixCase]] = {}
    for case in cases:
        if case.pair_id is not None:
            pair_groups.setdefault(case.pair_id, []).append(case)
        if case.comparison_group_id is not None:
            comparison_groups.setdefault(case.comparison_group_id, []).append(case)

    for pair_id, group in pair_groups.items():
        if len(group) != 2:
            raise ValueError(f"{pair_id} must contain exactly two rows")
        by_role = {case.pair_role: case for case in group}
        if set(by_role) != {"control", "enabled"}:
            raise ValueError(f"{pair_id} must contain control and enabled rows")
        control = by_role["control"]
        enabled = by_role["enabled"]
        if any(case.baseline_case_id != control.case_id for case in group):
            raise ValueError(f"{pair_id} baseline_case_id must name the control row")
        if control.optimization_stratum == "prefix":
            declared_fields = {"enable_prefix_caching"}
            if control.enable_prefix_caching or not enabled.enable_prefix_caching:
                raise ValueError(f"{pair_id} has invalid Prefix Caching roles")
        elif control.optimization_stratum == "mtp":
            declared_fields = {"enable_mtp"}
            if control.enable_mtp or not enabled.enable_mtp:
                raise ValueError(f"{pair_id} has invalid MTP roles")
        else:
            raise ValueError(f"{pair_id} has unsupported pair stratum")
        differences = _undeclared_pair_differences(
            control,
            enabled,
            declared_fields=declared_fields,
        )
        if differences:
            raise ValueError(
                f"{pair_id} changes undeclared fields: {', '.join(differences)}"
            )

    for comparison_group_id, group in comparison_groups.items():
        if len(group) == 1:
            continue
        if group[0].architecture == "pd-af-disaggregation":
            declared_fields = {"use_cuda_graph"}
            controls = [case for case in group if not case.use_cuda_graph]
            enabled_rows = [case for case in group if case.use_cuda_graph]
        else:
            declared_fields = {"decode_cuda_graph_mode"}
            chunked_prefill_varies = len(
                {case.enable_chunked_prefill for case in group}
            ) > 1
            if chunked_prefill_varies:
                declared_fields.add("enable_chunked_prefill")
                controls = [
                    case
                    for case in group
                    if case.decode_cuda_graph_mode == "none"
                    and not case.enable_chunked_prefill
                ]
                enabled_rows = [case for case in group if case not in controls]
            else:
                controls = [
                    case
                    for case in group
                    if case.decode_cuda_graph_mode == "none"
                ]
                enabled_rows = [
                    case
                    for case in group
                    if case.decode_cuda_graph_mode != "none"
                ]
        if len(controls) != 1 or not enabled_rows:
            raise ValueError(
                f"{comparison_group_id} requires one eager control and enabled rows"
            )
        control = controls[0]
        if any(case.baseline_case_id != control.case_id for case in group):
            raise ValueError(
                f"{comparison_group_id} baseline_case_id must name the control row"
            )
        for candidate in enabled_rows:
            differences = _undeclared_pair_differences(
                control,
                candidate,
                declared_fields=declared_fields,
            )
            if differences:
                raise ValueError(
                    f"{comparison_group_id} changes undeclared fields: "
                    f"{', '.join(differences)}"
                )


def build_matrix(repo_root: Path) -> list[MatrixCase]:
    """Build the deterministic 110-case matrix and validate its topology."""

    cases: list[MatrixCase] = []
    for architecture, case_count in ARCHITECTURE_CASE_COUNTS.items():
        for ordinal in range(case_count):
            model_kind = MODEL_ORDER[ordinal % len(MODEL_ORDER)]
            spec = MODEL_SPECS[model_kind]
            variant_index = (
                PD_AF_VARIANT_INDICES[ordinal % len(PD_AF_VARIANT_INDICES)]
                if architecture == "pd-af-disaggregation"
                else ordinal % len(VARIANTS)
            )
            distribution, workload_kind = VARIANTS[variant_index]
            if model_kind == "mixed":
                # The checked-in Step profile exposes only uniform_topk rows.
                distribution = "random"
            spec = (
                MODEL_SPECS["moe_standard"]
                if model_kind == "moe" and distribution != "random"
                else MODEL_SPECS[model_kind]
            )
            num_layers, moe_layer_ids = _model_layer_shape(str(spec["model_name"]))
            is_moe = model_kind != "dense"
            model_ordinal = ordinal // len(MODEL_ORDER)
            if not is_moe:
                ep_size = 1
            elif model_kind == "mixed":
                # Step's mixed profile needs TP=4 to fit the model shard in an
                # H800.  Its real linear/attention rows stop at TP=8, so the
                # mixed matrix uses EP<=2 (TP×EP<=8).  EP=4 remains covered by
                # the Phi/Qwen MoE populations.
                ep_size = (1, 2, 2)[model_ordinal % 3]
            else:
                ep_size = (1, 2, 4)[model_ordinal % 3]
            moe_tp = 4 if model_kind == "mixed" else 1
            pipeline_stages = 1
            replica_count = (1, 2, 4)[model_ordinal % 3]
            total_cards, topology = _case_cards(
                architecture, replica_count, ep_size, moe_tp, pipeline_stages
            )
            if total_cards > 32:
                raise AssertionError(
                    f"matrix topology exceeds 32 cards: {architecture} {model_kind} "
                    f"variant={variant_index} cards={total_cards}"
                )
            prefill_tokens, decode_tokens, num_requests = WORKLOADS[workload_kind]
            case_id = (
                f"{architecture.replace('-', '_')}_{model_kind}"
                f"_n{ordinal:02d}_v{variant_index:02d}"
            )
            baseline_id = f"{architecture.replace('-', '_')}_{model_kind}_baseline"
            case = MatrixCase(
                case_id=case_id,
                baseline_case_id=baseline_id,
                architecture=architecture,
                model_kind=model_kind,
                model_name=str(spec["model_name"]),
                device=str(spec.get("device", "h800")),
                routing_distribution=distribution if is_moe else "balanced",
                seed=42 + ordinal,
                workload_kind=workload_kind,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                num_requests=num_requests,
                ep_size=ep_size,
                moe_tensor_parallel_size=moe_tp,
                total_experts=int(spec["total_experts"]),
                router_topk=int(spec["router_topk"]),
                pipeline_stages=pipeline_stages,
                replica_count=replica_count,
                prefill_replicas=topology["prefill_replicas"],
                decode_replicas=topology["decode_replicas"],
                decode_attn_replicas=topology["decode_attn_replicas"],
                decode_ffn_replicas=topology["decode_ffn_replicas"],
                attn_tensor_parallel_size=1 if not is_moe else topology["attn_tp"],
                prefill_attn_tensor_parallel_size=topology["prefill_attn_tp"],
                decode_attn_tensor_parallel_size=topology["decode_attn_tp"],
                prefill_moe_tensor_parallel_size=moe_tp,
                decode_moe_tensor_parallel_size=moe_tp,
                prefill_moe_expert_parallel_size=ep_size,
                decode_moe_expert_parallel_size=ep_size,
                total_cards=total_cards
                if is_moe
                else _case_cards(architecture, replica_count, 1, 1, 1)[0],
                num_layers=num_layers,
                moe_layer_ids=moe_layer_ids,
            )
            validate_case_parallel_semantics(case)
            cases.append(case)
    if len(cases) != 110:
        raise AssertionError(f"expected 110 matrix cases, got {len(cases)}")
    return cases


def _optimization_topology(
    architecture: str,
    model_kind: str,
    model_name: str,
    total_cards: int,
    pipeline_stages: int,
    capacity_relation: str,
    *,
    world_size_override: int | None = None,
) -> dict[str, int]:
    if total_cards not in {8, 32}:
        raise ValueError(f"optimization cases require 8 or 32 cards, got {total_cards}")
    if pipeline_stages not in {1, 2}:
        raise ValueError(f"optimization cases require PP=1 or PP=2, got {pipeline_stages}")
    if capacity_relation not in {"equal", "gt", "lt"}:
        raise ValueError(f"unsupported capacity relation: {capacity_relation}")

    if world_size_override is not None:
        if world_size_override <= 0:
            raise ValueError(
                f"world_size_override must be positive, got {world_size_override}"
            )
        world_size = world_size_override
    elif architecture == "pd-af-disaggregation":
        if model_kind == "mixed" and total_cards == 8:
            raise ValueError("PD-AF mixed topology requires more than 8 cards")
        world_size = 2 if total_cards == 8 else 4
    elif model_name == str(OPTIMIZATION_MTP_MODEL_SPEC["model_name"]):
        world_size = 4
    elif model_kind == "dense":
        world_size = (
            2
            if architecture == "pd-disaggregation"
            and total_cards == 8
            and pipeline_stages == 2
            else 4
        )
    elif model_kind in {"moe", "mixed"}:
        world_size = 4 if total_cards == 8 else 8
        if (
            architecture == "pd-disaggregation"
            and total_cards == 8
            and pipeline_stages == 2
        ):
            world_size = 2
            if model_kind == "mixed":
                raise ValueError("PDD mixed 8-card topology cannot use PP=2")
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")

    if model_kind == "dense":
        moe_tp = world_size
        ep_size = 1
    elif model_kind == "moe":
        moe_tp = 1
        ep_size = world_size
    elif model_kind == "mixed":
        moe_tp = 4
        ep_size = world_size // moe_tp
    if moe_tp * ep_size != world_size:
        raise AssertionError(
            f"{model_kind} topology does not conserve world size: "
            f"moe_tp={moe_tp}, ep={ep_size}, world={world_size}"
        )

    if architecture == "co-location":
        denominator = world_size * pipeline_stages
        if total_cards % denominator:
            raise ValueError(
                f"co-location topology cannot model {total_cards} cards with "
                f"world={world_size}, pp={pipeline_stages}"
            )
        replica_count = total_cards // denominator
        topology = {
            "pipeline_stages": pipeline_stages,
            "replica_count": replica_count,
            "prefill_replicas": replica_count,
            "decode_replicas": replica_count,
            "decode_attn_replicas": replica_count,
            "decode_ffn_replicas": replica_count,
            "attn_tp": world_size,
            "prefill_attn_tp": world_size,
            "decode_attn_tp": world_size,
        }
    elif architecture == "pd-disaggregation":
        denominator = world_size * pipeline_stages
        if total_cards % denominator:
            raise ValueError(
                f"PDD topology cannot model {total_cards} cards with "
                f"world={world_size}, pp={pipeline_stages}"
            )
        role_replica_sum = total_cards // denominator
        if role_replica_sum < 2:
            raise ValueError("PDD requires at least one PREFILL and one DECODE Replica")
        if capacity_relation == "equal" or role_replica_sum == 2:
            if role_replica_sum % 2:
                raise ValueError("equal PDD capacity requires an even Replica sum")
            prefill_replicas = decode_replicas = role_replica_sum // 2
        elif capacity_relation == "gt":
            prefill_replicas, decode_replicas = role_replica_sum - 1, 1
        else:
            prefill_replicas, decode_replicas = 1, role_replica_sum - 1
        topology = {
            "pipeline_stages": pipeline_stages,
            "replica_count": max(prefill_replicas, decode_replicas),
            "prefill_replicas": prefill_replicas,
            "decode_replicas": decode_replicas,
            "decode_attn_replicas": decode_replicas,
            "decode_ffn_replicas": decode_replicas,
            "attn_tp": world_size,
            "prefill_attn_tp": world_size,
            "decode_attn_tp": world_size,
        }
    elif architecture == "pd-af-disaggregation":
        if (
            pipeline_stages == 1
            and total_cards == 4 * world_size
            and world_size in {2, 8}
        ):
            if capacity_relation == "equal":
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 2, 1, 1
            elif capacity_relation == "gt":
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 1, 2, 1
            else:
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 1, 1, 2
            decode_attn_tp = world_size
        elif pipeline_stages == 1 and total_cards == 32:
            if capacity_relation == "equal":
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 4, 2, 2
            elif capacity_relation == "gt":
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 4, 3, 1
            else:
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 4, 1, 3
            decode_attn_tp = 4
        elif pipeline_stages == 2 and total_cards == 32:
            if capacity_relation == "equal":
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 1, 2, 2
                decode_attn_tp = 4
            elif capacity_relation == "gt":
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 2, 2, 1
                decode_attn_tp = 4
            else:
                prefill_replicas, decode_attn_replicas, decode_ffn_replicas = 1, 1, 2
                decode_attn_tp = 8
        else:
            raise ValueError(
                f"unsupported PD-AF topology: cards={total_cards}, "
                f"pp={pipeline_stages}"
            )
        topology = {
            "pipeline_stages": pipeline_stages,
            "replica_count": prefill_replicas,
            "prefill_replicas": prefill_replicas,
            "decode_replicas": decode_attn_replicas,
            "decode_attn_replicas": decode_attn_replicas,
            "decode_ffn_replicas": decode_ffn_replicas,
            "attn_tp": world_size,
            "prefill_attn_tp": world_size,
            "decode_attn_tp": decode_attn_tp,
        }
    else:
        raise ValueError(f"unsupported architecture: {architecture}")

    if architecture == "co-location":
        computed_cards = (
            topology["replica_count"] * world_size * pipeline_stages
        )
    elif architecture == "pd-disaggregation":
        computed_cards = (
            topology["prefill_replicas"] + topology["decode_replicas"]
        ) * world_size * pipeline_stages
    else:
        computed_cards = (
            topology["prefill_replicas"] * world_size * pipeline_stages
            + topology["decode_attn_replicas"] * topology["decode_attn_tp"]
            + topology["decode_ffn_replicas"] * world_size * pipeline_stages
        )
    if computed_cards != total_cards:
        raise AssertionError(
            f"{architecture} {model_kind} topology models {computed_cards} cards, "
            f"expected {total_cards}"
        )
    return {
        **topology,
        "moe_tp": moe_tp,
        "ep_size": ep_size,
        "total_cards": total_cards,
    }


def _optimization_workload(
    optimization_stratum: str,
    workload_index: int,
    *,
    zero_routed: bool,
    simulation_mode: str,
) -> tuple[str, int, int, int]:
    if optimization_stratum == "prefix":
        return "prefix-trace", 32, 2, 2
    if zero_routed:
        return "zero-routed", 2, 4, 2 if simulation_mode == "online" else 1
    labels = ("prefill-heavy", "decode-heavy", "mixed")
    if optimization_stratum == "mtp":
        return (
            labels[workload_index % len(labels)],
            384 + 64 * workload_index,
            64 + 16 * (workload_index % 4),
            4 + workload_index % 3,
        )
    if optimization_stratum == "pd-af-cube":
        return (
            labels[workload_index % len(labels)],
            320 + 48 * workload_index,
            32 + 8 * (workload_index % 5),
            4 + workload_index % 4,
        )
    return (
        labels[workload_index % len(labels)],
        256 + 32 * workload_index,
        32 + 8 * (workload_index % 4),
        4 + workload_index % 3,
    )


def _make_optimization_case(
    *,
    repo_root: Path,
    case_id: str,
    baseline_case_id: str,
    seed: int,
    architecture: str,
    model_kind: str,
    use_mtp_model: bool,
    routing_distribution: str,
    simulation_mode: str,
    total_cards: int,
    enable_chunked_prefill: bool,
    decode_cuda_graph_mode: str,
    use_cuda_graph: bool,
    enable_prefix_caching: bool,
    enable_mtp: bool,
    request_source: str,
    optimization_stratum: str,
    pair_id: str | None,
    comparison_group_id: str | None,
    pair_role: str,
    pipeline_stages: int,
    capacity_relation: str,
    workload_index: int,
    zero_routed: bool = False,
) -> MatrixCase:
    if simulation_mode not in {"offline", "online"}:
        raise ValueError(f"unsupported simulation mode: {simulation_mode}")
    spec = OPTIMIZATION_MTP_MODEL_SPEC if use_mtp_model else MODEL_SPECS[model_kind]
    model_name = str(spec["model_name"])
    topology = _optimization_topology(
        architecture,
        model_kind,
        model_name,
        total_cards,
        pipeline_stages,
        capacity_relation,
        world_size_override=8 if zero_routed else None,
    )
    workload_kind, prefill_tokens, decode_tokens, num_requests = (
        _optimization_workload(
            optimization_stratum,
            workload_index,
            zero_routed=zero_routed,
            simulation_mode=simulation_mode,
        )
    )
    num_layers, moe_layer_ids = _model_layer_shape(model_name)
    case = MatrixCase(
        case_id=case_id,
        baseline_case_id=baseline_case_id,
        architecture=architecture,
        model_kind=model_kind,
        model_name=model_name,
        device="h800",
        routing_distribution=(
            routing_distribution if model_kind != "dense" else "balanced"
        ),
        seed=seed,
        workload_kind=workload_kind,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        num_requests=num_requests,
        ep_size=topology["ep_size"],
        moe_tensor_parallel_size=topology["moe_tp"],
        total_experts=int(spec["total_experts"]),
        router_topk=int(spec["router_topk"]),
        pipeline_stages=topology["pipeline_stages"],
        replica_count=topology["replica_count"],
        prefill_replicas=topology["prefill_replicas"],
        decode_replicas=topology["decode_replicas"],
        decode_attn_replicas=topology["decode_attn_replicas"],
        decode_ffn_replicas=topology["decode_ffn_replicas"],
        attn_tensor_parallel_size=topology["attn_tp"],
        prefill_attn_tensor_parallel_size=topology["prefill_attn_tp"],
        decode_attn_tensor_parallel_size=topology["decode_attn_tp"],
        prefill_moe_tensor_parallel_size=topology["moe_tp"],
        decode_moe_tensor_parallel_size=topology["moe_tp"],
        prefill_moe_expert_parallel_size=topology["ep_size"],
        decode_moe_expert_parallel_size=topology["ep_size"],
        total_cards=topology["total_cards"],
        num_layers=num_layers,
        moe_layer_ids=moe_layer_ids,
        simulation_mode=simulation_mode,
        enable_chunked_prefill=enable_chunked_prefill,
        decode_cuda_graph_mode=decode_cuda_graph_mode,
        use_cuda_graph=use_cuda_graph,
        enable_prefix_caching=enable_prefix_caching,
        enable_mtp=enable_mtp,
        request_source=request_source,
        optimization_stratum=optimization_stratum,
        pair_id=pair_id,
        comparison_group_id=comparison_group_id,
        pair_role=pair_role,
    )
    validate_case_parallel_semantics(case)
    validate_optimization_case(case)
    return case


def _shared_optimization_cases(
    repo_root: Path,
    architecture: str,
    *,
    seed_start: int,
) -> list[MatrixCase]:
    prefix = architecture.replace("-", "_")
    group_sizes = (3,) * 16 + (2,) * 2 + (1,) * 3
    if architecture == "co-location":
        model_routing = (
            (("dense", "balanced"),) * 10
            + (
                ("mixed", "balanced"),
                ("mixed", "skewed"),
                ("mixed", "skewed"),
                ("mixed", "zipf"),
                ("mixed", "zipf"),
                ("moe", "zipf"),
                ("mixed", "balanced"),
                ("mixed", "balanced"),
                ("dense", "balanced"),
                ("mixed", "skewed"),
                ("moe", "random"),
            )
        )
        offline_groups = set(range(8)) | {16, 18, 19}
        chunk_off_groups = set(range(8)) | {17, 18, 20}
    elif architecture == "pd-disaggregation":
        model_routing = (
            (("dense", "balanced"),) * 10
            + (
                ("moe", "balanced"),
                ("mixed", "skewed"),
                ("mixed", "skewed"),
                ("mixed", "zipf"),
                ("mixed", "zipf"),
                ("mixed", "zipf"),
                ("moe", "random"),
                ("mixed", "balanced"),
                ("mixed", "balanced"),
                ("mixed", "balanced"),
                ("mixed", "skewed"),
            )
        )
        offline_groups = set(range(8)) | {16, 18}
        chunk_off_groups = set(range(8)) | {16, 19}
    else:
        raise ValueError(f"unsupported shared optimization architecture: {architecture}")
    card8_groups = set(range(8)) | {17, 20}
    pp2_groups = set(range(6))

    factorial_context_sources = {0: 0, 1: 9}
    skipped_contexts = {8, 9}
    cases: list[MatrixCase] = []
    for group_index, (group_size, (model_kind, routing)) in enumerate(
        zip(group_sizes, model_routing, strict=True)
    ):
        if group_index in skipped_contexts:
            continue
        context_index = factorial_context_sources.get(group_index, group_index)
        graph_modes = (
            ("none", "full_decode_only", "piecewise")
            if group_size == 3
            else (("none", "full_decode_only") if group_size == 2 else ("none",))
        )
        comparison_group_id = f"{prefix}_ordinary_graph_{group_index:02d}"
        chunked_prefill_modes = (
            (False, True)
            if group_index in factorial_context_sources
            else (context_index not in chunk_off_groups,)
        )
        baseline_case_id = (
            f"{comparison_group_id}_none_chunk_off"
            if len(chunked_prefill_modes) == 2
            else f"{comparison_group_id}_none"
        )
        for graph_mode, enable_chunked_prefill in itertools.product(
            graph_modes,
            chunked_prefill_modes,
        ):
            case_id = (
                f"{comparison_group_id}_{graph_mode}_"
                f"chunk_{'on' if enable_chunked_prefill else 'off'}"
                if len(chunked_prefill_modes) == 2
                else f"{comparison_group_id}_{graph_mode}"
            )
            cases.append(
                _make_optimization_case(
                    repo_root=repo_root,
                    case_id=case_id,
                    baseline_case_id=baseline_case_id,
                    seed=seed_start + context_index,
                    architecture=architecture,
                    model_kind=model_kind,
                    use_mtp_model=False,
                    routing_distribution=routing,
                    simulation_mode=(
                        "offline" if context_index in offline_groups else "online"
                    ),
                    total_cards=8 if context_index in card8_groups else 32,
                    enable_chunked_prefill=enable_chunked_prefill,
                    decode_cuda_graph_mode=graph_mode,
                    use_cuda_graph=False,
                    enable_prefix_caching=False,
                    enable_mtp=False,
                    request_source="synthetic",
                    optimization_stratum="ordinary",
                    pair_id=None,
                    comparison_group_id=comparison_group_id,
                    pair_role=(
                        "control"
                        if graph_mode == "none" and not enable_chunked_prefill
                        else "enabled"
                    ),
                    pipeline_stages=2 if context_index in pp2_groups else 1,
                    capacity_relation=("equal", "gt", "lt")[context_index % 3],
                    workload_index=context_index,
                    zero_routed=(
                        architecture == "co-location" and context_index == 20
                    )
                    or (
                        architecture == "pd-disaggregation" and context_index == 16
                    ),
                )
            )

    prefix_routings = (
        "balanced",
        "skewed",
        "zipf",
        "balanced",
        "skewed",
        "zipf",
        "balanced",
        "skewed",
        "balanced",
        "skewed",
        "zipf",
    )
    prefix_graph_modes = (
        "none",
        "full_decode_only",
        "piecewise",
        "full_decode_only",
        "piecewise",
        "none",
        "full_decode_only",
        "piecewise",
        "none",
        "full_decode_only",
        "none",
    )
    prefix_models = (
        "moe",
        "mixed",
        "moe",
        "mixed",
        "moe",
        "mixed",
        "moe",
        "mixed",
        "moe",
        "mixed",
        "moe",
    )
    prefix_chunk_off = {0, 2, 5, 7, 9}
    prefix_offline = (
        {0, 2, 3, 5, 7, 10}
        if architecture == "co-location"
        else {1, 4, 6, 8, 9}
    )
    prefix_card8 = {0, 1, 3, 5, 7, 9}
    prefix_pp2 = {2, 4, 6, 8}
    for context_index in range(11):
        pair_id = f"{prefix}_prefix_{context_index:02d}"
        control_case_id = f"{pair_id}_control"
        for enabled in (False, True):
            cases.append(
                _make_optimization_case(
                    repo_root=repo_root,
                    case_id=(
                        f"{pair_id}_{'enabled' if enabled else 'control'}"
                    ),
                    baseline_case_id=control_case_id,
                    seed=seed_start + 100 + context_index,
                    architecture=architecture,
                    model_kind=prefix_models[context_index],
                    use_mtp_model=False,
                    routing_distribution=prefix_routings[context_index],
                    simulation_mode=(
                        "offline"
                        if context_index in prefix_offline
                        else "online"
                    ),
                    total_cards=8 if context_index in prefix_card8 else 32,
                    enable_chunked_prefill=context_index not in prefix_chunk_off,
                    decode_cuda_graph_mode=prefix_graph_modes[context_index],
                    use_cuda_graph=False,
                    enable_prefix_caching=enabled,
                    enable_mtp=False,
                    request_source="prefix-trace",
                    optimization_stratum="prefix",
                    pair_id=pair_id,
                    comparison_group_id=None,
                    pair_role="enabled" if enabled else "control",
                    pipeline_stages=2 if context_index in prefix_pp2 else 1,
                    capacity_relation=("equal", "gt", "lt")[context_index % 3],
                    workload_index=context_index,
                )
            )

    mtp_offline_count = 3 if architecture == "co-location" else 4
    for context_index in range(7):
        pair_id = f"{prefix}_mtp_{context_index:02d}"
        control_case_id = f"{pair_id}_control"
        for enabled in (False, True):
            cases.append(
                _make_optimization_case(
                    repo_root=repo_root,
                    case_id=(
                        f"{pair_id}_{'enabled' if enabled else 'control'}"
                    ),
                    baseline_case_id=control_case_id,
                    seed=seed_start + 200 + context_index,
                    architecture=architecture,
                    model_kind="moe",
                    use_mtp_model=True,
                    routing_distribution="random",
                    simulation_mode=(
                        "offline"
                        if context_index < mtp_offline_count
                        else "online"
                    ),
                    total_cards=8 if context_index < 4 else 32,
                    enable_chunked_prefill=context_index >= 4,
                    decode_cuda_graph_mode="none",
                    use_cuda_graph=False,
                    enable_prefix_caching=False,
                    enable_mtp=enabled,
                    request_source="synthetic",
                    optimization_stratum="mtp",
                    pair_id=pair_id,
                    comparison_group_id=None,
                    pair_role="enabled" if enabled else "control",
                    pipeline_stages=2 if context_index in {5, 6} else 1,
                    capacity_relation=("equal", "gt", "lt")[context_index % 3],
                    workload_index=context_index,
                )
            )

    expected_count = OPTIMIZATION_ARCHITECTURE_CASE_COUNTS[architecture]
    if len(cases) != expected_count:
        raise AssertionError(
            f"expected {expected_count} {architecture} optimization cases, "
            f"got {len(cases)}"
        )
    return cases


def _pdaf_optimization_cases(
    repo_root: Path,
    *,
    seed_start: int,
) -> list[MatrixCase]:
    paired_contexts = (
        ("offline", False, "dense", 8, "balanced", "equal", 1, False),
        ("offline", False, "moe", 8, "random", "gt", 1, False),
        ("offline", True, "mixed", 32, "balanced", "lt", 1, False),
        ("online", False, "dense", 32, "balanced", "gt", 1, False),
        ("online", True, "moe", 32, "skewed", "lt", 1, False),
        ("online", True, "mixed", 32, "zipf", "equal", 1, False),
    )
    cases: list[MatrixCase] = []
    for context_index, (
        simulation_mode,
        enable_chunked_prefill,
        model_kind,
        total_cards,
        routing,
        relation,
        pipeline_stages,
        zero_routed,
    ) in enumerate(paired_contexts):
        comparison_group_id = f"pd_af_cuda_graph_{context_index:02d}"
        baseline_case_id = f"{comparison_group_id}_control"
        for enabled in (False, True):
            cases.append(
                _make_optimization_case(
                    repo_root=repo_root,
                    case_id=(
                        f"{comparison_group_id}_"
                        f"{'enabled' if enabled else 'control'}"
                    ),
                    baseline_case_id=baseline_case_id,
                    seed=seed_start + context_index,
                    architecture="pd-af-disaggregation",
                    model_kind=model_kind,
                    use_mtp_model=False,
                    routing_distribution=routing,
                    simulation_mode=simulation_mode,
                    total_cards=total_cards,
                    enable_chunked_prefill=enable_chunked_prefill,
                    decode_cuda_graph_mode="none",
                    use_cuda_graph=enabled,
                    enable_prefix_caching=False,
                    enable_mtp=False,
                    request_source="synthetic",
                    optimization_stratum="pd-af-cube",
                    pair_id=None,
                    comparison_group_id=comparison_group_id,
                    pair_role="enabled" if enabled else "control",
                    pipeline_stages=pipeline_stages,
                    capacity_relation=relation,
                    workload_index=context_index,
                    zero_routed=zero_routed,
                )
            )

    standalone_contexts = (
        ("offline", False, False, "dense", 8, "balanced", "equal", False),
        ("offline", True, False, "moe", 8, "random", "equal", False),
        ("offline", True, True, "mixed", 32, "balanced", "gt", False),
        ("online", False, False, "dense", 32, "balanced", "gt", False),
        ("online", False, True, "moe", 32, "skewed", "lt", True),
        ("online", True, True, "mixed", 32, "zipf", "lt", False),
    )
    for context_index, (
        simulation_mode,
        enable_chunked_prefill,
        use_cuda_graph,
        model_kind,
        total_cards,
        routing,
        relation,
        zero_routed,
    ) in enumerate(standalone_contexts):
        case_id = f"pd_af_cube_standalone_{context_index:02d}"
        cases.append(
            _make_optimization_case(
                repo_root=repo_root,
                case_id=case_id,
                baseline_case_id=case_id,
                seed=seed_start + 100 + context_index,
                architecture="pd-af-disaggregation",
                model_kind=model_kind,
                use_mtp_model=False,
                routing_distribution=routing,
                simulation_mode=simulation_mode,
                total_cards=total_cards,
                enable_chunked_prefill=enable_chunked_prefill,
                decode_cuda_graph_mode="none",
                use_cuda_graph=use_cuda_graph,
                enable_prefix_caching=False,
                enable_mtp=False,
                request_source="synthetic",
                optimization_stratum="pd-af-cube",
                pair_id=None,
                comparison_group_id=f"pd_af_standalone_{context_index:02d}",
                pair_role="standalone",
                pipeline_stages=1,
                capacity_relation=relation,
                workload_index=6 + context_index,
                zero_routed=zero_routed,
            )
        )
    if len(cases) != 18:
        raise AssertionError(f"expected 18 PD-AF optimization cases, got {len(cases)}")
    return cases


def build_optimization_matrix(repo_root: Path) -> list[MatrixCase]:
    """Build the exact 200-case H800 optimization matrix."""

    cases = _shared_optimization_cases(
        repo_root,
        "co-location",
        seed_start=10_000,
    )
    cases.extend(
        _shared_optimization_cases(
            repo_root,
            "pd-disaggregation",
            seed_start=20_000,
        )
    )
    cases.extend(_pdaf_optimization_cases(repo_root, seed_start=30_000))
    if len(cases) != 200:
        raise AssertionError(f"expected 200 optimization cases, got {len(cases)}")
    validate_optimization_pairs(cases)
    return cases


def _script_for_case(case: MatrixCase, repo_root: Path) -> Path:
    root = repo_root / "examples" / "architecture"
    mode_suffix = "_online" if case.simulation_mode == "online" else ""
    mode_dir = case.simulation_mode
    if case.optimization_stratum == "prefix":
        names = {
            "co-location": (
                f"co-location/{mode_dir}/moe_prefix_caching{mode_suffix}.sh"
            ),
            "pd-disaggregation": (
                f"pdd/{mode_dir}/moe_prefix_caching{mode_suffix}.sh"
            ),
        }
    elif case.optimization_stratum == "mtp":
        names = {
            "co-location": f"co-location/{mode_dir}/moe_spec_dec{mode_suffix}.sh",
            "pd-disaggregation": f"pdd/{mode_dir}/moe_spec_dec{mode_suffix}.sh",
        }
    elif case.model_kind == "dense":
        names = {
            "co-location": (
                f"co-location/{mode_dir}/dense_model_basic{mode_suffix}.sh"
            ),
            "pd-disaggregation": (
                f"pdd/{mode_dir}/dense_model_basic{mode_suffix}.sh"
            ),
            "pd-af-disaggregation": (
                f"pd-af-disagg/{mode_dir}/dense_model_basic{mode_suffix}.sh"
            ),
        }
    else:
        names = {
            "co-location": (
                f"co-location/{mode_dir}/moe_model_basic{mode_suffix}.sh"
            ),
            "pd-disaggregation": (
                f"pdd/{mode_dir}/moe_model_basic{mode_suffix}.sh"
            ),
            "pd-af-disaggregation": (
                f"pd-af-disagg/{mode_dir}/moe_model_ep{mode_suffix}.sh"
            ),
        }
    try:
        relative_path = names[case.architecture]
    except KeyError as exc:
        raise ValueError(
            f"{case.optimization_stratum} is unsupported for {case.architecture}"
        ) from exc
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_shell_command(
    case: MatrixCase, repo_root: Path, output_root: Path
) -> tuple[str, dict[str, str]]:
    """Return an exact wrapper command and its environment for one case."""

    script = _script_for_case(case, repo_root)
    metrics_root = output_root / case.case_id / "metrics"
    env = {key: value for key, value in os.environ.items()}
    env.update(
        {
            "PYTHONPATH": str(repo_root),
            "PYTHON_BIN": sys.executable,
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
            "MODEL_NAME": case.model_name,
            "ENABLE_DUMMY_MODE": "false",
            "DECODE_CUDA_GRAPH_MODE": case.decode_cuda_graph_mode,
            "ENABLE_CUDA_GRAPH": str(case.use_cuda_graph).lower(),
            "ENABLE_CHUNKED_PREFILL": str(case.enable_chunked_prefill).lower(),
            "NUM_REQUESTS": str(case.num_requests),
            "PREFILL_TOKENS": str(case.prefill_tokens),
            "DECODE_TOKENS": str(case.decode_tokens),
            "QPS": "1.0",
            "RUN_ID": case.case_id,
            "METRICS_OUTPUT_DIR": str(metrics_root),
            "MOE_ROUTING_DISTRIBUTION_TYPE": case.routing_distribution,
            "MOE_ROUTING_SEED": str(case.seed),
            "TOTAL_EXPERTS": str(case.total_experts),
            "ROUTER_TOPK": str(case.router_topk),
            "MAX_TOKENS_IN_BATCH": str(
                min(64, max(1, case.prefill_tokens // 2))
                if case.enable_chunked_prefill
                else (
                    32
                    if case.optimization_stratum == "prefix"
                    else max(64, case.prefill_tokens)
                )
            ),
            "LONG_PREFILL_TOKEN_THRESHOLD": (
                "16" if case.enable_chunked_prefill else "0"
            ),
        }
    )
    if case.optimization_stratum == "prefix":
        env["TRACE_FILE"] = str(
            repo_root / "examples" / "fixtures" / "prefix_cache_shared_session_trace.csv"
        )
    if case.optimization_stratum == "mtp":
        env.update(
            {
                "SPEC_METHOD": "qwen3_next_mtp",
                "MTP_N_PREDICT": "2",
                "MTP_NUM_LAYERS": "1",
            }
        )
    if case.architecture == "co-location":
        env.update(
            {
                "NUM_REPLICAS": str(case.replica_count),
                "ATTN_TP": str(case.attn_tensor_parallel_size),
                "MOE_TP": str(case.moe_tensor_parallel_size),
                "MOE_EP": str(case.ep_size),
                "PP": str(case.pipeline_stages),
                "DEVICE": case.device,
            }
        )
    elif case.architecture == "pd-disaggregation":
        env.update(
            {
                "PREFILL_REPLICAS": str(case.prefill_replicas),
                "DECODE_REPLICAS": str(case.decode_replicas),
                "PREFILL_ATTN_TP": str(case.prefill_attn_tensor_parallel_size),
                "PREFILL_MOE_TP": str(case.prefill_moe_tensor_parallel_size),
                "PREFILL_MOE_EP": str(case.prefill_moe_expert_parallel_size),
                "DECODE_ATTN_TP": str(case.decode_attn_tensor_parallel_size),
                "DECODE_MOE_TP": str(case.decode_moe_tensor_parallel_size),
                "DECODE_MOE_EP": str(case.decode_moe_expert_parallel_size),
                "PREFILL_DEVICE": case.device,
                "DECODE_DEVICE": case.device,
                "PREFILL_PP": str(case.pipeline_stages),
                "DECODE_PP": str(case.pipeline_stages),
            }
        )
    else:
        env.update(
            {
                "PREFILL_REPLICAS": str(case.prefill_replicas),
                "DECODE_ATTN_REPLICAS": str(case.decode_attn_replicas),
                "DECODE_FFN_REPLICAS": str(case.decode_ffn_replicas),
                "PREFILL_ATTN_TP": str(case.prefill_attn_tensor_parallel_size),
                "PREFILL_MOE_TP": str(case.prefill_moe_tensor_parallel_size),
                "PREFILL_MOE_EP": str(case.prefill_moe_expert_parallel_size),
                "DECODE_ATTN_TP": str(case.decode_attn_tensor_parallel_size),
                "DECODE_FFN_MOE_TP": str(case.decode_moe_tensor_parallel_size),
                "DECODE_FFN_MOE_EP": str(case.decode_moe_expert_parallel_size),
                "PREFILL_DEVICE": case.device,
                "DECODE_ATTN_DEVICE": case.device,
                "DECODE_FFN_DEVICE": case.device,
                "PREFILL_PP": str(case.pipeline_stages),
                "DECODE_ATTN_PP": "1",
                "DECODE_FFN_PP": str(case.pipeline_stages),
            }
        )
    command_parts = ["bash", str(script), "--"]
    # The co-location MoE wrapper predates the explicit DEVICE environment
    # contract used by the dense/PDD/PD-AF wrappers.  Pass the device as a
    # regular CLI override so non-dummy profile lookup cannot silently fall
    # back to a different SKU.
    if case.architecture == "co-location" and case.model_kind != "dense":
        command_parts.extend(["--replica_config_device", case.device])
    if case.optimization_stratum == "prefix":
        command_parts.append(
            "--vllm_v1_scheduler_config_enable_prefix_caching"
            if case.enable_prefix_caching
            else "--no-vllm_v1_scheduler_config_enable_prefix_caching"
        )
    if case.optimization_stratum == "mtp":
        command_parts.append(
            "--speculative_decoding_config_enabled"
            if case.enable_mtp
            else "--no-speculative_decoding_config_enabled"
        )
    if case.architecture == "pd-af-disaggregation":
        if case.use_cuda_graph:
            command_parts.extend(
                ["--use_cuda_graph", "--cudagraph_capture_sizes", "8", "16", "32", "64"]
            )
        else:
            command_parts.append("--no-use_cuda_graph")
    # Keep predictor artifacts isolated from the repository's shared cache. A
    # previously interrupted non-dummy run can leave a truncated pickle there;
    # reusing it would make an otherwise valid case fail with an unrelated
    # deserialization error. The caller chooses a fresh output root for each
    # matrix campaign, so this path is deterministic and provenance-visible.
    predictor_cache_dir = (output_root / "_predictor_cache").resolve()
    command_parts.extend(["--metrics_config_cache_dir", str(predictor_cache_dir)])
    command = shlex.join(command_parts)
    return command, env


def validate_profile_inputs(case: MatrixCase, root: Path) -> list[Path]:
    """Require real profiling rows; never substitute dummy or synthetic data."""

    profile_root = root / "data" / "profiling" / "compute"
    if (root / case.device).is_dir() and not (profile_root).is_dir():
        profile_root = root
    model_dir = profile_root / case.device / case.model_name
    eager_attention_tps: set[int] = set()
    eager_linear_tps: set[int] = set()
    eager_moe_keys: set[tuple[int, int]] = set()
    kernel_attention_tps: set[int] = set()
    kernel_linear_tps: set[int] = set()
    kernel_moe_keys: set[tuple[int, int]] = set()
    moe_profile_requirements: dict[
        Path, tuple[str, set[_MoeProfileRequirement]]
    ] = {}

    def register_moe_requirement(
        path: Path,
        measurement_type: str,
        *,
        cluster_type: ClusterType,
        moe_tensor_parallel_size: int,
        moe_expert_parallel_size: int,
    ) -> None:
        requirement = _MoeProfileRequirement(
            cluster_type=cluster_type,
            moe_tensor_parallel_size=moe_tensor_parallel_size,
            moe_expert_parallel_size=moe_expert_parallel_size,
            routing_runtime_path=resolve_moe_gating_routing_runtime_path(
                case.routing_distribution
            ),
        )
        existing = moe_profile_requirements.get(path)
        if existing is None:
            moe_profile_requirements[path] = (measurement_type, {requirement})
            return
        existing_measurement_type, requirements = existing
        if existing_measurement_type != measurement_type:
            raise ValueError(
                f"{path} is required as both {existing_measurement_type} and "
                f"{measurement_type} for {case.case_id}"
            )
        requirements.add(requirement)

    if case.architecture == "co-location":
        eager_attention_tps.add(case.attn_tensor_parallel_size)
        eager_linear_tps.add(case.attn_tensor_parallel_size)
        if case.is_moe:
            eager_linear_tps.add(case.moe_tensor_parallel_size)
            eager_moe_keys.add((case.moe_tensor_parallel_size, case.ep_size))
            register_moe_requirement(
                model_dir / "moe.csv",
                "CUDA_EVENT",
                cluster_type=ClusterType.MONOLITHIC,
                moe_tensor_parallel_size=case.moe_tensor_parallel_size,
                moe_expert_parallel_size=case.ep_size,
            )
        if case.decode_cuda_graph_mode != "none":
            kernel_attention_tps.add(case.attn_tensor_parallel_size)
            kernel_linear_tps.update(eager_linear_tps)
            kernel_moe_keys.update(eager_moe_keys)
            if case.is_moe:
                register_moe_requirement(
                    model_dir / "moe_kernel_only.csv",
                    "KERNEL_ONLY",
                    cluster_type=ClusterType.MONOLITHIC,
                    moe_tensor_parallel_size=case.moe_tensor_parallel_size,
                    moe_expert_parallel_size=case.ep_size,
                )
    elif case.architecture == "pd-disaggregation":
        eager_attention_tps.add(case.prefill_attn_tensor_parallel_size)
        eager_linear_tps.add(case.prefill_attn_tensor_parallel_size)
        if case.is_moe:
            eager_linear_tps.add(case.prefill_moe_tensor_parallel_size)
            eager_moe_keys.add(
                (
                    case.prefill_moe_tensor_parallel_size,
                    case.prefill_moe_expert_parallel_size,
                )
            )
            register_moe_requirement(
                model_dir / "moe.csv",
                "CUDA_EVENT",
                cluster_type=ClusterType.PREFILL,
                moe_tensor_parallel_size=case.prefill_moe_tensor_parallel_size,
                moe_expert_parallel_size=case.prefill_moe_expert_parallel_size,
            )
        if case.decode_cuda_graph_mode == "none":
            eager_attention_tps.add(case.decode_attn_tensor_parallel_size)
            eager_linear_tps.add(case.decode_attn_tensor_parallel_size)
            if case.is_moe:
                eager_linear_tps.add(case.decode_moe_tensor_parallel_size)
                eager_moe_keys.add(
                    (
                        case.decode_moe_tensor_parallel_size,
                        case.decode_moe_expert_parallel_size,
                    )
                )
                register_moe_requirement(
                    model_dir / "moe.csv",
                    "CUDA_EVENT",
                    cluster_type=ClusterType.DECODE,
                    moe_tensor_parallel_size=case.decode_moe_tensor_parallel_size,
                    moe_expert_parallel_size=case.decode_moe_expert_parallel_size,
                )
        else:
            kernel_attention_tps.add(case.decode_attn_tensor_parallel_size)
            kernel_linear_tps.add(case.decode_attn_tensor_parallel_size)
            if case.is_moe:
                kernel_linear_tps.add(case.decode_moe_tensor_parallel_size)
                kernel_moe_keys.add(
                    (
                        case.decode_moe_tensor_parallel_size,
                        case.decode_moe_expert_parallel_size,
                    )
                )
                register_moe_requirement(
                    model_dir / "moe_kernel_only.csv",
                    "KERNEL_ONLY",
                    cluster_type=ClusterType.DECODE,
                    moe_tensor_parallel_size=case.decode_moe_tensor_parallel_size,
                    moe_expert_parallel_size=case.decode_moe_expert_parallel_size,
                )
    elif case.architecture == "pd-af-disaggregation":
        eager_attention_tps.update(
            {
                case.prefill_attn_tensor_parallel_size,
                case.decode_attn_tensor_parallel_size,
            }
        )
        eager_linear_tps.update(eager_attention_tps)
        kernel_attention_tps.add(case.decode_attn_tensor_parallel_size)
        kernel_linear_tps.add(case.decode_attn_tensor_parallel_size)
        decode_ffn_full_world_tp = (
            case.decode_moe_tensor_parallel_size
            * case.decode_moe_expert_parallel_size
        )
        kernel_linear_tps.add(decode_ffn_full_world_tp)
        if case.is_moe:
            eager_linear_tps.add(case.prefill_moe_tensor_parallel_size)
            kernel_linear_tps.add(case.decode_moe_tensor_parallel_size)
            eager_moe_keys.add(
                (
                    case.prefill_moe_tensor_parallel_size,
                    case.prefill_moe_expert_parallel_size,
                )
            )
            register_moe_requirement(
                model_dir / "moe.csv",
                "CUDA_EVENT",
                cluster_type=ClusterType.PREFILL,
                moe_tensor_parallel_size=case.prefill_moe_tensor_parallel_size,
                moe_expert_parallel_size=case.prefill_moe_expert_parallel_size,
            )
            kernel_moe_keys.add(
                (
                    case.decode_moe_tensor_parallel_size,
                    case.decode_moe_expert_parallel_size,
                )
            )
            register_moe_requirement(
                model_dir / "moe_kernel_only.csv",
                "KERNEL_ONLY",
                cluster_type=ClusterType.DECODE_FFN,
                moe_tensor_parallel_size=case.decode_moe_tensor_parallel_size,
                moe_expert_parallel_size=case.decode_moe_expert_parallel_size,
            )
    else:
        raise ValueError(f"unsupported architecture: {case.architecture}")

    profile_requirements: dict[Path, tuple[str, set[int]]] = {
        model_dir / "attention.csv": ("CUDA_EVENT", eager_attention_tps),
        model_dir / "linear_op.csv": ("CUDA_EVENT", eager_linear_tps),
    }
    if eager_moe_keys:
        profile_requirements[model_dir / "moe.csv"] = (
            "CUDA_EVENT",
            {tp for tp, _ep in eager_moe_keys},
        )
    if kernel_attention_tps:
        profile_requirements[model_dir / "attention_kernel_only.csv"] = (
            "KERNEL_ONLY",
            kernel_attention_tps,
        )
    if kernel_linear_tps:
        profile_requirements[model_dir / "linear_op_kernel_only.csv"] = (
            "KERNEL_ONLY",
            kernel_linear_tps,
        )
    if kernel_moe_keys:
        profile_requirements[model_dir / "moe_kernel_only.csv"] = (
            "KERNEL_ONLY",
            {tp for tp, _ep in kernel_moe_keys},
        )

    required = list(profile_requirements)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required non-dummy profiling files:\n" + "\n".join(str(p) for p in missing)
        )
    for path, (measurement_type, tp_sizes) in profile_requirements.items():
        _validate_profile_metadata(path)
        _validate_profile_family_and_tp(
            case,
            path,
            expected_measurement_type=measurement_type,
            required_tp_sizes=tp_sizes,
        )

    for path, (_measurement_type, requirements) in moe_profile_requirements.items():
        _validate_moe_profile_keys(
            case,
            path,
            requirements=requirements,
        )

    if case.enable_mtp:
        _validate_target_embedded_mtp_columns(
            case,
            model_dir / "linear_op.csv",
            required_tp_sizes={case.attn_tensor_parallel_size},
        )
    return required


def _preflight_blocker(stage: str, exc: Exception) -> dict[str, str]:
    """Serialize one expected static-validation failure for the JSONL ledger."""

    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }


def preflight_cases(
    cases: Sequence[MatrixCase],
    repo_root: Path,
    output_root: Path,
    *,
    matrix_kind: str = "optimization",
) -> list[dict[str, Any]]:
    """Validate a matrix without launching any simulator process.

    The preflight ledger is deliberately independent from the runtime result
    ledger.  A case is READY only when its topology contract, real profiling
    inputs, and wrapper command all validate.  Every expected validation
    failure is retained as a structured blocker so a campaign can stop before
    spending GPU time on a late predictor or wrapper error.
    """

    if matrix_kind not in {"regression", "optimization"}:
        raise ValueError(f"unsupported matrix kind for preflight: {matrix_kind}")

    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    rows: list[dict[str, Any]] = []
    for case in cases:
        blockers: list[dict[str, str]] = []
        required_profile_paths: list[str] = []
        command = ""
        environment: dict[str, str] = {}

        try:
            validate_case_parallel_semantics(case)
            if matrix_kind == "optimization":
                validate_optimization_case(case)
        except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(_preflight_blocker("topology", exc))

        try:
            required_profile_paths = [
                str(path.resolve()) for path in validate_profile_inputs(case, repo_root)
            ]
        except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(_preflight_blocker("profile", exc))

        try:
            command, env = build_shell_command(case, repo_root, output_root)
            environment = {
                key: env[key]
                for key in (
                    "MODEL_NAME",
                    "ENABLE_DUMMY_MODE",
                    "SIMULATION_MODE",
                    "DECODE_CUDA_GRAPH_MODE",
                    "ENABLE_CUDA_GRAPH",
                    "ENABLE_CHUNKED_PREFILL",
                    "LONG_PREFILL_TOKEN_THRESHOLD",
                    "MAX_TOKENS_IN_BATCH",
                    "ENABLE_PREFIX_CACHING",
                    "ENABLE_MTP",
                    "MOE_ROUTING_DISTRIBUTION_TYPE",
                    "MOE_ROUTING_SEED",
                    "TOTAL_EXPERTS",
                    "ROUTER_TOPK",
                    "TRACE_FILE",
                    "SPEC_METHOD",
                    "MTP_N_PREDICT",
                    "MTP_NUM_LAYERS",
                )
                if key in env
            }
        except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(_preflight_blocker("command", exc))

        rows.append(
            {
                "case_id": case.case_id,
                "baseline_case_id": case.baseline_case_id,
                "architecture": case.architecture,
                "model_kind": case.model_kind,
                "model_name": case.model_name,
                "device": case.device,
                "total_cards": case.total_cards,
                "simulation_mode": case.simulation_mode,
                "optimization_stratum": case.optimization_stratum,
                "pair_id": case.pair_id,
                "comparison_group_id": case.comparison_group_id,
                "pair_role": case.pair_role,
                "required_profile_paths": required_profile_paths,
                "command": command,
                "environment": environment,
                "blockers": blockers,
                "status": "READY" if not blockers else "BLOCKED",
                "preflight_only": True,
            }
        )
    return rows


@lru_cache(maxsize=None)
def _read_profile_table(
    path: Path,
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        rows = tuple(dict(row) for row in reader)
    return fieldnames, rows


@lru_cache(maxsize=None)
def _validate_profile_metadata(path: Path) -> None:
    """Require the predictor's immutable metadata contract before a run."""

    fieldnames, rows = _read_profile_table(path)
    missing = sorted(set(REQUIRED_PROFILE_METADATA_COLUMNS) - set(fieldnames))
    if missing:
        raise ValueError(
            f"{path} missing required profiling metadata columns: {', '.join(missing)}"
        )
    if not rows:
        raise ValueError(f"{path} contains no profiling rows")
    empty = [
        column
        for column in REQUIRED_PROFILE_METADATA_COLUMNS
        if any(not str(row.get(column, "")).strip() for row in rows)
    ]
    if empty:
        raise ValueError(
            f"{path} contains empty required profiling metadata columns: {', '.join(empty)}"
        )


def _profile_int_values(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    column: str,
) -> set[int]:
    fieldnames, _cached_rows = _read_profile_table(path)
    if column not in fieldnames:
        raise ValueError(f"{path} missing required profiling column: {column}")
    values: set[int] = set()
    for row in rows:
        raw_value = str(row.get(column, "")).strip()
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{path} contains non-integer {column}={raw_value!r}"
            ) from exc
        values.add(value)
    return values


def _validate_profile_family_and_tp(
    case: MatrixCase,
    path: Path,
    *,
    expected_measurement_type: str,
    required_tp_sizes: set[int],
) -> None:
    fieldnames, rows = _read_profile_table(path)
    if "measurement_type" not in fieldnames:
        raise ValueError(f"{path} missing required profiling column: measurement_type")
    measurement_types = {
        str(row.get("measurement_type", "")).strip() for row in rows
    }
    if measurement_types != {expected_measurement_type}:
        raise ValueError(
            f"{case.case_id} requires measurement_type={expected_measurement_type} "
            f"in {path}, but found {sorted(measurement_types)!r}"
        )
    available_tp_sizes = _profile_int_values(
        path,
        rows,
        "num_tensor_parallel_workers",
    )
    missing_tp_sizes = sorted(required_tp_sizes - available_tp_sizes)
    if missing_tp_sizes:
        raise ValueError(
            f"{case.case_id} requires TP rows {missing_tp_sizes} in {path}; "
            f"available TP rows are {sorted(available_tp_sizes)}"
        )


def _validate_moe_profile_keys(
    case: MatrixCase,
    path: Path,
    *,
    requirements: set[_MoeProfileRequirement],
) -> None:
    fieldnames, rows = _read_profile_table(path)
    required_columns = {"num_tensor_parallel_workers", "expert_parallel_size"}
    missing_columns = sorted(required_columns - set(fieldnames))
    if missing_columns:
        raise ValueError(
            f"{path} missing required MoE profiling columns: "
            f"{', '.join(missing_columns)}"
        )

    normalized_rows: list[tuple[dict[str, str], int, int]] = []
    for row in rows:
        try:
            tp_size = int(str(row["num_tensor_parallel_workers"]).strip())
            ep_size = int(str(row["expert_parallel_size"]).strip())
        except ValueError as exc:
            raise ValueError(f"{path} contains non-integer MoE TP/EP metadata") from exc
        normalized_rows.append((row, tp_size, ep_size))

    available_keys = sorted(
        {
            (
                tp_size,
                ep_size,
                str(row.get("routing_runtime_path", "")).strip(),
                str(row.get(MOE_GATING_RUNTIME_CONTEXT_COLUMN, "")).strip(),
            )
            for row, tp_size, ep_size in normalized_rows
        }
    )
    missing_requirements: list[str] = []
    for requirement in sorted(
        requirements,
        key=lambda item: (
            item.cluster_type.value,
            item.moe_tensor_parallel_size,
            item.moe_expert_parallel_size,
            item.routing_runtime_path,
        ),
    ):
        for operator in MOE_FAMILY.profiling_ops():
            op_name = operator.profiling_name()
            tp_key = resolve_moe_operator_tp_key(
                op_name,
                moe_tp_size=requirement.moe_tensor_parallel_size,
                cluster_type=requirement.cluster_type,
                family=MOE_FAMILY,
            )
            ep_agnostic = is_moe_operator_ep_agnostic(
                op_name,
                family=MOE_FAMILY,
            )
            candidates = [
                row
                for row, tp_size, ep_size in normalized_rows
                if tp_size == tp_key
                and (ep_agnostic or ep_size == requirement.moe_expert_parallel_size)
            ]
            requirement_parts = [
                f"cluster={requirement.cluster_type.name}",
                f"TP={tp_key}",
                (
                    "EP=ANY"
                    if ep_agnostic
                    else f"EP={requirement.moe_expert_parallel_size}"
                ),
            ]
            if op_name == "moe_gating_routing_topk":
                requirement_parts.append(
                    f"routing_runtime_path={requirement.routing_runtime_path}"
                )
                if "routing_runtime_path" not in fieldnames:
                    missing_requirements.append(
                        f"{op_name} requires {', '.join(requirement_parts)} "
                        "but routing_runtime_path column is missing"
                    )
                    continue
                candidates = [
                    row
                    for row in candidates
                    if str(row.get("routing_runtime_path", "")).strip()
                    == requirement.routing_runtime_path
                ]
            if operator.precision_name() == "moe_gating":
                requirement_parts.append(
                    "gating_runtime_context="
                    f"{DEFAULT_MOE_GATING_RUNTIME_CONTEXT}"
                )
                if MOE_GATING_RUNTIME_CONTEXT_COLUMN not in fieldnames:
                    missing_requirements.append(
                        f"{op_name} requires {', '.join(requirement_parts)} "
                        f"but {MOE_GATING_RUNTIME_CONTEXT_COLUMN} column is missing"
                    )
                    continue
                candidates = [
                    row
                    for row in candidates
                    if str(row.get(MOE_GATING_RUNTIME_CONTEXT_COLUMN, "")).strip()
                    == DEFAULT_MOE_GATING_RUNTIME_CONTEXT
                ]

            target_column = f"time_stats.{op_name}.median"
            requirement_parts.append(f"target={target_column}")
            if target_column not in fieldnames:
                missing_requirements.append(
                    f"{op_name} requires {', '.join(requirement_parts)} "
                    "but target column is missing"
                )
                continue
            has_valid_target = False
            for row in candidates:
                raw_value = str(row.get(target_column, "")).strip()
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(value) and value >= 0:
                    has_valid_target = True
                    break
            if not has_valid_target:
                missing_requirements.append(
                    f"{op_name} requires {', '.join(requirement_parts)} "
                    "with at least one finite non-negative value"
                )

    if missing_requirements:
        requirement_text = "\n  - ".join(missing_requirements)
        raise ValueError(
            f"{case.case_id} MoE profile contract failed before training in {path}.\n"
            "Missing op-level TP/EP/routing/context/timing coverage:\n"
            f"  - {requirement_text}\n"
            f"Available TP/EP/routing/context keys are {available_keys}"
        )


def _validate_target_embedded_mtp_columns(
    case: MatrixCase,
    path: Path,
    *,
    required_tp_sizes: set[int],
) -> None:
    fieldnames, rows = _read_profile_table(path)
    required_columns = (
        *TARGET_EMBEDDED_MTP_COLUMNS,
        *TARGET_EMBEDDED_MTP_SAME_TP_COLUMNS,
    )
    missing_columns = [
        column for column in required_columns if column not in fieldnames
    ]
    non_finite_by_tp: dict[int, list[str]] = {}
    for tp_size in sorted(required_tp_sizes):
        relevant_rows = [
            row
            for row in rows
            if int(str(row["num_tensor_parallel_workers"]).strip()) == tp_size
        ]
        invalid_columns = []
        for column in required_columns:
            if column not in fieldnames:
                continue
            for row in relevant_rows:
                raw_value = str(row.get(column, "")).strip()
                try:
                    value = float(raw_value)
                except ValueError:
                    invalid_columns.append(column)
                    break
                if not math.isfinite(value) or value < 0:
                    invalid_columns.append(column)
                    break
        if invalid_columns:
            non_finite_by_tp[tp_size] = invalid_columns
    if missing_columns or non_finite_by_tp:
        mtp_columns = ", ".join(TARGET_EMBEDDED_MTP_COLUMNS)
        same_tp_columns = ", ".join(TARGET_EMBEDDED_MTP_SAME_TP_COLUMNS)
        raise ValueError(
            f"{case.case_id} requires finite target-embedded MTP columns "
            f"{mtp_columns} and finite same-TP columns {same_tp_columns} "
            f"for TP sizes {sorted(required_tp_sizes)} in {path}; "
            f"missing={missing_columns}, non_finite_by_tp={non_finite_by_tp}"
        )


def _find_metrics_dir(
    output_root: Path,
    case: MatrixCase,
    *,
    started_at_ns: int | None = None,
) -> Path:
    root = output_root / case.case_id
    candidates = sorted(root.rglob("system_metrics.json"))
    if started_at_ns is not None:
        candidates = [
            path
            for path in candidates
            if path.stat().st_mtime_ns >= started_at_ns
        ]
        if not candidates:
            raise FileNotFoundError(
                f"no fresh system_metrics.json for {case.case_id} "
                f"under {output_root}"
            )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one system_metrics.json for {case.case_id}, found {len(candidates)}"
        )
    return candidates[0].parent


def _finite_metric_values(value: Any) -> Iterable[float]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _finite_metric_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _finite_metric_values(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"non-finite or negative metric value: {value!r}")
        yield number


def _read_request_metrics_rows(metrics_dir: Path) -> list[dict[str, str]]:
    path = metrics_dir / "request_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing request metrics file: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"request metrics file contains no rows: {path}")
    return rows


def _read_chunked_prefill_stage_ledger(
    case: MatrixCase,
    metrics_dir: Path,
) -> dict[str, Any]:
    path = metrics_dir / "frontier_stage_batch_ledger.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing stage-batch ledger: {path}")
    expected_cluster = {
        "co-location": "MONOLITHIC",
        "pd-disaggregation": "PREFILL",
        "pd-af-disaggregation": "PREFILL",
    }[case.architecture]
    prefill_components = (
        "attention_prefill_execution_time",
        "attn_mla_prefill_time",
    )
    grouped_payloads: dict[
        tuple[str, int, int],
        dict[
            int,
            tuple[
                tuple[str, ...],
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
            ],
        ],
    ] = {}
    prefill_row_count = 0

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid stage-batch ledger JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise ValueError(
                f"stage-batch ledger row is not an object at {path}:{line_number}"
            )
        if row.get("cluster_type") != expected_cluster:
            continue
        execution_time = row.get("execution_time")
        if not isinstance(execution_time, Mapping):
            continue
        component_ledger = execution_time.get("component_ledger_ms")
        if not isinstance(component_ledger, Mapping):
            continue
        prefill_component_total = 0.0
        for component in prefill_components:
            raw_value = component_ledger.get(component, 0.0)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(
                    f"{component} must be numeric at {path}:{line_number}"
                )
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{component} must be finite and non-negative "
                    f"at {path}:{line_number}"
                )
            prefill_component_total += value
        if prefill_component_total <= 0:
            continue
        integer_fields: dict[str, int] = {}
        for field in ("batch_id", "stage_id", "replica_id"):
            raw_value = row.get(field)
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise ValueError(
                    f"{field} must be an integer at {path}:{line_number}"
                )
            if raw_value < 0:
                raise ValueError(
                    f"{field} must be non-negative at {path}:{line_number}"
                )
            integer_fields[field] = raw_value

        request_ids = row.get("request_ids")
        request_runtime_epochs = row.get("request_runtime_epochs")
        request_num_tokens = row.get("request_num_tokens")
        request_num_prefill_tokens = row.get("request_num_prefill_tokens")
        if not isinstance(request_ids, list) or not isinstance(
            request_runtime_epochs,
            list,
        ) or not isinstance(
            request_num_tokens,
            list,
        ) or not isinstance(
            request_num_prefill_tokens,
            list,
        ):
            raise ValueError(
                "request_ids, request_num_tokens, and "
                "request_num_prefill_tokens must be lists "
                f"at {path}:{line_number}"
            )
        if (
            not request_ids
            or len(request_ids) != len(request_runtime_epochs)
            or len(request_ids) != len(request_num_tokens)
            or len(request_ids) != len(request_num_prefill_tokens)
        ):
            raise ValueError(
                "request_ids, request_runtime_epochs, request_num_tokens, and "
                "request_num_prefill_tokens must be non-empty and aligned "
                f"at {path}:{line_number}"
            )
        normalized_request_ids = tuple(str(request_id) for request_id in request_ids)
        if any(not request_id for request_id in normalized_request_ids):
            raise ValueError(f"request_ids must be non-empty at {path}:{line_number}")
        if len(set(normalized_request_ids)) != len(normalized_request_ids):
            raise ValueError(
                f"request_ids must be unique within a batch at {path}:{line_number}"
            )
        normalized_runtime_epochs: list[int] = []
        for runtime_epoch in request_runtime_epochs:
            if isinstance(runtime_epoch, bool) or not isinstance(runtime_epoch, int):
                raise ValueError(
                    "request_runtime_epochs must contain integers "
                    f"at {path}:{line_number}"
                )
            if runtime_epoch < 0:
                raise ValueError(
                    "request_runtime_epochs must be non-negative "
                    f"at {path}:{line_number}"
                )
            normalized_runtime_epochs.append(runtime_epoch)
        normalized_token_counts: list[int] = []
        normalized_prefill_token_counts: list[int] = []
        for token_count, prefill_token_count in zip(
            request_num_tokens,
            request_num_prefill_tokens,
        ):
            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise ValueError(
                    "request_num_tokens must contain integers "
                    f"at {path}:{line_number}"
                )
            if token_count <= 0 or token_count > case.prefill_tokens:
                raise ValueError(
                    "prefill chunks must be positive and no larger than the prompt "
                    f"at {path}:{line_number}: token_count={token_count}, "
                    f"prefill_tokens={case.prefill_tokens}"
                )
            if (
                isinstance(prefill_token_count, bool)
                or not isinstance(prefill_token_count, int)
            ):
                raise ValueError(
                    "request_num_prefill_tokens must contain integers "
                    f"at {path}:{line_number}"
                )
            if (
                prefill_token_count < 0
                or prefill_token_count > token_count
                or prefill_token_count > case.prefill_tokens
            ):
                raise ValueError(
                    "request prefill tokens must be non-negative and no larger "
                    "than the request batch tokens or prompt "
                    f"at {path}:{line_number}: "
                    f"prefill_token_count={prefill_token_count}, "
                    f"token_count={token_count}, "
                    f"prefill_tokens={case.prefill_tokens}"
                )
            normalized_token_counts.append(token_count)
            normalized_prefill_token_counts.append(prefill_token_count)
        if not any(normalized_prefill_token_counts):
            # Speculative verify batches may use the prefill-style attention
            # kernel while all requests are already in decode. They are not
            # prompt-prefill chunks and must not enter this ledger.
            continue

        prefill_row_count += 1
        if (
            row.get("execution_scope") != "FULL_STAGE_WORLD"
            or row.get("replica_local_id") is not None
        ):
            raise ValueError(
                "prefill stage-batch ledger row must use FULL_STAGE_WORLD "
                f"at {path}:{line_number}"
            )

        group_key = (
            expected_cluster,
            integer_fields["replica_id"],
            integer_fields["batch_id"],
        )
        stage_payloads = grouped_payloads.setdefault(group_key, {})
        stage_id = integer_fields["stage_id"]
        if stage_id in stage_payloads:
            raise ValueError(
                "duplicate pipeline-stage ledger row "
                f"cluster={group_key[0]} replica={group_key[1]} "
                f"batch={group_key[2]} stage={stage_id}"
            )
        stage_payloads[stage_id] = (
            normalized_request_ids,
            tuple(normalized_runtime_epochs),
            tuple(normalized_token_counts),
            tuple(normalized_prefill_token_counts),
        )

    expected_stage_ids = set(range(case.pipeline_stages))
    request_chunks: dict[str, list[tuple[int, int]]] = {}
    for group_key, stage_payloads in sorted(grouped_payloads.items()):
        actual_stage_ids = set(stage_payloads)
        if actual_stage_ids != expected_stage_ids:
            raise ValueError(
                "pipeline-stage coverage mismatch "
                f"cluster={group_key[0]} replica={group_key[1]} "
                f"batch={group_key[2]} expected={sorted(expected_stage_ids)} "
                f"actual={sorted(actual_stage_ids)}"
            )
        ordered_payloads = [
            stage_payloads[stage_id] for stage_id in sorted(stage_payloads)
        ]
        first_payload = ordered_payloads[0]
        if any(payload != first_payload for payload in ordered_payloads[1:]):
            raise ValueError(
                "pipeline-stage payload mismatch "
                f"cluster={group_key[0]} replica={group_key[1]} "
                f"batch={group_key[2]}"
            )
        request_ids, runtime_epochs, _token_counts, prefill_token_counts = first_payload
        for request_id, runtime_epoch, prefill_token_count in zip(
            request_ids,
            runtime_epochs,
            prefill_token_counts,
        ):
            if prefill_token_count > 0:
                request_chunks.setdefault(request_id, []).append(
                    (runtime_epoch, prefill_token_count)
                )

    if request_chunks and not case.enable_prefix_caching:
        if len(request_chunks) != case.num_requests:
            raise ValueError(
                "prefill request coverage mismatch "
                f"expected={case.num_requests} actual={len(request_chunks)}"
            )

    request_token_totals = {
        request_id: sum(token_count for _runtime_epoch, token_count in chunks)
        for request_id, chunks in sorted(request_chunks.items())
    }
    request_recompute_token_totals: dict[str, int] = {}
    request_prefill_preemption_counts: dict[str, int] = {}
    request_final_epoch_token_totals: dict[str, int] = {}
    request_rows_by_id: dict[str, Mapping[str, str]] = {}
    if any(total > case.prefill_tokens for total in request_token_totals.values()):
        for request_row in _read_request_metrics_rows(metrics_dir):
            request_id = str(request_row.get("Request Id", "")).strip()
            if request_id:
                request_rows_by_id[request_id] = request_row
    for request_id, total in request_token_totals.items():
        request_epochs = {}
        for runtime_epoch, token_count in request_chunks[request_id]:
            request_epochs[runtime_epoch] = (
                request_epochs.get(runtime_epoch, 0) + token_count
            )
        request_row = request_rows_by_id.get(request_id)
        final_epoch = max(request_epochs)
        request_final_epoch_token_totals[request_id] = request_epochs[final_epoch]
        preemption_count = 0
        if request_row is not None:
            preemption_count = _nonnegative_integer_metric(
                request_row,
                "request_prefill_preemption_count",
            )
        request_prefill_preemption_counts[request_id] = preemption_count
        if total > case.prefill_tokens:
            if request_row is None:
                raise ValueError(
                    "prefill-token conservation mismatch "
                    f"request_id={request_id} exceeds prompt without request metrics "
                    f"expected={case.prefill_tokens} actual={total}"
                )
            if len(request_epochs) != preemption_count + 1:
                raise ValueError(
                    "prefill recompute epoch count mismatch "
                    f"request_id={request_id} expected={preemption_count + 1} "
                    f"actual={len(request_epochs)}"
                )
            if request_epochs[final_epoch] != case.prefill_tokens:
                raise ValueError(
                    "prefill recompute final epoch mismatch "
                    f"request_id={request_id} expected={case.prefill_tokens} "
                    f"actual={request_epochs[final_epoch]}"
                )
            prior_epoch_total = sum(
                token_count
                for runtime_epoch, token_count in request_epochs.items()
                if runtime_epoch != final_epoch
            )
            if prior_epoch_total <= 0 or any(
                token_count >= case.prefill_tokens
                for runtime_epoch, token_count in request_epochs.items()
                if runtime_epoch != final_epoch
            ):
                raise ValueError(
                    "prefill recompute prior epoch must contain partial work "
                    f"request_id={request_id} epochs={request_epochs}"
                )
            request_recompute_token_totals[request_id] = prior_epoch_total
        elif not case.enable_prefix_caching and total != case.prefill_tokens:
            raise ValueError(
                "prefill-token conservation mismatch "
                f"request_id={request_id} expected={case.prefill_tokens} "
                f"actual={total}"
            )
        elif preemption_count > 0:
            raise ValueError(
                "prefill preemption has no recompute evidence "
                f"request_id={request_id} preemption_count={preemption_count}"
            )
    split_requests = {
        request_id: chunks
        for request_id, chunks in request_chunks.items()
        if len(chunks) > 1
    }
    full_conserving_split_requests = [
        request_id
        for request_id, chunks in split_requests.items()
        if request_final_epoch_token_totals[request_id] == case.prefill_tokens
    ]
    return {
        "chunked_prefill_stage_ledger_present": True,
        "chunked_prefill_prefill_row_count": prefill_row_count,
        "chunked_prefill_prefill_batch_count": len(grouped_payloads),
        "chunked_prefill_request_count": len(request_chunks),
        "chunked_prefill_split_request_count": len(split_requests),
        "chunked_prefill_split_count": sum(
            len(chunks) - 1 for chunks in split_requests.values()
        ),
        "chunked_prefill_full_conserving_split_request_count": len(
            full_conserving_split_requests
        ),
        "chunked_prefill_request_token_totals": request_token_totals,
        "chunked_prefill_request_final_epoch_token_totals": (
            request_final_epoch_token_totals
        ),
        "chunked_prefill_request_recompute_token_totals": request_recompute_token_totals,
        "chunked_prefill_prefill_preemption_count": sum(
            request_prefill_preemption_counts.values()
        ),
    }


def _nonnegative_integer_metric(
    row: Mapping[str, str],
    field: str,
) -> int:
    raw_value = str(row.get(field, "")).strip()
    if not raw_value:
        raise ValueError(f"request metrics row is missing {field}")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"request metrics {field} is not numeric: {raw_value!r}") from exc
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError(
            f"request metrics {field} must be a finite non-negative integer: {raw_value!r}"
        )
    return int(value)


def _check_optimization_activation(
    case: MatrixCase,
    text: str,
    metrics_dir: Path,
    metrics: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Require runtime evidence for every enabled optimization surface."""

    errors: list[str] = []
    evidence: dict[str, Any] = {}
    request_rows: list[dict[str, str]] | None = None

    def _request_rows() -> list[dict[str, str]]:
        nonlocal request_rows
        if request_rows is None:
            request_rows = _read_request_metrics_rows(metrics_dir)
        return request_rows

    if case.simulation_mode == "online":
        try:
            delays = []
            for row in _request_rows():
                raw_value = str(row.get("request_inter_arrival_delay", "")).strip()
                if not raw_value:
                    continue
                value = float(raw_value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        "request_inter_arrival_delay must be finite and non-negative"
                    )
                delays.append(value)
            positive_delays = [value for value in delays if value > 0]
            evidence["online_positive_inter_arrival_count"] = len(positive_delays)
            if not positive_delays:
                errors.append(
                    "online activation requires a finite positive "
                    "request_inter_arrival_delay"
                )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"online activation evidence invalid: {exc}")

    if case.enable_prefix_caching:
        try:
            stats = metrics.get("prefix_cache_statistics")
            if not isinstance(stats, Mapping):
                raise ValueError("missing prefix_cache_statistics")
            total_hit_blocks = _nonnegative_integer_metric(
                {key: str(value) for key, value in stats.items()},
                "total_hit_blocks",
            )
            request_hit_blocks = sum(
                _nonnegative_integer_metric(row, "request_prefix_cache_hit_blocks")
                for row in _request_rows()
            )
            evidence["prefix_cache_hit_blocks"] = total_hit_blocks
            evidence["prefix_cache_request_hit_blocks"] = request_hit_blocks
            if total_hit_blocks <= 0:
                errors.append(
                    "Prefix Cache activation requires total_hit_blocks > 0"
                )
            if request_hit_blocks != total_hit_blocks:
                errors.append(
                    "Prefix Cache activation totals disagree "
                    f"system={total_hit_blocks} request={request_hit_blocks}"
                )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"Prefix Cache activation evidence invalid: {exc}")

    if case.enable_mtp:
        try:
            stats = metrics.get("spec_decode_statistics")
            if not isinstance(stats, Mapping):
                raise ValueError("missing spec_decode_statistics")
            total_iterations = _nonnegative_integer_metric(
                {key: str(value) for key, value in stats.items()},
                "total_iterations",
            )
            total_committed_tokens = _nonnegative_integer_metric(
                {key: str(value) for key, value in stats.items()},
                "total_committed_tokens",
            )
            request_iterations = sum(
                _nonnegative_integer_metric(row, "request_spec_total_iterations")
                for row in _request_rows()
            )
            request_committed_tokens = sum(
                _nonnegative_integer_metric(row, "request_spec_committed_tokens")
                for row in _request_rows()
            )
            evidence["spec_decode_iterations"] = total_iterations
            evidence["spec_decode_committed_tokens"] = total_committed_tokens
            evidence["spec_decode_request_iterations"] = request_iterations
            evidence["spec_decode_request_committed_tokens"] = request_committed_tokens
            if total_iterations <= 0 or total_committed_tokens <= 0:
                errors.append(
                    "MTP activation requires positive total_iterations and "
                    "total_committed_tokens"
                )
            if request_iterations != total_iterations:
                errors.append(
                    "MTP iteration totals disagree "
                    f"system={total_iterations} request={request_iterations}"
                )
            if request_committed_tokens != total_committed_tokens:
                errors.append(
                    "MTP committed-token totals disagree "
                    f"system={total_committed_tokens} request={request_committed_tokens}"
                )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"MTP activation evidence invalid: {exc}")

    stage_ledger_path = metrics_dir / "frontier_stage_batch_ledger.jsonl"
    if case.enable_chunked_prefill or stage_ledger_path.is_file():
        try:
            chunked_prefill_evidence = _read_chunked_prefill_stage_ledger(
                case,
                metrics_dir,
            )
            evidence.update(chunked_prefill_evidence)
            split_count = int(
                chunked_prefill_evidence["chunked_prefill_split_count"]
            )
            full_conserving_split_count = int(
                chunked_prefill_evidence[
                    "chunked_prefill_full_conserving_split_request_count"
                ]
            )
            if case.enable_chunked_prefill:
                if split_count <= 0:
                    errors.append(
                        "Chunked Prefill activation requires multiple positive "
                        "prefill chunks for at least one request"
                    )
                elif full_conserving_split_count <= 0:
                    errors.append(
                        "Chunked Prefill activation requires at least one split "
                        "request with exact prefill-token conservation"
                    )
            elif split_count > 0:
                errors.append("Chunked Prefill control unexpectedly split a request")
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            evidence["chunked_prefill_stage_ledger_present"] = False
            evidence["chunked_prefill_split_count"] = 0
            errors.append(f"Chunked Prefill stage-ledger evidence invalid: {exc}")

    graph_enabled = (
        case.use_cuda_graph or case.decode_cuda_graph_mode != "none"
    )
    activation_marker = "[CUDA-GRAPH-ACTIVATION]"
    if not graph_enabled and activation_marker in text:
        errors.append("CUDA Graph control unexpectedly activated")
    if graph_enabled:
        expected_mode = (
            "GLOBAL"
            if case.use_cuda_graph
            else {
                "full_decode_only": "FULL",
                "piecewise": "PIECEWISE",
            }[case.decode_cuda_graph_mode]
        )
        expected_config_mode = (
            "global" if case.use_cuda_graph else case.decode_cuda_graph_mode
        )
        expected_roles = (
            ("MONOLITHIC",)
            if case.architecture == "co-location"
            else (
                ("DECODE",)
                if case.architecture == "pd-disaggregation"
                else ("DECODE_ATTN", "DECODE_FFN")
            )
        )
        try:
            matching_captures: list[Mapping[str, Any]] = []
            invalid_records: list[str] = []
            for line in text.splitlines():
                if activation_marker not in line:
                    continue
                payload_text = line.partition(activation_marker)[2].strip()
                capture = json.loads(payload_text)
                if not isinstance(capture, Mapping):
                    raise ValueError("activation record must be an object")
                role = capture.get("cluster_role")
                if role not in expected_roles:
                    invalid_records.append(f"unexpected cluster_role={role!r}")
                    continue
                if capture.get("config_mode") != expected_config_mode:
                    invalid_records.append(
                        f"{role} config_mode={capture.get('config_mode')!r}"
                    )
                    continue
                if capture.get("runtime_mode") != expected_mode:
                    invalid_records.append(
                        f"{role} runtime_mode={capture.get('runtime_mode')!r}"
                    )
                    continue
                if capture.get("capture_hit") is not True:
                    invalid_records.append(f"{role} capture_hit is not true")
                    continue
                if capture.get("measurement_family") != "kernel_only":
                    invalid_records.append(
                        f"{role} measurement_family="
                        f"{capture.get('measurement_family')!r}"
                    )
                    continue
                capture_sizes = capture.get("capture_sizes")
                original_tokens = capture.get("original_tokens")
                padded_tokens = capture.get("padded_tokens")
                if not isinstance(capture_sizes, list) or not capture_sizes:
                    raise ValueError("capture_sizes must be a non-empty list")
                if not isinstance(original_tokens, list) or not original_tokens:
                    raise ValueError("original_tokens must be a non-empty list")
                if not isinstance(padded_tokens, list) or not padded_tokens:
                    raise ValueError("padded_tokens must be a non-empty list")
                if not (
                    len(capture_sizes)
                    == len(original_tokens)
                    == len(padded_tokens)
                ):
                    raise ValueError(
                        "capture_sizes/original_tokens/padded_tokens lengths differ"
                    )
                for capture_size, original, padded in zip(
                    capture_sizes,
                    original_tokens,
                    padded_tokens,
                ):
                    if type(capture_size) is not int or capture_size <= 0:
                        raise ValueError(
                            "capture_sizes must contain positive integers"
                        )
                    if type(original) is not int or original < 0:
                        raise ValueError(
                            "original_tokens must contain non-negative integers"
                        )
                    if type(padded) is not int or padded < original:
                        raise ValueError(
                            "padded_tokens must contain integers >= original_tokens"
                        )
                    if capture_size != padded:
                        raise ValueError(
                            "capture_sizes must equal the selected padded_tokens"
                        )
                matching_captures.append(capture)
            matching_roles = {
                str(capture["cluster_role"]) for capture in matching_captures
            }
            evidence["cuda_graph_capture_count"] = len(matching_captures)
            evidence["cuda_graph_capture_roles"] = [
                role for role in expected_roles if role in matching_roles
            ]
            missing_roles = [
                role for role in expected_roles if role not in matching_roles
            ]
            if missing_roles:
                invalid_detail = (
                    f"; invalid records: {', '.join(invalid_records)}"
                    if invalid_records
                    else ""
                )
                errors.append(
                    "CUDA Graph activation requires production runtime records "
                    f"for roles={missing_roles}, runtime_mode={expected_mode}, "
                    f"measurement_family='kernel_only'{invalid_detail}"
                )
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"CUDA Graph activation evidence invalid: {exc}")

    return errors, evidence


def _parse_ep_workload_records(text: str) -> list[dict[str, Any]]:
    """Parse scheduler-emitted per-lane MoE workload records.

    The parser is deliberately strict: malformed records are ignored by the
    caller only as missing evidence, while a syntactically matching record is
    validated here so a negative/non-finite lane value cannot be mistaken for
    a successful workflow trace.
    """

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _EP_WORKLOAD_LINE_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        try:
            per_expert_tokens = ast.literal_eval(groups["per_expert_tokens"])
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "invalid per_expert_tokens literal in EP workload trace"
            ) from exc
        if not isinstance(per_expert_tokens, dict):
            raise ValueError("EP workload per_expert_tokens must be a dict")
        normalized_tokens: dict[int, int] = {}
        for expert_id, token_count in per_expert_tokens.items():
            if (
                type(expert_id) is not int
                or type(token_count) is not int
                or token_count < 0
            ):
                raise ValueError(
                    "EP workload per_expert_tokens must map exact integer expert IDs "
                    "to non-negative integer token counts"
                )
            normalized_tokens[int(expert_id)] = int(token_count)

        lane_compute_ms = float(groups["lane_compute_ms"])
        lane_comm_ms = float(groups["lane_comm_ms"])
        ep_id = int(groups["ep_id"])
        moe_ep_size = int(groups["moe_ep_size"])
        if moe_ep_size <= 0 or ep_id < 0 or ep_id >= moe_ep_size:
            raise ValueError(
                "EP workload ep_id must be within the declared moe_ep_size"
            )
        if not math.isfinite(lane_compute_ms) or lane_compute_ms < 0:
            raise ValueError("EP workload lane_compute_ms must be finite and non-negative")
        if not math.isfinite(lane_comm_ms) or lane_comm_ms < 0:
            raise ValueError("EP workload lane_comm_ms must be finite and non-negative")

        records.append(
            {
                "cluster": groups["cluster"],
                "batch_id": int(groups["batch_id"]),
                "layer_id": int(groups["layer_id"]),
                "ep_id": ep_id,
                "moe_ep_size": moe_ep_size,
                "per_expert_tokens": normalized_tokens,
                "lane_compute_ms": lane_compute_ms,
                "lane_comm_ms": lane_comm_ms,
            }
        )
    return records


def _expected_ep_size_for_cluster(case: MatrixCase, cluster: str) -> int | None:
    """Return the EP cardinality for one physical cluster role."""

    cluster_name = str(cluster).upper()
    if case.architecture == "co-location":
        return int(case.ep_size) if cluster_name == "MONOLITHIC" else None
    if case.architecture == "pd-af-disaggregation":
        if cluster_name == "PREFILL":
            return int(case.prefill_moe_expert_parallel_size)
        if cluster_name == "DECODE_FFN":
            return int(case.decode_moe_expert_parallel_size)
        return None
    if case.architecture == "pd-disaggregation":
        if cluster_name == "PREFILL":
            return int(case.prefill_moe_expert_parallel_size)
        if cluster_name == "DECODE":
            return int(case.decode_moe_expert_parallel_size)
        return None
    return None


def _expected_ep_roles(case: MatrixCase) -> tuple[str, ...]:
    """Return every cluster role that must execute each MoE layer."""

    if case.architecture == "co-location":
        return ("MONOLITHIC",)
    if case.architecture == "pd-disaggregation":
        return ("PREFILL", "DECODE")
    if case.architecture == "pd-af-disaggregation":
        return ("PREFILL", "DECODE_FFN")
    raise ValueError(f"unsupported architecture: {case.architecture}")


def _ep_event_positions(
    text: str,
) -> dict[tuple[str, int, int], dict[str, list[int]]]:
    """Index strict EP evidence by wave and log-line order."""

    positions: dict[tuple[str, int, int], dict[str, list[int]]] = {}
    event_patterns = (
        ("workload", _EP_WORKLOAD_LINE_RE),
        ("conservation", _EP_CONSERVATION_LINE_RE),
        ("barrier", _EP_BARRIER_LINE_RE),
    )
    for line_index, line in enumerate(text.splitlines()):
        for event_kind, pattern in event_patterns:
            match = pattern.search(line)
            if match is None:
                continue
            groups = match.groupdict()
            wave_key = (
                str(groups["cluster"]).upper(),
                int(groups["batch_id"]),
                int(groups["layer_id"]),
            )
            phase = str(groups["phase"]) if event_kind == "barrier" else event_kind
            positions.setdefault(wave_key, {}).setdefault(phase, []).append(
                line_index
            )
            break
    return positions


def _parse_ep_barrier_records(text: str) -> list[dict[str, Any]]:
    """Parse completed per-layer EP barrier records from a simulator log."""

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _EP_BARRIER_LINE_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        try:
            expected_ep_ids = ast.literal_eval(groups["expected_ep_ids"])
            arrived_ep_ids = ast.literal_eval(groups["arrived_ep_ids"])
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid EP barrier participant list") from exc
        if not isinstance(expected_ep_ids, list) or not isinstance(arrived_ep_ids, list):
            raise ValueError("EP barrier participant lists must be lists")
        if any(type(ep_id) is not int or ep_id < 0 for ep_id in expected_ep_ids):
            raise ValueError("EP barrier expected_ep_ids must be non-negative ints")
        if any(type(ep_id) is not int or ep_id < 0 for ep_id in arrived_ep_ids):
            raise ValueError("EP barrier arrived_ep_ids must be non-negative ints")
        if sorted(expected_ep_ids) != sorted(set(expected_ep_ids)):
            raise ValueError("EP barrier expected_ep_ids must be unique")
        if sorted(arrived_ep_ids) != sorted(set(arrived_ep_ids)):
            raise ValueError("EP barrier arrived_ep_ids must be unique")
        if sorted(expected_ep_ids) != sorted(arrived_ep_ids):
            raise ValueError("EP barrier arrived_ep_ids must equal expected_ep_ids")
        values = {
            name: float(groups[name])
            for name in ("max_lane_time_ms", "barrier_time_ms", "barrier_end_time_s")
        }
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("EP barrier times must be finite and non-negative")
        if values["barrier_time_ms"] < values["max_lane_time_ms"]:
            raise ValueError("EP barrier time is shorter than the slowest lane")
        batch_id = int(groups["batch_id"])
        layer_id = int(groups["layer_id"])
        if batch_id < 0 or layer_id < 0:
            raise ValueError("EP barrier batch_id/layer_id must be non-negative")
        records.append(
            {
                "cluster": groups["cluster"],
                "batch_id": batch_id,
                "layer_id": layer_id,
                "phase": groups["phase"],
                "expected_ep_ids": [int(ep_id) for ep_id in expected_ep_ids],
                "arrived_ep_ids": [int(ep_id) for ep_id in arrived_ep_ids],
                **values,
            }
        )
    return records


def _parse_ep_conservation_records(text: str) -> list[dict[str, Any]]:
    """Parse one exact routing-conservation record per materialized EP wave."""

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _EP_CONSERVATION_LINE_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        try:
            per_ep_routed_tokens = ast.literal_eval(groups["per_ep_routed_tokens"])
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid per_ep_routed_tokens literal") from exc
        if not isinstance(per_ep_routed_tokens, dict):
            raise ValueError("per_ep_routed_tokens must be a dict")
        normalized: dict[int, int] = {}
        for ep_id, token_count in per_ep_routed_tokens.items():
            if (
                type(ep_id) is not int
                or type(token_count) is not int
                or ep_id < 0
                or token_count < 0
            ):
                raise ValueError(
                    "per_ep_routed_tokens must map non-negative integer IDs "
                    "to non-negative integer counts"
                )
            normalized[int(ep_id)] = int(token_count)
        batch_id = int(groups["batch_id"])
        layer_id = int(groups["layer_id"])
        routing_token_count = int(groups["routing_token_count"])
        router_topk = int(groups["router_topk"])
        total_routed_assignments = int(groups["total_routed_assignments"])
        if batch_id < 0 or layer_id < 0:
            raise ValueError("conservation batch_id/layer_id must be non-negative")
        if router_topk <= 0:
            raise ValueError("conservation router_topk must be positive")
        if routing_token_count * router_topk != total_routed_assignments:
            raise ValueError(
                "conservation total does not equal routing_token_count * router_topk"
            )
        if sum(normalized.values()) != total_routed_assignments:
            raise ValueError(
                "conservation per_ep_routed_tokens total is inconsistent"
            )
        records.append(
            {
                "cluster": groups["cluster"],
                "batch_id": batch_id,
                "layer_id": layer_id,
                "routing_token_count": routing_token_count,
                "router_topk": router_topk,
                "total_routed_assignments": total_routed_assignments,
                "per_ep_routed_tokens": normalized,
            }
        )
    return records


def check_case_log(
    case: MatrixCase,
    log_path: Path,
    metrics_dir: Path,
    *,
    strict_layers: bool = False,
) -> dict[str, Any]:
    """Check workflow evidence and numeric metrics for one completed run."""

    errors: list[str] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback" in text:
        errors.append("Traceback")
    success_markers = {
        "Simulation completed successfully.",
        "Online simulation completed successfully.",
    }
    if not any(line.strip() in success_markers for line in text.splitlines()):
        errors.append("missing success marker")
    if "Dummy Mode: false" not in text:
        errors.append("dummy mode was not explicitly disabled")
    if re.search(r"(?i)(synthetic latency|scaling factor|visibility multiplier)", text):
        errors.append("forbidden synthetic/scaling wording in log")

    layer_ids = sorted({int(match) for match in re.findall(r"layer_id=(\d+)", text)})
    if not layer_ids:
        errors.append("no layer_id trace")
    if strict_layers and case.model_kind != "dense":
        if case.model_kind == "mixed":
            # A mixed model may legitimately aggregate dense-layer work.  The
            # correctness contract is that every declared MoE layer appears in
            # the per-layer MoE trace; dense layer IDs are not a substitute.
            moe_layer_ids_seen = sorted(
                {
                    int(match)
                    for line in text.splitlines()
                    if "[MOE]" in line
                    for match in re.findall(r"layer_id=(\d+)", line)
                }
            )
            expected = list(case.moe_layer_ids)
            if moe_layer_ids_seen != expected:
                errors.append(
                    "mixed MoE layer ids are incomplete "
                    f"expected={expected} actual={moe_layer_ids_seen}"
                )
        else:
            expected = list(range(case.num_layers))
            if layer_ids != expected:
                errors.append(
                    f"layer ids are not contiguous expected={expected} actual={layer_ids}"
                )

    moe_trace_count = text.count("[MOE]")
    ep_participant_records = text.count("per_expert_tokens extracted:")
    try:
        ep_workload_records = _parse_ep_workload_records(text)
    except ValueError as exc:
        ep_workload_records = []
        errors.append(f"invalid EP workload trace: {exc}")
    try:
        ep_barrier_records = _parse_ep_barrier_records(text)
    except ValueError as exc:
        ep_barrier_records = []
        errors.append(f"invalid EP barrier trace: {exc}")
    try:
        ep_conservation_records = _parse_ep_conservation_records(text)
    except ValueError as exc:
        ep_conservation_records = []
        errors.append(f"invalid EP conservation trace: {exc}")
    if case.is_moe:
        if "moe_grouped_gemm" not in text or "moe_shuffling" not in text:
            errors.append("missing MoE grouped-gemm/shuffling trace")
        if not ep_workload_records:
            errors.append("missing EP workload trace")
        elif strict_layers:
            expected_moe_layers = set(int(layer_id) for layer_id in case.moe_layer_ids)
            records_by_wave: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
            for record in ep_workload_records:
                cluster_name = str(record["cluster"]).upper()
                records_by_wave.setdefault(
                    (
                        cluster_name,
                        int(record["batch_id"]),
                        int(record["layer_id"]),
                    ),
                    [],
                ).append(record)
            expected_roles = _expected_ep_roles(case)
            complete_layers_by_role = {
                cluster_name: set() for cluster_name in expected_roles
            }
            per_ep_totals_by_wave: dict[
                tuple[str, int, int],
                dict[int, int],
            ] = {}
            for (cluster_name, _batch_id, layer_id), wave_records in records_by_wave.items():
                expected_ep_size = _expected_ep_size_for_cluster(case, cluster_name)
                if expected_ep_size is None:
                    errors.append(
                        "EP workload uses an unsupported cluster role "
                        f"architecture={case.architecture} cluster={cluster_name}"
                    )
                    continue
                expected_ep_ids = set(range(expected_ep_size))
                ep_id_counts = Counter(
                    int(record["ep_id"]) for record in wave_records
                )
                duplicate_ep_ids = sorted(
                    ep_id
                    for ep_id, count in ep_id_counts.items()
                    if count != 1
                )
                if duplicate_ep_ids:
                    errors.append(
                        "duplicate EP workload records "
                        f"cluster={cluster_name} batch_id={_batch_id} "
                        f"layer={layer_id} ep_ids={duplicate_ep_ids}"
                    )
                actual_ep_ids = set(ep_id_counts)
                if (
                    actual_ep_ids == expected_ep_ids
                    and not duplicate_ep_ids
                ):
                    complete_layers_by_role[cluster_name].add(layer_id)
                    per_ep_totals_by_wave[
                        (cluster_name, _batch_id, layer_id)
                    ] = {
                        int(record["ep_id"]): sum(
                            record["per_expert_tokens"].values()
                        )
                        for record in wave_records
                    }
                if any(
                    int(record["moe_ep_size"]) != expected_ep_size
                    for record in wave_records
                ):
                    errors.append(
                        "EP workload moe_ep_size mismatch "
                        f"cluster={cluster_name} layer={layer_id} "
                        f"expected={expected_ep_size}"
                    )
                if actual_ep_ids != expected_ep_ids:
                    errors.append(
                        "EP workload participants are incomplete "
                        f"cluster={cluster_name} batch_id={_batch_id} layer={layer_id} "
                        f"expected={sorted(expected_ep_ids)} actual={sorted(actual_ep_ids)}"
                    )
            for cluster_name in expected_roles:
                for expected_layer_id in sorted(expected_moe_layers):
                    if (
                        expected_layer_id
                        not in complete_layers_by_role[cluster_name]
                    ):
                        errors.append(
                            "EP workload has no complete participant wave "
                            f"cluster={cluster_name} layer={expected_layer_id}"
                        )
            conservation_records_by_wave: dict[
                tuple[str, int, int],
                list[dict[str, Any]],
            ] = {}
            for record in ep_conservation_records:
                conservation_records_by_wave.setdefault(
                    (
                        str(record["cluster"]).upper(),
                        int(record["batch_id"]),
                        int(record["layer_id"]),
                    ),
                    [],
                ).append(record)
            for wave_key, records in conservation_records_by_wave.items():
                if len(records) != 1:
                    errors.append(
                        "duplicate EP conservation records "
                        f"cluster={wave_key[0]} batch_id={wave_key[1]} "
                        f"layer={wave_key[2]} count={len(records)}"
                    )
            workload_wave_keys = set(records_by_wave)
            conservation_wave_keys = set(conservation_records_by_wave)
            missing_conservation_waves = sorted(
                workload_wave_keys - conservation_wave_keys
            )
            extra_conservation_waves = sorted(
                conservation_wave_keys - workload_wave_keys
            )
            for wave_key in missing_conservation_waves:
                errors.append(
                    "missing EP conservation evidence for wave "
                    f"cluster={wave_key[0]} batch_id={wave_key[1]} "
                    f"layer={wave_key[2]}"
                )
            if extra_conservation_waves:
                errors.append(
                    "EP conservation wave identity mismatch "
                    f"extra={extra_conservation_waves}"
                )
            for wave_key in sorted(
                workload_wave_keys & conservation_wave_keys
            ):
                records = conservation_records_by_wave[wave_key]
                if len(records) != 1:
                    continue
                conservation = records[0]
                expected_ep_size = _expected_ep_size_for_cluster(
                    case,
                    wave_key[0],
                )
                expected_ep_ids = set(range(expected_ep_size or 0))
                per_ep_totals = per_ep_totals_by_wave.get(wave_key)
                if per_ep_totals is None:
                    continue
                if set(per_ep_totals) != expected_ep_ids:
                    errors.append(
                        "EP conservation workload participant IDs do not match "
                        f"cluster={wave_key[0]} layer={wave_key[2]}"
                    )
                if per_ep_totals != dict(conservation["per_ep_routed_tokens"]):
                    errors.append(
                        "EP conservation per-lane totals disagree with workload "
                        f"cluster={wave_key[0]} batch_id={wave_key[1]} layer={wave_key[2]}"
                    )
                if sum(per_ep_totals.values()) != int(
                    conservation["total_routed_assignments"]
                ):
                    errors.append(
                        "EP conservation total disagrees with lane workload "
                        f"cluster={wave_key[0]} batch_id={wave_key[1]} layer={wave_key[2]}"
                    )

            barrier_records_by_wave_phase: dict[
                tuple[str, int, int, str],
                list[dict[str, Any]],
            ] = {}
            for record in ep_barrier_records:
                barrier_records_by_wave_phase.setdefault(
                    (
                        str(record["cluster"]).upper(),
                        int(record["batch_id"]),
                        int(record["layer_id"]),
                        str(record["phase"]),
                    ),
                    [],
                ).append(record)
            for barrier_key, records in barrier_records_by_wave_phase.items():
                if len(records) != 1:
                    errors.append(
                        "duplicate EP barrier records "
                        f"cluster={barrier_key[0]} batch_id={barrier_key[1]} "
                        f"layer={barrier_key[2]} phase={barrier_key[3]} "
                        f"count={len(records)}"
                    )
            required_phases = ("dispatch", "combine")
            for phase in required_phases:
                phase_wave_keys = {
                    key[:3]
                    for key in barrier_records_by_wave_phase
                    if key[3] == phase
                }
                missing_barrier_waves = sorted(
                    workload_wave_keys - phase_wave_keys
                )
                extra_barrier_waves = sorted(
                    phase_wave_keys - workload_wave_keys
                )
                if missing_barrier_waves:
                    errors.append(
                        "missing EP barrier evidence for workload waves "
                        f"phase={phase} waves={missing_barrier_waves}"
                    )
                if missing_barrier_waves or extra_barrier_waves:
                    errors.append(
                        "EP barrier wave identity mismatch "
                        f"phase={phase} missing={missing_barrier_waves} "
                        f"extra={extra_barrier_waves}"
                    )
                for wave_key in sorted(
                    workload_wave_keys & phase_wave_keys
                ):
                    records = barrier_records_by_wave_phase[
                        (*wave_key, phase)
                    ]
                    if len(records) != 1:
                        continue
                    expected_ep_size = _expected_ep_size_for_cluster(
                        case,
                        wave_key[0],
                    )
                    expected_ep_ids = set(range(expected_ep_size or 0))
                    workload_ep_ids = {
                        int(record["ep_id"])
                        for record in records_by_wave[wave_key]
                    }
                    barrier_ep_ids = set(records[0]["expected_ep_ids"])
                    if (
                        barrier_ep_ids != expected_ep_ids
                        or barrier_ep_ids != workload_ep_ids
                    ):
                        errors.append(
                            "EP barrier participants disagree with workload "
                            f"cluster={wave_key[0]} batch_id={wave_key[1]} "
                            f"layer={wave_key[2]} phase={phase} "
                            f"expected={sorted(expected_ep_ids)} "
                            f"workload={sorted(workload_ep_ids)} "
                            f"barrier={sorted(barrier_ep_ids)}"
                        )

            event_positions_by_wave = _ep_event_positions(text)
            ordered_waves_by_scope: dict[
                tuple[str, int],
                list[tuple[int, int, int]],
            ] = {}
            for wave_key in sorted(workload_wave_keys):
                event_positions = event_positions_by_wave.get(wave_key, {})
                workload_positions = event_positions.get("workload", [])
                conservation_positions = event_positions.get("conservation", [])
                dispatch_positions = event_positions.get("dispatch", [])
                combine_positions = event_positions.get("combine", [])
                if (
                    len(workload_positions) != len(records_by_wave[wave_key])
                    or len(conservation_positions) != 1
                    or len(dispatch_positions) != 1
                    or len(combine_positions) != 1
                ):
                    continue
                materialization_end = max(
                    [*workload_positions, *conservation_positions]
                )
                dispatch_position = dispatch_positions[0]
                combine_position = combine_positions[0]
                if not (
                    materialization_end < dispatch_position < combine_position
                ):
                    errors.append(
                        "EP wave event order is invalid "
                        f"cluster={wave_key[0]} batch_id={wave_key[1]} "
                        f"layer={wave_key[2]}"
                    )
                    continue
                wave_start = min(
                    [*workload_positions, *conservation_positions]
                )
                ordered_waves_by_scope.setdefault(
                    (wave_key[0], wave_key[1]),
                    [],
                ).append((wave_key[2], wave_start, combine_position))

            for (cluster_name, batch_id), ordered_waves in (
                ordered_waves_by_scope.items()
            ):
                ordered_waves.sort()
                for previous_wave, next_wave in zip(
                    ordered_waves,
                    ordered_waves[1:],
                ):
                    previous_layer, _previous_start, previous_combine = (
                        previous_wave
                    )
                    next_layer, next_start, _next_combine = next_wave
                    if next_start <= previous_combine:
                        errors.append(
                            "next MoE layer started before prior combine "
                            f"cluster={cluster_name} batch_id={batch_id} "
                            f"previous_layer={previous_layer} "
                            f"next_layer={next_layer}"
                        )
        if case.architecture == "pd-af-disaggregation" and ep_participant_records == 0:
            errors.append("missing DECODE_FFN EP participant maps")
        if case.expects_zero_routed_lane and not any(
            sum(record["per_expert_tokens"].values()) == 0
            for record in ep_workload_records
        ):
            errors.append("zero-routed case has no zero-total EP lane evidence")

    metric_path = metrics_dir / "system_metrics.json"
    numeric_metric_count = 0
    metrics: dict[str, Any] = {}
    activation_evidence: dict[str, Any] = {}
    if not metric_path.is_file():
        errors.append(f"missing metrics file: {metric_path}")
    else:
        try:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            numeric_metric_count = sum(1 for _ in _finite_metric_values(metrics))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid metrics: {exc}")

    activation_errors, activation_evidence = _check_optimization_activation(
        case,
        text,
        metrics_dir,
        metrics,
    )
    errors.extend(activation_errors)

    def _stat_value(name: str) -> float | None:
        value = metrics.get(name, {})
        if isinstance(value, Mapping):
            candidate = value.get("mean")
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return None

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": "; ".join(errors),
        "layer_ids": layer_ids,
        "moe_trace_count": moe_trace_count,
        "ep_participant_records": ep_participant_records,
        "ep_workload_records": len(ep_workload_records),
        "ep_barrier_records": len(ep_barrier_records),
        "ep_conservation_records": len(ep_conservation_records),
        "numeric_metric_count": numeric_metric_count,
        "ttft_mean_ms": _stat_value("ttft_statistics"),
        "tpot_mean_ms": _stat_value("tpot_statistics"),
        "e2e_mean_ms": _stat_value("request_e2e_time_statistics"),
        **activation_evidence,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_result_rows(path: Path) -> list[dict[str, Any]]:
    """Load an existing result ledger, failing on malformed evidence."""

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid result ledger JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"result ledger row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _serialized_case_payload(case: MatrixCase) -> dict[str, Any]:
    """Return the exact JSON representation persisted in manifests and ledgers."""

    payload = json.loads(json.dumps(asdict(case), sort_keys=True))
    if not isinstance(payload, dict):
        raise TypeError(f"case payload must serialize to an object: {case.case_id}")
    return payload


def _optimization_pair_specs(
    cases: Sequence[MatrixCase],
) -> list[dict[str, Any]]:
    pair_groups: dict[str, list[MatrixCase]] = {}
    comparison_groups: dict[str, list[MatrixCase]] = {}
    for case in cases:
        if case.pair_id is not None:
            pair_groups.setdefault(case.pair_id, []).append(case)
        if case.comparison_group_id is not None:
            comparison_groups.setdefault(case.comparison_group_id, []).append(case)

    specs: list[dict[str, Any]] = []
    for pair_id, group in sorted(pair_groups.items()):
        by_role = {case.pair_role: case for case in group}
        control = by_role["control"]
        enabled = by_role["enabled"]
        if control.optimization_stratum == "prefix":
            optimization = "prefix_cache"
            target_field = "enable_prefix_caching"
        elif control.optimization_stratum == "mtp":
            optimization = "mtp"
            target_field = "enable_mtp"
        else:
            raise ValueError(f"unsupported paired stratum: {pair_id}")
        specs.append(
            {
                "comparison_id": pair_id,
                "group_id": pair_id,
                "optimization": optimization,
                "target_field": target_field,
                "control": control,
                "enabled": enabled,
            }
        )

    for group_id, group in sorted(comparison_groups.items()):
        if len(group) < 2:
            continue
        if group[0].architecture == "pd-af-disaggregation":
            controls = [case for case in group if not case.use_cuda_graph]
            enabled_rows = [case for case in group if case.use_cuda_graph]
            if len(controls) != 1 or len(enabled_rows) != 1:
                raise ValueError(
                    f"{group_id} must contain one PD-AF graph control and enabled row"
                )
            specs.append(
                {
                    "comparison_id": f"{group_id}:cuda_graph",
                    "group_id": group_id,
                    "optimization": "cuda_graph",
                    "target_field": "use_cuda_graph",
                    "control": controls[0],
                    "enabled": enabled_rows[0],
                }
            )
            continue

        by_axes = {
            (case.decode_cuda_graph_mode, case.enable_chunked_prefill): case
            for case in group
        }
        chunk_modes = sorted(
            {case.enable_chunked_prefill for case in group}
        )
        graph_modes = sorted(
            {case.decode_cuda_graph_mode for case in group}
        )
        for chunk_enabled in chunk_modes:
            control = by_axes.get(("none", chunk_enabled))
            if control is None:
                continue
            for graph_mode in graph_modes:
                if graph_mode == "none":
                    continue
                enabled = by_axes.get((graph_mode, chunk_enabled))
                if enabled is None:
                    continue
                specs.append(
                    {
                        "comparison_id": (
                            f"{group_id}:cuda_graph:{graph_mode}:"
                            f"chunk_{'on' if chunk_enabled else 'off'}"
                        ),
                        "group_id": group_id,
                        "optimization": "cuda_graph",
                        "target_field": "decode_cuda_graph_mode",
                        "control": control,
                        "enabled": enabled,
                    }
                )

        if set(chunk_modes) == {False, True}:
            for graph_mode in graph_modes:
                control = by_axes.get((graph_mode, False))
                enabled = by_axes.get((graph_mode, True))
                if control is None or enabled is None:
                    continue
                specs.append(
                    {
                        "comparison_id": (
                            f"{group_id}:chunked_prefill:{graph_mode}"
                        ),
                        "group_id": group_id,
                        "optimization": "chunked_prefill",
                        "target_field": "enable_chunked_prefill",
                        "control": control,
                        "enabled": enabled,
                    }
                )
    return specs


def _optimization_activation_errors(
    optimization: str,
    control_check: Mapping[str, Any],
    enabled_check: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []

    def _count(check: Mapping[str, Any], field: str) -> int:
        value = check.get(field, 0)
        return int(value) if type(value) is int and value >= 0 else 0

    if optimization == "cuda_graph":
        control_count = _count(control_check, "cuda_graph_capture_count")
        enabled_count = _count(enabled_check, "cuda_graph_capture_count")
        if control_count != 0:
            errors.append("CUDA Graph control unexpectedly activated")
        if enabled_count <= 0:
            errors.append("CUDA Graph activation evidence is missing")
    elif optimization == "chunked_prefill":
        control_count = _count(control_check, "chunked_prefill_split_count")
        enabled_count = _count(enabled_check, "chunked_prefill_split_count")
        if control_count != 0:
            errors.append("Chunked Prefill control unexpectedly split a request")
        if enabled_count <= 0:
            errors.append("Chunked Prefill activation evidence is missing")
    elif optimization == "prefix_cache":
        control_count = _count(control_check, "prefix_cache_hit_blocks")
        enabled_count = _count(enabled_check, "prefix_cache_hit_blocks")
        if control_count != 0:
            errors.append("Prefix Cache control unexpectedly recorded hits")
        if enabled_count <= 0:
            errors.append("Prefix Cache activation evidence is missing")
    elif optimization == "mtp":
        control_iterations = _count(control_check, "spec_decode_iterations")
        enabled_iterations = _count(enabled_check, "spec_decode_iterations")
        enabled_tokens = _count(enabled_check, "spec_decode_committed_tokens")
        if control_iterations != 0:
            errors.append("MTP control unexpectedly recorded iterations")
        if enabled_iterations <= 0 or enabled_tokens <= 0:
            errors.append("MTP activation evidence is missing")
    else:
        raise ValueError(f"unsupported optimization comparison: {optimization}")
    return errors


def build_optimization_comparison(
    cases: Sequence[MatrixCase],
    result_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("optimization comparison cases contain duplicate case IDs")
    result_by_id: dict[str, Mapping[str, Any]] = {}
    for row in result_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in case_by_id:
            raise ValueError(f"unknown optimization result case_id={case_id!r}")
        if case_id in result_by_id:
            raise ValueError(f"duplicate optimization result case_id={case_id!r}")
        expected_case = _serialized_case_payload(case_by_id[case_id])
        if row.get("case") != expected_case:
            raise ValueError(
                "optimization result-row case metadata mismatch: "
                f"case_id={case_id!r}"
            )
        result_by_id[case_id] = row
    missing_results = sorted(set(case_by_id) - set(result_by_id))
    if missing_results:
        raise ValueError(
            f"optimization comparison is missing result rows: {missing_results}"
        )

    metric_fields = ("ttft_mean_ms", "tpot_mean_ms", "e2e_mean_ms")
    pair_rows: list[dict[str, Any]] = []
    for spec in _optimization_pair_specs(cases):
        control = spec["control"]
        enabled = spec["enabled"]
        control_result = result_by_id[control.case_id]
        enabled_result = result_by_id[enabled.case_id]
        control_check = control_result.get("check")
        enabled_check = enabled_result.get("check")
        errors: list[str] = []
        if not isinstance(control_check, Mapping) or not isinstance(
            enabled_check, Mapping
        ):
            raise ValueError(
                f"{spec['comparison_id']} result rows require check objects"
            )
        for label, result, check in (
            ("control", control_result, control_check),
            ("enabled", enabled_result, enabled_check),
        ):
            if result.get("status") != "PASS" or check.get("status") != "PASS":
                errors.append(f"{label} runtime workflow did not PASS")

        ignored_fields = {
            "case_id",
            "baseline_case_id",
            "pair_id",
            "comparison_group_id",
            "pair_role",
        }
        changed_fields = sorted(
            field_name
            for field_name in asdict(control)
            if field_name not in ignored_fields
            and getattr(control, field_name) != getattr(enabled, field_name)
        )
        target_field = str(spec["target_field"])
        if changed_fields != [target_field]:
            errors.append(
                f"pair changes fields={changed_fields}, expected={[target_field]}"
            )

        errors.extend(
            _optimization_activation_errors(
                str(spec["optimization"]),
                control_check,
                enabled_check,
            )
        )
        metrics: dict[str, dict[str, float | None]] = {}
        for metric_field in metric_fields:
            raw_control = control_check.get(metric_field)
            raw_enabled = enabled_check.get(metric_field)
            control_value = (
                float(raw_control)
                if isinstance(raw_control, (int, float))
                and not isinstance(raw_control, bool)
                and math.isfinite(float(raw_control))
                and float(raw_control) >= 0
                else None
            )
            enabled_value = (
                float(raw_enabled)
                if isinstance(raw_enabled, (int, float))
                and not isinstance(raw_enabled, bool)
                and math.isfinite(float(raw_enabled))
                and float(raw_enabled) >= 0
                else None
            )
            if control_value is None or enabled_value is None:
                delta = None
                relative_delta = None
            else:
                delta = enabled_value - control_value
                relative_delta = (
                    None
                    if control_value == 0
                    else delta / control_value * 100.0
                )
            metrics[metric_field] = {
                "control_ms": control_value,
                "enabled_ms": enabled_value,
                "delta_ms": delta,
                "relative_delta_percent": relative_delta,
            }

        pair_rows.append(
            {
                "comparison_id": spec["comparison_id"],
                "group_id": spec["group_id"],
                "optimization": spec["optimization"],
                "target_field": target_field,
                "control_case_id": control.case_id,
                "enabled_case_id": enabled.case_id,
                "architecture": control.architecture,
                "model_kind": control.model_kind,
                "simulation_mode": control.simulation_mode,
                "total_cards": control.total_cards,
                "changed_fields": changed_fields,
                "latency_oracle": "report_only",
                "status": "PASS" if not errors else "FAIL",
                "errors": "; ".join(errors),
                "metrics": metrics,
            }
        )

    failed_pair_count = sum(row["status"] != "PASS" for row in pair_rows)
    return {
        "status": "PASS" if failed_pair_count == 0 else "FAIL",
        "case_count": len(cases),
        "pair_count": len(pair_rows),
        "failed_pair_count": failed_pair_count,
        "latency_oracle": "report_only",
        "optimization_counts": dict(
            sorted(Counter(row["optimization"] for row in pair_rows).items())
        ),
        "pairs": pair_rows,
    }


def write_optimization_comparison_artifacts(
    task_dir: Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    """Write machine-readable and human-readable paired comparison reports."""

    task_dir.mkdir(parents=True, exist_ok=True)
    json_path = task_dir / "moe_ep_non_dummy_optimization_comparison.json"
    csv_path = task_dir / "moe_ep_non_dummy_optimization_comparison.csv"
    markdown_path = task_dir / "moe_ep_non_dummy_optimization_comparison.md"

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metric_fields = ("ttft_mean_ms", "tpot_mean_ms", "e2e_mean_ms")
    csv_fields = [
        "comparison_id",
        "group_id",
        "optimization",
        "target_field",
        "control_case_id",
        "enabled_case_id",
        "architecture",
        "model_kind",
        "simulation_mode",
        "total_cards",
        "changed_fields",
        "latency_oracle",
        "status",
        "errors",
    ]
    for metric_field in metric_fields:
        metric_name = metric_field.removesuffix("_ms")
        csv_fields.extend(
            [
                f"control_{metric_name}_ms",
                f"enabled_{metric_name}_ms",
                f"{metric_name}_delta_ms",
                f"{metric_name}_relative_delta_percent",
            ]
        )

    csv_rows: list[dict[str, Any]] = []
    for pair in report.get("pairs", ()):
        row = {field: pair.get(field, "") for field in csv_fields[:14]}
        row["changed_fields"] = ",".join(pair.get("changed_fields", ()))
        metrics = pair.get("metrics", {})
        for metric_field in metric_fields:
            metric_name = metric_field.removesuffix("_ms")
            values = metrics.get(metric_field, {})
            row[f"control_{metric_name}_ms"] = values.get("control_ms")
            row[f"enabled_{metric_name}_ms"] = values.get("enabled_ms")
            row[f"{metric_name}_delta_ms"] = values.get("delta_ms")
            row[f"{metric_name}_relative_delta_percent"] = values.get(
                "relative_delta_percent"
            )
        csv_rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    markdown_lines = [
        "# Frontier Optimization Paired Comparison",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Cases: `{report.get('case_count')}`",
        f"- Pairs: `{report.get('pair_count')}`",
        f"- Failed pairs: `{report.get('failed_pair_count')}`",
        "- Latency values are report-only; they do not determine PASS/FAIL.",
        "",
        "| Pair | Optimization | Architecture | Control | Enabled | Status | "
        "TTFT control/enabled/delta (ms) | TPOT control/enabled/delta (ms) | "
        "E2E control/enabled/delta (ms) | Errors |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]

    def metric_triplet(pair: Mapping[str, Any], metric_field: str) -> str:
        values = pair["metrics"][metric_field]
        return (
            f"{values['control_ms']} / {values['enabled_ms']} / "
            f"{values['delta_ms']}"
        )

    for pair in report.get("pairs", ()):
        errors = str(pair.get("errors", "")).replace("|", "\\|")
        markdown_lines.append(
            "| {comparison_id} | {optimization} | {architecture} | "
            "`{control_case_id}` | `{enabled_case_id}` | {status} | "
            "{ttft} | {tpot} | {e2e} | {errors} |".format(
                comparison_id=pair["comparison_id"],
                optimization=pair["optimization"],
                architecture=pair["architecture"],
                control_case_id=pair["control_case_id"],
                enabled_case_id=pair["enabled_case_id"],
                status=pair["status"],
                ttft=metric_triplet(pair, "ttft_mean_ms"),
                tpot=metric_triplet(pair, "tpot_mean_ms"),
                e2e=metric_triplet(pair, "e2e_mean_ms"),
                errors=errors,
            )
        )
    markdown_path.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    return json_path, csv_path, markdown_path


def _validate_result_ledger_provenance(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    output_root: Path,
    results_path: Path,
) -> None:
    """Reject ledger rows produced by another run/output generation.

    A case ID is not a sufficient identity for evidence: two worktrees can
    produce the same ID with identical metrics but different scheduler traces.
    Every persisted row must therefore carry the canonical output and ledger
    paths used by the current invocation.
    """

    expected_repo = repo_root.resolve()
    expected_output = output_root.resolve()
    expected_results = results_path.resolve()
    for row in rows:
        case_id = row.get("case_id", "<missing>")
        row_repo = row.get("repo_root")
        row_output = row.get("output_root")
        row_results = row.get("results_path")
        row_log = row.get("log_path")
        row_metrics = row.get("metrics_path")
        if not all(
            isinstance(value, str)
            for value in (row_repo, row_output, row_results, row_log, row_metrics)
        ):
            raise ValueError(
                "result ledger row is missing canonical provenance: "
                f"case_id={case_id!r}"
            )
        if Path(row_repo).resolve() != expected_repo:
            raise ValueError(
                "result ledger repo_root provenance mismatch: "
                f"case_id={case_id!r}, row={row_repo!r}, expected={str(expected_repo)!r}"
            )
        if Path(row_output).resolve() != expected_output:
            raise ValueError(
                "result ledger output_root provenance mismatch: "
                f"case_id={case_id!r}, row={row_output!r}, expected={str(expected_output)!r}"
            )
        if Path(row_results).resolve() != expected_results:
            raise ValueError(
                "result ledger results_path provenance mismatch: "
                f"case_id={case_id!r}, row={row_results!r}, expected={str(expected_results)!r}"
            )
        if not case_id or not isinstance(case_id, str):
            raise ValueError("result ledger case_id must be a non-empty string")
        case_root = expected_output / case_id
        log_path = Path(row_log).resolve()
        if not log_path.is_relative_to(case_root):
            raise ValueError(
                "result ledger log_path is outside its canonical case directory: "
                f"case_id={case_id!r}, path={row_log!r}"
            )
        if row_metrics:
            metrics_path = Path(row_metrics).resolve()
            if not metrics_path.is_relative_to(case_root):
                raise ValueError(
                    "result ledger metrics_path is outside its canonical case directory: "
                    f"case_id={case_id!r}, path={row_metrics!r}"
                )
        elif row.get("status") == "PASS":
            raise ValueError(
                "result ledger PASS row has no canonical metrics_path: "
                f"case_id={case_id!r}"
            )


def _validate_persisted_case_metadata(
    cases: Sequence[MatrixCase],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require on-disk case metadata to match the active matrix exactly."""

    case_by_id = {case.case_id: case for case in cases}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in case_by_id:
            raise ValueError(f"unknown optimization result case_id={case_id!r}")
        log_path = row.get("log_path")
        if not isinstance(log_path, str) or not log_path:
            raise ValueError(
                f"optimization result row has no log_path: case_id={case_id!r}"
            )
        metadata_path = Path(log_path).resolve().parent / "case_metadata.json"
        if not metadata_path.is_file():
            raise ValueError(
                "persisted case metadata is missing: "
                f"case_id={case_id!r}, path={metadata_path}"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "persisted case metadata is invalid JSON: "
                f"case_id={case_id!r}, path={metadata_path}"
            ) from exc
        expected_case = _serialized_case_payload(case_by_id[case_id])
        if not isinstance(metadata, Mapping) or metadata.get("case") != expected_case:
            raise ValueError(
                "persisted case metadata mismatch: "
                f"case_id={case_id!r}, path={metadata_path}"
            )


def _merge_result_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Merge rerun rows without losing prior case evidence.

    Existing and new rows must refer only to the current manifest and must have
    unique case IDs within each input.  A rerun replaces the prior row for the
    same case; cases not selected in the current invocation remain intact.
    Output follows manifest order and omits cases that have not run yet.
    """

    expected = tuple(expected_case_ids)
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("manifest contains duplicate case IDs")

    merged: dict[str, dict[str, Any]] = {}
    for source_name, rows in (("existing", existing_rows), ("new", new_rows)):
        seen: set[str] = set()
        for row in rows:
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{source_name} result row has no non-empty case_id")
            if case_id not in expected_set:
                raise ValueError(
                    f"{source_name} result row has unknown case_id={case_id!r}"
                )
            if case_id in seen:
                raise ValueError(f"{source_name} result rows repeat case_id={case_id!r}")
            seen.add(case_id)
            merged[case_id] = dict(row)
    return [merged[case_id] for case_id in expected if case_id in merged]


def write_manifest(path: Path, cases: Sequence[MatrixCase]) -> None:
    _write_jsonl(path, [asdict(case) for case in cases])


def _load_manifest(path: Path) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["moe_layer_ids"] = tuple(record["moe_layer_ids"])
        cases.append(MatrixCase(**record))
    return cases


def _run_case(
    case: MatrixCase,
    command: str,
    env: dict[str, str],
    *,
    repo_root: Path,
    output_root: Path,
    results_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    case_dir = output_root / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / f"{case.case_id}.log"
    metadata_path = case_dir / "case_metadata.json"
    run_started_at_s = time.time()
    metadata_path.write_text(
        json.dumps(
            {
                "case": asdict(case),
                "command": command,
                "environment": {
                    key: env[key]
                    for key in (
                        "MODEL_NAME",
                        "ENABLE_DUMMY_MODE",
                        "DECODE_CUDA_GRAPH_MODE",
                        "MOE_ROUTING_DISTRIBUTION_TYPE",
                        "MOE_ROUTING_SEED",
                        "TOTAL_EXPERTS",
                        "ROUTER_TOPK",
                    )
                    if key in env
                },
                "repo_root": str(repo_root),
                "output_root": str(output_root),
                "results_path": str(results_path),
                "run_started_at_unix_s": run_started_at_s,
                "trace_schema_version": 1,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    run_freshness_marker_ns = metadata_path.stat().st_mtime_ns
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"MATRIX_COMMAND: {command}\n")
        stream.flush()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=repo_root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            exit_code = 124
            stream.write(f"MATRIX_TIMEOUT after {timeout_seconds}s\n")
    elapsed = time.monotonic() - started
    run_finished_at_s = time.time()
    try:
        if (
            not log_path.is_file()
            or log_path.stat().st_mtime_ns < run_freshness_marker_ns
        ):
            raise FileNotFoundError(
                f"no fresh case log for {case.case_id} under {output_root}"
            )
        metrics_dir = _find_metrics_dir(
            output_root,
            case,
            started_at_ns=run_freshness_marker_ns,
        )
        check = check_case_log(case, log_path, metrics_dir, strict_layers=True)
        metrics_path = str(metrics_dir)
    except (FileNotFoundError, OSError) as exc:
        check = {"status": "FAIL", "errors": str(exc)}
        metrics_path = ""
    status = "PASS" if exit_code == 0 and check.get("status") == "PASS" else "FAIL"
    return {
        "case_id": case.case_id,
        "case": _serialized_case_payload(case),
        "architecture": case.architecture,
        "model_kind": case.model_kind,
        "total_cards": case.total_cards,
        "exit_code": exit_code,
        "elapsed_seconds": round(elapsed, 3),
        "log_path": str(log_path),
        "metrics_path": metrics_path,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "results_path": str(results_path),
        "run_started_at_unix_s": run_started_at_s,
        "run_finished_at_unix_s": run_finished_at_s,
        "trace_schema_version": 1,
        "status": status,
        "check": check,
    }


def run_cases(
    cases: Sequence[MatrixCase],
    repo_root: Path,
    output_root: Path,
    results_path: Path,
    *,
    start: int = 0,
    limit: int | None = None,
    timeout_seconds: int = 600,
    continue_on_failure: bool = False,
    max_parallel_cases: int = 1,
) -> list[dict[str, Any]]:
    if type(max_parallel_cases) is not int or max_parallel_cases <= 0:
        raise ValueError(
            "max_parallel_cases must be a positive integer, "
            f"got {max_parallel_cases!r}"
        )
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    results_path = results_path.resolve()
    selected = list(cases[start : (start + limit) if limit is not None else None])
    expected_case_ids = tuple(case.case_id for case in cases)
    existing_rows = _load_result_rows(results_path)
    _validate_result_ledger_provenance(
        existing_rows,
        repo_root=repo_root,
        output_root=output_root,
        results_path=results_path,
    )
    persisted = _merge_result_rows(
        existing_rows,
        (),
        expected_case_ids=expected_case_ids,
    )
    launch_specs: list[tuple[MatrixCase, str, dict[str, str]]] = []
    for case in selected:
        validate_profile_inputs(case, repo_root)
        command, env = build_shell_command(case, repo_root, output_root)
        launch_specs.append((case, command, env))

    results_by_index: dict[int, dict[str, Any]] = {}

    def persist_result(index: int, result: dict[str, Any]) -> None:
        nonlocal persisted
        results_by_index[index] = result
        persisted = _merge_result_rows(
            persisted,
            (result,),
            expected_case_ids=expected_case_ids,
        )
        _write_jsonl(results_path, persisted)

    if max_parallel_cases == 1 or len(launch_specs) <= 1:
        for index, (case, command, env) in enumerate(launch_specs):
            result = _run_case(
                case,
                command,
                env,
                repo_root=repo_root,
                output_root=output_root,
                results_path=results_path,
                timeout_seconds=timeout_seconds,
            )
            persist_result(index, result)
            if result["status"] != "PASS" and not continue_on_failure:
                raise RuntimeError(
                    f"matrix case failed: {json.dumps(result, sort_keys=True)}"
                )
        return [results_by_index[index] for index in range(len(launch_specs))]

    failure_result: dict[str, Any] | None = None
    next_index = 0
    with ThreadPoolExecutor(max_workers=max_parallel_cases) as executor:
        active = {}

        def submit_ready_cases() -> None:
            nonlocal next_index
            while (
                failure_result is None
                and next_index < len(launch_specs)
                and len(active) < max_parallel_cases
            ):
                case, command, env = launch_specs[next_index]
                future = executor.submit(
                    _run_case,
                    case,
                    command,
                    env,
                    repo_root=repo_root,
                    output_root=output_root,
                    results_path=results_path,
                    timeout_seconds=timeout_seconds,
                )
                active[future] = next_index
                next_index += 1

        submit_ready_cases()
        while active:
            done, _pending = wait(active, return_when=FIRST_COMPLETED)
            completed = sorted(
                ((active.pop(future), future) for future in done),
                key=lambda item: item[0],
            )
            for index, future in completed:
                result = future.result()
                persist_result(index, result)
                if (
                    result["status"] != "PASS"
                    and not continue_on_failure
                    and failure_result is None
                ):
                    failure_result = result
            submit_ready_cases()

    if failure_result is not None:
        raise RuntimeError(
            f"matrix case failed: {json.dumps(failure_result, sort_keys=True)}"
        )
    return [
        results_by_index[index]
        for index in range(len(launch_specs))
        if index in results_by_index
    ]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("generate", "preflight", "run", "compare"),
        default="generate",
    )
    parser.add_argument(
        "--matrix-kind",
        choices=("regression", "optimization"),
        default="regression",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("task_memory/task_2026-08-12_moe_ep_rank_stragger_analysis"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help=(
            "Canonical JSONL ledger path. Required when resuming a ledger; "
            "rows from another output root or worktree are rejected."
        ),
    )
    parser.add_argument(
        "--preflight-path",
        type=Path,
        default=None,
        help="Independent JSONL ledger path for static preflight evidence.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-parallel-cases", type=int, default=1)
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.repo_root.resolve()
    task_dir = args.task_dir if args.task_dir.is_absolute() else repo_root / args.task_dir
    if args.matrix_kind == "optimization":
        manifest_name = "moe_ep_non_dummy_optimization_matrix_manifest.jsonl"
        results_name = "moe_ep_non_dummy_optimization_matrix_results.jsonl"
        default_output_root = Path(
            "/data/ycfeng/tmp/frontier_non_dummy_optimization_matrix"
        )
        cases = build_optimization_matrix(repo_root)
    else:
        manifest_name = "moe_ep_non_dummy_matrix_manifest.jsonl"
        results_name = "moe_ep_non_dummy_matrix_results.jsonl"
        default_output_root = Path("/data/ycfeng/tmp/frontier_non_dummy_matrix")
        cases = build_matrix(repo_root)
    manifest_path = task_dir / manifest_name
    results_path = (
        args.results_path
        if args.results_path is not None
        else task_dir / results_name
    )
    output_root = (
        args.output_root if args.output_root is not None else default_output_root
    )
    write_manifest(manifest_path, cases)
    print(f"manifest={manifest_path} cases={len(cases)}")
    if args.mode == "generate":
        return 0
    if args.mode == "compare":
        if args.matrix_kind != "optimization":
            raise ValueError("compare mode requires --matrix-kind optimization")
        result_rows = _load_result_rows(results_path)
        _validate_result_ledger_provenance(
            result_rows,
            repo_root=repo_root,
            output_root=output_root,
            results_path=results_path,
        )
        _validate_persisted_case_metadata(cases, result_rows)
        report = build_optimization_comparison(cases, result_rows)
        json_path, csv_path, markdown_path = (
            write_optimization_comparison_artifacts(task_dir, report)
        )
        print(f"comparison_json={json_path}")
        print(f"comparison_csv={csv_path}")
        print(f"comparison_markdown={markdown_path}")
        return 0 if report["status"] == "PASS" else 1
    if args.mode == "preflight":
        preflight_path = (
            args.preflight_path
            if args.preflight_path is not None
            else task_dir
            / f"moe_ep_non_dummy_{args.matrix_kind}_preflight.jsonl"
        )
        preflight_rows = preflight_cases(
            cases,
            repo_root,
            output_root,
            matrix_kind=args.matrix_kind,
        )
        _write_jsonl(preflight_path, preflight_rows)
        ready = sum(row["status"] == "READY" for row in preflight_rows)
        blocked = len(preflight_rows) - ready
        print(f"preflight={preflight_path} ready={ready} blocked={blocked}")
        return 0 if blocked == 0 else 1
    if args.matrix_kind == "optimization":
        preflight_path = (
            args.preflight_path
            if args.preflight_path is not None
            else task_dir / "moe_ep_non_dummy_optimization_preflight.jsonl"
        )
        preflight_rows = preflight_cases(
            cases,
            repo_root,
            output_root,
            matrix_kind=args.matrix_kind,
        )
        _write_jsonl(preflight_path, preflight_rows)
        ready = sum(row["status"] == "READY" for row in preflight_rows)
        blocked = len(preflight_rows) - ready
        print(f"preflight={preflight_path} ready={ready} blocked={blocked}")
        if blocked:
            return 1
    results = run_cases(
        cases,
        repo_root,
        output_root,
        results_path,
        start=args.start,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        continue_on_failure=args.continue_on_failure,
        max_parallel_cases=args.max_parallel_cases,
    )
    passed = sum(result["status"] == "PASS" for result in results)
    failed = len(results) - passed
    print(f"results={results_path} passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
