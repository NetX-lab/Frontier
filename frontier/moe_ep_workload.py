"""Shared per-layer MoE expert-parallel workload materialization.

This module is intentionally pure.  It owns routing-ratio validation,
deterministic integerization, and Replica-local expert ownership splitting;
it does not own scheduler state, predictor caches, communication events, or
request accounting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import floor, isfinite
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias


ExpertTokenMap: TypeAlias = Mapping[int, int]
ExpertOwnership: TypeAlias = Mapping[int, int]
RoutingDetails: TypeAlias = Mapping[int, Mapping[int, Mapping[int, Real]]]


def _require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _freeze_map(values: Mapping[int, int]) -> MappingProxyType:
    return MappingProxyType(dict(values))


def _freeze_nested_map(
    values: Mapping[int, Mapping[int, int]],
) -> MappingProxyType:
    return MappingProxyType(
        {
            outer_key: MappingProxyType(dict(inner_values))
            for outer_key, inner_values in values.items()
        }
    )


@dataclass(frozen=True)
class LayerEPWorkload:
    """Immutable aggregate physical workload for one MoE layer operation."""

    target_replica_id: int
    global_layer_id: int
    routing_token_count: int
    router_topk: int
    total_routed_assignments: int
    global_per_expert_tokens: ExpertTokenMap
    per_ep_per_expert_tokens: Mapping[int, ExpertTokenMap]
    per_ep_routed_tokens: Mapping[int, int]
    participant_ep_ids: tuple[int, ...]
    expert_to_ep: ExpertOwnership

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_per_expert_tokens",
            _freeze_map(self.global_per_expert_tokens),
        )
        object.__setattr__(
            self,
            "per_ep_per_expert_tokens",
            _freeze_nested_map(self.per_ep_per_expert_tokens),
        )
        object.__setattr__(
            self,
            "per_ep_routed_tokens",
            _freeze_map(self.per_ep_routed_tokens),
        )
        object.__setattr__(self, "expert_to_ep", _freeze_map(self.expert_to_ep))
        object.__setattr__(self, "participant_ep_ids", tuple(self.participant_ep_ids))


def resolve_routing_details(
    routing_details: RoutingDetails,
    target_replica_id: int,
    global_layer_id: int,
) -> dict[int, Real]:
    """Resolve one exact Replica/layer routing-ratio map.

    A copy is returned so architecture wrappers cannot mutate predictor-owned
    ``routing_details`` while materializing a layer workload.
    """

    if not isinstance(routing_details, Mapping):
        raise ValueError("routing_details must be a mapping")
    if target_replica_id not in routing_details:
        raise ValueError(
            "routing_details missing target_replica_id "
            f"{target_replica_id}"
        )
    replica_details = routing_details[target_replica_id]
    if not isinstance(replica_details, Mapping):
        raise ValueError(
            "routing_details target_replica_id entry must be a mapping"
        )
    if global_layer_id not in replica_details:
        raise ValueError(
            "routing_details missing global_layer_id "
            f"{global_layer_id} for target_replica_id {target_replica_id}"
        )
    layer_details = replica_details[global_layer_id]
    if not isinstance(layer_details, Mapping):
        raise ValueError("routing_details global_layer_id entry must be a mapping")
    return dict(layer_details)


def build_contiguous_expert_ownership(
    total_expert_num: int,
    moe_expert_parallel_size: int,
) -> dict[int, int]:
    """Build the current equal-size contiguous global-expert ownership map."""

    total_expert_num = _require_int(total_expert_num, "total_expert_num", minimum=1)
    moe_expert_parallel_size = _require_int(
        moe_expert_parallel_size,
        "moe_expert_parallel_size",
        minimum=1,
    )
    if total_expert_num % moe_expert_parallel_size != 0:
        raise ValueError(
            "contiguous equal-size ownership requires total_expert_num to be "
            "divisible by moe_expert_parallel_size"
        )
    experts_per_ep = total_expert_num // moe_expert_parallel_size
    return {
        expert_id: expert_id // experts_per_ep
        for expert_id in range(total_expert_num)
    }


def _validate_ownership(
    expert_to_ep: ExpertOwnership,
    *,
    total_expert_num: int,
    moe_expert_parallel_size: int,
) -> dict[int, int]:
    if not isinstance(expert_to_ep, Mapping):
        raise ValueError("expert ownership must be a mapping")
    expected = set(range(total_expert_num))
    if set(expert_to_ep) != expected:
        raise ValueError("ownership expert key set must equal all global expert IDs")
    for expert_id, ep_id in expert_to_ep.items():
        _require_int(expert_id, "ownership expert ID", minimum=0)
        _require_int(ep_id, "ownership EP ID", minimum=0)
        if ep_id >= moe_expert_parallel_size:
            raise ValueError("ownership EP ID is outside moe_expert_parallel_size")

    expected_ownership = build_contiguous_expert_ownership(
        total_expert_num,
        moe_expert_parallel_size,
    )
    if dict(expert_to_ep) != expected_ownership:
        raise ValueError(
            "expert ownership must be contiguous equal-size ownership"
        )
    return dict(expected_ownership)


def _validate_routing_ratios(
    routing_ratios: Mapping[int, Real],
    *,
    total_expert_num: int,
) -> dict[int, float]:
    if not isinstance(routing_ratios, Mapping) or not routing_ratios:
        raise ValueError("routing ratios must be a non-empty mapping")
    expected = set(range(total_expert_num))
    if set(routing_ratios) != expected:
        raise ValueError("routing ratios expert key set must equal all global expert IDs")

    values: dict[int, float] = {}
    for expert_id, ratio in routing_ratios.items():
        _require_int(expert_id, "routing expert ID", minimum=0)
        if isinstance(ratio, bool) or not isinstance(ratio, Real):
            raise ValueError("routing ratios must contain numeric values")
        value = float(ratio)
        if not isfinite(value):
            raise ValueError("routing ratios must be finite")
        if value < 0:
            raise ValueError("routing ratios must be non-negative")
        values[expert_id] = value

    ratio_sum = sum(values.values())
    if ratio_sum <= 0:
        raise ValueError("routing ratios must have a positive sum")
    return {expert_id: value / ratio_sum for expert_id, value in values.items()}


def materialize_layer_ep_workload(
    *,
    routing_ratios: Mapping[int, Real],
    target_replica_id: int,
    global_layer_id: int,
    routing_token_count: int,
    router_topk: int,
    total_expert_num: int,
    moe_expert_parallel_size: int,
    expert_to_ep: ExpertOwnership,
) -> LayerEPWorkload:
    """Materialize one exact aggregate workload for a Replica-local EP wave."""

    target_replica_id = _require_int(
        target_replica_id,
        "target_replica_id",
        minimum=0,
    )
    global_layer_id = _require_int(global_layer_id, "global_layer_id", minimum=0)
    routing_token_count = _require_int(
        routing_token_count,
        "routing_token_count",
        minimum=0,
    )
    router_topk = _require_int(router_topk, "router_topk", minimum=1)
    total_expert_num = _require_int(
        total_expert_num,
        "total_expert_num",
        minimum=1,
    )
    moe_expert_parallel_size = _require_int(
        moe_expert_parallel_size,
        "moe_expert_parallel_size",
        minimum=1,
    )
    ownership = _validate_ownership(
        expert_to_ep,
        total_expert_num=total_expert_num,
        moe_expert_parallel_size=moe_expert_parallel_size,
    )
    normalized_ratios = _validate_routing_ratios(
        routing_ratios,
        total_expert_num=total_expert_num,
    )

    total_routed_assignments = routing_token_count * router_topk
    quotas = {
        expert_id: total_routed_assignments * ratio
        for expert_id, ratio in normalized_ratios.items()
    }
    global_per_expert_tokens = {
        expert_id: int(floor(quota))
        for expert_id, quota in quotas.items()
    }
    remainder = total_routed_assignments - sum(global_per_expert_tokens.values())
    if remainder < 0 or remainder >= total_expert_num:
        raise ValueError("Hamilton remainder is outside the valid expert range")

    ranked_experts = sorted(
        quotas,
        key=lambda expert_id: (
            -(quotas[expert_id] - global_per_expert_tokens[expert_id]),
            expert_id,
        ),
    )
    for expert_id in ranked_experts[:remainder]:
        global_per_expert_tokens[expert_id] += 1

    if sum(global_per_expert_tokens.values()) != total_routed_assignments:
        raise ValueError("global expert token conservation failed")

    per_ep_per_expert_tokens: dict[int, dict[int, int]] = {
        ep_id: {} for ep_id in range(moe_expert_parallel_size)
    }
    for expert_id in range(total_expert_num):
        per_ep_per_expert_tokens[ownership[expert_id]][expert_id] = (
            global_per_expert_tokens[expert_id]
        )
    per_ep_routed_tokens = {
        ep_id: sum(expert_tokens.values())
        for ep_id, expert_tokens in per_ep_per_expert_tokens.items()
    }
    if sum(per_ep_routed_tokens.values()) != total_routed_assignments:
        raise ValueError("per-EP token conservation failed")

    return LayerEPWorkload(
        target_replica_id=target_replica_id,
        global_layer_id=global_layer_id,
        routing_token_count=routing_token_count,
        router_topk=router_topk,
        total_routed_assignments=total_routed_assignments,
        global_per_expert_tokens=global_per_expert_tokens,
        per_ep_per_expert_tokens=per_ep_per_expert_tokens,
        per_ep_routed_tokens=per_ep_routed_tokens,
        participant_ep_ids=tuple(range(moe_expert_parallel_size)),
        expert_to_ep=ownership,
    )


__all__ = [
    "ExpertOwnership",
    "ExpertTokenMap",
    "LayerEPWorkload",
    "RoutingDetails",
    "build_contiguous_expert_ownership",
    "materialize_layer_ep_workload",
    "resolve_routing_details",
]
