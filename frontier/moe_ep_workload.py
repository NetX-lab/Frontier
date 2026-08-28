"""Shared per-layer MoE expert-parallel workload materialization.

This module is intentionally pure.  It owns routing-ratio validation,
deterministic integerization, and Replica-local expert ownership splitting;
it does not own scheduler state, predictor caches, communication events, or
request accounting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
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


def _normalize_token_map(
    values: Mapping[int, int],
    name: str,
) -> dict[int, int]:
    """Validate and copy one sparse expert-token map."""

    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized: dict[int, int] = {}
    for expert_id, token_count in values.items():
        _require_int(expert_id, f"{name} expert ID", minimum=0)
        _require_int(token_count, f"{name} token count", minimum=0)
        normalized[expert_id] = token_count
    return normalized


def _require_exact_key_domain(
    values: Mapping[object, object],
    expected_keys: set[int],
    name: str,
) -> None:
    """Reject bool/numeric aliases before comparing mapping key domains."""

    if any(type(key) is not int or key < 0 for key in values):
        raise ValueError(f"{name} keys must be exact non-negative ints")
    if set(values) != expected_keys:
        raise ValueError(f"{name} keys do not match the canonical domain")


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
        target_replica_id = _require_int(
            self.target_replica_id,
            "target_replica_id",
            minimum=0,
        )
        global_layer_id = _require_int(
            self.global_layer_id,
            "global_layer_id",
            minimum=0,
        )
        routing_token_count = _require_int(
            self.routing_token_count,
            "routing_token_count",
            minimum=0,
        )
        router_topk = _require_int(self.router_topk, "router_topk", minimum=1)
        total_routed_assignments = _require_int(
            self.total_routed_assignments,
            "total_routed_assignments",
            minimum=0,
        )
        expected_assignments = routing_token_count * router_topk
        if total_routed_assignments != expected_assignments:
            raise ValueError(
                "total_routed_assignments must equal routing_token_count * "
                f"router_topk: expected={expected_assignments}, "
                f"got={total_routed_assignments}"
            )

        global_tokens = _normalize_token_map(
            self.global_per_expert_tokens,
            "global expert map",
        )
        if not global_tokens:
            raise ValueError("global expert map must contain at least one expert")
        total_expert_num = len(global_tokens)
        expected_expert_ids = set(range(total_expert_num))
        if set(global_tokens) != expected_expert_ids:
            raise ValueError(
                "global expert map keys must contain every global expert exactly once"
            )

        participant_ep_ids = tuple(self.participant_ep_ids)
        if not participant_ep_ids:
            raise ValueError("participant_ep_ids must contain at least one EP lane")
        if any(
            type(ep_id) is not int or ep_id < 0 for ep_id in participant_ep_ids
        ):
            raise ValueError("participant_ep_ids must contain exact non-negative ints")
        expected_participants = tuple(range(len(participant_ep_ids)))
        if participant_ep_ids != expected_participants:
            raise ValueError(
                "participant_ep_ids must be the canonical contiguous tuple: "
                f"expected={expected_participants}, got={participant_ep_ids}"
            )
        moe_expert_parallel_size = len(participant_ep_ids)
        if total_expert_num % moe_expert_parallel_size != 0:
            raise ValueError(
                "global expert count must be divisible by the participant EP size"
            )

        if not isinstance(self.expert_to_ep, Mapping):
            raise ValueError("expert_to_ep must be a mapping")
        _require_exact_key_domain(
            self.expert_to_ep,
            expected_expert_ids,
            "expert_to_ep",
        )
        ownership = {
            expert_id: _require_int(
                ep_id,
                "expert ownership EP ID",
                minimum=0,
            )
            for expert_id, ep_id in self.expert_to_ep.items()
        }
        if set(ownership) != expected_expert_ids:
            raise ValueError(
                "expert_to_ep keys must contain every global expert exactly once"
            )
        if any(ep_id >= moe_expert_parallel_size for ep_id in ownership.values()):
            raise ValueError("expert_to_ep contains an EP ID outside participant_ep_ids")
        expected_ownership = build_contiguous_expert_ownership(
            total_expert_num,
            moe_expert_parallel_size,
        )
        if ownership != expected_ownership:
            raise ValueError(
                "expert_to_ep must use canonical contiguous equal-size ownership"
            )

        if not isinstance(self.per_ep_per_expert_tokens, Mapping):
            raise ValueError("per_ep_per_expert_tokens must be a mapping")
        _require_exact_key_domain(
            self.per_ep_per_expert_tokens,
            set(participant_ep_ids),
            "per_ep_per_expert_tokens",
        )
        per_ep_tokens: dict[int, dict[int, int]] = {}
        for ep_id in participant_ep_ids:
            normalized = _normalize_token_map(
                self.per_ep_per_expert_tokens[ep_id],
                f"per-EP expert map ep_id={ep_id}",
            )
            for expert_id in normalized:
                if expert_id not in ownership:
                    raise ValueError(
                        "per-EP expert map contains an expert outside the global "
                        f"expert domain: ep_id={ep_id}, expert_id={expert_id}"
                    )
                if ownership[expert_id] != ep_id:
                    raise ValueError(
                        "per-EP expert map contains an expert outside its owned "
                        f"lane: ep_id={ep_id}, expert_id={expert_id}"
                    )
            per_ep_tokens[ep_id] = normalized

        for expert_id in sorted(expected_expert_ids):
            owner_ep_id = ownership[expert_id]
            local_count = per_ep_tokens[owner_ep_id].get(expert_id, 0)
            if local_count != global_tokens[expert_id]:
                raise ValueError(
                    "per-EP expert maps must agree with the global expert map: "
                    f"expert_id={expert_id}, global={global_tokens[expert_id]}, "
                    f"local={local_count}"
                )

        if not isinstance(self.per_ep_routed_tokens, Mapping):
            raise ValueError("per_ep_routed_tokens must be a mapping")
        _require_exact_key_domain(
            self.per_ep_routed_tokens,
            set(participant_ep_ids),
            "per_ep_routed_tokens",
        )
        per_ep_routed_tokens = {
            ep_id: _require_int(
                self.per_ep_routed_tokens[ep_id],
                f"per-EP routed token count ep_id={ep_id}",
                minimum=0,
            )
            for ep_id in participant_ep_ids
        }
        for ep_id in participant_ep_ids:
            expected_lane_tokens = sum(per_ep_tokens[ep_id].values())
            if per_ep_routed_tokens[ep_id] != expected_lane_tokens:
                raise ValueError(
                    "per-EP routed token count must equal its expert-map sum: "
                    f"ep_id={ep_id}, expected={expected_lane_tokens}, "
                    f"got={per_ep_routed_tokens[ep_id]}"
                )
        if sum(global_tokens.values()) != total_routed_assignments:
            raise ValueError(
                "global expert token sum must equal total_routed_assignments: "
                f"expected={total_routed_assignments}, got={sum(global_tokens.values())}"
            )
        if sum(per_ep_routed_tokens.values()) != total_routed_assignments:
            raise ValueError(
                "per-EP routed token sum must equal total_routed_assignments: "
                f"expected={total_routed_assignments}, "
                f"got={sum(per_ep_routed_tokens.values())}"
            )

        object.__setattr__(self, "target_replica_id", target_replica_id)
        object.__setattr__(self, "global_layer_id", global_layer_id)
        object.__setattr__(self, "routing_token_count", routing_token_count)
        object.__setattr__(self, "router_topk", router_topk)
        object.__setattr__(
            self,
            "total_routed_assignments",
            total_routed_assignments,
        )
        object.__setattr__(
            self,
            "global_per_expert_tokens",
            _freeze_map(global_tokens),
        )
        object.__setattr__(
            self,
            "per_ep_per_expert_tokens",
            _freeze_nested_map(per_ep_tokens),
        )
        object.__setattr__(
            self,
            "per_ep_routed_tokens",
            _freeze_map(per_ep_routed_tokens),
        )
        object.__setattr__(self, "expert_to_ep", _freeze_map(ownership))
        object.__setattr__(self, "participant_ep_ids", participant_ep_ids)

    def lane(self, ep_id: int) -> "EPLaneWorkload":
        """Return the canonical physical workload for one materialized lane."""

        if type(ep_id) is not int or ep_id < 0:
            raise ValueError("ep_id must be an exact non-negative int")
        if ep_id not in self.per_ep_per_expert_tokens:
            raise ValueError(
                f"ep_id={ep_id} is not present in the materialized EP workload"
            )
        total_expert_num = len(self.global_per_expert_tokens)
        moe_expert_parallel_size = len(self.participant_ep_ids)
        return _build_lane_descriptor(
            ep_id=ep_id,
            moe_expert_parallel_size=moe_expert_parallel_size,
            total_expert_num=total_expert_num,
            per_expert_tokens=self.per_ep_per_expert_tokens[ep_id],
            router_topk=self.router_topk,
        )


@dataclass(frozen=True)
class EPLaneWorkload:
    """Immutable physical workload for one local expert-parallel lane."""

    ep_id: int
    moe_expert_parallel_size: int
    total_expert_num: int
    owned_expert_ids: tuple[int, ...]
    local_token_counts: tuple[int, ...]
    routed_token_count: int
    router_topk: int

    def __post_init__(self) -> None:
        if type(self.ep_id) is not int or self.ep_id < 0:
            raise ValueError("ep_id must be an exact non-negative int")
        if (
            type(self.moe_expert_parallel_size) is not int
            or self.moe_expert_parallel_size <= 0
        ):
            raise ValueError("moe_expert_parallel_size must be an exact positive int")
        if type(self.total_expert_num) is not int or self.total_expert_num <= 0:
            raise ValueError("total_expert_num must be an exact positive int")
        if self.ep_id >= self.moe_expert_parallel_size:
            raise ValueError("ep_id is outside moe_expert_parallel_size")
        if self.total_expert_num % self.moe_expert_parallel_size != 0:
            raise ValueError(
                "total_expert_num must be divisible by moe_expert_parallel_size"
            )
        expected_width = self.total_expert_num // self.moe_expert_parallel_size
        owned_expert_ids = tuple(self.owned_expert_ids)
        expected_owned_ids = tuple(
            range(self.ep_id * expected_width, (self.ep_id + 1) * expected_width)
        )
        if owned_expert_ids != expected_owned_ids:
            raise ValueError(
                "owned_expert_ids do not match canonical contiguous ownership: "
                f"expected={expected_owned_ids}, got={owned_expert_ids}"
            )
        local_token_counts = tuple(self.local_token_counts)
        if len(local_token_counts) != expected_width:
            raise ValueError(
                "local_token_counts must have the fixed local expert width: "
                f"expected={expected_width}, got={len(local_token_counts)}"
            )
        if any(type(value) is not int or value < 0 for value in local_token_counts):
            raise ValueError(
                "local_token_counts must contain exact non-negative integers"
            )
        if type(self.routed_token_count) is not int or self.routed_token_count < 0:
            raise ValueError(
                "routed_token_count must be an exact non-negative integer"
            )
        if sum(local_token_counts) != self.routed_token_count:
            raise ValueError(
                "routed_token_count must equal the sum of local_token_counts: "
                f"routed={self.routed_token_count}, sum={sum(local_token_counts)}"
            )
        if type(self.router_topk) is not int or self.router_topk <= 0:
            raise ValueError("router_topk must be an exact positive integer")
        object.__setattr__(self, "owned_expert_ids", owned_expert_ids)
        object.__setattr__(self, "local_token_counts", local_token_counts)

    @property
    def local_expert_width(self) -> int:
        """Return the configured number of experts owned by this lane."""

        return self.total_expert_num // self.moe_expert_parallel_size

    @property
    def per_expert_tokens(self) -> ExpertTokenMap:
        """Return a read-only compatibility view keyed by global expert ID."""

        return _freeze_map(dict(zip(self.owned_expert_ids, self.local_token_counts)))

    @property
    def num_experts_per_device(self) -> int:
        """Compatibility alias for the canonical local expert width."""

        return self.local_expert_width

    def materialize_expert_token_counts(self) -> tuple[int, ...]:
        """Return the fixed-width local token vector."""

        return self.local_token_counts


def _build_lane_descriptor(
    *,
    ep_id: int,
    moe_expert_parallel_size: int,
    total_expert_num: int,
    per_expert_tokens: Mapping[int, int],
    router_topk: int,
) -> EPLaneWorkload:
    """Build one descriptor and densify its canonical local expert map."""

    ep_id = _require_int(ep_id, "ep_id", minimum=0)
    moe_expert_parallel_size = _require_int(
        moe_expert_parallel_size,
        "moe_expert_parallel_size",
        minimum=1,
    )
    total_expert_num = _require_int(
        total_expert_num,
        "total_expert_num",
        minimum=1,
    )
    router_topk = _require_int(router_topk, "router_topk", minimum=1)
    if total_expert_num % moe_expert_parallel_size != 0:
        raise ValueError(
            "total_expert_num must be divisible by moe_expert_parallel_size"
        )
    if ep_id >= moe_expert_parallel_size:
        raise ValueError("ep_id is outside moe_expert_parallel_size")
    if not isinstance(per_expert_tokens, Mapping):
        raise ValueError("per_expert_tokens must be a mapping")
    width = total_expert_num // moe_expert_parallel_size
    owned_ids = tuple(range(ep_id * width, (ep_id + 1) * width))
    owned_id_set = set(owned_ids)
    normalized: dict[int, int] = {}
    for expert_id, token_count in per_expert_tokens.items():
        if type(expert_id) is not int or expert_id < 0:
            raise ValueError("lane expert IDs must be exact non-negative integers")
        if type(token_count) is not int or token_count < 0:
            raise ValueError(
                "lane expert token counts must be exact non-negative integers"
            )
        if expert_id not in owned_id_set:
            raise ValueError(
                "lane workload contains an expert outside canonical ownership: "
                f"ep_id={ep_id}, expert_id={expert_id}"
            )
        normalized[expert_id] = token_count
    local_token_counts = tuple(normalized.get(expert_id, 0) for expert_id in owned_ids)
    return EPLaneWorkload(
        ep_id=ep_id,
        moe_expert_parallel_size=moe_expert_parallel_size,
        total_expert_num=total_expert_num,
        owned_expert_ids=owned_ids,
        local_token_counts=local_token_counts,
        routed_token_count=sum(local_token_counts),
        router_topk=router_topk,
    )


def split_global_expert_tokens_into_lanes(
    global_per_expert_tokens: Mapping[int, int],
    *,
    total_expert_num: int,
    moe_expert_parallel_size: int,
    router_topk: int = 1,
) -> tuple[EPLaneWorkload, ...]:
    """Split a complete global expert map using canonical contiguous ownership."""

    total_expert_num = _require_int(total_expert_num, "total_expert_num", minimum=1)
    moe_expert_parallel_size = _require_int(
        moe_expert_parallel_size,
        "moe_expert_parallel_size",
        minimum=1,
    )
    ownership = build_contiguous_expert_ownership(
        total_expert_num,
        moe_expert_parallel_size,
    )
    if not isinstance(global_per_expert_tokens, Mapping):
        raise ValueError("global_per_expert_tokens must be a mapping")
    if set(global_per_expert_tokens) != set(range(total_expert_num)):
        raise ValueError(
            "global expert map must contain every global expert exactly once"
        )
    for expert_id, token_count in global_per_expert_tokens.items():
        if type(expert_id) is not int or expert_id < 0:
            raise ValueError("global expert IDs must be exact non-negative integers")
        if type(token_count) is not int or token_count < 0:
            raise ValueError(
                "global expert token counts must be exact non-negative integers"
            )
        if expert_id not in ownership:
            raise ValueError(
                "global expert ID is outside total_expert_num: "
                f"expert_id={expert_id}"
            )
    per_ep: dict[int, dict[int, int]] = {
        ep_id: {
            expert_id: int(global_per_expert_tokens[expert_id])
            for expert_id, owner_ep_id in ownership.items()
            if owner_ep_id == ep_id
        }
        for ep_id in range(moe_expert_parallel_size)
    }
    return tuple(
        _build_lane_descriptor(
            ep_id=ep_id,
            moe_expert_parallel_size=moe_expert_parallel_size,
            total_expert_num=total_expert_num,
            per_expert_tokens=per_ep[ep_id],
            router_topk=router_topk,
        )
        for ep_id in range(moe_expert_parallel_size)
    )


def resolve_ep_lane_workload(
    source: object,
    *,
    required: bool = True,
) -> EPLaneWorkload | None:
    """Resolve the canonical physical lane descriptor from a boundary value.

    Callers may pass the descriptor itself or an entity that exposes the
    descriptor through ``lane_workload``.  The helper deliberately does not
    accept a raw expert-token mapping: aggregate and lane-local workloads have
    different physical domains and must be materialized before prediction.
    """

    if isinstance(source, EPLaneWorkload):
        return source
    missing = object()
    try:
        static_candidate = inspect.getattr_static(source, "lane_workload")
    except AttributeError:
        static_candidate = missing
    if static_candidate is missing:
        if required:
            raise ValueError(
                "an EPLaneWorkload descriptor is required for EP-lane workload "
                "consumption"
            )
        return None
    candidate = (
        getattr(source, "lane_workload")
        if isinstance(static_candidate, property)
        else static_candidate
    )
    if candidate is None:
        if required:
            raise ValueError(
                "an EPLaneWorkload descriptor is required for EP-lane workload "
                "consumption"
            )
        return None
    if not isinstance(candidate, EPLaneWorkload):
        raise TypeError(
            "lane_workload must be an EPLaneWorkload descriptor, got "
            f"{type(candidate).__name__}"
        )
    return candidate


def resolve_routing_details(
    routing_details: RoutingDetails,
    target_replica_id: int,
    global_layer_id: int,
) -> dict[int, Real]:
    """Resolve one exact Replica/layer routing-ratio map.

    A copy is returned so architecture wrappers cannot mutate predictor-owned
    ``routing_details`` while materializing a layer workload.
    """

    target_replica_id = _require_int(
        target_replica_id,
        "target_replica_id",
        minimum=0,
    )
    global_layer_id = _require_int(
        global_layer_id,
        "global_layer_id",
        minimum=0,
    )
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
    "EPLaneWorkload",
    "ExpertOwnership",
    "ExpertTokenMap",
    "LayerEPWorkload",
    "RoutingDetails",
    "build_contiguous_expert_ownership",
    "materialize_layer_ep_workload",
    "resolve_ep_lane_workload",
    "resolve_routing_details",
    "split_global_expert_tokens_into_lanes",
]
