"""Routing-count policies and immutable request-owner source frames."""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable


class WorkloadDistributionType(Enum):
    BALANCED = "balanced"
    RANDOM = "random"
    SKEWED = "skewed"
    ZIPF = "zipf"
    POSITION_CYCLIC = "position_cyclic"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact int >= {minimum}")


@dataclass(frozen=True)
class TokenInterval:
    start: int
    length: int

    def __post_init__(self):
        _integer(self.start, "interval start")
        _integer(self.length, "interval length", 1)


@dataclass(frozen=True)
class SourceLaneRouting:
    lane_id: int
    source_batch_id: int
    source_global_id: int
    forward_group_id: int
    graph_mode: str
    gating_runtime_context: str
    physical_rows: int
    live_rows: int
    intervals: tuple[TokenInterval, ...]

    def __post_init__(self):
        from frontier.moe_gating_runtime import validate_moe_gating_runtime_context
        for name in ("lane_id", "source_batch_id", "forward_group_id", "live_rows"):
            _integer(getattr(self, name), name)
        _integer(self.source_global_id, "source_global_id", -1)
        _integer(self.physical_rows, "physical_rows", 1)
        if self.graph_mode not in ("NONE", "PIECEWISE", "FULL"):
            raise ValueError("source graph_mode must be NONE, PIECEWISE or FULL")
        validate_moe_gating_runtime_context(self.gating_runtime_context)
        intervals = tuple(self.intervals)
        if any(not isinstance(interval, TokenInterval) for interval in intervals):
            raise ValueError("source intervals must be TokenInterval values")
        if sum(interval.length for interval in intervals) != self.live_rows or self.live_rows > self.physical_rows:
            raise ValueError("source live rows must equal interval lengths and fit physical rows")
        object.__setattr__(self, "intervals", intervals)


@dataclass(frozen=True)
class WaveRoutingSource:
    policy_id: str
    source_lanes: tuple[SourceLaneRouting, ...]
    gathered_physical_rows: int

    def __post_init__(self):
        policy = resolve_routing_count_policy(self.policy_id)
        if not policy.requires_source:
            raise ValueError("WaveRoutingSource requires a source-based policy")
        lanes = tuple(self.source_lanes)
        if not lanes or any(not isinstance(lane, SourceLaneRouting) for lane in lanes):
            raise ValueError("source_lanes must contain immutable source lanes")
        if tuple(lane.lane_id for lane in lanes) != tuple(range(len(lanes))):
            raise ValueError("source lanes must be in contiguous DP order")
        if len({lane.forward_group_id for lane in lanes}) != 1:
            raise ValueError("source lanes must belong to one admitted forward group")
        _integer(self.gathered_physical_rows, "gathered_physical_rows", 1)
        if self.gathered_physical_rows != sum(lane.physical_rows for lane in lanes):
            raise ValueError("gathered physical rows must equal source widths in no-SP scope")
        object.__setattr__(self, "source_lanes", lanes)

    @property
    def live_rows(self):
        return sum(lane.live_rows for lane in self.source_lanes)


def build_wave_routing_source(source_batches, *, policy_id, cluster_type, gathered_physical_rows, model_config=None):
    """Freeze only explicit, forward-bound request-owner batches, never EP copies."""
    from frontier.entities.batch import Batch, DummyForwardBatch
    from frontier.moe_gating_runtime import (
        DEFAULT_MOE_GATING_RUNTIME_CONTEXT, PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
        should_use_prefill_hot_moe_gating_context,
    )
    lanes = []
    for lane_id, batch in sorted(source_batches.items()):
        if not isinstance(batch, Batch) or getattr(batch, "lane_workload", None) is not None:
            raise ValueError("routing source must contain actual request-owner batches")
        binding = getattr(batch, "forward_start_descriptor", None)
        if binding is None:
            raise ValueError("routing source requires admitted forward_start_descriptor")
        if batch.get_effective_total_tokens_for_compute(cluster_type) != binding.num_tokens:
            raise ValueError("source physical width differs from its forward binding")
        intervals = ()
        if not isinstance(batch, DummyForwardBatch):
            queries, contexts = tuple(batch.num_tokens), tuple(batch.num_context_tokens)
            if len(batch.requests) != len(queries) or len(queries) != len(contexts) or sum(queries) != batch.total_num_tokens:
                raise ValueError("source request/query/context arrays disagree")
            intervals = tuple(TokenInterval(context, query) for context, query in zip(contexts, queries))
        hot = should_use_prefill_hot_moe_gating_context(batch=batch, model_config=model_config)
        lanes.append(SourceLaneRouting(
            lane_id, batch.id, batch.global_id, binding.group_id, binding.graph_mode,
            PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT if hot else DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
            binding.num_tokens, sum(interval.length for interval in intervals), intervals,
        ))
    return WaveRoutingSource(policy_id, tuple(lanes), gathered_physical_rows)


def position_cyclic_expert_counts(intervals, top_k, total_expert_num):
    """Count the consecutive assignment residues of each logical interval."""
    _integer(top_k, "top_k", 1)
    _integer(total_expert_num, "total_expert_num", 1)
    if top_k > total_expert_num:
        raise ValueError("top_k must not exceed total_expert_num")
    counts = dict.fromkeys(range(total_expert_num), 0)
    for interval in intervals:
        if not isinstance(interval, TokenInterval):
            raise ValueError("counts require TokenInterval values")
        base, remainder = divmod(interval.length * top_k, total_expert_num)
        start = interval.start * top_k % total_expert_num
        for expert in counts:
            counts[expert] += base
        for offset in range(remainder):
            counts[(start + offset) % total_expert_num] += 1
    return counts


def _ratio_counts(*, routing_ratios, routing_source, routing_token_count, router_topk, total_expert_num):
    from frontier.moe_ep_workload import materialize_expert_token_counts
    if routing_source is not None:
        raise ValueError("ratio counts do not accept a routing source")
    return materialize_expert_token_counts(
        routing_ratios=routing_ratios, total_routed_assignments=routing_token_count * router_topk,
        total_expert_num=total_expert_num,
    )


def _position_counts(*, routing_ratios, routing_source, routing_token_count, router_topk, total_expert_num):
    if not isinstance(routing_source, WaveRoutingSource) or routing_ratios is not None:
        raise ValueError("position counts require an explicit source, not ratios")
    if routing_source.gathered_physical_rows != routing_token_count:
        raise ValueError("routing source and gathered physical width disagree")
    return position_cyclic_expert_counts(
        (interval for lane in routing_source.source_lanes for interval in lane.intervals),
        router_topk, total_expert_num,
    )


@dataclass(frozen=True)
class RoutingCountPolicy:
    runtime_path: str
    requires_source: bool
    count_experts: Callable


RATIO_COUNT_POLICY = RoutingCountPolicy("standard_fused_topk", False, _ratio_counts)
ROUTING_COUNT_POLICIES = MappingProxyType({
    "balanced": RATIO_COUNT_POLICY,
    "random": RoutingCountPolicy("uniform_topk", False, _ratio_counts),
    "skewed": RATIO_COUNT_POLICY,
    "zipf": RATIO_COUNT_POLICY,
    "position_cyclic": RoutingCountPolicy("position_cyclic", True, _position_counts),
})


def resolve_routing_count_policy(distribution_type):
    name = str(distribution_type).strip().lower()
    if name not in ROUTING_COUNT_POLICIES:
        raise ValueError(f"Unsupported moe_routing_distribution_type={distribution_type!r}; expected {sorted(ROUTING_COUNT_POLICIES)}")
    return ROUTING_COUNT_POLICIES[name]
