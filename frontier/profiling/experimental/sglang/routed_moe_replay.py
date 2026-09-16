"""Exact routed MoE replay inputs for experimental SGLang/AITER profiling.

Routing is supplied by Frontier's deterministic workload generator or by an
explicit expert-count JSON file.  The module never monkeypatches or observes a
live SGLang router and does not fit a cost model from route histograms.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from frontier.moe_ep_workload import generate_moe_routing_ratios

from .moe import reconstruct_topk_ids as _reconstruct_topk_ids


@dataclass(frozen=True)
class RoutedMoEInput:
    """Validated route histogram and its deterministic token assignments."""

    physical_size: int
    num_experts: int
    top_k: int
    expert_counts: tuple[int, ...]
    assignments: tuple[tuple[int, ...], ...]
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "physical_size": self.physical_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_counts": list(self.expert_counts),
            "assignments": [list(row) for row in self.assignments],
            "source": self.source,
        }


def frontier_routing_ratios(**kwargs: Any) -> dict[int, float]:
    """Call the shared simulator routing helper without changing its semantics."""

    return generate_moe_routing_ratios(**kwargs)


def reconstruct_topk_ids(
    physical_size: int,
    expert_counts: Sequence[int],
    spec: Mapping[str, Any] | None = None,
    *,
    num_experts: int | None = None,
    top_k: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Use the shared deterministic assignment rule with a convenient API."""

    if spec is None:
        if num_experts is None or top_k is None:
            raise ValueError("num_experts and top_k are required when spec is omitted")
        spec = {"num_experts": num_experts, "top_k": top_k, "block_size_m": 32}
    else:
        spec = dict(spec)
        if num_experts is not None and int(spec.get("num_experts", -1)) != num_experts:
            raise ValueError("num_experts disagrees with the supplied spec")
        if top_k is not None and int(spec.get("top_k", -1)) != top_k:
            raise ValueError("top_k disagrees with the supplied spec")
    return _reconstruct_topk_ids(physical_size, tuple(expert_counts), spec)


def _validate_counts(
    physical_size: int,
    expert_counts: Sequence[int],
    *,
    num_experts: int,
    top_k: int,
) -> tuple[int, ...]:
    if (
        type(physical_size) is not int
        or physical_size < 1
        or type(num_experts) is not int
        or num_experts < 1
        or type(top_k) is not int
        or not 1 <= top_k <= num_experts
    ):
        raise ValueError("physical_size, num_experts, and top_k are invalid")
    counts = tuple(expert_counts)
    if len(counts) != num_experts or any(
        type(value) is not int or value < 0 or value > physical_size
        for value in counts
    ):
        raise ValueError("expert counts must be non-negative integers in the expert domain")
    if sum(counts) != physical_size * top_k:
        raise ValueError(
            "total routed assignments must equal physical_size * top_k: "
            f"expected={physical_size * top_k}, got={sum(counts)}"
        )
    if sum(value > 0 for value in counts) < top_k:
        raise ValueError("expert counts cannot form unique per-token top-k assignments")
    return counts


def validate_topk_assignments(
    assignments: Sequence[Sequence[int]],
    expert_counts: Sequence[int],
    *,
    num_experts: int,
    top_k: int,
    physical_size: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Validate top-k conservation and per-token distinct-expert feasibility."""

    rows = tuple(tuple(row) for row in assignments)
    if physical_size is None:
        physical_size = len(rows)
    counts = _validate_counts(
        physical_size, expert_counts, num_experts=num_experts, top_k=top_k
    )
    if len(rows) != physical_size:
        raise ValueError("assignment row count does not equal physical_size")
    rebuilt = [0] * num_experts
    for row in rows:
        if len(row) != top_k or len(set(row)) != top_k:
            raise ValueError("each token must select exactly top_k distinct experts")
        if any(type(expert) is not int or not 0 <= expert < num_experts for expert in row):
            raise ValueError("assignment expert IDs are outside the expert domain")
        for expert in row:
            rebuilt[expert] += 1
    if tuple(rebuilt) != counts:
        raise ValueError(
            "top-k conservation failed: rebuilt expert counts disagree with the input histogram"
        )
    return rows


def build_routed_input(
    physical_size: int,
    expert_counts: Sequence[int],
    *,
    num_experts: int,
    top_k: int,
    source: str,
) -> RoutedMoEInput:
    counts = _validate_counts(
        physical_size, expert_counts, num_experts=num_experts, top_k=top_k
    )
    assignments = reconstruct_topk_ids(
        physical_size,
        counts,
        {"num_experts": num_experts, "top_k": top_k, "block_size_m": 32},
    )
    assignments = validate_topk_assignments(
        assignments,
        counts,
        num_experts=num_experts,
        top_k=top_k,
        physical_size=physical_size,
    )
    return RoutedMoEInput(
        physical_size=physical_size,
        num_experts=num_experts,
        top_k=top_k,
        expert_counts=counts,
        assignments=assignments,
        source=str(source),
    )


def load_expert_count_json(path: str | Path) -> dict[str, Any]:
    """Load a compact explicit route histogram without guessing missing fields."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("expert-count JSON must contain an object")
    required = {"num_experts", "top_k", "counts"}
    if set(payload) != required:
        raise ValueError("expert-count JSON requires exactly num_experts, top_k, and counts")
    num_experts, top_k = payload["num_experts"], payload["top_k"]
    if type(num_experts) is not int or type(top_k) is not int or not 1 <= top_k <= num_experts:
        raise ValueError("expert-count JSON has an invalid expert domain")
    counts = tuple(payload["counts"])
    if len(counts) != num_experts or any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("expert-count JSON counts must cover every expert with non-negative ints")
    return {"num_experts": num_experts, "top_k": top_k, "counts": counts}


def input_from_explicit_json(
    path: str | Path, *, physical_size: int | None = None
) -> RoutedMoEInput:
    payload = load_expert_count_json(path)
    counts = payload["counts"]
    if physical_size is None:
        total = sum(counts)
        if total % payload["top_k"]:
            raise ValueError("explicit expert counts do not encode an integral physical size")
        physical_size = total // payload["top_k"]
    return build_routed_input(
        physical_size,
        counts,
        num_experts=payload["num_experts"],
        top_k=payload["top_k"],
        source=f"explicit_json:{Path(path)}",
    )


def input_from_frontier_config(
    *,
    physical_size: int,
    total_expert_num: int,
    router_topk: int,
    distribution_type: str,
    seed: int,
    layer_id: int,
) -> RoutedMoEInput:
    """Materialize a route histogram with the exact Frontier ratio/integer rules."""

    ratios = frontier_routing_ratios(
        total_expert_num=total_expert_num,
        distribution_type=distribution_type,
        seed=seed,
        layer_id=layer_id,
    )
    raw = [ratios[expert] * physical_size * router_topk for expert in range(total_expert_num)]
    floors = [math.floor(value) for value in raw]
    remaining = physical_size * router_topk - sum(floors)
    order = sorted(
        range(total_expert_num), key=lambda expert: (-(raw[expert] - floors[expert]), expert)
    )
    for expert in order[:remaining]:
        floors[expert] += 1
    return build_routed_input(
        physical_size,
        floors,
        num_experts=total_expert_num,
        top_k=router_topk,
        source="frontier_config",
    )


def profile_routed_graph(query, count, repetitions, model, group, *, trace: bool = False):
    """Return the common replay row and optional trace-replay callback.

    Routed primitives use the same correctness checks, mutable-buffer resets,
    warmup and representative trace capture as the other isolated primitives.
    Heavy runtime imports remain behind the validated replay plan.
    """

    from .graph_replay import profile_graph, validate_plan
    from .moe import MOE_ROUTED_PRIMITIVES

    physical_size = query["physical_size"] if isinstance(query, Mapping) else query.physical_size
    validate_plan((physical_size,), (count,), repetitions, "validation")

    component = query["component"] if isinstance(query, Mapping) else query.component
    counts = query["physical_expert_counts"] if isinstance(query, Mapping) else query.expert_counts
    if component not in MOE_ROUTED_PRIMITIVES:
        raise ValueError("Unsupported routed primitive query")
    return profile_graph(
        component, physical_size, count, repetitions, model, group, group.rank_in_group,
        physical_expert_counts=tuple(counts), trace=trace,
    )


__all__ = [
    "RoutedMoEInput", "frontier_routing_ratios", "validate_topk_assignments",
    "build_routed_input", "load_expert_count_json", "input_from_explicit_json",
    "input_from_frontier_config", "profile_routed_graph",
]
