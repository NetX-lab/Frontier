"""CPU planning for balanced routed MoE ranks and SP-local gate rows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from collections import Counter
from collections.abc import Iterator

from frontier.moe_ep_workload import materialize_balanced_rank_classes

CLASS_COLUMNS = ("local_routed_rows", "local_experts_with_extra_row", "base_rows_per_expert")


def parse_dp_lane_local_rows(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("DP lane rows must be comma-separated integers") from exc
    if not rows or any(row <= 0 for row in rows):
        raise argparse.ArgumentTypeError("DP lane rows must be positive")
    return rows


def validate_lane_rows_args(args: argparse.Namespace) -> None:
    if not getattr(args, "dp_lane_local_rows", None):
        return
    if args.num_tokens_list is not None or args.extra_num_tokens is not None:
        raise ValueError("--dp_lane_local_rows replaces --num_tokens_list and --extra_num_tokens")
    if args.expert_parallel_sizes is None or len(args.expert_parallel_sizes) != 1:
        raise ValueError("--dp_lane_local_rows requires exactly one EP size")
    if args.num_tensor_parallel_workers != [1]:
        raise ValueError("--dp_lane_local_rows requires --num_tensor_parallel_workers 1")
    if not args.enable_load_imbalance or args.load_distributions != ["balanced"]:
        raise ValueError("--dp_lane_local_rows requires --load_distributions balanced")
    if args.routing_runtime_path != "balanced_topk":
        raise ValueError("--dp_lane_local_rows requires --routing_runtime_path balanced_topk")
    ep = args.expert_parallel_sizes[0]
    if ep <= 0 or any(ep % len(rows) for rows in args.dp_lane_local_rows):
        raise ValueError("EP must be positive and divisible by each DP lane vector length")


@dataclass(frozen=True)
class MoERankTask:
    expert_parallel_size: int
    num_tokens: int
    ep_rank: int
    gating_num_tokens: int
    local_routed_rows: int
    local_experts_with_extra_row: int
    base_rows_per_expert: int
    dp_lane_local_rows: tuple[int, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["dp_lane_local_rows"] = list(self.dp_lane_local_rows)
        return result


def distinct_lane_permutations(rows: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Enumerate each placement once, including vectors with repeated widths."""
    remaining = Counter(rows)
    prefix = []

    def visit():
        if len(prefix) == len(rows):
            yield tuple(prefix)
            return
        for width in sorted(remaining):
            if remaining[width]:
                remaining[width] -= 1
                prefix.append(width)
                yield from visit()
                prefix.pop()
                remaining[width] += 1

    yield from visit()


def plan_dp_lane_rows(
    lane_vectors: list[tuple[int, ...]], *, expert_parallel_size: int,
    num_experts: int, router_topk: int,
) -> list[MoERankTask]:
    """Take all lane placements, selecting the lowest rank of each (EP,M,class,L).

    Ranks are DP-major. The retained vector witnesses the representative rank's
    L; permutations affect L but never the gathered routing population M.
    """
    tasks = {}
    for rows in lane_vectors:
        if not rows or any(row <= 0 for row in rows):
            raise ValueError("DP lane rows must be positive")
        if expert_parallel_size <= 0 or expert_parallel_size % len(rows):
            raise ValueError("EP must be positive and divisible by the DP lane vector length")
        ranks_per_lane = expert_parallel_size // len(rows)
        gathered = ranks_per_lane * sum(rows)
        classes = materialize_balanced_rank_classes(
            routing_token_count=gathered, router_topk=router_topk,
            total_expert_num=num_experts, moe_expert_parallel_size=expert_parallel_size,
        )
        for placement in distinct_lane_permutations(rows):
            for rank, rank_class in enumerate(classes):
                local = placement[rank // ranks_per_lane]
                key = (expert_parallel_size, gathered, rank_class, local)
                task = MoERankTask(expert_parallel_size, gathered, rank, local,
                                   *rank_class, placement)
                previous = tasks.get(key)
                if previous is None or rank < previous.ep_rank:
                    tasks[key] = task
    return [tasks[key] for key in sorted(tasks)]
