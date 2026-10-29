"""C1 section 4.1 attention classes and compression targets, shared on CPU.

The seven multi-tail coordinates are single observed windows, not a Cartesian
extension of the 39-class serving grid. FULL W1 dummy is shared by the two roles.
The context allowance and tail mixes follow decision D-G1H-GRID-AMENDMENT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

from frontier.profiling.attention.dsa_plan import (
    DP_FULL, DP_NONE, DP_PIECEWISE, PHASE_DUMMY, PHASE_FIRST_TAIL, PHASE_PREFILL,
)

CATEGORY_MULTI_TAIL = "multi_tail"

C1 = {
    "decision": "D-G1H-GRID-AMENDMENT",
    "widths": (1, 2, 4, 8, 16, 24),
    "margin_min_own": 17,
    "cached_low": 6144,
    "cached_high": 7166,
    "prefill_query": 6144,
    "ratios": (4, 128),
    "profiles": ("low", "high", "spread"),
    "multi_tail": ((2, 2), (3, 2), (3, 3), (4, 4), (5, 2), (7, 2), (8, 3)),
    "max_writes": {
        1: (1, 1), 2: (2, 2), 3: (3, 3), 4: (4, 4), 5: (4, 2),
        6: (5, 3), 7: (6, 3), 8: (7, 4), 9: (6, 2), 10: (7, 2),
        11: (7, 2), 12: (7, 2), 13: (6, 1),
    },
    "unobserved_max_writes": (7, 4),
}
# D-G1H-GRID-AMENDMENT: near means within one compression group.
C1["context_tolerance"] = max(C1["ratios"]) - 1


def max_writes(own: int) -> tuple[int, int]:
    return C1["max_writes"].get(own, tuple(min(own, k) for k in C1["unobserved_max_writes"]))


@cache
def serving_classes() -> tuple[tuple[int, int], ...]:
    return tuple((width, own) for width in C1["widths"]
                 for own in range(C1["margin_min_own"] if width == max(C1["widths"]) else 1, width + 1))


def natural_width(own: int) -> int:
    return next(width for width in C1["widths"] if width >= own)


@dataclass(frozen=True)
class AttentionWindow:
    category: str
    mode: str
    width: int
    own: int
    tails: int
    profile: str | None
    mix: tuple[int, int] | None

    @property
    def requests(self) -> int:
        return self.width if self.mode == DP_FULL else max(self.own, 1)

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "requests": self.requests}


@cache
def required_windows() -> tuple[AttentionWindow, ...]:
    windows = []
    for category, mode, tails in ((DP_FULL, DP_FULL, 0), (DP_PIECEWISE, DP_PIECEWISE, 0),
                                  (PHASE_FIRST_TAIL, DP_PIECEWISE, 1)):
        for width, own in serving_classes():
            k4, k128 = max_writes(own)
            for profile in C1["profiles"]:
                for mix in ((0, 0), (k4, 0), (k128, k128)):
                    mix = tuple(max(count, tails) for count in mix)
                    windows.append(AttentionWindow(category, mode, width, own, tails, profile, mix))
    windows.extend(AttentionWindow(CATEGORY_MULTI_TAIL, DP_PIECEWISE, natural_width(own), own,
                                   tails, "low", (tails, tails))
                   for own, tails in C1["multi_tail"])
    windows.extend(AttentionWindow(PHASE_DUMMY, mode, width, 0, 0, None, None)
                   for mode in (DP_FULL, DP_PIECEWISE) for width in C1["widths"])
    windows.append(AttentionWindow(PHASE_DUMMY, DP_NONE, 1, 0, 0, None, None))
    windows.append(AttentionWindow(PHASE_PREFILL, DP_NONE, C1["prefill_query"], 1, 0, None, (1, 1)))
    return tuple(dict.fromkeys(windows))


def profile_targets(profile: str, count: int) -> tuple[int, ...]:
    low, high = C1["cached_low"], C1["cached_high"]
    if profile in ("low", "high"):
        return (low if profile == "low" else high,) * count
    # The one-request spread degenerates to the low endpoint. With no continuing
    # request (tail alone), all profiles share the same physical coordinate.
    return tuple(round(low + (high - low) * i / max(count - 1, 1)) for i in range(count))


def matching_profiles(requests: list[dict[str, int]]) -> tuple[str, ...]:
    contexts = sorted(r["num_computed_tokens"] for r in requests
                      if r["num_computed_tokens"] >= r["prompt_len"])
    if any(c < C1["cached_low"] or c > C1["cached_high"] for c in contexts):
        return ()
    return tuple(profile for profile in C1["profiles"]
                 if all(abs(c - target) <= C1["context_tolerance"]
                        for c, target in zip(contexts, profile_targets(profile, len(contexts)))))


def describe_step(step: dict[str, Any]) -> dict[str, Any]:
    rows, dispatch = step["requests"], step["cudagraph_dispatch"]
    tails = [i for i, r in enumerate(rows) if r["num_scheduled_tokens"] == 1
             and r["num_computed_tokens"] + 1 == r["prompt_len"]]
    writers = {str(ratio): [i for i, r in enumerate(rows)
                           if (r["num_computed_tokens"] + r["num_scheduled_tokens"]) // ratio
                           > r["num_computed_tokens"] // ratio] for ratio in C1["ratios"]}
    own = len(rows)
    mode, width = dispatch["cg_mode"], dispatch["num_tokens"]
    category = (PHASE_DUMMY if step["phase"] == PHASE_DUMMY else PHASE_PREFILL if step["phase"] == PHASE_PREFILL
                else CATEGORY_MULTI_TAIL if len(tails) > 1 else PHASE_FIRST_TAIL if tails else mode)
    attention_rows = [{"cached_tokens": r["num_computed_tokens"],
                       "query_tokens": r["num_scheduled_tokens"]} for r in rows]
    if category == PHASE_DUMMY:
        attention_rows = ([{"cached_tokens": 0, "query_tokens": 1} for _ in range(width)]
                          if mode == DP_FULL else [{"cached_tokens": 0, "query_tokens": width}])
    elif mode == DP_FULL:
        attention_rows.extend({"cached_tokens": 0, "query_tokens": 1, "padding": True}
                              for _ in range(width - own))
    return {"category": category, "mode": mode, "width": width,
            "requests": dispatch["num_reqs"] if mode == DP_FULL else max(own, 1), "own": own,
            "own_requests": list(range(own)), "first_tails": tails,
            "cached_tokens": [r["num_computed_tokens"] for r in rows],
            "query_tokens": [r["num_scheduled_tokens"] for r in rows],
            "attention_request_states": attention_rows,
            "padding_cached_tokens": [0] * (width - own) if mode == DP_FULL else [],
            "writers": writers, "mix": tuple(len(writers[str(r)]) for r in C1["ratios"])}


def assign_windows(step: dict[str, Any]) -> tuple[AttentionWindow, ...]:
    point = describe_step(step)
    if point["category"] not in (PHASE_DUMMY, PHASE_PREFILL) and any(
            q != 1 for q in point["query_tokens"]):
        return ()
    if point["category"] not in (PHASE_DUMMY, PHASE_PREFILL) and any(
            r["num_computed_tokens"] < r["prompt_len"] and i not in point["first_tails"]
            for i, r in enumerate(step["requests"])):
        return ()
    if any(step["requests"][i]["num_computed_tokens"] != C1["prefill_query"] - 1
           for i in point["first_tails"]):
        return ()
    if point["category"] == PHASE_PREFILL and (
            point["cached_tokens"] != [0] or point["query_tokens"] != [C1["prefill_query"]]):
        return ()
    profiles = matching_profiles(step["requests"])
    return tuple(window for window in required_windows()
                 if (window.category, window.mode, window.width, window.requests, window.own, window.tails)
                 == (point["category"], point["mode"], point["width"], point["requests"],
                     point["own"], len(point["first_tails"]))
                 and (window.profile is None or window.profile in profiles)
                 and (window.mix is None or window.mix == point["mix"]))


def native_class_in_grid(record: dict[str, Any]) -> bool:
    """Reference counters carry has_prefill, not the number of first tails."""
    mode, width, own = record["mode"], record["W"], record["own_reqs"]
    return ((width, own) in serving_classes()
            and record["own_tokens"] == own
            and ((mode == DP_FULL and record["R"] == width and not record["has_prefill"])
                 or (mode == DP_PIECEWISE and record["R"] == own)))
