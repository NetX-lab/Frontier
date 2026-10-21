"""CUDA graph families and the shared token capture resolver."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphPolicy:
    uniform_decode: str
    other: str


GRAPH_POLICIES = {
    "none": GraphPolicy("NONE", "NONE"),
    "full_decode_only": GraphPolicy("FULL", "NONE"),
    "piecewise": GraphPolicy("PIECEWISE", "PIECEWISE"),
    "full_and_piecewise": GraphPolicy("FULL", "PIECEWISE"),
}
GRAPH_MODE_ORDER = {"NONE": 0, "PIECEWISE": 1, "FULL": 2}


def graph_policy(mode):
    try:
        return GRAPH_POLICIES[mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported CUDA graph policy: {mode!r}") from exc


def resolve_capture_size(total_tokens, capture_sizes):
    for size in sorted(capture_sizes):
        if total_tokens <= size:
            return True, int(size)
    return False, int(total_tokens)
