"""CUDA graph families and the shared token capture resolver."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphPolicy:
    uniform_decode: str
    other: str

    def resolve_capture(self, total_tokens, capture_sizes, *, uniform_decode, request_capacity):
        """Select a no-MTP graph whose token and request dimensions were captured."""
        candidates = (self.uniform_decode, self.other) if uniform_decode else (self.other,)
        for mode in candidates:
            if mode == "NONE":
                continue
            sizes = (size for size in capture_sizes if mode != "FULL" or size <= request_capacity)
            hit, width = resolve_capture_size(total_tokens, sizes)
            if hit:
                return mode, width
        return "NONE", int(total_tokens)


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
