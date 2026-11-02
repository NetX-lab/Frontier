"""The producer's balanced strategy pin, shared with native attention rows."""

import hashlib
from pathlib import Path

from frontier.profiling.attention.dsv4_attention_rows import BALANCED_ROUTING_SHA256


def balanced_routing_columns() -> dict[str, str]:
    path = Path(__file__).with_name("balanced_routing.py")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BALANCED_ROUTING_SHA256:
        raise ValueError(f"Balanced routing strategy SHA-256 mismatch: {path}")
    return {"routing_strategy_sha256": digest}
