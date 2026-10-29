"""Step owners in the existing integer-layer replay record format.

The reserved layer -1 means the whole runner step; real layer IDs are >= 0.
Readers must test both fields, so malformed owners never disappear silently.
"""
from __future__ import annotations

STEP_LAYER_ID = -1
ZERO_SCOPE = "kv_cache_block_zero"
WRITE_SCOPE = "dsv4_block_table_write"
INPUT_SCOPE = "dsv4_input_preparation"
SAMPLING_SCOPE = "dsv4_sampling"
LOGITS_SCOPE = "dsv4_compute_logits"
STEP_SCOPES = (ZERO_SCOPE, WRITE_SCOPE, INPUT_SCOPE, SAMPLING_SCOPE, LOGITS_SCOPE)


def is_step_scope(row):
    scope, layer = row.get("scope"), row.get("global_layer_id")
    if scope in STEP_SCOPES or layer == STEP_LAYER_ID:
        if scope not in STEP_SCOPES or layer != STEP_LAYER_ID:
            raise ValueError(f"Step scope requires owner ({STEP_LAYER_ID}, step scope), got {(layer, scope)!r}")
        return True
    return False


def attention_owned(row):
    return row.get("scope") is not None and not is_step_scope(row)
