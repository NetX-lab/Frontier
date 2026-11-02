from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from frontier.spec_decode.model_identity import resolve_canonical_model_key
from frontier.spec_decode.mtp_registry import get_mtp_model_alias_map


@dataclass(frozen=True)
class DecodeDraftProposerLatencyProfileEntry:
    method: str
    model_name: str
    attn_tp_size: int
    num_speculative_tokens: int
    spec_verify_request_count: int
    latency_ms: float


def normalize_decode_draft_proposer_model_name(model_name: str) -> str:
    return resolve_canonical_model_key(
        model_name,
        explicit_alias_map=get_mtp_model_alias_map(),
    )


def load_decode_draft_proposer_latency_profile(
    *,
    profile_file: str,
    supported_methods: set[str],
) -> Dict[Tuple[str, str, int, int, int], float]:
    if not profile_file:
        return {}
    if not os.path.isfile(profile_file):
        raise ValueError(
            "SpeculativeDecodingConfig.decode_draft_proposer_latency_profile_file "
            f"does not exist: {profile_file!r}"
        )
    try:
        with open(profile_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "SpeculativeDecodingConfig.decode_draft_proposer_latency_profile_file "
            f"must be valid JSON: {profile_file!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "SpeculativeDecodingConfig.decode_draft_proposer_latency_profile_file "
            f"must contain a JSON object, got={type(payload).__name__}"
        )

    allowed_top_level_keys = {"entries", "metadata"}
    unexpected_keys = sorted(set(payload.keys()) - allowed_top_level_keys)
    if unexpected_keys:
        raise ValueError(
            "Unsupported keys in decode draft proposer latency profile file: "
            f"{unexpected_keys}, supported={sorted(allowed_top_level_keys)}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(
            "SpeculativeDecodingConfig.decode_draft_proposer_latency_profile_file "
            "must contain a non-empty 'entries' list."
        )

    lookup: Dict[Tuple[str, str, int, int, int], float] = {}
    required_entry_keys = {
        "method",
        "model_name",
        "attn_tp_size",
        "num_speculative_tokens",
        "spec_verify_request_count",
        "latency_ms",
    }
    for idx, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                "decode draft proposer latency profile entries must be JSON objects, "
                f"got index={idx}, type={type(raw_entry).__name__}"
            )
        missing_keys = sorted(required_entry_keys - set(raw_entry.keys()))
        if missing_keys:
            raise ValueError(
                "decode draft proposer latency profile entry is missing keys: "
                f"index={idx}, missing={missing_keys}"
            )
        method = str(raw_entry["method"])
        if method not in supported_methods:
            raise ValueError(
                "decode draft proposer latency profile contains unsupported method "
                f"{method!r}; supported={sorted(supported_methods)}"
            )
        model_name = normalize_decode_draft_proposer_model_name(
            raw_entry["model_name"]
        )
        attn_tp_size = int(raw_entry["attn_tp_size"])
        num_speculative_tokens = int(raw_entry["num_speculative_tokens"])
        spec_verify_request_count = int(raw_entry["spec_verify_request_count"])
        latency_ms = float(raw_entry["latency_ms"])
        if not model_name:
            raise ValueError(
                "decode draft proposer latency profile model_name "
                f"must be non-empty, got index={idx}"
            )
        if attn_tp_size <= 0:
            raise ValueError(
                "decode draft proposer latency profile attn_tp_size "
                f"must be > 0, got index={idx}, value={attn_tp_size}"
            )
        if num_speculative_tokens <= 0:
            raise ValueError(
                "decode draft proposer latency profile num_speculative_tokens "
                f"must be > 0, got index={idx}, value={num_speculative_tokens}"
            )
        if spec_verify_request_count <= 0:
            raise ValueError(
                "decode draft proposer latency profile spec_verify_request_count "
                f"must be > 0, got index={idx}, value={spec_verify_request_count}"
            )
        if latency_ms < 0.0:
            raise ValueError(
                "decode draft proposer latency profile latency_ms "
                f"must be >= 0, got index={idx}, value={latency_ms!r}"
            )
        key = (
            method,
            model_name,
            attn_tp_size,
            num_speculative_tokens,
            spec_verify_request_count,
        )
        if key in lookup:
            raise ValueError(
                "decode draft proposer latency profile contains duplicate workload key: "
                f"{key}"
            )
        lookup[key] = latency_ms

    return lookup


def get_decode_draft_proposer_latency_ms(
    *,
    lookup: Optional[Dict[Tuple[str, str, int, int, int], float]],
    method: str,
    model_name: str,
    attn_tp_size: int,
    num_speculative_tokens: int,
    spec_verify_request_count: int,
) -> float:
    if not lookup:
        raise ValueError("decode draft proposer latency profile lookup is empty")
    key = (
        str(method),
        normalize_decode_draft_proposer_model_name(model_name),
        int(attn_tp_size),
        int(num_speculative_tokens),
        int(spec_verify_request_count),
    )
    if key not in lookup:
        raise ValueError(
            "No decode draft proposer latency profile for workload key "
            f"{key}"
        )
    return float(lookup[key])
