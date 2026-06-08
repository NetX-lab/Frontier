from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from frontier.config.config import SpeculativeDecodingConfig
from frontier.spec_decode.model_identity import resolve_canonical_model_key
from frontier.spec_decode.mtp_registry import (
    get_mtp_model_alias_map,
    get_registered_mtp_model_contract,
)


SUPPORTED_SPEC_METHODS = {
    "ngram",
    "medusa",
    "eagle",
    "eagle3",
    "deepseek_mtp",
    "ernie_mtp",
    "qwen3_moe_mtp",
    "qwen3_next_mtp",
}

MTP_METHOD_FAMILIES = {
    "deepseek_mtp": "draft_model_mtp",
    "ernie_mtp": "draft_model_mtp",
    "qwen3_moe_mtp": "target_embedded_mtp",
    "qwen3_next_mtp": "target_embedded_mtp",
}

LOOKAHEAD_SLOT_METHODS = {
    "eagle",
    "eagle3",
    "deepseek_mtp",
    "ernie_mtp",
    "qwen3_moe_mtp",
    "qwen3_next_mtp",
}

PREFIX_MATCHING_DISABLED_METHODS = {
    "qwen3_next_mtp",
}

TARGET_EMBEDDED_MTP_SAME_TP_LINEAR_OPS = frozenset(
    {
        "emb",
        "input_layernorm",
        "post_attention_layernorm",
    }
)


@dataclass(frozen=True)
class SpecDecodeIterationOutcome:
    planned_draft_tokens: int
    verify_tokens: int
    accepted_draft_tokens: int
    rejected_draft_tokens: int
    committed_tokens: int


def is_spec_decode_enabled(config: Optional[SpeculativeDecodingConfig]) -> bool:
    return bool(config is not None and config.enabled)


def method_uses_lookahead_slots(method: str) -> bool:
    if method not in SUPPORTED_SPEC_METHODS:
        raise ValueError(
            "Speculative method name must match vLLM names, "
            f"got={method!r}, supported={sorted(SUPPORTED_SPEC_METHODS)}"
        )
    return method in LOOKAHEAD_SLOT_METHODS


def method_requires_prefix_matching_disabled(method: str) -> bool:
    if method not in SUPPORTED_SPEC_METHODS:
        raise ValueError(
            "Speculative method name must match vLLM names, "
            f"got={method!r}, supported={sorted(SUPPORTED_SPEC_METHODS)}"
        )
    return method in PREFIX_MATCHING_DISABLED_METHODS


def get_mtp_method_family(method: str) -> str:
    if method not in SUPPORTED_SPEC_METHODS:
        raise ValueError(
            "Speculative method name must match vLLM names, "
            f"got={method!r}, supported={sorted(SUPPORTED_SPEC_METHODS)}"
        )
    if method not in MTP_METHOD_FAMILIES:
        raise ValueError(f"Method {method!r} is not an MTP method")
    return MTP_METHOD_FAMILIES[method]


def is_target_embedded_mtp_enabled(
    config: Optional[SpeculativeDecodingConfig],
) -> bool:
    if not is_spec_decode_enabled(config):
        return False
    method = str(getattr(config, "method", "")).strip()
    if method not in MTP_METHOD_FAMILIES:
        return False
    return MTP_METHOD_FAMILIES[method] == "target_embedded_mtp"


def _build_default_mtp_policy_contract(_mtp_family: str) -> dict[str, object]:
    return {
        "norm_policy": {
            "embedding": "rms_norm",
            "hidden_states": "rms_norm",
        },
        "decoder_layer_policy": "reuse_target_decoder_layer",
        "lm_head_policy": "tie_word_embeddings_based",
        "tp_policy": "reuse_target_attn_tp",
    }


def get_mtp_static_contract(
    method: str,
    *,
    model_name: Optional[str] = None,
    attn_tp_size: Optional[int] = None,
) -> dict[str, object]:
    mtp_family = get_mtp_method_family(method)
    canonical_model_key = ""
    if model_name is not None:
        canonical_model_key = resolve_canonical_model_key(
            str(model_name),
            explicit_alias_map=get_mtp_model_alias_map(),
        )
    policy_contract = _build_default_mtp_policy_contract(mtp_family)
    contract_source = "comparison_declared_method_family_contract"
    if canonical_model_key:
        registered_contract = get_registered_mtp_model_contract(
            canonical_model_key,
            mtp_family=mtp_family,
        )
        if registered_contract is None:
            raise ValueError(
                "Unknown MTP static contract model: "
                f"method={method!r}, model_name={canonical_model_key!r}. "
                "Current static contract helper is comparison-only and requires "
                "an explicit known-model registry entry."
            )
        policy_contract = {
            key: value
            for key, value in registered_contract.items()
            if key != "mtp_family"
        }
        contract_source = "comparison_registry_known_model"
    contract: dict[str, object] = {
        "method": str(method),
        "comparison_only_declaration": True,
        "contract_source": contract_source,
        "mtp_family": mtp_family,
        "spec_model_required": mtp_family == "draft_model_mtp",
        "requires_prefix_caching_disabled": method_requires_prefix_matching_disabled(
            method
        ),
        "uses_target_hidden_states": True,
        "uses_input_embeddings": True,
        **policy_contract,
    }
    if canonical_model_key:
        contract["model_name"] = canonical_model_key
    elif model_name is not None:
        normalized_model_name = str(model_name).strip().rstrip("/")
        if normalized_model_name:
            contract["model_name"] = normalized_model_name
    if attn_tp_size is not None:
        contract["attn_tp_size"] = int(attn_tp_size)
    return contract


def get_planned_draft_tokens(
    config: SpeculativeDecodingConfig,
    remaining_decode_tokens: int,
    iteration_index: Optional[int] = None,
    request_id: Optional[str] = None,
) -> int:
    if not config.enabled:
        return 0
    if remaining_decode_tokens <= 0:
        return 0
    per_request_planned_draft_trace = getattr(
        config, "_per_request_scheduled_draft_tokens_trace", None
    )
    if per_request_planned_draft_trace is not None:
        if request_id is None:
            raise ValueError(
                "request_id is required when acceptance_trace_file contains "
                "per-request scheduled draft tokens"
            )
        request_id_normalized = str(request_id)
        if request_id_normalized not in per_request_planned_draft_trace:
            raise ValueError(
                "per-request scheduled draft trace missing request_id="
                f"{request_id_normalized!r}"
            )
        if iteration_index is None:
            raise ValueError(
                "iteration_index is required when acceptance_trace_file contains "
                "per-request scheduled draft tokens"
            )
        request_trace = per_request_planned_draft_trace[request_id_normalized]
        idx = int(iteration_index)
        if idx < 0:
            raise ValueError(f"iteration_index must be >= 0, got={idx}")
        if idx >= len(request_trace):
            raise ValueError(
                "per-request scheduled draft trace exhausted: "
                f"request_id={request_id_normalized!r}, "
                f"iteration_index={idx}, trace_len={len(request_trace)}"
            )
        return int(request_trace[idx])
    planned_draft_trace = getattr(config, "_scheduled_draft_tokens_trace", None)
    if planned_draft_trace is not None:
        if iteration_index is None:
            raise ValueError(
                "iteration_index is required when acceptance_trace_file contains "
                "scheduled_draft_tokens_per_iteration"
            )
        idx = int(iteration_index)
        if idx < 0:
            raise ValueError(f"iteration_index must be >= 0, got={idx}")
        if idx >= len(planned_draft_trace):
            raise ValueError(
                "scheduled draft trace exhausted: "
                f"iteration_index={idx}, trace_len={len(planned_draft_trace)}"
            )
        return int(planned_draft_trace[idx])
    if remaining_decode_tokens <= 1:
        return 0
    return min(
        int(config.num_speculative_tokens),
        int(remaining_decode_tokens - 1),
    )


def _get_committed_tokens_from_deterministic_source(
    config: SpeculativeDecodingConfig,
    iteration_index: Optional[int],
    request_id: Optional[str],
) -> int:
    per_request_committed_trace = getattr(
        config, "_per_request_committed_tokens_trace", None
    )
    if per_request_committed_trace is not None:
        if request_id is None:
            raise ValueError(
                "request_id is required when acceptance_trace_file contains "
                "per-request committed tokens"
            )
        request_id_normalized = str(request_id)
        if request_id_normalized not in per_request_committed_trace:
            raise ValueError(
                "per-request acceptance trace missing request_id="
                f"{request_id_normalized!r}"
            )
        if iteration_index is None:
            raise ValueError(
                "iteration_index is required when acceptance_trace_file is configured"
            )
        request_trace = per_request_committed_trace[request_id_normalized]
        idx = int(iteration_index)
        if idx < 0:
            raise ValueError(f"iteration_index must be >= 0, got={idx}")
        if idx >= len(request_trace):
            raise ValueError(
                "per-request acceptance trace exhausted: "
                f"request_id={request_id_normalized!r}, "
                f"iteration_index={idx}, trace_len={len(request_trace)}"
            )
        return int(request_trace[idx])
    committed_trace = getattr(config, "_committed_tokens_trace", None)
    if committed_trace is not None:
        if iteration_index is None:
            raise ValueError(
                "iteration_index is required when acceptance_trace_file is configured"
            )
        idx = int(iteration_index)
        if idx < 0:
            raise ValueError(f"iteration_index must be >= 0, got={idx}")
        if idx >= len(committed_trace):
            raise ValueError(
                "acceptance trace exhausted: "
                f"iteration_index={idx}, trace_len={len(committed_trace)}"
            )
        return int(committed_trace[idx])
    return int(config.committed_tokens_per_iteration)


def compute_iteration_outcome(
    config: SpeculativeDecodingConfig,
    remaining_decode_tokens: int,
    planned_draft_tokens: Optional[int] = None,
    iteration_index: Optional[int] = None,
    request_id: Optional[str] = None,
) -> SpecDecodeIterationOutcome:
    if remaining_decode_tokens <= 0:
        return SpecDecodeIterationOutcome(
            planned_draft_tokens=0,
            verify_tokens=0,
            accepted_draft_tokens=0,
            rejected_draft_tokens=0,
            committed_tokens=0,
        )

    if not config.enabled:
        return SpecDecodeIterationOutcome(
            planned_draft_tokens=0,
            verify_tokens=1,
            accepted_draft_tokens=0,
            rejected_draft_tokens=0,
            committed_tokens=1,
        )

    if planned_draft_tokens is None:
        planned_drafts = get_planned_draft_tokens(
            config,
            remaining_decode_tokens,
            iteration_index=iteration_index,
            request_id=request_id,
        )
    else:
        planned_drafts = int(planned_draft_tokens)
        if planned_drafts < 0:
            raise ValueError(
                f"planned_draft_tokens must be >= 0, got={planned_drafts}"
            )

    committed_tokens_raw = _get_committed_tokens_from_deterministic_source(
        config, iteration_index, request_id
    )
    if committed_tokens_raw < 0:
        raise ValueError(
            "deterministic committed tokens must be >= 0, "
            f"got={committed_tokens_raw}"
        )
    max_committable_tokens = min(1 + planned_drafts, int(remaining_decode_tokens))
    committed_tokens = min(committed_tokens_raw, max_committable_tokens)

    # Phase 1 hard constraint: committed tokens must never exceed remaining decode.
    if committed_tokens > remaining_decode_tokens:
        committed_tokens = int(remaining_decode_tokens)

    accepted_drafts = max(0, committed_tokens - 1)
    rejected_drafts = max(0, planned_drafts - accepted_drafts)
    verify_tokens = 1 + planned_drafts

    if committed_tokens < 0:
        raise ValueError(
            "committed_tokens must be >= 0 when remaining_decode_tokens > 0, "
            f"got={committed_tokens}"
        )
    if committed_tokens > remaining_decode_tokens:
        raise ValueError(
            "committed_tokens exceeded remaining_decode_tokens after clamp, "
            f"committed={committed_tokens}, remaining={remaining_decode_tokens}"
        )

    return SpecDecodeIterationOutcome(
        planned_draft_tokens=planned_drafts,
        verify_tokens=verify_tokens,
        accepted_draft_tokens=accepted_drafts,
        rejected_draft_tokens=rejected_drafts,
        committed_tokens=committed_tokens,
    )
