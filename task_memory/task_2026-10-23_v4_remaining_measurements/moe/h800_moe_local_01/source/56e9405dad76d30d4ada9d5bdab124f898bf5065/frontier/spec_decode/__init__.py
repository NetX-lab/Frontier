from importlib import import_module

__all__ = [
    "SUPPORTED_SPEC_METHODS",
    "LOOKAHEAD_SLOT_METHODS",
    "PREFIX_MATCHING_DISABLED_METHODS",
    "SpecDecodeIterationOutcome",
    "is_spec_decode_enabled",
    "get_mtp_method_family",
    "is_target_embedded_mtp_enabled",
    "get_mtp_static_contract",
    "method_uses_lookahead_slots",
    "method_requires_prefix_matching_disabled",
    "get_planned_draft_tokens",
    "compute_iteration_outcome",
    "normalize_model_alias",
    "resolve_canonical_model_key",
    "MTPRuntimeContract",
    "build_mtp_runtime_contract",
    "DecodeDraftProposerLatencyProfileEntry",
    "normalize_decode_draft_proposer_model_name",
    "load_decode_draft_proposer_latency_profile",
    "get_decode_draft_proposer_latency_ms",
]

_RUNTIME_EXPORTS = {
    "SUPPORTED_SPEC_METHODS",
    "LOOKAHEAD_SLOT_METHODS",
    "PREFIX_MATCHING_DISABLED_METHODS",
    "SpecDecodeIterationOutcome",
    "is_spec_decode_enabled",
    "get_mtp_method_family",
    "is_target_embedded_mtp_enabled",
    "get_mtp_static_contract",
    "method_uses_lookahead_slots",
    "method_requires_prefix_matching_disabled",
    "get_planned_draft_tokens",
    "compute_iteration_outcome",
}

_PROFILE_EXPORTS = {
    "DecodeDraftProposerLatencyProfileEntry",
    "normalize_decode_draft_proposer_model_name",
    "load_decode_draft_proposer_latency_profile",
    "get_decode_draft_proposer_latency_ms",
}

_IDENTITY_EXPORTS = {
    "normalize_model_alias",
    "resolve_canonical_model_key",
}

_MTP_RUNTIME_EXPORTS = {
    "MTPRuntimeContract",
    "build_mtp_runtime_contract",
}


def __getattr__(name: str):
    if name in _RUNTIME_EXPORTS:
        module = import_module("frontier.spec_decode.runtime")
        return getattr(module, name)
    if name in _PROFILE_EXPORTS:
        module = import_module("frontier.spec_decode.proposer_profile")
        return getattr(module, name)
    if name in _IDENTITY_EXPORTS:
        module = import_module("frontier.spec_decode.model_identity")
        return getattr(module, name)
    if name in _MTP_RUNTIME_EXPORTS:
        module = import_module("frontier.spec_decode.mtp_runtime")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
