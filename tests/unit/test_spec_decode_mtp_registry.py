from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_target_embedded_mtp_methods_are_registry_source_for_runtime_helpers() -> None:
    import frontier.spec_decode.mtp_registry as registry
    from frontier.spec_decode.runtime import (
        SUPPORTED_SPEC_METHODS,
        get_mtp_method_family,
        method_requires_prefix_matching_disabled,
        method_uses_lookahead_slots,
    )

    target_methods = set(registry.get_target_embedded_mtp_methods())

    assert target_methods == {"qwen3_moe_mtp", "qwen3_next_mtp"}
    assert target_methods.issubset(SUPPORTED_SPEC_METHODS)
    for method in target_methods:
        assert registry.is_target_embedded_mtp_method(method)
        assert get_mtp_method_family(method) == "target_embedded_mtp"
        assert method_uses_lookahead_slots(method) is True
    assert method_requires_prefix_matching_disabled("qwen3_next_mtp") is True
    assert method_requires_prefix_matching_disabled("qwen3_moe_mtp") is False


def test_target_embedded_runtime_contract_reads_registry_policy() -> None:
    import frontier.spec_decode.mtp_registry as registry
    from frontier.spec_decode.mtp_runtime import build_mtp_runtime_contract

    policy = registry.get_target_embedded_mtp_runtime_policy("qwen3_next_mtp")

    contract = build_mtp_runtime_contract(
        method="qwen3_next_mtp",
        target_model_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
        spec_model_name="",
        attn_tp_size=4,
        mtp_n_predict=1,
        mtp_num_layers=1,
    )

    assert contract.mtp_family == "target_embedded_mtp"
    assert contract.proposer_model_name == contract.target_model_name
    assert contract.fusion_op_name == policy["fusion_op_name"]
    assert contract.fusion_is_tp_sharded is policy["fusion_is_tp_sharded"]
    assert contract.fusion_requires_allgather is policy["fusion_requires_allgather"]
    assert contract.norm_op_name == policy["norm_op_name"]
    assert contract.lm_head_op_name == policy["lm_head_op_name"]


def test_draft_model_mtp_still_requires_independent_spec_model_contract() -> None:
    from frontier.spec_decode.mtp_runtime import build_mtp_runtime_contract

    with pytest.raises(ValueError, match="draft-model MTP requires non-empty spec_model_name"):
        build_mtp_runtime_contract(
            method="deepseek_mtp",
            target_model_name="target",
            spec_model_name="",
            attn_tp_size=4,
            mtp_n_predict=1,
            mtp_num_layers=1,
        )


def test_target_embedded_same_tp_ops_are_registry_backed() -> None:
    import frontier.spec_decode.mtp_registry as registry

    assert registry.get_target_embedded_mtp_same_tp_linear_ops() == (
        "emb",
        "input_layernorm",
        "post_attention_layernorm",
    )


def test_target_embedded_method_literals_are_not_duplicated_in_runtime_contract_table() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "frontier/spec_decode/mtp_runtime.py").read_text(
        encoding="utf-8"
    )

    assert '"qwen3_next_mtp"' not in source
    assert '"qwen3_moe_mtp"' not in source


def _param_counter_for_spec_method(
    *,
    method: str,
    spec_model_name: str = "",
):
    from frontier.types import ClusterType
    from frontier.utils.param_counter import ParamCounter

    model_config = SimpleNamespace(
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        embedding_dim=16,
        mlp_hidden_dim=32,
        use_gated_mlp=True,
        is_moe=False,
        vocab_size=64,
        tie_word_embeddings=True,
        get_head_dim=lambda: 4,
    )
    replica_config = SimpleNamespace(
        model_name="unit-target-model",
        model_config=model_config,
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        num_pipeline_stages=1,
        speculative_decoding_config=SimpleNamespace(
            enabled=True,
            method=method,
            spec_model_name=spec_model_name,
            mtp_n_predict=1,
            mtp_num_layers=1,
        ),
    )
    return ParamCounter(replica_config, ClusterType.MONOLITHIC)


@pytest.mark.parametrize("method", ["deepseek_mtp", "ernie_mtp"])
def test_param_counter_draft_model_mtp_missing_spec_model_fails_fast(
    method: str,
) -> None:
    counter = _param_counter_for_spec_method(method=method, spec_model_name="")

    with pytest.raises(ValueError, match="draft-model MTP requires non-empty spec_model_name"):
        counter.get_num_mtp_parameters_per_device()


def test_param_counter_non_mtp_spec_method_has_no_mtp_parameters() -> None:
    counter = _param_counter_for_spec_method(method="eagle", spec_model_name="")

    assert counter.get_num_mtp_parameters_per_device() == 0


def test_param_counter_target_embedded_mtp_allows_empty_spec_model_name() -> None:
    counter = _param_counter_for_spec_method(method="qwen3_next_mtp", spec_model_name="")

    assert counter.get_num_mtp_parameters_per_device() > 0


def test_speculative_decoding_config_accepts_runtime_mtp_registry_methods() -> None:
    from frontier.config.config import SpeculativeDecodingConfig
    from frontier.spec_decode.runtime import MTP_METHOD_FAMILIES

    for method in MTP_METHOD_FAMILIES:
        SpeculativeDecodingConfig(
            enabled=True,
            method=method,
            mtp_n_predict=1,
            mtp_num_layers=1,
        )


def test_static_mtp_contract_helper_is_not_used_by_production_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    allowed_paths = {
        repo_root / "frontier/spec_decode/runtime.py",
        repo_root / "frontier/spec_decode/__init__.py",
    }

    production_hits = []
    for path in (repo_root / "frontier").rglob("*.py"):
        if path in allowed_paths:
            continue
        source = path.read_text(encoding="utf-8")
        if "get_mtp_static_contract" in source:
            production_hits.append(path.relative_to(repo_root).as_posix())

    assert production_hits == []
