from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config.model_config import BaseModelConfig, ModelArch
from frontier.model_architectures import (
    ExpertParallelCollective,
    LinearAttentionImplementation,
    ModelArchitectureProfile,
    ModelArchitectureRegistry,
    ResidualAddPolicy,
    get_model_architecture_profile,
)
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.types import ActivationType, ClusterType, NormType


def _profiling_config(**overrides):
    values = dict(
        model_type="unit_custom_model",
        model_arch="generic",
        model_architecture_profile=None,
        embedding_dim=128,
        mlp_hidden_dim=256,
        num_q_heads=8,
        num_kv_heads=4,
        no_tensor_parallel=False,
        is_moe=True,
        post_attn_norm=True,
        share_expert_dim=64,
    )
    values.update(overrides)
    cfg = SimpleNamespace(**values)
    cfg.supports_share_expert = lambda: bool(get_model_architecture_profile(cfg).supports_share_expert(cfg))
    return cfg


def _runtime_model_config(**overrides) -> BaseModelConfig:
    values = dict(
        num_layers=2,
        num_q_heads=8,
        num_kv_heads=4,
        embedding_dim=128,
        mlp_hidden_dim=256,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        model_type="unit_custom_model",
        model_arch=ModelArch.GENERIC,
        is_moe=True,
        share_expert_dim=64,
    )
    values.update(overrides)
    return BaseModelConfig(**values)


def _real_profiling_model_config(**overrides) -> ProfilingModelConfig:
    values = dict(
        name="unit-model",
        num_layers=2,
        num_q_heads=8,
        num_kv_heads=4,
        embedding_dim=128,
        mlp_hidden_dim=256,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation="silu",
        norm="rms_norm",
        post_attn_norm=True,
        vocab_size=32000,
        model_type="unit_custom_model",
        model_arch=ModelArch.GENERIC,
        is_moe=True,
        share_expert_dim=64,
    )
    values.update(overrides)
    return ProfilingModelConfig(**values)


def test_explicit_profile_id_reuses_step3_contract_for_new_model_name() -> None:
    cfg = _profiling_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
    )

    profile = get_model_architecture_profile(cfg)

    assert profile.profile_id == "step3_text"
    assert profile.linear_attention.sharded_impl is LinearAttentionImplementation.STEP3_TEXT
    assert profile.linear_attention.replicated_ops == (
        "attn_pre_proj_qkv",
        "attn_pre_proj_q_norm",
    )
    assert profile.expert_parallel_collective is ExpertParallelCollective.ALLTOALL
    assert profile.uses_expert_parallel_alltoall(ClusterType.MONOLITHIC, expected_ep_size=2)


def test_explicit_step3_profile_drives_profiling_plan_without_model_type_branch() -> None:
    cfg = _profiling_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
    )

    plan = build_profiling_plan(
        cfg,
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        is_moe=True,
    )

    assert "attn_pre_proj_qkv" in plan["replicated_ops"]
    assert "attn_pre_proj_q_norm" in plan["replicated_ops"]
    assert "attn_pre_proj_wq" in plan["enabled_ops"]
    assert "share_expert_up_proj" in plan["enabled_ops"]


def test_explicit_step2_profile_drives_profiling_plan_without_model_type_branch() -> None:
    cfg = _profiling_config(
        model_type="unit_new_step2_like",
        model_architecture_profile="step2_mini",
    )

    plan = build_profiling_plan(
        cfg,
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        is_moe=True,
    )

    assert "attn_inter_norm" in plan["enabled_ops"]
    assert "attn_wq_proj" in plan["enabled_ops"]
    assert "share_expert_up_proj" in plan["enabled_ops"]


def test_local_registry_can_plugin_custom_profile_without_global_model_branch() -> None:
    registry = ModelArchitectureRegistry()
    registry.register(ModelArchitectureProfile.generic())
    registry.register(
        ModelArchitectureProfile.step3_text(
            profile_id="unit_step3_plugin",
            match=lambda cfg: getattr(cfg, "model_type", None) == "unit_plugin_model",
        )
    )

    profile = registry.resolve(_profiling_config(model_type="unit_plugin_model"))

    assert profile.profile_id == "unit_step3_plugin"
    assert profile.linear_attention.sharded_ops == (
        "attn_pre_proj",
        "attn_rope",
        "attn_post_proj",
        "attn_pre_proj_wq",
    )


def test_runtime_config_explicit_profile_drives_legacy_compat_methods() -> None:
    cfg = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
    )

    assert cfg.get_model_architecture_profile().profile_id == "step3_text"
    assert cfg.is_step3_text()
    assert not cfg.is_step2_mini()
    assert cfg.supports_share_expert()


def test_profiling_config_explicit_profile_drives_legacy_compat_methods() -> None:
    cfg = _real_profiling_model_config(
        model_type="unit_new_step2_like",
        model_arch=ModelArch.GENERIC,
        model_architecture_profile="step2_mini",
    )

    assert cfg.get_model_architecture_profile().profile_id == "step2_mini"
    assert cfg.is_step2_mini
    assert not cfg.is_step3_text()
    assert cfg.supports_share_expert()


def test_ep_collective_resolver_uses_profile_not_step3_model_type() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    step3_alias_cfg = _profiling_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
    )
    generic_named_step3_cfg = _profiling_config(
        model_type="step3_text",
        model_architecture_profile="generic",
    )

    assert (
        resolve_ep_collective_kind(
            step3_alias_cfg,
            ClusterType.MONOLITHIC,
            expected_ep_size=2,
        )
        is ExpertParallelCollective.ALLTOALL
    )
    assert (
        resolve_ep_collective_kind(
            generic_named_step3_cfg,
            ClusterType.MONOLITHIC,
            expected_ep_size=2,
        )
        is ExpertParallelCollective.ALLGATHER
    )


def test_ep_collective_resolver_fails_fast_without_model_config() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    with pytest.raises(
        ValueError,
        match="EP collective resolution requires replica_config.model_config",
    ):
        resolve_ep_collective_kind(
            None,
            ClusterType.MONOLITHIC,
            expected_ep_size=2,
        )


def test_phase2_consumers_do_not_directly_branch_on_step2_step3_identity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    consumer_paths = (
        repo_root / "frontier/profiling/linear_op/linear_op_impl.py",
        repo_root / "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py",
        repo_root / "frontier/metrics/metrics_store.py",
    )
    forbidden_snippets = (
        'config.model_type == "step3_text"',
        'config.model_type == "step2_mini"',
        'model_config.model_type == "step3_text"',
        "model_config.is_step3_text()",
    )

    violations = []
    for path in consumer_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append(f"{path.relative_to(repo_root)}: {snippet}")

    assert violations == []





def test_step3_profile_declares_residual_add_policy_capability() -> None:
    assert ModelArchitectureProfile.generic().residual_add_policy is ResidualAddPolicy.STANDARD
    assert (
        ModelArchitectureProfile.step3_text().residual_add_policy
        is ResidualAddPolicy.FFN_RESIDUAL_ONLY
    )


def test_predictor_metadata_validates_architecture_profile_id() -> None:
    import pandas as pd

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _ConcretePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = object.__new__(_ConcretePredictor)
    predictor._model_config = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
        is_moe=True,
        share_expert_dim=64,
    )
    df = pd.DataFrame(
        {
            "profiling_precision": ["fp16"],
            "model_arch": ["generic"],
            "model_architecture_profile": ["step3_text"],
            "quant_signature": [predictor._model_config.get_quant_signature()],
            "measurement_type": ["cuda_event"],
        }
    )

    metadata = predictor._get_profiling_metadata(df, "unit.csv")

    assert metadata.model_arch == "generic"
    assert metadata.model_architecture_profile == "step3_text"


def test_predictor_metadata_rejects_profile_mismatch() -> None:
    import pandas as pd
    import pytest

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _ConcretePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = object.__new__(_ConcretePredictor)
    predictor._model_config = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
        is_moe=True,
        share_expert_dim=64,
    )
    df = pd.DataFrame(
        {
            "profiling_precision": ["fp16"],
            "model_arch": ["generic"],
            "model_architecture_profile": ["generic"],
            "quant_signature": [predictor._model_config.get_quant_signature()],
            "measurement_type": ["cuda_event"],
        }
    )

    with pytest.raises(ValueError, match="model_architecture_profile mismatch"):
        predictor._get_profiling_metadata(df, "unit.csv")


def test_predictor_rejects_invalid_architecture_profile_contract() -> None:
    from typing import cast

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _ConcretePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = object.__new__(_ConcretePredictor)
    # This test intentionally injects a malformed structural config.
    predictor._model_config = cast(
        BaseModelConfig,
        SimpleNamespace(get_model_architecture_profile=lambda: object()),
    )

    try:
        predictor._get_model_architecture_profile()
    except TypeError as exc:
        assert "must return ModelArchitectureProfile" in str(exc)
    else:
        raise AssertionError("Expected invalid architecture profile contract to fail")


def test_moe_predictor_rejects_invalid_architecture_profile_contract() -> None:
    from typing import cast

    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
        SklearnMoEExecutionTimePredictor,
    )

    class _ConcreteMoEPredictor(SklearnMoEExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = object.__new__(_ConcreteMoEPredictor)
    # This test intentionally injects a malformed structural config.
    predictor._model_config = cast(
        BaseModelConfig,
        SimpleNamespace(get_model_architecture_profile=lambda: object()),
    )

    try:
        predictor._apply_share_expert_tp_allreduce_overlap(1.0)
    except TypeError as exc:
        assert "must return ModelArchitectureProfile" in str(exc)
    else:
        raise AssertionError("Expected invalid architecture profile contract to fail")


def test_step3_profile_requires_moe_runtime_config() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires? is_moe=True"):
        _runtime_model_config(
            model_type="unit_invalid_step3_like",
            model_architecture_profile="step3_text",
            is_moe=False,
            share_expert_dim=64,
        )


def test_step3_profile_requires_moe_profiling_config() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires? is_moe=True"):
        _real_profiling_model_config(
            model_type="unit_invalid_step3_like",
            model_architecture_profile="step3_text",
            is_moe=False,
            share_expert_dim=64,
        )


def test_mtp_structural_adapter_uses_explicit_architecture_profile() -> None:
    from frontier.spec_decode.mtp_runtime import StructuralModelConfigAdapter

    profiling_config = _real_profiling_model_config(
        model_type="unit_new_step3_like",
        model_arch=ModelArch.GENERIC,
        model_architecture_profile="step3_text",
        is_moe=True,
        share_expert_dim=64,
    )
    adapter = StructuralModelConfigAdapter(profiling_config)

    assert adapter.get_model_architecture_profile().profile_id == "step3_text"
    assert adapter.is_step3_text()
    assert not adapter.is_step2_mini()
    assert adapter.supports_share_expert()


def test_mtp_json_fallback_preserves_explicit_architecture_profile(
    monkeypatch,
    tmp_path,
) -> None:
    import json

    from frontier.spec_decode.mtp_runtime import (
        _load_structural_model_config_from_json,
    )

    config_dir = tmp_path / "data" / "config" / "models"
    config_dir.mkdir(parents=True)
    (config_dir / "unit-json-step3-like.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "hidden_size": 128,
                "intermediate_size": 256,
                "max_position_embeddings": 4096,
                "vocab_size": 32000,
                "hidden_act": "silu",
                "model_type": "unit_new_step3_like",
                "model_arch": "generic",
                "model_architecture_profile": "step3_text",
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "share_expert_dim": 64,
                "share_q_dim": 16,
                "head_dim": 16,
                "torch_dtype": "float16",
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    adapter = _load_structural_model_config_from_json("unit-json-step3-like")

    assert adapter.get_model_architecture_profile().profile_id == "step3_text"
    assert adapter.is_step3_text()
    assert adapter.supports_share_expert()


def test_mtp_structural_loader_does_not_mask_internal_profiling_errors(
    monkeypatch,
) -> None:
    import pytest

    from frontier.spec_decode import mtp_runtime

    def raise_internal_error(model_name: str):  # noqa: ARG001
        raise RuntimeError("profiling registry exploded")

    monkeypatch.setattr(
        mtp_runtime.ProfilingModelConfig,
        "from_model_name",
        staticmethod(raise_internal_error),
    )

    with pytest.raises(RuntimeError, match="profiling registry exploded"):
        mtp_runtime.load_mtp_structural_model_config("unit-json-step3-like")


def test_mtp_structural_loader_preserves_json_fallback_for_value_error(
    monkeypatch,
    tmp_path,
) -> None:
    import json

    from frontier.spec_decode import mtp_runtime

    def raise_model_lookup_error(model_name: str):  # noqa: ARG001
        raise ValueError("profiling model config unavailable")

    config_dir = tmp_path / "data" / "config" / "models"
    config_dir.mkdir(parents=True)
    (config_dir / "unit-json-step3-like.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "hidden_size": 128,
                "intermediate_size": 256,
                "max_position_embeddings": 4096,
                "vocab_size": 32000,
                "hidden_act": "silu",
                "model_type": "unit_new_step3_like",
                "model_arch": "generic",
                "model_architecture_profile": "step3_text",
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "share_expert_dim": 64,
                "share_q_dim": 16,
                "head_dim": 16,
                "torch_dtype": "float16",
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        mtp_runtime.ProfilingModelConfig,
        "from_model_name",
        staticmethod(raise_model_lookup_error),
    )

    adapter = mtp_runtime.load_mtp_structural_model_config("unit-json-step3-like")

    assert adapter.get_model_architecture_profile().profile_id == "step3_text"
    assert adapter.supports_share_expert()


def test_param_counter_share_expert_uses_profile_for_new_model_name() -> None:
    from types import SimpleNamespace
    from typing import cast

    from frontier.config import ReplicaConfig
    from frontier.utils.param_counter import ParamCounter

    model_config = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
        embedding_dim=128,
        share_expert_dim=64,
        use_gated_mlp=True,
        is_moe=True,
    )
    replica_config = cast(
        ReplicaConfig,
        SimpleNamespace(
            model_config=model_config,
            attn_tensor_parallel_size=1,
            moe_tensor_parallel_size=2,
            moe_expert_parallel_size=1,
            num_pipeline_stages=1,
        ),
    )

    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    assert counter._get_share_expert_params_per_layer(tensor_parallel_size=2) == 12288


def test_phase2_predictor_consumers_do_not_use_step2_step3_identity_wrappers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    consumer_paths = (
        repo_root / "frontier/execution_time_predictor/sklearn_execution_time_predictor.py",
        repo_root / "frontier/execution_time_predictor/shared_prediction_model_manager.py",
        repo_root / "frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py",
        repo_root / "frontier/utils/param_counter.py",
        repo_root / "frontier/profiling/utils/confirmation.py",
    )
    forbidden_snippets = (
        "is_step2_mini()",
        "is_step3_text()",
        "model_type == \"step3_text\"",
        "model_type not in {\"step2_mini\", \"step3_text\"}",
        "model_arch == \"step2_mini\"",
        "step3_text_compatible",
        "_log_step3_attention_shape",
        "[STEP3_SHAPE]",
    )

    violations = []
    for path in consumer_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append(f"{path.relative_to(repo_root)}: {snippet}")

    assert violations == []
