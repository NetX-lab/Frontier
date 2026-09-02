"""Regression coverage for profile-owned typed dimension sources."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.model_architectures import (
    LayerKind,
    ModelArchitectureProfile,
    _moe_share_expert_requirements,
)
from frontier.profiling.linear_op.linear_op_impl import _supports_share_expert
from frontier.profiling.non_kv_cache_overhead import runtime_estimator
from frontier.types import ClusterType
from frontier.types import MeasurementType
from frontier.utils.param_counter import ParamCounter


def _alias_only_config(*, shared_width: int | None = 64) -> SimpleNamespace:
    """Build a config whose shared width is exposed only under the HF alias."""

    config = SimpleNamespace(
        is_moe=True,
        num_layers=4,
        num_q_heads=8,
        num_kv_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layers_enum="1,2",
        model_type="unit_alias_only",
        model_arch="generic",
        model_architecture_profile="generic",
        embedding_dim=128,
        mlp_hidden_dim=128,
        dense_mlp_hidden_dim=256,
        routed_mlp_hidden_dim=128,
        share_expert_dim=None,
        shared_expert_intermediate_size=shared_width,
    )
    config.get_model_architecture_profile = ModelArchitectureProfile.generic
    return config


def _cache_replica(config: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=8,
        attn_dp=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        num_pipeline_stages=1,
    )


def test_shared_alias_is_resolved_by_every_typed_consumer(monkeypatch) -> None:
    """One alias-only width must resolve identically across all consumers."""

    config = _alias_only_config()
    profile = ModelArchitectureProfile.generic()

    assert profile.supports_share_expert(config) is True
    assert _supports_share_expert(config) is True
    assert runtime_estimator._supports_share_expert_weights(config) is True

    resolved = profile.resolve_layer_contract(
        config,
        layer_kind=LayerKind.SHARED,
        tensor_parallel_size=1,
        expert_parallel_size=1,
    )
    assert resolved.effective_ffn_width == 64

    counter = object.__new__(ParamCounter)
    counter._model_config = config
    counter._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    counter_contract = counter._resolve_profile_layer_contract(
        LayerKind.SHARED,
        tensor_parallel_size=1,
    )
    assert counter_contract is not None
    assert counter_contract.effective_ffn_width == 64

    context = OpTraceContext(
        cluster_type=ClusterType.MONOLITHIC,
        model_config=config,
        replica_config=SimpleNamespace(
            attn_tensor_parallel_size=1,
            attn_dp=1,
            moe_tensor_parallel_size=1,
            moe_expert_parallel_size=1,
            num_pipeline_stages=1,
            router_topk=2,
        ),
        total_tokens=1,
        effective_tokens_compute=1,
        effective_tokens_transfer=1,
        effective_tokens_rounded=1,
        tokens_are_post_routing=False,
    )
    assert context.share_expert_dim == 64

    for requirement in _moe_share_expert_requirements():
        requirement.validate(profile, config)


def test_canonical_shared_width_wins_over_compatibility_alias() -> None:
    """The canonical field remains authoritative when both fields exist."""

    config = _alias_only_config(shared_width=64)
    config.share_expert_dim = 96
    resolved = ModelArchitectureProfile.generic().resolve_layer_contract(
        config,
        layer_kind=LayerKind.SHARED,
        tensor_parallel_size=1,
        expert_parallel_size=1,
    )
    assert resolved.effective_ffn_width == 96


def test_zero_canonical_shared_width_continues_to_declared_alias() -> None:
    """A zero optional sentinel does not hide a valid compatibility alias."""

    config = _alias_only_config(shared_width=5120)
    config.share_expert_dim = 0

    resolved = ModelArchitectureProfile.generic().resolve_layer_contract(
        config,
        layer_kind=LayerKind.SHARED,
        tensor_parallel_size=1,
        expert_parallel_size=1,
    )

    assert resolved.effective_ffn_width == 5120


def test_dense_resolution_ignores_synthesized_legacy_width() -> None:
    """Dynamic adapters cannot satisfy a missing profile-owned dense field."""

    class DynamicConfig:
        is_moe = False
        num_layers = 1
        dense_mlp_hidden_dim = None

        def __getattr__(self, name: str) -> object:
            if name == "mlp_hidden_dim":
                return 999
            raise AttributeError(name)

    with pytest.raises(ValueError, match="dense layer width"):
        ModelArchitectureProfile.generic().resolve_layer_contract(
            DynamicConfig(),
            layer_kind=LayerKind.DENSE,
            tensor_parallel_size=1,
        )


def test_missing_shared_width_fails_fast_after_alias_resolution() -> None:
    """A shared contract without either declared width must stay explicit."""

    config = _alias_only_config(shared_width=None)
    with pytest.raises(ValueError, match="shared|share_expert"):
        ModelArchitectureProfile.generic().resolve_layer_contract(
            config,
            layer_kind=LayerKind.SHARED,
            tensor_parallel_size=1,
            expert_parallel_size=1,
        )


def test_runtime_cache_key_separates_typed_width_contracts(monkeypatch) -> None:
    """Equal legacy fields cannot make different typed contracts share a cache."""

    config_a = _alias_only_config()
    config_a.get_name = lambda: "same-model"
    config_a.get_head_dim = lambda: 16
    config_a.dense_mlp_hidden_dim = 18432
    config_a.routed_mlp_hidden_dim = 5120
    config_a.mlp_hidden_dim = 5120
    config_a.share_expert_dim = 5120
    config_a.shared_expert_intermediate_size = None

    config_b = _alias_only_config()
    config_b.get_name = lambda: "same-model"
    config_b.get_head_dim = lambda: 16
    config_b.dense_mlp_hidden_dim = 12288
    config_b.routed_mlp_hidden_dim = 5120
    config_b.mlp_hidden_dim = 5120
    config_b.share_expert_dim = None
    config_b.shared_expert_intermediate_size = 4096

    monkeypatch.setattr(
        runtime_estimator,
        "_get_current_cuda_total_memory_cache_key",
        lambda: "unit-cuda",
    )
    key_a = runtime_estimator._build_cache_key(
        replica_config=_cache_replica(config_a),
        cluster_type=ClusterType.MONOLITHIC,
        max_num_batched_tokens=8,
        weights_memory_bytes=10,
        weights_memory_source="param_counter",
    )
    key_b = runtime_estimator._build_cache_key(
        replica_config=_cache_replica(config_b),
        cluster_type=ClusterType.MONOLITHIC,
        max_num_batched_tokens=8,
        weights_memory_bytes=10,
        weights_memory_source="param_counter",
    )

    assert key_a != key_b


class _HashProbePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        raise AssertionError("not used")

    def _get_estimator(self):
        raise AssertionError("not used")


def _standalone_hash_probe(config: object) -> _HashProbePredictor:
    """Build the smallest predictor shell that exercises production cache hashing."""

    predictor = _HashProbePredictor.__new__(_HashProbePredictor)
    predictor._config = SimpleNamespace(
        get_type=lambda: "unit",
        k_fold_cv_splits=2,
        prediction_max_prefill_chunk_size=8,
        prediction_max_batch_size=4,
    )
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    predictor._model_config = config
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._block_size = 16
    predictor._max_tokens = 128
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._model_manager = None
    for attribute in (
        "_compute_input_file",
        "_attention_input_file",
        "_compute_input_file_eager",
        "_attention_input_file_eager",
        "_compute_input_file_kernel_only",
        "_attention_input_file_kernel_only",
        "_all_reduce_input_file",
        "_send_recv_input_file",
        "_cpu_overhead_input_file",
        "_pp_stage_boundary_input_file",
        "_pp_receiver_head_input_file",
        "_pp_producer_send_path_input_file",
        "_pp_prefill_consumer_active_input_file",
    ):
        setattr(predictor, attribute, None)
    return predictor


def test_standalone_predictor_cache_identity_includes_typed_contract() -> None:
    """Different typed widths must not share standalone model or prediction caches."""

    config_a = BaseModelConfig.create_from_name("step3-moe-noquant")
    config_b = BaseModelConfig.create_from_name("step3-moe-noquant")
    config_b.dense_mlp_hidden_dim = 12288

    predictor_a = _standalone_hash_probe(config_a)
    predictor_b = _standalone_hash_probe(config_b)

    assert predictor_a._get_model_hash("mlp_up_proj") != predictor_b._get_model_hash(
        "mlp_up_proj"
    )
    model = SimpleNamespace()
    assert predictor_a._get_prediction_cache_hash(
        "mlp_up_proj", model
    ) != predictor_b._get_prediction_cache_hash("mlp_up_proj", model)


def test_standalone_predictor_cache_identity_includes_moe_layer_map() -> None:
    """Changing only MoE layer placement must invalidate standalone caches."""

    config_official = BaseModelConfig.create_from_name("step3-moe-noquant")
    config_alternate = BaseModelConfig.create_from_name("step3-moe-noquant")
    config_alternate.moe_layers_enum = ",".join(
        str(layer_id) for layer_id in range(0, 56)
    )

    predictor_official = _standalone_hash_probe(config_official)
    predictor_alternate = _standalone_hash_probe(config_alternate)

    assert predictor_official._get_model_hash_identity(
        "mlp_up_proj"
    ) != predictor_alternate._get_model_hash_identity("mlp_up_proj")


def test_profile_identity_includes_model_depth_when_moe_map_is_unchanged() -> None:
    """Equal typed widths and layer IDs must not hide a different model depth."""

    config_a = BaseModelConfig.create_from_name("step3-moe-noquant")
    config_b = BaseModelConfig.create_from_name("step3-moe-noquant")
    config_b.num_layers = config_a.num_layers + 1

    profile = config_a.get_model_architecture_profile()
    assert profile.get_layer_contract_identity(config_a) != (
        profile.get_layer_contract_identity(config_b)
    )


def test_op_trace_uses_profile_owned_width_and_tp_domains_for_mixed_step3() -> None:
    """Trace shapes must follow each typed Step3 layer contract."""

    config = SimpleNamespace(
        is_moe=True,
        num_layers=4,
        num_q_heads=8,
        num_kv_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layers_enum="1,2",
        model_type="step3_text",
        model_arch="generic",
        model_architecture_profile="step3_text",
        embedding_dim=16,
        mlp_hidden_dim=32,
        dense_mlp_hidden_dim=24,
        routed_mlp_hidden_dim=32,
        share_expert_dim=12,
        shared_expert_intermediate_size=None,
        get_model_architecture_profile=ModelArchitectureProfile.step3_text,
        get_head_dim=lambda: 2,
    )
    replica = SimpleNamespace(
        attn_tensor_parallel_size=4,
        attn_dp=1,
        moe_tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        num_pipeline_stages=1,
        router_topk=2,
    )
    context = OpTraceContext(
        cluster_type=ClusterType.MONOLITHIC,
        model_config=config,
        replica_config=replica,
        total_tokens=8,
        effective_tokens_compute=8,
        effective_tokens_transfer=8,
        effective_tokens_rounded=8,
        tokens_are_post_routing=False,
    )

    dense = compute_op_trace_meta("mlp_up_proj", "COMPUTE", context)
    routed = compute_op_trace_meta("moe_grouped_gemm", "COMPUTE", context)
    shared = compute_op_trace_meta("share_expert_up_proj", "COMPUTE", context)

    assert dense["tensor_shape"]["output"] == [8, 6]
    assert routed["tensor_shape"]["output"] == [16, 16]
    assert shared["tensor_shape"]["output"] == [8, 3]


def test_op_trace_shared_tp_uses_role_domain_for_decode_ffn() -> None:
    """Decode-FFN maps the Step3 attention TP contract to its FFN role TP."""

    config = SimpleNamespace(
        is_moe=True,
        num_layers=4,
        num_q_heads=8,
        num_kv_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layers_enum="0,1,2,3",
        model_type="step3_text",
        model_architecture_profile="step3_text",
        embedding_dim=16,
        mlp_hidden_dim=32,
        dense_mlp_hidden_dim=24,
        routed_mlp_hidden_dim=32,
        share_expert_dim=12,
        get_model_architecture_profile=ModelArchitectureProfile.step3_text,
        get_head_dim=lambda: 2,
    )
    replica = SimpleNamespace(
        attn_tensor_parallel_size=4,
        attn_dp=1,
        moe_tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        num_pipeline_stages=1,
        router_topk=2,
    )
    context = OpTraceContext(
        cluster_type=ClusterType.DECODE_FFN,
        model_config=config,
        replica_config=replica,
        total_tokens=8,
        effective_tokens_compute=8,
        effective_tokens_transfer=8,
        effective_tokens_rounded=8,
        tokens_are_post_routing=False,
    )

    shared = compute_op_trace_meta("share_expert_up_proj", "COMPUTE", context)

    assert context.share_expert_tp == 2
    assert shared["tensor_shape"]["output"] == [8, 6]
