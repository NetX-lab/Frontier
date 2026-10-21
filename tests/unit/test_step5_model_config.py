"""Step text representation contracts; these checks do not qualify execution."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.families import get_attention_family
from frontier.attention.gdn.config import LayerAttentionSpec
from frontier.attention.model_binding import bind_attention_family, bind_layer_attention
from frontier.attention.step5 import (
    resolve_step5_attention_geometry,
    resolve_step5_model_precision,
)
from frontier.config import global_vars
from frontier.config.model_config import (
    BaseModelConfig,
    QuantizationConfig,
    _normalize_hf_text_config,
)
from frontier.config.precision_type import PrecisionType
from frontier.config.utils import dataclass_to_dict
from frontier.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)
from frontier.model_architectures import LayerKind, get_model_architecture_profile
from frontier.operators.binding import build_operator_manifest
from frontier.operators.spec import TensorParallelMode
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.types import ActivationType, NormType


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "stepfun-ai/Step-5-Preview"
MODEL_PATH = REPO_ROOT / "data/config/models/stepfun-ai__Step-5-Preview.json"
MODEL_SHA256 = "e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63"
# StructuralRequirement wraps ValueError at the model construction boundary.
STRUCTURAL_CONFIG_ERROR = "Step text configuration is incomplete or unsupported"
LAYER_ARRAYS = (
    ("rope_theta", "layer_rope_thetas"),
    ("partial_rotary_factors", "partial_rotary_factors"),
    ("swiglu_limits", "swiglu_limits"),
    ("swiglu_limits_shared", "swiglu_limits_shared"),
)


@pytest.fixture(autouse=True)
def repository_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def native() -> dict:
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def model() -> BaseModelConfig:
    return BaseModelConfig.create_from_name(MODEL_NAME)


@pytest.fixture
def profiling() -> ProfilingModelConfig:
    return ProfilingModelConfig.from_model_name(MODEL_NAME)


@pytest.fixture
def raw_model_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise the real JSON boundary using an unrelated model name."""
    models_dir = tmp_path / "data/config/models"
    models_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    def load(payload: dict) -> BaseModelConfig:
        (models_dir / "unit__text_fixture.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return BaseModelConfig.create_from_name("unit/text_fixture")

    return load


def test_step_metadata_matches_the_locked_json_bytes() -> None:
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256


def test_nested_normalization_retains_the_entire_outer_fp8_policy(native: dict) -> None:
    original = deepcopy(native)
    normalized = _normalize_hf_text_config(native)
    assert native == original
    assert normalized["model_type"] == "step5"
    assert normalized["architectures"] == ["Step5ForCausalLM"]
    assert "text_config" not in normalized
    assert normalized["vision_config"]["width"] == original["vision_config"]["width"]
    assert normalized["vision_config"]["mlp_ratio"] == 8960 / 1536
    assert normalized["projector_bias"] is False
    assert normalized["quantization_config"] == original["quantization_config"]
    quantization = QuantizationConfig.from_dict(normalized["quantization_config"])
    assert len(quantization.ignored_layers) == 1406
    assert quantization.ignored_layers == (
        original["quantization_config"]["modules_to_not_convert"]
    )
    assert quantization.quant_method == "fp8"
    assert quantization.fmt == "e4m3"
    assert quantization.activation_scheme == "dynamic"
    assert quantization.weight_block_size == (128, 128)
    assert quantization.is_checkpoint_fp8_serialized is True
    assert normalized["layer_types"] == tuple(original["text_config"]["layer_types"][:92])
    assert normalized["mtp_layer_types"] == ("sliding_attention",) * 3
    assert normalized["rope_theta"] is None
    assert normalized["uses_fused_add_norm"] is True
    for source, target in LAYER_ARRAYS:
        assert normalized[target] == tuple(original["text_config"][source])
        assert len(normalized[target]) == 95


@pytest.mark.parametrize(("width", "fp32_residual", "expected"), [
    (4096, True, True),
    (16384, True, True),
    (16416, True, False),
    (4097, True, False),
    (4096, False, False),
])
def test_fused_add_norm_metadata_follows_residual_dtype_and_native_kernel_width(
    native: dict, width: int, fp32_residual: bool, expected: bool,
) -> None:
    native["text_config"]["hidden_size"] = width
    native["text_config"]["fp32_residual_connection"] = fp32_residual
    normalized = _normalize_hf_text_config(native)
    assert normalized["uses_fused_add_norm"] is expected


@pytest.mark.parametrize(("width", "fp32_residual", "explicit_flag"), [
    (4096, True, False),
    (4096, False, True),
    (4097, True, True),
])
def test_explicit_fused_add_norm_flag_cannot_override_native_source_metadata(
    native: dict, raw_model_loader,
    width: int, fp32_residual: bool, explicit_flag: bool,
) -> None:
    native["text_config"]["hidden_size"] = width
    native["text_config"]["fp32_residual_connection"] = fp32_residual
    native["text_config"]["uses_fused_add_norm"] = explicit_flag
    with pytest.raises(ValueError, match="uses_fused_add_norm"):
        raw_model_loader(native)


def test_nested_and_native_text_json_resolve_the_same_contract(
    native: dict, raw_model_loader,
) -> None:
    nested = raw_model_loader(native)
    text = deepcopy(native["text_config"])
    text["quantization_config"] = deepcopy(native["quantization_config"])
    unwrapped = raw_model_loader(text)
    text_payload = dataclass_to_dict(nested)
    text_payload.pop("vision_config")
    text_payload.pop("projector_bias")
    assert dataclass_to_dict(unwrapped) == text_payload
    assert unwrapped.vision_config is None
    assert unwrapped.get_layer_attention_specs() == nested.get_layer_attention_specs()
    assert unwrapped.get_model_architecture_profile().profile_id == "step5"
    assert unwrapped.get_name() == "unit/text_fixture"


def test_transformers_backbone_and_mtp_schedule_form_is_preserved(
    native: dict, raw_model_loader,
) -> None:
    original = raw_model_loader(native)
    text = native["text_config"]
    text["layer_types_with_mtp"] = list(text["layer_types"])
    text["layer_types"] = text["layer_types"][:92]
    normalized = raw_model_loader(native)
    assert normalized.get_layer_attention_specs() == original.get_layer_attention_specs()
    assert normalized.mtp_layer_types == original.mtp_layer_types
    assert normalized.layer_rope_thetas == original.layer_rope_thetas


def test_step_binds_every_backbone_layer_and_inventories_mtp_separately(
    model: BaseModelConfig,
) -> None:
    expected = tuple(
        LayerAttentionSpec(
            layer_id, "step5_attention", "sparse_gqa" if layer_id % 4 == 3 else "swa"
        )
        for layer_id in range(92)
    )
    assert model.num_layers == 92
    assert model.num_nextn_predict_layers == 3
    assert model.mtp_layer_types == ("sliding_attention",) * 3
    assert model.get_layer_attention_specs() == expected
    assert Counter(spec.variant_id for spec in expected) == {"swa": 69, "sparse_gqa": 23}
    assert [bind_layer_attention(model, layer_id) for layer_id in range(92)] == list(expected)
    assert all(spec.index_producer_layer_id is None for spec in expected)
    assert model.get_gdn_config() is None


@pytest.mark.parametrize("layer_id", [-1, 92, 93, 94, 95, True, 1.5, "3"])
def test_step_layer_binding_rejects_mtp_and_invalid_backbone_indices(
    model: BaseModelConfig, layer_id: object,
) -> None:
    with pytest.raises(ValueError, match="global_layer_id"):
        bind_layer_attention(model, layer_id)


def test_step_ffn_contracts_keep_all_dense_routed_and_shared_layers(
    model: BaseModelConfig,
) -> None:
    assert model.get_moe_layer_ids() == list(range(3, 91))
    assert model.get_num_moe_layers() == 88
    assert model.num_experts == 352
    assert model.num_experts_per_tok == 8
    assert model.supports_share_expert() is True
    profile = model.get_model_architecture_profile()
    assert profile.profile_id == "step5"
    contracts = [
        profile.resolve_layer_contract(
            model, layer_id=layer_id, attention_tp_size=8,
            moe_tp_size=1, expert_parallel_size=32,
        )
        for layer_id in range(92)
    ]
    assert [
        contract.layer_id
        for contract in contracts
        if contract.layer_kind is LayerKind.DENSE
    ] == [0, 1, 2, 91]
    assert Counter(contract.layer_kind for contract in contracts) == {
        LayerKind.DENSE: 4, LayerKind.ROUTED: 88,
    }
    for contract in contracts:
        if contract.layer_kind is LayerKind.DENSE:
            assert (contract.width, contract.tp_mode, contract.tensor_parallel_size) == (
                13824, TensorParallelMode.ATTENTION_TP, 8,
            )
            assert contract.expert_parallel_size is None
        else:
            assert (contract.width, contract.tp_mode, contract.tensor_parallel_size) == (
                1536, TensorParallelMode.MOE_TP, 1,
            )
            assert contract.expert_parallel_size == 32
            shared = profile.resolve_layer_contract(
                model, layer_id=contract.layer_id, layer_kind=LayerKind.SHARED,
                attention_tp_size=8, moe_tp_size=1, expert_parallel_size=32,
            )
            assert (shared.width, shared.tp_mode, shared.tensor_parallel_size) == (
                1536, TensorParallelMode.ATTENTION_TP, 8,
            )
            assert shared.expert_parallel_size is None


def test_runtime_and_profiling_serialization_preserve_the_same_step_contract(
    native: dict, model: BaseModelConfig, profiling: ProfilingModelConfig,
) -> None:
    runtime_payload = json.loads(json.dumps(dataclass_to_dict(model)))
    restored = ProfilingModelConfig(**json.loads(json.dumps(profiling.to_dict())))
    fields = (
        "num_layers", "num_q_heads", "num_kv_heads", "embedding_dim",
        "num_experts", "num_experts_per_tok", "moe_layers_enum", "layer_types",
        "num_nextn_predict_layers", "mtp_layer_types", "layer_rope_thetas",
        "partial_rotary_factors", "swiglu_limits", "swiglu_limits_shared",
        "sparse_config", "attention_other_setting", "sliding_window",
        "dense_mlp_hidden_dim", "routed_mlp_hidden_dim", "share_expert_dim",
        "attn_output_gate", "use_qk_norm", "norm_dtype", "rms_norm_eps",
        "fp32_residual_connection", "norm_expert_weight", "moe_router_activation",
        "moe_router_scaling_factor", "need_fp32_gate", "use_moe_router_bias",
        "zero_centered", "yarn_only_types", "use_rope_layers", "tie_word_embeddings",
        "fused_add_norm_capability", "vision_config", "projector_bias",
    )
    assert model.quantization_config is not None
    runtime_quantization = QuantizationConfig.from_dict(runtime_payload["quantization_config"])
    assert runtime_quantization.to_dict() == model.quantization_config.to_dict()
    runtime_fields = dict(runtime_payload)
    runtime_fields.pop("name")
    runtime_fields["activation"] = ActivationType(runtime_fields["activation"])
    runtime_fields["norm"] = NormType(runtime_fields["norm"])
    runtime_fields["quantization_config"] = runtime_quantization
    restored_runtime = type(model)(**runtime_fields)
    for source, target in LAYER_ARRAYS:
        assert tuple(runtime_payload[target]) == tuple(native["text_config"][source])
        assert len(runtime_payload[target]) == 95
    assert "text_config" not in runtime_payload
    assert runtime_payload["vision_config"] == model.vision_config
    assert runtime_payload["projector_bias"] is False
    assert not any(key.startswith("_") for key in runtime_payload)
    for config in (restored_runtime, profiling, restored):
        assert config.get_model_architecture_profile().profile_id == "step5"
        assert resolve_step5_model_precision(config) is PrecisionType.BF16
        assert config.uses_fused_add_norm is model.uses_fused_add_norm is True
        assert config.get_head_dim() == model.get_head_dim() == 192
        assert config.get_layer_attention_specs() == model.get_layer_attention_specs()
        assert config.get_moe_layer_ids() == list(range(3, 91))
        assert config.get_quant_signature() == model.get_quant_signature()
        assert config.quantization_config.to_dict() == model.quantization_config.to_dict()
        for name in fields:
            assert getattr(config, name) == getattr(model, name), name
        profile = config.get_model_architecture_profile()
        layer_kinds = (
            profile.resolve_layer_contract(config, layer_id=i).layer_kind
            for i in range(92)
        )
        assert Counter(layer_kinds) == {LayerKind.DENSE: 4, LayerKind.ROUTED: 88}
        for layer_id in range(92):
            assert resolve_step5_attention_geometry(
                config, layer_id, tensor_parallel_size=8,
            ) == resolve_step5_attention_geometry(
                model, layer_id, tensor_parallel_size=8,
            )
    assert resolve_step5_model_precision(model) is PrecisionType.BF16


@pytest.mark.parametrize("config_kind", ["runtime", "profiling"])
def test_tp8_step_geometry_preserves_kv_and_provider_replication(
    model: BaseModelConfig, profiling: ProfilingModelConfig, config_kind: str,
) -> None:
    config = model if config_kind == "runtime" else profiling
    sparse = resolve_step5_attention_geometry(config, 3, tensor_parallel_size=8)
    assert (sparse.num_q_heads, sparse.num_kv_heads, sparse.head_dim) == (64, 4, 192)
    assert (sparse.local_q_heads, sparse.local_kv_heads, sparse.kv_replication) == (8, 1, 2)
    assert (sparse.local_provider_groups, sparse.provider_replication) == (1, 2)
    assert (
        sparse.local_index_q_heads, sparse.local_index_k_heads, sparse.index_k_replication,
    ) == (4, 1, 8)
    assert sparse.proxy_dim == 256
    assert sparse.q_width == 12288
    assert sparse.local_qkv_width == 1920
    assert sparse.local_fused_indexer_width == 1544
    assert sparse.local_kv_heads * 8 == sparse.num_kv_heads * sparse.kv_replication
    swa = resolve_step5_attention_geometry(config, 0, tensor_parallel_size=8)
    assert (swa.local_q_heads, swa.local_kv_heads, swa.kv_replication) == (8, 1, 2)
    assert swa.local_provider_groups == swa.local_index_q_heads == swa.local_index_k_heads == 0
    assert swa.local_fused_indexer_width == 0
    assert config.sparse_config["topk"] == 512
    assert config.sparse_config["region_block_size"] == 8


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
def test_legacy_profiling_and_step_share_exact_kv_partitioning(
    model: BaseModelConfig, profiling: ProfilingModelConfig, tp_size: int,
) -> None:
    sparse = resolve_step5_attention_geometry(model, 3, tensor_parallel_size=tp_size)
    assert profiling.get_num_kv_heads(ParallelConfig(tensor_parallel_size=tp_size)) == sparse.local_kv_heads
    assert sparse.local_kv_heads * tp_size == model.num_kv_heads * sparse.kv_replication


def test_geometry_uses_the_selected_layer_attention_override(model: BaseModelConfig) -> None:
    changed = replace(model, attention_other_setting={
        "attention_type": "sliding_attention", "num_attention_heads": 32,
        "num_attention_groups": 4, "head_dim": 128,
    })
    swa = resolve_step5_attention_geometry(changed, 0, tensor_parallel_size=8)
    sparse = resolve_step5_attention_geometry(changed, 3, tensor_parallel_size=8)
    assert (swa.num_q_heads, swa.head_dim, swa.local_q_heads) == (32, 128, 4)
    assert (sparse.num_q_heads, sparse.head_dim, sparse.local_q_heads) == (64, 192, 8)


@pytest.mark.parametrize("tp_size", [0, True, 3, 6, 32])
def test_geometry_rejects_invalid_tp_and_unsupported_local_head_ratios(
    model: BaseModelConfig, tp_size: object,
) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size|local Q heads"):
        resolve_step5_attention_geometry(model, 3, tensor_parallel_size=tp_size)


def test_geometry_rejects_kv_groups_that_cannot_be_sharded_or_replicated(
    model: BaseModelConfig,
) -> None:
    changed = replace(model, attention_other_setting={
        "attention_type": "sliding_attention", "num_attention_heads": 48,
        "num_attention_groups": 3, "head_dim": 192,
    })
    with pytest.raises(ValueError, match="KV groups.*divide"):
        resolve_step5_attention_geometry(changed, 0, tensor_parallel_size=8)


@pytest.mark.parametrize(("model_type", "name", "expected"), [
    ("step5", "unit/model", "step5"),
    ("unit_model", MODEL_NAME, "generic"),
    (None, MODEL_NAME, "generic"),
])
def test_step_profile_identity_does_not_dispatch_on_the_model_name(
    model_type: str | None, name: str, expected: str,
) -> None:
    config = SimpleNamespace(model_type=model_type, name=name)
    assert get_model_architecture_profile(config).profile_id == expected


@pytest.mark.parametrize("source", [name for name, _ in LAYER_ARRAYS] + ["layer_types"])
def test_raw_step_arrays_must_include_all_backbone_and_mtp_entries(
    native: dict, raw_model_loader, source: str,
) -> None:
    native["text_config"][source] = native["text_config"][source][:-1]
    with pytest.raises(ValueError, match=source):
        raw_model_loader(native)


@pytest.mark.parametrize(("name", "value", "message"), [
    ("num_hidden_layers", True, "num_hidden_layers"),
    ("num_attention_heads", 64.5, "num_attention_heads"),
    ("num_attention_groups", 0, "num_attention_groups"),
    ("head_dim", "192", "head_dim"),
    ("head_dim", None, "head_dim"),
    ("hidden_size", False, "hidden_size"),
    ("moe_num_experts", "352", "moe_num_experts"),
    ("moe_top_k", 8.0, "moe_top_k"),
    ("num_nextn_predict_layers", -1, "num_nextn_predict_layers"),
    ("moe_layers_enum", "", "moe_layers_enum"),
    ("moe_layer_list", [3, True], "moe_layer_list"),
    ("norm_expert_weight", "true", "norm_expert_weight"),
    ("num_key_value_heads", 8, "Conflicting Step fields"),
    ("num_experts_per_tok", 4, "Conflicting Step fields"),
])
def test_raw_step_integer_boolean_and_alias_contracts_fail_closed(
    native: dict, raw_model_loader, name: str, value: object, message: str,
) -> None:
    native["text_config"][name] = value
    with pytest.raises(ValueError, match=message):
        raw_model_loader(native)


@pytest.mark.parametrize(("name", "message"), [
    ("head_dim", "head_dim"),
    ("sparse_config", STRUCTURAL_CONFIG_ERROR),
    ("norm_expert_weight", "norm_expert_weight"),
    ("moe_layers_enum", "moe_layers_enum"),
])
def test_raw_step_missing_required_fields_cannot_select_generic_attention(
    native: dict, raw_model_loader, name: str, message: str,
) -> None:
    del native["text_config"][name]
    with pytest.raises(ValueError, match=message):
        raw_model_loader(native)


@pytest.mark.parametrize("name", ["use_bias", "use_qkv_bias", "qkv_bias", "attention_bias"])
def test_raw_step_bias_flags_cannot_admit_a_native_bias_free_path(
    native: dict, raw_model_loader, name: str,
) -> None:
    native["text_config"][name] = True
    with pytest.raises((ValueError, NotImplementedError), match="bias"):
        raw_model_loader(native)


def test_raw_step_zero_centered_false_cannot_admit_unsupported_native_rmsnorm(
    native: dict, raw_model_loader,
) -> None:
    native["text_config"]["zero_centered"] = False
    with pytest.raises(NotImplementedError, match="zero.centered"):
        raw_model_loader(native)


@pytest.mark.parametrize("malformation", [
    "outer_type", "inner_type", "non_object", "double_nesting", "conflicting_quantization",
])
def test_nested_step_configuration_rejects_wrong_identity_and_conflicting_precision(
    native: dict, raw_model_loader, malformation: str,
) -> None:
    if malformation == "outer_type":
        native["model_type"] = "unit_model"
    elif malformation == "inner_type":
        native["text_config"]["model_type"] = "step3_text"
    elif malformation == "non_object":
        native["text_config"] = []
    elif malformation == "double_nesting":
        native["text_config"]["text_config"] = {}
    else:
        native["text_config"]["quantization_config"] = {"quant_method": "fp4"}
    with pytest.raises(ValueError, match="nested text_config|Conflicting outer"):
        raw_model_loader(native)


@pytest.mark.parametrize(("name", "value", "error", "message"), [
    ("torch_dtype", "float16", NotImplementedError, "BF16"),
    ("use_mla", True, ValueError, STRUCTURAL_CONFIG_ERROR),
    ("use_mfa", True, ValueError, STRUCTURAL_CONFIG_ERROR),
    ("head_dim", 96, NotImplementedError, "head_dim 128 or 192"),
    ("num_q_heads", 65, ValueError, STRUCTURAL_CONFIG_ERROR),
    ("layer_types", ("sliding_attention",) * 92, ValueError, STRUCTURAL_CONFIG_ERROR),
    ("mtp_layer_types", ("full_attention",), ValueError, STRUCTURAL_CONFIG_ERROR),
])
def test_resolved_step_configurations_reject_wrong_attention_contracts(
    model: BaseModelConfig, name: str, value: object, error: type[Exception], message: str,
) -> None:
    with pytest.raises(error, match=message):
        replace(model, **{name: value}).get_layer_attention_specs()


@pytest.mark.parametrize(("name", "value", "error", "message"), [
    ("proxy_dim", 128, NotImplementedError, "proxy_dim"),
    ("region_block_size", 4, NotImplementedError, "region_block_size"),
    ("sparse_indexer_num_k_heads", 2, NotImplementedError, "sparse_indexer_num_k_heads"),
    ("num_provider_groups", 8, ValueError, STRUCTURAL_CONFIG_ERROR),
    ("sparse_indexer_num_heads", 15, ValueError, STRUCTURAL_CONFIG_ERROR),
    ("apply_to_layer_types", ["sliding_attention"], ValueError, STRUCTURAL_CONFIG_ERROR),
    ("enabled", 1, NotImplementedError, "enabled"),
])
def test_resolved_step_sparse_geometry_fails_closed_on_wrong_units_and_heads(
    model: BaseModelConfig, name: str, value: object, error: type[Exception], message: str,
) -> None:
    sparse_config = dict(model.sparse_config, **{name: value})
    with pytest.raises(error, match=message):
        replace(model, sparse_config=sparse_config).get_layer_attention_specs()


@pytest.mark.parametrize(("name", "value", "error", "message"), [
    ("layer_rope_thetas", float("nan"), ValueError, STRUCTURAL_CONFIG_ERROR),
    ("partial_rotary_factors", True, ValueError, STRUCTURAL_CONFIG_ERROR),
    ("partial_rotary_factors", 2.0, ValueError, STRUCTURAL_CONFIG_ERROR),
    ("swiglu_limits", 8.0, NotImplementedError, "routed SwiGLU"),
    ("swiglu_limits_shared", -1.0, ValueError, STRUCTURAL_CONFIG_ERROR),
])
def test_per_layer_array_values_are_validated_beyond_the_first_layer(
    model: BaseModelConfig, name: str, value: object,
    error: type[Exception], message: str,
) -> None:
    values = list(getattr(model, name))
    values[88] = value
    with pytest.raises(error, match=message):
        replace(model, **{name: tuple(values)}).get_layer_attention_specs()


def test_operator_manifest_inventories_both_step_variants_and_mixed_ffn(
    model: BaseModelConfig, profiling: ProfilingModelConfig,
) -> None:
    expected = {
        ("step5_attention", "swa"), ("step5_attention", "sparse_gqa"),
        ("memory", "replicated"), ("ffn", "dense"),
        ("moe", "routed"), ("share_expert", "shared_dense"),
        ("step5_ffn", "fp32_branch_merge"),
    }
    for config in (model, profiling):
        manifest = build_operator_manifest(config)
        assert {
            (binding.family_id, binding.variant_id)
            for binding in manifest.family_bindings
        } == expected
        assert len(manifest.family_bindings) == len(expected)
        operators = manifest.operators()
        assert len({operator.name for operator in operators}) == len(operators)
        for binding in manifest.family_bindings:
            if binding.family_id in {"step5_attention", "step5_ffn"}:
                assert binding.frozen is True
                with pytest.raises(NotImplementedError, match="frozen"):
                    binding.require_enabled_for_execution()


@pytest.mark.parametrize("dummy_mode", [False, True])
def test_step_predictor_rejects_measured_mode_and_unadmitted_structural_layout(
    model: BaseModelConfig, monkeypatch: pytest.MonkeyPatch, dummy_mode: bool,
) -> None:
    from frontier.config import MetricsConfig, RandomForrestExecutionTimePredictorConfig, ReplicaConfig, VllmV1SchedulerConfig
    from frontier.types import ClusterType

    monkeypatch.setattr(global_vars, "SYS_ARCH", "co-location")
    replica = ReplicaConfig(
        model_name=MODEL_NAME, device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, attn_dp=4, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=32, attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )
    replica.model_config = model
    # Full EP over the flattened TP x DP ranks is admitted only on PD roles.
    expected_error = "MONOLITHIC, PREFILL or DECODE" if dummy_mode else "Step attention"
    with pytest.raises(NotImplementedError, match=expected_error):
        RandomForrestExecutionTimePredictor(
            predictor_config=RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=dummy_mode),
            replica_config=replica, replica_scheduler_config=VllmV1SchedulerConfig(block_size=64),
            metrics_config=MetricsConfig(), cluster_type=ClusterType.MONOLITHIC,
        )


def test_step_generic_profiler_and_attention_execution_remain_guarded(
    model: BaseModelConfig, profiling: ProfilingModelConfig,
) -> None:
    family = get_attention_family("step5_attention")
    assert family.execution_enabled is False
    assert family.structural_execution_enabled is True
    with pytest.raises(NotImplementedError, match="Step attention"):
        family.require_enabled_for_execution()
    family.require_enabled_for_execution(structural=True)
    for config in (model, profiling):
        with pytest.raises(ValueError, match="explicit global layer id"):
            bind_attention_family(config)
        with pytest.raises(NotImplementedError, match="Step native.*producer"):
            build_profiling_plan(
                config, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1], is_moe=True,
            )
