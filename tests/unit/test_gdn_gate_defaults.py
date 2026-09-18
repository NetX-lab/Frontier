"""Preserve GDN gate defaults, raw values and validation boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.gdn.config import GatedDeltaNetConfig, resolve_gdn_shape
from frontier.config.model_config import BaseModelConfig
from frontier.profiling.common.model_config import ModelConfig
from frontier.types import ActivationType, NormType


GATE_CASES = [
    pytest.param({}, "silu", "silu", id="omitted"),
    pytest.param({"gdn_output_gate_type": None}, None, None, id="null"),
    pytest.param({"gdn_output_gate_type": "swish"}, "swish", "silu", id="swish"),
    pytest.param({"gdn_output_gate_type": " SWISH "}, " SWISH ", "silu", id="alias"),
    pytest.param({"gdn_output_gate_type": "sigmoid"}, "sigmoid", "sigmoid", id="sigmoid"),
    pytest.param({"gdn_output_gate_type": ""}, "", None, id="empty"),
]


def _model_values(hybrid):
    return dict(
        num_layers=4, num_q_heads=4, num_kv_heads=2, embedding_dim=256,
        mlp_hidden_dim=64, max_position_embeddings=4096, vocab_size=1024,
        use_gated_mlp=True, use_bias=False, use_qkv_bias=False,
        activation=ActivationType.SILU, norm=NormType.RMS_NORM,
        post_attn_norm=True, is_moe=True, num_experts=8, num_experts_per_tok=2,
        model_type="qwen3_5_moe_text" if hybrid else "qwen3_next",
        linear_conv_kernel_dim=4, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_num_key_heads=2,
        linear_num_value_heads=4, full_attention_interval=4,
    )


def _gate_error(raw):
    return f"GDN output_gate_type must be silu/swish or sigmoid, got {str(raw)!r}"


@pytest.mark.parametrize("overrides,raw,normalized", GATE_CASES)
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("config_type", [BaseModelConfig, ModelConfig])
def test_gate_constructor_preserves_raw_and_resolved_values(
    config_type, hybrid, overrides, raw, normalized
):
    values = _model_values(hybrid) | overrides
    if config_type is ModelConfig:
        values["name"] = "gate-contract"
    if hybrid and normalized is None:
        with pytest.raises(ValueError, match="both GDN and full-attention layers") as error:
            config_type(**values)
        assert str(error.value.__cause__) == _gate_error(raw)
        return

    config = config_type(**values)
    assert config.gdn_output_gate_type == (str(raw) if config_type is ModelConfig else raw)
    gdn = config.get_gdn_config()
    if hybrid:
        assert gdn.output_gate_type == normalized
        assert config.get_num_gdn_layers() == 3
        assert config.get_layer_attention_spec(3).is_full_attention
    else:
        assert gdn is None
        assert config.get_num_full_attention_layers() == 4


@pytest.mark.parametrize("overrides,raw,normalized", GATE_CASES)
@pytest.mark.parametrize("config_type", [BaseModelConfig, ModelConfig])
def test_hf_gate_values_preserve_loader_behavior(
    monkeypatch, tmp_path, config_type, overrides, raw, normalized
):
    source = Path(__file__).resolve().parents[2] / "data/config/models/Qwen3.8-2.4T-A95B-Quark-MXFP4.json"
    values = json.loads(source.read_text(encoding="utf-8"))
    if overrides:
        values["output_gate_type"] = overrides["gdn_output_gate_type"]
    directory = tmp_path / "data/config/models"
    directory.mkdir(parents=True)
    (directory / "gate-contract.json").write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loader = (
        BaseModelConfig.create_from_name
        if config_type is BaseModelConfig
        else ModelConfig.from_model_name
    )
    if normalized is None:
        with pytest.raises(ValueError, match="both GDN and full-attention layers") as error:
            loader("gate-contract")
        assert str(error.value.__cause__) == _gate_error(raw)
        return

    config = loader("gate-contract")
    assert config.gdn_output_gate_type == raw
    assert config.get_gdn_config().output_gate_type == normalized
    assert config.get_num_gdn_layers() == 69
    assert config.get_num_full_attention_layers() == 23


@pytest.mark.parametrize("overrides,raw,normalized", GATE_CASES)
def test_shape_resolver_preserves_missing_and_explicit_gate_values(
    overrides, raw, normalized
):
    config = SimpleNamespace(**(_model_values(True) | overrides))
    if normalized is None:
        with pytest.raises(ValueError) as error:
            resolve_gdn_shape(config)
        assert str(error.value) == _gate_error(raw)
    else:
        assert resolve_gdn_shape(config).output_gate_type == normalized


def test_gate_default_does_not_change_absent_shape_semantics():
    assert GatedDeltaNetConfig(4, 32, 32, 2, 4).output_gate_type == "silu"
    assert resolve_gdn_shape(SimpleNamespace(gdn_output_gate_type=None)) is None
    with pytest.raises(ValueError, match="^GDN shape configuration is incomplete$"):
        resolve_gdn_shape(SimpleNamespace(linear_conv_kernel_dim=4))
