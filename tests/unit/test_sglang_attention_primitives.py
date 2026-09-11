from copy import deepcopy

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.runtime.sglang_attention import (
    ATTENTION_PRIMITIVES,
    attention_decode_spec,
    validate_attention_decode_spec,
    validate_attention_workload,
)
from frontier.profiling.runtime.sglang_primitives import DEFAULT_PRIMITIVES
from frontier.runtime_cost.primitives import PrimitiveCalibration
from frontier.runtime_cost.sglang import CostQuery, DecodeWorkload
from tests.unit.test_primitive_calibration import IDENTITY, profiles


@pytest.fixture
def model():
    return ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")


def test_qwen38_tp8_attention_decode_contract(model):
    spec = attention_decode_spec(model, 8)
    assert spec["q_heads"] == 8
    assert spec["kv_heads"] == 1
    assert spec["head_dim"] == 256
    assert spec["rotary_dim"] == 64
    assert spec["projected_qkv_gate_width"] == 4608
    assert spec["local_qkv_weight_shape"] == (4608, 8192)
    assert spec["attention_backend"] == "aiter.paged_attention_ragged"
    assert spec["kv_cache_layout"] == "NHD"
    assert spec["page_size"] == 1
    assert spec["output_gate"]
    assert not spec["includes_output_projection"]
    validate_attention_decode_spec(spec, 8)


def test_context_sensitive_primitives_are_opt_in(model):
    assert not set(ATTENTION_PRIMITIVES) & set(DEFAULT_PRIMITIVES)
    assert validate_attention_workload(
        24, 20, (1029,) * 20 + (1,) * 4, attention_decode_spec(model, 8)
    ) == {
        "logical_size": 20,
        "padding_count": 4,
        "active_context_length": 1029,
        "padding_context_length": 1,
    }


def attention_profiles(model, split="calibration", sizes=(16, 32), context=1029):
    result = profiles(split, sizes)
    spec = attention_decode_spec(model, 8)
    for payload in result:
        base = [row for row in payload["rows"] if row["primitive"] == "gemma_norm"]
        payload["rows"] = []
        for name in ATTENTION_PRIMITIVES:
            for row in base:
                row = deepcopy(row)
                size = row["physical_size"]
                logical = size - 4
                contexts = (context,) * logical + (1,) * 4
                row.update(
                    primitive=name,
                    logical_size=logical,
                    physical_context_lens=list(contexts),
                    attention_decode_spec=deepcopy(spec),
                    attention_workload=validate_attention_workload(
                        size, logical, contexts, spec),
                )
                payload["rows"].append(row)
    return result


def test_all_attention_scopes_fit_and_price_exact_context(model):
    fit = PrimitiveCalibration(attention_profiles(model), identity=IDENTITY)
    validation = attention_profiles(model, "validation", (24,))
    assert fit.validate(validation)["all_primitives_passed"]
    workload = DecodeWorkload(20, (1029,) * 20 + (1,) * 4)
    for component in ATTENTION_PRIMITIVES:
        query = CostQuery(IDENTITY, 3, "attention", component, workload)
        assert fit(query).basis == "isolated_compute"


@pytest.mark.parametrize("contexts", [
    (1030,) * 20 + (1,) * 4,
    (1029,) * 20 + (2,) * 4,
    (1029,) * 24,
])
def test_query_cannot_change_context_or_padding_ownership(model, contexts):
    fit = PrimitiveCalibration(attention_profiles(model), identity=IDENTITY)
    logical = 24 if len(set(contexts)) == 1 else 20
    query = CostQuery(
        IDENTITY, 3, "attention", "attn_decode", DecodeWorkload(logical, contexts))
    with pytest.raises(ValueError, match="Attention"):
        fit(query)


@pytest.mark.parametrize("mutation", ["shape", "summary", "fit_context", "validation_context"])
def test_changed_attention_contract_fails_closed(model, mutation):
    data = attention_profiles(model)
    if mutation == "shape":
        data[0]["rows"][0]["attention_decode_spec"]["head_dim"] = 128
        with pytest.raises(ValueError):
            PrimitiveCalibration(data, identity=IDENTITY)
    elif mutation == "summary":
        data[0]["rows"][0]["attention_workload"]["active_context_length"] = 7
        with pytest.raises(ValueError, match="summary"):
            PrimitiveCalibration(data, identity=IDENTITY)
    elif mutation == "fit_context":
        for payload in data:
            for row in payload["rows"]:
                if row["physical_size"] == 32:
                    row["physical_context_lens"][:28] = [1030] * 28
                    row["attention_workload"]["active_context_length"] = 1030
        with pytest.raises(ValueError, match="context lengths"):
            PrimitiveCalibration(data, identity=IDENTITY)
    else:
        fit = PrimitiveCalibration(data, identity=IDENTITY)
        validation = attention_profiles(model, "validation", (24,), context=1030)
        with pytest.raises(ValueError, match="context ownership"):
            fit.validate(validation)
