from copy import deepcopy

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.runtime.sglang_gdn import gdn_core_spec, validate_gdn_core_spec
from frontier.profiling.runtime.sglang_primitives import DEFAULT_PRIMITIVES
from frontier.runtime_cost.primitives import PrimitiveCalibration
from frontier.runtime_cost.sglang import CostQuery, DecodeWorkload
from tests.unit.test_primitive_calibration import IDENTITY, profiles


@pytest.fixture
def model():
    return ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")


def test_qwen38_tp8_packed_decode_state_contract(model):
    spec = gdn_core_spec(model, 8)
    assert spec["q_heads"] == 2
    assert spec["value_heads"] == 16
    assert spec["mixed_qkv_width"] == 2560
    assert spec["projected_qkvz_width"] == 4608
    assert spec["projected_ba_width"] == 32
    assert spec["conv_state_shape"] == (2560, 3)
    assert spec["recurrent_state_shape"] == (16, 128, 128)
    assert spec["padding_cache_index"] == -1
    assert spec["includes_input_reordering"]
    assert spec["includes_gate_materialization"]
    assert not spec["includes_gated_norm"]
    validate_gdn_core_spec(spec, 8)


def test_stateful_core_is_opt_in_to_preserve_profiler_defaults():
    assert "gdn_core_decode" not in DEFAULT_PRIMITIVES
    assert "gdn_output_projection" in DEFAULT_PRIMITIVES


def core_profiles(model, split="calibration", sizes=(16, 32)):
    result = profiles(split, sizes)
    for payload in result:
        base = [row for row in payload["rows"] if row["primitive"] == "gemma_norm"]
        payload["rows"] = []
        for row in base:
            row = deepcopy(row)
            row.update(primitive="gdn_core_decode", logical_size=row["physical_size"] - 4,
                       gdn_core_spec=gdn_core_spec(model, 8))
            payload["rows"].append(row)
    return result


def test_core_fit_is_bounded_to_matching_padding_ownership(model):
    fit = PrimitiveCalibration(core_profiles(model), identity=IDENTITY)
    query = CostQuery(IDENTITY, 0, "gdn", "gdn_core_decode",
                      DecodeWorkload(20, (1029,) * 20 + (1,) * 4))
    assert fit(query).basis == "isolated_compute"
    assert fit.validate(core_profiles(model, "validation", (24,)))["all_primitives_passed"]
    unpadded = CostQuery(IDENTITY, 0, "gdn", "gdn_core_decode",
                         DecodeWorkload(24, (1029,) * 24))
    with pytest.raises(ValueError, match="padding"):
        fit(unpadded)


@pytest.mark.parametrize("mutation", ["state", "logical", "padding_fit", "validation_padding"])
def test_malformed_or_changed_core_contract_fails(model, mutation):
    data = core_profiles(model)
    if mutation == "state":
        data[0]["rows"][0]["gdn_core_spec"]["recurrent_state_shape"] = (8, 128, 128)
        with pytest.raises(ValueError):
            PrimitiveCalibration(data, identity=IDENTITY)
    elif mutation == "logical":
        data[0]["rows"][0]["logical_size"] = 0
        with pytest.raises(ValueError):
            PrimitiveCalibration(data, identity=IDENTITY)
    elif mutation == "padding_fit":
        for payload in data:
            for row in payload["rows"]:
                if row["physical_size"] == 32:
                    row["logical_size"] = 24
        with pytest.raises(ValueError, match="fixed padding"):
            PrimitiveCalibration(data, identity=IDENTITY)
    else:
        fit = PrimitiveCalibration(data, identity=IDENTITY)
        validation = core_profiles(model, "validation", (24,))
        for payload in validation:
            for row in payload["rows"]:
                row["logical_size"] = 24
        with pytest.raises(ValueError, match="padding ownership"):
            fit.validate(validation)
