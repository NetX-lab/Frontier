from copy import deepcopy
from dataclasses import replace

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.runtime.sglang_dense import DENSE_PRIMITIVES, dense_primitive_spec, validate_dense_spec
from frontier.runtime_cost.primitives import PrimitiveCalibration
from frontier.runtime_cost.sglang import CostQuery, DecodeWorkload
from tests.unit.test_primitive_calibration import IDENTITY, profiles


@pytest.fixture
def model():
    return ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")


@pytest.mark.parametrize("name,width,outputs,weights", [
    ("shared_expert_gate_up", 8192, (512,), ((512, 8192),)),
    ("shared_expert_activation", 512, (256,), ()),
    ("shared_expert_down", 256, (8192,), ((8192, 256),)),
    ("moe_router_linear", 8192, (512,), ((512, 8192),)),
    ("gdn_input_projections", 8192, (4608, 32), ((4608, 8192), (32, 8192))),
    ("gdn_output_projection", 2048, (8192,), ((8192, 2048),)),
])
def test_native_shards_and_both_gdn_input_projections(model, name, width, outputs, weights):
    spec = dense_primitive_spec(name, model, 8)
    assert (spec["input_width"], spec["output_widths"], spec["local_weight_shapes"]) == (width, outputs, weights)
    assert not spec["includes_reduction"]
    assert spec["input_transform"] == ("gdn_rmsnorm_gated" if name == "gdn_output_projection" else "none")
    validate_dense_spec(spec, 8, name)


def test_router_is_replicated_and_gdn_heads_use_model_override(model):
    assert dense_primitive_spec("moe_router_linear", model, 4)["local_weight_shapes"] == ((512, 8192),)
    assert dense_primitive_spec("shared_expert_gate_up", model, 4)["output_widths"] == (1024,)
    model.linear_num_value_heads = 64
    assert dense_primitive_spec("gdn_input_projections", model, 8)["output_widths"] == (2560, 16)


@pytest.mark.parametrize("tp", [0, True, 3])
def test_invalid_sharding_fails(model, tp):
    with pytest.raises(ValueError): dense_primitive_spec("gdn_input_projections", model, tp)


def test_quantized_dense_weights_cannot_be_labeled_bf16(model):
    model.quantization_config = replace(model.quantization_config, quantized_operations=None)
    with pytest.raises(ValueError, match="unquantized"):
        dense_primitive_spec("shared_expert_gate_up", model, 8)


def dense_profiles(model, split="calibration", sizes=(16, 32)):
    result = profiles(split, sizes)
    for payload in result:
        base = payload["rows"][:2 * len(sizes)]
        payload["rows"] = []
        for name in DENSE_PRIMITIVES:
            for row in base:
                row = deepcopy(row)
                row.update(primitive=name, dense_spec=dense_primitive_spec(name, model, 8))
                payload["rows"].append(row)
    return result


def test_all_dense_scopes_integrate_without_covering_core_or_routed_experts(model):
    fit = PrimitiveCalibration(dense_profiles(model), identity=IDENTITY)
    assert fit.validate(dense_profiles(model, "validation", (24,)))["all_primitives_passed"]
    for component in DENSE_PRIMITIVES:
        query = CostQuery(IDENTITY, 0, "gdn", component, DecodeWorkload(20, (1029,) * 20 + (1,) * 4))
        assert fit(query).basis == "isolated_compute"
    for component in ("gdn_core_decode", "mlp_tp_allreduce", "moe_routing_topk"):
        with pytest.raises(ValueError, match="Missing"):
            fit(CostQuery(IDENTITY, 0, "gdn", component, DecodeWorkload(1, (1029,))))


@pytest.mark.parametrize("field,value", [("includes_reduction", True), ("weight_dtype", "float32"),
    ("local_weight_shapes", ((64, 8192),)), ("input_width", True), ("output_partitions", ((513,),))])
def test_malformed_rank_shape_evidence_fails(model, field, value):
    data = dense_profiles(model)
    data[0]["rows"][0]["dense_spec"][field] = value
    with pytest.raises(ValueError): PrimitiveCalibration(data, identity=IDENTITY)


def test_validation_cannot_silently_change_dense_shape(model):
    fit = PrimitiveCalibration(dense_profiles(model), identity=IDENTITY)
    validation = dense_profiles(model, "validation", (24,))
    for payload in validation:
        for row in payload["rows"]:
            if row["primitive"] == "moe_router_linear":
                spec = row["dense_spec"]
                spec.update(output_partitions=((256,),), output_widths=(256,), local_weight_shapes=((256, 8192),))
    with pytest.raises(ValueError, match="shape contract changed"):
        fit.validate(validation)


def test_gdn_output_cannot_drop_its_gated_norm():
    model = ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")
    spec = dense_primitive_spec("gdn_output_projection", model, 8)
    spec["input_transform"] = "none"
    with pytest.raises(ValueError, match="ownership"):
        validate_dense_spec(spec, 8, "gdn_output_projection")
