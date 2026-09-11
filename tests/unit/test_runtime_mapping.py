import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.validation.runtime_mapping import map_runtime_components


def mapping(components, **kwargs):
    options = dict(phase="decode", layer_id=0, measurement_type="HIP_GRAPH_EVENT",
        model_config=ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4"))
    options.update(kwargs)
    return map_runtime_components(components, **options)


def test_combined_mlp_reduction_and_missing_gates_cannot_be_imported_twice():
    result = mapping(["mlp_tp_allreduce", "shared_gate_sigmoid_mul_add"])
    reduction, gate = result["mappings"]
    assert reduction["relationship"] == "fused_collective"
    assert len(reduction["native_targets"]) == 2
    assert gate["relationship"] == "native_gap" and not gate["native_targets"]
    assert not result["native_measurement_family_supported"]
    assert not any(row["native_csv_export_admitted"] for row in result["mappings"])
    attention = mapping(["attn_post_proj_gate"], layer_id=3, measurement_type="KERNEL_ONLY")
    assert attention["mappings"][0]["relationship"] == "native_gap"
    assert attention["native_measurement_family_supported"]


def test_fused_residuals_keep_cross_layer_ownership():
    first = mapping(["input_layernorm"])["mappings"][0]
    later = mapping(["input_layernorm"], layer_id=1)["mappings"][0]
    assert len(first["native_targets"]) == 1
    assert later["native_targets"][1] == {"operator": "add_ffn_residual", "family": "memory", "layer_offset": -1}
    assert later["relationship"] == "fused_cross_layer"


def test_partial_event_scopes_are_not_whole_native_operators():
    result = mapping(["gdn_attn", "gdn_out_proj", "moe_experts"])
    assert [row["relationship"] for row in result["mappings"]] == ["partial", "partial", "composite"]
    assert {t["operator"] for t in result["mappings"][2]["native_targets"]} == {"moe_shuffling", "moe_grouped_gemm"}


@pytest.mark.parametrize("components,kwargs", [
    (["invented"], {}), (["gdn_core_prefill"], {}), (["gdn"], {"layer_id": 3}),
    (["full_attention"], {}), (["gdn"], {"phase": "mixed"}),
    (["gdn", "gdn"], {}), (["gdn"], {"measurement_type": "EVENT_AS_KERNELS"}),
])
def test_unknown_or_mismatched_runtime_mapping_fails(components, kwargs):
    with pytest.raises(ValueError):
        mapping(components, **kwargs)
