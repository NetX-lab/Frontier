import pandas as pd

from frontier.entities.time_components import (
    AttentionOperatorTimes,
    AttentionTime,
    canonical_operator_execution_time_attrs,
)
from frontier.gdn import GATED_DELTA_NET_FAMILY
from frontier.gdn.profiling_schema import (
    get_required_gdn_profiling_columns,
    validate_gdn_profiling_dataframe,
)
from frontier.operators.families import get_operator_family
from frontier.operators.spec import OperatorPhase, OperatorRole


def test_gdn_family_is_registered_with_phase_specific_cores() -> None:
    assert get_operator_family("gated_delta_net") is GATED_DELTA_NET_FAMILY
    operators = {operator.name: operator for operator in GATED_DELTA_NET_FAMILY.operators}

    assert operators["gdn_core_prefill"].role is OperatorRole.PREFILL_KERNEL
    assert operators["gdn_core_prefill"].phases == (OperatorPhase.PREFILL,)
    assert operators["gdn_core_decode"].role is OperatorRole.DECODE_KERNEL
    assert operators["gdn_core_decode"].phases == (OperatorPhase.DECODE,)
    assert operators["gdn_core_mixed"].role is OperatorRole.MIXED_KERNEL
    assert operators["gdn_core_mixed"].phases == (OperatorPhase.MIXED,)


def test_gdn_structured_times_replace_legacy_attention_carriers_once() -> None:
    operator_times = AttentionOperatorTimes(
        {
            "gdn_input_projections": 1.0,
            "gdn_core_prefill": 2.0,
            "gdn_output_projection": 3.0,
        }
    )
    component = AttentionTime(
        attention_layer_pre_proj_execution_time=1.0,
        attention_prefill_execution_time=2.0,
        attention_layer_post_proj_execution_time=3.0,
        operator_times=operator_times,
    )

    assert component.total_time() == 6.0
    attrs = canonical_operator_execution_time_attrs()
    assert attrs["gdn_input_projections"] == (
        "attention_layer_pre_proj_execution_time"
    )
    assert attrs["gdn_core_decode"] == "attention_decode_execution_time"


def test_gdn_csv_schema_requires_backend_state_and_validation_timing() -> None:
    columns = get_required_gdn_profiling_columns()
    row = {column: 1 for column in columns}
    row.update(
        {
            "measurement_type": "CUDA_EVENT",
            "model_architecture_profile": "qwen3_5_moe",
            "quant_signature": "none",
            "device": "mi355x",
            "runtime_stack_signature": "unit",
            "gdn_runtime_backend": "vllm_rocm",
            "gdn_rank_aggregation": "single_rank",
            "gdn_prefill_backend": "triton",
            "gdn_decode_backend": "packed_recurrent_triton",
            "gqa_interleaved_layout": False,
            "packed_recurrent_decode": True,
            "model_dtype": "bfloat16",
            "conv_state_dtype": "bfloat16",
            "recurrent_state_dtype": "bfloat16",
            "has_initial_state": False,
        }
    )

    validate_gdn_profiling_dataframe(pd.DataFrame([row]))


def test_gdn_csv_schema_rejects_missing_runtime_backend() -> None:
    columns = [
        column
        for column in get_required_gdn_profiling_columns()
        if column != "gdn_runtime_backend"
    ]
    row = {column: 1 for column in columns}
    row["measurement_type"] = "CUDA_EVENT"

    try:
        validate_gdn_profiling_dataframe(pd.DataFrame([row]))
    except ValueError as exc:
        assert "gdn_runtime_backend" in str(exc)
    else:
        raise AssertionError("missing GDN runtime backend was accepted")
