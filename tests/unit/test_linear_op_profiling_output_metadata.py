"""Unit tests for linear-op profiling output metadata contract."""

from __future__ import annotations

import pandas as pd
import pytest

from frontier.profiling.linear_op import main as linear_op_main


def test_attach_linear_op_output_metadata_writes_model_architecture_profile() -> None:
    df = pd.DataFrame(
        [
            {
                "num_tokens": 1,
                "time_stats.attn_rope.mean": 2.5,
            }
        ]
    )

    output = linear_op_main._attach_linear_op_output_metadata(  # pylint: disable=protected-access
        df,
        precision_str="FP16",
        model_arch="generic",
        model_architecture_profile="step2_mini",
        quant_signature="none",
        measurement_type="KERNEL_ONLY",
    )

    assert output.loc[0, "profiling_precision"] == "FP16"
    assert output.loc[0, "model_arch"] == "generic"
    assert output.loc[0, "model_architecture_profile"] == "step2_mini"
    assert output.loc[0, "quant_signature"] == "none"
    assert output.loc[0, "measurement_type"] == "KERNEL_ONLY"
    assert output.loc[0, "time_stats.attn_rope.mean"] == 2.5


@pytest.mark.parametrize(
    ("column_name", "existing_value", "expected_value"),
    [
        ("profiling_precision", "FP16", "BF16"),
        ("measurement_type", "CUDA_EVENT", "KERNEL_ONLY"),
        ("model_arch", "generic", "step2_mini"),
        ("model_architecture_profile", "generic", "step2_mini"),
        ("quant_signature", "none", "fp8_w8a8"),
    ],
)
def test_attach_linear_op_output_metadata_rejects_conflicting_metadata(
    column_name: str,
    existing_value: str,
    expected_value: str,
) -> None:
    df = pd.DataFrame(
        [
            {
                "num_tokens": 1,
                column_name: existing_value,
                "time_stats.attn_rope.mean": 2.5,
            }
        ]
    )

    kwargs = {
        "precision_str": "FP16",
        "model_arch": "generic",
        "model_architecture_profile": "generic",
        "quant_signature": "none",
        "measurement_type": "KERNEL_ONLY",
    }
    if column_name == "profiling_precision":
        kwargs["precision_str"] = expected_value
    elif column_name == "measurement_type":
        kwargs["measurement_type"] = expected_value
    elif column_name == "model_arch":
        kwargs["model_arch"] = expected_value
    elif column_name == "model_architecture_profile":
        kwargs["model_architecture_profile"] = expected_value
    elif column_name == "quant_signature":
        kwargs["quant_signature"] = expected_value
    else:
        raise AssertionError(f"Unhandled metadata column: {column_name}")

    with pytest.raises(ValueError, match=column_name):
        linear_op_main._attach_linear_op_output_metadata(  # pylint: disable=protected-access
            df,
            **kwargs,
        )
