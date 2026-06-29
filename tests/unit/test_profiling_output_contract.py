"""Contracts for profiling output paths and measurement aliases."""

from pathlib import Path

import pytest

from frontier.profiling.utils import (
    ProfileMethod,
    build_profiling_output_path,
    profile_method_to_measurement_type,
)
from frontier.types import MeasurementType


def test_build_profiling_output_path_matches_release_schema() -> None:
    path = build_profiling_output_path(
        output_root="data/profiling",
        profiling_type="compute",
        hardware="a800",
        model_name="mixtral_8x7b_moe",
        op_name="attention",
    )

    assert path == Path("data/profiling/compute/a800/mixtral_8x7b_moe/attention.csv")


def test_build_profiling_output_path_preserves_nested_hf_model_names() -> None:
    path = build_profiling_output_path(
        output_root="data/profiling",
        profiling_type="compute",
        hardware="rtx_pro_6000",
        model_name="meta-llama/Llama-2-7b-hf",
        op_name="linear_op",
    )

    assert path == Path(
        "data/profiling/compute/rtx_pro_6000/meta-llama/Llama-2-7b-hf/linear_op.csv"
    )


@pytest.mark.parametrize(
    ("profile_method", "expected"),
    [
        ("cuda", MeasurementType.CUDA_EVENT),
        ("kernel_only", MeasurementType.KERNEL_ONLY),
        (ProfileMethod.CUDA.value, MeasurementType.CUDA_EVENT),
        (ProfileMethod.KERNEL_ONLY.value, MeasurementType.KERNEL_ONLY),
    ],
)
def test_profile_method_accepts_release_time_metric_aliases(
    profile_method: str, expected: MeasurementType
) -> None:
    assert profile_method_to_measurement_type(profile_method) == expected


def test_build_profiling_output_path_uses_kernel_only_suffix_for_kernel_metric() -> None:
    from frontier.profiling.utils import build_profile_method_output_path

    path = build_profile_method_output_path(
        output_root="data/profiling",
        profiling_type="compute",
        hardware="rtx_pro_6000",
        model_name="Qwen3-30B-A3B-tiny",
        op_name="moe",
        profile_method="kernel_only",
    )

    assert path == Path(
        "data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/moe_kernel_only.csv"
    )


def test_build_profiling_output_path_keeps_cuda_metric_canonical_filename() -> None:
    from frontier.profiling.utils import build_profile_method_output_path

    path = build_profile_method_output_path(
        output_root="data/profiling",
        profiling_type="compute",
        hardware="rtx_pro_6000",
        model_name="qwen2_dense_test",
        op_name="linear_op",
        profile_method="cuda",
    )

    assert path == Path(
        "data/profiling/compute/rtx_pro_6000/qwen2_dense_test/linear_op.csv"
    )


def test_attention_standard_output_uses_profile_method_output_path() -> None:
    source = (Path(__file__).resolve().parents[2] / "frontier/profiling/attention/main.py").read_text(
        encoding="utf-8"
    )
    standard_save_block = source.split("# Save standard attention results", 1)[1].split(
        "# Save mixed-batch results separately", 1
    )[0]

    assert "build_profile_method_output_path" in standard_save_block
    assert "build_profiling_output_path" not in standard_save_block
    assert "_prepare_standard_attention_output_dataframe" in standard_save_block
