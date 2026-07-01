"""Unit tests for true mixed orchestration and combined output schema contract."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import get_required_profiling_columns
from frontier.profiling.attention import main as attention_main


def _cli_args(**overrides: object) -> SimpleNamespace:
    defaults = {
        "profile_only_prefill": False,
        "profile_only_decode": False,
        "decode_kv_cache_size_list": None,
        "enable_true_mixed": False,
        "attention_backend": "FLASHINFER",
        "vllm_mla_cuda_op_log": None,
        "models": ["meta-llama/Llama-2-7b-hf"],
        "num_tensor_parallel_workers": [1],
        "enable_mixed_prefill": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _dense_standard_row(measurement_type: str) -> dict[str, object]:
    row = {column: 1 for column in get_required_profiling_columns(DENSE_ATTENTION_FAMILY)}
    row.update(
        {
            "measurement_type": measurement_type,
            "attention_backend": "FLASHINFER",
            "is_prefill": True,
        }
    )
    return row


def test_validate_cli_conflicts_rejects_invalid_combinations() -> None:
    with pytest.raises(ValueError, match="cannot both be enabled"):
        attention_main._validate_cli_conflicts(  # pylint: disable=protected-access
            _cli_args(
                profile_only_prefill=True,
                profile_only_decode=True,
            )
        )

    with pytest.raises(ValueError, match="--enable_true_mixed requires profiling both"):
        attention_main._validate_cli_conflicts(  # pylint: disable=protected-access
            _cli_args(
                profile_only_prefill=True,
                enable_true_mixed=True,
            )
        )

    with pytest.raises(ValueError, match="--enable_true_mixed requires profiling both"):
        attention_main._validate_cli_conflicts(  # pylint: disable=protected-access
            _cli_args(
                profile_only_decode=True,
                enable_true_mixed=True,
            )
        )


def test_validate_cli_conflicts_accepts_valid_configuration() -> None:
    attention_main._validate_cli_conflicts(  # pylint: disable=protected-access
        _cli_args(
            enable_true_mixed=True,
        )
    )


def test_build_attention_combined_df_normalizes_partition_markers() -> None:
    standard_df = pd.DataFrame(
        [
            {
                "batch_size": 1,
                "prefill_chunk_size": 1024,
                "is_prefill": True,
            }
        ]
    )
    mixed_prefill_df = pd.DataFrame(
        [
            {
                "batch_size": 2,
                "total_tokens": 2048,
                "is_prefill": True,
            }
        ]
    )
    true_mixed_df = pd.DataFrame(
        [
            {
                "batch_size": 3,
                "num_prefill_seqs": 1,
                "num_decode_seqs": 2,
                "is_prefill": True,
                "total_prefill_tokens": 1024,
                "is_true_mixed_batch": True,
            }
        ]
    )

    combined = attention_main._build_attention_combined_df(  # pylint: disable=protected-access
        standard_df,
        mixed_prefill_df,
        true_mixed_df,
    )

    assert list(combined["is_true_mixed_batch"]) == [False, False, True]
    assert list(combined["is_mixed_batch"]) == [False, True, False]
    assert combined.loc[2, "total_prefill_tokens"] == 1024


def test_build_attention_combined_df_returns_empty_when_no_partitions() -> None:
    combined = attention_main._build_attention_combined_df(  # pylint: disable=protected-access
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert combined.empty


@pytest.mark.parametrize("measurement_type", ["CUDA_EVENT", "KERNEL_ONLY"])
def test_prepare_standard_attention_output_validates_dense_schema(
    measurement_type: str,
) -> None:
    df = pd.DataFrame([_dense_standard_row(measurement_type)]).drop(
        columns=["measurement_type"]
    )

    output = attention_main._prepare_standard_attention_output_dataframe(  # pylint: disable=protected-access
        df,
        precision_str="bf16",
        model_arch="llama",
        quant_signature="none",
        measurement_type=measurement_type,
    )

    assert output.loc[0, "measurement_type"] == measurement_type
    assert output.loc[0, "profiling_precision"] == "bf16"
    assert output.loc[0, "model_arch"] == "llama"
    assert output.loc[0, "quant_signature"] == "none"


def test_prepare_standard_attention_output_rejects_missing_dense_schema_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "attention_backend": "FLASHINFER",
                "n_q_head": 32,
                "n_kv_head": 8,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing required attention profiling columns",
    ):
        attention_main._prepare_standard_attention_output_dataframe(  # pylint: disable=protected-access
            df,
            precision_str="bf16",
            model_arch="llama",
            quant_signature="none",
            measurement_type="CUDA_EVENT",
        )
