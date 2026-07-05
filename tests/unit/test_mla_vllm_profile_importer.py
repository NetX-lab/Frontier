"""Unit tests for importing vLLM MLA timing rows into Frontier profiling schema."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import get_profiling_metric_names
from frontier.types import MeasurementType
from tests.unit.mla_h800_fixture import h800_mla_mixed_rows

REQUIRED_SCOPES = get_profiling_metric_names(LATENT_MLA_ATTENTION_FAMILY)


def _base_meta() -> dict[str, object]:
    return {
        "attention_backend": "FLASHINFER_MLA",
        "use_mla": True,
        "runtime_num_kv_heads": 1,
        "runtime_head_size": 576,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 192,
        "v_head_dim": 128,
        "block_size": 64,
        "kv_cache_dtype": "auto",
        "calculate_kv_scales": False,
        "attn_module_sliding_window": None,
        "alibi_slopes": None,
        "logits_soft_cap": None,
        "attn_type": "decoder",
        "max_seqlen_q": 1,
        "max_seqlen_k": 65,
        "num_actual_tokens": 1,
    }


def _sample_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, scope in enumerate(REQUIRED_SCOPES, 1):
        rows.append(
            {
                "batch_id": 7,
                "batch_size": 1,
                "batch_num_tokens": 1,
                "batch_num_prefill_tokens": 0,
                "batch_num_decode_tokens": 1,
                "batch_request_num_tokens": [1],
                "op_name": scope,
                "cuda_time_ms": float(idx) / 100.0,
                "count": 1,
                "meta": _base_meta(),
            }
        )
    rows.append(
        {
            "batch_id": 7,
            "op_name": "mlp_up_proj",
            "cuda_time_ms": 999.0,
            "count": 1,
            "meta": _base_meta(),
        }
    )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_vllm_mla_rows_convert_to_frontier_attention_profile_dataframe(
    tmp_path: Path,
) -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
        load_vllm_mla_profile_dataframe,
    )

    df = build_frontier_mla_profile_dataframe(
        _sample_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="cuda_event",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["model_name"] == "deepseek-ai/DeepSeek-V2-Lite"
    assert row["model_arch"] == "deepseek_v2"
    assert row["precision"] == "bf16"
    assert row["quant_signature"] == "none"
    assert row["measurement_type"] == "CUDA_EVENT"
    assert row["attention_backend"] == "FLASHINFER_MLA"
    assert row["n_q_head"] == 128
    assert row["n_kv_head"] == 1
    assert row["head_size"] == 576
    assert row["qk_head_dim"] == 192
    assert row["kv_lora_rank"] == 512
    assert row["block_size"] == 64
    assert row["num_tensor_parallel_workers"] == 1
    assert row["batch_size"] == 1
    assert row["batch_num_tokens"] == 1
    assert row["batch_num_prefill_tokens"] == 0
    assert row["batch_num_decode_tokens"] == 1
    assert row["max_seqlen_q"] == 1
    assert row["max_seqlen_k"] == 65
    assert row["num_actual_tokens"] == 1
    assert row["max_seq_len"] == 65
    assert bool(row["is_prefill"]) is False
    assert bool(row["is_mla_profile_import"]) is True

    for idx, scope in enumerate(REQUIRED_SCOPES, 1):
        expected_ms = float(idx) / 100.0
        assert row[f"time_stats.{scope}.min"] == pytest.approx(expected_ms)
        assert row[f"time_stats.{scope}.max"] == pytest.approx(expected_ms)
        assert row[f"time_stats.{scope}.mean"] == pytest.approx(expected_ms)
        assert row[f"time_stats.{scope}.median"] == pytest.approx(expected_ms)
        assert row[f"time_stats.{scope}.std"] == pytest.approx(0.0)
        assert row[f"time_stats.{scope}.count"] == 1

    input_log = tmp_path / "cuda_ops.jsonl"
    _write_jsonl(input_log, _sample_rows())
    loaded_df = load_vllm_mla_profile_dataframe(
        input_log,
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="cuda_event",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )
    pd.testing.assert_frame_equal(df, loaded_df)


@pytest.mark.parametrize(
    ("measurement_type", "expected_value"),
    [
        ("cuda_event", "CUDA_EVENT"),
        ("KERNEL_ONLY", "KERNEL_ONLY"),
        (MeasurementType.CUDA_EVENT, "CUDA_EVENT"),
    ],
)
def test_vllm_mla_profile_importer_normalizes_measurement_type_contract(
    measurement_type: str | MeasurementType,
    expected_value: str,
) -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    df = build_frontier_mla_profile_dataframe(
        _sample_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type=measurement_type,
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )

    assert df.loc[0, "measurement_type"] == expected_value


def test_vllm_mla_profile_importer_rejects_invalid_measurement_type() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    with pytest.raises(ValueError, match="Unsupported measurement_type"):
        build_frontier_mla_profile_dataframe(
            _sample_rows(),
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="timer_magic",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_importer_builds_numeric_groundtruth_comparison() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
        build_mla_profile_groundtruth_comparison,
    )

    rows = _sample_rows()
    df = build_frontier_mla_profile_dataframe(
        rows,
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )

    comparison = build_mla_profile_groundtruth_comparison(rows, df)

    assert tuple(comparison.columns) == (
        "scope",
        "vllm_cuda_time_ms",
        "frontier_profile_median_ms",
        "absolute_error_ms",
        "relative_error_pct",
        "vllm_sample_count",
    )
    assert tuple(comparison["scope"]) == REQUIRED_SCOPES
    for idx, row in enumerate(comparison.to_dict("records"), 1):
        expected_ms = float(idx) / 100.0
        assert row["vllm_cuda_time_ms"] == pytest.approx(expected_ms)
        assert row["frontier_profile_median_ms"] == pytest.approx(expected_ms)
        assert row["absolute_error_ms"] == pytest.approx(0.0)
        assert row["relative_error_pct"] == pytest.approx(0.0)
        assert row["vllm_sample_count"] == 1


def test_vllm_mla_groundtruth_comparison_reports_nonzero_error() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
        build_mla_profile_groundtruth_comparison,
    )

    rows = _sample_rows()
    df = build_frontier_mla_profile_dataframe(
        rows,
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )
    df.loc[0, "time_stats.attn_mla_decode.median"] = 0.055

    comparison = build_mla_profile_groundtruth_comparison(rows, df)
    decode_row = comparison[
        comparison["scope"] == "attn_mla_decode"
    ].iloc[0]

    assert decode_row["vllm_cuda_time_ms"] == pytest.approx(0.05)
    assert decode_row["frontier_profile_median_ms"] == pytest.approx(0.055)
    assert decode_row["absolute_error_ms"] == pytest.approx(0.005)
    assert decode_row["relative_error_pct"] == pytest.approx(10.0)


def test_vllm_mla_groundtruth_comparison_rejects_nonzero_frontier_for_zero_vllm() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
        build_mla_profile_groundtruth_comparison,
    )

    rows = _sample_rows()
    rows[4]["cuda_time_ms"] = 0.0
    df = build_frontier_mla_profile_dataframe(
        rows,
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )
    df.loc[0, "time_stats.attn_mla_decode.median"] = 0.001

    with pytest.raises(ValueError, match="zero vLLM median"):
        build_mla_profile_groundtruth_comparison(rows, df)


@pytest.mark.parametrize(
    ("profile_method", "expected_measurement_type", "expected_filename"),
    [
        ("cuda_event", "CUDA_EVENT", "attention.csv"),
        ("kernel_only", "KERNEL_ONLY", "attention_kernel_only.csv"),
        ("record_function", "KERNEL_ONLY", "attention_kernel_only.csv"),
    ],
)
def test_attention_profile_main_imports_vllm_mla_log_for_cuda_event_and_kernel_only(
    tmp_path: Path,
    profile_method: str,
    expected_measurement_type: str,
    expected_filename: str,
) -> None:
    from argparse import Namespace

    from frontier.profiling.attention.main import _run_vllm_mla_profile_import

    input_log = tmp_path / "cuda_ops.jsonl"
    _write_jsonl(input_log, _sample_rows())
    args = Namespace(
        vllm_mla_cuda_op_log=input_log,
        models=["deepseek-ai/DeepSeek-V2-Lite"],
        model_arch="deepseek_v2",
        precision="BF16",
        output_dir=str(tmp_path / "profiling"),
        device="h100",
        profile_method=profile_method,
        num_tensor_parallel_workers=[1],
        max_model_len=163840,
        attention_backend="FLASHINFER_MLA",
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
        enable_true_mixed=False,
        model_architecture_profile="generic",
    )

    output_file = _run_vllm_mla_profile_import(args)

    assert output_file.name == expected_filename
    df = pd.read_csv(output_file)
    assert df.loc[0, "measurement_type"] == expected_measurement_type
    assert df.loc[0, "time_stats.attn_mla_decode.median"] == pytest.approx(0.05)

    comparison_file = output_file.with_name(
        f"{output_file.stem}_vllm_mla_groundtruth_comparison.csv"
    )
    assert comparison_file.name == (
        f"{expected_filename.removesuffix('.csv')}"
        "_vllm_mla_groundtruth_comparison.csv"
    )
    assert comparison_file.exists()
    comparison = pd.read_csv(comparison_file)
    assert tuple(comparison["scope"]) == REQUIRED_SCOPES
    assert comparison["absolute_error_ms"].max() == pytest.approx(0.0)
    assert comparison["relative_error_pct"].max() == pytest.approx(0.0)
    assert comparison["vllm_sample_count"].sum() == len(REQUIRED_SCOPES)


def test_vllm_mla_profile_importer_fails_fast_on_missing_scope() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = [
        row
        for row in _sample_rows()
        if row["op_name"] != "attn_mla_decode_q_latent_proj"
    ]
    with pytest.raises(ValueError, match="Missing required MLA attention scopes"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_rejects_unexpected_query_head_count() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    rows[0]["meta"] = {**_base_meta(), "n_q_head": 64}
    with pytest.raises(ValueError, match="n_q_head"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_rejects_dense_metadata() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    rows[0]["meta"] = {**_base_meta(), "runtime_num_kv_heads": 128}
    with pytest.raises(ValueError, match="runtime_num_kv_heads"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_rejects_inconsistent_runtime_head_size() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    rows[0]["meta"] = {**_base_meta(), "runtime_head_size": 575}
    with pytest.raises(ValueError, match="runtime_head_size"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_rejects_unsupported_block_size() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    rows[0]["meta"] = {**_base_meta(), "block_size": 16}
    with pytest.raises(ValueError, match="Unsupported FlashInfer MLA block_size"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_accepts_formula_consistent_runtime_variant() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    variant_meta = {
        **_base_meta(),
        "runtime_head_size": 320,
        "kv_lora_rank": 256,
        "qk_nope_head_dim": 96,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 160,
    }
    for row in rows:
        if row["op_name"] in REQUIRED_SCOPES:
            row["meta"] = variant_meta

    df = build_frontier_mla_profile_dataframe(
        rows,
        model_name="deepseek-ai/DeepSeek-V2-Lite-Variant",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="cuda_event",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )

    assert df.loc[0, "head_size"] == 320
    assert df.loc[0, "kv_lora_rank"] == 256
    assert df.loc[0, "qk_rope_head_dim"] == 64
    assert df.loc[0, "qk_head_dim"] == 160


def test_vllm_mla_profile_importer_rejects_mixed_scope_metadata() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    rows[1]["meta"] = {**_base_meta(), "block_size": 32}
    with pytest.raises(ValueError, match="Inconsistent runtime meta"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_rejects_mixed_dynamic_profile_shape() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    rows[1]["batch_size"] = 2
    rows[1]["meta"] = {**_base_meta(), "max_seqlen_k": 129}
    with pytest.raises(ValueError, match="Inconsistent dynamic profile shape"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_rejects_token_count_metadata_mismatch() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    for row in rows:
        if row.get("op_name") in REQUIRED_SCOPES:
            row["batch_num_tokens"] = 2
            row["batch_request_num_tokens"] = [2]
            row["meta"] = {**_base_meta(), "max_seqlen_q": 2}

    with pytest.raises(ValueError, match="num_actual_tokens.*batch_num_tokens"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_rejects_negative_timings() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    rows[0]["cuda_time_ms"] = -0.1
    with pytest.raises(ValueError, match="Negative CUDA timing"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_rejects_missing_cuda_timing() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
    )

    rows = _sample_rows()
    del rows[0]["cuda_time_ms"]
    with pytest.raises(ValueError, match="Missing cuda_time_ms"):
        build_frontier_mla_profile_dataframe(
            rows,
            model_name="deepseek-ai/DeepSeek-V2-Lite",
            model_arch="deepseek_v2",
            precision="bf16",
            quant_signature="none",
            measurement_type="cuda_event",
            num_tensor_parallel_workers=1,
            max_model_len=163840,
        )


def test_vllm_mla_profile_importer_derives_required_scopes_from_family_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import frontier.profiling.attention.vllm_mla_profile_importer as importer

    original_scopes = get_profiling_metric_names(LATENT_MLA_ATTENTION_FAMILY)
    derived_scopes = (
        original_scopes[0],
        original_scopes[1],
        original_scopes[2],
        original_scopes[3],
        "catalog_attn_mla_decode",
        original_scopes[5],
    )

    rows = _sample_rows()
    for row in rows:
        if row.get("op_name") == original_scopes[4]:
            row["op_name"] = "catalog_attn_mla_decode"

    catalog_family = replace(
        LATENT_MLA_ATTENTION_FAMILY,
        operators=tuple(
            replace(operator, name="catalog_attn_mla_decode")
            if operator.name == original_scopes[4]
            else operator
            for operator in LATENT_MLA_ATTENTION_FAMILY.operators
        ),
    )
    monkeypatch.setattr(importer, "LATENT_MLA_ATTENTION_FAMILY", catalog_family)

    df = importer.build_frontier_mla_profile_dataframe(
        rows,
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="cuda_event",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )

    row = df.iloc[0]
    assert row["time_stats.catalog_attn_mla_decode.median"] == pytest.approx(0.05)
    assert "time_stats.attn_mla_decode.median" not in df.columns


def test_attention_profile_main_imports_vllm_mla_log_to_canonical_csv(
    tmp_path: Path,
) -> None:
    from argparse import Namespace

    from frontier.profiling.attention.main import _run_vllm_mla_profile_import

    input_log = tmp_path / "cuda_ops.jsonl"
    _write_jsonl(input_log, _sample_rows())
    args = Namespace(
        vllm_mla_cuda_op_log=input_log,
        models=["deepseek-ai/DeepSeek-V2-Lite"],
        model_arch="deepseek_v2",
        precision="BF16",
        output_dir=str(tmp_path / "profiling"),
        device="h100",
        profile_method="cuda_event",
        num_tensor_parallel_workers=[1],
        max_model_len=163840,
        attention_backend="FLASHINFER_MLA",
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
        enable_true_mixed=False,
        model_architecture_profile="generic",
    )

    output_file = _run_vllm_mla_profile_import(args)

    assert output_file == (
        tmp_path
        / "profiling"
        / "compute"
        / "h100"
        / "deepseek-ai"
        / "DeepSeek-V2-Lite"
        / "attention.csv"
    )
    df = pd.read_csv(output_file)
    assert len(df) == 1
    assert df.loc[0, "model_name"] == "deepseek-ai/DeepSeek-V2-Lite"
    assert df.loc[0, "model_arch"] == "deepseek_v2"
    assert df.loc[0, "profiling_precision"] == "bf16"
    assert df.loc[0, "measurement_type"] == "CUDA_EVENT"
    assert df.loc[0, "attention_backend"] == "FLASHINFER_MLA"
    assert df.loc[0, "time_stats.attn_mla_decode.median"] == pytest.approx(0.05)


def test_vllm_mla_import_mode_rejects_ambiguous_cli_shape(tmp_path: Path) -> None:
    from argparse import Namespace

    from frontier.profiling.attention.main import _run_vllm_mla_profile_import

    input_log = tmp_path / "cuda_ops.jsonl"
    _write_jsonl(input_log, _sample_rows())
    args = Namespace(
        vllm_mla_cuda_op_log=input_log,
        models=["model-a", "model-b"],
        model_arch="deepseek_v2",
        precision="BF16",
        output_dir=str(tmp_path / "profiling"),
        device="h100",
        profile_method="cuda_event",
        num_tensor_parallel_workers=[1],
        max_model_len=163840,
        attention_backend="FLASHINFER_MLA",
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
        enable_true_mixed=False,
    )

    with pytest.raises(ValueError, match="exactly one model"):
        _run_vllm_mla_profile_import(args)


def test_vllm_mla_import_mode_rejects_multiple_tensor_parallel_sizes(
    tmp_path: Path,
) -> None:
    from argparse import Namespace

    from frontier.profiling.attention.main import _run_vllm_mla_profile_import

    input_log = tmp_path / "cuda_ops.jsonl"
    _write_jsonl(input_log, _sample_rows())
    args = Namespace(
        vllm_mla_cuda_op_log=input_log,
        models=["deepseek-ai/DeepSeek-V2-Lite"],
        model_arch="deepseek_v2",
        precision="BF16",
        output_dir=str(tmp_path / "profiling"),
        device="h100",
        profile_method="cuda_event",
        num_tensor_parallel_workers=[1, 2],
        max_model_len=163840,
        attention_backend="FLASHINFER_MLA",
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
        enable_true_mixed=False,
    )

    with pytest.raises(ValueError, match="exactly one tensor parallel size"):
        _run_vllm_mla_profile_import(args)


@pytest.mark.parametrize("attention_backend", ["FLASHINFER", "NO_OP"])
def test_vllm_mla_import_mode_requires_flashinfer_mla_backend(
    tmp_path: Path, attention_backend: str
) -> None:
    from argparse import Namespace

    from frontier.profiling.attention.main import _run_vllm_mla_profile_import

    input_log = tmp_path / "cuda_ops.jsonl"
    _write_jsonl(input_log, _sample_rows())
    args = Namespace(
        vllm_mla_cuda_op_log=input_log,
        models=["deepseek-ai/DeepSeek-V2-Lite"],
        model_arch="deepseek_v2",
        precision="BF16",
        output_dir=str(tmp_path / "profiling"),
        device="h100",
        profile_method="cuda_event",
        num_tensor_parallel_workers=[1],
        max_model_len=163840,
        attention_backend=attention_backend,
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
        enable_true_mixed=False,
    )

    with pytest.raises(ValueError, match="requires --attention_backend FLASHINFER_MLA"):
        _run_vllm_mla_profile_import(args)


@pytest.mark.parametrize(
    ("enable_mixed_prefill", "enable_true_mixed"),
    [(True, False), (False, True)],
)
def test_vllm_mla_import_mode_rejects_mixed_profiling_modes(
    tmp_path: Path, enable_mixed_prefill: bool, enable_true_mixed: bool
) -> None:
    from argparse import Namespace

    from frontier.profiling.attention.main import _run_vllm_mla_profile_import

    input_log = tmp_path / "cuda_ops.jsonl"
    _write_jsonl(input_log, _sample_rows())
    args = Namespace(
        vllm_mla_cuda_op_log=input_log,
        models=["deepseek-ai/DeepSeek-V2-Lite"],
        model_arch="deepseek_v2",
        precision="BF16",
        output_dir=str(tmp_path / "profiling"),
        device="h100",
        profile_method="cuda_event",
        num_tensor_parallel_workers=[1],
        max_model_len=163840,
        attention_backend="FLASHINFER_MLA",
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=enable_mixed_prefill,
        enable_true_mixed=enable_true_mixed,
    )

    with pytest.raises(ValueError, match="cannot be combined with mixed profiling modes"):
        _run_vllm_mla_profile_import(args)


def test_flashinfer_mla_backend_requires_explicit_vllm_import_log() -> None:
    from argparse import Namespace

    from frontier.profiling.attention.main import _validate_cli_conflicts

    args = Namespace(
        vllm_mla_cuda_op_log=None,
        models=["deepseek-ai/DeepSeek-V2-Lite"],
        num_tensor_parallel_workers=[1],
        attention_backend="FLASHINFER_MLA",
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
        enable_true_mixed=False,
        decode_kv_cache_size_list=None,
    )

    with pytest.raises(ValueError, match="requires --vllm_mla_cuda_op_log"):
        _validate_cli_conflicts(args)

def test_vllm_mla_importer_preserves_h800_mixed_rows_as_sparse_profile_rows() -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
        build_mla_profile_groundtruth_comparison,
    )

    rows = h800_mla_mixed_rows()
    df = build_frontier_mla_profile_dataframe(
        rows,
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="KERNEL_ONLY",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )

    assert len(df) == 5
    non_nan_counts = {
        scope: int(df[f"time_stats.{scope}.median"].notna().sum())
        for scope in REQUIRED_SCOPES
    }
    assert non_nan_counts == {
        "attn_mla_kv_cache_save": 3,
        "attn_mla_prefill_kv_up_proj": 2,
        "attn_mla_prefill": 2,
        "attn_mla_decode_q_latent_proj": 2,
        "attn_mla_decode": 2,
        "attn_mla_v_up_proj": 2,
    }

    mixed_decode_row = df[
        (df["batch_size"] == 2)
        & (df["batch_num_tokens"] == 65)
        & (df["max_seqlen_q"] == 1)
        & (df["num_actual_tokens"] == 1)
    ].iloc[0]
    assert mixed_decode_row["time_stats.attn_mla_decode.median"] == pytest.approx(
        0.069984
    )
    assert pd.isna(mixed_decode_row["time_stats.attn_mla_prefill.median"])

    comparison = build_mla_profile_groundtruth_comparison(rows, df)
    assert len(comparison) == 13
    assert comparison["vllm_sample_count"].sum() == 13
    assert comparison["absolute_error_ms"].max() == pytest.approx(0.0)
    assert comparison["relative_error_pct"].max() == pytest.approx(0.0)
    assert {
        "profile_row_index",
        "batch_size",
        "batch_num_tokens",
        "batch_num_prefill_tokens",
        "batch_num_decode_tokens",
        "max_seqlen_q",
        "max_seqlen_k",
        "num_actual_tokens",
    }.issubset(comparison.columns)


def test_vllm_mla_import_mode_requires_model_architecture_profile_for_unknown_model(
    tmp_path: Path,
) -> None:
    from argparse import Namespace

    from frontier.profiling.attention.main import _run_vllm_mla_profile_import

    input_log = tmp_path / "cuda_ops.jsonl"
    _write_jsonl(input_log, _sample_rows())
    args = Namespace(
        vllm_mla_cuda_op_log=input_log,
        models=["deepseek-ai/DeepSeek-V2-Lite"],
        model_arch="deepseek_v2",
        precision="BF16",
        output_dir=str(tmp_path / "profiling"),
        device="h100",
        profile_method="cuda_event",
        num_tensor_parallel_workers=[1],
        max_model_len=163840,
        attention_backend="FLASHINFER_MLA",
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
        enable_true_mixed=False,
    )

    with pytest.raises(ValueError, match="model_architecture_profile"):
        _run_vllm_mla_profile_import(args)
