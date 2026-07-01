#!/usr/bin/env python3
"""Unit tests for DeepSeek-V2 MLA Stage 2 profile/model artifacts."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
def _installed_flashinfer_python_version() -> str | None:
    try:
        return version("flashinfer-python")
    except PackageNotFoundError:
        return None


BUILDER = (
    PROJECT_ROOT
    / "tests"
    / "analysis"
    / "mla_deepseek_v2"
    / "build_mla_stage2_profile_model.py"
)


def _load_builder_module():
    try:
        return importlib.import_module(
            "tests.analysis.mla_deepseek_v2.build_mla_stage2_profile_model"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing MLA Stage 2 builder module: {exc}")


def _sample_rows() -> list[dict[str, object]]:
    base_meta = {
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
    }
    return [
        {
            "batch_id": 0,
            "batch_size": 1,
            "batch_num_tokens": 64,
            "batch_num_prefill_tokens": 64,
            "batch_num_decode_tokens": 0,
            "batch_request_num_tokens": [64],
            "op_name": "attn_mla_kv_cache_save",
            "cuda_time_ms": 0.0016,
            "count": 1,
            "meta": {**base_meta, "max_seqlen_q": 64, "max_seqlen_k": 64, "num_actual_tokens": 64},
        },
        {
            "batch_id": 0,
            "batch_size": 1,
            "batch_num_tokens": 64,
            "batch_num_prefill_tokens": 64,
            "batch_num_decode_tokens": 0,
            "batch_request_num_tokens": [64],
            "op_name": "attn_mla_prefill_kv_up_proj",
            "cuda_time_ms": 0.028416,
            "count": 1,
            "meta": {**base_meta, "max_seqlen_q": 64, "max_seqlen_k": 64, "num_actual_tokens": 64},
        },
        {
            "batch_id": 0,
            "batch_size": 1,
            "batch_num_tokens": 64,
            "batch_num_prefill_tokens": 64,
            "batch_num_decode_tokens": 0,
            "batch_request_num_tokens": [64],
            "op_name": "attn_mla_prefill",
            "cuda_time_ms": 0.011552,
            "count": 1,
            "meta": {**base_meta, "max_seqlen_q": 64, "max_seqlen_k": 64, "num_actual_tokens": 64},
        },
        {
            "batch_id": 1,
            "batch_size": 1,
            "batch_num_tokens": 1,
            "batch_num_prefill_tokens": 0,
            "batch_num_decode_tokens": 1,
            "batch_request_num_tokens": [1],
            "op_name": "attn_mla_kv_cache_save",
            "cuda_time_ms": 0.00144,
            "count": 1,
            "meta": {**base_meta, "max_seqlen_q": 1, "max_seqlen_k": 65, "num_actual_tokens": 1},
        },
        {
            "batch_id": 1,
            "batch_size": 1,
            "batch_num_tokens": 1,
            "batch_num_prefill_tokens": 0,
            "batch_num_decode_tokens": 1,
            "batch_request_num_tokens": [1],
            "op_name": "attn_mla_decode_q_latent_proj",
            "cuda_time_ms": 0.01472,
            "count": 1,
            "meta": {**base_meta, "max_seqlen_q": 1, "max_seqlen_k": 65, "num_actual_tokens": 1},
        },
        {
            "batch_id": 1,
            "batch_size": 1,
            "batch_num_tokens": 1,
            "batch_num_prefill_tokens": 0,
            "batch_num_decode_tokens": 1,
            "batch_request_num_tokens": [1],
            "op_name": "attn_mla_decode",
            "cuda_time_ms": 0.050079,
            "count": 1,
            "meta": {**base_meta, "max_seqlen_q": 1, "max_seqlen_k": 65, "num_actual_tokens": 1},
        },
        {
            "batch_id": 1,
            "batch_size": 1,
            "batch_num_tokens": 1,
            "batch_num_prefill_tokens": 0,
            "batch_num_decode_tokens": 1,
            "batch_request_num_tokens": [1],
            "op_name": "attn_mla_v_up_proj",
            "cuda_time_ms": 0.015648,
            "count": 1,
            "meta": {**base_meta, "max_seqlen_q": 1, "max_seqlen_k": 65, "num_actual_tokens": 1},
        },
    ]


def test_mla_required_scopes_are_declared_from_shared_attention_mapping() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.setdefault(node.module, set()).update(
                alias.name for alias in node.names
            )

    literal_required_scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SCOPES"
            for target in node.targets
        )
        and isinstance(node.value, ast.Tuple)
    ]
    assert literal_required_scopes == []
    assert "LATENT_MLA_ATTENTION_FAMILY" in imports.get(
        "frontier.attention.families", set()
    )
    assert "get_profiling_metric_names" in imports.get(
        "frontier.attention.profiling_mapping", set()
    )

    from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
    from frontier.attention.profiling_mapping import get_profiling_metric_names

    builder = _load_builder_module()
    assert builder.REQUIRED_SCOPES == get_profiling_metric_names(
        LATENT_MLA_ATTENTION_FAMILY
    )


def test_mla_scope_summary_requires_flashinfer_mla_runtime_metadata() -> None:
    builder = _load_builder_module()
    summaries = builder.summarize_mla_scopes(_sample_rows())

    for scope in builder.REQUIRED_SCOPES:
        summary = summaries[scope]
        assert summary.backend == "FLASHINFER_MLA"
        assert summary.use_mla is True
        assert summary.runtime_num_kv_heads == 1
        assert summary.runtime_head_size == 576
        assert summary.kv_lora_rank == 512
        assert summary.qk_head_dim == 192
        assert summary.block_size == 64
        assert summary.attn_type == "decoder"


def test_mla_memory_metrics_use_latent_page_formula_and_reject_dense_factor() -> None:
    builder = _load_builder_module()

    config64 = builder.DeepSeekV2MlaStage2Config(block_size=64)
    memory64 = builder.compute_mla_memory_metrics(
        config=config64,
        observed_max_context_tokens=65,
    )
    assert memory64["native_tp1_page_bytes_per_layer"] == 73_728
    assert memory64["observed_context_tokens"] == 65
    assert memory64["observed_context_blocks"] == 2
    assert memory64["observed_context_bytes_per_layer_tp1"] == 147_456
    assert memory64["observed_context_bytes_per_worker_tp1"] == 8_847_360

    config32 = builder.DeepSeekV2MlaStage2Config(block_size=32)
    memory32 = builder.compute_mla_memory_metrics(
        config=config32,
        observed_max_context_tokens=65,
    )
    assert memory32["native_tp1_page_bytes_per_layer"] == 36_864
    assert memory32["observed_context_blocks"] == 3
    assert memory32["observed_context_bytes_per_layer_tp1"] == 110_592

    with pytest.raises(ValueError, match="dense K/V factor"):
        builder.compute_mla_memory_metrics(
            config=config64,
            observed_max_context_tokens=65,
            dense_kv_factor=2,
        )


def test_mla_error_matrix_converges_for_transformed_flashinfer_mla_rows() -> None:
    builder = _load_builder_module()
    config = builder.DeepSeekV2MlaStage2Config(block_size=64)
    matrix = builder.build_error_matrix(_sample_rows(), config)

    acceptance_ops = {
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    }
    assert {row["op_name"] for row in matrix["latency"]} == acceptance_ops
    for row in matrix["latency"]:
        assert row["actual_ms"] > 0.0
        assert row["predicted_ms"] == pytest.approx(row["actual_ms"])
        assert row["relative_error_pct"] == pytest.approx(0.0)
        assert row["passes_5pct"] is True
        assert row["backend"] == "FLASHINFER_MLA"

    for row in matrix["memory"]:
        assert row["actual_bytes"] > 0
        assert row["predicted_bytes"] == row["actual_bytes"]
        assert row["relative_error_pct"] == pytest.approx(0.0)
        assert row["passes_5pct"] is True


def test_mla_stage2_cli_writes_reproducible_artifacts(tmp_path: Path) -> None:
    input_log = tmp_path / "cuda_ops.jsonl"
    input_log.write_text(
        "\n".join(json.dumps(row) for row in _sample_rows()) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage2"

    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--cuda-op-log",
            str(input_log),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "MLA Stage 2 profile/model artifacts written" in completed.stdout
    assert (output_dir / "mla_attention.csv").exists()
    assert (output_dir / "error_matrix.json").exists()
    assert (output_dir / "error_matrix.md").exists()
    summary = json.loads((output_dir / "error_matrix.json").read_text(encoding="utf-8"))
    assert summary["environment"]["python_bin"] == sys.executable
    assert summary["environment"]["flashinfer_python_version"] == _installed_flashinfer_python_version()
    assert summary["environment"]["flashinfer_python_expected_version"] == "0.3.1.post1"
    assert summary["config"]["attention_backend"] == "FLASHINFER_MLA"
    assert summary["config"]["use_mla"] is True
