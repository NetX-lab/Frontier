from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import tests.analysis.mha_phi3.build_mha_stage2_profile_model as builder_module

from tests.analysis.mha_phi3.build_mha_stage2_profile_model import (
    MhaStage2Config,
    build_error_matrix,
    build_frontier_attention_rows,
    compute_mha_memory_metrics,
    load_cuda_op_rows,
    summarize_attention_scopes,
    write_stage2_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER = (
    PROJECT_ROOT
    / "tests"
    / "analysis"
    / "mha_phi3"
    / "build_mha_stage2_profile_model.py"
)
LIVE_CUDA_OPS = Path("/tmp/frontier_mha_phi3_flash_attn_live_probe/cuda_ops.jsonl")


def test_mha_required_scopes_are_declared_from_shared_dense_attention_mapping() -> None:
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
    assert "DENSE_ATTENTION_FAMILY" in imports.get(
        "frontier.attention.families", set()
    )
    assert "get_profiling_metric_names" in imports.get(
        "frontier.attention.profiling_mapping", set()
    )

    from frontier.attention.families import DENSE_ATTENTION_FAMILY
    from frontier.attention.profiling_mapping import get_profiling_metric_names

    assert builder_module.REQUIRED_SCOPES == get_profiling_metric_names(
        DENSE_ATTENTION_FAMILY
    )


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "batch_id": 0,
            "batch_size": 1,
            "batch_num_tokens": 64,
            "batch_num_prefill_tokens": 64,
            "batch_num_decode_tokens": 0,
            "batch_request_num_tokens": [64],
            "op_name": "attn_kv_cache_save",
            "cuda_time_ms": 0.001984,
            "count": 1,
            "meta": {
                "attention_backend": "FLASH_ATTN_VLLM_V1",
                "head_dim": 96,
                "num_q_heads": 32,
                "num_kv_heads": 32,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": 2047,
                "flash_attention_window_size": [2046, 0],
                "kv_cache_spec_type": "SlidingWindowSpec",
                "max_seqlen_q": 64,
                "max_seqlen_k": 64,
                "num_actual_tokens": 64,
            },
        },
        {
            "batch_id": 0,
            "batch_size": 1,
            "batch_num_tokens": 64,
            "batch_num_prefill_tokens": 64,
            "batch_num_decode_tokens": 0,
            "batch_request_num_tokens": [64],
            "op_name": "attn_prefill",
            "cuda_time_ms": 0.005952,
            "count": 1,
            "meta": {
                "attention_backend": "FLASH_ATTN_VLLM_V1",
                "head_dim": 96,
                "num_q_heads": 32,
                "num_kv_heads": 32,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": 2047,
                "flash_attention_window_size": [2046, 0],
                "kv_cache_spec_type": "SlidingWindowSpec",
                "max_seqlen_q": 64,
                "max_seqlen_k": 64,
                "num_actual_tokens": 64,
            },
        },
        {
            "batch_id": 2,
            "batch_size": 1,
            "batch_num_tokens": 1,
            "batch_num_prefill_tokens": 0,
            "batch_num_decode_tokens": 1,
            "batch_request_num_tokens": [1],
            "op_name": "attn_kv_cache_save",
            "cuda_time_ms": 0.001760,
            "count": 1,
            "meta": {
                "attention_backend": "FLASH_ATTN_VLLM_V1",
                "head_dim": 96,
                "num_q_heads": 32,
                "num_kv_heads": 32,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": 2047,
                "flash_attention_window_size": [2046, 0],
                "kv_cache_spec_type": "SlidingWindowSpec",
                "max_seqlen_q": 1,
                "max_seqlen_k": 65,
                "num_actual_tokens": 1,
            },
        },
        {
            "batch_id": 2,
            "batch_size": 1,
            "batch_num_tokens": 1,
            "batch_num_prefill_tokens": 0,
            "batch_num_decode_tokens": 1,
            "batch_request_num_tokens": [1],
            "op_name": "attn_decode",
            "cuda_time_ms": 0.005984,
            "count": 1,
            "meta": {
                "attention_backend": "FLASH_ATTN_VLLM_V1",
                "head_dim": 96,
                "num_q_heads": 32,
                "num_kv_heads": 32,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": 2047,
                "flash_attention_window_size": [2046, 0],
                "kv_cache_spec_type": "SlidingWindowSpec",
                "max_seqlen_q": 1,
                "max_seqlen_k": 65,
                "num_actual_tokens": 1,
            },
        },
    ]


def test_stage2_loader_rejects_empty_cuda_op_log(tmp_path: Path) -> None:
    empty_log = tmp_path / "cuda_ops.jsonl"
    empty_log.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_cuda_op_rows(empty_log)


def test_stage2_scope_summary_preserves_flashattention_runtime_metadata() -> None:
    summaries = summarize_attention_scopes(_sample_rows())

    assert summaries["attn_kv_cache_save"].rows == 2
    assert summaries["attn_kv_cache_save"].total_cuda_time_ms == pytest.approx(
        0.003744
    )
    assert summaries["attn_kv_cache_save"].median_cuda_time_ms == pytest.approx(
        0.001872
    )
    assert summaries["attn_prefill"].rows == 1
    assert summaries["attn_decode"].rows == 1

    for scope in ["attn_kv_cache_save", "attn_prefill", "attn_decode"]:
        summary = summaries[scope]
        assert summary.backend == "FLASH_ATTN_VLLM_V1"
        assert summary.head_dim == 96
        assert summary.num_q_heads == 32
        assert summary.num_kv_heads == 32
        assert summary.sliding_window == 2047
        assert summary.kv_cache_spec_type == "SlidingWindowSpec"


def test_stage2_frontier_attention_rows_match_prediction_contract() -> None:
    config = MhaStage2Config()
    rows = build_frontier_attention_rows(_sample_rows(), config)

    prefill_rows = [row for row in rows if row["is_prefill"] is True]
    decode_rows = [row for row in rows if row["is_prefill"] is False]
    assert len(prefill_rows) == 1
    assert len(decode_rows) == 1

    prefill = prefill_rows[0]
    assert prefill["n_embd"] == 3072
    assert prefill["n_q_head"] == 32
    assert prefill["n_kv_head"] == 32
    assert prefill["block_size"] == 16
    assert prefill["num_tensor_parallel_workers"] == 1
    assert prefill["measurement_type"] == "cuda_event"
    assert prefill["attention_backend"] == "FLASH_ATTN_VLLM_V1"
    assert prefill["prefill_chunk_size"] == 64
    assert prefill["kv_cache_size"] == 0
    assert prefill["prefill_chunk_size_squared"] == 4096
    assert prefill["time_stats.attn_prefill.median"] == pytest.approx(0.005952)
    assert prefill["time_stats.attn_decode.median"] == 0.0
    assert prefill["time_stats.attn_kv_cache_save.median"] == pytest.approx(0.001984)

    decode = decode_rows[0]
    assert decode["prefill_chunk_size"] == 0
    assert decode["batch_size"] == 1
    assert decode["kv_cache_size"] == 65
    assert decode["time_stats.attn_decode.median"] == pytest.approx(0.005984)
    assert decode["time_stats.attn_prefill.median"] == 0.0
    assert decode["time_stats.attn_kv_cache_save.median"] == pytest.approx(0.001760)


def test_stage2_memory_metrics_match_dense_page_and_vllm_sliding_window_budget() -> None:
    config = MhaStage2Config()
    memory = compute_mha_memory_metrics(config=config, observed_max_context_tokens=65)

    assert memory["tp1_page_bytes_per_layer"] == 196_608
    assert memory["tp8_page_bytes_per_layer"] == 24_576
    assert memory["observed_context_tokens"] == 65
    assert memory["observed_context_blocks"] == 5
    assert memory["observed_context_bytes_per_layer_tp1"] == 983_040
    assert memory["sliding_window_budget_tokens"] == 2_302
    assert memory["sliding_window_budget_blocks"] == 145
    assert memory["vllm_sliding_window_budget_bytes_per_layer_tp1"] == 28_508_160


def test_stage2_error_matrix_converges_for_transformed_live_rows() -> None:
    config = MhaStage2Config()
    rows = build_frontier_attention_rows(_sample_rows(), config)
    matrix = build_error_matrix(_sample_rows(), rows, config)

    for row in matrix["latency"]:
        assert row["actual_ms"] > 0.0
        assert row["predicted_ms"] == pytest.approx(row["actual_ms"])
        assert row["relative_error_pct"] == pytest.approx(0.0)
        assert row["passes_5pct"] is True

    for row in matrix["memory"]:
        assert row["actual_bytes"] > 0
        assert row["predicted_bytes"] == row["actual_bytes"]
        assert row["relative_error_pct"] == pytest.approx(0.0)
        assert row["passes_5pct"] is True


def test_stage2_cli_writes_reproducible_artifacts(tmp_path: Path) -> None:
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

    assert "MHA Stage 2 profile/model artifacts written" in completed.stdout
    assert (output_dir / "attention.csv").exists()
    assert (output_dir / "error_matrix.json").exists()
    assert (output_dir / "error_matrix.md").exists()
    summary = json.loads((output_dir / "error_matrix.json").read_text(encoding="utf-8"))
    assert summary["environment"]["python_bin"] == sys.executable
    assert summary["config"]["flashinfer_python_expected_version"] == "0.3.1.post1"


@pytest.mark.skipif(
    not LIVE_CUDA_OPS.exists(),
    reason="MHA CR-005 live-probe artifact is not present under /tmp",
)
def test_stage2_builder_accepts_cr005_live_probe_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "live"
    write_stage2_outputs(LIVE_CUDA_OPS, output_dir, MhaStage2Config())

    summary = json.loads((output_dir / "error_matrix.json").read_text(encoding="utf-8"))
    latency_by_scope = {row["metric"]: row for row in summary["latency"]}
    assert latency_by_scope["attn_kv_cache_save"]["actual_ms"] == pytest.approx(
        0.001888
    )
    assert latency_by_scope["attn_prefill"]["actual_ms"] == pytest.approx(0.005936)
    assert latency_by_scope["attn_decode"]["actual_ms"] == pytest.approx(0.005984)
    assert all(row["passes_5pct"] for row in summary["latency"])
    assert all(row["passes_5pct"] for row in summary["memory"])
