from __future__ import annotations

import ast
import json
import importlib
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER = (
    PROJECT_ROOT
    / "tests"
    / "analysis"
    / "mqa_falcon"
    / "build_mqa_stage2_profile_model.py"
)
LIVE_CUDA_OPS = Path("/tmp/frontier_mqa_falcon_flashinfer_live_probe/cuda_ops.jsonl")


def _load_builder_module():
    try:
        return importlib.import_module(
            "tests.analysis.mqa_falcon.build_mqa_stage2_profile_model"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing MQA Stage 2 builder module: {exc}")


def test_mqa_required_scopes_are_declared_from_shared_dense_attention_mapping() -> None:
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

    builder = _load_builder_module()
    assert builder.REQUIRED_SCOPES == get_profiling_metric_names(
        DENSE_ATTENTION_FAMILY
    )


def _sample_rows() -> list[dict[str, object]]:
    base_meta = {
        "attention_backend": "FLASHINFER_VLLM_V1",
        "head_dim": 64,
        "num_q_heads": 71,
        "num_kv_heads": 1,
        "kv_cache_dtype": "auto",
        "calculate_kv_scales": False,
        "attn_module_sliding_window": None,
        "flashinfer_window_left": -1,
        "kv_cache_spec_type": "FullAttentionSpec",
    }
    return [
        {
            "batch_id": 0,
            "batch_size": 1,
            "batch_num_tokens": 64,
            "batch_num_prefill_tokens": 64,
            "batch_num_decode_tokens": 0,
            "batch_request_num_tokens": [64],
            "op_name": "attn_kv_cache_save",
            "cuda_time_ms": 0.002112,
            "count": 1,
            "meta": {
                **base_meta,
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
            "cuda_time_ms": 0.006464,
            "count": 1,
            "meta": {
                **base_meta,
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
            "cuda_time_ms": 0.001920,
            "count": 1,
            "meta": {
                **base_meta,
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
            "cuda_time_ms": 0.005696,
            "count": 1,
            "meta": {
                **base_meta,
                "max_seqlen_q": 1,
                "max_seqlen_k": 65,
                "num_actual_tokens": 1,
            },
        },
    ]


def test_mqa_scope_summary_preserves_flashinfer_runtime_metadata() -> None:
    builder = _load_builder_module()
    summaries = builder.summarize_attention_scopes(_sample_rows())

    assert summaries["attn_kv_cache_save"].rows == 2
    assert summaries["attn_kv_cache_save"].median_cuda_time_ms == pytest.approx(
        0.002016
    )
    assert summaries["attn_prefill"].median_cuda_time_ms == pytest.approx(0.006464)
    assert summaries["attn_decode"].median_cuda_time_ms == pytest.approx(0.005696)

    for scope in ["attn_kv_cache_save", "attn_prefill", "attn_decode"]:
        summary = summaries[scope]
        assert summary.backend == "FLASHINFER_VLLM_V1"
        assert summary.head_dim == 64
        assert summary.num_q_heads == 71
        assert summary.num_kv_heads == 1
        assert summary.sliding_window is None
        assert summary.kv_cache_spec_type == "FullAttentionSpec"


def test_mqa_frontier_attention_rows_match_native_falcon_contract() -> None:
    builder = _load_builder_module()
    config = builder.FalconMqaStage2Config()
    rows = builder.build_frontier_attention_rows(_sample_rows(), config)

    prefill_rows = [row for row in rows if row["is_prefill"] is True]
    decode_rows = [row for row in rows if row["is_prefill"] is False]
    assert len(prefill_rows) == 1
    assert len(decode_rows) == 1

    prefill = prefill_rows[0]
    assert prefill["n_embd"] == 4544
    assert prefill["n_q_head"] == 71
    assert prefill["n_kv_head"] == 1
    assert prefill["block_size"] == 16
    assert prefill["num_tensor_parallel_workers"] == 1
    assert prefill["attention_backend"] == "FLASHINFER_VLLM_V1"
    assert prefill["model_arch"] == "falcon_mqa"
    assert prefill["prefill_chunk_size"] == 64
    assert prefill["kv_cache_size"] == 0
    assert prefill["time_stats.attn_prefill.median"] == pytest.approx(0.006464)
    assert prefill["time_stats.attn_kv_cache_save.median"] == pytest.approx(0.002112)

    decode = decode_rows[0]
    assert decode["prefill_chunk_size"] == 0
    assert decode["kv_cache_size"] == 65
    assert decode["time_stats.attn_decode.median"] == pytest.approx(0.005696)
    assert decode["time_stats.attn_kv_cache_save.median"] == pytest.approx(0.001920)


def test_mqa_native_tp_constraints_and_synthetic_replication_coverage() -> None:
    builder = _load_builder_module()
    native = builder.FalconMqaStage2Config()
    assert builder.local_q_heads(native.num_q_heads, 1) == 71
    assert builder.local_kv_heads(native.num_kv_heads, 1) == 1

    for tp_size in [2, 4, 8]:
        with pytest.raises(ValueError, match="native Falcon-7B Q heads"):
            builder.local_q_heads(native.num_q_heads, tp_size, native_falcon=True)

    synthetic = builder.FalconMqaStage2Config(
        hidden_size=4608,
        num_q_heads=72,
        num_kv_heads=1,
        synthetic_tp_mock=True,
    )
    for tp_size, expected_local_q in [(2, 36), (4, 18), (8, 9)]:
        assert builder.local_q_heads(synthetic.num_q_heads, tp_size) == expected_local_q
        assert builder.local_kv_heads(synthetic.num_kv_heads, tp_size) == 1


def test_mqa_memory_metrics_match_native_page_bytes() -> None:
    builder = _load_builder_module()
    config = builder.FalconMqaStage2Config()
    memory = builder.compute_mqa_memory_metrics(
        config=config,
        observed_max_context_tokens=65,
    )

    assert memory["native_tp1_page_bytes_per_layer"] == 4_096
    assert memory["observed_context_tokens"] == 65
    assert memory["observed_context_blocks"] == 5
    assert memory["observed_context_bytes_per_layer_tp1"] == 20_480


def test_mqa_error_matrix_converges_for_transformed_native_live_rows() -> None:
    builder = _load_builder_module()
    config = builder.FalconMqaStage2Config()
    rows = builder.build_frontier_attention_rows(_sample_rows(), config)
    matrix = builder.build_error_matrix(_sample_rows(), rows, config)

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


def test_mqa_stage2_cli_writes_reproducible_artifacts(tmp_path: Path) -> None:
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

    assert "MQA Stage 2 profile/model artifacts written" in completed.stdout
    assert (output_dir / "attention.csv").exists()
    assert (output_dir / "error_matrix.json").exists()
    assert (output_dir / "error_matrix.md").exists()
    summary = json.loads((output_dir / "error_matrix.json").read_text(encoding="utf-8"))
    assert summary["environment"]["python_bin"] == sys.executable
    assert summary["config"]["flashinfer_python_expected_version"] == "0.3.1.post1"
    assert summary["config"]["attention_backend"] == "FLASHINFER_VLLM_V1"


@pytest.mark.skipif(
    not LIVE_CUDA_OPS.exists() or LIVE_CUDA_OPS.stat().st_size == 0,
    reason="MQA FlashInfer live-probe artifact is not present under /tmp",
)
def test_mqa_stage2_builder_accepts_live_probe_artifact(tmp_path: Path) -> None:
    builder = _load_builder_module()
    output_dir = tmp_path / "live"
    builder.write_stage2_outputs(LIVE_CUDA_OPS, output_dir, builder.FalconMqaStage2Config())

    summary = json.loads((output_dir / "error_matrix.json").read_text(encoding="utf-8"))
    assert all(row["passes_5pct"] for row in summary["latency"])
    assert all(row["passes_5pct"] for row in summary["memory"])
