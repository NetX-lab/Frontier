from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from tests.analysis.mla_deepseek_v2 import build_mla_stage3_online_trace as stage3_builder
from tests.analysis.mla_deepseek_v2.build_mla_stage3_online_trace import (
    DeepSeekV2MlaStage3TraceConfig,
    build_frontier_trace_events,
    build_stage3_error_matrix,
    write_stage3_outputs,
)
from tests.unit.test_mla_stage2_profile_modeling import _sample_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER = (
    PROJECT_ROOT
    / "tests"
    / "analysis"
    / "mla_deepseek_v2"
    / "build_mla_stage3_online_trace.py"
)
COMPARATOR = (
    PROJECT_ROOT
    / "tests"
    / "comparison"
    / "chunked_prefill_online"
    / "compare_online_per_op.py"
)
def _installed_flashinfer_python_version() -> str | None:
    try:
        return version("flashinfer-python")
    except PackageNotFoundError:
        return None


LIVE_CUDA_OPS = Path(
    "/tmp/frontier_mla_deepseek_v2_flashinfer_mla_live_probe/cuda_ops.jsonl"
)
LIVE_BATCH_LOG = Path(
    "/tmp/frontier_mla_deepseek_v2_flashinfer_mla_live_probe/batch_log.jsonl"
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_mla_stage3_flashinfer_environment_fails_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_package(_dist_name: str) -> str:
        raise PackageNotFoundError("flashinfer-python")

    monkeypatch.setattr(stage3_builder, "version", raise_missing_package)

    with pytest.raises(ValueError, match="flashinfer-python is not installed"):
        stage3_builder._flashinfer_environment(DeepSeekV2MlaStage3TraceConfig())


def test_mla_stage3_flashinfer_environment_fails_on_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage3_builder, "version", lambda _dist_name: "0.3.1.post0")

    with pytest.raises(ValueError, match="flashinfer-python version mismatch"):
        stage3_builder._flashinfer_environment(DeepSeekV2MlaStage3TraceConfig())


def test_mla_stage3_flashinfer_environment_records_exact_expected_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage3_builder, "version", lambda _dist_name: "0.3.1.post1")

    environment = stage3_builder._flashinfer_environment(
        DeepSeekV2MlaStage3TraceConfig()
    )

    assert environment["python_bin"] == sys.executable
    assert environment["flashinfer_python_version"] == "0.3.1.post1"
    assert environment["flashinfer_python_expected_version"] == "0.3.1.post1"


def test_mla_stage3_trace_builder_preserves_flashinfer_mla_scope_shape() -> None:
    rows = _sample_rows()
    events = build_frontier_trace_events(rows, DeepSeekV2MlaStage3TraceConfig())

    assert [event["name"] for event in events] == [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_kv_cache_save",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    assert all(event["type"] == "COMPUTE" for event in events)
    assert all(event["layer_id"] == 0 for event in events)

    prefill_event = events[2]
    assert prefill_event["batch_id"] == 0
    assert prefill_event["cluster"] == "PREFILL"
    assert prefill_event["duration_ms"] == pytest.approx(rows[2]["cuda_time_ms"])
    assert prefill_event["meta"]["num_layers"] == 1
    assert prefill_event["meta"]["model_name"] == "mla-deepseek-v2-stage3-mock"
    assert prefill_event["meta"]["request_ids"] == ["0"]
    assert prefill_event["meta"]["num_tokens"] == [64]
    assert prefill_event["meta"]["batch_size"] == 1
    assert prefill_event["meta"]["effective_total_tokens_compute"] == 64
    assert prefill_event["meta"]["attention_backend"] == "FLASHINFER_MLA"
    assert prefill_event["meta"]["use_mla"] is True
    assert prefill_event["meta"]["runtime_num_kv_heads"] == 1
    assert prefill_event["meta"]["runtime_head_size"] == 576
    assert prefill_event["meta"]["kv_lora_rank"] == 512
    assert prefill_event["meta"]["qk_rope_head_dim"] == 64
    assert prefill_event["meta"]["block_size"] == 64
    assert prefill_event["meta"]["flashinfer_python_expected_version"] == "0.3.1.post1"

    decode_event = events[-2]
    assert decode_event["batch_id"] == 1
    assert decode_event["cluster"] == "DECODE"
    assert decode_event["meta"]["request_ids"] == ["0"]
    assert decode_event["meta"]["effective_total_tokens_compute"] == 1
    assert decode_event["meta"]["max_seqlen_q"] == 1
    assert decode_event["meta"]["max_seqlen_k"] == 65


def test_mla_stage3_comparator_consumes_builder_output_with_zero_attention_error(
    tmp_path: Path,
) -> None:
    cuda_log = tmp_path / "cuda_ops.jsonl"
    trace_path = tmp_path / "op_traces.jsonl"
    output_json = tmp_path / "per_op_comparison.json"
    output_md = tmp_path / "per_op_comparison.md"
    _write_jsonl(cuda_log, _sample_rows())

    events = build_frontier_trace_events(_sample_rows(), DeepSeekV2MlaStage3TraceConfig())
    _write_jsonl(trace_path, events)

    subprocess.run(
        [
            sys.executable,
            str(COMPARATOR),
            "--vllm-op-log",
            str(cuda_log),
            "--frontier-op-traces",
            str(trace_path),
            "--model-profile",
            "dense",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--threshold-percent",
            "5.0",
        ],
        check=True,
    )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    matrix = build_stage3_error_matrix(summary)

    assert summary["status"] == "PASS"
    assert matrix["latency_ops_evaluated"] == [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    for row in matrix["latency"]:
        assert row["relative_error_percent"] == pytest.approx(0.0)
        assert row["passes_5pct"] is True


def test_mla_stage3_cli_writes_trace_and_error_matrix(tmp_path: Path) -> None:
    cuda_log = tmp_path / "cuda_ops.jsonl"
    batch_log = tmp_path / "batch_log.jsonl"
    output_dir = tmp_path / "stage3"
    _write_jsonl(cuda_log, _sample_rows())
    _write_jsonl(
        batch_log,
        [
            {
                "batch_id": 0,
                "batch_size": 1,
                "batch_num_tokens": 64,
                "batch_num_prefill_tokens": 64,
                "batch_num_decode_tokens": 0,
                "batch_execution_time_ms": 10.0,
                "timestamp": 1.0,
                "request_ids": ["0"],
                "request_num_tokens": [64],
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--cuda-op-log",
            str(cuda_log),
            "--batch-log",
            str(batch_log),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "MLA Stage 3 online trace artifacts written" in completed.stdout
    assert (output_dir / "op_traces.jsonl").exists()
    assert (output_dir / "per_op_comparison.json").exists()
    assert (output_dir / "per_op_comparison.md").exists()
    assert (output_dir / "stage3_error_matrix.json").exists()
    assert (output_dir / "stage3_error_matrix.md").exists()

    matrix = json.loads(
        (output_dir / "stage3_error_matrix.json").read_text(encoding="utf-8")
    )
    assert matrix["environment"]["python_bin"] == sys.executable
    assert matrix["environment"]["flashinfer_python_version"] == _installed_flashinfer_python_version()
    assert matrix["environment"]["flashinfer_python_expected_version"] == "0.3.1.post1"
    assert matrix["runtime"]["attention_backend"] == "FLASHINFER_MLA"
    assert matrix["runtime"]["use_mla"] is True
    assert matrix["runtime"]["runtime_num_kv_heads"] == 1
    assert matrix["runtime"]["runtime_head_size"] == 576
    assert matrix["runtime"]["block_size"] == 64
    assert matrix["latency_ops_evaluated"] == [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    assert all(row["passes_5pct"] for row in matrix["latency"])
    assert all(row["passes_5pct"] for row in matrix["memory"])


@pytest.mark.skipif(
    not (LIVE_CUDA_OPS.exists() and LIVE_BATCH_LOG.exists()),
    reason="MLA FlashInfer live-probe artifacts are not present under /tmp",
)
def test_mla_stage3_builder_accepts_flashinfer_mla_live_probe_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "live"
    summary = write_stage3_outputs(
        LIVE_CUDA_OPS,
        LIVE_BATCH_LOG,
        output_dir,
        DeepSeekV2MlaStage3TraceConfig(),
    )

    assert summary["error_matrix"]["status"] == "PASS"
    assert summary["error_matrix"]["latency_ops_evaluated"] == [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    for row in summary["error_matrix"]["latency"]:
        assert row["relative_error_percent"] == pytest.approx(0.0)
        assert row["passes_5pct"] is True
    for row in summary["error_matrix"]["memory"]:
        assert row["relative_error_percent"] == pytest.approx(0.0)
        assert row["passes_5pct"] is True
