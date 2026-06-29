from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = (
    PROJECT_ROOT
    / "tests"
    / "analysis"
    / "mqa_falcon"
    / "run_vllm_flashinfer_live_probe.sh"
)
VALIDATOR = (
    PROJECT_ROOT
    / "tests"
    / "analysis"
    / "mqa_falcon"
    / "validate_flashinfer_live_probe.py"
)


def test_mqa_validator_required_scopes_are_declared_from_shared_dense_mapping() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
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

    validator = importlib.import_module(
        "tests.analysis.mqa_falcon.validate_flashinfer_live_probe"
    )
    assert validator.REQUIRED_SCOPES == get_profiling_metric_names(
        DENSE_ATTENTION_FAMILY
    )


def test_mqa_flashinfer_probe_script_uses_required_backend_and_env() -> None:
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert 'PYTHON_BIN="${PYTHON_BIN:-/local/ycfeng/anaconda3/envs/frontier/bin/python}"' in source
    assert 'FRONTIER_CUDA_HOME="${FRONTIER_CUDA_HOME:-/usr/local/cuda-13.2}"' in source
    assert (
        'FLASHINFER_PYTHON_EXPECTED_VERSION="${FLASHINFER_PYTHON_EXPECTED_VERSION:-0.3.1.post1}"'
        in source
    )
    assert 'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"' in source
    assert 'EXPECTED_BACKEND="${EXPECTED_BACKEND:-FLASHINFER_VLLM_V1}"' in source
    assert "VLLM_FRONTIER_INSTRUMENTATION=1" in source
    assert "VLLM_FRONTIER_RUNTIME_META_ENABLED=1" in source
    assert "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH" in source
    assert "attn_kv_cache_save,attn_prefill,attn_decode" in source
    assert "flashinfer_version = version(\"flashinfer-python\")" in source
    assert 'export PYTHONPATH="$PROJECT_ROOT:$VLLM_ROOT:${PYTHONPATH:-}"' in source
    assert 'export CUDA_HOME="$FRONTIER_CUDA_HOME"' in source
    assert 'export PYTORCH_NVCC="$FRONTIER_CUDA_HOME/bin/nvcc"' in source
    assert "--tensor-parallel-size \"$TP_SIZE\"" in source
    assert "validate_flashinfer_live_probe.py" in source


def test_mqa_probe_writes_falcon_mqa_dummy_config() -> None:
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    for snippet in [
        '"architectures": ["FalconForCausalLM"]',
        '"model_type": "falcon"',
        '"hidden_size": 4544',
        '"num_hidden_layers": 1',
        '"num_attention_heads": 71',
        '"multi_query": true',
        '"new_decoder_architecture": false',
        '"parallel_attn": true',
        '"alibi": false',
        '"bias": false',
        '"torch_dtype": "bfloat16"',
        '"vocab_size": 65024',
    ]:
        assert snippet in source


def test_mqa_validator_accepts_flashinfer_mqa_runtime_meta(tmp_path: Path) -> None:
    cuda_log = tmp_path / "cuda_ops.jsonl"
    run_log = tmp_path / "run.log"
    rows = [
        {
            "batch_id": 0,
            "op_name": "attn_kv_cache_save",
            "cuda_time_ms": 0.0021,
            "count": 1,
            "meta": {
                "attention_backend": "FLASHINFER_VLLM_V1",
                "head_dim": 64,
                "num_q_heads": 71,
                "num_kv_heads": 1,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": None,
                "flashinfer_window_left": -1,
                "kv_cache_spec_type": "FullAttentionSpec",
                "max_seqlen_q": 64,
                "max_seqlen_k": 64,
                "num_actual_tokens": 64,
            },
        },
        {
            "batch_id": 0,
            "op_name": "attn_prefill",
            "cuda_time_ms": 0.0064,
            "count": 1,
            "meta": {
                "attention_backend": "FLASHINFER_VLLM_V1",
                "head_dim": 64,
                "num_q_heads": 71,
                "num_kv_heads": 1,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": None,
                "flashinfer_window_left": -1,
                "kv_cache_spec_type": "FullAttentionSpec",
                "max_seqlen_q": 64,
                "max_seqlen_k": 64,
                "num_actual_tokens": 64,
            },
        },
        {
            "batch_id": 2,
            "op_name": "attn_decode",
            "cuda_time_ms": 0.0055,
            "count": 1,
            "meta": {
                "attention_backend": "FLASHINFER_VLLM_V1",
                "head_dim": 64,
                "num_q_heads": 71,
                "num_kv_heads": 1,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": None,
                "flashinfer_window_left": -1,
                "kv_cache_spec_type": "FullAttentionSpec",
                "max_seqlen_q": 1,
                "max_seqlen_k": 65,
                "num_actual_tokens": 1,
            },
        },
    ]
    cuda_log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run_log.write_text(
        "Initializing a V1 LLM engine\n"
        "Using FlashInfer backend on V1 engine\n"
        "tensor_parallel_size=1\n"
        "chunked_prefill_enabled=False\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--cuda-op-log",
            str(cuda_log),
            "--run-log",
            str(run_log),
            "--expected-backend",
            "FLASHINFER_VLLM_V1",
            "--expected-head-dim",
            "64",
            "--expected-q-heads",
            "71",
            "--expected-kv-heads",
            "1",
            "--expected-tp",
            "1",
            "--expected-chunked-prefill-enabled",
            "False",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "FlashInfer MQA live probe validation passed." in completed.stdout
    assert "attn_kv_cache_save: rows=1" in completed.stdout
    assert "attn_prefill: rows=1" in completed.stdout
    assert "attn_decode: rows=1" in completed.stdout


def test_mqa_validator_rejects_materialized_kv_repeat_metadata(tmp_path: Path) -> None:
    cuda_log = tmp_path / "cuda_ops.jsonl"
    run_log = tmp_path / "run.log"
    rows = []
    for scope, max_seqlen_q, max_seqlen_k in [
        ("attn_kv_cache_save", 64, 64),
        ("attn_prefill", 64, 64),
        ("attn_decode", 1, 65),
    ]:
        rows.append({
            "batch_id": 0,
            "op_name": scope,
            "cuda_time_ms": 0.0064,
            "count": 1,
            "meta": {
            "attention_backend": "FLASHINFER_VLLM_V1",
            "head_dim": 64,
            "num_q_heads": 71,
            "num_kv_heads": 71,
            "kv_cache_dtype": "auto",
            "calculate_kv_scales": False,
            "attn_module_sliding_window": None,
            "flashinfer_window_left": -1,
            "kv_cache_spec_type": "FullAttentionSpec",
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "num_actual_tokens": max_seqlen_q,
            },
        })
    cuda_log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run_log.write_text(
        "Initializing a V1 LLM engine\n"
        "Using FlashInfer backend on V1 engine\n"
        "tensor_parallel_size=1\n"
        "chunked_prefill_enabled=False\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--cuda-op-log",
            str(cuda_log),
            "--run-log",
            str(run_log),
            "--expected-backend",
            "FLASHINFER_VLLM_V1",
            "--expected-head-dim",
            "64",
            "--expected-q-heads",
            "71",
            "--expected-kv-heads",
            "1",
            "--expected-tp",
            "1",
            "--expected-chunked-prefill-enabled",
            "False",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert "Unexpected num_kv_heads" in completed.stderr
