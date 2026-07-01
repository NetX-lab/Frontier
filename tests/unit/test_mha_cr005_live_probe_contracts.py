from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = PROJECT_ROOT / "tests" / "analysis" / "mha_phi3" / "run_vllm_flash_attn_live_probe.sh"
VALIDATOR = PROJECT_ROOT / "tests" / "analysis" / "mha_phi3" / "validate_flash_attn_live_probe.py"
def test_mha_validator_required_scopes_are_declared_from_shared_dense_mapping() -> None:
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
        "tests.analysis.mha_phi3.validate_flash_attn_live_probe"
    )
    assert validator.REQUIRED_SCOPES == get_profiling_metric_names(
        DENSE_ATTENTION_FAMILY
    )


def test_mha_flash_attention_probe_script_uses_required_backend_and_instrumentation() -> None:
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert "VLLM_ATTENTION_BACKEND=\"${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}\"" in source
    assert "EXPECTED_BACKEND=\"${EXPECTED_BACKEND:-FLASH_ATTN_VLLM_V1}\"" in source
    assert "VLLM_FRONTIER_INSTRUMENTATION=1" in source
    assert "VLLM_FRONTIER_RUNTIME_META_ENABLED=1" in source
    assert 'FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/frontier_flashinfer_workspace}"' in source
    assert "export FLASHINFER_WORKSPACE_BASE" in source
    assert "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH" in source
    assert "attn_kv_cache_save,attn_prefill,attn_decode" in source
    assert "examples/offline_inference/measure_ttft.py" in source
    assert "--enforce-eager" in source
    assert "validate_flash_attn_live_probe.py" in source
    assert 'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"' not in source
    assert 'EXPECTED_BACKEND="${EXPECTED_BACKEND:-FLASHINFER_VLLM_V1}"' not in source


def test_mha_flash_attention_probe_script_defaults_to_current_frontier_env() -> None:
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert 'PYTHON_BIN="${PYTHON_BIN:-/local/ycfeng/anaconda3/envs/frontier/bin/python}"' in source
    assert 'FRONTIER_CUDA_HOME="${FRONTIER_CUDA_HOME:-/usr/local/cuda-13.2}"' in source
    assert 'FLASHINFER_PYTHON_EXPECTED_VERSION="${FLASHINFER_PYTHON_EXPECTED_VERSION:-0.3.1.post1}"' in source
    assert 'export CUDA_HOME="$FRONTIER_CUDA_HOME"' in source
    assert 'export CUDA_PATH="$FRONTIER_CUDA_HOME"' in source
    assert 'export PYTORCH_NVCC="$FRONTIER_CUDA_HOME/bin/nvcc"' in source
    assert 'export PATH="$FRONTIER_CUDA_HOME/bin:$(dirname "$PYTHON_BIN"):$PATH"' in source
    assert 'export LD_LIBRARY_PATH="$FRONTIER_CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' in source
    assert 'export LIBRARY_PATH="$FRONTIER_CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"' in source
    assert 'flashinfer_version = version("flashinfer-python")' in source
    assert 'if flashinfer_version != expected_flashinfer_version:' in source
    assert (
        'VLLM_FRONTIER_COMPILED_PACKAGE="${VLLM_FRONTIER_COMPILED_PACKAGE:-'
        '/local/ycfeng/anaconda3/envs/frontier/lib/python3.10/site-packages/vllm}"'
        in source
    )
    assert (
        'VLLM_FRONTIER_VLLM_FLASH_ATTN_PACKAGE="${VLLM_FRONTIER_VLLM_FLASH_ATTN_PACKAGE:-'
        '/local/ycfeng/anaconda3/envs/frontier/lib/python3.10/site-packages/vllm/vllm_flash_attn}"'
        in source
    )
    assert "vllm-bs-0.10.2" not in source






def test_validator_accepts_required_scopes_and_runtime_meta(tmp_path: Path) -> None:
    cuda_log = tmp_path / "cuda_ops.jsonl"
    run_log = tmp_path / "run.log"
    rows = [
        {
            "batch_id": 0,
            "op_name": "attn_kv_cache_save",
            "cuda_time_ms": 0.031,
            "count": 1,
            "meta": {
                "attention_backend": "FLASH_ATTN_VLLM_V1",
                "head_dim": 96,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": None,
                "kv_cache_spec_type": "FullAttentionSpec",
                "max_seqlen_q": 16,
            },
        },
        {
            "batch_id": 0,
            "op_name": "attn_prefill",
            "cuda_time_ms": 1.25,
            "count": 1,
            "meta": {
                "attention_backend": "FLASH_ATTN_VLLM_V1",
                "head_dim": 96,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": None,
                "kv_cache_spec_type": "FullAttentionSpec",
                "max_seqlen_q": 16,
            },
        },
        {
            "batch_id": 1,
            "op_name": "attn_decode",
            "cuda_time_ms": 0.17,
            "count": 1,
            "meta": {
                "attention_backend": "FLASH_ATTN_VLLM_V1",
                "head_dim": 96,
                "kv_cache_dtype": "auto",
                "calculate_kv_scales": False,
                "attn_module_sliding_window": None,
                "kv_cache_spec_type": "FullAttentionSpec",
                "max_seqlen_q": 1,
            },
        },
    ]
    cuda_log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run_log.write_text(
        "Initializing a V1 LLM engine\n"
        "Using Flash Attention backend on V1 engine\n"
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
            "FLASH_ATTN_VLLM_V1",
            "--expected-head-dim",
            "96",
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

    assert "FlashAttention live probe validation passed." in completed.stdout
    assert "attn_kv_cache_save: rows=1" in completed.stdout
    assert "attn_prefill: rows=1" in completed.stdout
    assert "attn_decode: rows=1" in completed.stdout


def test_validator_rejects_missing_decode_scope(tmp_path: Path) -> None:
    cuda_log = tmp_path / "cuda_ops.jsonl"
    run_log = tmp_path / "run.log"
    row = {
        "batch_id": 0,
        "op_name": "attn_prefill",
        "cuda_time_ms": 1.25,
        "count": 1,
        "meta": {
            "attention_backend": "FLASH_ATTN_VLLM_V1",
            "head_dim": 96,
            "kv_cache_dtype": "auto",
            "calculate_kv_scales": False,
            "attn_module_sliding_window": None,
            "kv_cache_spec_type": "FullAttentionSpec",
            "max_seqlen_q": 16,
        },
    }
    cuda_log.write_text(json.dumps(row) + "\n", encoding="utf-8")
    run_log.write_text(
        "Initializing a V1 LLM engine\n"
        "Using Flash Attention backend on V1 engine\n"
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
            "FLASH_ATTN_VLLM_V1",
            "--expected-head-dim",
            "96",
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
    assert "Missing required CUDA scopes" in completed.stderr
    assert "attn_decode" in completed.stderr
