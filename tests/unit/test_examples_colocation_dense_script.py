#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = (
    _REPO_ROOT
    / "examples"
    / "architecture"
    / "co-location"
    / "offline"
    / "dense_model_basic.sh"
)
_DENSE_SCRIPTS = (
    _SCRIPT_PATH,
    _REPO_ROOT
    / "examples"
    / "architecture"
    / "co-location"
    / "online"
    / "dense_model_basic_online.sh",
)


def test_dense_colocation_example_uses_valid_attn_data_parallel_flag() -> None:
    script_text = _SCRIPT_PATH.read_text()

    assert "--replica_config_attn_data_parallel_size" in script_text
    assert '"$DP"' in script_text
    assert "--replica_config_attn_dataa t_parallel_size" not in script_text


def test_dense_colocation_example_uses_python3_compatible_entrypoint() -> None:
    script_text = _SCRIPT_PATH.read_text()

    assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in script_text
    assert "CMD=(" in script_text
    assert '"$PYTHON_BIN" -m frontier.main' in script_text


def test_dense_colocation_examples_explain_none_mode_when_kernel_only_profiles_are_missing() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": "/bin/true",
        "ENABLE_DUMMY_MODE": "false",
        "DEVICE": "missing_kernel_only_device",
        "MODEL_NAME": "missing_kernel_only_model",
        "DECODE_CUDA_GRAPH_MODE": "full_decode_only",
    }

    for script in _DENSE_SCRIPTS:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=_REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert "Kernel-only profiling CSVs are required" in result.stderr, script
        assert "DECODE_CUDA_GRAPH_MODE=none" in result.stderr, script
