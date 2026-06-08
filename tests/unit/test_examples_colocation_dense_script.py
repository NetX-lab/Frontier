#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "architecture"
    / "co-location"
    / "dense_model_basic.sh"
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


def test_dense_colocation_example_defaults_to_astra_sim_analytical_backend() -> None:
    script_text = _SCRIPT_PATH.read_text()

    assert 'CC_BACKEND_CONFIG_TYPE="${CC_BACKEND_CONFIG_TYPE:-astra_sim_analytical}"' in script_text
    assert '--cc_backend_config_type "$CC_BACKEND_CONFIG_TYPE"' in script_text
    assert "--cc_backend_config_type collective_sim" not in script_text
