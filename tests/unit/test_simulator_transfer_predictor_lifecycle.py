"""Simulator transfer-predictor lifecycle contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from frontier.config import SimulationConfig
from frontier.m2n_transfer import M2NTransferPredictorRegistry
from frontier.simulator import Simulator


REPO_ROOT = Path(__file__).resolve().parents[2]


def _capture_simulator_argv(tmp_path: Path, script: Path) -> list[str]:
    capture_path = tmp_path / f"{script.stem}.argv"
    fake_python = tmp_path / "capture_python.sh"
    fake_python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_PATH": str(capture_path),
            "PYTHON_BIN": str(fake_python),
            "NUM_REQUESTS": "1",
            "PREFILL_TOKENS": "16",
            "DECODE_TOKENS": "4",
            "METRICS_OUTPUT_DIR": str(tmp_path / "metrics"),
            "RUN_ID": script.stem,
        }
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    argv = capture_path.read_text(encoding="utf-8").splitlines()
    assert argv[:2] == ["-m", "frontier.main"]
    return argv[2:]


@pytest.mark.parametrize(
    ("relative_script", "expected_m2n_constructions"),
    (
        ("examples/architecture/pdd/offline/dense_model_basic.sh", 0),
        ("examples/architecture/pd-af-disagg/offline/dense_model_basic.sh", 1),
    ),
)
def test_simulator_constructs_m2n_predictor_only_for_pdaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_script: str,
    expected_m2n_constructions: int,
) -> None:
    simulator_argv = _capture_simulator_argv(tmp_path, REPO_ROOT / relative_script)
    monkeypatch.setattr(sys, "argv", ["frontier.main", *simulator_argv])
    config = SimulationConfig.create_from_cli_args()

    registry_get = Mock(wraps=M2NTransferPredictorRegistry.get)
    monkeypatch.setattr(M2NTransferPredictorRegistry, "get", registry_get)
    monkeypatch.setattr(Simulator, "_init_simulation_mode", lambda self: None)

    Simulator(config)

    assert registry_get.call_count == expected_m2n_constructions, (
        f"sys_arch={config.sys_arch}: expected M2N predictor constructions="
        f"{expected_m2n_constructions}, actual={registry_get.call_count}, "
        f"delta={registry_get.call_count - expected_m2n_constructions}"
    )
