from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "architecture" / "run_dummy_smoke_matrix.sh"
README = REPO_ROOT / "examples" / "architecture" / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dummy_smoke_matrix_script_exists_and_is_shell_valid() -> None:
    assert SCRIPT.exists(), f"Missing dummy-mode example runner: {SCRIPT}"

    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_dummy_smoke_matrix_covers_dense_moe_colocation_pdd_online_offline() -> None:
    text = _read(SCRIPT)

    expected_cases = [
        "co-location/offline/dense_model_basic.sh",
        "co-location/offline/moe_model_basic.sh",
        "co-location/online/dense_model_basic_online.sh",
        "co-location/online/moe_model_basic_online.sh",
        "pdd/offline/dense_model_basic.sh",
        "pdd/offline/moe_model_basic.sh",
        "pdd/online/dense_model_basic_online.sh",
        "pdd/online/moe_model_basic_online.sh",
    ]
    for case in expected_cases:
        assert f'"{case}"' in text

    assert "ENABLE_DUMMY_MODE=true" in text
    assert "DUMMY_EXEC_TIME_MS" in text
    assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in text
    assert 'PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in text
    assert "random_forrest_execution_time_predictor_config_linear_op_input_file" not in text
    assert "random_forrest_execution_time_predictor_config_atten_input_file" not in text
    assert "random_forrest_execution_time_predictor_config_moe_input_file" not in text


def test_architecture_readme_documents_dummy_smoke_matrix() -> None:
    readme = _read(README)

    assert "run_dummy_smoke_matrix.sh" in readme
    assert "does not consume profiling CSV datasets" in readme
    assert "dense/MoE" in readme
    assert "co-location/PDD" in readme
    assert "offline/online" in readme
