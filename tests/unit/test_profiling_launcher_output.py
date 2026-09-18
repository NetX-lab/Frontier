"""Execute profiling shell postflight with a native-producer stand-in."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {"linear_op": "profile_linear_op.sh", "attention": "profile_attention_chunked_prefill.sh", "moe": "profile_moe.sh"}


@pytest.mark.parametrize("operator", SCRIPTS)
@pytest.mark.parametrize("method,suffix", [
    ("cuda_event", ""), ("cuda", ""), ("device_event", "_device_event"),
    ("kernel_only", "_kernel_only"), ("record_function", "_kernel_only"),
])
def test_selected_output_passes_postflight(tmp_path, operator, method, suffix):
    expected = tmp_path / "compute/test_device/test_model" / f"{operator}{suffix}.csv"
    result = run_launcher(tmp_path, operator, method, expected, produce=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected.is_file()
    assert f"Resolved output: {expected}" in result.stdout


@pytest.mark.parametrize("operator", SCRIPTS)
def test_stale_cuda_file_cannot_satisfy_device_event_postflight(tmp_path, operator):
    directory = tmp_path / "compute/test_device/test_model"
    directory.mkdir(parents=True)
    (directory / f"{operator}.csv").write_text("stale CUDA data\n")
    expected = directory / f"{operator}_device_event.csv"
    result = run_launcher(tmp_path, operator, "device_event", expected, produce=False)
    assert result.returncode != 0
    assert str(expected) in result.stderr


def run_launcher(root, operator, method, expected, *, produce):
    executable = root / "producer-python"
    executable.write_text(f"#!{sys.executable}\n" + '''import os
from pathlib import Path
import sys
if len(sys.argv) > 2 and sys.argv[1] == "-m" and sys.argv[2].endswith(".main"):
    if os.environ["STANDIN_PRODUCE"] == "1":
        path = Path(os.environ["STANDIN_OUTPUT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("producer output\\n")
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
    executable.chmod(0o755)
    return subprocess.run(
        ["bash", str(ROOT / "examples/profiling" / SCRIPTS[operator]),
         "--model", "test_model", "--device", "test_device", "--output-root", str(root),
         "--profile-method", method], cwd=ROOT,
        env={**os.environ, "PYTHON_BIN": str(executable), "STANDIN_OUTPUT": str(expected),
             "STANDIN_PRODUCE": "1" if produce else "0"},
        text=True, capture_output=True, timeout=60,
    )
