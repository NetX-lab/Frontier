import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "architecture" / "pdd" / "offline" / "dense_model_basic.sh"
RELEASE_GUARD_MESSAGE = "Disaggregated architecture support is currently being optimized"
RUN_ID = "pytest_pdd_dense_example_smoke"
EXPECTED_KV_TRANSFER_BYTES = 4_194_304


def test_pdd_dense_example_script_runs_sequential_smoke(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHON_BIN": sys.executable,
            "METRICS_OUTPUT_DIR": str(tmp_path / "metrics"),
            "RUN_ID": RUN_ID,
            "NUM_REQUESTS": "1",
            "PREFILL_TOKENS": "8",
            "DECODE_TOKENS": "2",
            "QPS": "1000.0",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    combined_output = result.stdout + result.stderr
    assert RELEASE_GUARD_MESSAGE not in combined_output
    assert result.returncode == 0, combined_output
    assert "Simulation completed successfully." in result.stdout

    metrics_root = tmp_path / "metrics"
    run_dirs = [path for path in metrics_root.rglob(RUN_ID) if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    request_metrics_path = run_dir / "request_metrics.csv"
    system_metrics_path = run_dir / "system_metrics.json"
    assert request_metrics_path.is_file()
    assert system_metrics_path.is_file()

    with request_metrics_path.open(newline="") as request_metrics_file:
        request_rows = list(csv.DictReader(request_metrics_file))
    with system_metrics_path.open() as system_metrics_file:
        system_metrics = json.load(system_metrics_file)

    assert len(request_rows) == 1
    request_row = request_rows[0]
    assert float(request_row["request_e2e_time"]) > 0.0
    assert float(request_row["ttft"]) > 0.0
    assert float(request_row["transfer_kv_cache"]) > 0.0

    simulation_metadata = system_metrics["simulation_metadata"]
    assert simulation_metadata["total_requests"] == 1
    assert simulation_metadata["completed_requests"] == 1

    transfer_statistics = system_metrics["kv_cache_transfer_statistics"]
    assert transfer_statistics["total_transfers"] == 1
    assert system_metrics["kv_cache_transfer_total_bytes"] == EXPECTED_KV_TRANSFER_BYTES
    assert transfer_statistics["total_transfer_time_ms"] > 0.0
