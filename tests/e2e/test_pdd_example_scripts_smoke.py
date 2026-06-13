from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PDD_DIR = REPO_ROOT / "examples" / "architecture" / "pdd"

SCRIPT_CASES = (
    ("compat_dense", PDD_DIR / "dense_model_basic.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 1),
    ("offline_dense", PDD_DIR / "offline" / "dense_model_basic.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 1),
    ("offline_moe", PDD_DIR / "offline" / "moe_model_basic.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 1),
    ("offline_thinking", PDD_DIR / "offline" / "thinking_mode_basic.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 2),
    ("offline_spec_dec", PDD_DIR / "offline" / "moe_spec_dec.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 1),
    ("offline_prefix", PDD_DIR / "offline" / "moe_prefix_caching.sh", {}, 2, 2),
    ("online_dense", PDD_DIR / "online" / "dense_model_basic_online.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 1),
    ("online_moe", PDD_DIR / "online" / "moe_model_basic_online.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 1),
    ("online_thinking", PDD_DIR / "online" / "thinking_mode_basic_online.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 2),
    ("online_spec_dec", PDD_DIR / "online" / "moe_spec_dec_online.sh", {"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"}, 1, 1),
    ("online_prefix", PDD_DIR / "online" / "moe_prefix_caching_online.sh", {}, 2, 2),
)


@pytest.mark.parametrize(("case_id", "script_path", "overrides", "expected_requests", "expected_transfers"), SCRIPT_CASES)
def test_pdd_example_script_emits_valid_metrics(case_id, script_path, overrides, expected_requests, expected_transfers, tmp_path):
    run_id = f"pytest_pdd_{case_id}"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHON_BIN": sys.executable,
            "METRICS_OUTPUT_DIR": str(tmp_path / "metrics"),
            "RUN_ID": run_id,
            "QPS": "1000.0",
            "MAX_TOKENS_IN_BATCH": "128",
            "LONG_PREFILL_TOKEN_THRESHOLD": "4",
        }
    )
    env.update(overrides)

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    combined_output = result.stdout + result.stderr
    assert "Disaggregated architecture support is currently being optimized" not in combined_output
    assert result.returncode == 0, combined_output
    assert "completed successfully" in result.stdout

    run_dirs = [path for path in (tmp_path / "metrics").rglob(run_id) if path.is_dir()]
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]

    request_metrics_path = run_dir / "request_metrics.csv"
    system_metrics_path = run_dir / "system_metrics.json"
    assert request_metrics_path.is_file()
    assert system_metrics_path.is_file()

    with request_metrics_path.open(newline="") as request_metrics_file:
        request_rows = list(csv.DictReader(request_metrics_file))
    with system_metrics_path.open() as system_metrics_file:
        system_metrics = json.load(system_metrics_file)

    assert len(request_rows) == expected_requests
    metadata = system_metrics["simulation_metadata"]
    assert metadata["total_requests"] == expected_requests
    assert metadata["completed_requests"] == expected_requests
    assert system_metrics["kv_cache_transfer_statistics"]["total_transfers"] == expected_transfers
    assert system_metrics["kv_cache_transfer_total_bytes"] > 0
    assert system_metrics["kv_cache_transfer_statistics"]["total_transfer_time_ms"] > 0.0
    for request_row in request_rows:
        assert float(request_row["request_e2e_time"]) > 0.0
        assert float(request_row["ttft"]) > 0.0
        assert float(request_row["transfer_kv_cache"]) > 0.0
