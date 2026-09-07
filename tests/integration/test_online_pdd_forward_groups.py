"""Run the online PDD workload that strands unequal attention-DP lane histories."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("num_requests", [2, 8])
def test_online_pdd_unequal_lane_histories_complete(tmp_path, num_requests):
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ, "PYTHON_BIN": sys.executable, "PYTHONPATH": str(root),
        "PYTHONDONTWRITEBYTECODE": "1", "WANDB_DISABLED": "true", "VIDUR_DISABLE_WANDB": "1",
        "TMPDIR": str(tmp_path), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "ENABLE_DUMMY_MODE": "true", "DUMMY_EXEC_TIME_MS": "1", "NUM_REQUESTS": str(num_requests),
        "PREFILL_TOKENS": "96", "DECODE_TOKENS": "8", "QPS": "1000",
        "DECODE_CUDA_GRAPH_MODE": "none", "ENABLE_CHUNKED_PREFILL": "true",
        "LONG_PREFILL_TOKEN_THRESHOLD": "64", "METRICS_OUTPUT_DIR": str(tmp_path / "metrics"),
        "RUN_ID": "forward_groups",
    }
    command = [
        "bash", str(root / "examples/architecture/pdd/online/moe_model_basic_online.sh"), "--",
        "--metrics_config_cache_dir", str(tmp_path / "cache"), "--replica_config_attn_dp", "2",
    ]
    for role in ("prefill", "decode"):
        prefix = f"--cluster_config_{role}_replica_config_"
        command.extend([prefix + "attn_tensor_parallel_size", "4", prefix + "moe_tensor_parallel_size", "1",
                        prefix + "moe_expert_parallel_size", "8"])
    (tmp_path / "invocation.json").write_text(json.dumps({
        "cwd": str(root), "command": command,
        "environment_overrides": {key: value for key, value in env.items() if os.environ.get(key) != value},
    }, indent=2))
    log = tmp_path / "run.log"
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, log.read_text()[-6000:]
    paths = list((tmp_path / "metrics").rglob("request_metrics.csv"))
    assert len(paths) == 1
    with paths[0].open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == num_requests
    assert len({row["Request Id"] for row in rows}) == num_requests
    assert all(float(row["request_num_prefill_tokens"]) == 96 for row in rows)
    assert all(float(row["request_num_decode_tokens"]) == 8 for row in rows)
