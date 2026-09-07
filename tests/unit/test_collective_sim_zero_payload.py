"""Exercise zero-byte collective payloads through the real runner boundary."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


BACKEND = Path(__file__).resolve().parents[2] / "frontier/cc_backend/backends/collective-sim"
if not (BACKEND / "sim/datacenter/htsim_ndp").is_file():
    pytest.skip("The optional collective-sim backend must be initialized and built.", allow_module_level=True)
sys.path.insert(0, str(BACKEND / "python"))
from collective_sim_core.predictor import predict_collective_time
from collective_sim_core.schema import Scenario


def scenario(payload, output):
    return Scenario.from_dict({
        "cluster": {"servers": 1, "gpus_per_server": 8},
        "parallelism": {"tp": 1, "cp": 1, "dp": 1, "ep": 8},
        "collective": {"kind": "alltoall", "tensor_bytes": payload,
                       "domain_dims": ["EP"], "placement_order": ["TP", "CP", "DP", "EP"],
                       "participant_ranks": list(range(8)), "exclude_intra_server": True,
                       "alltoall_model": "pairwise_steps"},
        "intra_server": {"model": "nvlink_analytic", "nvlink_one_way_bw_GBps": 450,
                         "nvlink_latency_us": 0.5, "nvlink_efficiency": 0.8},
        "runner": {"out_dir": str(output)},
    })


@pytest.mark.parametrize("payload", [0, 32768])
def test_real_predictor_retains_intra_server_latency(payload, tmp_path):
    result = predict_collective_time(scenario(payload, tmp_path), repo_root=BACKEND)
    expected_ms = (7 * 0.5 + (7 / 8 * payload) / (450e9 * 0.8) * 1e6) / 1000
    assert result["predicted_time_ms"] == pytest.approx(expected_ms)
    assert result["breakdown"]["network_ms"] == 0
    assert result["assumptions"]["estimated_bytes_per_rank"] == pytest.approx(7 / 8 * payload)


@pytest.mark.parametrize("payload,override,error", [
    (None, None, "missing required fields: ['tensor_bytes']"),
    (-1, None, "tensor_bytes must be >= 0"),
    (32768, 0, None),
])
def test_runner_distinguishes_missing_invalid_and_explicit_zero(payload, override, error, tmp_path):
    spec = scenario(32768, tmp_path).to_runner_spec()
    if payload is None:
        del spec["collective"]["tensor_bytes"]
    else:
        spec["collective"]["tensor_bytes"] = payload
    source = tmp_path / "scenario.json"
    source.write_text(json.dumps(spec))
    command = [sys.executable, str(BACKEND / "htsim_runner.py"), "--spec", str(source),
               "--out-dir", str(tmp_path / "runner")]
    if override is not None:
        command.extend(["--tensor-bytes", str(override)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if error:
        assert result.returncode == 2
        assert error in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.splitlines()[-1])
        assert payload["tensor_bytes"] == 0
