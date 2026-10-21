"""Exercise the public graph policy through the unmodified main subprocess.

This regression stops before arrival to isolate initialization. Full request
execution remains a separate acceptance check in run_step5_measured_main.py.
"""

import json
import os
import subprocess

from frontier.types import ClusterType
from tests.integration.run_step5_measured_main import REPO, command, write_read_cost
from tests.integration.step5_measured_compute_fixture import generate


def test_public_full_decode_only_initializes_real_main(tmp_path):
    fixture = generate(tmp_path / "fixture", device="h800", gpus_per_role=16)
    case = tmp_path / "online"
    case.mkdir()
    (case / "requests.csv").write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens\n10.0,6144,1024\n"
    )
    read_cost = tmp_path / "read_cost.json"
    write_read_cost(read_cost)
    argv = command("online", case, fixture, read_cost) + ["--time_limit", "1"]
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "WANDB_DISABLED": "true",
           "VIDUR_DISABLE_WANDB": "1"}
    with (case / "main.log").open("w") as log:
        result = subprocess.run(argv, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, (case / "main.log").read_text()
    [config_path] = (case / "metrics").rglob("config.json")
    config = json.loads(config_path.read_text())
    assert config["decode_cuda_graph_mode"] == "full_decode_only"
    assert config["cluster_config"]["cluster_scheduler_config"] == {
        "policy_version": "v0.28.0", "name": "vllm_load_balancing",
    }
    system = json.loads((config_path.parent / "system_metrics.json").read_text())
    assert system["simulation_metadata"]["completed_requests"] == 0
    assert set(fixture["bundles"][str(ClusterType.DECODE.value)]) == {"eager", "kernel_only"}
