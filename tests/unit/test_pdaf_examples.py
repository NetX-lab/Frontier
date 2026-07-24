"""Contract and smoke checks for the public pd-af-disaggregation examples."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "architecture" / "pd-af-disagg"

OFFLINE_CASES = (
    "offline/dense_model_basic.sh",
    "offline/moe_model_basic.sh",
    "offline/moe_model_ep.sh",
    "offline/dense_cuda_graph.sh",
    "offline/moe_cuda_graph.sh",
)
ONLINE_CASES = (
    "online/dense_model_basic_online.sh",
    "online/moe_model_basic_online.sh",
    "online/moe_model_ep_online.sh",
    "online/dense_cuda_graph_online.sh",
    "online/moe_cuda_graph_online.sh",
)
ALL_CASES = OFFLINE_CASES + ONLINE_CASES


def test_pdaf_example_surface_contains_offline_and_online_cases() -> None:
    expected = {*ALL_CASES, "run_all.sh"}
    actual = {
        path.relative_to(EXAMPLES_ROOT).as_posix()
        for path in EXAMPLES_ROOT.rglob("*.sh")
    }
    assert actual == expected


def _capture_cli(tmp_path: Path, relative: str) -> list[str]:
    capture_path = tmp_path / (relative.replace("/", "_") + ".argv")
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
        }
    )
    result = subprocess.run(
        ["bash", str(EXAMPLES_ROOT / relative)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return capture_path.read_text(encoding="utf-8").splitlines()


def _option_value(argv: list[str], option: str) -> str:
    index = argv.index(option)
    return argv[index + 1]


@pytest.mark.parametrize("relative", ALL_CASES)
def test_pdaf_example_scripts_emit_the_release_cli_contract(
    tmp_path: Path, relative: str
) -> None:
    argv = _capture_cli(tmp_path, relative)
    expected_mode = "online" if relative.startswith("online/") else "offline"

    assert argv[:2] == ["-m", "frontier.main"]
    assert _option_value(argv, "--simulation_mode") == expected_mode
    assert _option_value(argv, "--sys_arch") == "pd-af-disaggregation"
    assert "--no-enable_parallel_clusters" in argv
    assert _option_value(argv, "--cc_backend_config_type") == "analytical"
    assert _option_value(argv, "--m2n_transfer_config_type") == "analytical"
    assert "--metrics_config_write_metrics" in argv
    assert "--metrics_config_store_request_metrics" in argv

    forbidden_options = {
        "--enable_thinking_mode",
        "--replica_scheduler_config_enable_prefix_caching",
        "--speculative_decoding_config_enabled",
    }
    assert forbidden_options.isdisjoint(argv)


@pytest.mark.parametrize("relative", ALL_CASES)
def test_pdaf_example_scripts_are_executable_and_shell_valid(relative: str) -> None:
    script = EXAMPLES_ROOT / relative
    assert os.access(script, os.X_OK), relative
    result = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "relative",
    (
        "offline/dense_model_basic.sh",
        "offline/dense_cuda_graph.sh",
        "online/dense_model_basic_online.sh",
        "online/dense_cuda_graph_online.sh",
    ),
)
def test_pdaf_dense_recipes_use_dense_safe_defaults(
    tmp_path: Path, relative: str
) -> None:
    argv = _capture_cli(tmp_path, relative)
    assert _option_value(argv, "--replica_config_model_name") == (
        "llama2_7b_dense_example"
    )
    assert _option_value(
        argv, "--cluster_config_prefill_replica_config_router_topk"
    ) == "1"
    assert _option_value(
        argv, "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size"
    ) == "1"


@pytest.mark.parametrize(
    "relative",
    ("offline/moe_model_ep.sh", "online/moe_model_ep_online.sh"),
)
def test_pdaf_ep_recipes_exercise_ep_greater_than_one_by_default(
    tmp_path: Path, relative: str
) -> None:
    argv = _capture_cli(tmp_path, relative)
    assert _option_value(
        argv, "--cluster_config_prefill_replica_config_attn_tensor_parallel_size"
    ) == "2"
    assert _option_value(
        argv, "--cluster_config_prefill_replica_config_attn_data_parallel_size"
    ) == "1"
    assert _option_value(
        argv, "--cluster_config_prefill_replica_config_moe_tensor_parallel_size"
    ) == "1"
    assert _option_value(
        argv, "--cluster_config_prefill_replica_config_moe_expert_parallel_size"
    ) == "2"
    assert _option_value(
        argv,
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size",
    ) == "2"
    assert _option_value(
        argv,
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size",
    ) == "1"
    assert _option_value(
        argv, "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size"
    ) == "1"
    assert _option_value(
        argv, "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size"
    ) == "2"
    assert _option_value(
        argv, "--cluster_config_decode_attn_micro_batch_size"
    ) == "1"


@pytest.mark.parametrize(
    "relative",
    (
        "offline/dense_cuda_graph.sh",
        "offline/moe_cuda_graph.sh",
        "online/dense_cuda_graph_online.sh",
        "online/moe_cuda_graph_online.sh",
    ),
)
def test_pdaf_cuda_graph_recipes_enable_the_global_pdaf_mode(
    tmp_path: Path, relative: str
) -> None:
    argv = _capture_cli(tmp_path, relative)
    assert "--use_cuda_graph" in argv
    capture_index = argv.index("--cudagraph_capture_sizes")
    assert argv[capture_index + 1 : capture_index + 5] == ["8", "16", "32", "64"]
    assert "--decode_cuda_graph_mode" not in argv


def test_pdaf_run_all_contains_each_case_exactly_once() -> None:
    text = (EXAMPLES_ROOT / "run_all.sh").read_text(encoding="utf-8")
    match = re.search(r"CASES=\(\n(?P<body>.*?)\n\)", text, flags=re.DOTALL)
    assert match is not None
    cases = tuple(re.findall(r'^\s*"([^"]+\.sh)"\s*$', match.group("body"), re.MULTILINE))
    assert cases == ALL_CASES


@pytest.mark.parametrize(
    "relative",
    (
        "offline/moe_model_basic.sh",
        "online/dense_model_basic_online.sh",
        "offline/moe_cuda_graph.sh",
    ),
)
def test_pdaf_representative_examples_write_complete_metrics(
    tmp_path: Path, relative: str
) -> None:
    output_dir = tmp_path / "metrics"
    env = os.environ.copy()
    env.update(
        {
            "NUM_REQUESTS": "1",
            "PREFILL_TOKENS": "16",
            "DECODE_TOKENS": "4",
            "QPS": "1",
            "METRICS_OUTPUT_DIR": str(output_dir),
            "RUN_ID": "unit_smoke",
        }
    )
    script = EXAMPLES_ROOT / relative
    result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    metric_dirs = list(output_dir.rglob("request_metrics.csv"))
    assert len(metric_dirs) == 1
    request_metrics = metric_dirs[0]
    system_metrics = request_metrics.with_name("system_metrics.json")
    assert system_metrics.is_file()
    with request_metrics.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["Request Id"] == "0"
    payload = json.loads(system_metrics.read_text())
    assert payload["simulation_metadata"]["total_requests"] == 1
    assert payload["simulation_metadata"]["completed_requests"] == 1
    assert payload["kv_cache_transfer_statistics"]["total_transfers"] > 0
    assert payload["m2n_transfer_statistics"]["total_transfers"] > 0
