"""64 CPU-only serving scenarios guarding unchanged simulator execution.

These are structural regression checks using dummy timings, not numerical
correlation evidence. Run explicitly when changing capture/replay integration.
"""

from itertools import product
from pathlib import Path
import os
import subprocess
import sys

import pytest


CASES = list(product(
    ("meta-llama/Llama-2-7b-hf", "Phi-tiny-MoE-instruct"),
    ("offline", "online"), (32, 128), (1, 4), (1.0, 8.0), (False, True),
))


@pytest.mark.parametrize("model,mode,length,count,qps,graphs", CASES)
def test_capture_integration_preserves_serving_matrix(tmp_path, model, mode, length, count, qps, graphs):
    command = [sys.executable, "-m", "frontier.main",
        "--simulation_mode", mode, "--sys_arch", "co-location",
        "--cc_backend_config_type", "analytical", "--cluster_config_num_replicas", "1",
        "--replica_config_model_name", model, "--replica_config_device", "mi355x",
        "--replica_config_network_device", "mi355x_ubb",
        "--replica_config_attn_tensor_parallel_size", "2",
        "--replica_config_moe_tensor_parallel_size", "2",
        "--replica_config_moe_expert_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_scheduler_config_type", "sglang",
        "--sglang_scheduler_config_block_size", "1",
        "--sglang_scheduler_config_num_blocks_mode", "explicit",
        "--sglang_scheduler_config_num_blocks", "4096",
        "--sglang_scheduler_config_max_tokens_in_batch", "256",
        "--sglang_scheduler_config_enable_chunked_prefill",
        "--decode_cuda_graph_mode", "full_decode_only" if graphs else "none",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(count),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(length),
        "--fixed_request_length_generator_config_decode_tokens", "3",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(qps),
        "--metrics_config_output_dir", str(tmp_path),
        "--metrics_config_run_id", "regression",
        "--metrics_config_write_metrics", "--metrics_config_store_request_metrics",
        "--no-metrics_config_store_plots", "--no-metrics_config_write_json_trace",
        "--no-metrics_config_enable_chrome_trace"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                            env={**os.environ, "FRONTIER_LOG_LEVEL": "ERROR"},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    metrics = list(tmp_path.rglob("request_metrics.csv"))
    assert len(metrics) == 1
    import pandas as pd
    requests = pd.read_csv(metrics[0])
    assert len(requests) == count
    assert (requests["ttft"] >= 0).all()
