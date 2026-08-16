"""Prove scheduler reachability for held-out cache-write feature tuples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

from frontier.config import SimulationConfig
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType
from frontier.utils.random import set_seeds


SCENARIOS = {
    "single_prefill_gap": {
        "expected_key": (8, 0, 1),
        "max_tokens_in_batch": 4096,
    },
    "default_batch_extrapolation": {
        # The per-request 4096-token serving cap includes one decode token,
        # leaving 4095 prefill tokens per request.
        "expected_key": (16380, 0, 4),
        "max_tokens_in_batch": 16384,
    },
    "batch4_seq3072": {
        "expected_key": (12288, 0, 4),
        "max_tokens_in_batch": 16384,
    },
    "batch4_seq3584": {
        "expected_key": (14336, 0, 4),
        "max_tokens_in_batch": 16384,
    },
    "batch4_seq3840": {
        "expected_key": (15360, 0, 4),
        "max_tokens_in_batch": 16384,
    },
    "batch4_seq4032": {
        "expected_key": (16128, 0, 4),
        "max_tokens_in_batch": 16384,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _cli_args(args: argparse.Namespace) -> list[str]:
    scenario = SCENARIOS[args.scenario]
    return [
        "--simulation_mode",
        "offline",
        "--sys_arch",
        "co-location",
        "--no-enable_parallel_clusters",
        "--cc_backend_config_type",
        "analytical",
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_device",
        "h800",
        "--replica_config_model_name",
        "llama2_7b_dense_example",
        "--replica_config_num_pipeline_stages",
        "1",
        "--replica_config_attn_tensor_parallel_size",
        "1",
        "--replica_config_attn_data_parallel_size",
        "1",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--decode_cuda_graph_mode",
        "none",
        "--vllm_v1_scheduler_config_max_tokens_in_batch",
        str(scenario["max_tokens_in_batch"]),
        "--vllm_v1_scheduler_config_batch_size_cap",
        "128",
        "--vllm_v1_scheduler_config_block_size",
        "16",
        "--vllm_v1_scheduler_config_num_blocks",
        "2048",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type",
        "trace_replay",
        "--trace_request_generator_config_trace_file",
        str(args.trace_file),
        "--trace_request_generator_config_max_tokens",
        "4096",
        "--offline_use_generated_request_arrivals",
        "--metrics_config_output_dir",
        str(args.metrics_dir),
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
        "0.01",
        "--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling",
    ]


def main() -> None:
    args = _parse_args()
    if not args.trace_file.is_file():
        raise ValueError(f"trace file does not exist: {args.trace_file}")
    if args.metrics_dir.exists():
        raise ValueError(f"metrics directory must be absent: {args.metrics_dir}")
    if args.output_json.exists():
        raise ValueError(f"output JSON must be absent: {args.output_json}")

    expected_key = tuple(SCENARIOS[args.scenario]["expected_key"])
    batch_records: list[dict[str, object]] = []
    original_create_batch = VLLMv1EngineReplicaScheduler._create_batch

    def captured_create_batch(scheduler, requests, num_tokens):
        batch = original_create_batch(scheduler, requests, num_tokens)
        if scheduler._cluster_type == ClusterType.MONOLITHIC:
            batch_records.append(
                {
                    "batch_id": int(batch.id),
                    "cache_save_key": [
                        int(batch.total_num_tokens),
                        0 if int(batch.num_decode_tokens) == 0 else None,
                        len(batch.requests),
                    ],
                    "num_prefill_tokens": int(batch.num_prefill_tokens),
                    "num_decode_tokens": int(batch.num_decode_tokens),
                    "scheduled_tokens": [int(value) for value in batch.num_tokens],
                    "request_ids": [int(request.id) for request in batch.requests],
                }
            )
        return batch

    sys.argv = [sys.argv[0], *_cli_args(args)]
    with mock.patch.object(
        VLLMv1EngineReplicaScheduler,
        "_create_batch",
        captured_create_batch,
    ):
        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        simulator = Simulator(config)
        simulator.run()

    matching_batches = [
        record
        for record in batch_records
        if tuple(record["cache_save_key"]) == expected_key
    ]
    if len(matching_batches) != 1:
        raise AssertionError(
            f"expected exactly one batch with cache-save key {expected_key}, "
            f"got {matching_batches}; all batches={batch_records}"
        )

    result = {
        "scenario": args.scenario,
        "trace_file": str(args.trace_file),
        "serving_max_tokens_per_request": 4096,
        "scheduler_max_tokens_in_batch": int(
            SCENARIOS[args.scenario]["max_tokens_in_batch"]
        ),
        "expected_cache_save_key": list(expected_key),
        "matching_batch": matching_batches[0],
        "all_batch_count": len(batch_records),
        "status": "PASS",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
