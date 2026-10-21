"""Run the actual main CLI with explicitly synthetic CPU Step P16/D16 inputs.

The supplied profile tables come from step5_measured_compute_fixture.py. This
runner never patches simulator classes and establishes no native/accuracy gate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

from frontier.attention.memory import AttentionKVCacheSelection, get_attention_kv_transfer_payload
from frontier.attention.step5_memory import resolve_step5_kv_cache_capacity
from frontier.config import ReplicaConfig


REPO = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    "Synthetic CPU compute profiles and 3.5 ms READ cost; no native admission or accuracy claim.",
    "Fixture max_model_len=8192, not canonical 7168; actual run matches fixture capacity.",
    "One request per mode, not formal 128 requests, 16 warmups, or three repetitions.",
    "CPU overhead is explicitly skipped; collective_sim uses the built-in H800 topology.",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_profile(path):
    fixture = json.loads(path.read_text())
    for key, value in (("evidence_kind", "cpu_fixture"), ("native_admitted", False),
                       ("device", "h800"), ("attention_tp", 8), ("attention_dp", 2),
                       ("moe_tp", 1), ("moe_ep", 16), ("max_model_len", 8192),
                       ("block_size", 16), ("batch_width", 1)):
        require(fixture[key] == value, f"Unsupported fixture {key}: {fixture[key]}")
    return fixture


def write_read_cost(path):
    replica = ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, attn_dp=2, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16, total_expert_num=352, router_topk=8,
        attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )
    capacity = resolve_step5_kv_cache_capacity(
        replica.model_config, 16, AttentionKVCacheSelection.from_replica(replica, block_size=16),
    )
    payload = get_attention_kv_transfer_payload(
        replica.model_config, capacity, prompt_tokens=6128, tensor_parallel_size=8,
        target_reserved_blocks=4 * (6144 // 16),
    )
    document = dict(
        schema="nixl_read_cost.v1", model_architecture_profile="step5",
        profile_identity="CPU_CONTRACT_FIXTURE_NOT_NATIVE",
        source_device="h800", target_device="h800",
        source_attention_tp=8, target_attention_tp=8, duration_scope="read_to_gpu_ready",
        cache_capacity=asdict(capacity), measurements=[dict(
            num_prefill_tokens=6144, num_computed_tokens=6128,
            transfer_bytes_per_rank=payload.transfer_bytes_per_rank, time_ms=3.5,
        )],
    )
    write_json(path, document)
    return document


def command(mode, root, fixture, read_cost):
    args = [
        "--simulation_mode", mode, "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters", "--no-enable_thinking_mode",
        "--no-speculative_decoding_config_enabled", "--no-use_cuda_graph",
        "--decode_cuda_graph_mode", "full_decode_only", "--cudagraph_capture_sizes", "1",
        "--cc_backend_config_type", "collective_sim",
        "--collective_sim_cc_backend_config_scenario_profile", "h800_rail",
        "--collective_sim_cc_backend_config_cache_dir", str(root / "cc_cache"),
        "--cluster_scheduler_config_type", "vllm_load_balancing",
        "--vllm_load_balancing_cluster_scheduler_config_policy_version", "v0.28.0",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        "--replica_config_model_name", "stepfun-ai/Step-5-Preview",
        "--replica_config_device", "h800", "--replica_config_network_device", "h800_dgx",
        "--replica_config_attn_dp", "2", "--replica_config_attention_backend", "STEP5_DSA",
        "--replica_config_kv_cache_dtype", "auto",
        "--replica_config_moe_routing_distribution_type", "balanced",
        "--replica_config_moe_routing_seed", "42",
        "--replica_scheduler_config_type", "vllm_v1",
        "--no-vllm_v1_scheduler_config_enable_prefix_caching",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--vllm_v1_scheduler_config_batch_size_cap", "1",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_max_model_len", str(fixture["max_model_len"]),
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "8192",
        "--vllm_v1_scheduler_config_gpu_memory_utilization", "0.92",
        "--vllm_v1_scheduler_config_num_blocks", "0",
        "--vllm_v1_scheduler_config_num_blocks_mode", "memory_planner",
        "--no-vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling",
        "--analytical_kv_cache_transfer_config_kv_connector", "nixl",
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(root / "requests.csv"),
        "--trace_request_generator_config_max_tokens", "8192",
        "--seed", "42", "--log_level", "warning",
        "--metrics_config_output_dir", str(root / "metrics"),
        "--metrics_config_cache_dir", fixture["cache_dir"],
        "--metrics_config_run_id", f"step5_measured_cpu_main_{mode}",
        "--metrics_config_write_metrics", "--metrics_config_store_request_metrics",
        "--metrics_config_enable_metrics_ground_truth_trace",
        "--metrics_config_keep_individual_batch_metrics",
        "--no-metrics_config_store_frontier_stage_batch_ledger",
        "--metrics_config_store_frontier_stage_batch_ledger_summary",
        "--no-metrics_config_enable_op_level_tracing", "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace", "--no-metrics_config_write_json_trace",
    ]
    for role in ("prefill", "decode"):
        prefix = f"--cluster_config_{role}_replica_config_"
        for key, value in dict(num_pipeline_stages=1, attn_tensor_parallel_size=8,
                               moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
                               total_expert_num=352, router_topk=8, device="h800").items():
            args.extend((prefix + key, str(value)))
    options = {**fixture["predictor_config"], "nixl_read_profile": str(read_cost)}
    prefix = "random_forrest_execution_time_predictor_config_"
    for key, value in options.items():
        if isinstance(value, bool):
            args.append(f"--{'' if value else 'no-'}{prefix}{key}")
        else:
            args.extend((f"--{prefix}{key}", *map(str, value if isinstance(value, list) else [value])))
    return [sys.executable, "-B", "-m", "frontier.main", *args]


def validate(root, mode, document):
    [config_path] = list((root / "metrics").rglob("config.json"))
    run = config_path.parent
    config = json.loads(config_path.read_text())
    cluster = config["cluster_config"]
    require(config["simulation_mode"] == mode, "Wrong mode")
    require(config["decode_cuda_graph_mode"] == "full_decode_only", "Wrong graph mode")
    for role in ("prefill", "decode"):
        replica = cluster[f"{role}_replica_config"]
        require([replica[k] for k in ("attn_tensor_parallel_size", "attn_dp",
                "moe_tensor_parallel_size", "moe_expert_parallel_size", "world_size")]
                == [8, 2, 1, 16, 16], f"Wrong {role} topology")
    scheduler = cluster["replica_scheduler_config"]
    require(scheduler["max_model_len"] == 8192, "Run must match fixture capacity")
    require(not scheduler["enable_prefix_caching"] and not scheduler["enable_chunked_prefill"],
            "Prefix/chunked prefill must stay disabled")
    system = json.loads((run / "system_metrics.json").read_text())
    metadata = system["simulation_metadata"]
    require(metadata["total_requests"] == metadata["completed_requests"] == 1, "Incomplete run")
    with (run / "request_metrics.csv").open() as stream:
        [request] = list(csv.DictReader(stream))
    require(int(request["request_num_prefill_tokens"]) == 6144, "Wrong input count")
    require(int(request["request_num_decode_tokens"]) == 1024, "Wrong output count")
    [transfer] = [json.loads(line) for line in (run / "frontier_transfer_ledger.jsonl").read_text().splitlines()]
    require(transfer["status"] == "completed" and transfer["target_bound"], "Incomplete READ")
    require(transfer["bytes"] == document["measurements"][0]["transfer_bytes_per_rank"], "Wrong READ bytes")
    require(abs(transfer["duration_ms"] - 3.5) < 1e-6, "Wrong READ cost")
    return dict(mode=mode, status="PASS", evidence_kind="cpu_fixture", native_admitted=False,
                run_dir=str(run), request=request, transfer=transfer, system_metrics=system,
                limitations=LIMITATIONS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("offline", "online", "all"), default="all")
    args = parser.parse_args()
    fixture = read_profile(args.fixture_manifest)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    document = write_read_cost(root / "synthetic_read_cost.json")
    write_json(root / "environment.json", dict(python=sys.version, executable=sys.executable,
               platform=platform.platform(), task_id="D20-ACTUAL-MAIN", limitations=LIMITATIONS))
    results = []
    for mode in (("offline", "online") if args.mode == "all" else (args.mode,)):
        case = root / mode
        case.mkdir()
        (case / "requests.csv").write_text(
            "arrived_at,num_prefill_tokens,num_decode_tokens\n"
            f"{0.0 if mode == 'offline' else 0.25},6144,1024\n")
        argv = command(mode, case, fixture, root / "synthetic_read_cost.json")
        write_json(case / "command.json", argv)
        (case / "command.bash").write_text(shlex.join(argv) + "\n")
        env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "WANDB_DISABLED": "true",
               "VIDUR_DISABLE_WANDB": "1"}
        started = time.monotonic()
        with (case / "run.log").open("w") as stream:
            result = subprocess.run(argv, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write_json(case / "execution.json", dict(returncode=result.returncode,
                   wall_s=time.monotonic() - started))
        require(result.returncode == 0, f"Actual main failed; inspect {case / 'run.log'}")
        receipt = validate(case, mode, document)
        write_json(case / "validation.json", receipt)
        results.append(receipt)
    write_json(root / "summary.json", results)
    print(json.dumps([dict(mode=r["mode"], status=r["status"], run_dir=r["run_dir"]) for r in results]))


if __name__ == "__main__":
    main()
