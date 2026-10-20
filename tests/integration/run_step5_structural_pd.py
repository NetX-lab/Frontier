"""Run and validate the dummy Step-5-Preview structural PDD path.

One PREFILL and one DECODE replica each hold 32 logical GPUs: attention TP8 x
DP4 and MoE TP1 x EP32, PP1. This is CPU-only structural evidence. It does not
qualify native NIXL payloads or descriptors, the native 25-token D recompute
tail, measured latency, CUDA graphs, or H800 execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from math import ceil
import os
from pathlib import Path
import subprocess
import sys
import time

from tests.integration.run_glm53_structural_pd import read_jsonl, require, source_identity


REPO = Path(__file__).resolve().parents[2]
MODES = ("offline", "online")
ATTENTION_DP = 4
BLOCK_SIZE = 16
# Prompts below, on and above the 8-token sparse region, the 16-token page,
# and the 512-token window plus one page; then the canonical 4105-token prompt
# and its 4096/4104 neighbors. Twelve requests give three per attention lane.
SCENARIOS = {
    "single": ((4105,), 9),
    "boundaries": ((7, 8, 9, 15, 16, 17, 527, 528, 529, 4096, 4104, 4105), 3),
}
ONLINE_ARRIVAL_GAP_S = 0.01
# Per-rank page bytes at TP8, block 16, BF16; see test_step5_structural_pd.py.
SWA_LAYERS, SPARSE_LAYERS = 69, 23
RAW_PAGE_BYTES, SUMMARY_PAGE_BYTES, OPEN_REGION_PAGE_BYTES = 12288, 512, 2060
SWA_TRANSFER_PAGES = 33
NETWORK_LATENCY_MS = 0.1
NETWORK_BYTES_PER_MS = 100.0e9 / 8 / 1e3
LIMITS = [
    "Dummy CPU structural simulation at 0.25 ms per operator; no timing fidelity claim.",
    "Transfer bytes are modeled per-TP-rank pages, not native NIXL wire bytes or descriptors.",
    "P computes the whole prompt and D recomputes one token; native P stops 25 tokens early.",
    "The open-region accumulator page and the partial region-summary page are sent; "
    "native vLLM resets the open region and sends full pages only.",
    "D reserves full-prompt pages per window group, as allocated before window retirement.",
    "Block size 16 differs from the 64-token MONOLITHIC Step structural block.",
]


def expected_rank_bytes(prompt_tokens: int) -> int:
    pages = ceil(prompt_tokens / BLOCK_SIZE)
    swa_pages = min(pages, SWA_TRANSFER_PAGES)
    open_region = int(prompt_tokens % 8 != 0)
    return (
        SWA_LAYERS * swa_pages * RAW_PAGE_BYTES
        + SPARSE_LAYERS * pages * (RAW_PAGE_BYTES + SUMMARY_PAGE_BYTES)
        + SPARSE_LAYERS * open_region * OPEN_REGION_PAGE_BYTES
    )


def expected_transfer_ms(prompt_tokens: int) -> float:
    return NETWORK_LATENCY_MS + expected_rank_bytes(prompt_tokens) / NETWORK_BYTES_PER_MS


def command(mode: str, scenario: str, case_root: Path) -> list[str]:
    roles = []
    for role in ("prefill", "decode"):
        prefix = f"--cluster_config_{role}_replica_config_"
        roles += [
            prefix + "num_pipeline_stages", "1",
            prefix + "attn_tensor_parallel_size", "8",
            prefix + "moe_tensor_parallel_size", "1",
            prefix + "moe_expert_parallel_size", "32",
            prefix + "total_expert_num", "352",
            prefix + "router_topk", "8",
            prefix + "device", "h800",
        ]
    # The stage ledger also writes about 15 MB of EP wave rows per batch, so
    # only the single-request case keeps it for chunk and step counts.
    ledger_flag = (
        "--metrics_config_store_frontier_stage_batch_ledger" if scenario == "single"
        else "--no-metrics_config_store_frontier_stage_batch_ledger"
    )
    args = [
        "--simulation_mode", mode,
        "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--log_level", "warning",
        "--decode_cuda_graph_mode", "none",
        "--cc_backend_config_type", "analytical",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        *roles,
        "--replica_config_model_name", "stepfun-ai/Step-5-Preview",
        "--replica_config_network_device", "h800_dgx",
        "--replica_config_attn_dp", str(ATTENTION_DP),
        "--replica_config_attention_backend", "STEP5_DSA",
        "--replica_config_kv_cache_dtype", "auto",
        "--replica_config_moe_routing_distribution_type", "balanced",
        "--replica_config_moe_routing_seed", "42",
        "--no-speculative_decoding_config_enabled",
        "--replica_scheduler_config_type", "vllm_v1",
        "--no-vllm_v1_scheduler_config_enable_prefix_caching",
        "--vllm_v1_scheduler_config_batch_size_cap", "1",
        "--vllm_v1_scheduler_config_block_size", str(BLOCK_SIZE),
        "--vllm_v1_scheduler_config_num_blocks", "0",
        "--vllm_v1_scheduler_config_num_blocks_mode", "memory_planner",
        "--no-vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling",
        "--vllm_v1_scheduler_config_max_model_len", "8192",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "512",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "512",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "0.25",
        "--analytical_kv_cache_transfer_config_kv_connector", "nixl",
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps", "100.0",
        "--analytical_kv_cache_transfer_config_network_latency_ms", str(NETWORK_LATENCY_MS),
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(case_root / "requests.csv"),
        "--trace_request_generator_config_max_tokens", "8192",
        "--seed", "42",
        "--metrics_config_output_dir", str(case_root / "metrics"),
        "--metrics_config_cache_dir", str(case_root / "cache"),
        "--metrics_config_run_id", f"step5_structural_pd_{scenario}_{mode}",
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        ledger_flag,
        "--no-metrics_config_enable_op_level_tracing",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]
    return [sys.executable, "-m", "frontier.main", *args]


def validate_config(config: dict, mode: str) -> None:
    require(config["simulation_mode"] == mode, "wrong simulation mode")
    require(config["sys_arch"] == "pd-disaggregation", "wrong architecture")
    require(not config["enable_parallel_clusters"], "parallel PDD is unsupported")
    require(config["kv_cache_transfer_config"]["kv_connector"] == "nixl", "wrong connector")
    cluster = config["cluster_config"]
    require(cluster["prefill_cluster_num_replicas"] == 1, "wrong P replica count")
    require(cluster["decode_cluster_num_replicas"] == 1, "wrong D replica count")
    for role in ("prefill_replica_config", "decode_replica_config"):
        replica = cluster[role]
        for field, value in (
            ("num_pipeline_stages", 1), ("attn_tensor_parallel_size", 8),
            ("attn_dp", ATTENTION_DP), ("moe_tensor_parallel_size", 1),
            ("moe_expert_parallel_size", 32), ("world_size", 32),
        ):
            require(replica[field] == value, f"wrong {role} {field}")
        require(replica["model_config"]["num_layers"] == 92, "Step must retain all 92 layers")
    scheduler = cluster["replica_scheduler_config"]
    require(scheduler["block_size"] == BLOCK_SIZE, "wrong block size")
    require(scheduler["num_blocks_mode"] == "memory_planner", "allocator must be planned")
    require(not scheduler["enable_prefix_caching"], "prefix caching must stay off")


def validate_artifacts(run_dir: Path, mode: str, scenario: str) -> dict:
    prompts, decode_tokens = SCENARIOS[scenario]
    count = len(prompts)
    validate_config(json.loads((run_dir / "config.json").read_text()), mode)
    system = json.loads((run_dir / "system_metrics.json").read_text())
    metadata = system["simulation_metadata"]
    require(metadata["total_requests"] == count, "wrong request count")
    require(metadata["completed_requests"] == count, "requests did not complete")

    transfers = read_jsonl(run_dir / "frontier_transfer_ledger.jsonl")
    require(len(transfers) == count, "one transfer per request is required")
    source_lanes = Counter()
    target_lanes = Counter()
    for row in transfers:
        request_id = int(row["request_ids"][0])
        prompt_tokens = prompts[request_id]
        require(row["status"] == "completed" and row["target_bound"], "incomplete transfer")
        require((row["source_cluster"], row["target_cluster"]) == ("PREFILL", "DECODE"),
                "wrong transfer roles")
        require(row["bytes"] == expected_rank_bytes(prompt_tokens),
                f"request {request_id} transfer bytes differ from the hand computation")
        require(abs(row["duration_ms"] - expected_transfer_ms(prompt_tokens)) < 1e-6,
                f"request {request_id} transfer time is wrong")
        # Round robin over one replica's DP lanes places request i on lane i % 4.
        require(row["source_replica_local_id"] == request_id % ATTENTION_DP,
                f"request {request_id} ran on the wrong P lane")
        source_lanes[row["source_replica_local_id"]] += 1
        target_lanes[(row["target_replica_id"], row["target_replica_local_id"])] += 1
    require(len({row["target_replica_id"] for row in transfers}) == 1, "one D replica expected")
    lane_counts = sorted(target_lanes.values())
    require(len(target_lanes) == min(count, ATTENTION_DP) and lane_counts[-1] - lane_counts[0] <= 1,
            "D round robin must spread transfers evenly over its lanes")
    total_bytes = sum(row["bytes"] for row in transfers)
    stats = system["kv_cache_transfer_statistics"]
    require(stats["total_transfers"] == count, "wrong transfer total")
    require(stats["total_data_transferred_bytes"] == total_bytes, "wrong transfer bytes")

    with (run_dir / "request_metrics.csv").open(newline="") as stream:
        requests = {int(row["Request Id"]): row for row in csv.DictReader(stream)}
    require(sorted(requests) == list(range(count)), "missing request metrics")
    for request_id, row in requests.items():
        require(int(row["request_num_prefill_tokens"]) == prompts[request_id], "wrong prompt")
        require(int(row["request_num_decode_tokens"]) == decode_tokens, "wrong decode length")
        arrival_ms = 0.0 if mode == "offline" else request_id * ONLINE_ARRIVAL_GAP_S * 1e3
        prefill_ready_ms = arrival_ms + float(row["prefill_e2e_time"])
        require(float(row["transfer_kv_cache_request_start_ts"]) >= prefill_ready_ms - 1e-6,
                "transfer cannot start before P finishes")

    result = {
        "mode": mode, "scenario": scenario, "requests_completed": count,
        "transfer_bytes_per_request": {str(prompts[int(row["request_ids"][0])]): row["bytes"]
                                       for row in transfers},
        "transfer_bytes_total": total_bytes,
        "source_lane_counts": {str(lane): n for lane, n in sorted(source_lanes.items())},
        "target_lane_counts": {f"{replica}:{lane}": n
                               for (replica, lane), n in sorted(target_lanes.items())},
        "max_request_e2e_ms": max(float(row["request_e2e_time"]) for row in requests.values()),
    }
    if scenario == "single":
        stage_rows = read_jsonl(run_dir / "frontier_stage_batch_ledger.jsonl")
        prefill_chunks = [row["request_num_prefill_tokens"][0]
                          for row in stage_rows if row["cluster_type"] == "PREFILL"]
        decode_steps = [row["request_num_tokens"][0]
                        for row in stage_rows if row["cluster_type"] == "DECODE"]
        require(prefill_chunks == [512] * 8 + [9], "4105 tokens must run as 8 x 512 + 9 chunks")
        # D recomputes the last prompt token, then runs eight decode inputs.
        require(decode_steps == [1] * decode_tokens, "D must run one step per output token")
        result.update(prefill_chunks=len(prefill_chunks), decode_steps=len(decode_steps))
    return result


def run_case(mode: str, scenario: str, output_root: Path) -> dict:
    case_root = output_root / f"{scenario}_{mode}"
    case_root.mkdir(parents=True)
    prompts, decode_tokens = SCENARIOS[scenario]
    with (case_root / "requests.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arrived_at", "num_prefill_tokens", "num_decode_tokens"))
        for request_id, prompt_tokens in enumerate(prompts):
            arrival_s = 0.0 if mode == "offline" else request_id * ONLINE_ARRIVAL_GAP_S
            writer.writerow((arrival_s, prompt_tokens, decode_tokens))
    before = source_identity(Path(__file__))
    argv = command(mode, scenario, case_root)
    (case_root / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    started = time.monotonic()
    with (case_root / "run.log").open("w") as stream:
        process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(process.pid, 0)
    wall_s = time.monotonic() - started
    require(os.waitstatus_to_exitcode(status) == 0,
            f"{scenario} {mode} failed; see {case_root / 'run.log'}")
    require(source_identity(Path(__file__)) == before, "source changed during structural run")
    [config_path] = list((case_root / "metrics").rglob("config.json"))
    result = validate_artifacts(config_path.parent, mode, scenario)
    result.update(wall_time_s=round(wall_s, 1), peak_rss_mib=round(usage.ru_maxrss / 1024, 1),
                  source_identity=before, limits=LIMITS)
    (case_root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    modes = MODES if args.mode == "all" else (args.mode,)
    scenarios = tuple(SCENARIOS) if args.scenario == "all" else (args.scenario,)
    args.output_root.mkdir(parents=True, exist_ok=False)
    results = [run_case(mode, scenario, args.output_root)
               for scenario in scenarios for mode in modes]
    (args.output_root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
