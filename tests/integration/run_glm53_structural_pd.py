"""Run and validate the canonical dummy GLM-5.3 structural PDD path.

This is CPU-only structural evidence. It does not qualify native NIXL payloads,
cross-process ordering, measured latency, CUDA graphs, or H800 execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
CASES = ("offline", "online")
# A D reservation holds ceil(prompt / 64) of the lane's three blocks. The tail
# needs two, and the earlier request on its lane leaves one, so the tail waits
# for that request to finish.
PROMPT_TOKENS = (128, 65, 65, 65, 65)
REQUEST_COUNT = len(PROMPT_TOKENS)
DECODE_TOKENS = 3
PAGE_BYTES_PER_WORKER_LANE = 5_928_192


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_identity() -> dict:
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO)
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def command(case: str, output_root: Path) -> list[str]:
    run_id = f"glm53_structural_pd_{case}"
    trace_path = output_root / "requests.csv"
    args = [
        "--simulation_mode", case,
        "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--log_level", "warning",
        "--decode_cuda_graph_mode", "none",
        "--cc_backend_config_type", "analytical",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "8",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "32",
        "--cluster_config_prefill_replica_config_total_expert_num", "256",
        "--cluster_config_prefill_replica_config_router_topk", "8",
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_decode_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_replica_config_attn_tensor_parallel_size", "8",
        "--cluster_config_decode_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_replica_config_moe_expert_parallel_size", "32",
        "--cluster_config_decode_replica_config_total_expert_num", "256",
        "--cluster_config_decode_replica_config_router_topk", "8",
        "--cluster_config_decode_replica_config_device", "h800",
        "--replica_config_model_name", "zai-org/GLM-5.3",
        "--replica_config_network_device", "h800_dgx",
        "--replica_config_attn_dp", "4",
        "--replica_config_moe_routing_distribution_type", "balanced",
        "--replica_config_moe_routing_seed", "42",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_batch_size_cap", "1",
        "--vllm_v1_scheduler_config_block_size", "64",
        "--vllm_v1_scheduler_config_num_blocks", "3",
        "--vllm_v1_scheduler_config_num_blocks_mode", "explicit",
        "--vllm_v1_scheduler_config_max_model_len", "256",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "64",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "64",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "0.01",
        "--analytical_kv_cache_transfer_config_kv_connector", "nixl",
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps", "100.0",
        "--analytical_kv_cache_transfer_config_network_latency_ms", "0.1",
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(trace_path),
        "--trace_request_generator_config_max_tokens", "256",
        "--seed", "42",
        "--metrics_config_output_dir", str(output_root / "metrics"),
        "--metrics_config_cache_dir", str(output_root / "cache"),
        "--metrics_config_run_id", run_id,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics",
        "--metrics_config_store_token_completion_metrics",
        "--metrics_config_store_frontier_stage_batch_ledger",
        "--metrics_config_store_frontier_stage_batch_ledger_summary",
        "--metrics_config_enable_op_level_tracing",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]
    return [sys.executable, "-m", "frontier.main", *args]


def validate_config(config: dict, case: str) -> None:
    require(config["simulation_mode"] == case, "wrong simulation mode")
    require(config["sys_arch"] == "pd-disaggregation", "wrong architecture")
    require(not config["enable_parallel_clusters"], "parallel PDD is unsupported")
    require(config["decode_cuda_graph_mode"] == "none", "graph mode must be none")
    require(config["kv_cache_transfer_config"]["kv_connector"] == "nixl", "wrong connector")
    cluster = config["cluster_config"]
    require(cluster["prefill_cluster_num_replicas"] == 1, "wrong P replica count")
    require(cluster["decode_cluster_num_replicas"] == 1, "wrong D replica count")
    for role in ("prefill_replica_config", "decode_replica_config"):
        replica = cluster[role]
        require(replica["num_pipeline_stages"] == 1, f"wrong {role} PP")
        require(replica["attn_tensor_parallel_size"] == 8, f"wrong {role} attention TP")
        require(replica["attn_dp"] == 4, f"wrong {role} attention DP")
        require(replica["moe_tensor_parallel_size"] == 1, f"wrong {role} MoE TP")
        require(replica["moe_expert_parallel_size"] == 32, f"wrong {role} MoE EP")
        require(replica["world_size"] == 32, f"wrong {role} logical GPU count")
        model = replica["model_config"]
        require(model["num_layers"] == 78, "GLM must retain all 78 layers")
        require(set(map(int, model["moe_layers_enum"].split(","))) == set(range(3, 78)),
                "GLM must retain 3 dense and 75 MoE layers")
    scheduler = cluster["replica_scheduler_config"]
    require(scheduler["block_size"] == 64, "structural path requires block size 64")
    require(scheduler["num_blocks"] == 3, "near-full path requires three blocks")
    require(scheduler["num_blocks_mode"] == "explicit", "allocator must be explicit")


def find_tail_capacity_owner(transfers: list[dict], tail_request_id: int) -> int:
    """Return the earlier request that owns the tail's selected D lane."""
    by_request = {int(row["request_ids"][0]): row for row in transfers}
    require(tail_request_id in by_request, "tail transfer is missing")
    tail = by_request[tail_request_id]
    tail_lane = (tail["target_replica_id"], tail["target_replica_local_id"])
    owners = [
        request_id
        for request_id, row in by_request.items()
        if request_id != tail_request_id
        and (row["target_replica_id"], row["target_replica_local_id"])
        == tail_lane
        and row["end_ts_s"] <= tail["start_ts_s"]
    ]
    require(
        len(owners) == 1,
        "tail must wait behind exactly one completed transfer on its selected D lane",
    )
    return owners[0]


def validate_artifacts(run_dir: Path, case: str) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    validate_config(config, case)
    system = json.loads((run_dir / "system_metrics.json").read_text())
    metadata = system["simulation_metadata"]
    require(metadata["total_requests"] == REQUEST_COUNT, "wrong request count")
    require(metadata["completed_requests"] == REQUEST_COUNT, "requests did not complete")
    transfers = read_jsonl(run_dir / "frontier_transfer_ledger.jsonl")
    require(len(transfers) == REQUEST_COUNT, "one transfer per request is required")
    for row in transfers:
        require(row["status"] == "completed" and row["target_bound"], "incomplete transfer")
        require(row["source_cluster"] == "PREFILL" and row["target_cluster"] == "DECODE",
                "wrong transfer roles")
        request_id = row["request_ids"][0]
        expected_pages = (PROMPT_TOKENS[request_id] + 63) // 64
        require(row["bytes"] == expected_pages * PAGE_BYTES_PER_WORKER_LANE,
                "payload must contain source-backed prompt pages only")
        require(row["start_ts_s"] <= row["end_ts_s"], "transfer timestamps are reversed")
    transfer_stats = system["kv_cache_transfer_statistics"]
    require(transfer_stats["total_transfers"] == REQUEST_COUNT, "wrong transfer total")
    expected_bytes = sum((tokens + 63) // 64 for tokens in PROMPT_TOKENS) * (
        PAGE_BYTES_PER_WORKER_LANE
    )
    require(transfer_stats["total_data_transferred_bytes"] == expected_bytes,
            "wrong transfer bytes")
    with (run_dir / "request_metrics.csv").open(newline="") as stream:
        requests = list(csv.DictReader(stream))
    require(len(requests) == REQUEST_COUNT, "missing request metrics")
    require(sorted(int(row["request_num_prefill_tokens"]) for row in requests) ==
            sorted(PROMPT_TOKENS), "wrong prompt lengths")
    require({int(row["request_num_decode_tokens"]) for row in requests} == {DECODE_TOKENS},
            "wrong decode lengths")
    requests_by_id = {int(row["Request Id"]): row for row in requests}
    capacity_owner_id = find_tail_capacity_owner(transfers, tail_request_id=4)
    capacity_owner_arrival_ms = 0.0 if case == "offline" else float(capacity_owner_id)
    capacity_owner_completion_ms = capacity_owner_arrival_ms + float(
        requests_by_id[capacity_owner_id]["request_e2e_time"]
    )
    tail_transfer_start_ms = float(
        requests_by_id[4]["transfer_kv_cache_request_start_ts"]
    )
    tail_prefill_ready_ms = float(requests_by_id[4]["prefill_e2e_time"])
    if case == "online":
        tail_prefill_ready_ms += 4.0
    require(
        tail_prefill_ready_ms < tail_transfer_start_ms,
        "tail must finish P before waiting for D capacity",
    )
    require(
        tail_transfer_start_ms >= capacity_owner_completion_ms,
        "tail transfer must wait until its actual D-lane owner completes",
    )
    stage_path = run_dir / "frontier_stage_batch_ledger.jsonl"
    stage_rows = read_jsonl(stage_path)
    require(stage_rows, "full stage ledger is required")
    require({row["cluster_type"] for row in stage_rows} == {"PREFILL", "DECODE"},
            "both P and D stage ledgers are required")
    require(all(row["execution_time"]["component_ledger_ms"] for row in stage_rows),
            "stage component ledgers must be populated")
    for row in stage_rows:
        ledger = row["execution_time"]["component_ledger_ms"]
        require(ledger["attn_dsa_indexer"] == 0.21,
                "stage must contain 21 full-index DSA layers")
        require(ledger["attn_dsa_mla"] == 0.78,
                "stage must contain 78 DSA MLA layers")
        require(ledger["attention_prefill_execution_time"] == 0.0,
                "DSA stage cannot charge generic prefill attention")
        require(ledger["attention_kv_cache_save_execution_time"] == 0.0,
                "DSA stage cannot charge generic KV-save attention")
        require(ledger["mlp_layer_up_proj_execution_time"] == 0.03,
                "stage must contain three dense FFN layers")
    head_prefill_chunks = [
        row["request_num_prefill_tokens"][0]
        for row in stage_rows
        if row["cluster_type"] == "PREFILL" and row["request_ids"] == ["0"]
    ]
    require(head_prefill_chunks == [64, 64],
            "128-token head request must execute as two real prefill chunks")

    transfer_traces = []
    indexer_layers = set()
    mla_layers = set()
    dense_layers = set()
    op_trace_count = 0
    with (run_dir / "op_traces.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            name = row.get("name")
            if name is None:
                continue
            op_trace_count += 1
            meta = row.get("meta", {})
            if name == "kv_cache_transfer":
                transfer_traces.append(row)
            elif name == "attn_dsa_indexer":
                indexer_layers.update(meta.get("global_layer_ids", []))
            elif name == "attn_dsa_mla":
                mla_layers.update(meta.get("global_layer_ids", []))
            elif name == "mlp_up_proj":
                dense_layers.update(meta.get("global_layer_ids", []))
    require(len(transfer_traces) == REQUEST_COUNT,
            "op trace must contain one component-aware transfer per request")
    for row in transfer_traces:
        meta = row["meta"]
        request_id = int(meta["request_ids"][0])
        expected_pages = (PROMPT_TOKENS[request_id] + 63) // 64
        require(meta["payload_kind"] ==
                "structural_whole_prompt_pages_per_worker_lane",
                "op trace must label structural whole-page payloads")
        require(meta["byte_scope"] == "per_worker_lane_resident_pages",
                "op trace byte scope must remain per worker/lane")
        require(meta["source_prompt_common_blocks"] == expected_pages,
                "op trace source page count is wrong")
        require(meta["transfer_size_bytes"] ==
                expected_pages * PAGE_BYTES_PER_WORKER_LANE,
                "op trace component bytes do not match the transfer")
        require(meta["tensor_dtypes"] == {
                    "main_mla": "BF16", "indexer_k": "FP8", "indexer_scale": "FP32"},
                "op trace must preserve mixed DSA cache dtypes")
        require({item["component_id"] for item in meta["cache_components"]} ==
                {"main_mla", "indexer_k_and_scale"},
                "op trace must expose both canonical GLM cache components")
        require(meta["native_wire_bytes_known"] is False,
                "structural bytes cannot be labeled native wire bytes")
    require(indexer_layers == {0, 1, 2, *range(6, 75, 4)},
            "op trace must cover exactly the 21 full-index layers")
    require(mla_layers == set(range(78)),
            "op trace must cover MLA work for all 78 layers")
    require(dense_layers == {0, 1, 2},
            "op trace must keep dense FFN work on the first three layers")
    return {
        "case": case,
        "logical_gpu_count": 64,
        "requests_completed": REQUEST_COUNT,
        "transfer_count": len(transfers),
        "transfer_bytes": sum(row["bytes"] for row in transfers),
        "stage_rows": len(stage_rows),
        "op_trace_count": op_trace_count,
        "tail_target_capacity_owner_request_id": capacity_owner_id,
        "tail_target_capacity_owner_completion_ms": capacity_owner_completion_ms,
        "tail_capacity_wait_ms": tail_transfer_start_ms - tail_prefill_ready_ms,
        "limits": [
            "dummy CPU structural simulation",
            "whole-page per-worker/lane payload, not native wire or rank aggregate",
            "same-time D-ready then P-release is a simulator ordering approximation",
            "native NIXL lifecycle, delay, descriptors, and hardware behavior remain unqualified",
        ],
    }


def run_case(case: str, output_root: Path) -> dict:
    case_root = output_root / case
    case_root.mkdir(parents=True, exist_ok=False)
    with (case_root / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("arrived_at", "num_prefill_tokens", "num_decode_tokens"),
        )
        writer.writeheader()
        for request_id, prompt_tokens in enumerate(PROMPT_TOKENS):
            writer.writerow(
                {
                    "arrived_at": 0.0 if case == "offline" else request_id / 1000,
                    "num_prefill_tokens": prompt_tokens,
                    "num_decode_tokens": DECODE_TOKENS,
                }
            )
    before = source_identity()
    argv = command(case, case_root)
    (case_root / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    with (case_root / "run.log").open("w") as stream:
        result = subprocess.run(
            argv,
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    require(result.returncode == 0, f"{case} simulator failed; see {case_root / 'run.log'}")
    after = source_identity()
    require(before == after, "source changed during structural run")
    candidates = list((case_root / "metrics").rglob("config.json"))
    require(len(candidates) == 1, "expected exactly one metrics run directory")
    validation = validate_artifacts(candidates[0].parent, case)
    validation["source_identity"] = before
    (case_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    selected = CASES if args.case == "all" else (args.case,)
    args.output_root.mkdir(parents=True, exist_ok=False)
    results = [run_case(case, args.output_root) for case in selected]
    (args.output_root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
