"""Exercise full 43-layer V4 serving with explicitly synthetic dummy timing.

Eager PP1 co-location includes the existing request-free idle DP forwards.
Source SP geometry is derived from scheduled/dummy rows; destination expert
ratios are synthetic balanced inputs. Native routing, latency, overlap,
workspace capacity, CUDA graphs and PD remain unqualified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


SOURCE_FILES = (
    "frontier/attention/deepseek_v4.py", "frontier/attention/families.py",
    "frontier/config/quantization_manager.py", "frontier/entities/execution_time.py",
    "frontier/entities/stage_execution_time.py", "frontier/entities/time_components.py",
    "frontier/execution_time_predictor/deepseek_v4.py",
    "frontier/execution_time_predictor/sklearn_execution_time_predictor.py",
    "frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py",
    "frontier/metrics/constants.py", "frontier/metrics/ep_wave_metrics.py",
    "frontier/metrics/metrics_store.py", "frontier/metrics/op_trace_utils.py",
    "frontier/operators/families.py",
    "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py",
    "frontier/scheduler/utils/ep_wave.py", "frontier/scheduler/utils/expert_parallel.py",
    "frontier/scheduler/utils/layer_workload.py",
    "tests/integration/run_dsv4_structural.py",
)


def source_identity():
    return {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            for path in SOURCE_FILES}


def validate_artifacts(run_dir, dp):
    system = json.loads((run_dir / "system_metrics.json").read_text())
    metadata = system["simulation_metadata"]
    require(metadata["total_requests"] == metadata["completed_requests"] == dp,
            "all and only submitted requests must complete")
    with (run_dir / "request_metrics.csv").open(newline="") as stream:
        requests = list(csv.DictReader(stream))
    require(len(requests) == dp, "request metrics coverage differs from input")
    require({int(r["request_num_prefill_tokens"]) for r in requests} == {17}, "wrong prompts")
    require({int(r["request_num_decode_tokens"]) for r in requests} == {2}, "wrong decode length")
    request_ids = {str(row["Request Id"]) for row in requests}
    stage_rows = list(read_jsonl(run_dir / "frontier_stage_batch_ledger.jsonl"))
    stage_by_batch = {row["batch_id"]: row for row in stage_rows}
    stage_ops = defaultdict(Counter)
    phase_ops = defaultdict(Counter)
    attention_layers = defaultdict(dict)
    counts = Counter()
    aliases = {"attn_pre_proj", "attn_post_proj", "attn_prefill", "attn_decode",
               "attn_rope", "attn_kv_cache_save", "add_attn_residual", "add_ffn_residual",
               "input_layernorm", "post_attention_layernorm", "moe_all_to_all"}
    for row in read_jsonl(run_dir / "op_traces.jsonl"):
        if "name" not in row:
            continue
        counts["trace_rows"] += 1
        name, meta = row["name"], row.get("meta", {})
        require(name not in aliases, f"legacy alias would duplicate physical work: {name}")
        require(row["duration_ms"] > 0, "trace contains non-positive work")
        if "phase" in meta:
            key = (row["batch_id"], meta["ep_id"], meta["global_layer_ids"][0], meta["phase"])
            phase_ops[key][name] += row["duration_ms"]
            require(meta["native_routed_assignments"] == "UNKNOWN", "synthetic routes claimed native")
            require(meta["destination_provenance"] == "configured synthetic expert ratios", "missing ratio origin")
            require(set(meta["request_ids"]) <= request_ids, "virtual destination identities credited as requests")
        elif row["batch_id"] in stage_by_batch:
            stage_ops[row["batch_id"]][name] += row["duration_ms"]
            if name == "dsv4_sparse_attention":
                layers = meta.get("global_layer_ids", [row["layer_id"]])
                for layer in layers:
                    require(layer not in attention_layers[row["batch_id"]], "attention double ownership")
                    attention_layers[row["batch_id"]][layer] = meta["attention_variant_id"]
    cpu_names = {"schedule_time": "schedule", "sampler_e2e_time": "sampler_e2e",
                 "prepare_inputs_e2e_time": "prepare_inputs_e2e",
                 "process_model_outputs_time": "process_model_outputs", "ray_comm_time": "ray_comm_time"}
    scheduled = defaultdict(list)
    max_stage_error = 0.0
    for row in stage_rows:
        require(set(row["request_ids"]) <= request_ids, "idle forward credited as real request")
        components = row["execution_time"]["component_ledger_ms"]
        error = abs(sum(components.values()) - row["execution_time"]["total_time_ms"])
        max_stage_error = max(max_stage_error, error)
        require(error < 1e-6, "stage component ledger is not conserved")
        observed = stage_ops[row["batch_id"]]
        for name, duration in components.items():
            require(abs(observed[cpu_names.get(name, name)] - duration) < 1e-6,
                    f"stage trace/component disagreement: batch={row['batch_id']} {name}")
        require(set(attention_layers[row["batch_id"]]) == set(range(43)), "stage lacks43physicalattentionlayers")
        require(Counter(attention_layers[row["batch_id"]].values()) ==
                {"uncompressed": 2, "csa": 21, "hca": 20}, "wrong43-layer variants")
        require(abs(components["dsv4_mhc_first_pre"] - 0.01) < 1e-9, "first mHC counted incorrectly")
        require(abs(components["dsv4_mhc_ffn_post_attn_pre"] - 0.42) < 1e-9, "cross-layer mHC counted incorrectly")
        for name in cpu_names:
            require(abs(components[name] - 0.01) < 1e-9, "stage CPU overhead must occur once")
        require(abs(components["dsv4_embedding"] - 0.01) < 1e-9 and
                abs(components["dsv4_embedding_allreduce"] - 0.01) < 1e-9,
                "local embedding and its reduction must each occur once")
        for request_id, tokens in zip(row["request_ids"], row["request_num_tokens"]):
            scheduled[request_id].append((row["stage_start_ts"], tokens))
    require(set(scheduled) == request_ids, "stage/request identity coverage differs")
    for values in scheduled.values():
        require([tokens for _, tokens in sorted(values)] == [16, 1, 1],
                "expected16+1chunkedprefill then one decode forward for2sampled tokens")
    waves = defaultdict(list)
    max_phase_error = 0.0
    for row in read_jsonl(run_dir / "frontier_ep_wave_lane_ledger.jsonl"):
        layer = row["global_layer_ids"][0]
        waves[(row["wave_start_time_s"], layer)].append(row)
        local = (row["source_rows"] + 7) // 8
        valid = max(0, min(local, row["source_rows"] - row["source_tp_rank"] * local))
        require(row["physical_local_rows"] == local and row["valid_local_rows"] == valid,
                "source SP geometry differs from its batch")
        idle = row["source_forward_kind"] == "request-free dummy"
        require(row["request_valid_local_rows"] == (0 if idle else valid), "dummy request credit")
        require(not idle or (row["source_rows"] == 1 and not row["request_ids"]), "wrong idle contract")
        counts["idle_ep_lanes" if idle else "real_ep_lanes"] += 1
        require(row["routing_operator_variant"] == ("hash" if layer < 3 else "score"), "wrong routing variant")
        for phase, details in row["phases"].items():
            predicted = details["operators_ms"]
            observed = phase_ops.pop((row["batch_id"], row["ep_id"], layer, phase), Counter())
            error = abs(sum(predicted.values()) + details.get("kernel_gap_ms", 0) - details["duration_ms"])
            max_phase_error = max(max_phase_error, error)
            require(error < 1e-6, "EP phase ledger is not conserved")
            for name, duration in predicted.items():
                require(abs(observed[name] - duration) < 1e-6, "EP trace/phase disagreement")
        order = ("pre_dispatch", "dispatch", "routed_compute", "combine", "post_combine")
        for previous, following in zip(order, order[1:]):
            end = row["phases"][previous]["start_time_s"] + row["phases"][previous]["duration_ms"] / 1000
            require(end <= row["phases"][following]["start_time_s"] + 1e-12,
                    "structural serial phase boundaries overlap")
        pre = row["phases"]["pre_dispatch"]["operators_ms"]
        for name in ("dsv4_mhc_attn_post_ffn_pre", "share_expert_up_proj", "share_expert_act",
                     "share_expert_down_proj", "moe_gating_linear", "moe_gating_routing_topk"):
            require(abs(pre[name] - 0.01) < 1e-9, "missing/duplicated source FFN physical scope")
        post = row["phases"]["post_combine"]["operators_ms"]
        require(("dsv4_mhc_final_post" in post) == (layer == 42), "final mHC placement")
        require(("dsv4_lm_head" in post) == (layer == 42 and not idle), "selected LM-row ownership")
    require(not phase_ops, "EP trace has work without a phase-ledger owner")
    for rows in waves.values():
        require(len(rows) == dp * 8, "EP wave is missing source-rank participants")
        physical = sum(row["physical_local_rows"] for row in rows)
        require(sum(row["synthetic_destination_assignments"] for row in rows) == physical * 6,
                "synthetic destination assignments do not conserve physical source rows")
    require(counts["real_ep_lanes"] > 0 and counts["idle_ep_lanes"] > 0, "real/idle scope not exercised")
    return {**counts, "stage_rows": len(stage_rows), "ep_waves": len(waves),
            "max_stage_conservation_error_ms": max_stage_error,
            "max_phase_conservation_error_ms": max_phase_error}


def command(case, output_root, dp):
    return [sys.executable, "-m", "frontier.main",
        "--simulation_mode", case, "--sys_arch", "co-location",
        "--log_level", "warning", "--decode_cuda_graph_mode", "none",
        "--cc_backend_config_type", "analytical",
        "--replica_config_model_name", "deepseek-ai/DeepSeek-V4-Flash",
        "--replica_config_device", "h800", "--replica_config_network_device", "h800_dgx",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_config_attn_tensor_parallel_size", "8",
        "--replica_config_attn_dp", str(dp),
        "--replica_config_moe_tensor_parallel_size", "1",
        "--replica_config_moe_expert_parallel_size", str(8 * dp),
        "--replica_config_total_expert_num", "256", "--replica_config_router_topk", "6",
        "--replica_config_moe_routing_distribution_type", "balanced",
        "--replica_config_attention_backend", "FLASHMLA_SPARSE_DSV4",
        "--replica_config_kv_cache_dtype", "fp8",
        "--replica_config_indexer_kv_cache_dtype", "fp8",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_batch_size_cap", "1",
        "--vllm_v1_scheduler_config_block_size", "256",
        "--vllm_v1_scheduler_config_num_blocks", "256",
        "--vllm_v1_scheduler_config_num_blocks_mode", "explicit",
        "--vllm_v1_scheduler_config_max_model_len", "256",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "16",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "16",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "0.01",
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(output_root / "requests.csv"),
        "--trace_request_generator_config_max_tokens", "256", "--seed", "42",
        "--metrics_config_output_dir", str(output_root / "metrics"),
        "--metrics_config_cache_dir", str(output_root / "cache"),
        "--metrics_config_run_id", f"dsv4_structural_{case}",
        "--metrics_config_write_metrics", "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics", "--metrics_config_store_operation_metrics",
        "--metrics_config_store_frontier_stage_batch_ledger",
        "--metrics_config_store_frontier_stage_batch_ledger_summary",
        "--metrics_config_enable_op_level_tracing", "--metrics_config_enable_per_layer_expansion",
        "--metrics_config_num_requests_to_trace_per_layer", str(dp),
        "--no-metrics_config_store_plots", "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace"]


def run_case(case, output_root, dp):
    root = output_root / case
    root.mkdir(parents=True, exist_ok=False)
    with (root / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "arrived_at", "num_prefill_tokens", "num_decode_tokens"))
        writer.writeheader()
        for _ in range(dp):
            writer.writerow(dict(arrived_at=0.0, num_prefill_tokens=17,
                                 num_decode_tokens=2))
    argv = command(case, root, dp)
    (root / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    before = source_identity()
    with (root / "run.log").open("w") as stream:
        completed = subprocess.run(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=900)
    (root / "source_identity.json").write_text(json.dumps({
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "c17_sources_before": before,
        "c17_sources_after": source_identity(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    require(before == source_identity(), "C17 source files changed during the run")
    require(completed.returncode == 0, f"simulator failed: {root / 'run.log'}")
    runs = list((root / "metrics").rglob("config.json"))
    require(len(runs) == 1, "one metrics run is required")
    run_dir = runs[0].parent
    validation = validate_artifacts(run_dir, dp)
    return {"case": case, "logical_gpus": dp * 8, "run_dir": str(run_dir),
            "requests_completed": dp, **validation,
            "evidence_level": "CPU dummy structural; native assignments and latency UNKNOWN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dp", type=int, choices=(2, 4), default=4)
    parser.add_argument("--case", choices=("offline", "online", "both"), default="both")
    args = parser.parse_args()
    cases = ("offline", "online") if args.case == "both" else (args.case,)
    results = [run_case(case, args.output_root, args.dp) for case in cases]
    (args.output_root / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
