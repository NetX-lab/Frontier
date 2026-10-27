"""Run Frontier on a native Step P/D workload case and compare it with the native metrics.

`run` simulates the formal requests of the frozen trace once, on the topology and engine
limits of the native run definition, and summarizes them with the native metric
definitions. `compare` reports the error of each metric against `workload_metrics.py`
output. Diagnostic ledgers and traces stay off; a run produces request metrics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tests.integration.step5_pd.canonical_replay import formal_metrics

MODEL = "stepfun-ai/Step-5-Preview"
CASES = json.loads((Path(__file__).with_name("priority_cases.json")).read_text())
# Frontier device and network-device names for each native GPU type.
DEVICES = {"H200": ("h200", "h200_dgx"), "H800": ("h800", "h800_dgx")}
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")
STATISTICS = ("mean", "p50", "p90", "p95")


def capture_sizes(max_size: int) -> list[int]:
    """vLLM's default CUDA Graph capture sizes up to `max_cudagraph_capture_size`."""
    return [size for size in [1, 2, 4, *range(8, 256, 8), *range(256, max_size + 1, 16)]
            if size <= max_size]


def collective_args(network_device: str, tp: int, dp: int, measured: bool) -> list[str]:
    """Analytical collectives, or vLLM's allgather_reducescatter MoE path priced from the
    measured curves of this network device and TP/DP layout."""
    if not measured:
        return ["--cc_backend_config_type", "analytical"]
    curves = REPO / f"data/profiling/network/{network_device}/step5_tp{tp}_dp{dp}/summary.csv"
    if not curves.is_file():
        raise FileNotFoundError(f"No measured collective curves for {network_device} TP{tp}/DP{dp}: {curves}")
    return ["--cc_backend_config_type", "measured",
            "--measured_cc_backend_config_summary_csv_path", str(curves),
            "--replica_config_moe_all2all_backend", "allgather_reducescatter"]


def command(case: str, engine: dict, trace: Path, output_dir: Path, predictor_options: dict,
            measured_collectives: bool = False) -> list[str]:
    from frontier.config import ReplicaConfig

    topology = CASES[case]
    device, network_device = DEVICES[topology["gpu_type"]]
    tp, dp = engine["tensor_parallel_size"], topology["data_parallel_size"]
    if engine["pipeline_parallel_size"] != 1 or engine["enable_chunked_prefill"] or engine["enable_prefix_caching"]:
        raise ValueError("The Step workload runner supports PP1 without chunked prefill or prefix caching")
    if topology["replica"] * topology["gpu"] != 2 * tp * dp:
        raise ValueError(f"{case}: each role must hold TP x DP = {tp} x {dp} GPUs")
    model = ReplicaConfig(model_name=MODEL, device=device, network_device=network_device,
                          attn_tensor_parallel_size=tp, attn_dp=dp, moe_tensor_parallel_size=1,
                          moe_expert_parallel_size=tp * dp, attention_backend="STEP5_DSA").model_config
    compilation = engine["compilation_config"]
    args = [
        "--simulation_mode", "online", "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters", "--no-enable_thinking_mode",
        "--no-speculative_decoding_config_enabled", "--no-use_cuda_graph",
        "--decode_cuda_graph_mode", compilation["cudagraph_mode"].lower(),
        "--cudagraph_capture_sizes", *map(str, capture_sizes(compilation["max_cudagraph_capture_size"])),
        *collective_args(network_device, tp, dp, measured_collectives),
        "--cluster_scheduler_config_type", "vllm_load_balancing",
        "--vllm_load_balancing_cluster_scheduler_config_policy_version", "v0.28.0",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        "--replica_config_model_name", MODEL,
        "--replica_config_device", device, "--replica_config_network_device", network_device,
        "--replica_config_attn_dp", str(dp), "--replica_config_attention_backend", "STEP5_DSA",
        "--replica_config_kv_cache_dtype", engine.get("kv_cache_dtype", "auto"),
        # The native position_cyclic router gives every expert the same token count, which
        # is what `balanced` materializes; position_cyclic in Frontier is TP8/DP2-only.
        "--replica_config_moe_routing_distribution_type", "balanced",
        "--replica_config_moe_routing_seed", str(engine["seed"]),
        "--replica_scheduler_config_type", "vllm_v1",
        "--no-vllm_v1_scheduler_config_enable_prefix_caching",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--vllm_v1_scheduler_config_batch_size_cap", str(engine["max_num_seqs"]),
        "--vllm_v1_scheduler_config_block_size", str(engine["block_size"]),
        "--vllm_v1_scheduler_config_max_model_len", str(engine["max_model_len"]),
        "--vllm_v1_scheduler_config_max_tokens_in_batch", str(engine["max_num_batched_tokens"]),
        "--vllm_v1_scheduler_config_gpu_memory_utilization", str(engine["gpu_memory_utilization"]),
        "--vllm_v1_scheduler_config_num_blocks", "0",
        "--vllm_v1_scheduler_config_num_blocks_mode", "memory_planner",
        "--no-vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling",
        "--analytical_kv_cache_transfer_config_kv_connector", "nixl",
        # Native PREFILL and DECODE engines keep two batches in flight with vLLM's
        # async scheduling; DECODE engine steps post, find and report NIXL READs.
        "--analytical_kv_cache_transfer_config_nixl_read_completion", "engine_step",
        "--cluster_config_prefill_replica_scheduler_config_max_concurrent_batches", "2",
        "--cluster_config_decode_replica_scheduler_config_max_concurrent_batches", "2",
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(trace),
        "--trace_request_generator_config_max_tokens", str(engine["max_num_batched_tokens"]),
        "--seed", str(engine["seed"]), "--log_level", "warning",
        "--metrics_config_output_dir", str(output_dir / "metrics"),
        "--metrics_config_cache_dir", str(output_dir / "cache"),
        "--metrics_config_run_id", case,
        "--metrics_config_write_metrics", "--metrics_config_store_request_metrics",
        "--no-metrics_config_store_frontier_stage_batch_ledger",
        "--no-metrics_config_store_batch_metrics", "--no-metrics_config_store_utilization_metrics",
        "--no-metrics_config_enable_op_level_tracing", "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
    ]
    for role in ("prefill", "decode"):
        prefix = f"--cluster_config_{role}_replica_config_"
        for key, value in dict(num_pipeline_stages=1, attn_tensor_parallel_size=tp,
                               moe_tensor_parallel_size=1, moe_expert_parallel_size=tp * dp,
                               total_expert_num=model.num_experts, router_topk=model.num_experts_per_tok,
                               device=device).items():
            args.extend((prefix + key, str(value)))
    prefix = "random_forrest_execution_time_predictor_config_"
    for key, value in predictor_options.items():
        if isinstance(value, bool):
            args.append(f"--{'' if value else 'no-'}{prefix}{key}")
        else:
            args.extend((f"--{prefix}{key}", *map(str, value if isinstance(value, list) else [value])))
    return [sys.executable, "-B", "-m", "frontier.main", *args]


def summarize(run_dir: Path, trace: Path) -> dict:
    """Apply the native `formal_metrics` to Frontier's per-request times."""
    with trace.open() as stream:
        arrivals = list(csv.DictReader(stream))
    with (run_dir / "request_metrics.csv").open() as stream:
        requests = list(csv.DictReader(stream))
    if len(requests) != len(arrivals):
        raise ValueError(f"Frontier completed {len(requests)} of {len(arrivals)} requests")
    records = []
    for row in requests:
        source = arrivals[int(row["Request Id"])]
        start = float(source["arrived_at"])
        first = start + float(row["client_ttft"]) / 1000
        last = start + float(row["request_e2e_time"]) / 1000
        output = int(row["request_num_decode_tokens"])
        records.append(dict(status="completed", dispatch_monotonic=start, last_token_monotonic=last,
                            num_prefill_tokens=int(row["request_num_prefill_tokens"]),
                            num_decode_tokens=output, ttft_ms=1000 * (first - start),
                            e2e_ms=1000 * (last - start), tpot_ms=1000 * (last - first) / (output - 1)))
    return formal_metrics(records)


def native_reference(native: dict) -> dict:
    """Average each native metric over the formal repetitions."""
    repeats = native["repetitions"]
    reference = {field: {name: sum(r[field][name] for r in repeats) / len(repeats) for name in STATISTICS}
                 for field in METRICS}
    reference["total_thpt_tokens_s"] = native["mean_total_thpt_tokens_s"]
    reference["output_thpt_tokens_s"] = native["mean_output_thpt_tokens_s"]
    return reference


def compare(frontier: dict, native: dict, gate: float) -> dict:
    reference = native_reference(native)
    rows = []
    for field in METRICS:
        for name in STATISTICS:
            rows.append((f"{field}.{name}", frontier[field][name], reference[field][name]))
    for field in ("total_thpt_tokens_s", "output_thpt_tokens_s"):
        rows.append((field, frontier[field], reference[field]))
    table = []
    for metric, predicted, actual in rows:
        error = predicted - actual
        relative = error / actual if actual else None
        table.append(dict(metric=metric, frontier=predicted, native=actual, abs_error=error,
                          rel_error=relative,
                          status="UNDEFINED" if relative is None else ("PASS" if abs(relative) <= gate else "FAIL")))
    return dict(gate=gate, rows=table, status="PASS" if all(r["status"] == "PASS" for r in table) else "FAIL")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--case", choices=sorted(CASES), required=True)
    run.add_argument("--run-definition", type=Path, default=Path(__file__).with_name("canonical_run_h200.json"))
    run.add_argument("--trace", type=Path,
                     default=REPO / "data/processed_traces/canonical_fixed_6144_1024_req128_qps1/trace.csv")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--predictor-options", type=Path,
                     help="JSON object of execution-time predictor options; omit for dummy mode")
    run.add_argument("--measured-collectives", action="store_true",
                     help="Run vLLM's allgather_reducescatter MoE path and price collectives from "
                          "data/profiling/network/<network_device>/step5_tp<TP>_dp<DP>/summary.csv")
    check = commands.add_parser("compare")
    check.add_argument("--frontier", type=Path, required=True, help="frontier_summary.json from `run`")
    check.add_argument("--native", type=Path, required=True, help="workload_metrics.py output")
    check.add_argument("--gate", type=float, required=True, help="Relative error limit, e.g. 0.3")
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.action == "compare":
        result = compare(json.loads(args.frontier.read_text())["metrics"],
                         json.loads(args.native.read_text()), args.gate)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for row in result["rows"]:
            relative = "undefined" if row["rel_error"] is None else f"{100 * row['rel_error']:+.1f}%"
            print(f"{row['metric']:<22} frontier={row['frontier']:.3f} native={row['native']:.3f} "
                  f"rel={relative} {row['status']}")
        return 0 if result["status"] == "PASS" else 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    engine = json.loads(args.run_definition.read_text())["engine"]
    options = ({"enable_dummy_mode": True} if args.predictor_options is None
               else json.loads(args.predictor_options.read_text()))
    argv = command(args.case, engine, args.trace.resolve(), output_dir, options, args.measured_collectives)
    (output_dir / "command.json").write_text(json.dumps(argv, indent=1) + "\n")
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
           "WANDB_DISABLED": "true", "VIDUR_DISABLE_WANDB": "1"}
    started = time.monotonic()
    with (output_dir / "run.log").open("w") as log:
        returncode = subprocess.run(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    execution = dict(returncode=returncode, wall_s=time.monotonic() - started,
                     commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    if returncode == 0:
        [run_dir] = (output_dir / "metrics").rglob("request_metrics.csv")
        execution["metrics"] = summarize(run_dir.parent, args.trace.resolve())
    (output_dir / "frontier_summary.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps({key: execution[key] for key in ("returncode", "wall_s")}))
    return returncode


if __name__ == "__main__":
    sys.exit(main())
