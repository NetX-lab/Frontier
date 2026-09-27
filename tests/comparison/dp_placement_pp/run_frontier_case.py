#!/usr/bin/env python3
"""Run the DP-placement calibration case on Frontier (plan §18.6, `simulator-run`).

The case's vLLM engine settings are the source of every setting that has a
Frontier counterpart, so a semantic difference can only come from the mapping
in `case_config`, which the semantic-alignment table reads back from the
constructed configuration (`effective_settings.json`). The model name is an
engine setting. Expert count and router top-k come from that model's config.
Frontier-only settings (the execution-time source, device labels, the KV block
count, the MoE routing distribution and its seed, and the analytical backend)
come from the case's Frontier settings file.

The execution-time source is either a fixed dummy time
(`dummy_execution_time_ms`), for timing-independent checks, or a directory of
profiled operator tables (`profiling_dir`, relative to the settings file) that
holds `linear_op.csv`, `attention.csv` and `moe.csv`. With profiled tables the
predictor's prediction ranges are the engine's token budget and model length.

Three runs share the trace and differ in one thing each:

``vllm_load_balancing``
    The policy under test.
``completion_reporting_control``
    The same policy with the reporting it had before schedule-time reports
    (plan §18.11 P9-04 controls).
``round_robin``
    A load-blind policy, for reference only.

Each run executes in its own process, so Frontier's request ids restart at 0
and equal the trace row index that `request_ids.json` maps to case request ids.
Observation reuses `run_case` of `tests/integration/test_vllm_dp_placement_runtime.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_TEST = REPO_ROOT / "tests" / "integration" / "test_vllm_dp_placement_runtime.py"

RUNS = {
    "vllm_load_balancing": dict(policy_name="vllm_load_balancing"),
    "completion_reporting_control": dict(
        policy_name="vllm_load_balancing", completion_reporting_control=True
    ),
    "round_robin": dict(policy_name="round_robin"),
}


def execution_time_predictor_config(engine: dict, frontier: dict, frontier_dir: Path):
    from frontier.config import RandomForrestExecutionTimePredictorConfig

    if ("profiling_dir" in frontier) == ("dummy_execution_time_ms" in frontier):
        raise ValueError(
            "the Frontier settings name exactly one execution-time source: "
            "profiling_dir or dummy_execution_time_ms"
        )
    if "dummy_execution_time_ms" in frontier:
        return RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True,
            dummy_execution_time_ms=frontier["dummy_execution_time_ms"],
        )
    profiling_dir = (frontier_dir / frontier["profiling_dir"]).resolve()
    return RandomForrestExecutionTimePredictorConfig(
        linear_op_input_file=str(profiling_dir / "linear_op.csv"),
        atten_input_file=str(profiling_dir / "attention.csv"),
        moe_input_file=str(profiling_dir / "moe.csv"),
        prediction_max_prefill_chunk_size=engine["max_num_batched_tokens"],
        prediction_max_tokens_per_request=engine["max_model_len"],
    )


def case_config(
    engine: dict, frontier: dict, frontier_dir: Path, trace_path: Path, root: Path, policy
):
    from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
    from frontier.config.parallel_semantics import resolve_frontier_parallelism_mapping
    from frontier.config import (
        ClusterConfig,
        MetricsConfig,
        ReplicaConfig,
        SimulationConfig,
        TraceRequestGeneratorConfig,
        VllmV1SchedulerConfig,
    )

    if frontier["cc_backend"] != "analytical":
        raise ValueError(f"unsupported cc_backend {frontier['cc_backend']!r}")
    if not engine["enforce_eager"]:
        raise ValueError(
            "unmapped setting enforce_eager=false: vLLM 0.10.2's non-eager V1 "
            "default is PIECEWISE, which this harness does not map"
        )
    mapping = resolve_frontier_parallelism_mapping(
        model_profile="moe",
        tensor_parallel_size=engine["tensor_parallel_size"],
        num_replicas=1,
        enable_expert_parallel=engine["enable_expert_parallel"],
        attn_dp=engine["data_parallel_size"],
    )
    replica = ReplicaConfig(
        model_name=engine["model_name"],
        device=frontier["device"],
        network_device=frontier["network_device"],
        num_pipeline_stages=engine["pipeline_parallel_size"],
        attn_tensor_parallel_size=mapping.attn_tensor_parallel_size,
        attn_dp=mapping.attn_dp,
        moe_tensor_parallel_size=mapping.moe_tensor_parallel_size,
        moe_expert_parallel_size=mapping.moe_expert_parallel_size,
        moe_routing_distribution_type=frontier["moe_routing_distribution_type"],
        moe_routing_seed=frontier["moe_routing_seed"],
    )
    scheduler = VllmV1SchedulerConfig(
        num_blocks=frontier["num_blocks"],
        num_blocks_mode="explicit",
        block_size=engine["block_size"],
        batch_size_cap=engine["max_num_seqs"],
        max_tokens_in_batch=engine["max_num_batched_tokens"],
        enable_chunked_prefill=engine["enable_chunked_prefill"],
        enable_prefix_caching=engine["enable_prefix_caching"],
        scheduling_policy="fcfs",
    )
    cluster = ClusterConfig(
        replica_config=replica,
        replica_scheduler_config=scheduler,
        cluster_scheduler_config=policy(),
        execution_time_predictor_config=execution_time_predictor_config(
            engine, frontier, frontier_dir
        ),
        cc_backend_config=AnalyticalCCBackendConfig(),
    )
    return SimulationConfig(
        simulation_mode="online",
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=cluster,
        metrics_config=MetricsConfig(
            output_dir=str(root / "metrics"),
            cache_dir=str(root / "cache"),
            run_id="dp_pp_case",
            write_metrics=True,
            store_request_metrics=True,
            store_batch_metrics=False,
            store_operation_metrics=False,
            store_utilization_metrics=False,
            store_plots=False,
            enable_chrome_trace=False,
            write_json_trace=False,
        ),
        # vLLM rejects a request longer than max_model_len; the generator's
        # default 4096 cap would instead shorten the long prompt silently.
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_path), max_tokens=engine["max_model_len"]
        ),
    )


def effective_settings(config) -> dict:
    cluster = config.cluster_config
    replica = cluster.replica_config
    scheduler = cluster.replica_scheduler_config
    predictor = cluster.execution_time_predictor_config
    if predictor.enable_dummy_mode:
        execution_time = {"dummy_execution_time_ms": predictor.dummy_execution_time_ms}
    else:
        execution_time = {
            "linear_op_input_file": predictor.linear_op_input_file,
            "atten_input_file": predictor.atten_input_file,
            "moe_input_file": predictor.moe_input_file,
            "prediction_max_prefill_chunk_size": predictor.prediction_max_prefill_chunk_size,
            "prediction_max_tokens_per_request": predictor.prediction_max_tokens_per_request,
        }
    return {
        "attn_dp": replica.attn_dp,
        "attn_tensor_parallel_size": replica.attn_tensor_parallel_size,
        "moe_tensor_parallel_size": replica.moe_tensor_parallel_size,
        "moe_expert_parallel_size": replica.moe_expert_parallel_size,
        "num_pipeline_stages": replica.num_pipeline_stages,
        "num_replicas": cluster.num_replicas,
        "model_name": replica.model_name,
        "num_layers": replica.model_config.num_layers,
        "total_expert_num": replica.total_expert_num,
        "router_topk": replica.router_topk,
        "moe_routing_distribution_type": str(replica.moe_routing_distribution_type),
        "cluster_scheduler": str(cluster.cluster_scheduler_config.get_type()),
        "replica_scheduler": str(scheduler.get_type()),
        "max_tokens_in_batch": scheduler.max_tokens_in_batch,
        "batch_size_cap": scheduler.batch_size_cap,
        "block_size": scheduler.block_size,
        "num_blocks": scheduler.num_blocks,
        "enable_chunked_prefill": scheduler.enable_chunked_prefill,
        "enable_prefix_caching": scheduler.enable_prefix_caching,
        "long_prefill_token_threshold": scheduler.long_prefill_token_threshold,
        "scheduling_policy": scheduler.scheduling_policy,
        "decode_cuda_graph_mode": config.decode_cuda_graph_mode,
        **execution_time,
        "trace_max_tokens": config.request_generator_config.max_tokens,
    }


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("dp_placement_runtime", RUNTIME_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(args: argparse.Namespace) -> None:
    engine = json.loads(args.engine_config.read_text())
    frontier = json.loads(args.frontier_config.read_text())
    request_ids = json.loads((args.trace_dir / "request_ids.json").read_text())
    root = args.output_dir / args.run
    root.mkdir(parents=True, exist_ok=True)
    trace_path = args.trace_dir / "trace.csv"
    built = {}

    def build(policy):
        built["config"] = case_config(
            engine, frontier, args.frontier_config.parent, trace_path, root, policy
        )
        return built["config"]

    evidence = load_runtime_module().run_case(
        root, build_config=build, attn_dp=engine["data_parallel_size"], **RUNS[args.run]
    )
    rows = request_ids["rows"]
    # A load-balancing run selects once per placement, in the same order; a
    # round-robin run makes no selection.
    for selection, frontier_id in zip(evidence["selections"], evidence["placement_request_ids"]):
        selection["request_id"] = rows[frontier_id]["request_id"]
    evidence["placement_by_request_id"] = {
        rows[frontier_id]["request_id"]: lane
        for frontier_id, lane in zip(evidence["placement_request_ids"], evidence["placements"])
    }
    (root / "evidence.json").write_text(json.dumps(evidence, indent=1))
    (root / "effective_settings.json").write_text(
        json.dumps(effective_settings(built["config"]), indent=1)
    )


def summarize(output_dir: Path, request_ids: dict) -> dict:
    bursts: dict[str, list[dict]] = {}
    for row in request_ids["rows"]:
        if row["segment"] == "burst":
            bursts.setdefault(row["burst"], []).append(row)
    evidence = {run: json.loads((output_dir / run / "evidence.json").read_text()) for run in RUNS}
    summary = {"runs": {
        run: {key: evidence[run][key] for key in ("completed_requests", "num_requests", "tokens_conserved")}
        for run in RUNS
    }, "bursts": {}}
    for name, burst in bursts.items():
        probe_id = burst[-1]["request_id"] if burst[-1]["probe"] else None
        burst_start = burst[0]["arrived_at"]
        summary["bursts"][name] = {"probe_request_id": probe_id, "runs": {}}
        for run in RUNS:
            placement = evidence[run]["placement_by_request_id"]
            selections = {selection["request_id"]: selection for selection in evidence[run]["selections"]}
            burst_ends = [
                record["time"] for record in evidence[run]["records"]
                if record["kind"] == "end" and record["time"] >= burst_start
            ]
            run_summary = {
                "burst_placements": [placement[row["request_id"]] for row in burst],
                "first_completion_after_burst_s": (
                    min(burst_ends) - burst_start if burst_ends else None
                ),
            }
            if probe_id is not None:
                run_summary["probe_placement"] = placement[probe_id]
                run_summary["probe_selection"] = selections.get(probe_id)
            summary["bursts"][name]["runs"][run] = run_summary
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--frontier-config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", choices=sorted(RUNS))
    args = parser.parse_args(argv)

    if args.run is not None:
        run_one(args)
        return 0

    for run in RUNS:
        child_args = list(argv) if argv is not None else sys.argv[1:]
        command = [sys.executable, str(Path(__file__).resolve()), *child_args, "--run", run]
        result = subprocess.run(
            command,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{run}.log").write_text(result.stdout)
        if result.returncode != 0:
            print(result.stdout[-5000:])
            return result.returncode
    request_ids = json.loads((args.trace_dir / "request_ids.json").read_text())
    summary = summarize(args.output_dir, request_ids)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
