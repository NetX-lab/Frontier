"""Translates a parsed real sglang/vLLM bench_serving sweep into matching frontier.main invocations.

Known fidelity gaps (not solved by this translator, just carried forward honestly):
  - The real server uses CUDA graphs (up to batch size 128) and the engine's own attention/MoE
    kernels (aiter/triton for sglang, vLLM's own for vLLM); Frontier's MI355X MLA attention
    profiling used a portable torch-SDPA backend (not peak-tuned), and decode_cuda_graph_mode
    defaults to "none" here since no cuda-graph decode training data was collected for MLA on
    this device. Expect Frontier to under-predict peak throughput and over-predict decode
    latency to some degree as a result.
  - Real per-request scheduler admission (chunked-prefill sizing, token budget) is the real
    engine's own default policy, not discoverable from the client-side benchmark log; the values
    below are reasonable Frontier vLLM-v1-scheduler defaults for this workload shape. When the
    real run was itself served by vLLM, this is a much closer match than a "reasonable default"
    -- Frontier's scheduler model is literally based on vLLM v1 -- but it is still not a confirmed
    match against that specific server's actual --max-num-batched-tokens/--block-size flags,
    which aren't visible from the client-side log either.

Request generation defaults to closed_loop mode (Frontier's concurrency-capped request generator,
matching the real benchmark's request_rate=inf + max_concurrency=N client behavior exactly). The
older poisson mode (open-loop, QPS calibrated to the real run's observed throughput) is kept
available via request_mode="poisson" for direct before/after comparison -- it's known to inflate
TTFT under load once injected rate exceeds the simulated system's real capacity, since an open-loop
process has no backpressure the way a closed-loop client structurally does.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

from frontier.utils.output_paths import sanitize_output_component
from tools.validation.real_log_aggregator import AggregatedRun, load_and_aggregate
from tools.validation.real_log_parser import BenchmarkResult, BenchmarkRun


@dataclass
class Topology:
    """The real server's parallelism layout, translated to Frontier's 4-way TP/DP/EP/PP split."""

    device: str
    model_name: str
    attn_tensor_parallel_size: int
    attn_data_parallel_size: int = 1
    moe_tensor_parallel_size: int = 0  # defaults to attn_tensor_parallel_size below if unset
    moe_expert_parallel_size: int = 1
    num_pipeline_stages: int = 1
    num_replicas: int = 1

    def __post_init__(self) -> None:
        if not self.moe_tensor_parallel_size:
            self.moe_tensor_parallel_size = self.attn_tensor_parallel_size


@dataclass
class SimPoint:
    """One translated frontier.main invocation, matching one real benchmark-phase concurrency level."""

    concurrency: int
    run_id: str
    request_mode: str
    args: List[str]
    real: BenchmarkResult
    calibrated_qps: Optional[float] = None  # only set when request_mode == "poisson"

    @property
    def command(self) -> List[str]:
        return ["python3", "-m", "frontier.main", *self.args]

    def command_str(self) -> str:
        """Pretty-print as a multi-line shell command, one `--flag value` pair per line."""
        head = "python3 -m frontier.main"
        lines = []
        i = 0
        while i < len(self.args):
            if self.args[i].startswith("--") and i + 1 < len(self.args) and not self.args[i + 1].startswith("--"):
                lines.append(f"{self.args[i]} {self.args[i + 1]}")
                i += 2
            else:
                lines.append(self.args[i])
                i += 1
        return " \\\n  ".join([head, *lines])


def build_sim_point(
    real: BenchmarkResult,
    topology: Topology,
    output_dir: str,
    *,
    block_size: int = 32,
    cc_backend: str = "analytical",
    network_device: Optional[str] = None,
    decode_cuda_graph_mode: str = "none",
    enable_chunked_prefill: bool = True,
    max_tokens_in_batch: Optional[int] = None,
    atten_input_file: Optional[str] = None,
    request_mode: str = "closed_loop",
) -> SimPoint:
    if real.random_input_len is None or real.random_output_len is None:
        raise ValueError(f"Missing prefill/decode length on real record (concurrency={real.concurrency})")
    if request_mode not in ("closed_loop", "poisson"):
        raise ValueError(f"Unknown request_mode: {request_mode!r} (expected 'closed_loop' or 'poisson')")

    calibrated_qps: Optional[float] = None
    # Frontier's own run-id validation rejects "/" outright (and would silently normalize other
    # punctuation), so HF-style model names like "openai/gpt-oss-120b" must be sanitized here --
    # otherwise both the log path below and --metrics_config_run_id itself break.
    run_id = f"{sanitize_output_component(topology.model_name, 'model_name')}_conc{real.concurrency}"
    token_budget = max_tokens_in_batch or max(real.random_input_len, 8192)
    # The "dense" attention family's execution-time predictor (default type: random_forrest)
    # pre-computes a lookup grid capped at prediction_max_prefill_chunk_size (prefill chunk size)
    # and prediction_max_tokens_per_request (kv_cache_size, i.e. context length reached at the
    # last decode step) -- both default to 4096. That cap is auto-widened by max_tokens_in_batch
    # for the linear-op grid but NOT for the attention-prefill grid, so any workload with an
    # 8192-token prefill (like these) hits a bare KeyError deep in simulation once training
    # finishes, unless these are raised explicitly to cover the actual request shape.
    max_context_len = real.random_input_len + real.random_output_len

    args = [
        "--log_level", "warning",  # per-op INFO trace logging is significant overhead at real request counts
        "--simulation_mode", "online",
        "--sys_arch", "co-location",
        "--cc_backend_config_type", cc_backend,
        "--cluster_config_num_replicas", str(topology.num_replicas),
        "--replica_config_device", topology.device,
        "--replica_config_model_name", topology.model_name,
        "--replica_config_attn_tensor_parallel_size", str(topology.attn_tensor_parallel_size),
        "--replica_config_attn_data_parallel_size", str(topology.attn_data_parallel_size),
        "--replica_config_moe_tensor_parallel_size", str(topology.moe_tensor_parallel_size),
        "--replica_config_moe_expert_parallel_size", str(topology.moe_expert_parallel_size),
        "--replica_config_num_pipeline_stages", str(topology.num_pipeline_stages),
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_block_size", str(block_size),
        "--vllm_v1_scheduler_config_max_tokens_in_batch", str(token_budget),
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size", str(token_budget),
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request", str(max_context_len),
        "--decode_cuda_graph_mode", decode_cuda_graph_mode,
        "--metrics_config_output_dir", output_dir,
        "--metrics_config_run_id", run_id,
        # metrics_extractor reads system_metrics.json/request_metrics.csv only -- Frontier's own
        # per-column plots and chrome trace are pure overhead here (minutes per run, unused output).
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
    ]
    if atten_input_file:
        # Points the predictor at a specific attention profiling CSV -- e.g. a block-size- or
        # backend-tagged variant (attention_combined_block16.csv) -- instead of the default
        # ./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv. Must be the *combined* file
        # (standard + true-mixed rows merged): the predictor loads one file and filters
        # is_true_mixed_batch rows out of it, it never reads attention_true_mixed.csv separately.
        args.extend([
            "--random_forrest_execution_time_predictor_config_atten_input_file", atten_input_file,
        ])

    if request_mode == "closed_loop":
        args.extend([
            "--request_generator_config_type", "closed_loop",
            "--closed_loop_request_generator_config_num_requests", str(real.num_prompts),
            "--closed_loop_request_generator_config_max_concurrency", str(real.concurrency),
            "--closed_loop_length_generator_config_type", "fixed",
            "--fixed_request_length_generator_config_prefill_tokens", str(real.random_input_len),
            "--fixed_request_length_generator_config_decode_tokens", str(real.random_output_len),
        ])
    else:
        if real.request_throughput_req_s is None:
            raise ValueError(f"Missing request_throughput_req_s on real record (concurrency={real.concurrency})")
        calibrated_qps = real.request_throughput_req_s
        args.extend([
            "--request_generator_config_type", "synthetic",
            "--synthetic_request_generator_config_num_requests", str(real.num_prompts),
            "--length_generator_config_type", "fixed",
            "--fixed_request_length_generator_config_prefill_tokens", str(real.random_input_len),
            "--fixed_request_length_generator_config_decode_tokens", str(real.random_output_len),
            "--interval_generator_config_type", "poisson",
            "--poisson_request_interval_generator_config_qps", f"{calibrated_qps:.6f}",
        ])

    if enable_chunked_prefill:
        args.append("--vllm_v1_scheduler_config_enable_chunked_prefill")
    else:
        args.append("--no-vllm_v1_scheduler_config_enable_chunked_prefill")
    if network_device:
        args.extend(["--replica_config_network_device", network_device])

    return SimPoint(
        concurrency=real.concurrency,
        run_id=run_id,
        request_mode=request_mode,
        args=args,
        real=real,
        calibrated_qps=calibrated_qps,
    )


def build_sweep(
    run: Union[BenchmarkRun, AggregatedRun], topology: Topology, output_dir: str, **kwargs
) -> List[SimPoint]:
    """One SimPoint per real Benchmark-phase concurrency level (warmup phase is intentionally skipped).

    `run` may be a single-run BenchmarkRun or a multi-repetition AggregatedRun -- both expose
    `.benchmark` as a plain list of per-concurrency BenchmarkResult (AggregatedRun's is the
    canonical repetition per concurrency level), which is all this needs: workload shape, not
    the real-side stats used for reporting.
    """
    return [build_sim_point(r, topology, output_dir, **kwargs) for r in run.benchmark]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        help="benchmark_results/run_<label>/ directory (or a bare bench_output.txt), or a "
        "directory containing several repeated run_<label>/ subdirectories -- workload shape "
        "is taken from one canonical repetition per concurrency level",
    )
    parser.add_argument("--device", required=True, help="Frontier device SKU, e.g. mi355x")
    parser.add_argument("--model-name", required=True, help="Frontier --replica_config_model_name, e.g. deepseek-r1-0528")
    parser.add_argument("--attn-tp", type=int, required=True)
    parser.add_argument("--attn-dp", type=int, default=1)
    parser.add_argument("--moe-tp", type=int, default=0, help="defaults to --attn-tp if unset")
    parser.add_argument("--moe-ep", type=int, default=1)
    parser.add_argument("--pipeline-stages", type=int, default=1)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument("--cc-backend", default="analytical", help="e.g. analytical or vidur")
    parser.add_argument("--network-device", default=None, help="e.g. mi355x_8gpu (required for --cc-backend vidur)")
    parser.add_argument("--request-mode", default="closed_loop", choices=["closed_loop", "poisson"],
                         help="closed_loop (default, matches real max-concurrency semantics) or poisson (legacy calibrated-QPS approximation)")
    parser.add_argument("--block-size", type=int, default=32,
                         help="Must match the paged-attention block size the model's profiling data was collected "
                              "at (check data/profiling/compute/<device>/<model>/attention.csv's block_size column) "
                              "-- otherwise the execution-time predictor's training data comes back empty. "
                              "Default 32 matches deepseek-r1-0528; e.g. qwen3-a3b-30b-moe/gpt-oss-* use 16.")
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True,
                         help="Default on, matching real vLLM/sglang server behavior. Disable only as a workaround "
                              "when the model's profiling data has zero is_mixed_batch rows (check "
                              "data/profiling/compute/<device>/<model>/attention.csv) -- chunked prefill lets "
                              "prefill and decode share a batch, which needs a trained attn_decode_in_mixed "
                              "predictor; without it, true mixed batches raise instead of silently mispredicting.")
    parser.add_argument("--max-tokens-in-batch", type=int, default=None,
                         help="Per-iteration scheduler token budget (vLLM's --max-num-batched-tokens / "
                              "SGLang's --chunked-prefill-size -- an aggregate cap shared across everything "
                              "packed into one scheduling step, not a per-request limit). Defaults to "
                              "max(prefill_tokens, 8192) when unset; pass the real server's actual value "
                              "here when known (check its startup logs/launch flags) for an accurate match.")
    parser.add_argument("--atten-input-file", default=None,
                         help="Path to a specific attention profiling CSV, overriding the default "
                              "./data/profiling/compute/<device>/<model>/attention.csv -- e.g. a block-size- "
                              "or backend-tagged variant like attention_combined_block16.csv. Must be the "
                              "*combined* file (standard + true-mixed rows merged), not the plain or "
                              "true-mixed-only ones -- the predictor loads one file and filters "
                              "is_true_mixed_batch rows out of it.")
    parser.add_argument("--output-dir", default="outputs/mi355x_deepseek_r1_validation")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of shell commands")
    args = parser.parse_args()

    run = load_and_aggregate(args.run_dir)

    topology = Topology(
        device=args.device,
        model_name=args.model_name,
        attn_tensor_parallel_size=args.attn_tp,
        attn_data_parallel_size=args.attn_dp,
        moe_tensor_parallel_size=args.moe_tp,
        moe_expert_parallel_size=args.moe_ep,
        num_pipeline_stages=args.pipeline_stages,
        num_replicas=args.num_replicas,
    )

    sim_points = build_sweep(
        run, topology, args.output_dir,
        cc_backend=args.cc_backend, network_device=args.network_device, request_mode=args.request_mode,
        block_size=args.block_size, enable_chunked_prefill=args.enable_chunked_prefill,
        max_tokens_in_batch=args.max_tokens_in_batch, atten_input_file=args.atten_input_file,
    )

    if args.json:
        print(json.dumps([{"concurrency": sp.concurrency, "run_id": sp.run_id, "request_mode": sp.request_mode,
                            "calibrated_qps": sp.calibrated_qps, "command": sp.command} for sp in sim_points], indent=2))
        return

    for sp in sim_points:
        qps_note = f", calibrated_qps={sp.calibrated_qps:.2f}" if sp.calibrated_qps is not None else ""
        print(f"# concurrency={sp.concurrency}  mode={sp.request_mode}  num_prompts={sp.real.num_prompts}{qps_note}")
        print(sp.command_str())
        print()


if __name__ == "__main__":
    main()
