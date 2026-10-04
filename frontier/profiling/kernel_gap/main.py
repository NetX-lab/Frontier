"""Per-kernel device gap of eager launches and CUDA-graph replays (kernel_gap.csv).

A forward step's device time is its kernels' durations plus the idle time
between consecutive kernels. The kernel-only operator tables price the first
part; this profiler measures the second, per kernel, for the two ways a
forward runs:

- eager: one forward of the linear-op profiler's model (its decoder block
  repeated, with dummy weights) for --model at each token count in
  --num_tokens, so the chain has the kernel mix of a layer's norms, GEMMs and
  rotary embedding. A spin kernel holds the stream until the host has queued
  every launch of the chain, so the idle time between two kernels is the
  device's own gap and never a wait for the host;
- cuda_graph: the decode CUDA graphs of a vLLM engine with dummy weights for
  --model and the deployment's engine arguments (--vllm_engine_args). The gap
  inside a graph replay depends on the process that replays it (on H800 the
  same chain graph idled 0.05-0.45 us per gap by how many graphs the process
  held and by model and TP), so it is measured on the engine's own graphs.
  Request i of max_num_seqs requests stops after i + 1 tokens, so the decode
  batch shrinks by one per step and every capture size's graph is replayed
  under vLLM's torch profiler; only the kernels of each cudaGraphLaunch are
  measured, on every rank. The eager chain runs first, before the engine
  starts.

The gap before a kernel is the device idle time since the kernels before it
ended (0 when it overlaps them), from the CUPTI kernel records of a
torch.profiler trace; each execution mode's row holds the mean over all its
gaps, the device idle per kernel. Each eager token count's and each rank
trace's mean, median and percentiles are printed as JSON lines.

    python -m frontier.profiling.kernel_gap.main --device h800 --output_dir data/profiling \
        --model Qwen3-235B-A22B --num_tensor_parallel_workers 8 --num_tokens 16 2048 \
        --vllm_engine_args engine_args.json --vllm_attention_backend FLASHINFER

engine_args.json holds vllm.LLM keyword arguments other than the model, e.g.
{"load_format": "dummy", "skip_tokenizer_init": true, "tensor_parallel_size": 8,
"compilation_config": {"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}, "max_num_seqs": 64}.

Output: <output_dir>/compute/<device>/<model>/kernel_gap.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from frontier.profiling.kernel_gap.schema import EXECUTION_MODE_COLUMN, EXECUTION_MODES, KERNEL_GAP_US_COLUMN
from frontier.profiling.utils import build_profiling_output_path

# torch.cuda._sleep launches this kernel; it holds the stream and is not part of a chain.
SPIN_KERNEL_NAME = "spin_kernel"
LAUNCH_API_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx")
GRAPH_LAUNCH_API_NAMES = ("cudaGraphLaunch", "cuGraphLaunch")


def chain_kernels(trace_events: Sequence[dict]) -> list[dict]:
    """Kernel records of the measured eager chains in device order (spin kernels removed)."""

    kernels = [event for event in trace_events
               if event.get("cat") == "kernel" and SPIN_KERNEL_NAME not in event["name"]]
    return sorted(kernels, key=lambda event: event["ts"])


def split_chains(kernels: Sequence[dict], repeats: int) -> list[Sequence[dict]]:
    """The kernels of each repeat of the chain; every repeat launches the same kernel sequence."""

    if len(kernels) % repeats:
        raise ValueError(f"{len(kernels)} chain kernels in the trace do not split into {repeats} repeats")
    per_chain = len(kernels) // repeats
    chains = [kernels[start:start + per_chain] for start in range(0, len(kernels), per_chain)]
    sequences = {tuple(kernel["name"] for kernel in chain) for chain in chains}
    if len(sequences) != 1:
        raise ValueError(f"the {repeats} repeats of the chain launched {len(sequences)} different kernel sequences")
    return chains


def graph_launch_chains(trace_events: Sequence[dict]) -> list[list[dict]]:
    """Kernel records of each CUDA-graph launch in a trace, joined to its cudaGraphLaunch call by correlation id."""

    kernels_by_correlation: dict[int, list[dict]] = defaultdict(list)
    for event in trace_events:
        if event.get("cat") == "kernel":
            kernels_by_correlation[event["args"]["correlation"]].append(event)
    chains = [sorted(kernels_by_correlation[event["args"]["correlation"]], key=lambda kernel: kernel["ts"])
              for event in trace_events
              if event.get("cat") in ("cuda_runtime", "cuda_driver")
              and event["name"].startswith(GRAPH_LAUNCH_API_NAMES)]
    if not chains:
        raise ValueError("the trace holds no CUDA-graph launch")
    return chains


def chain_gaps_us(chains: Sequence[Sequence[dict]]) -> list[float]:
    """Device idle time before each kernel but the first of each chain, in us (0 when it overlaps earlier kernels)."""

    gaps = []
    for chain in chains:
        covered_end = chain[0]["ts"] + chain[0]["dur"]
        for kernel in chain[1:]:
            gaps.append(max(0.0, kernel["ts"] - covered_end))
            covered_end = max(covered_end, kernel["ts"] + kernel["dur"])
    return gaps


def require_host_ahead(trace_events: Sequence[dict], chains: Sequence[Sequence[dict]]) -> None:
    """Fail unless every launch of a chain returned before its first kernel started."""

    launch_end = {event["args"]["correlation"]: event["ts"] + event["dur"] for event in trace_events
                  if event.get("cat") in ("cuda_runtime", "cuda_driver") and event["name"] in LAUNCH_API_NAMES}
    for chain in chains:
        last_launch_end = max(launch_end[kernel["args"]["correlation"]] for kernel in chain)
        if last_launch_end > chain[0]["ts"]:
            raise RuntimeError(
                "the host was not ahead of the device: the chain's last launch returned "
                f"{last_launch_end - chain[0]['ts']:.1f} us after its first kernel started; "
                "raise --host_ahead_ms"
            )


def profiled_trace(run: Callable[[], None], trace_dir: Path) -> list[dict]:
    import torch

    trace_path = trace_dir / "kernel_gap_trace.json"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        # CUPTI can drop the first kernel record of a fresh session; these
        # spin kernels absorb that and are excluded from every chain.
        for _ in range(4):
            torch.cuda._sleep(1000)
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    return json.loads(trace_path.read_text())["traceEvents"]


def spin_cycles_per_ms() -> float:
    import torch

    cycles = 10_000_000
    torch.cuda._sleep(cycles)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    end.synchronize()
    return cycles / start.elapsed_time(end)


def eager_chains(launch_chain: Callable[[], None], repeats: int, host_ahead_ms: float,
                 trace_dir: Path) -> list[Sequence[dict]]:
    """Kernel records of each of `repeats` eager chains, each queued behind a spin kernel."""

    import torch

    launch_chain()
    torch.cuda.synchronize()
    spin_cycles = int(host_ahead_ms * spin_cycles_per_ms())

    def run() -> None:
        for _ in range(repeats):
            torch.cuda._sleep(spin_cycles)
            launch_chain()
            torch.cuda.synchronize()

    events = profiled_trace(run, trace_dir)
    chains = split_chains(chain_kernels(events), repeats)
    require_host_ahead(events, chains)
    return chains


def vllm_decode_graph_gaps(model: str, engine_args: dict, attention_backend: str | None, prompt_tokens: int,
                           work_dir: Path) -> dict[str, list[float]]:
    """Per rank trace: the gaps inside every decode CUDA-graph launch of a vLLM engine with dummy weights."""

    trace_dir = work_dir / "vllm_traces"
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(trace_dir)
    # vLLM records Python stacks unless told otherwise, at a host cost per operator.
    os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "0"
    if attention_backend is not None:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model_dir = work_dir / "model"
    model_dir.mkdir()
    shutil.copyfile(Path("data/config/models") / f"{model.replace('/', '__')}.json", model_dir / "config.json")
    llm = LLM(model=str(model_dir), **engine_args)
    num_requests = llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
    prompts = [TokensPrompt(prompt_token_ids=[index + 1] * prompt_tokens) for index in range(num_requests)]
    params = [SamplingParams(max_tokens=index + 1, ignore_eos=True, detokenize=False) for index in range(num_requests)]
    llm.generate(prompts, params, use_tqdm=False)
    llm.start_profile()
    llm.generate(prompts, params, use_tqdm=False)
    llm.stop_profile()

    gaps = {}
    for trace_path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        with gzip.open(trace_path, "rt") as handle:
            gaps[trace_path.name] = chain_gaps_us(graph_launch_chains(json.load(handle)["traceEvents"]))
    if not gaps:
        raise RuntimeError(f"vLLM wrote no profiler trace to {trace_dir}")
    return gaps


def decoder_layer_chain(model: str, num_tensor_parallel_workers: int,
                        trace_dir: Path) -> Callable[[int], Callable[[], None]]:
    """A function from a token count to the launch of one chain forward of that many tokens."""

    import torch

    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
    from frontier.profiling.utils import build_profile_position_indices

    model_config = ModelConfig.from_model_name(model)
    # The record_function method keeps the block's operator timers on the host,
    # so they add no device work and the block can be captured in a graph.
    wrapper = LinearOpWrapper(model_config, num_tensor_parallel_workers, "record_function",
                              rank=0, output_dir=str(trace_dir))

    def chain_launch(tokens: int) -> Callable[[], None]:
        input_ids = torch.randint(0, wrapper.padded_vocab_size // num_tensor_parallel_workers, (tokens,),
                                  device="cuda", dtype=torch.long)
        positions = torch.tensor(build_profile_position_indices(tokens, model_config.max_position_embeddings),
                                 device="cuda", dtype=torch.long)

        def launch_chain() -> None:
            with torch.inference_mode():
                wrapper.model(input_ids, positions)

        return launch_chain

    return chain_launch


def driver_version() -> str:
    import pynvml

    pynvml.nvmlInit()
    try:
        return pynvml.nvmlSystemGetDriverVersion()
    finally:
        pynvml.nvmlShutdown()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", type=str, default="data/profiling",
                        help="Root output directory for profiling results (default: data/profiling)")
    parser.add_argument("--device", type=str, required=True,
                        help="Hardware SKU for the output path (e.g. h800)")
    parser.add_argument("--model", type=str, required=True, help="Model name of data/config/models")
    parser.add_argument("--num_tensor_parallel_workers", type=int, default=1, help="Attention TP shard")
    parser.add_argument("--num_tokens", type=int, nargs="+", default=[16, 2048],
                        help="Token counts of the eager forward")
    parser.add_argument("--repeats", type=int, default=20, help="Eager chains per token count")
    parser.add_argument("--host_ahead_ms", type=float, default=50.0,
                        help="Spin time before each eager chain, so the host queues every launch first")
    parser.add_argument("--vllm_engine_args", type=str, required=True,
                        help="JSON file of the deployment's vllm.LLM keyword arguments, without the model")
    parser.add_argument("--vllm_attention_backend", type=str, default=None,
                        help="VLLM_ATTENTION_BACKEND of the deployment (default: vLLM's own selection)")
    parser.add_argument("--vllm_prompt_tokens", type=int, default=16,
                        help="Prompt length of each request of the profiled vLLM decode run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    gaps = {mode: [] for mode in EXECUTION_MODES}

    def record(mode: str, setting: dict, setting_gaps: list[float]) -> None:
        print(json.dumps({EXECUTION_MODE_COLUMN: mode, "model": args.model, **setting,
                          "num_gaps": len(setting_gaps),
                          "mean_us": float(np.mean(setting_gaps)),
                          "median_us": float(np.median(setting_gaps)),
                          "p10_us": float(np.percentile(setting_gaps, 10)),
                          "p90_us": float(np.percentile(setting_gaps, 90)),
                          "p99_us": float(np.percentile(setting_gaps, 99))}), flush=True)
        gaps[mode].extend(setting_gaps)

    with tempfile.TemporaryDirectory() as temporary_dir:
        work_dir = Path(temporary_dir)
        chain_launch = decoder_layer_chain(args.model, args.num_tensor_parallel_workers, work_dir)
        for tokens in args.num_tokens:
            record("eager", {"num_tensor_parallel_workers": args.num_tensor_parallel_workers, "num_tokens": tokens},
                   chain_gaps_us(eager_chains(chain_launch(tokens), args.repeats, args.host_ahead_ms, work_dir)))
        # vLLM's workers start only when the device has the free memory the engine asks for.
        del chain_launch
        torch.cuda.empty_cache()
        engine_args = json.loads(Path(args.vllm_engine_args).read_text())
        for trace_name, trace_gaps in vllm_decode_graph_gaps(args.model, engine_args, args.vllm_attention_backend,
                                                             args.vllm_prompt_tokens, work_dir).items():
            record("cuda_graph", {"trace": trace_name}, trace_gaps)
    sources = {"eager": f"decoder_layer:{args.model}:tp{args.num_tensor_parallel_workers}",
               "cuda_graph": f"vllm_decode_graphs:{args.model}"}
    rows = [{
        EXECUTION_MODE_COLUMN: mode,
        KERNEL_GAP_US_COLUMN: float(np.mean(mode_gaps)),
        "num_gaps": len(mode_gaps),
        "chain": sources[mode],
        "gpu_name": torch.cuda.get_device_name(),
        "driver_version": driver_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    } for mode, mode_gaps in gaps.items()]
    output_path = build_profiling_output_path(output_root=args.output_dir, profiling_type="compute",
                                              hardware=args.device, model_name=args.model, op_name="kernel_gap")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} kernel-gap rows to {output_path}")


if __name__ == "__main__":
    main()
