"""Per-kernel device gap of eager launches and CUDA-graph replays (kernel_gap.csv).

A forward step's device time is its kernels' durations plus the idle time
between consecutive kernels. The kernel-only operator tables price the first
part; this profiler measures the second, per kernel, for the two ways a
forward runs. The chain is one forward of the linear-op profiler's model (its
decoder block repeated, with dummy weights) for --model, so it has the kernel
mix of a layer's norms, GEMMs and rotary embedding:

- cuda_graph: the chain at each of --cuda_graph_num_tokens, the case's vLLM
  CUDA-graph capture sizes, captured as one graph per size and replayed. vLLM
  captures all of its decode graphs at startup, and on H800 a process that
  holds more than three captured graphs replays each of them with about
  0.4 us between kernels instead of about 0.05 us; so every graph is captured
  before any is measured, and all stay alive. Only the kernels of each graph
  launch are measured, not PyTorch's replay prologue;
- eager: the chain at each token count in --num_tokens, measured after the
  graphs in the same process. A spin kernel holds the stream until the host
  has queued every launch of the chain, so the idle time between two kernels
  is the device's own gap and never a wait for the host.

The gap before a chain kernel is the device idle time since the kernels before
it ended (0 when it overlaps them), from the CUPTI kernel records of a
torch.profiler trace; each execution mode's row holds the mean over all gaps
of all token counts and repeats, the device idle per kernel. Each setting's
mean, median and percentiles are printed as JSON lines.

    python -m frontier.profiling.kernel_gap.main --device h800 --output_dir data/profiling \
        --model Qwen3-235B-A22B --num_tensor_parallel_workers 8 \
        --cuda_graph_num_tokens 1 2 4 8 16 24 32 40 48 56 64 --num_tokens 16 2048

Output: <output_dir>/compute/<device>/kernel_gap.csv
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

EXECUTION_MODES = ("eager", "cuda_graph")
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


def graph_launch_chains(trace_events: Sequence[dict], repeats: int) -> list[list[dict]]:
    """Kernel records of each of `repeats` graph replays, joined to their cudaGraphLaunch call by correlation id."""

    kernels_by_correlation: dict[int, list[dict]] = defaultdict(list)
    for event in trace_events:
        if event.get("cat") == "kernel":
            kernels_by_correlation[event["args"]["correlation"]].append(event)
    chains = [sorted(kernels_by_correlation[event["args"]["correlation"]], key=lambda kernel: kernel["ts"])
              for event in trace_events
              if event.get("cat") in ("cuda_runtime", "cuda_driver")
              and event["name"].startswith(GRAPH_LAUNCH_API_NAMES)]
    if len(chains) != repeats or len({len(chain) for chain in chains}) != 1:
        raise ValueError(f"expected {repeats} graph launches with the same kernel count, "
                         f"found kernel counts {[len(chain) for chain in chains]}")
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


def captured_graph(launch_chain: Callable[[], None]):
    """The chain captured as a CUDA graph, replayed once."""

    import torch

    launch_chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_chain()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def graph_replay_chains(graph, repeats: int, trace_dir: Path) -> list[list[dict]]:
    """Kernel records of each of `repeats` replays of a captured graph."""

    import torch

    def run() -> None:
        for _ in range(repeats):
            graph.replay()
            torch.cuda.synchronize()

    return graph_launch_chains(profiled_trace(run, trace_dir), repeats)


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
    parser.add_argument("--cuda_graph_num_tokens", type=int, nargs="+", required=True,
                        help="The case's vLLM CUDA-graph capture sizes; one graph is captured per size")
    parser.add_argument("--num_tokens", type=int, nargs="+", default=[16, 2048],
                        help="Token counts of the eager forward")
    parser.add_argument("--repeats", type=int, default=20, help="Chains per setting and mode")
    parser.add_argument("--host_ahead_ms", type=float, default=50.0,
                        help="Spin time before each eager chain, so the host queues every launch first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    gaps = {mode: [] for mode in EXECUTION_MODES}

    def record(mode: str, tokens: int, setting_gaps: list[float]) -> None:
        print(json.dumps({"execution_mode": mode, "model": args.model,
                          "num_tensor_parallel_workers": args.num_tensor_parallel_workers, "num_tokens": tokens,
                          "num_gaps": len(setting_gaps),
                          "mean_us": float(np.mean(setting_gaps)),
                          "median_us": float(np.median(setting_gaps)),
                          "p10_us": float(np.percentile(setting_gaps, 10)),
                          "p90_us": float(np.percentile(setting_gaps, 90)),
                          "p99_us": float(np.percentile(setting_gaps, 99))}), flush=True)
        gaps[mode].extend(setting_gaps)

    with tempfile.TemporaryDirectory() as work_dir:
        trace_dir = Path(work_dir)
        chain_launch = decoder_layer_chain(args.model, args.num_tensor_parallel_workers, trace_dir)
        graphs = {tokens: captured_graph(chain_launch(tokens)) for tokens in args.cuda_graph_num_tokens}
        for tokens, graph in graphs.items():
            record("cuda_graph", tokens, chain_gaps_us(graph_replay_chains(graph, args.repeats, trace_dir)))
        for tokens in args.num_tokens:
            record("eager", tokens, chain_gaps_us(eager_chains(chain_launch(tokens), args.repeats,
                                                               args.host_ahead_ms, trace_dir)))
    rows = [{
        "execution_mode": mode,
        "kernel_gap_us": float(np.mean(mode_gaps)),
        "num_gaps": len(mode_gaps),
        "chain": f"decoder_layer:{args.model}:tp{args.num_tensor_parallel_workers}",
        "gpu_name": torch.cuda.get_device_name(),
        "driver_version": driver_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    } for mode, mode_gaps in gaps.items()]
    output_path = Path(args.output_dir) / "compute" / args.device / "kernel_gap.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} kernel-gap rows to {output_path}")


if __name__ == "__main__":
    main()
