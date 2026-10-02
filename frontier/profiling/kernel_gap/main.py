"""Per-kernel device gap of eager launches and CUDA-graph replays (kernel_gap.csv).

A forward step's device time is its kernels' durations plus the idle time
between consecutive kernels. The kernel-only operator tables price the first
part; this profiler measures the second, per kernel, for the two ways a
forward runs:

- eager: a spin kernel holds the stream until the host has queued every launch
  of the chain, so the idle time between two kernels is the device's own gap
  and never a wait for the host;
- cuda_graph: the same chain captured once and replayed.

Two chains are available:

- elementwise: a run of in-place adds on one tensor, for each element count in
  --num_elements;
- decoder_layer: one forward of the linear-op profiler's model (its decoder
  block repeated, with dummy weights) for --model at each token count in
  --num_tokens, so the chain has the kernel mix of a layer's norms, GEMMs and
  rotary embedding.

The gap of one pair of consecutive chain kernels is the next kernel's start
minus the previous kernel's end in the CUPTI kernel records of a
torch.profiler trace; each execution mode's row holds the median over all
pairs of all chain settings and repeats. The per-setting medians are printed
as JSON lines.

    python -m frontier.profiling.kernel_gap.main --device h800 --output_dir data/profiling \
        --chain decoder_layer --model Qwen3-235B-A22B --num_tensor_parallel_workers 8

Output: <output_dir>/compute/<device>/kernel_gap.csv
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

EXECUTION_MODES = ("eager", "cuda_graph")
# torch.cuda._sleep launches this kernel; it holds the stream and is not part of a chain.
SPIN_KERNEL_NAME = "spin_kernel"
LAUNCH_API_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx")


def chain_kernels(trace_events: Sequence[dict]) -> list[dict]:
    """Kernel records of the measured chains in device order (spin kernels removed)."""

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


def chain_gaps_us(chains: Sequence[Sequence[dict]]) -> list[float]:
    """Idle time between consecutive kernels inside each repeat of a chain, in us."""

    return [following["ts"] - (previous["ts"] + previous["dur"])
            for chain in chains for previous, following in zip(chain, chain[1:])]


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


def measure_gaps(mode: str, launch_chain: Callable[[], None], repeats: int,
                 host_ahead_ms: float, trace_dir: Path) -> list[float]:
    import torch

    launch_chain()
    torch.cuda.synchronize()
    if mode == "eager":
        spin_cycles = int(host_ahead_ms * spin_cycles_per_ms())

        def run() -> None:
            for _ in range(repeats):
                torch.cuda._sleep(spin_cycles)
                launch_chain()
                torch.cuda.synchronize()
    elif mode == "cuda_graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_chain()
        graph.replay()
        torch.cuda.synchronize()

        def run() -> None:
            for _ in range(repeats):
                graph.replay()
                torch.cuda.synchronize()
    else:
        raise ValueError(f"unknown execution mode {mode!r}; expected one of {EXECUTION_MODES}")

    events = profiled_trace(run, trace_dir)
    chains = split_chains(chain_kernels(events), repeats)
    if mode == "eager":
        require_host_ahead(events, chains)
    return chain_gaps_us(chains)


def elementwise_chains(num_elements: Sequence[int], num_kernels: int) -> Iterator[tuple[dict, Callable[[], None]]]:
    import torch

    for elements in num_elements:
        tensor = torch.zeros(elements, device="cuda", dtype=torch.float32)

        def launch_chain(tensor=tensor) -> None:
            for _ in range(num_kernels):
                tensor.add_(1.0)

        yield {"num_elements": elements, "num_kernels": num_kernels}, launch_chain


def decoder_layer_chains(model: str, num_tensor_parallel_workers: int, num_tokens: Sequence[int],
                         trace_dir: Path) -> Iterator[tuple[dict, Callable[[], None]]]:
    import torch

    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
    from frontier.profiling.utils import build_profile_position_indices

    model_config = ModelConfig.from_model_name(model)
    # The record_function method keeps the block's operator timers on the host,
    # so they add no device work and the block can be captured in a graph.
    wrapper = LinearOpWrapper(model_config, num_tensor_parallel_workers, "record_function",
                              rank=0, output_dir=str(trace_dir))
    for tokens in num_tokens:
        input_ids = torch.randint(0, wrapper.padded_vocab_size // num_tensor_parallel_workers, (tokens,),
                                  device="cuda", dtype=torch.long)
        positions = torch.tensor(build_profile_position_indices(tokens, model_config.max_position_embeddings),
                                 device="cuda", dtype=torch.long)

        def launch_chain(input_ids=input_ids, positions=positions) -> None:
            with torch.inference_mode():
                wrapper.model(input_ids, positions)

        yield {"model": model, "num_tensor_parallel_workers": num_tensor_parallel_workers,
               "num_tokens": tokens}, launch_chain


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
    parser.add_argument("--chain", choices=("elementwise", "decoder_layer"), required=True)
    parser.add_argument("--num_elements", type=int, nargs="+", default=[512, 67584, 1081344],
                        help="elementwise: element counts of the chain tensor (float32)")
    parser.add_argument("--num_kernels", type=int, default=256, help="elementwise: kernels per chain")
    parser.add_argument("--model", type=str, help="decoder_layer: model name of data/config/models")
    parser.add_argument("--num_tensor_parallel_workers", type=int, default=1, help="decoder_layer: TP shard")
    parser.add_argument("--num_tokens", type=int, nargs="+", default=[16, 2048],
                        help="decoder_layer: token counts of the forward")
    parser.add_argument("--repeats", type=int, default=20, help="Chains per setting and mode")
    parser.add_argument("--host_ahead_ms", type=float, default=50.0,
                        help="Spin time before each eager chain, so the host queues every launch first")
    args = parser.parse_args()
    if args.chain == "decoder_layer" and not args.model:
        parser.error("--chain decoder_layer requires --model")
    return args


def main() -> None:
    args = parse_args()
    import torch

    gaps = {mode: [] for mode in EXECUTION_MODES}
    with tempfile.TemporaryDirectory() as trace_dir:
        chains = (elementwise_chains(args.num_elements, args.num_kernels) if args.chain == "elementwise"
                  else decoder_layer_chains(args.model, args.num_tensor_parallel_workers, args.num_tokens,
                                            Path(trace_dir)))
        for setting, launch_chain in chains:
            for mode in EXECUTION_MODES:
                setting_gaps = measure_gaps(mode, launch_chain, args.repeats, args.host_ahead_ms, Path(trace_dir))
                print(json.dumps({"execution_mode": mode, **setting, "num_gaps": len(setting_gaps),
                                  "median_us": float(np.median(setting_gaps)),
                                  "p10_us": float(np.percentile(setting_gaps, 10)),
                                  "p90_us": float(np.percentile(setting_gaps, 90))}), flush=True)
                gaps[mode].extend(setting_gaps)
    chain = args.chain if args.chain == "elementwise" else f"decoder_layer:{args.model}:tp{args.num_tensor_parallel_workers}"
    rows = [{
        "execution_mode": mode,
        "kernel_gap_us": float(np.median(mode_gaps)),
        "num_gaps": len(mode_gaps),
        "chain": chain,
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
