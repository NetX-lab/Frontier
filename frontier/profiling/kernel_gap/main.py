"""Per-kernel device gap of eager launches and CUDA-graph replays (kernel_gap.csv).

A forward step's device time is its kernels' durations plus the idle time
between consecutive kernels. The kernel-only operator tables price the first
part; this profiler measures the second, per kernel, for the two ways a
forward runs:

- eager: a spin kernel holds the stream until the host has queued every launch
  of the chain, so the idle time between two kernels is the device's own gap
  and never a wait for the host;
- cuda_graph: the same chain captured once and replayed.

A chain is a run of in-place elementwise kernels on one tensor, repeated for
each element count in --num_elements. The gap of one pair of consecutive
chain kernels is the next kernel's start minus the previous kernel's end in
the CUPTI kernel records of a torch.profiler trace; each execution mode's row
holds the median over all pairs of all chains and repeats. The per-element-count
medians are printed as JSON lines.

    python -m frontier.profiling.kernel_gap.main --device h800 --output_dir data/profiling

Output: <output_dir>/compute/<device>/kernel_gap.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

EXECUTION_MODES = ("eager", "cuda_graph")
# torch.cuda._sleep launches this kernel; it holds the stream and is not part of a chain.
SPIN_KERNEL_NAME = "spin_kernel"
LAUNCH_API_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC")


def chain_kernels(trace_events: Sequence[dict]) -> list[dict]:
    """Kernel records of the measured chains in device order (spin kernels removed)."""

    kernels = [event for event in trace_events
               if event.get("cat") == "kernel" and SPIN_KERNEL_NAME not in event["name"]]
    return sorted(kernels, key=lambda event: event["ts"])


def chain_gaps_us(kernels: Sequence[dict], num_kernels: int, repeats: int) -> list[float]:
    """Idle time between consecutive kernels inside each repeat of a chain, in us."""

    if len(kernels) != num_kernels * repeats:
        raise ValueError(
            f"expected {num_kernels} x {repeats} chain kernels in the trace, found {len(kernels)}"
        )
    names = {kernel["name"] for kernel in kernels}
    if len(names) != 1:
        raise ValueError(f"a chain must launch one kernel name, found {sorted(names)}")
    gaps = []
    for start in range(0, len(kernels), num_kernels):
        chain = kernels[start:start + num_kernels]
        gaps.extend(following["ts"] - (previous["ts"] + previous["dur"])
                    for previous, following in zip(chain, chain[1:]))
    return gaps


def require_host_ahead(trace_events: Sequence[dict], kernels: Sequence[dict],
                       num_kernels: int) -> None:
    """Fail unless every launch of a chain returned before its first kernel started."""

    launches = sorted((event for event in trace_events
                       if event.get("cat") in ("cuda_runtime", "cuda_driver")
                       and event["name"] in LAUNCH_API_NAMES),
                      key=lambda event: event["ts"])
    correlation_end = {launch["args"]["correlation"]: launch["ts"] + launch["dur"] for launch in launches}
    for start in range(0, len(kernels), num_kernels):
        chain = kernels[start:start + num_kernels]
        last_launch_end = max(correlation_end[kernel["args"]["correlation"]] for kernel in chain)
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


def measure_gaps(mode: str, num_elements: int, num_kernels: int, repeats: int,
                 host_ahead_ms: float, trace_dir: Path) -> list[float]:
    import torch

    tensor = torch.zeros(num_elements, device="cuda", dtype=torch.float32)

    def launch_chain() -> None:
        for _ in range(num_kernels):
            tensor.add_(1.0)

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
    kernels = chain_kernels(events)
    gaps = chain_gaps_us(kernels, num_kernels, repeats)
    if mode == "eager":
        require_host_ahead(events, kernels, num_kernels)
    return gaps


def driver_version() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
                          check=True, capture_output=True, text=True).stdout.strip()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", type=str, default="data/profiling",
                        help="Root output directory for profiling results (default: data/profiling)")
    parser.add_argument("--device", type=str, required=True,
                        help="Hardware SKU for the output path (e.g. h800)")
    parser.add_argument("--num_elements", type=int, nargs="+", default=[512, 67584, 1081344],
                        help="Element counts of the chain tensor (float32)")
    parser.add_argument("--num_kernels", type=int, default=256, help="Kernels per chain")
    parser.add_argument("--repeats", type=int, default=20, help="Chains per element count and mode")
    parser.add_argument("--host_ahead_ms", type=float, default=50.0,
                        help="Spin time before each eager chain, so the host queues every launch first")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as trace_dir:
        for mode in EXECUTION_MODES:
            gaps = []
            for num_elements in args.num_elements:
                chain = measure_gaps(mode, num_elements, args.num_kernels, args.repeats,
                                     args.host_ahead_ms, Path(trace_dir))
                print(json.dumps({"execution_mode": mode, "num_elements": num_elements,
                                  "num_gaps": len(chain),
                                  "median_us": float(np.median(chain)),
                                  "p10_us": float(np.percentile(chain, 10)),
                                  "p90_us": float(np.percentile(chain, 90))}), flush=True)
                gaps.extend(chain)
            rows.append({
                "execution_mode": mode,
                "kernel_gap_us": float(np.median(gaps)),
                "num_gaps": len(gaps),
                "gpu_name": torch.cuda.get_device_name(),
                "driver_version": driver_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            })
    output_path = Path(args.output_dir) / "compute" / args.device / "kernel_gap.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} kernel-gap rows to {output_path}")


if __name__ == "__main__":
    main()
