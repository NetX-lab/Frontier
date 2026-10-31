"""One-GPU CUDA launch queue probe; CUDA imports occur only in run_probe.

Each trial gates the launch stream behind a sleep, records every host launch,
and stops at the first blocking launch, exhausted sleep, or enqueue budget.
Unblocked trials are censored, never reported as a measured capacity.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def probe_cases(config: dict) -> list[dict]:
    kernels = config["kernels_per_graph"]
    if not kernels or len(set(kernels)) != len(kernels) or any(
        type(k) is not int or k < 1 or k & (k - 1) for k in kernels
    ):
        raise ValueError("kernels_per_graph must be distinct positive powers of two")
    if min(config["sleep_ms"]) <= 0 or config["trials"] < 1:
        raise ValueError("Sleep and trials must be positive")
    return [dict(mode=mode, kernels_per_graph=k, sleep_ms=s, trial=t)
            for mode, k in [("graph", k) for k in kernels] + [("plain", 1)]
            for s in config["sleep_ms"] for t in range(config["trials"])]


def blocking_depth(record: dict, fraction: float = 0.25, slowdown: float = 50) -> dict:
    launches = record["launches"]
    baseline = float(record["baseline_launch_us"])
    sleep_us = float(record["actual_sleep_us"])
    if baseline <= 0 or sleep_us <= 0:
        raise ValueError("A trial requires positive calibrated durations")
    threshold = max(fraction * sleep_us, slowdown * baseline)
    for i, launch in enumerate(launches):
        if (launch["host_us"] >= threshold and launch["start_us"] < sleep_us * 0.8
                and launch.get("gate_pending", True)):
            return dict(depth=i, censored=False, threshold_us=threshold,
                        blocking_launch_us=launch["host_us"])
    return dict(depth=len(launches), censored=True, threshold_us=threshold,
                reason=record.get("stop_reason", "no blocking launch"))


def run_probe(config: dict) -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Launch-depth probe requires exactly one visible CUDA GPU")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    tensor = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    # Calibrate cycles with device events, independently of host launch latency.
    cycles = 10_000_000
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    with torch.cuda.stream(stream):
        begin.record()
        torch.cuda._sleep(cycles)
        end.record()
    end.synchronize()
    cycles_per_us = cycles / (begin.elapsed_time(end) * 1000)
    (output / "runtime.json").write_text(json.dumps(dict(
        torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0), config=config,
        cycles_per_us=cycles_per_us), indent=2) + "\n")
    graphs = {}
    for k in config["kernels_per_graph"]:
        with torch.cuda.stream(stream):
            for _ in range(3):
                tensor.add_(1)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(k):
                tensor.add_(1)
        stream.synchronize()
        # Separate untimed census verifies the requested graph node count.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
        count = sum(e.device_type == torch.autograd.DeviceType.CUDA and
                    e.name != "[memory]" for e in prof.events())
        prof.export_chrome_trace(str(output / f"graph_k{k}.json"))
        if count != k:
            raise RuntimeError(f"Graph k={k} census found {count} device activities")
        graphs[k] = graph
    with (output / "launches.jsonl").open("a", buffering=1) as log:
        for case in probe_cases(config):
            k = case["kernels_per_graph"]
            launch = graphs[k].replay if case["mode"] == "graph" else lambda: tensor.add_(1)
            baseline = []
            with torch.cuda.stream(stream):
                for _ in range(20):
                    start = time.perf_counter_ns()
                    launch()
                    baseline.append((time.perf_counter_ns() - start) / 1000)
                    stream.synchronize()
                origin = time.perf_counter_ns()
                begin.record()
                torch.cuda._sleep(int(case["sleep_ms"] * 1000 * cycles_per_us))
                end.record()
                gate_enqueue_us = (time.perf_counter_ns() - origin) / 1000
                launches = []
                stop = "enqueue_budget"
                budget = min(config["max_launches"], config["max_enqueued_kernels"] // k)
                for _ in range(budget):
                    if end.query():
                        stop = "sleep_finished"
                        break
                    start = time.perf_counter_ns()
                    launch()
                    elapsed = (time.perf_counter_ns() - start) / 1000
                    launches.append(dict(start_us=(start - origin) / 1000, host_us=elapsed,
                                         gate_pending=True))
                    if elapsed >= max(case["sleep_ms"] * 1000 * 0.25,
                                      statistics.median(baseline) * 50):
                        stop = "blocking_launch"
                        break
                    if (time.perf_counter_ns() - origin) / 1e6 >= case["sleep_ms"] * 0.8:
                        stop = "sleep_budget"
                        break
            stream.synchronize()
            actual_sleep = begin.elapsed_time(end) * 1000
            record = {**case, "baseline_launch_us": statistics.median(baseline),
                      "actual_sleep_us": actual_sleep, "launches": launches,
                      "stop_reason": stop, "gate_enqueue_us": gate_enqueue_us}
            record["analysis"] = blocking_depth(record)
            log.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.list_cases:
        for case in probe_cases(config):
            print(json.dumps(case, sort_keys=True))
        return
    run_probe(config)


if __name__ == "__main__":
    main()
