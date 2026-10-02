"""Reduction of vLLM torch-profiler traces to per-forward device timelines.

The reduction runs on the GPU worker after the replay, so a wrong join or idle
split would only show after a multi-GPU run; a synthetic kineto trace checks it.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tests.comparison.calibration.forward_device_timeline import main

BASE_NS = 1_790_000_000_000_000_000


def complete(cat: str, name: str, ts: float, dur: float, pid: int = 7, tid: int = 7, **args) -> dict:
    return {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid, "ts": ts, "dur": dur, "args": args}


def write_trace(path: Path, events: list[dict], rank: int | None) -> None:
    # Kineto's layout: header members one per line, then traceEvents, then trailing members.
    lines = ["{", '  "schemaVersion": 1,', '  "deviceProperties": [', "  ],"]
    if rank is not None:
        info = {"backend": "nccl", "rank": rank, "world_size": 2, "pg_config": [{"ranks": [0, 1]}]}
        lines.append(f'      "distributedInfo": {json.dumps(info)},')
    lines += ['  "displayTimeUnit": "ms",', f'  "baseTimeNanoseconds": {BASE_NS},', '  "traceEvents": [',
              ",\n".join(json.dumps(event) for event in events), "  ],", '  "traceName": "t"', "}"]
    with gzip.open(path, "wt") as trace:
        trace.write("\n".join(lines))


def rank0_events() -> list[dict]:
    return [
        # Launched before the forward: its kernel runs inside the forward's range but is not its work.
        complete("cuda_runtime", "cudaMemcpyAsync", 900, 5, correlation=6),
        complete("gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", 1030, 4, pid=0, tid=7, correlation=6),
        complete("user_annotation", "Forward", 1000, 100),
        complete("cpu_op", "aten::mm", 1008, 8),
        complete("cuda_runtime", "cudaLaunchKernel", 1010, 5, correlation=1),
        complete("kernel", "gemm", 1020, 20, pid=0, tid=7, correlation=1),
        # The device idles 1040-1056; the launch call returns at 1054: 14 waiting for the host, 2 other.
        complete("cuda_driver", "cuLaunchKernelEx", 1050, 4, correlation=2),
        complete("kernel", "fused_moe_kernel", 1056, 14, pid=0, tid=7, correlation=2),
        # Overlaps the previous kernel on a second stream: busy grows by 1070-1080 only.
        complete("cuda_runtime", "cudaLaunchKernel", 1055, 2, correlation=3),
        complete("kernel", "rms_norm", 1060, 20, pid=0, tid=8, correlation=3),
        # Launched long before the device frees up: the gap 1080-1090 is other idle time.
        complete("cuda_runtime", "cudaMemcpyAsync", 1060, 3, correlation=4),
        complete("gpu_memcpy", "Memcpy DtoD (Device -> Device)", 1090, 5, pid=0, tid=7, correlation=4),
        # Another thread's launch inside the forward's range is not the forward's work.
        complete("cuda_runtime", "cudaLaunchKernel", 1041, 1, tid=9, correlation=5),
        complete("kernel", "other_thread_kernel", 1041, 4, pid=0, tid=7, correlation=5),
        # A graph replay: its kernels carry the cudaGraphLaunch call's correlation id.
        complete("user_annotation", "Forward", 2000, 50),
        complete("cuda_runtime", "cudaGraphLaunch", 2005, 10, correlation=10),
        complete("kernel", "graph_kernel", 2020, 10, pid=0, tid=7, correlation=10),
        complete("kernel", "cross_device_reduce_1stage", 2032, 8, pid=0, tid=7, correlation=10),
    ]


def rank1_events() -> list[dict]:
    # Rank 1 replays the same graph; its all-reduce waits less (5 against 8) and ends with rank 0's.
    # It has no eager forward in the window, so rank 0's eager forward gets no cross-rank minimum.
    return [
        complete("user_annotation", "Forward", 2001, 40),
        complete("cuda_runtime", "cudaGraphLaunch", 2004, 8, correlation=20),
        complete("kernel", "graph_kernel", 2018, 9, pid=1, tid=7, correlation=20),
        complete("kernel", "cross_device_reduce_1stage", 2035, 5, pid=1, tid=7, correlation=20),
    ]


def test_each_forward_splits_its_device_span_into_busy_and_idle_time(tmp_path: Path) -> None:
    write_trace(tmp_path / "api_server.async_llm.pt.trace.json.gz", [], rank=None)
    write_trace(tmp_path / "worker_rank1.pt.trace.json.gz", rank1_events(), rank=1)
    write_trace(tmp_path / "worker_rank0.pt.trace.json.gz", rank0_events(), rank=0)
    output = tmp_path / "device_timeline.json"

    assert main(["--trace-dir", str(tmp_path), "--output", str(output)]) == 0

    result = json.loads(output.read_text())
    assert result["trace_file"] == "worker_rank0.pt.trace.json.gz"
    assert result["distributed_info"] == {"backend": "nccl", "rank": 0, "world_size": 2}
    [forward] = result["forwards"]
    base_us = BASE_NS / 1000
    assert forward["host_start_us"] == base_us + 1000 and forward["host_end_us"] == base_us + 1100
    assert forward["launch_calls"] == 4 and forward["device_activities"] == 4
    assert forward["device_start_us"] == base_us + 1020 and forward["device_end_us"] == base_us + 1095
    assert forward["device_span_ms"] == pytest.approx(0.075)
    assert forward["device_sum_ms"] == pytest.approx(0.059)
    assert forward["busy_ms"] == pytest.approx(0.049)
    assert forward["idle_ms"] == pytest.approx(0.026)
    assert forward["idle_host_wait_ms"] == pytest.approx(0.014)
    assert forward["idle_other_ms"] == pytest.approx(0.012)
    assert forward["device_ms_by_name"] == pytest.approx({
        "gemm": 0.020, "rms_norm": 0.020, "fused_moe_kernel": 0.014, "Memcpy DtoD (Device -> Device)": 0.005,
    })
    assert forward["all_reduce_us"] == [] and forward["all_reduce_min_us"] is None

    [graph] = result["graph_forwards"]
    assert graph["launch_calls"] == 1 and graph["device_activities"] == 2
    assert graph["device_start_us"] == base_us + 2020 and graph["device_end_us"] == base_us + 2040
    assert graph["busy_ms"] == pytest.approx(0.018)
    # The graph launch returned at 2015, before the gap 2030-2032 opened.
    assert graph["idle_host_wait_ms"] == 0 and graph["idle_other_ms"] == pytest.approx(0.002)
    assert graph["all_reduce_us"] == [8] and graph["all_reduce_min_us"] == [5]


def test_the_reduction_needs_one_trace_per_rank(tmp_path: Path) -> None:
    write_trace(tmp_path / "api_server.pt.trace.json.gz", [], rank=None)
    write_trace(tmp_path / "worker_rank0.pt.trace.json.gz", rank0_events(), rank=0)
    with pytest.raises(SystemExit, match="expected one trace per rank"):
        main(["--trace-dir", str(tmp_path), "--output", str(tmp_path / "out.json")])
