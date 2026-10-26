"""Per-iteration arithmetic of the native decode device timeline on a synthetic kineto trace."""

import gzip
import json

import pytest

from tests.integration.step5_pd import native_decode_timeline


def event(cat, name, ts, dur, pid=1, tid=1, **args):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "pid": pid, "tid": tid, "args": args}


def test_iterations_split_device_busy_and_idle_by_launching_thread(tmp_path):
    trace = {"traceEvents": [
        event("user_annotation", "execute_context_0(0)_generation_4(4)", 0, 100),
        event("user_annotation", "execute_context_0(0)_generation_4(4)", 1000, 100),
        event("user_annotation", "execute_context_0(0)_generation_5(5)", 2000, 100),
        event("cuda_runtime", "cudaGraphLaunch", 10, 20, correlation=1),
        event("cuda_runtime", "cudaLaunchKernel", 500, 10, correlation=2),
        event("cuda_runtime", "cudaMemcpyAsync", 1010, 5, correlation=3),
        event("kernel", "ncclDevKernel_AllReduce_Sum_bf16", 40, 100, pid=0, tid=7, correlation=1),
        event("kernel", "fused_moe_kernel", 140, 200, pid=0, tid=7, correlation=1),
        event("kernel", "elementwise_kernel", 520, 50, pid=0, tid=7, correlation=2),
        event("gpu_memcpy", "Memcpy DtoH", 1020, 30, pid=0, tid=8, correlation=3),
        # Launched by another thread: counts toward the wall-clock busy time only.
        event("kernel", "sampler_kernel", 900, 200, pid=0, tid=9, correlation=9),
    ]}
    path = tmp_path / "load_t016_dp1_pp0_tp3_dcp0_ep11_rank11.1700.pt.trace.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump(trace, handle)
    output = tmp_path / "timeline.json"
    assert native_decode_timeline.main([str(tmp_path), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    [record] = result["traces"]
    assert (record["window"], record["dp"], record["tp"], record["ep"]) == ("load_t016", 1, 3, 11)
    first, second = record["iterations"]
    assert first["decode_requests"] == 4 and first["period_ms"] == pytest.approx(1.0)
    assert first["execute_ms"] == pytest.approx(0.1)
    assert first["busy_ms"] == pytest.approx(0.35) and first["span_ms"] == pytest.approx(0.53)
    # The 180 us gap before the elementwise kernel: its launch returned 170 us into it.
    assert first["idle_host_wait_ms"] == pytest.approx(0.17) and first["idle_other_ms"] == pytest.approx(0.01)
    assert first["wall_busy_ms"] == pytest.approx(0.45)
    assert first["device_ms_by_category"] == pytest.approx(
        {"tp_all_reduce": 0.1, "moe_compute": 0.2, "other": 0.05})
    assert second["busy_ms"] == pytest.approx(0.03) and second["wall_busy_ms"] == pytest.approx(0.1)
    assert second["device_ms_by_category"] == pytest.approx({"copy": 0.03})
    bucket = result["summary"]["load_t016"]["dp1"]["4"]
    assert bucket["iterations"] == 2 and bucket["busy_ms"] == pytest.approx(0.19)
    assert result["other_kernels_ms"] == pytest.approx({"sampler_kernel": 0.2, "elementwise_kernel": 0.05})
