"""Contracts for profiling timing statistics emitted to CSV schemas."""

from __future__ import annotations

import json
from pathlib import Path

from frontier.profiling.utils import record_function_tracer as tracer_module
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer
from frontier.profiling.utils.singleton import Singleton


def _reset_timer_stats_singleton() -> None:
    Singleton._instances.pop(TimerStatsStore, None)  # pylint: disable=protected-access


def test_cuda_event_timer_stats_include_sample_count() -> None:
    _reset_timer_stats_singleton()
    store = TimerStatsStore(profile_method="cuda")
    store.record_time("vidur_attn_prefill", 1.25)
    store.record_time("vidur_attn_prefill", 2.75)

    stats = store.get_stats()

    assert stats["attn_prefill"]["count"] == 2
    assert stats["attn_prefill"]["min"] == 1.25
    assert stats["attn_prefill"]["max"] == 2.75


def test_record_function_tracer_stats_include_sample_count(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "cat": "user_annotation",
                        "name": "vidur_attn_prefill",
                        "ts": 0,
                        "dur": 100,
                    },
                    {
                        "cat": "cuda_runtime",
                        "ts": 10,
                        "dur": 5,
                        "args": {"correlation": 1},
                    },
                    {
                        "cat": "kernel",
                        "ts": 20,
                        "dur": 1000,
                        "args": {"correlation": 1},
                    },
                    {
                        "cat": "user_annotation",
                        "name": "vidur_attn_prefill",
                        "ts": 200,
                        "dur": 100,
                    },
                    {
                        "cat": "cuda_driver",
                        "ts": 210,
                        "dur": 5,
                        "args": {"correlation": 2},
                    },
                    {
                        "cat": "kernel",
                        "ts": 220,
                        "dur": 3000,
                        "args": {"correlation": 2},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    tracer = RecordFunctionTracer(str(tmp_path))
    tracer.trace_path = str(trace_path)

    stats = tracer.get_operation_time_stats()

    assert stats["attn_prefill"]["count"] == 2
    assert stats["attn_prefill"]["min"] == 1.0
    assert stats["attn_prefill"]["max"] == 3.0


def test_record_function_tracer_primes_cuda_capture_before_measured_scopes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[object] = []

    class FakeProfiler:
        def __enter__(self):
            calls.append("profiler_enter")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            calls.append("profiler_exit")

        def export_chrome_trace(self, trace_path: str) -> None:
            calls.append(("export", trace_path))

    class FakeTensor:
        def __init__(self, label: str):
            self.label = label

        def __add__(self, value: int):
            calls.append(("add", self.label, value))
            return FakeTensor(f"{self.label}+{value}")

    def fake_profile(*, activities):
        calls.append(("profile", tuple(activities)))
        return FakeProfiler()

    def fake_empty(shape, *, device):
        calls.append(("empty", tuple(shape), device))
        return FakeTensor("seed")

    def fake_synchronize() -> None:
        calls.append("synchronize")

    monkeypatch.setattr(tracer_module.torch.profiler, "profile", fake_profile)
    monkeypatch.setattr(tracer_module.torch, "empty", fake_empty)
    monkeypatch.setattr(tracer_module.torch.cuda, "synchronize", fake_synchronize)

    tracer = RecordFunctionTracer(str(tmp_path))
    returned = tracer.__enter__()
    calls.append("returned")

    assert returned is tracer
    assert calls.index("profiler_enter") < calls.index(("empty", (1,), "cuda"))
    assert calls.index(("empty", (1,), "cuda")) < calls.index("synchronize")
    assert calls.index("synchronize") < calls.index("returned")
    add_calls = [call for call in calls if isinstance(call, tuple) and call[0] == "add"]
    assert len(add_calls) >= 4
    assert getattr(tracer, "_cuda_profiler_prime_tensor").label.endswith("+1")


def test_record_function_tracer_ignores_non_vidur_cuda_priming_events(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "cat": "user_annotation",
                        "name": "profiler_cuda_capture_prime",
                        "ts": 0,
                        "dur": 50,
                    },
                    {
                        "cat": "cuda_runtime",
                        "ts": 10,
                        "dur": 5,
                        "args": {"correlation": 100},
                    },
                    {
                        "cat": "kernel",
                        "ts": 20,
                        "dur": 9000,
                        "args": {"correlation": 100},
                    },
                    {
                        "cat": "user_annotation",
                        "name": "vidur_attn_kv_cache_save",
                        "ts": 100,
                        "dur": 50,
                    },
                    {
                        "cat": "cuda_runtime",
                        "ts": 110,
                        "dur": 5,
                        "args": {"correlation": 101},
                    },
                    {
                        "cat": "kernel",
                        "ts": 120,
                        "dur": 2000,
                        "args": {"correlation": 101},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    tracer = RecordFunctionTracer(str(tmp_path))
    tracer.trace_path = str(trace_path)

    stats = tracer.get_operation_time_stats()

    assert set(stats) == {"attn_kv_cache_save"}
    assert stats["attn_kv_cache_save"]["mean"] == 2.0
