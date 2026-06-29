"""Contracts for profiling timing statistics emitted to CSV schemas."""

from __future__ import annotations

import json
from pathlib import Path

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
