"""Inputs the DP-placement harness builds for a ground-truth run.

A wrong trace row or server switch is only visible after a multi-GPU run, so
the pieces that decide them are checked here.
"""

from __future__ import annotations

from pathlib import Path

from tests.comparison.dp_placement_pp.make_trace import build_rows
from tests.comparison.dp_placement_pp.vllm_replay import server_env


def workload(bursts: list[dict]) -> dict:
    return {
        "request_id_namespace": "t",
        "warmups": {"count": 1, "interval_s": 1.0, "num_prefill_tokens": 8, "num_decode_tokens": 2},
        "idle_gap_s": 20.0,
        "bursts": bursts,
        "steady": {
            "gap_after_burst_s": 5.0, "count": 1, "interval_s": 0.1,
            "num_prefill_tokens": [16], "num_decode_tokens": [4],
        },
    }


def test_a_burst_has_a_probe_only_when_it_names_an_offset_and_may_set_its_own_gap():
    rows = build_rows(workload([
        {"name": "q", "order": ["short", "short"], "short_prefill_tokens": 32,
         "num_decode_tokens": 4, "idle_gap_s": 3.0, "probe_offset_s": 0.0005,
         "probe_prefill_tokens": 32, "probe_decode_tokens": 4},
        {"name": "l", "order": ["long"], "long_prefill_tokens": 4096, "num_decode_tokens": 1},
    ]))

    burst = [
        (row["request_id"], row["arrived_at"], row["num_prefill_tokens"], row["probe"])
        for row in rows if row["segment"] == "burst"
    ]
    assert burst == [
        ("t-q-b1", 3.0, 32, False),
        ("t-q-b2", 3.0, 32, False),
        ("t-q-b3", 3.0005, 32, True),
        ("t-l-b1", 23.0005, 4096, False),
    ]
    assert rows[-1]["arrived_at"] == 28.0005


def test_the_mode_alone_sets_the_frontier_switches_of_the_server():
    inherited = {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_SCHED_LOG_PATH": "/tmp/sched.jsonl",
        "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH": "/tmp/old.jsonl",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    }
    out = Path("/run")

    clean, clean_set = server_env("clean", {}, out, inherited)
    instrumented, _ = server_env(
        "instrumented", {"attention_backend": "FLASHINFER"}, out, inherited
    )

    assert clean == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH": "/run/request_metrics.jsonl",
    }
    assert clean_set == {key: value for key, value in clean.items() if key != "PATH"}
    assert instrumented == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_INSTRUMENTATION": "1",
        "VLLM_FRONTIER_MOE_ROUTING_LOG_PATH": "/run/moe_routing.jsonl",
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
    }
