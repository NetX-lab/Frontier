"""Inputs the DP-placement harness builds for a ground-truth run.

A wrong trace row or server switch is only visible after a multi-GPU run, so
the pieces that decide them are checked here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tests.comparison.dp_placement_pp.make_trace import build_rows
from tests.comparison.dp_placement_pp.vllm_replay import OP_TIMING_SCOPES, server_command, server_env


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


def test_warmup_groups_precede_seeded_poisson_arrivals():
    workload = {
        "request_id_namespace": "t",
        "warmups": {"groups": [
            {"name": "single", "count": 2, "interval_s": 1.0,
             "num_prefill_tokens": 2048, "num_decode_tokens": 16},
            {"name": "batch", "gap_before_s": 3.0, "count": 2, "interval_s": 0.03,
             "num_prefill_tokens": 2048, "num_decode_tokens": 64},
        ]},
        "poisson": {"gap_before_s": 10.0, "count": 3, "qps": 2.0, "seed": 0,
                    "num_prefill_tokens": 2048, "num_decode_tokens": 256},
    }
    rows = build_rows(workload)

    assert [(row["request_id"], row["arrived_at"]) for row in rows if row["role"] == "warmup"] == [
        ("t-single-w0", 0.0), ("t-single-w1", 1.0), ("t-batch-w0", 4.0), ("t-batch-w1", 4.03),
    ]
    formal = [row for row in rows if row["role"] == "formal"]
    assert [row["request_id"] for row in formal] == ["t-p000", "t-p001", "t-p002"]
    assert {row["segment"] for row in formal} == {"poisson"}
    assert {(row["num_prefill_tokens"], row["num_decode_tokens"]) for row in formal} == {(2048, 256)}
    intervals = random.Random(0)
    expected = [14.03]
    expected.append(expected[-1] + intervals.expovariate(2.0))
    expected.append(expected[-1] + intervals.expovariate(2.0))
    assert [row["arrived_at"] for row in formal] == [round(value, 6) for value in expected]
    assert build_rows(workload) == rows


def test_the_mode_alone_sets_the_frontier_switches_of_the_server():
    inherited = {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_SCHED_LOG_PATH": "/tmp/sched.jsonl",
        "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH": "/tmp/old.jsonl",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_V1_ALLOW_NO_CHUNKED_PREFILL": "1",
        "VLLM_MOE_UNIFORM_ROUTING": "1",
        "VLLM_TORCH_PROFILER_DIR": "/tmp/old_traces",
        "VLLM_TORCH_PROFILER_RECORD_SHAPES": "1",
        "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1",
    }
    out = Path("/run")

    clean, clean_set = server_env("clean", {"enable_chunked_prefill": True}, out, inherited)
    instrumented, _ = server_env(
        "instrumented",
        {"attention_backend": "FLASHINFER", "enable_chunked_prefill": False,
         "moe_uniform_routing": True},
        out,
        inherited,
    )
    op_timing, _ = server_env("op_timing", {"enable_chunked_prefill": True}, out, inherited)
    kernel_timing, _ = server_env("kernel_timing", {"enable_chunked_prefill": True}, out, inherited)
    schedule_timing, _ = server_env("schedule_timing", {"enable_chunked_prefill": True}, out, inherited)
    cpu, _ = server_env("cpu", {"enable_chunked_prefill": True}, out, inherited)
    kv_save_timing, _ = server_env("kv_save_timing", {"enable_chunked_prefill": True}, out, inherited)
    kv_save_device_ids, _ = server_env("kv_save_device_ids", {"enable_chunked_prefill": True}, out, inherited)
    device_timeline, _ = server_env("device_timeline", {"enable_chunked_prefill": True}, out, inherited,
                                    Path("/traces"))

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
        "VLLM_MOE_UNIFORM_ROUTING": "1",
        "VLLM_V1_ALLOW_NO_CHUNKED_PREFILL": "1",
    }
    assert op_timing == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_INSTRUMENTATION": "1",
        "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH": "/run/op_timing.jsonl",
        "VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES": ",".join(OP_TIMING_SCOPES),
        "VLLM_FRONTIER_OP_TIMING_MODE": "cuda_event",
        "VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE": "default",
        "VLLM_FRONTIER_OP_AGG_MODE": "per_scope",
        "VLLM_FRONTIER_BATCH_LOG_PATH": "/run/batch_log.jsonl",
        "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH": "/run/pp_boundary.jsonl",
        "VLLM_FRONTIER_SCHED_LOG_PATH": "/run/schedule.jsonl",
    }
    assert kernel_timing == op_timing | {"VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE": "kernel_only"}
    assert schedule_timing == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_SCHED_LOG_PATH": "/run/schedule.jsonl",
    }
    assert cpu == {
        "PATH": "/usr/bin",
        "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": "/run/dp_placement",
        "VLLM_FRONTIER_CPU_PROBE_LOG_PATH": "/run/cpu_probe.jsonl",
    }
    assert kv_save_timing == cpu | {"VLLM_FRONTIER_KV_SAVE_PROBE_LOG_PATH": "/run/kv_save_probe.jsonl"}
    assert kv_save_device_ids == kv_save_timing | {"VLLM_FRONTIER_KV_SAVE_DEVICE_BLOCK_IDS": "1"}
    assert device_timeline == cpu | {
        "VLLM_TORCH_PROFILER_DIR": "/traces",
        "VLLM_TORCH_PROFILER_WITH_STACK": "0",
        "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1",
    }
    # The fork's default scope list has no dense MLP scope; the timing modes name them.
    assert {"mlp_up_proj", "mlp_act", "mlp_down_proj"} <= set(OP_TIMING_SCOPES)


def test_the_engine_file_alone_sets_the_cuda_graph_setup_of_the_server():
    engine = {
        "load_format": "dummy", "dtype": "float16", "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1, "data_parallel_size": 1, "max_num_batched_tokens": 16384,
        "max_num_seqs": 64, "block_size": 16, "max_model_len": 4096,
        "gpu_memory_utilization": 0.9, "seed": 0, "skip_tokenizer_init": True,
        "enable_expert_parallel": False, "enforce_eager": True,
        "enable_chunked_prefill": False, "enable_prefix_caching": False,
    }
    graph = engine | {
        "enforce_eager": False,
        "compilation_config": {"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
    }

    eager_command = server_command(engine, Path("/m"), 8000)
    graph_command = server_command(graph, Path("/m"), 8000)

    assert "--enforce-eager" in eager_command
    assert "--compilation-config" not in eager_command
    assert "--enforce-eager" not in graph_command
    config = graph_command[graph_command.index("--compilation-config") + 1]
    assert json.loads(config) == {"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
