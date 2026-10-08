#!/usr/bin/env python3
"""Serve the DP-placement case with `vllm serve` and replay its trace over HTTP.

Runs inside the ``vllm/vllm-openai:v0.10.2`` image with the instrumented
overlay first on ``PYTHONPATH``. The server is the deployment the placement is
measured on (plan §18.1): one API server, whose ``DPLBAsyncMPClient`` routes
every request, plus the DP coordinator and one engine core per DP rank.

Each trace row becomes one ``/v1/completions`` request posted at its
``arrived_at`` offset from a common origin, with deterministic prompt token ids,
``max_tokens = min_tokens = num_decode_tokens`` and ``ignore_eos``, and the
header ``X-Request-Id: <request_id>``; the server names the engine request
``cmpl-<request_id>-0``. Request bodies are encoded before the origin so that
dispatch times are not delayed by encoding.

The run mode (calibration contract, "Run Modes") sets the server's Frontier
switches; none is inherited from the worker. Every mode writes the placement
records to ``dp_placement/``, declared scheduler-level workflow evidence.

``clean``
    E2E request metrics on (``request_metrics.jsonl``), operator probes off.
``instrumented``
    Instrumentation on with the MoE routing records (``moe_routing.jsonl``),
    E2E request metrics off.
``op_timing``, ``kernel_timing``
    Instrumentation on with CUDA-event timing per operator scope
    (``op_timing.jsonl``) and the batch, PP boundary and schedule records
    (``batch_log.jsonl``, ``pp_boundary.jsonl``, ``schedule.jsonl``); MoE
    routing records and E2E request metrics off. ``op_timing`` times each scope
    as it runs, launch gaps included. ``kernel_timing`` synchronizes the device
    before each scope, so a scope holds its kernels only. The step times of
    neither mode are ground truth. Both time the scopes in ``OP_TIMING_SCOPES``.
``schedule_timing``
    The scheduler's per-step record (``schedule.jsonl``) only: instrumentation,
    operator probes and E2E request metrics off. The scheduler writes one line
    per step on the host, with no device sync, so the step timestamps measure
    the steps of an otherwise clean engine. E2E metrics come from ``clean`` runs.
``cpu``
    The CPU probe only (``cpu_probe.jsonl``): per-step host timestamps of
    schedule, input preparation, the forward call, logits and sampling,
    bookkeeping and ``update_from_output``, kept in memory and written when the
    engine core shuts down. With DP > 1 the engine core of DP rank d writes
    ``cpu_probe_dp<d>.jsonl``. With PP > 1 the TP rank 0 worker of each PP
    stage also writes one record per forward (PP receive and send stamps,
    forward start, forward device time, padded input tokens, CUDA graph mode)
    to ``cpu_probe[_dp<d>]_pp<p>.jsonl`` when it shuts down. Instrumentation,
    operator probes and E2E request metrics off.
``kv_save_timing``, ``kv_save_device_ids``
    Diagnostic modes of a pd-disaggregation case, for a vLLM-BS diagnostic
    commit only: the CPU probe plus the producer connector's KV-save probe
    (``kv_save_probe.jsonl``), per-layer host stamps of each request's KV
    extraction, logger write and send queueing, each send in the send thread
    and each ``wait_for_save``, written when the engine core shuts down.
    ``kv_save_device_ids`` also sets ``VLLM_FRONTIER_KV_SAVE_DEVICE_BLOCK_IDS``,
    so the connector indexes the KV cache with a device copy of the block ids.
    Instrumentation, operator probes and E2E request metrics off.
``device_timeline``
    The CPU probe plus vLLM's own torch profiler (``VLLM_TORCH_PROFILER_DIR``,
    CPU and CUDA activities, no Python stacks) with the model runner's
    ``Forward`` range of each step (``VLLM_CUSTOM_SCOPES_FOR_PROFILING``). The
    replay opens the profiler through ``/start_profile`` at
    ``--profile-start-s`` after the first formal arrival and closes it through
    ``/stop_profile`` ``--profile-duration-s`` later; every process of the
    server then writes its trace to ``--torch-trace-dir``, outside the output
    directory. Step times inside the window carry the profiler's host
    overhead; the probe's steps before it carry none. Instrumentation, operator
    probes and E2E request metrics off.

The attention backend is an engine setting: ``VLLM_ATTENTION_BACKEND`` is set
from the engine file's ``attention_backend`` and is otherwise left to vLLM's
own selection. So is the CUDA graph setup: the engine file's
``compilation_config`` is passed to ``--compilation-config`` as given, and
without it vLLM keeps its own default. Outputs in ``--output-dir``: ``server.log``,
``client_requests.jsonl``, the mode's record file, ``dp_placement/``,
``replay_summary.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_admission_pp"))
from vllm_burst_driver import prompt_token_ids, write_model_dir  # noqa: E402

SERVED_MODEL_NAME = "dp_pp_case"
# Longest wait of the replay's event loop, which bounds the dispatch lateness.
HEARTBEAT_S = 0.01
# Operator-timing modes and the CUDA-event scope mode each one runs.
OP_TIMING_SCOPE_MODES = {"op_timing": "default", "kernel_timing": "kernel_only"}
# Operator scopes the vLLM-BS Llama and Qwen3-MoE models open in the calibration
# cases (EP by all-to-all, P2P KV transfer, TP all-reduce). The fork's default
# scope list leaves out the dense MLP scopes. The TP all-reduce scopes open only
# at TP > 1; the first two run inside attn_post_proj and mlp_down_proj.
OP_TIMING_SCOPES = (
    "input_layernorm", "attn_pre_proj", "attn_rope", "attn_kv_cache_save", "attn_prefill",
    "attn_decode", "attn_post_proj", "post_attention_layernorm",
    "mlp_up_proj", "mlp_act", "mlp_down_proj",
    "moe_gating", "moe_shuffling", "moe_grouped_gemm", "add",
    "expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine",
    "attn_post_proj_tp_allreduce", "mlp_down_proj_tp_allreduce",
    "moe_tensor_parallel_allreduce", "tensor_parallel_allreduce",
    "kv_p2p_send", "kv_p2p_recv",
)
# Modes shared with pd_replay.py. device_timeline profiles the one server this
# script starts, so it is this script's own.
MODES = ("clean", "instrumented", *OP_TIMING_SCOPE_MODES, "schedule_timing", "cpu",
         "kv_save_timing", "kv_save_device_ids")
KV_CACHE_LINE = re.compile(r"\((EngineCore_DP\d+) pid=\d+\).*GPU KV cache size: ([\d,]+) tokens")


def server_env(mode: str, engine: dict, output_dir: Path, inherited: dict,
               torch_trace_dir: Path | None = None) -> tuple[dict, dict]:
    """Return the server environment and the variables the mode set in it."""

    env = {
        key: value for key, value in inherited.items()
        if not key.startswith(("VLLM_FRONTIER_", "VLLM_TORCH_PROFILER_"))
        and key != "VLLM_CUSTOM_SCOPES_FOR_PROFILING"
        and key not in ("VLLM_ATTENTION_BACKEND", "VLLM_V1_ALLOW_NO_CHUNKED_PREFILL",
                        "VLLM_MOE_UNIFORM_ROUTING")
    }
    mode_env = {"VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": str(output_dir / "dp_placement")}
    if mode == "clean":
        mode_env["VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH"] = str(output_dir / "request_metrics.jsonl")
    elif mode == "instrumented":
        mode_env["VLLM_FRONTIER_INSTRUMENTATION"] = "1"
        mode_env["VLLM_FRONTIER_MOE_ROUTING_LOG_PATH"] = str(output_dir / "moe_routing.jsonl")
    elif mode == "schedule_timing":
        mode_env["VLLM_FRONTIER_SCHED_LOG_PATH"] = str(output_dir / "schedule.jsonl")
    elif mode == "cpu":
        mode_env["VLLM_FRONTIER_CPU_PROBE_LOG_PATH"] = str(output_dir / "cpu_probe.jsonl")
    elif mode in ("kv_save_timing", "kv_save_device_ids"):
        mode_env["VLLM_FRONTIER_CPU_PROBE_LOG_PATH"] = str(output_dir / "cpu_probe.jsonl")
        mode_env["VLLM_FRONTIER_KV_SAVE_PROBE_LOG_PATH"] = str(output_dir / "kv_save_probe.jsonl")
        if mode == "kv_save_device_ids":
            mode_env["VLLM_FRONTIER_KV_SAVE_DEVICE_BLOCK_IDS"] = "1"
    elif mode == "device_timeline":
        mode_env |= {
            "VLLM_FRONTIER_CPU_PROBE_LOG_PATH": str(output_dir / "cpu_probe.jsonl"),
            "VLLM_TORCH_PROFILER_DIR": str(torch_trace_dir),
            # vLLM records Python stacks unless told otherwise, at a host cost per operator.
            "VLLM_TORCH_PROFILER_WITH_STACK": "0",
            "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1",
        }
    else:
        mode_env |= {
            "VLLM_FRONTIER_INSTRUMENTATION": "1",
            "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH": str(output_dir / "op_timing.jsonl"),
            "VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES": ",".join(OP_TIMING_SCOPES),
            "VLLM_FRONTIER_OP_TIMING_MODE": "cuda_event",
            "VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE": OP_TIMING_SCOPE_MODES[mode],
            "VLLM_FRONTIER_OP_AGG_MODE": "per_scope",
            "VLLM_FRONTIER_BATCH_LOG_PATH": str(output_dir / "batch_log.jsonl"),
            "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH": str(output_dir / "pp_boundary.jsonl"),
            "VLLM_FRONTIER_SCHED_LOG_PATH": str(output_dir / "schedule.jsonl"),
        }
    if "attention_backend" in engine:
        mode_env["VLLM_ATTENTION_BACKEND"] = engine["attention_backend"]
    if engine.get("moe_uniform_routing"):
        # vLLM-BS routes token i to experts (i * top_k + k) mod num_experts,
        # the same split as Frontier's balanced routing distribution.
        mode_env["VLLM_MOE_UNIFORM_ROUTING"] = "1"
    if not engine["enable_chunked_prefill"]:
        # vLLM-BS refuses V1 generation without chunked prefill unless this is
        # set (EngineArgs._set_default_args_v1).
        mode_env["VLLM_V1_ALLOW_NO_CHUNKED_PREFILL"] = "1"
    return env | mode_env, mode_env


def server_command(engine: dict, model_dir: Path, port: int) -> list[str]:
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_dir),
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "127.0.0.1", "--port", str(port),
        "--load-format", engine["load_format"],
        "--dtype", engine["dtype"],
        "--tensor-parallel-size", str(engine["tensor_parallel_size"]),
        "--pipeline-parallel-size", str(engine["pipeline_parallel_size"]),
        "--data-parallel-size", str(engine["data_parallel_size"]),
        "--max-num-batched-tokens", str(engine["max_num_batched_tokens"]),
        "--max-num-seqs", str(engine["max_num_seqs"]),
        "--block-size", str(engine["block_size"]),
        "--max-model-len", str(engine["max_model_len"]),
        "--gpu-memory-utilization", str(engine["gpu_memory_utilization"]),
        "--seed", str(engine["seed"]),
    ]
    flags = {
        "--skip-tokenizer-init": engine["skip_tokenizer_init"],
        "--enable-expert-parallel": engine["enable_expert_parallel"],
        "--enforce-eager": engine["enforce_eager"],
        "--enable-chunked-prefill": engine["enable_chunked_prefill"],
        "--no-enable-chunked-prefill": not engine["enable_chunked_prefill"],
        "--enable-prefix-caching": engine["enable_prefix_caching"],
        "--no-enable-prefix-caching": not engine["enable_prefix_caching"],
    }
    command += [flag for flag, enabled in flags.items() if enabled]
    if "compilation_config" in engine:
        command += ["--compilation-config", json.dumps(engine["compilation_config"])]
    return command


def wait_until_ready(server: subprocess.Popen, port: int, timeout_s: float) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if server.poll() is not None:
            raise RuntimeError(f"vllm serve exited with {server.returncode} during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except OSError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"vllm serve not ready after {timeout_s} s")


async def replay(rows: list[dict], port: int, vocab_size: int, origin_lead_s: float,
                 profile_window_s: tuple[float, float] | None = None) -> tuple[dict, list[dict]]:
    """Post every row at its arrival offset; with profile_window_s, also open and
    close the server's torch profiler at those two offsets of the same clock."""
    import aiohttp

    url = f"http://127.0.0.1:{port}/v1/completions"
    bodies = {
        row["request_id"]: json.dumps({
            "model": SERVED_MODEL_NAME,
            "prompt": prompt_token_ids(row["request_id"], row["num_prefill_tokens"], vocab_size),
            "max_tokens": row["num_decode_tokens"],
            "min_tokens": row["num_decode_tokens"],
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": False,
        })
        for row in rows
    }
    records: list[dict] = []
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        origin = time.monotonic() + origin_lead_s
        origin_wall = time.time() + origin_lead_s

        async def post(row: dict) -> None:
            await asyncio.sleep(max(0.0, origin + row["arrived_at"] - time.monotonic()))
            dispatched = time.monotonic()
            async with session.post(
                url, data=bodies[row["request_id"]],
                headers={"Content-Type": "application/json", "X-Request-Id": row["request_id"]},
            ) as response:
                payload = await response.json(content_type=None)
            record = {
                "request_id": row["request_id"],
                "engine_request_id": f"cmpl-{row['request_id']}-0",
                "segment": row["segment"],
                "role": row["role"],
                "arrived_at": row["arrived_at"],
                "num_prefill_tokens": row["num_prefill_tokens"],
                "num_decode_tokens": row["num_decode_tokens"],
                "dispatch_offset_s": dispatched - origin,
                "dispatch_monotonic": dispatched,
                "response_monotonic": time.monotonic(),
                "http_status": response.status,
            }
            if response.status == 200:
                record["usage"] = payload.get("usage")
                record["finish_reason"] = payload["choices"][0].get("finish_reason")
            else:
                record["error"] = payload
            records.append(record)

        profile_marks: dict = {}

        async def profile(start_s: float, stop_s: float) -> None:
            for endpoint, at in (("start_profile", start_s), ("stop_profile", stop_s)):
                await asyncio.sleep(max(0.0, origin + at - time.monotonic()))
                requested = time.monotonic()
                async with session.post(f"http://127.0.0.1:{port}/{endpoint}") as response:
                    response.raise_for_status()
                profile_marks[endpoint] = {"requested_monotonic": requested,
                                           "returned_monotonic": time.monotonic()}

        async def heartbeat() -> None:
            # Linux lets a timed wait overshoot by up to about 0.1% of its
            # timeout, so after an idle gap of seconds the event loop would wake
            # a due post milliseconds late.
            while True:
                await asyncio.sleep(HEARTBEAT_S)

        posts = [post(row) for row in rows]
        if profile_window_s is not None:
            posts.append(profile(*profile_window_s))
        beat = asyncio.ensure_future(heartbeat())
        await asyncio.gather(*posts)
        beat.cancel()
    clock = {"origin_monotonic": origin, "origin_wall": origin_wall}
    if profile_window_s is not None:
        clock["profile_marks"] = profile_marks
    return clock, records


def stop_server(server: subprocess.Popen, timeout_s: float) -> dict:
    for sent in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        server.send_signal(sent)
        try:
            return {"signal": sent.name, "returncode": server.wait(timeout=timeout_s)}
        except subprocess.TimeoutExpired:
            continue
    return {"signal": "SIGKILL", "returncode": server.wait()}


def kv_cache_tokens(server_log: Path) -> dict[str, list[int]]:
    tokens: dict[str, list[int]] = {}
    for line in server_log.read_text(errors="replace").splitlines():
        match = KV_CACHE_LINE.search(line)
        if match:
            tokens.setdefault(match.group(1), []).append(int(match.group(2).replace(",", "")))
    return tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=(*MODES, "device_timeline"), required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--origin-lead-s", type=float, default=1.0)
    parser.add_argument("--drain-s", type=float, default=3.0)
    parser.add_argument("--stop-timeout-s", type=float, default=60.0)
    parser.add_argument("--torch-trace-dir", type=Path,
                        help="device_timeline: the profiler's trace directory, outside --output-dir")
    parser.add_argument("--profile-start-s", type=float, default=20.0,
                        help="device_timeline: profiler start, seconds after the first formal arrival")
    parser.add_argument("--profile-duration-s", type=float, default=10.0,
                        help="device_timeline: seconds the profiler stays open")
    args = parser.parse_args(argv)
    if args.mode == "device_timeline" and args.torch_trace_dir is None:
        parser.error("mode device_timeline needs --torch-trace-dir")

    engine = json.loads(args.engine_config.read_text())
    request_ids = json.loads((args.trace_dir / "request_ids.json").read_text())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = write_model_dir(
        REPO_ROOT / "data/config/models" / f"{engine['model_name']}.json",
        output_dir / "model",
    )

    env, mode_env = server_env(args.mode, engine, output_dir, dict(os.environ), args.torch_trace_dir)
    profile_window_s = None
    if args.mode == "device_timeline":
        first_formal = min(row["arrived_at"] for row in request_ids["rows"] if row["role"] == "formal")
        start = first_formal + args.profile_start_s
        profile_window_s = (start, start + args.profile_duration_s)
    command = server_command(engine, output_dir / "model", args.port)
    summary: dict = {
        "mode": args.mode,
        "engine_config": engine,
        "server_command": command,
        "server_env_set": mode_env,
        "server_env_removed": sorted(
            name for name in os.environ if name not in env or name in mode_env
        ),
        "trace_dir": str(args.trace_dir),
        "profile_window_s": profile_window_s,
    }
    with (output_dir / "server.log").open("w") as server_log:
        server = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            summary["startup_s"] = wait_until_ready(server, args.port, args.startup_timeout_s)
            clock, records = asyncio.run(replay(
                request_ids["rows"], args.port, int(model_config["vocab_size"]), args.origin_lead_s,
                profile_window_s,
            ))
            summary.update(clock)
            time.sleep(args.drain_s)
        finally:
            summary["server_stop"] = stop_server(server, args.stop_timeout_s)
    records.sort(key=lambda record: record["dispatch_monotonic"])
    (output_dir / "client_requests.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    failed = [record["request_id"] for record in records if record["http_status"] != 200]
    summary["num_requests"] = len(records)
    summary["failed_request_ids"] = failed
    summary["kv_cache_tokens_by_engine"] = kv_cache_tokens(output_dir / "server.log")
    (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=1))
    print("REPLAY_DONE", json.dumps({
        "requests": len(records), "failed": len(failed),
        "kv_cache_tokens_by_engine": summary["kv_cache_tokens_by_engine"],
    }))
    return 0 if not failed else 4


if __name__ == "__main__":
    raise SystemExit(main())
