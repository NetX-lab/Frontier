#!/usr/bin/env python3
"""Serve the DP-placement case with `vllm serve` and replay its trace over HTTP.

Runs inside the ``vllm/vllm-openai:v0.10.2`` image with the instrumented
overlay first on ``PYTHONPATH``. The server is the deployment the placement is
measured on (plan §18.1): one API server, whose ``DPLBAsyncMPClient`` routes
every request, plus the DP coordinator and one engine core per DP rank.

The explicit ``vllm_version=0.28.0`` adapter instead accepts existing local GLM
metadata, dummy weights and clean/light observer JSON. It prepares a standalone
local serve command; it does not orchestrate the approved multi-node deployment
or verify the installed image, process inventory or native qualification. The
legacy modes and outputs described below remain the v0.10.2 defaults.

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
    The CPU probe only: the engine core writes per-step host timestamps of
    schedule and ``update_from_output`` to ``cpu_probe.jsonl``, or
    ``cpu_probe_dp<d>.jsonl`` for DP rank d, when it shuts down. The TP rank 0
    worker of each PP stage, a single stage included, writes one record per
    forward (PP receive and send stamps, input preparation, forward-context
    entry, the DP token-count all-reduce, the forward call, logits and
    sampling on the last stage, the forward's device time, padded input tokens,
    CUDA graph mode) to ``cpu_probe[_dp<d>]_pp<p>.jsonl`` when it shuts down.
    Instrumentation, operator probes and E2E request metrics off.
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
import hashlib
import ipaddress
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
V028_SOURCE = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
V028_IMAGE = ("hub.i.basemind.com/vllm-openai/vllm-openai@sha256:"
              "2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635")
KV_CACHE_LINE = re.compile(r"\((EngineCore_DP\d+) pid=\d+\).*GPU KV cache size: ([\d,]+) tokens")
V028_NODE_FIELDS = ("data_parallel_size_local", "data_parallel_address", "data_parallel_rpc_port",
                    "data_parallel_start_rank", "headless")


def runtime_version(engine: dict) -> str:
    """Select the explicit harness adapter without claiming installed-runtime identity."""
    if type(engine) is not dict:
        raise ValueError("Engine configuration must be a JSON object")
    version = engine.get("vllm_version", "0.10.2")
    if version not in ("0.10.2", "0.28.0"):
        raise ValueError(f"Unsupported vLLM harness version: {version!r}")
    if version == "0.28.0":
        reference = engine.get("native_reference")
        if (type(reference) is not dict or set(reference) != {
                "source_sha", "instrumentation_sha", "image"}
                or reference["source_sha"] != V028_SOURCE
                or reference["image"] != V028_IMAGE
                or not isinstance(reference["instrumentation_sha"], str)
                or re.fullmatch(r"[0-9a-f]{40}", reference["instrumentation_sha"]) is None):
            raise ValueError("v0.28 requires the locked native source/image and instrumentation SHA")
        if engine.get("load_format") != "dummy" or engine.get("skip_tokenizer_init") is not True:
            raise ValueError("v0.28 harness requires dummy weights and skip_tokenizer_init")
        if engine.get("moe_uniform_routing") or engine.get("enable_return_routed_experts"):
            raise ValueError("v0.28 routing content and forced routing must remain OFF")
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
            if type(engine.get(name)) is not int or engine[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if "num_gpu_blocks_override" in engine:
            blocks = engine["num_gpu_blocks_override"]
            if type(blocks) is not int or blocks <= 0:
                raise ValueError("num_gpu_blocks_override must be a positive integer")
        for name in ("enable_expert_parallel", "enforce_eager", "enable_chunked_prefill",
                     "enable_prefix_caching"):
            if type(engine.get(name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if any(name in engine for name in V028_NODE_FIELDS):
            if (engine.get("data_parallel_rank") is not None or engine.get("data_parallel_hybrid_lb")
                    or engine.get("data_parallel_external_lb")):
                raise ValueError("Native DP node replay supports internal load balancing only")
            if any(name not in engine for name in V028_NODE_FIELDS[:3]):
                raise ValueError("Native DP nodes require explicit local size, head address and RPC port")
            local, dp = engine["data_parallel_size_local"], engine["data_parallel_size"]
            headless = engine.get("headless", False)
            start = engine.get("data_parallel_start_rank")
            rpc_port = engine["data_parallel_rpc_port"]
            if (engine["pipeline_parallel_size"] != 1 or type(local) is not int
                    or not 0 < local < dp or type(headless) is not bool
                    or type(rpc_port) is not int or not 0 < rpc_port < 65536):
                raise ValueError("Native DP nodes require PP1, valid local size, headless flag and RPC port")
            if headless:
                if type(start) is not int or start <= 0 or start + local > dp:
                    raise ValueError("Headless nodes require a valid nonzero native DP start rank")
            elif "data_parallel_start_rank" in engine:
                raise ValueError("An internal-DP API head must omit data_parallel_start_rank")
            for name in ("data_parallel_address", "host"):
                value = engine.get(name, "127.0.0.1")
                if type(value) is not str:
                    raise ValueError(f"{name} requires an explicit IPv4 address")
                address = ipaddress.IPv4Address(value)
                if (address.is_multicast
                        or name == "data_parallel_address" and (address.is_loopback or address.is_unspecified)
                        or name == "host" and not headless and address.is_loopback):
                    raise ValueError(f"{name} must permit inter-node communication")
    return version


def validate_observer(observer: dict, engine: dict, mode: str) -> None:
    """Check explicitly supplied recorder inputs; never infer process completeness."""
    fields = {"run_id", "source_sha", "instrumentation_sha", "config_id", "replica_id",
              "role", "expected_ranks", "output_dir", "mode", "capacity_bytes"}
    if type(observer) is not dict or set(observer) != fields:
        raise ValueError("Observer requires exact serving identity/mode/capacity fields")
    strings = fields - {"expected_ranks", "capacity_bytes"}
    if any(type(observer[key]) is not str or not observer[key] for key in strings):
        raise ValueError("Observer identity fields must be nonempty strings")
    if observer["mode"] != mode or mode not in ("clean", "light"):
        raise ValueError("Observer mode conflicts with explicit clean/light mode")
    if observer["role"] not in ("MONOLITHIC", "PREFILL", "DECODE"):
        raise ValueError("Unsupported observer role")
    reference = engine["native_reference"]
    if any(observer[key] != reference[key] for key in ("source_sha", "instrumentation_sha")):
        raise ValueError("Observer source/instrumentation conflicts with native reference")
    ranks = observer["expected_ranks"]
    if (type(ranks) is not list or not ranks or any(type(rank) is not int or rank < 0 for rank in ranks)
            or len(set(ranks)) != len(ranks)):
        raise ValueError("Observer expected_ranks must contain unique nonnegative integers")
    if len(ranks) != (engine["tensor_parallel_size"] * engine["pipeline_parallel_size"]
                      * engine["data_parallel_size"]):
        raise ValueError("Observer expected_ranks count conflicts with declared topology")
    if type(observer["capacity_bytes"]) is not int or observer["capacity_bytes"] <= 0:
        raise ValueError("Observer capacity_bytes must be a positive integer")
    output = Path(observer["output_dir"])
    if not output.is_absolute() or not output.resolve().is_relative_to("/mnt/codesign-exp/ycfeng"):
        raise ValueError("Observer output must be inside the personal cloud root")


def server_env(mode: str, engine: dict, output_dir: Path, inherited: dict,
               torch_trace_dir: Path | None = None, *, observer: dict | None = None) -> tuple[dict, dict]:
    """Return the server environment and the variables the mode set in it."""

    env = {
        key: value for key, value in inherited.items()
        if not key.startswith(("VLLM_FRONTIER_", "VLLM_TORCH_PROFILER_"))
        and key != "VLLM_CUSTOM_SCOPES_FOR_PROFILING"
        and key not in ("VLLM_ATTENTION_BACKEND", "VLLM_V1_ALLOW_NO_CHUNKED_PREFILL",
                        "VLLM_MOE_UNIFORM_ROUTING")
    }
    if runtime_version(engine) == "0.28.0":
        if mode not in ("clean", "light"):
            raise ValueError("v0.28 supports only clean/light observer modes")
        if mode == "light" and observer is None:
            raise ValueError("Light mode requires explicitly supplied observer JSON")
        mode_env = {}
        if observer is not None:
            validate_observer(observer, engine, mode)
            mode_env["VLLM_FRONTIER_OBSERVER"] = json.dumps(observer, sort_keys=True)
        return env | mode_env, mode_env
    if observer is not None:
        raise ValueError("Explicit observer JSON requires the v0.28 adapter")
    if mode not in (*MODES, "device_timeline"):
        raise ValueError(f"Unsupported legacy mode: {mode!r}")
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
    version = runtime_version(engine)
    if version == "0.28.0":
        local_model_config(model_dir)
        if engine.get("data_parallel_rpc_port") == port and not engine.get("headless"):
            raise ValueError("HTTP and DP RPC ports must be distinct")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_dir),
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", engine.get("host", "127.0.0.1") if version == "0.28.0" else "127.0.0.1",
        "--port", str(port),
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
    if version == "0.28.0":
        for name in ("attention_backend", "quantization", "shutdown_timeout",
                      "distributed_executor_backend", "data_parallel_size_local",
                      "data_parallel_start_rank", "data_parallel_address", "data_parallel_rpc_port",
                      "num_gpu_blocks_override", "worker_extension_cls", "kv_cache_dtype"):
            if name in engine:
                command += ["--" + name.replace("_", "-"), str(engine[name])]
        if "async_scheduling" in engine:
            if type(engine["async_scheduling"]) is not bool:
                raise ValueError("async_scheduling must be boolean")
            command += ["--async-scheduling" if engine["async_scheduling"] else "--no-async-scheduling"]
        if engine.get("headless"):
            command.append("--headless")
    return command


def local_model_config(model_dir: Path) -> dict:
    """Read supported original-model metadata without preparing or downloading weights."""
    if not model_dir.is_absolute() or not (model_dir / "config.json").is_file():
        raise ValueError("v0.28 requires an absolute local metadata directory with config.json")
    config = json.loads((model_dir / "config.json").read_text())
    supported = {"GlmMoeDsaForCausalLM", "DeepseekV4ForCausalLM"}
    if (type(config) is not dict or type(config.get("architectures")) is not list
            or len(config["architectures"]) != 1 or config["architectures"][0] not in supported):
        raise ValueError("Native metadata must identify original GLM or DeepSeek V4")
    if type(config.get("vocab_size")) is not int or config["vocab_size"] <= 200:
        raise ValueError("Local metadata requires an integer vocab_size above the prompt generator's 200-token margin")
    return config


def observer_evidence(engine: dict, observer: dict | None) -> dict:
    """Retain file identities and declared process expectations, without raw-trace admission."""
    files = []
    directory = Path(observer["output_dir"]) if observer is not None else None
    errors = []
    if directory is not None:
        try:
            for path in sorted(directory.rglob("*.jsonl")):
                digest = hashlib.sha256()
                size = 0
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                files.append({"path": str(path), "bytes": size,
                              "sha256": digest.hexdigest()})
        except OSError as error:
            errors.append(f"{type(error).__name__}: {error}")
    return {
        "declared_native_reference": engine["native_reference"],
        "installed_runtime_verified": False, "qualification_verified": False,
        "process_expectations": {
            "declared_dp_size": engine["data_parallel_size"],
            "declared_worker_ranks": None if observer is None else observer["expected_ranks"],
            "engine_owner_inventory": None, "worker_inventory": None,
            "inventory_verified": False,
        },
        "observer": observer, "trace_directory_exists": directory is not None and directory.is_dir(),
        "trace_files": files, "collection_errors": errors,
        "native_close_and_device_completion": "unverified",
    }


def wait_until_ready(server: subprocess.Popen | None, port: int, timeout_s: float, *,
                     host: str = "127.0.0.1") -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"vllm serve exited with {server.returncode} during startup")
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except OSError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"vllm serve not ready after {timeout_s} s")


async def sse_events(content):
    """Yield complete SSE data events and their client receipt timestamps."""
    pending = b""
    data = []
    async for chunk in content.iter_any():
        received = time.monotonic()
        pending += chunk
        while match := re.search(rb"\r\n|\r|\n", pending):
            if pending[match.start():] == b"\r":
                break  # A CRLF delimiter can itself straddle network chunks.
            line, pending = pending[:match.start()], pending[match.end():]
            if not line:
                if data:
                    yield b"\n".join(data).decode("utf-8"), received
                    data = []
            elif line.startswith(b"data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(b" ") else value)
    if pending or data:
        raise ValueError("Unterminated SSE event at EOF")


async def read_token_stream(response, record):
    """Read v0.28 delta token IDs, never treating text or usage as tokens."""
    record.update(output_token_count=0, stream_done=False,
                  first_token_monotonic=None, last_token_monotonic=None)
    async for data, received in sse_events(response.content):
        if record["stream_done"]:
            raise ValueError("SSE data after DONE")
        if data == "[DONE]":
            record["stream_done"] = True
            continue
        payload = json.loads(data)
        if "error" in payload:
            record["error"] = {"type": "STREAM_ERROR", "detail": payload["error"]}
            continue
        if payload.get("usage") is not None:
            record["usage"] = payload["usage"]
        for choice in payload.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Expected one completion choice at index zero")
            tokens = choice.get("token_ids")
            if tokens is not None:
                if type(tokens) is not list or any(type(token) is not int or token < 0 for token in tokens):
                    raise ValueError("Invalid delta token_ids")
                if tokens:
                    if record["first_token_monotonic"] is None:
                        record["first_token_monotonic"] = received
                    record["last_token_monotonic"] = received
                    record["output_token_count"] += len(tokens)
            if choice.get("finish_reason") is not None:
                record["finish_reason"] = choice["finish_reason"]
    errors = record["validation_errors"]
    if not record["stream_done"]:
        errors.append("Missing SSE DONE")
    if record["output_token_count"] != record["num_decode_tokens"]:
        errors.append("Delta output token count mismatch")
    usage = record.get("usage") or {}
    expected = {"prompt_tokens": record["num_prefill_tokens"],
                "completion_tokens": record["num_decode_tokens"],
                "total_tokens": record["num_prefill_tokens"] + record["num_decode_tokens"]}
    if any(type(usage.get(key)) is not int or usage[key] != value for key, value in expected.items()):
        errors.append("Native usage token count mismatch or missing usage")
    if record.get("finish_reason") != "length":
        errors.append("Missing length finish reason")
    if record["first_token_monotonic"] is None:
        errors.append("No actual output token observed")


def request_records(rows: list[dict]) -> list[dict]:
    """Allocate one mutable record per planned request before any submission."""
    return [row | {"segment": row.get("segment", row["role"]), "dispatch_monotonic": None,
                   "http_status": None, "status": "not_submitted", "validation_errors": []}
            for row in rows]


async def replay(rows: list[dict], port: int, vocab_size: int, origin_lead_s: float,
                 profile_window_s: tuple[float, float] | None = None, *,
                 vllm_version: str = "0.10.2", records: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Post every row at its arrival offset; with profile_window_s, also open and
    close the server's torch profiler at those two offsets of the same clock.

    Callers retaining partial results pass one preallocated request record per row.
    """
    import aiohttp

    url = f"http://127.0.0.1:{port}/v1/completions"
    bodies = {
        row["request_id"]: json.dumps({
            "model": SERVED_MODEL_NAME,
            "prompt": prompt_token_ids(row.get("template_request_id", row["request_id"]),
                                       row["num_prefill_tokens"], vocab_size),
            "max_tokens": row["num_decode_tokens"],
            "min_tokens": row["num_decode_tokens"],
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": vllm_version == "0.28.0",
            **({"return_token_ids": True, "stream_options": {"include_usage": True}}
               if vllm_version == "0.28.0" else {}),
        })
        for row in rows
    }
    if records is None:
        records = request_records(rows)
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        origin = time.monotonic() + origin_lead_s
        origin_wall = time.time() + origin_lead_s

        async def post(record: dict) -> None:
            if vllm_version == "0.10.2":
                record["engine_request_id"] = f"cmpl-{record['request_id']}-0"
            try:
                due = origin + record["arrived_at"]
                await asyncio.sleep(max(0.0, due - time.monotonic()))
                dispatched = time.monotonic()
                record.update(dispatch_monotonic=dispatched, dispatch_offset_s=dispatched - origin,
                              send_lateness_s=dispatched - due, status="submitted")
                async with session.post(
                    url, data=bodies[record["request_id"]],
                    headers={"Content-Type": "application/json", "X-Request-Id": record["request_id"]},
                ) as response:
                    record["http_status"] = response.status
                    if response.status != 200:
                        record["error"] = {"type": "HTTP_ERROR", "body": await response.text()}
                    elif vllm_version == "0.28.0":
                        await read_token_stream(response, record)
                    else:
                        payload = await response.json(content_type=None)
                        record["usage"] = payload.get("usage")
                        record["finish_reason"] = payload["choices"][0].get("finish_reason")
            except asyncio.CancelledError:
                record["error"] = {"type": "CancelledError", "message": "Request unfinished at client cancellation"}
                raise
            except Exception as error:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
            finally:
                record["response_monotonic"] = time.monotonic()
                record["status"] = ("unfinished" if record.get("error", {}).get("type") == "CancelledError"
                                    else "failed" if record.get("error") or record["validation_errors"] else "completed")
                first, last = record.get("first_token_monotonic"), record.get("last_token_monotonic")
                if first is not None and last is not None:
                    record.update(ttft_ms=(first - record["dispatch_monotonic"]) * 1000,
                                  e2e_ms=(last - record["dispatch_monotonic"]) * 1000,
                                  tpot_ms=((last - first) * 1000 / (record["num_decode_tokens"] - 1)
                                           if record["num_decode_tokens"] > 1 else None))

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

        posts = [post(record) for record in records]
        if profile_window_s is not None:
            posts.append(profile(*profile_window_s))
        beat = asyncio.ensure_future(heartbeat())
        try:
            await asyncio.gather(*posts)
        finally:
            beat.cancel()
    clock = {"origin_monotonic": origin, "origin_wall": origin_wall}
    if profile_window_s is not None:
        clock["profile_marks"] = profile_marks
    return clock, records


def stop_server(server: subprocess.Popen, timeout_s: float) -> dict:
    if server.poll() is not None:
        return {"signal": None, "returncode": server.returncode}
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
    parser.add_argument("--mode", choices=(*MODES, "device_timeline", "light"), required=True)
    parser.add_argument("--observer-config", type=Path,
                        help="Explicit v0.28 serving recorder JSON; never inherited")
    parser.add_argument("--model-dir", type=Path,
                        help="v0.28: existing local GLM metadata directory")
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
    version = runtime_version(engine)
    observer = json.loads(args.observer_config.read_text()) if args.observer_config else None
    env, mode_env = server_env(args.mode, engine, args.output_dir, dict(os.environ),
                               args.torch_trace_dir, observer=observer)
    if observer is not None and observer["role"] != "MONOLITHIC":
        raise ValueError("DP replay requires MONOLITHIC observer role")
    if version == "0.28.0":
        if args.model_dir is None:
            parser.error("v0.28 requires --model-dir with local GLM metadata")
        model_config = local_model_config(args.model_dir)
        model_dir = args.model_dir
    elif args.model_dir is not None:
        parser.error("--model-dir requires the explicit v0.28 adapter")
    request_ids = json.loads((args.trace_dir / "request_ids.json").read_text())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if version == "0.10.2":
        model_dir = output_dir / "model"
        model_config = write_model_dir(
            REPO_ROOT / "data/config/models" / f"{engine['model_name']}.json", model_dir,
        )
    profile_window_s = None
    if args.mode == "device_timeline":
        first_formal = min(row["arrived_at"] for row in request_ids["rows"] if row["role"] == "formal")
        start = first_formal + args.profile_start_s
        profile_window_s = (start, start + args.profile_duration_s)
    command = server_command(engine, model_dir, args.port)
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
    if version == "0.28.0":
        summary.update(vllm_version=version, qualification_verified=False,
                       model_config_sha256=hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest())
    records = request_records(request_ids["rows"])
    with (output_dir / "server.log").open("w") as server_log:
        server = None
        try:
            server = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
            summary["startup_s"] = wait_until_ready(server, args.port, args.startup_timeout_s)
            clock, records = asyncio.run(replay(
                request_ids["rows"], args.port, int(model_config["vocab_size"]), args.origin_lead_s,
                profile_window_s, vllm_version=version,
                records=records,
            ))
            summary.update(clock)
            time.sleep(args.drain_s)
        except BaseException as error:
            summary["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            if server is not None:
                summary["server_stop"] = stop_server(server, args.stop_timeout_s)
            if version == "0.28.0":
                summary["observer_evidence"] = observer_evidence(engine, observer)
            (output_dir / "client_requests.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=1))
    records.sort(key=lambda record: record["dispatch_monotonic"])
    (output_dir / "client_requests.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    failed = [record["request_id"] for record in records if record["status"] != "completed"]
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
