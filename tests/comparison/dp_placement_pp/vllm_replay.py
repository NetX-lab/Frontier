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
STEP5_IMAGE = ("a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:"
               "69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
# The locked Step-5 image reports this version; it is recorded evidence, not a check on the
# installed package, because a served container cannot be inspected from the client node.
STEP5_INSTALLED = "0.1.dev22912+g8224bb82f"
KV_CACHE_LINE = re.compile(r"\((EngineCore_DP\d+) pid=\d+\).*GPU KV cache size: ([\d,]+) tokens")
NATIVE_NODE_FIELDS = ("data_parallel_size_local", "data_parallel_address", "data_parallel_rpc_port",
                      "data_parallel_start_rank", "headless")
# Native-serving engine keys passed through to the server command as flags.
NATIVE_SERVING_FIELDS = ("attention_backend", "quantization", "shutdown_timeout",
                         "distributed_executor_backend", "data_parallel_size_local",
                         "data_parallel_start_rank", "data_parallel_address",
                         "data_parallel_rpc_port", "num_gpu_blocks_override")

# One record per explicit harness adapter. `connector` selects the P/D transfer protocol,
# `native_serving` whether the record serves a pinned native image (which enables the native
# engine checks, the node flags and the local model metadata), and `model_metadata` names the
# metadata files, the architecture and the key path of `vocab_size` in `config.json`.
RUNTIME_ADAPTERS = {
    "0.10.2": {
        "installed_version": "0.10.2",
        "connector": "p2p_nccl",
        "native_serving": False,
        "modes": (*MODES, "device_timeline"),
        "model_metadata": None,
        "native_reference": None,
        "skip_tokenizer_init": None,
        "kv_connector_extra_config": {},
    },
    "0.28.0": {
        "installed_version": "0.28.0",
        "connector": "nixl_pull",
        "native_serving": True,
        "modes": ("clean", "light"),
        "model_metadata": {"files": ("config.json",),
                           "architecture": "GlmMoeDsaForCausalLM",
                           "vocab_size_keys": ("vocab_size",)},
        "native_reference": {"source_sha": V028_SOURCE,
                             "instrumentation_sha": re.compile(r"[0-9a-f]{40}"),
                             "image": V028_IMAGE},
        "skip_tokenizer_init": True,
        "kv_connector_extra_config": {},
    },
    # Step-5-Preview transfers sparse-summary state, so the image accepts only a CUDA-buffer
    # NixlConnector or MooncakeStoreConnector; this record keeps the pull connector. The image
    # carries no Frontier instrumentation, so only clean mode is legal. The text configuration
    # nests `vocab_size` under `text_config`, and the tokenizer stays initialized because the
    # `step5_vl` processor needs it at startup.
    "step5": {
        "installed_version": STEP5_INSTALLED,
        "connector": "nixl_pull",
        "native_serving": True,
        "modes": ("clean",),
        "model_metadata": {"files": ("config.json", "tokenizer_config.json", "tokenizer.json"),
                           "architecture": "Step5ForConditionalGeneration",
                           "vocab_size_keys": ("text_config", "vocab_size")},
        "native_reference": {"image": STEP5_IMAGE},
        "skip_tokenizer_init": False,
        # Verified against the pinned image: base_worker defaults to UCX and a 30 s lease.
        "kv_connector_extra_config": {"backends": ["UCX"], "kv_lease_duration": 30},
    },
}


def adapter_record(version: str) -> dict:
    """Return the declared adapter for a harness version label."""
    if version not in RUNTIME_ADAPTERS:
        raise ValueError(f"Unsupported vLLM harness version: {version!r}")
    return RUNTIME_ADAPTERS[version]


def uses_nixl_pull(version: str) -> bool:
    """Whether the adapter transfers KV by having the decode instance read prefill blocks."""
    return adapter_record(version)["connector"] == "nixl_pull"


def native_serving(version: str) -> bool:
    """Whether the adapter serves a pinned native image instead of the legacy instrumented one."""
    return adapter_record(version)["native_serving"]


def check_native_reference(reference, expected: dict) -> None:
    """Require exactly the declared reference fields; a pattern value must match in full."""
    if type(reference) is not dict or set(reference) != set(expected):
        raise ValueError(f"Adapter requires the locked native reference {sorted(expected)}")
    for name, wanted in expected.items():
        value = reference[name]
        matches = (type(value) is str and wanted.fullmatch(value) is not None
                   if isinstance(wanted, re.Pattern)
                   else value == wanted)
        if not matches:
            raise ValueError(f"Adapter rejects native reference {name}: {value!r}")


def runtime_version(engine: dict) -> str:
    """Select the explicit harness adapter without claiming installed-runtime identity."""
    if type(engine) is not dict:
        raise ValueError("Engine configuration must be a JSON object")
    version = engine.get("vllm_version", "0.10.2")
    adapter = adapter_record(version)
    if not adapter["native_serving"]:
        return version
    check_native_reference(engine.get("native_reference"), adapter["native_reference"])
    expected_init = adapter["skip_tokenizer_init"]
    if (engine.get("load_format") != "dummy"
            or engine.get("skip_tokenizer_init") is not expected_init):
        raise ValueError(
            f"{version} harness requires dummy weights and skip_tokenizer_init={expected_init}")
    if engine.get("moe_uniform_routing") or engine.get("enable_return_routed_experts"):
        raise ValueError(f"{version} routing content and forced routing must remain OFF")
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
    if any(name in engine for name in NATIVE_NODE_FIELDS):
        if (engine.get("data_parallel_rank") is not None or engine.get("data_parallel_hybrid_lb")
                or engine.get("data_parallel_external_lb")):
            raise ValueError("Native DP node replay supports internal load balancing only")
        if any(name not in engine for name in NATIVE_NODE_FIELDS[:3]):
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
    if native_serving(runtime_version(engine)):
        adapter = adapter_record(runtime_version(engine))
        if mode not in adapter["modes"]:
            raise ValueError(f"The native adapter supports only {'/'.join(adapter['modes'])} modes")
        if mode == "light" and observer is None:
            raise ValueError("Light mode requires explicitly supplied observer JSON")
        if observer is not None and "instrumentation_sha" not in adapter["native_reference"]:
            raise ValueError("Explicit observer JSON requires an instrumented native reference")
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
    native = native_serving(version)
    if native:
        local_model_config(model_dir, version)
        if engine.get("data_parallel_rpc_port") == port and not engine.get("headless"):
            raise ValueError("HTTP and DP RPC ports must be distinct")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_dir),
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", engine.get("host", "127.0.0.1") if native else "127.0.0.1",
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
    if native:
        for name in NATIVE_SERVING_FIELDS:
            if name in engine:
                command += ["--" + name.replace("_", "-"), str(engine[name])]
        if "async_scheduling" in engine:
            if type(engine["async_scheduling"]) is not bool:
                raise ValueError("async_scheduling must be boolean")
            command += ["--async-scheduling" if engine["async_scheduling"] else "--no-async-scheduling"]
        if "cudagraph_metrics" in engine:
            if type(engine["cudagraph_metrics"]) is not bool:
                raise ValueError("cudagraph_metrics must be boolean")
            command += ["--cudagraph-metrics" if engine["cudagraph_metrics"] else "--no-cudagraph-metrics"]
        if "api_server_count" in engine and not engine.get("headless"):
            command += ["--api-server-count", str(engine["api_server_count"])]
        if engine.get("headless"):
            command.append("--headless")
    return command


def local_model_config(model_dir: Path, version: str = "0.28.0") -> dict:
    """Read the local metadata the adapter declares; this harness never prepares weights."""
    metadata = adapter_record(version)["model_metadata"]
    if metadata is None:
        raise ValueError(f"The {version} adapter does not read local model metadata")
    if not model_dir.is_absolute() or not all((model_dir / name).is_file()
                                             for name in metadata["files"]):
        raise ValueError("A native adapter requires an absolute local metadata directory with "
                         + ", ".join(metadata["files"]))
    config = json.loads((model_dir / "config.json").read_text())
    if type(config) is not dict or config.get("architectures") != [metadata["architecture"]]:
        raise ValueError(f"The {version} adapter prepares only {metadata['architecture']}")
    value = config
    for key in metadata["vocab_size_keys"]:
        value = value.get(key) if type(value) is dict else None
    if type(value) is not int or value <= 200:
        raise ValueError("Local metadata requires an integer vocab_size above the prompt "
                         "generator's 200-token margin")
    # Downstream replay reads a single scalar, which Step-5 declares inside its text config.
    return config | {"vocab_size": value}


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


async def read_completion_stream(response, *, cancel_after_first_content: bool = False,
                                 return_token_ids: bool = False,
                                 cancel_decode_client: bool = False) -> tuple[dict, dict]:
    """Read OpenAI completion SSE lines and time the first nonempty text choice.

    aiohttp's line iterator joins fragmented network reads, including UTF-8 bytes. Empty
    choices, SSE comments and the final usage event are not content. All timestamps use
    the client's host-wide monotonic clock, also used by its local P/D proxy.
    """
    payload = {"choices": [{"finish_reason": None}], "usage": None}
    timing = {"stream": True, "first_token_monotonic": None, "stream_completed": False}
    if return_token_ids:
        timing.update(last_token_monotonic=None, token_id_deltas=[], generated_token_count=0,
                      accepted_prompt_token_count=None, accepted_prompt_sha256=None)
    async for line in response.content:
        arrived = time.monotonic()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            timing["stream_completed"] = True
            continue
        event = json.loads(data)
        if cancel_decode_client and event.get("step5_decode_cancelled") is True:
            timing.update(client_cancelled=True, cancel_origin="proxy_native_decode_client")
            break
        if "error" in event:
            timing["stream_error"] = event["error"]
        if event.get("usage") is not None:
            payload["usage"] = event["usage"]
        for choice in event.get("choices", []):
            if return_token_ids and choice.get("prompt_token_ids") is not None:
                prompt = choice["prompt_token_ids"]
                timing["accepted_prompt_token_count"] = len(prompt)
                timing["accepted_prompt_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            content = choice.get("token_ids") if return_token_ids else choice.get("text")
            if return_token_ids and content:
                if (choice.get("index", 0) != 0 or type(content) is not list
                        or any(type(token) is not int or token < 0 for token in content)):
                    raise ValueError("Expected one completion choice with native token-ID deltas")
                timing["token_id_deltas"].append({"monotonic": arrived, "token_ids": content})
                timing["generated_token_count"] += len(content)
                timing["last_token_monotonic"] = arrived
            if content and timing["first_token_monotonic"] is None:
                timing["first_token_monotonic"] = arrived
                if cancel_after_first_content:
                    timing["client_cancelled"] = True
            if choice.get("finish_reason") is not None:
                payload["choices"][0]["finish_reason"] = choice["finish_reason"]
        if timing.get("client_cancelled"):
            break
    return payload, timing


async def replay(rows: list[dict], port: int, vocab_size: int, origin_lead_s: float,
                  profile_window_s: tuple[float, float] | None = None, *,
                  vllm_version: str = "0.10.2", records: list[dict] | None = None,
                  checkpoint=None, completion_url: str | None = None) -> tuple[dict, list[dict]]:
    """Post every row at its arrival offset; with profile_window_s, also open and
    close the server's torch profiler at those two offsets of the same clock.
    An optional prompt_id repeats identical token IDs under distinct request IDs.
    Rows with stream=true opt into client first-content timing and final usage events;
    rows without it keep the existing nonstreaming body and record format."""
    import aiohttp

    url = completion_url or f"http://127.0.0.1:{port}/v1/completions"
    bodies = {
        row["request_id"]: json.dumps({
            "model": SERVED_MODEL_NAME,
            "prompt": prompt_token_ids(row.get("prompt_id", row["request_id"]),
                                       row["num_prefill_tokens"], vocab_size),
            "max_tokens": row["num_decode_tokens"],
            "min_tokens": row["num_decode_tokens"],
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": row.get("stream", False),
             **({"stream_options": {"include_usage": True}} if row.get("stream") else {}),
            **({"return_token_ids": True, "add_special_tokens": False, "echo": False}
               if row.get("return_token_ids") else {}),
        })
        for row in rows
    }
    if records is None:
        records = []
    planned = {}
    for row in rows:
        if row.get("return_token_ids"):
            record = dict(row, status="scheduled", http_status=None, dispatch_monotonic=None,
                          response_monotonic=None, first_token_monotonic=None,
                          last_token_monotonic=None)
            record["prompt_sha256"] = hashlib.sha256(
                json.dumps(json.loads(bodies[row["request_id"]])["prompt"]).encode()).hexdigest()
            planned[row["request_id"]] = record
            records.append(record)

    def persist():
        if checkpoint is not None:
            try:
                checkpoint()
            except Exception as error:
                print("NATIVE_PD_AUXILIARY_UNVERIFIED checkpoint", repr(error), flush=True)

    persist()
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        origin = time.monotonic() + origin_lead_s
        origin_wall = time.time() + origin_lead_s

        async def post(row: dict) -> None:
            await asyncio.sleep(max(0.0, origin + row["arrived_at"] - time.monotonic()))
            dispatched = time.monotonic()
            timing = {"stream": True} if row.get("stream") else {}
            record = planned.get(row["request_id"])
            if record is not None:
                record.update(status="dispatched", dispatch_monotonic=dispatched,
                              dispatch_offset_s=dispatched - origin,
                              dispatch_lateness_s=dispatched - origin - row["arrived_at"])
                persist()
            try:
                async with session.post(
                    url, data=bodies[row["request_id"]],
                    headers={"Content-Type": "application/json", "X-Request-Id": row["request_id"],
                             **({"X-Step5-Cancel-Decode-After-S": str(row["cancel_after_decode_submit_s"])}
                                if "cancel_after_decode_submit_s" in row else {})},
                ) as response:
                    if record is not None:
                        record["http_status"] = response.status
                    if row.get("stream") and response.status == 200:
                        options = {"cancel_after_first_content": row.get("cancel_after_first_content", False)}
                        if "cancel_after_decode_submit_s" in row:
                            options["cancel_decode_client"] = True
                        if row.get("return_token_ids"):
                            options["return_token_ids"] = True
                        payload, timing = await read_completion_stream(response, **options)
                    else:
                        payload = await response.json(content_type=None)
                    if record is not None and type(payload) is not dict:
                        raise ValueError("Native completion payload must be an object")
            except Exception as error:
                if record is None:
                    raise
                record.update(status="failed", error=f"{type(error).__name__}: {error}",
                              response_monotonic=time.monotonic())
                persist()
                return
            if record is not None:
                record.update(timing, response_monotonic=time.monotonic(),
                              usage=payload.get("usage"),
                              finish_reason=(payload.get("choices") or [{}])[0].get("finish_reason"),
                              status="received" if response.status == 200 else "failed")
                if response.status != 200:
                    record["error"] = payload
                persist()
                return
            record = {
                "request_id": row["request_id"],
                "segment": row["segment"],
                "role": row["role"],
                "arrived_at": row["arrived_at"],
                "num_prefill_tokens": row["num_prefill_tokens"],
                "num_decode_tokens": row["num_decode_tokens"],
                "dispatch_offset_s": dispatched - origin,
                "dispatch_monotonic": dispatched,
                "response_monotonic": time.monotonic(),
                "http_status": response.status,
                **timing,
            }
            if not native_serving(vllm_version):
                record["engine_request_id"] = f"cmpl-{row['request_id']}-0"
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
                        help="Explicit native serving recorder JSON; never inherited")
    parser.add_argument("--model-dir", type=Path,
                        help="native adapters: existing local metadata directory")
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
    native = native_serving(version)
    observer = json.loads(args.observer_config.read_text()) if args.observer_config else None
    env, mode_env = server_env(args.mode, engine, args.output_dir, dict(os.environ),
                               args.torch_trace_dir, observer=observer)
    if observer is not None and observer["role"] != "MONOLITHIC":
        raise ValueError("DP replay requires MONOLITHIC observer role")
    if native:
        if args.model_dir is None:
            parser.error(f"{version} requires --model-dir with the local metadata it declares")
        model_config = local_model_config(args.model_dir, version)
        model_dir = args.model_dir
    elif args.model_dir is not None:
        parser.error("--model-dir requires a native adapter")
    request_ids = json.loads((args.trace_dir / "request_ids.json").read_text())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not native:
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
    if native:
        summary.update(vllm_version=version, qualification_verified=False,
                       model_config_sha256=hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest())
    records = []
    with (output_dir / "server.log").open("w") as server_log:
        server = None
        try:
            server = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
            summary["startup_s"] = wait_until_ready(server, args.port, args.startup_timeout_s)
            clock, records = asyncio.run(replay(
                request_ids["rows"], args.port, int(model_config["vocab_size"]), args.origin_lead_s,
                profile_window_s, vllm_version=version,
            ))
            summary.update(clock)
            time.sleep(args.drain_s)
        except BaseException as error:
            summary["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            if server is not None:
                summary["server_stop"] = stop_server(server, args.stop_timeout_s)
            if native:
                summary["observer_evidence"] = observer_evidence(engine, observer)
            (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=1))
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
