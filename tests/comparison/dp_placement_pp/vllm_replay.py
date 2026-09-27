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
switches; none is inherited from the worker. Both modes write the placement
records to ``dp_placement/``, declared scheduler-level workflow evidence.

``clean``
    E2E request metrics on (``request_metrics.jsonl``), operator probes off.
``instrumented``
    Instrumentation on with the MoE routing records (``moe_routing.jsonl``),
    E2E request metrics off.

The attention backend is an engine setting: ``VLLM_ATTENTION_BACKEND`` is set
from the engine file's ``attention_backend`` and is otherwise left to vLLM's
own selection. Outputs in ``--output-dir``: ``server.log``,
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
MODES = ("clean", "instrumented")
KV_CACHE_LINE = re.compile(r"\((EngineCore_DP\d+) pid=\d+\).*GPU KV cache size: ([\d,]+) tokens")


def server_env(mode: str, engine: dict, output_dir: Path, inherited: dict) -> tuple[dict, dict]:
    """Return the server environment and the variables the mode set in it."""

    env = {
        key: value for key, value in inherited.items()
        if not key.startswith("VLLM_FRONTIER_") and key != "VLLM_ATTENTION_BACKEND"
    }
    mode_env = {"VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR": str(output_dir / "dp_placement")}
    if mode == "clean":
        mode_env["VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH"] = str(output_dir / "request_metrics.jsonl")
    else:
        mode_env["VLLM_FRONTIER_INSTRUMENTATION"] = "1"
        mode_env["VLLM_FRONTIER_MOE_ROUTING_LOG_PATH"] = str(output_dir / "moe_routing.jsonl")
    if "attention_backend" in engine:
        mode_env["VLLM_ATTENTION_BACKEND"] = engine["attention_backend"]
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
    return command + [flag for flag, enabled in flags.items() if enabled]


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


async def replay(rows: list[dict], port: int, vocab_size: int, origin_lead_s: float) -> tuple[dict, list[dict]]:
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

        await asyncio.gather(*(post(row) for row in rows))
    return {"origin_monotonic": origin, "origin_wall": origin_wall}, records


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
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--origin-lead-s", type=float, default=1.0)
    parser.add_argument("--drain-s", type=float, default=3.0)
    parser.add_argument("--stop-timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)

    engine = json.loads(args.engine_config.read_text())
    request_ids = json.loads((args.trace_dir / "request_ids.json").read_text())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = write_model_dir(
        REPO_ROOT / "data/config/models" / f"{engine['model_name']}.json",
        output_dir / "model",
    )

    env, mode_env = server_env(args.mode, engine, output_dir, dict(os.environ))
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
    }
    with (output_dir / "server.log").open("w") as server_log:
        server = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            summary["startup_s"] = wait_until_ready(server, args.port, args.startup_timeout_s)
            clock, records = asyncio.run(replay(
                request_ids["rows"], args.port, int(model_config["vocab_size"]), args.origin_lead_s
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
