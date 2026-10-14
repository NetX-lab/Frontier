#!/usr/bin/env python3
"""Serve a 1P1D vLLM pair joined by the P2pNccl connector and replay a trace through pd_proxy.py.

Runs inside the ``vllm/vllm-openai:v0.10.2`` image with the instrumented
overlay first on ``PYTHONPATH`` (run_pd_worker.sh). The deployment follows
calibration decisions Q4 and Q5: a prefill instance on the first visible GPU and
a decode instance on the second, each a ``vllm serve`` of the engine file, joined
by P2pNcclConnector with ``send_type`` PUT_ASYNC, with pd_proxy.py in front.

The connector addresses are static. Each instance binds its ZMQ socket on
``127.0.0.1:<kv_port>``, and the proxy writes both addresses into every proxied
request id, so the connector's proxy discovery (``proxy_ip``/``proxy_port``)
stays off. The engine file's ``kv_transfer`` block sets the connector values, and
none has a default here:

``kv_buffer_size``
    bytes of received KV the decode instance holds on its GPU; beyond it the
    connector moves received tensors to a pinned host pool.
``mem_pool_size_gb``
    size of that pool, which each instance allocates at start.
``prefill_kv_port``, ``decode_kv_port``
    ZMQ ports of the two instances.
``nccl_num_channels``
    NCCL channels of the P2P communicator.

The trace is replayed with ``vllm_replay.replay`` against the proxy, and the
run mode sets the Frontier switches of both instances with
``vllm_replay.server_env``. Every mode also writes each instance's per-layer
KV-transfer log (``VLLM_FRONTIER_KV_TRANSFER_LOG_PATH``), declared PD workflow
evidence.

After the replay each server log must hold the connector's init line with
PUT_ASYNC and no line of ``CONNECTOR_FAULTS``, the pool fallback among them. In
clean mode pd_metrics.py then joins the two instances' request metrics with the
proxy log. Outputs in ``--output-dir``: ``prefill/`` and ``decode/``
(``server.log``, the mode's record file, ``kv_transfer.jsonl``,
``dp_placement/``), ``proxy.log``, ``proxy_requests.jsonl``,
``client_requests.jsonl``, ``pd_request_metrics.jsonl`` and
``pd_join_report.json`` (clean mode), ``replay_summary.json``.

Exit status: 0 pass, 4 a request failed, 5 a connector check failed, 6 the
clean-mode join is incomplete.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HARNESS_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(HARNESS_DIR), str(HARNESS_DIR.parent / "dp_placement_pp")]
from pd_metrics import engine_request_id, write_pd_rows  # noqa: E402
from pd_proxy import proxied_request_id  # noqa: E402
from vllm_replay import (  # noqa: E402
    MODES, REPO_ROOT, kv_cache_tokens, replay, server_command, server_env, stop_server,
    wait_until_ready, write_model_dir,
)

ROLES = ("prefill", "decode")
KV_ROLES = {"prefill": "kv_producer", "decode": "kv_consumer"}
KV_IP = "127.0.0.1"
KV_TRANSFER_FIELDS = ("kv_buffer_size", "mem_pool_size_gb", "prefill_kv_port", "decode_kv_port",
                      "nccl_num_channels")
CONNECTOR_INIT = "P2pNcclEngine init"
# Connector log lines after which a KV tensor did not go GPU to GPU into the
# decode instance's cache as it was sent (vLLM-BS ff328f82a,
# vllm/distributed/kv_transfer/kv_connector/v1/p2p/).
CONNECTOR_FAULTS = {
    "[PUT]Recv Tensor, Out Of Threshold": "received tensor moved to the pinned host pool",
    "[PUT]Recv Tensor, Out Of Memory": "decode instance could not allocate a received tensor",
    "Send Tensor, Peer Out Of Memory/Threshold": "decode instance rejected a sent tensor",
    "[PUT]Recv From": "decode instance read an empty tensor",
    "kv_cache is None": "a layer was left without received KV",
    "kv_cache does not match": "received block count differs from the decode allocation",
}


def kv_transfer_config(engine: dict, role: str, http_port: int) -> dict:
    kv = engine["kv_transfer"]
    return {
        "kv_connector": "P2pNcclConnector",
        "kv_role": KV_ROLES[role],
        "kv_buffer_size": kv["kv_buffer_size"],
        "kv_ip": KV_IP,
        "kv_port": kv[f"{role}_kv_port"],
        # The connector reads these as strings (os.environ, int(), float()),
        # as in vLLM's disaggregated_serving_p2p_nccl_xpyd example.
        "kv_connector_extra_config": {
            "http_port": str(http_port),
            "send_type": "PUT_ASYNC",
            "nccl_num_channels": str(kv["nccl_num_channels"]),
            "mem_pool_size_gb": str(kv["mem_pool_size_gb"]),
        },
    }


def instance_devices(inherited: dict) -> dict[str, str]:
    """Prefill takes the first visible GPU, decode the second."""

    # Without CUDA_VISIBLE_DEVICES, CUDA numbers the visible GPUs from 0.
    visible = inherited.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else ["0", "1"]
    if len(devices) < 2:
        raise SystemExit(f"1P1D needs two GPUs, CUDA_VISIBLE_DEVICES={visible!r}")
    return dict(zip(ROLES, devices))


def instance_env(mode: str, engine: dict, role_dir: Path, inherited: dict,
                 device: str) -> tuple[dict, dict]:
    """Return one instance's environment and the variables the harness set in it."""

    env, set_env = server_env(mode, engine, role_dir, inherited)
    role_env = {
        "CUDA_VISIBLE_DEVICES": device,
        "VLLM_FRONTIER_KV_TRANSFER_LOG_PATH": str(role_dir / "kv_transfer.jsonl"),
    }
    return env | role_env, set_env | role_env


def connector_check(server_log: Path) -> dict:
    lines = server_log.read_text(errors="replace").splitlines()
    init_lines = [line for line in lines if CONNECTOR_INIT in line]
    faults = {pattern: sum(pattern in line for line in lines) for pattern in CONNECTOR_FAULTS}
    return {
        "init_lines": init_lines,
        "faults": {pattern: count for pattern, count in faults.items() if count},
        "passed": bool(init_lines)
        and all("send_type:PUT_ASYNC" in line for line in init_lines)
        and not any(faults.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--decode-port", type=int, default=8200)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--origin-lead-s", type=float, default=1.0)
    parser.add_argument("--drain-s", type=float, default=3.0)
    parser.add_argument("--stop-timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)

    engine = json.loads(args.engine_config.read_text())
    missing = [name for name in KV_TRANSFER_FIELDS if name not in engine.get("kv_transfer", {})]
    if missing:
        raise SystemExit(f"{args.engine_config}: kv_transfer lacks {missing}")
    rows = json.loads((args.trace_dir / "request_ids.json").read_text())["rows"]
    kv_addresses = {role: f"{KV_IP}:{engine['kv_transfer'][f'{role}_kv_port']}" for role in ROLES}
    # Every trace id must survive the connector's address parsing before a GPU is used.
    proxied = {
        row["request_id"]: proxied_request_id(row["request_id"], kv_addresses["prefill"],
                                              kv_addresses["decode"])
        for row in rows
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = write_model_dir(
        REPO_ROOT / "data/config/models" / f"{engine['model_name']}.json",
        output_dir / "model",
    )
    devices = instance_devices(dict(os.environ))
    ports = {"prefill": args.prefill_port, "decode": args.decode_port}
    proxy_command = [
        sys.executable, str(HARNESS_DIR / "pd_proxy.py"), "--port", str(args.proxy_port),
        "--prefill-url", f"http://127.0.0.1:{ports['prefill']}",
        "--decode-url", f"http://127.0.0.1:{ports['decode']}",
        "--prefill-kv-address", kv_addresses["prefill"],
        "--decode-kv-address", kv_addresses["decode"],
        "--log-path", str(output_dir / "proxy_requests.jsonl"),
    ]
    summary: dict = {
        "mode": args.mode,
        "engine_config": engine,
        "trace_dir": str(args.trace_dir),
        "devices": devices,
        "kv_addresses": kv_addresses,
        "proxy_command": proxy_command,
        "instances": {},
    }
    processes: dict[str, subprocess.Popen] = {}
    with contextlib.ExitStack() as logs:
        try:
            launched = time.monotonic()
            for role in ROLES:
                role_dir = output_dir / role
                role_dir.mkdir()
                env, set_env = instance_env(args.mode, engine, role_dir, dict(os.environ), devices[role])
                command = server_command(engine, output_dir / "model", ports[role]) + [
                    "--kv-transfer-config", json.dumps(kv_transfer_config(engine, role, ports[role])),
                ]
                summary["instances"][role] = {
                    "server_command": command,
                    "server_env_set": set_env,
                    "server_env_removed": sorted(
                        name for name in os.environ if name not in env or name in set_env
                    ),
                }
                server_log = logs.enter_context((role_dir / "server.log").open("w"))
                processes[role] = subprocess.Popen(
                    command, env=env, stdout=server_log, stderr=subprocess.STDOUT
                )
            for role in ROLES:
                wait_until_ready(processes[role], ports[role], args.startup_timeout_s)
                summary["instances"][role]["ready_after_launch_s"] = time.monotonic() - launched
            proxy_log = logs.enter_context((output_dir / "proxy.log").open("w"))
            processes["proxy"] = subprocess.Popen(
                proxy_command, stdout=proxy_log, stderr=subprocess.STDOUT
            )
            wait_until_ready(processes["proxy"], args.proxy_port, args.startup_timeout_s)
            clock, records = asyncio.run(replay(
                rows, args.proxy_port, int(model_config["vocab_size"]), args.origin_lead_s
            ))
            summary.update(clock)
            time.sleep(args.drain_s)
        finally:
            summary["process_stop"] = {
                name: stop_server(processes[name], args.stop_timeout_s) for name in reversed(processes)
            }

    for record in records:
        record["engine_request_id"] = engine_request_id(proxied[record["request_id"]])
    records.sort(key=lambda record: record["dispatch_monotonic"])
    (output_dir / "client_requests.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    failed = [record["request_id"] for record in records if record["http_status"] != 200]
    summary["num_requests"] = len(records)
    summary["failed_request_ids"] = failed
    for role in ROLES:
        server_log = output_dir / role / "server.log"
        summary["instances"][role]["kv_cache_tokens_by_engine"] = kv_cache_tokens(server_log)
        summary["instances"][role]["connector_check"] = connector_check(server_log)
    connector_failed = [
        role for role in ROLES if not summary["instances"][role]["connector_check"]["passed"]
    ]
    summary["connector_check_failed"] = connector_failed
    join_complete = True
    if args.mode == "clean":
        summary["pd_join"] = write_pd_rows(output_dir)
        join_complete = summary["pd_join"]["complete"]
    status = 4 if failed else 5 if connector_failed else 0 if join_complete else 6
    summary["status"] = status
    (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=1))
    print("REPLAY_DONE", json.dumps({
        "status": status, "requests": len(records), "failed": len(failed),
        "connector_check_failed": connector_failed,
        "joined": summary.get("pd_join", {}).get("joined"),
        "kv_cache_tokens": {
            role: summary["instances"][role]["kv_cache_tokens_by_engine"] for role in ROLES
        },
    }))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
