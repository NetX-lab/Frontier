#!/usr/bin/env python3
"""Serve a 1P1D vLLM pair joined by the P2pNccl connector and replay a trace through pd_proxy.py.

Runs inside the ``vllm/vllm-openai:v0.10.2`` image with the instrumented
overlay first on ``PYTHONPATH`` (run_pd_worker.sh). The deployment follows
calibration decisions Q4 and Q5: a prefill instance on the first visible GPU and
a decode instance on the second, each a ``vllm serve`` of the engine file, joined
by P2pNcclConnector with ``send_type`` PUT_ASYNC, with pd_proxy.py in front.

The explicit v0.28 adapter prepares PP1 NIXL pull roles and forwards P's
native response metadata through that same proxy. It retains declared observer
inputs and terminal diagnostics without applying the legacy PUT checks or metric
join. With explicit external P/D endpoints, it owns only the proxy and client;
the worker coordinator owns native multi-node startup and drain. This adapter
does not establish native READ/device/D-ready completion. The remaining description is legacy.

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
import csv
from datetime import datetime
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from urllib.parse import urlsplit

HARNESS_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(HARNESS_DIR), str(HARNESS_DIR.parent / "dp_placement_pp")]
from pd_metrics import engine_request_id, write_pd_rows  # noqa: E402
from pd_proxy import proxied_request_id  # noqa: E402
from vllm_replay import (  # noqa: E402
    MODES, REPO_ROOT, kv_cache_tokens, replay, server_command, server_env, stop_server,
    wait_until_ready, write_model_dir, runtime_version, local_model_config, observer_evidence,
    SERVED_MODEL_NAME,
    request_records,
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
    if role not in ROLES:
        raise ValueError(f"Unsupported P/D role: {role!r}")
    version = runtime_version(engine)
    kv = engine.get("kv_transfer") if version == "0.28.0" else engine["kv_transfer"]
    if version == "0.28.0":
        fields = {"kv_buffer_device", "side_channel_host", "prefill_side_channel_port",
                  "decode_side_channel_port"}
        if (type(kv) is not dict or set(kv) != fields or kv["kv_buffer_device"] != "cuda"
                or type(kv["side_channel_host"]) is not str
                or engine["pipeline_parallel_size"] != 1):
            raise ValueError("v0.28 pull requires explicit PP1 CUDA NIXL configuration")
        address = ipaddress.IPv4Address(kv["side_channel_host"])
        if (address.is_multicast or address.is_unspecified
                or "data_parallel_size_local" in engine and address.is_loopback):
            raise ValueError("NIXL side-channel host requires a reachable IPv4 address")
        ports = [kv[f"{name}_side_channel_port"] for name in ROLES]
        dp = engine["data_parallel_size"]
        if (type(dp) is not int or dp <= 0
                or any(type(port) is not int or not 0 < port <= 65536 - dp for port in ports)
                or abs(ports[0] - ports[1]) < dp):
            raise ValueError("NIXL side-channel port windows must be valid and disjoint for DP lanes")
        service_ports = [http_port] + ([engine["data_parallel_rpc_port"]]
                                      if "data_parallel_rpc_port" in engine else [])
        if any(start <= port < start + dp for start in ports for port in service_ports):
            raise ValueError("HTTP and DP RPC ports must be outside NIXL side-channel windows")
        return {"kv_connector": "NixlConnector", "kv_role": KV_ROLES[role],
                "kv_buffer_device": "cuda", "kv_load_failure_policy": "fail"}
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


def instance_devices(inherited: dict, *, engine: dict | None = None,
                     role: str | None = None) -> dict[str, str]:
    """Select explicit local devices, or the legacy first GPU of each role."""

    # Without CUDA_VISIBLE_DEVICES, CUDA numbers the visible GPUs from 0.
    visible = inherited.get("CUDA_VISIBLE_DEVICES")
    if engine is not None and runtime_version(engine) == "0.28.0":
        roles = engine.get("role_devices")
        selected_roles = (role,) if role is not None else ROLES
        if any(name not in ROLES for name in selected_roles):
            raise ValueError(f"Unsupported P/D role: {role!r}")
        dp = (engine.get("data_parallel_size_local", engine["data_parallel_size"])
              if role else engine["data_parallel_size"])
        count = engine["tensor_parallel_size"] * dp
        if (type(roles) is not dict or set(roles) != set(selected_roles)
                or engine["pipeline_parallel_size"] != 1
                or any(type(values) is not list or len(values) != count
                       or any(type(value) is not str or not value or "," in value for value in values)
                       for values in roles.values())):
            raise ValueError("v0.28 requires explicit PP1 per-role devices matching TP * local DP")
        selected = [device for name in selected_roles for device in roles[name]]
        if len(set(selected)) != len(selected) or (visible and not set(selected) <= set(visible.split(","))):
            raise ValueError("P/D devices must be disjoint and included in CUDA_VISIBLE_DEVICES")
        return {name: ",".join(roles[name]) for name in selected_roles}
    devices = visible.split(",") if visible else ["0", "1"]
    if len(devices) < 2:
        raise SystemExit(f"1P1D needs two GPUs, CUDA_VISIBLE_DEVICES={visible!r}")
    return dict(zip(ROLES, devices))


def instance_env(mode: str, engine: dict, role_dir: Path, inherited: dict,
                 device: str, *, observer: dict | None = None,
                 role: str | None = None) -> tuple[dict, dict]:
    """Return one instance's environment and the variables the harness set in it."""

    env, set_env = server_env(mode, engine, role_dir, inherited, observer=observer)
    if runtime_version(engine) == "0.28.0":
        kv_transfer_config(engine, role, 0)
        if observer is not None and observer["role"] != role.upper():
            raise ValueError("Observer role conflicts with P/D instance role")
        kv = engine["kv_transfer"]
        role_env = {"CUDA_VISIBLE_DEVICES": device,
                    "VLLM_NIXL_SIDE_CHANNEL_HOST": kv["side_channel_host"],
                    "VLLM_NIXL_SIDE_CHANNEL_PORT": str(kv[f"{role}_side_channel_port"])}
        if engine.get("worker_extension_cls") == "native_full_structure.FullModelInventory":
            # The server and its spawned workers do not inherit this process's sys.path.
            role_env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(HARNESS_DIR), env.get("PYTHONPATH"))))
        return env | role_env, set_env | role_env
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


def canonical_workload(trace_dir):
    """Read the existing frozen artifact layout without rewriting any input."""
    path = trace_dir / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "frontier.canonical_fixed_workload.v1":
        return None
    members = {"manifest.json", "trace.csv", "request_ids.json", "warmup_pool.csv"}
    checked = set()
    for line in (trace_dir / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        if name not in members or name in checked:
            raise ValueError("Unexpected canonical checksum member")
        checked.add(name)
        if hashlib.sha256((trace_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Canonical workload checksum mismatch: {name}")
    if checked != members:
        raise ValueError("Incomplete canonical checksum manifest")
    formal = json.loads((trace_dir / "request_ids.json").read_text())["rows"]
    with (trace_dir / "warmup_pool.csv").open() as stream:
        warmup = [row | {"arrived_at": float(row["arrived_at"]),
                         "num_prefill_tokens": int(row["num_prefill_tokens"]),
                         "num_decode_tokens": int(row["num_decode_tokens"]), "role": "warmup"}
                  for row in csv.DictReader(stream)]
    for phase, rows, count in (
        ("formal", formal, manifest["selection"]["num_formal_requests"]),
        ("warmup", warmup, manifest["warmup"]["num_requests_per_repetition"]),
    ):
        if len(rows) != count or any(
            row["role"] != phase or row["arrived_at"] != index * manifest["arrivals"]["inter_arrival_s"]
            or row["num_prefill_tokens"] != manifest["lengths"]["prompt_tokens_per_request"]
            or row["num_decode_tokens"] != manifest["lengths"]["output_tokens_per_request"]
            for index, row in enumerate(rows)
        ):
            raise ValueError(f"Canonical {phase} row/count mismatch")
    ids = [row["request_id"] for row in formal + warmup]
    if len(set(ids)) != len(ids):
        raise ValueError("Canonical formal/warmup template IDs must be disjoint")
    return manifest, formal, warmup


def dedicated_services(root, urls, dp_size):
    """Bind fresh Popen/readiness receipts to this prelaunch manifest and DP layout."""
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    written = datetime.fromisoformat(manifest["written_before_launch_utc"])
    generation = manifest["run_generation"]
    if (manifest["mode"] != "clean" or type(generation) is not str or not generation
            or written.tzinfo is None):
        raise ValueError("Missing dedicated service prelaunch identity")
    lanes, services = {role: set() for role in ROLES}, {}
    paths = sorted((root / "nodes").glob("*/result.json"))
    if len(paths) != manifest["resources"]["replica"]:
        raise ValueError("Missing dedicated service node receipts")
    for path in paths:
        result = json.loads(path.read_text())
        plan = result["plan"]
        role, lane, engine = plan["role"], plan["lane"], plan["engine"]
        started = datetime.fromisoformat(result["server_started_at_utc"])
        if (role not in lanes or engine["data_parallel_size"] != dp_size
                or type(result["server_pid"]) is not int or result["server_pid"] <= 0
                or result["run_manifest_sha256"] != digest or result["run_generation"] != generation
                or started.tzinfo is None or started < written or plan["urls"] != urls):
            raise ValueError("Unbound dedicated native service launch")
        owned = set(range(lane, lane + engine["data_parallel_size_local"]))
        if not owned or lanes[role] & owned or not owned <= set(range(dp_size)):
            raise ValueError("Invalid dedicated service DP lane ownership")
        lanes[role].update(owned)
        if lane == 0:
            ready = json.loads((path.parent / "server_ready.json").read_text())
            stamp = datetime.fromisoformat(ready["ready_at_utc"])
            if (ready["role"] != role or ready["url"] != urls[role] or ready["health_status"] != 200
                    or stamp.tzinfo is None or stamp < started
                    or any(ready[key] != result[key] for key in ("server_pid", "server_started_at_utc",
                                                               "run_manifest_sha256", "run_generation"))):
                raise ValueError("Unbound dedicated native service readiness")
            services[role] = ready
    if set(services) != set(ROLES) or any(values != set(range(dp_size)) for values in lanes.values()):
        raise ValueError("Missing dedicated native service DP lanes")
    return {"run_manifest_sha256": digest, "run_generation": generation, "services": services}


def native_idle_metrics(text, dp_size):
    """Retain published gauges and explicit counters; never fill missing samples."""
    names = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")
    lanes = {str(index): {} for index in range(dp_size)}
    counters = {str(index): {} for index in range(dp_size)}
    for line in text.splitlines():
        match = re.fullmatch(r'vllm:(\w+)\{(.*)\}\s+([^\s]+)(?:\s+\d+)?', line)
        if match is None:
            continue
        name, labels, value = match.groups()
        if name not in (*names, "request_success_total"):
            continue
        labels = dict(re.findall(r'(\w+)="([^"\\]*)"', labels))
        if labels.get("model_name") != SERVED_MODEL_NAME:
            continue
        lane = labels.get("engine")
        if lane not in lanes:
            raise ValueError(f"Unexpected native metrics engine: {lane}")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Invalid native idle metric")
        if name == "request_success_total":
            reason = labels.get("finished_reason", "")
            if reason in counters[lane]:
                raise ValueError("Duplicate or invalid native completion counter")
            counters[lane][reason] = number
        else:
            if name in lanes[lane]:
                raise ValueError("Duplicate native idle gauge")
            lanes[lane][name] = number
    if any(not values for values in counters.values()):
        raise ValueError("Missing native completion counter for a declared lane")
    return {"lanes": lanes, "completion_counters": counters,
            "completed_requests": sum(sum(values.values()) for values in counters.values()),
            "idle": all(set(values) == set(names) and all(value == 0 for value in values.values())
                        for values in lanes.values())}


def idle_metrics(text, dp_size):
    """Require native metrics for every declared request-owner DP lane."""
    observed = native_idle_metrics(text, dp_size)
    if any(len(values) != 3 for values in observed["lanes"].values()):
        raise ValueError("Missing native running/waiting/KV gauge for a declared lane")
    return observed


async def observe_idle(urls, dp_size, timeout_s, poll_s, *, completed_after=None, fresh_services=None):
    """Observe idle as an auxiliary soft gate; never stop native workers."""
    import aiohttp

    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Idle observation timeout must be positive and finite")
    deadline = asyncio.get_running_loop().time() + timeout_s
    observations = []
    try:
        async with asyncio.timeout_at(deadline):
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as session:
                async def scrape(role, sample):
                    try:
                        async with session.get(urls[role] + "/metrics") as response:
                            response.raise_for_status()
                            text = await response.text()
                            observed = native_idle_metrics(text, dp_size) if fresh_services else idle_metrics(text, dp_size)
                            if fresh_services:
                                # v0.28 FinishReason children are created eagerly (loggers.py:713-724).
                                reasons = {"stop", "length", "abort", "error", "repetition"}
                                if (any(set(values) != reasons or any(values.values())
                                        for values in observed["completion_counters"].values())
                                        or any(observed["lanes"].values()) and not observed["idle"]):
                                    raise ValueError("Cold start requires explicit all-lane zero counters and unpublished or idle gauges")
                            sample["roles"][role] = observed
                    except Exception as error:
                        sample["errors"].append(f"{role}: {type(error).__name__}: {error}")

                while True:
                    sample = {"monotonic": time.monotonic(), "roles": {}, "errors": []}
                    observations.append(sample)
                    await asyncio.gather(*(scrape(role, sample) for role in ROLES))
                    fresh = completed_after is None or all(
                        sample["roles"].get(role, {}).get("completed_requests", -1) >= completed_after[role]
                        for role in ROLES)
                    if (not sample["errors"] and fresh
                            and (fresh_services or all(sample["roles"][role]["idle"] for role in ROLES))):
                        return {"verified": True, "observations": observations}
                    await asyncio.sleep(poll_s)
    except TimeoutError:
        return {"verified": False, "observations": observations,
                "error": "Idle observation deadline exceeded; idle/KV release or fresh completion metrics unverified"}


def formal_metrics(records):
    """Compute only a complete population's uncorrected client-token metrics."""
    if not records or any(record["status"] != "completed" for record in records):
        return None
    start = min(record["dispatch_monotonic"] for record in records)
    stop = max(record["last_token_monotonic"] for record in records)
    total = sum(record["num_prefill_tokens"] + record["num_decode_tokens"] for record in records)
    metrics = {"first_submission_monotonic": start, "last_token_monotonic": stop,
               "window_s": stop - start, "total_tokens": total, "total_thpt_tokens_s": total / (stop - start)}
    for field in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = sorted(record[field] for record in records)
        metrics[field] = {"mean": statistics.mean(values), "p50": statistics.median(values),
                          "p95": values[max(0, math.ceil(len(values) * .95) - 1)]}
    return metrics


async def replay_repetitions(workload, port, vocab_size, origin_lead_s, urls, dp_size,
                             idle_timeout_s, idle_poll_s, *, checkpoint=None, records=None, fresh_services=None):
    """Run canonical warmup/formal phases in one existing proxy/client lifetime."""
    manifest, formal, warmup = workload
    result = {"repetitions": [], "idle_boundaries": [], "observation_errors": [],
              "reference_admission": "UNVERIFIED"}
    if records is None:
        records = []
    phases = {}
    for repeat in manifest["repetitions"]["repeat_indices"]:
        for phase, templates in (("warmup", warmup), ("formal", formal)):
            phases[repeat, phase] = request_records([row | {
                "template_request_id": row["request_id"],
                "request_id": manifest["repetitions"]["runtime_request_id_rule"].format(
                    request_id=row["request_id"], repeat=repeat),
                "repeat": repeat, "segment": phase,
            } for row in templates])
            records.extend(phases[repeat, phase])

    def persist():
        if checkpoint is not None:
            try:
                checkpoint(result, records)
            except Exception as error:
                result["observation_errors"].append(f"{type(error).__name__}: {error}")
                print("NATIVE_PD_AUXILIARY_UNVERIFIED", result["observation_errors"][-1], flush=True)

    async def boundary(name, completed_after=None, *, first=False):
        try:
            dispatched = sum(row["dispatch_monotonic"] is not None for row in records)
            cold = (dedicated_services(fresh_services, urls, dp_size)
                    if first and fresh_services and dispatched == 0 else None)
            observed = await observe_idle(urls, dp_size, idle_timeout_s, idle_poll_s,
                                          completed_after=completed_after, fresh_services=cold)
            if first and cold and observed["verified"]:
                observed.update(proof="dedicated_services_ready_client_zero_native_counters_zero",
                                fresh_services=cold, client_dispatched_requests=dispatched)
            elif not first and completed_after is None:
                observed.update(verified=False, error="Initial native completion baseline is unverified")
            if first and dispatched:
                observed.update(verified=False, error="Existing client dispatches invalidate cold start")
            if not first:
                observed["completed_after"] = completed_after
        except Exception as error:
            observed = {"verified": False, "error": f"{type(error).__name__}: {error}"}
        observed["boundary"] = name
        if not observed["verified"]:
            print("NATIVE_PD_AUXILIARY_UNVERIFIED", name, observed.get("error"), flush=True)
        result["idle_boundaries"].append(observed)
        persist()
        return observed

    idle = await boundary("before_repeat01_warmup", first=True)
    completed = ({role: idle["observations"][-1]["roles"][role]["completed_requests"] for role in ROLES}
                 if idle["verified"] else None)
    for repeat in manifest["repetitions"]["repeat_indices"]:
        current = {"repeat": repeat, "phases": {}, "client_workload_verified": idle["verified"]}
        result["repetitions"].append(current)
        for phase in ("warmup", "formal"):
            rows = phases[repeat, phase]
            try:
                clock, phase_records = await replay(rows, port, vocab_size, origin_lead_s,
                                                    vllm_version="0.28.0", records=rows)
            finally:
                persist()
            failed = [record["request_id"] for record in phase_records if record["status"] != "completed"]
            current["phases"][phase] = clock | {"planned_requests": len(rows),
                                               "recorded_requests": len(phase_records),
                                               "failed_request_ids": failed}
            current["client_workload_verified"] &= not failed and len(phase_records) == len(rows)
            persist()
            if completed is not None:
                completed = {role: count + len(rows) for role, count in completed.items()}
            idle = await boundary(f"after_repeat{repeat:02d}_{phase}", completed)
            current["client_workload_verified"] &= idle["verified"]
            if phase == "formal":
                try:
                    current["metrics"] = formal_metrics(phase_records) if len(phase_records) == len(rows) else None
                except Exception as error:
                    current["metrics"] = None
                    result["observation_errors"].append(f"{type(error).__name__}: {error}")
                    print("NATIVE_PD_AUXILIARY_UNVERIFIED", result["observation_errors"][-1], flush=True)
                    current["client_workload_verified"] = False
        persist()
    result["client_workload_verified"] = (all(row["client_workload_verified"] for row in result["repetitions"])
                                           and not result["observation_errors"])
    result["status"] = "CLIENT_WORKLOAD_VERIFIED" if result["client_workload_verified"] else "REFERENCE_UNSTABLE"
    persist()
    if result["observation_errors"]:
        result.update(client_workload_verified=False, status="REFERENCE_UNSTABLE")
    return result, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=(*MODES, "light"), required=True)
    parser.add_argument("--observer-config", type=Path,
                        help="v0.28: explicit JSON object with prefill/decode recorder inputs")
    parser.add_argument("--model-dir", type=Path,
                        help="v0.28: existing local GLM metadata directory")
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--decode-port", type=int, default=8200)
    parser.add_argument("--prefill-url", help="v0.28: externally managed prefill HTTP endpoint")
    parser.add_argument("--decode-url", help="v0.28: externally managed decode HTTP endpoint")
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--origin-lead-s", type=float, default=1.0)
    parser.add_argument("--drain-s", type=float, default=3.0)
    parser.add_argument("--stop-timeout-s", type=float, default=60.0)
    parser.add_argument("--idle-timeout-s", type=float, default=180.0)
    parser.add_argument("--idle-poll-s", type=float, default=1.0)
    args = parser.parse_args(argv)

    engine = json.loads(args.engine_config.read_text())
    version = runtime_version(engine)
    workload = canonical_workload(args.trace_dir)
    if workload is not None and (version != "0.28.0" or args.mode != "clean"):
        raise ValueError("Canonical repetitions require the native v0.28 clean mode")
    if args.idle_timeout_s <= 0 or args.idle_poll_s <= 0:
        raise ValueError("Idle observation timeout and poll interval must be positive")
    external = args.prefill_url is not None or args.decode_url is not None
    if external and (version != "0.28.0" or args.prefill_url is None or args.decode_url is None):
        raise ValueError("External P/D endpoints require v0.28 and both --prefill-url and --decode-url")
    if external:
        urls = {role: getattr(args, f"{role}_url").rstrip("/") for role in ROLES}
        for url in urls.values():
            endpoint = urlsplit(url)
            if (endpoint.scheme != "http" or not endpoint.hostname or not endpoint.port
                    or endpoint.username is not None or endpoint.password is not None
                    or endpoint.path or endpoint.query or endpoint.fragment):
                raise ValueError("External endpoints require explicit http://IPv4:port origins")
            address = ipaddress.IPv4Address(endpoint.hostname)
            if (address.is_unspecified or address.is_multicast
                    or address.is_loopback and endpoint.port == args.proxy_port):
                raise ValueError("External endpoints must be reachable and distinct from the local proxy")
        if urls["prefill"] == urls["decode"]:
            raise ValueError("External P/D endpoints must be distinct")
    else:
        if version == "0.28.0" and "data_parallel_size_local" in engine:
            raise ValueError("Native DP node partitions require externally managed P/D endpoints")
        host = engine.get("host", "127.0.0.1") if version == "0.28.0" else "127.0.0.1"
        host = "127.0.0.1" if host == "0.0.0.0" else host
        urls = {role: f"http://{host}:{getattr(args, f'{role}_port')}" for role in ROLES}
    observers = json.loads(args.observer_config.read_text()) if args.observer_config else None
    if version == "0.28.0":
        if args.model_dir is None:
            parser.error("v0.28 requires --model-dir with local GLM metadata")
        model_dir = args.model_dir
        model_config = local_model_config(model_dir)
        if observers is not None and (type(observers) is not dict or set(observers) != set(ROLES)
                                      or any(type(value) is not dict for value in observers.values())):
            raise ValueError("PD observer JSON must contain exactly prefill/decode inputs")
        if observers is not None and observers["prefill"].get("run_id") != observers["decode"].get("run_id"):
            raise ValueError("PD observers require one run identity")
    else:
        if args.model_dir is not None or observers is not None:
            parser.error("Explicit metadata/observer inputs require the v0.28 adapter")
        missing = [name for name in KV_TRANSFER_FIELDS if name not in engine.get("kv_transfer", {})]
        if missing:
            raise SystemExit(f"{args.engine_config}: kv_transfer lacks {missing}")
    rows = json.loads((args.trace_dir / "request_ids.json").read_text())["rows"]
    kv_addresses = ({role: f"{KV_IP}:{engine['kv_transfer'][f'{role}_kv_port']}" for role in ROLES}
                    if version == "0.10.2" else {})
    # Every trace id must survive the connector's address parsing before a GPU is used.
    proxied = {
        row["request_id"]: proxied_request_id(row["request_id"], kv_addresses["prefill"],
                                              kv_addresses["decode"])
        for row in rows
    } if version == "0.10.2" else {}
    output_dir = args.output_dir
    devices = {} if external else instance_devices(dict(os.environ), engine=engine)
    ports = {"prefill": args.prefill_port, "decode": args.decode_port}
    if not external and len({args.proxy_port, *ports.values()}) != 3:
        raise ValueError("Proxy and P/D HTTP ports must be distinct")
    if version == "0.10.2":
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = output_dir / "model"
        model_config = write_model_dir(
            REPO_ROOT / "data/config/models" / f"{engine['model_name']}.json", model_dir,
        )
    plans = {}
    for role in ROLES:
        if external:
            server_env(args.mode, engine, output_dir / role, dict(os.environ),
                       observer=None if observers is None else observers[role])
            if observers is not None and observers[role]["role"] != role.upper():
                raise ValueError("Observer role conflicts with P/D instance role")
            continue
        env, changed = instance_env(args.mode, engine, output_dir / role, dict(os.environ), devices[role],
                                    observer=None if observers is None else observers[role], role=role)
        command = server_command(engine, model_dir, ports[role]) + [
            "--kv-transfer-config", json.dumps(kv_transfer_config(engine, role, ports[role])),
        ]
        plans[role] = (env, changed, command)
    if version == "0.28.0":
        if observers is not None:
            prefill_dir, decode_dir = (Path(observers[role]["output_dir"]).resolve() for role in ROLES)
            if prefill_dir.is_relative_to(decode_dir) or decode_dir.is_relative_to(prefill_dir):
                raise ValueError("PD observers require disjoint output directories")
        if not 0 < args.proxy_port < 65536:
            raise ValueError("Proxy port must be valid")
        if not external:
            http_ports = [args.proxy_port, *ports.values()]
            kv = engine["kv_transfer"]
            dp = engine["data_parallel_size"]
            if (any(not 0 < port < 65536 for port in http_ports)
                    or any(kv[f"{role}_side_channel_port"] <= port
                           < kv[f"{role}_side_channel_port"] + dp
                           for role in ROLES for port in http_ports)):
                raise ValueError("HTTP ports must be valid and outside both NIXL side-channel windows")
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_command = [
        sys.executable, str(HARNESS_DIR / "pd_proxy.py"), "--port", str(args.proxy_port),
        "--prefill-url", urls["prefill"],
        "--decode-url", urls["decode"],
        "--log-path", str(output_dir / "proxy_requests.jsonl"),
    ]
    if version == "0.10.2":
        proxy_command += ["--prefill-kv-address", kv_addresses["prefill"],
                          "--decode-kv-address", kv_addresses["decode"]]
    else:
        proxy_command += ["--vllm-version", version]
    summary: dict = {
        "mode": args.mode,
        "engine_config": engine,
        "trace_dir": str(args.trace_dir),
        "devices": devices,
        "kv_addresses": kv_addresses,
        "proxy_command": proxy_command,
        "instances": {},
    }
    if version == "0.28.0":
        summary.update(vllm_version=version, qualification_verified=False,
                       model_config_sha256=hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest())
    if external:
        summary["instances"] = {role: {"http_url": urls[role], "lifecycle_owner": "external"}
                                for role in ROLES}
    for role, (env, set_env, command) in plans.items():
        summary["instances"][role] = {
            "server_command": command,
            "server_env_set": set_env,
            "server_env_removed": sorted(
                name for name in os.environ if name not in env or name in set_env
            ),
        }
    processes: dict[str, subprocess.Popen] = {}
    records = []
    with contextlib.ExitStack() as logs:
        try:
            launched = time.monotonic()
            for role in plans:
                role_dir = output_dir / role
                role_dir.mkdir()
                env, set_env, command = plans[role]
                server_log = logs.enter_context((role_dir / "server.log").open("w"))
                processes[role] = subprocess.Popen(
                    command, env=env, stdout=server_log, stderr=subprocess.STDOUT
                )
            for role in ROLES:
                endpoint = urlsplit(urls[role])
                if external or endpoint.hostname != "127.0.0.1":
                    health_wait_s = wait_until_ready(
                        processes.get(role), endpoint.port, args.startup_timeout_s, host=endpoint.hostname)
                else:
                    wait_until_ready(processes[role], ports[role], args.startup_timeout_s)
                if external:
                    summary["instances"][role]["health_wait_s"] = health_wait_s
                else:
                    summary["instances"][role]["ready_after_launch_s"] = time.monotonic() - launched
            proxy_log = logs.enter_context((output_dir / "proxy.log").open("w"))
            processes["proxy"] = subprocess.Popen(
                proxy_command, stdout=proxy_log, stderr=subprocess.STDOUT
            )
            wait_until_ready(processes["proxy"], args.proxy_port, args.startup_timeout_s)
            if workload is not None:
                fresh_services = output_dir.parent if external and (output_dir.parent / "run_manifest.json").is_file() else None
                def checkpoint(result, phase_records):
                    (output_dir / "client_requests.jsonl").write_text(
                        "".join(json.dumps(record) + "\n" for record in phase_records))
                    (output_dir / "canonical_repetitions.json").write_text(json.dumps(result, indent=1))

                repetitions, records = asyncio.run(replay_repetitions(
                    workload, args.proxy_port, int(model_config["vocab_size"]), args.origin_lead_s,
                    urls, engine["data_parallel_size"], args.idle_timeout_s, args.idle_poll_s,
                    checkpoint=checkpoint, records=records,
                    **({"fresh_services": fresh_services} if fresh_services else {})))
                summary["canonical_repetitions"] = repetitions
            else:
                clock, records = asyncio.run(replay(
                    rows, args.proxy_port, int(model_config["vocab_size"]), args.origin_lead_s,
                    vllm_version=version,
                ))
                summary.update(clock)
            time.sleep(args.drain_s)
        except BaseException as error:
            summary["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            summary["process_stop"] = {
                name: stop_server(processes[name], args.stop_timeout_s) for name in reversed(processes)
            }
            if version == "0.28.0":
                for role in ROLES:
                    summary["instances"].setdefault(role, {})["observer_evidence"] = observer_evidence(
                        engine, None if observers is None else observers[role])
            (output_dir / "client_requests.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=1))

    if version == "0.10.2":
        for record in records:
            record["engine_request_id"] = engine_request_id(proxied[record["request_id"]])
    records.sort(key=lambda record: (record["dispatch_monotonic"] is None,
                                     record["dispatch_monotonic"] or 0))
    (output_dir / "client_requests.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    failed = [record["request_id"] for record in records
              if record["http_status"] != 200 or record.get("status", "completed") != "completed"]
    summary["num_requests"] = len(records)
    summary["failed_request_ids"] = failed
    if version == "0.28.0":
        summary["native_transfer_verification"] = {
            "status": "UNVERIFIED", "expected_protocol": "NixlConnector pull READ",
            "native_done": "unobserved", "device_completion": "unobserved",
            "decode_readiness": "unobserved", "metric_normalization": "not_applied",
        }
        summary["status"] = 4 if (failed or workload is not None
                                  and not repetitions["client_workload_verified"]) else 0
        (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=1))
        print("REPLAY_DONE", json.dumps({"status": summary["status"], "requests": len(records),
                                         "qualification_verified": False}))
        return summary["status"]
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
