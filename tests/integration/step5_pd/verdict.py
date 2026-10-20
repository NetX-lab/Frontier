"""Judge a finished native P/D run from its task-local evidence (design01 section 3.6).

Reads `config.json`, `trace/` and `evidence/` of a stage directory: the per-node records that
native_pd_worker.py mirrored (`nodes/<host>/`) and the wave outputs of run_waves.py
(`replay/`). The platform terminal status is an argument, because only the launcher sees
it. Each of the nine checks reports PASS, FAIL or UNVERIFIED; a check whose evidence is
missing or unreadable is UNVERIFIED with the error, never FAIL. The first wave in `waves.json`
marked as gating decides the checks; the other waves are reported as observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "comparison" / "calibration"))
from serving_metrics import (  # noqa: E402
    KV_CACHE_USAGE, PROMPT_TOKEN_SOURCES, engine_values, parse_prometheus, transfer_checks,
    transfer_delta,
)

ROLES = ("prefill", "decode")
NIXL_SCHEDULER = "Initializing NIXL Scheduler"
RESOLVED_LAYOUT = re.compile(r"Using (\w+) KV cache layout\.")
WORKER_LAYOUT = re.compile(r"Detected kv cache layout (\w+)")
KV_CACHE_SIZE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
STEP5_GUARD = "Step5 DSA sparse summaries are only transferred by"
EXPECTED_LAYOUT = "LBNHC"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def judge(stage: Path, platform_status: str) -> dict:
    config = json.loads((stage / "config.json").read_text())
    waves = json.loads((stage / "trace" / "waves.json").read_text())
    evidence = stage / "evidence"
    replay = evidence / "replay"
    gating = waves["gating_wave"]
    engine = config["engine"]
    checks: dict = {}
    observations: dict = {}

    def check(name, function):
        try:
            checks[name] = function()
        except Exception:
            checks[name] = {"status": "UNVERIFIED", "error": traceback.format_exc()}

    nodes = {path.name: path for path in sorted((evidence / "nodes").iterdir())}
    results = {host: json.loads((path / "result.json").read_text()) for host, path in nodes.items()}
    rows = json.loads((stage / "trace" / gating / "request_ids.json").read_text())["rows"]
    request_id = rows[0]["request_id"]

    def wave_records(wave):
        client = {row["request_id"]: row for row in read_jsonl(replay / wave / "client_requests.jsonl")}
        proxy = {row["request_id"]: row for row in read_jsonl(replay / wave / "proxy_requests.jsonl")}
        return client, proxy

    def pages(point):
        return {role: parse_prometheus((replay / "metrics" / f"{point}_{role}.prom").read_text())
                for role in ROLES}

    def placement():
        ready = {host: json.loads((path / "ready.json").read_text()) for host, path in nodes.items()}
        hosts = sorted(ready)
        roles = {host: results[host]["plan"]["role"] for host in hosts}
        half = config["replica"] // 2
        expected_roles = {host: ROLES[index // half] for index, host in enumerate(hosts)}
        ucx = {host: results[host].get("ucx_devices") for host in hosts}
        gpus = {host: results[host]["nvidia_smi"].count(config["gpu_type"]) for host in hosts}
        passed = (len(hosts) == config["replica"]
                  and len({row["ip"] for row in ready.values()}) == config["replica"]
                  and roles == expected_roles and all(ucx.values())
                  and all(count == config["gpu"] for count in gpus.values()))
        return {"status": status(passed), "hosts": {host: ready[host]["ip"] for host in hosts},
                "roles": roles, "ucx_devices": ucx, "gpus_matching_type": gpus}

    def startup():
        per_node = {}
        for host, path in nodes.items():
            log = (path / "server.log").read_text(errors="replace")
            per_node[host] = {
                "role": results[host]["plan"]["role"],
                "nixl_scheduler_lines": log.count(NIXL_SCHEDULER),
                "resolved_layouts": RESOLVED_LAYOUT.findall(log),
                "worker_layouts": WORKER_LAYOUT.findall(log),
                "kv_cache_tokens": [int(value.replace(",", "")) for value in KV_CACHE_SIZE.findall(log)],
                "step5_guard_error": STEP5_GUARD in log,
            }
        tp = engine["tensor_parallel_size"]
        passed = all(row["nixl_scheduler_lines"] >= 1
                     and row["resolved_layouts"] and set(row["resolved_layouts"]) == {EXPECTED_LAYOUT}
                     and row["worker_layouts"].count(EXPECTED_LAYOUT) == len(row["worker_layouts"]) == tp
                     and row["kv_cache_tokens"] and not row["step5_guard_error"]
                     for row in per_node.values())
        return {"status": status(passed), "nodes": per_node}

    def prefill_response():
        _, proxy = wave_records(gating)
        record = proxy[request_id]
        params = record.get("kv_transfer_params") or {}
        passed = (record.get("prefill_status") == 200 and params.get("do_remote_prefill") is True
                  and bool(params.get("remote_block_ids"))
                  and all(type(params.get(key)) is str and params[key]
                          for key in ("remote_engine_id", "remote_request_id", "remote_host"))
                  and type(params.get("remote_port")) is int
                  and params.get("transfer_mode") == "pull"
                  and params.get("tp_size") == engine["tensor_parallel_size"])
        return {"status": status(passed), "prefill_status": record.get("prefill_status"),
                "kv_transfer_params": params,
                "remote_block_counts": [len(group) for group in params.get("remote_block_ids") or []]}

    def decode_output():
        client, proxy = wave_records(gating)
        record, row = client[request_id], rows[0]
        tokens = (record.get("usage") or {}).get("completion_tokens")
        passed = (record.get("http_status") == 200 and proxy[request_id].get("decode_status") == 200
                  and tokens == row["num_decode_tokens"] and record.get("finish_reason") == "length")
        return {"status": status(passed), "http_status": record.get("http_status"),
                "decode_status": proxy[request_id].get("decode_status"), "usage": record.get("usage"),
                "finish_reason": record.get("finish_reason")}

    def transfer():
        before, after = pages("baseline"), pages(gating)
        deltas = {role: transfer_delta(before[role], after[role]) for role in ROLES}
        return transfer_checks(deltas["prefill"], deltas["decode"],
                               waves["expected_decode_prompt_sources"][gating])

    def release():
        start, end = pages("baseline"), pages("released")
        expired = {role: transfer_delta(start[role], end[role])["nixl_failures"][
            "vllm:nixl_num_kv_expired_reqs"] for role in ROLES}
        usage = {role: engine_values(end[role], KV_CACHE_USAGE) for role in ROLES}
        no_expiry = all(row["present"] and row["total"] == 0 for row in expired.values())
        zero = all(values and all(value == 0 for value in values.values()) for values in usage.values())
        verdict = "PASS" if zero and no_expiry else "FAIL" if not no_expiry else "UNVERIFIED"
        return {"status": verdict, "kv_cache_usage_perc": usage, "expired_requests_change": expired,
                "note": ("The gauge is published with scheduler statistics, so an engine that stays "
                         "idle after freeing blocks can keep reporting its last busy value; a nonzero "
                         "gauge without lease expiry is therefore UNVERIFIED, not FAIL.")}

    def ordering():
        client, proxy = wave_records(gating)
        sent, routed = client[request_id], proxy[request_id]
        times = [sent["dispatch_monotonic"], routed["proxy_arrival_monotonic"],
                 routed["prefill_response_monotonic"], routed["decode_response_monotonic"],
                 sent["response_monotonic"]]
        return {"status": status(times == sorted(times) and len(set(times)) == len(times)),
                "client_dispatch": times[0], "proxy_arrival": times[1], "prefill_response": times[2],
                "decode_response": times[3], "client_response": times[4],
                "note": ("Client and proxy share the prefill head's monotonic clock. The proxy sends "
                         "to D after the prefill response in program order; that send time is not "
                         "recorded separately.")}

    def exit_status():
        metrics = json.loads((replay / "native_transfer_metrics.json").read_text())
        per_node = {host: {"workload_returncode": result.get("workload_returncode"),
                           "server_stop": result.get("server_stop"),
                           "failed": (nodes[host] / "failed.json").exists(),
                           "stage": result.get("stage")} for host, result in results.items()}
        passed = (platform_status == "Succeeded" and metrics["statuses"].get(gating) == 0
                  and all(row["workload_returncode"] == 0 and not row["failed"]
                          and (row["server_stop"] or {}).get("returncode") == 0
                          for row in per_node.values()))
        return {"status": status(passed), "platform_status": platform_status,
                "wave_statuses": metrics["statuses"], "wave_errors": metrics["errors"],
                "nodes": per_node}

    for name, function in (("1_placement", placement), ("2_startup", startup),
                           ("3_prefill_response", prefill_response), ("4_decode_output", decode_output),
                           ("5_6_transfer", transfer), ("7_release", release), ("8_ordering", ordering),
                           ("9_exit", exit_status)):
        check(name, function)
    if "nixl_counters" in checks.get("5_6_transfer", {}):
        transfer_result = checks.pop("5_6_transfer")
        counters = transfer_result["nixl_counters"]
        # vLLM registers every failure series per engine at startup, so an absent one means the
        # page could not be read as expected, not that a failure was hidden.
        absent = any(value is None for role in counters["failure_changes"].values()
                     for value in role.values())
        checks["5_nixl_counters"] = dict(counters, status=(
            status(counters["passed"]) if not absent or not all(
                value > 0 for value in counters["decode_reads"].values()) else "UNVERIFIED"))
        checks["6_decode_prompt_sources"] = dict(transfer_result["decode_prompt_sources"],
                                                 status=status(transfer_result["decode_prompt_sources"]["passed"]))
    elif "5_6_transfer" in checks:
        failed = checks.pop("5_6_transfer")
        checks["5_nixl_counters"] = checks["6_decode_prompt_sources"] = failed

    try:
        observations["requests"] = {}
        single_request_waves = set()
        for wave in waves["waves"]:
            trace_rows = json.loads((stage / "trace" / wave / "request_ids.json").read_text())["rows"]
            if len(trace_rows) == 1:
                single_request_waves.add(wave)
            client, proxy = wave_records(wave)
            for rid, record in client.items():
                routed = proxy.get(rid, {})
                params = routed.get("kv_transfer_params") or {}

                def interval(start, end, source=routed):
                    return source[end] - source[start] if start in source and end in source else None
                observations["requests"][rid] = {
                    "wave": wave, "num_prefill_tokens": record["num_prefill_tokens"],
                    "http_status": record["http_status"], "usage": record.get("usage"),
                    "client_send_monotonic": record["dispatch_monotonic"],
                    "client_first_token_monotonic": None,
                    "client_first_token_status": "UNVERIFIED_NONSTREAMING",
                    "proxy_prefill_response_monotonic": routed.get("prefill_response_monotonic"),
                    "client_completion_monotonic": record["response_monotonic"],
                    "output_tokens": (record.get("usage") or {}).get("completion_tokens"),
                    "client_e2e_s": record["response_monotonic"] - record["dispatch_monotonic"],
                    "prefill_phase_s": interval("proxy_arrival_monotonic", "prefill_response_monotonic"),
                    "decode_phase_s": interval("prefill_response_monotonic", "decode_response_monotonic"),
                    "prefill_lane": (params["remote_port"] - engine["kv_transfer"]["prefill_side_channel_port"]
                                     if type(params.get("remote_port")) is int else None)}
        points = ["baseline", *waves["waves"]]
        observations["wave_changes"] = {}
        for start, end in zip(points, points[1:]):
            before, after = pages(start), pages(end)
            delta = {role: transfer_delta(before[role], after[role]) for role in ROLES}
            observations["wave_changes"][end] = {
                "decode_prompt_sources": {source: delta["decode"]["prompt_tokens_by_source"][source]
                                          for source in PROMPT_TOKEN_SOURCES},
                "expected_decode_prompt_sources": waves["expected_decode_prompt_sources"][end],
                "decode_nixl_transfers": delta["decode"]["nixl_transfers"],
                "decode_requests": delta["decode"]["requests"],
                "prefill_requests": delta["prefill"]["requests"]}
        observations["warm_requests"] = []
        for rid, row in observations["requests"].items():
            wave = row["wave"]
            window = observations["wave_changes"][wave]
            sources = window["decode_prompt_sources"]
            requests = window["decode_requests"]
            completed = requests["vllm:e2e_request_latency_seconds"]["count"]
            attributed = (wave in single_request_waves and completed["present"]
                          and completed["total"] == 1
                          and all(series["present"] for series in sources.values())
                          and sum(series["total"] for series in sources.values()) == row["num_prefill_tokens"])
            ttft = requests["vllm:time_to_first_token_seconds"]
            row.update(
                decode_external_tokens=sources["external_kv_transfer"]["total"] if attributed else None,
                decode_computed_prompt_tokens=sources["local_compute"]["total"] if attributed else None,
                decode_prompt_attribution="single_request_window" if attributed else "UNVERIFIED",
                decode_prompt_sources_per_engine={name: series["per_engine"] for name, series in sources.items()},
                decode_engine_ttft_s=(ttft["sum"]["total"] if attributed and ttft["sum"]["present"]
                                      and ttft["count"]["total"] == 1 else None))
            if wave != gating:
                observations["warm_requests"].append({"request_id": rid, **row})
        observations["client_first_token_note"] = (
            "The existing replay and proxy are nonstreaming. Client first-token timestamps are "
            "UNVERIFIED, not response completion times. decode_engine_ttft_s is the native engine "
            "histogram delta for a single-request window, not client TTFT. Warmup and boundary "
            "waves are observations; only W1 decides the unchanged nine conditions.")
        observations["observation_errors"] = {host: result.get("observation_errors")
                                              for host, result in results.items()}
    except Exception:
        observations["error"] = traceback.format_exc()
    return {"stage": str(stage), "gating_wave": gating, "checks": checks, "observations": observations,
            "all_pass": all(row.get("status") == "PASS" for row in checks.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--platform-status", required=True,
                        help="Phase of the RJob object as the platform reports it, e.g. Succeeded; "
                             "the launcher's lowercase FINAL_STATUS value does not match")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verdict = judge(args.stage_dir, args.platform_status)
    args.output.write_text(json.dumps(verdict, indent=1))
    print(json.dumps({name: row.get("status") for name, row in verdict["checks"].items()}))
