"""Consume the common fixed workload through the existing native P/D proxy and client.

This driver owns phase boundaries and admission diagnostics, not another HTTP replay
client. Auxiliary observations are soft gates: failed diagnostics invalidate reference
admission, but never cancel healthy requests or prevent later repetitions.

An optional length sweep runs once before the first repetition. A native decode step that
mixes decodes with a new request's prompt tail plans the DSA scoring kernel for
``(longest sequence + 7) // 8`` regions, a compile-time constant of the Triton kernel. The
native warm-up compiles only one region count, so the first mixed step at each new longest
length compiles a variant on the critical path (about 0.4 s, stalling every request of the
engine). The sweep makes each DP engine compile every variant the formal phases reach, so the
repetitions measure steady-state serving.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

CALIBRATION = Path(__file__).resolve().parents[2] / "comparison/calibration"
sys.path[:0] = [str(CALIBRATION), str(CALIBRATION.parent / "dp_placement_pp")]
from pd_proxy import build_app  # noqa: E402
from pd_replay import canonical_workload as read_canonical_workload  # noqa: E402
from serving_metrics import parse_prometheus, transfer_delta  # noqa: E402
from vllm_replay import SERVED_MODEL_NAME, local_model_config, replay, runtime_version  # noqa: E402

MANIFEST_SHA256 = "5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3"
SUMS_SHA256 = "594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9"
ROLES = ("prefill", "decode")
GAUGES = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")
LONG_SWEEP_PREFIX = "length-sweep-long-"
SHORT_SWEEP_PREFIX = "length-sweep-short-"


def canonical_workload(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    """Check the D18 pins and all common package members without rewriting them."""
    for name, expected in (("manifest.json", MANIFEST_SHA256), ("SHA256SUMS", SUMS_SHA256)):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"D18 canonical pin mismatch: {name}")
    return read_canonical_workload(directory)


def phase_rows(manifest: dict, templates: list[dict], repeat: int, phase: str) -> list[dict]:
    return [row | {"template_request_id": row["request_id"], "prompt_id": row["request_id"],
                   "request_id": manifest["repetitions"]["runtime_request_id_rule"].format(
                       request_id=row["request_id"], repeat=repeat),
                   "repeat": repeat, "segment": phase, "stream": True, "return_token_ids": True}
            for row in templates]


def length_sweep_rows(formal: list[dict], settings: dict) -> list[dict]:
    """Long requests of the longest formal shape and a stream of short requests.

    ``long_requests_per_lane`` long requests per DP lane arrive together, so the native DP load
    balancer spreads them over the lanes, and the longest sequence of every lane grows through
    the formal range. A region count lasts eight decode steps; short requests, one per
    ``lane_interval_s / dp_size`` for ``duration_s``, give every lane a mixed step within that
    window as long as ``lane_interval_s`` is shorter than eight decode steps.
    """
    sweep, lanes = settings["length_sweep"], settings["dp_size"]
    longest = max(formal, key=lambda row: row["num_prefill_tokens"] + row["num_decode_tokens"])
    common = {"segment": "length_sweep", "role": "diagnostic", "repeat": 0, "stream": True,
              "return_token_ids": True}
    rows = [common | {"request_id": f"{LONG_SWEEP_PREFIX}{index:02d}", "arrived_at": 0.0,
                      "num_prefill_tokens": longest["num_prefill_tokens"],
                      "num_decode_tokens": longest["num_decode_tokens"]}
            for index in range(sweep["long_requests_per_lane"] * lanes)]
    interval = sweep["lane_interval_s"] / lanes
    return rows + [common | {"request_id": f"{SHORT_SWEEP_PREFIX}{index:05d}",
                             "arrived_at": round(index * interval, 6),
                             "num_prefill_tokens": sweep["short_prompt_tokens"],
                             "num_decode_tokens": sweep["short_output_tokens"]}
                   for index in range(math.ceil(sweep["duration_s"] / interval))]


def length_sweep_coverage(records: list[dict]) -> dict:
    """The sweep covers the formal range when short requests keep arriving until every long
    request has produced its last token."""
    long = [record for record in records if record["request_id"].startswith(LONG_SWEEP_PREFIX)]
    dispatches = sorted(record["dispatch_monotonic"] for record in records
                        if record["request_id"].startswith(SHORT_SWEEP_PREFIX)
                        and record["dispatch_monotonic"] is not None)
    if not long or any(record["status"] != "completed" for record in long) or len(dispatches) < 2:
        return {"coverage_verified": False}
    last_token = max(record["last_token_monotonic"] for record in long)
    return {"last_long_token_monotonic": last_token, "last_short_dispatch_monotonic": dispatches[-1],
            "max_short_dispatch_gap_s": max(later - earlier for earlier, later in zip(dispatches, dispatches[1:])),
            "coverage_verified": dispatches[-1] >= last_token}


def validate_record(record: dict) -> list[str]:
    """Require native counts, prompt identity and real token timestamps, never SSE counts."""
    errors = []
    usage = record.get("usage") or {}
    expected_prompt, expected_output = record["num_prefill_tokens"], record["num_decode_tokens"]
    if record.get("http_status") != 200 or record.get("status") == "failed":
        errors.append("HTTP or transport failure")
    if (not record.get("stream_completed") or record.get("stream_error")
            or record.get("finish_reason") != "length"):
        errors.append("Incomplete or failed fixed-length stream")
    if (usage.get("prompt_tokens") != expected_prompt
            or record.get("accepted_prompt_token_count") != expected_prompt
            or record.get("accepted_prompt_sha256") != record.get("prompt_sha256")):
        errors.append("Accepted prompt count or token-ID identity mismatch")
    if (usage.get("completion_tokens") != expected_output
            or record.get("generated_token_count") != expected_output):
        errors.append("Native output-token count mismatch")
    stamps = [record.get(name) for name in (
        "dispatch_monotonic", "first_token_monotonic", "last_token_monotonic", "response_monotonic")]
    if (any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in stamps)
            or stamps != sorted(stamps)):
        errors.append("Missing or unordered native token timestamps")
    record["validation_errors"] = errors
    record["status"] = "completed" if not errors else "failed"
    if not errors:
        start, first, last, _ = stamps
        record.update(ttft_ms=1000 * (first - start), e2e_ms=1000 * (last - start),
                      tpot_ms=1000 * (last - first) / (expected_output - 1))
    return errors


def formal_metrics(records: list[dict]) -> dict | None:
    if not records or any(record.get("status") != "completed" for record in records):
        return None
    start = min(record["dispatch_monotonic"] for record in records)
    stop = max(record["last_token_monotonic"] for record in records)
    total = sum(record["num_prefill_tokens"] + record["num_decode_tokens"] for record in records)
    output = sum(record["num_decode_tokens"] for record in records)
    result = {"first_submission_monotonic": start, "last_output_monotonic": stop,
              "window_s": stop - start, "formal_requests": len(records), "total_tokens": total,
              "total_thpt_tokens_s": total / (stop - start), "output_tokens": output,
              "output_thpt_tokens_s": output / (stop - start),
              "request_thpt_requests_s": len(records) / (stop - start)}
    for field in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = sorted(record[field] for record in records)
        result[field] = {"mean": statistics.mean(values), "p50": statistics.median(values),
                          "p90": values[math.ceil(len(values) * .90) - 1],
                          "p95": values[math.ceil(len(values) * .95) - 1]}
    return result


def idle_metrics(text: str, dp_size: int) -> dict:
    lanes = {str(index): {} for index in range(dp_size)}
    completed = {str(index): 0.0 for index in range(dp_size)}
    success_lanes = set()
    for name, labels, value in parse_prometheus(text):
        if labels.get("model_name") != SERVED_MODEL_NAME:
            continue
        key = name.removeprefix("vllm:")
        if key not in (*GAUGES, "request_success_total"):
            continue
        lane = labels.get("engine")
        if lane not in lanes or not math.isfinite(value) or value < 0:
            raise ValueError("Unexpected native metrics lane or value")
        if key == "request_success_total":
            completed[lane] += value
            success_lanes.add(lane)
        else:
            if key in lanes[lane]:
                raise ValueError("Duplicate native idle gauge")
            lanes[lane][key] = value
    if (any(set(values) != set(GAUGES) for values in lanes.values())
            or success_lanes != set(lanes)):
        raise ValueError("Missing native running/waiting/KV/completion series for a DP lane")
    return {"lanes": lanes, "completed_requests": sum(completed.values()),
            "completed_per_lane": completed,
            "idle": all(value == 0 for values in lanes.values() for value in values.values())}


def boundary_reached(sample: dict, completed_after: dict | None) -> bool:
    """Every lane of both roles is idle and each role counts at least the expected completions."""
    return not sample["errors"] and all(
        sample["roles"][role]["idle"]
        and (completed_after is None or sample["roles"][role]["completed_requests"] >= completed_after[role])
        for role in ROLES)


def phase_completions(phase: dict) -> int:
    """Requests of a phase the client saw complete; each one completed on both roles.

    A request that failed at the client may never have reached a role (a reused proxy
    connection can reset before the request is sent), so it raises no expected count.
    """
    return phase["recorded_requests"] - len(phase["failed_request_ids"])


async def observe_idle(urls: dict, dp_size: int, timeout_s: float, poll_s: float,
                       completed_after: dict | None = None, *, diagnostic_only=False) -> dict:
    """Observe fresh per-lane native gauges; a timeout never stops subsequent work."""
    import aiohttp

    observations = []
    try:
        async with asyncio.timeout(timeout_s):
            async with aiohttp.ClientSession() as session:
                while True:
                    sample = {"monotonic": time.monotonic(), "roles": {}, "errors": [], "pages": {}}
                    observations.append(sample)

                    async def read(role):
                        try:
                            async with session.get(urls[role] + "/metrics") as response:
                                response.raise_for_status()
                                text = await response.text()
                                sample["pages"][role] = text
                                sample["roles"][role] = idle_metrics(text, dp_size)
                        except Exception as error:
                            sample["errors"].append(f"{role}: {type(error).__name__}: {error}")

                    await asyncio.gather(*(read(role) for role in ROLES))
                    if diagnostic_only:
                        return {"verified": not sample["errors"] and all(
                            sample["roles"][role]["idle"] for role in ROLES),
                            "diagnostic_only": True, "observations": observations}
                    if boundary_reached(sample, completed_after):
                        return {"verified": True, "observations": observations}
                    await asyncio.sleep(poll_s)
    except TimeoutError:
        return {"verified": False, "observations": observations, "error": "Idle/fresh completion deadline"}


async def replay_repetitions(workload, port, vocab_size, origin_lead_s, urls, settings, *, checkpoint=None):
    manifest, formal, warmup = workload
    records = []
    result = {"repetitions": [], "idle_boundaries": [], "observation_errors": [],
               "replay_kind": settings["kind"],
               "reference_admission": "UNVERIFIED", "runtime_qualification": "UNVERIFIED",
              "clock": "prefill-head Linux monotonic; proxy and client on the same host"}

    def persist():
        if checkpoint is not None:
            try:
                checkpoint(result, records)
            except Exception as error:
                result["observation_errors"].append(f"checkpoint: {type(error).__name__}: {error}")
                print("NATIVE_PD_AUXILIARY_UNVERIFIED", result["observation_errors"][-1], flush=True)

    async def boundary(name, completed_after=None):
        try:
            observed = await observe_idle(urls, settings["dp_size"], settings["idle_timeout_s"],
                                          settings["idle_poll_s"], completed_after)
        except Exception as error:
            observed = {"verified": False, "error": f"{type(error).__name__}: {error}"}
        observed["boundary"] = name
        result["idle_boundaries"].append(observed)
        if not observed["verified"]:
            print("NATIVE_PD_AUXILIARY_UNVERIFIED", name, observed.get("error"), flush=True)
        persist()
        return observed

    async def run_phase(rows, windows=()):
        """Replay one phase; return its summary, records and record validation errors."""
        phase_records = []
        copied = 0

        def persist_phase():
            nonlocal copied
            records.extend(phase_records[copied:])
            copied = len(phase_records)

        try:
            clock, _ = await replay(rows, port, vocab_size, origin_lead_s, windows, vllm_version="step5",
                                    records=phase_records,
                                    checkpoint=lambda: persist_phase(), profile_url=urls["decode"])
        finally:
            persist_phase()
            persist()
        errors = []
        for record in phase_records:
            try:
                errors.extend(validate_record(record))
            except Exception as error:
                message = f"record validation: {type(error).__name__}: {error}"
                record.update(status="failed", validation_errors=[message])
                errors.append(message)
        summary = clock | {"planned_requests": len(rows), "recorded_requests": len(phase_records),
                           "failed_request_ids": [row["request_id"] for row in phase_records
                                                  if row["status"] != "completed"]}
        return summary, phase_records, errors

    # Native mostrecent gauges may not exist before the first scheduler update.
    # Retain one truthful startup snapshot, never infer idle from missing series.
    try:
        idle = await observe_idle(urls, settings["dp_size"], settings["idle_timeout_s"],
                                  settings["idle_poll_s"], diagnostic_only=True)
    except Exception as error:
        idle = {"verified": False, "diagnostic_only": True, "error": f"{type(error).__name__}: {error}"}
    idle["diagnostic_only"] = True
    result["startup_observation"] = idle
    # These are expected completions of this newly started run, not observed gauges.
    completed = {role: 0 for role in ROLES}
    qualification = settings["kind"] == "qualification"
    repeats = [1] if qualification else manifest["repetitions"]["repeat_indices"]
    if "length_sweep" in settings:
        # A soft gate: the sweep only prepares the engines, and its verdict never touches the
        # client workload verdict. Later boundaries count completions from the observed ones.
        summary, sweep_records, _ = await run_phase(length_sweep_rows(formal, settings))
        result["length_sweep"] = summary | length_sweep_coverage(sweep_records)
        if not result["length_sweep"]["coverage_verified"] or summary["failed_request_ids"]:
            print("NATIVE_PD_AUXILIARY_UNVERIFIED length_sweep coverage",
                  result["length_sweep"]["coverage_verified"], "failed", len(summary["failed_request_ids"]),
                  flush=True)
        idle = await boundary("after_length_sweep")
        completed = ({role: idle["observations"][-1]["roles"][role]["completed_requests"] for role in ROLES}
                     if idle["verified"] else None)
    for repeat in repeats:
        # Only the startup observation, which precedes the first repetition without a sweep, may be
        # diagnostic; every other boundary must have been verified idle.
        current = {"repeat": repeat, "phases": {},
                   "client_workload_verified": idle.get("diagnostic_only", False) or idle["verified"]}
        result["repetitions"].append(current)
        phases = (("warmup", warmup), ("concurrent", formal[:settings["concurrent_requests"]])) if qualification else (
            ("warmup", warmup), ("formal", formal))
        for phase, templates in phases:
            rows = phase_rows(manifest, templates, repeat, phase)
            if qualification:
                rows = [row | {"request_id": "qualification-" + row["request_id"], "role": "diagnostic",
                               **({"arrived_at": 0.0} if phase == "concurrent" else {})} for row in rows]
            # Diagnostic device-timeline runs profile the decode servers during the first formal phase.
            windows = (settings.get("decode_profile_windows", [])
                       if phase == "formal" and repeat == repeats[0] else [])
            current["phases"][phase], phase_records, errors = await run_phase(rows, windows)
            current["client_workload_verified"] &= not errors and len(phase_records) == len(rows)
            if phase == "formal":
                try:
                    current["metrics"] = formal_metrics(phase_records)
                except Exception as error:
                    current["metrics"] = None
                    current["client_workload_verified"] = False
                    result["observation_errors"].append(f"metrics: {type(error).__name__}: {error}")
            if completed is not None:
                completed = {role: count + phase_completions(current["phases"][phase])
                             for role, count in completed.items()}
            before = idle
            idle = await boundary(f"after_repeat{repeat:02d}_{phase}", completed)
            current["client_workload_verified"] &= idle["verified"]
            try:
                if before["verified"] and idle["verified"]:
                    current["phases"][phase]["transfer_delta"] = {
                        role: transfer_delta(parse_prometheus(before["observations"][-1]["pages"][role]),
                                             parse_prometheus(idle["observations"][-1]["pages"][role]))
                        for role in ROLES}
            except Exception as error:
                result["observation_errors"].append(f"transfer: {type(error).__name__}: {error}")
            persist()
    result["client_workload_verified"] = (all(row["client_workload_verified"] for row in result["repetitions"])
                                            and not result["observation_errors"])
    result["status"] = "CLIENT_WORKLOAD_VERIFIED" if result["client_workload_verified"] else "REFERENCE_UNSTABLE"
    result["repeat_metric_means"] = {
        field: statistics.mean(row["metrics"][field]["mean"] for row in result["repetitions"])
        for field in ("ttft_ms", "tpot_ms", "e2e_ms")
    } if not qualification and all(row.get("metrics") is not None for row in result["repetitions"]) else None
    if result["repeat_metric_means"] is not None:
        result["repeat_metric_means"]["total_thpt_tokens_s"] = statistics.mean(
            row["metrics"]["total_thpt_tokens_s"] for row in result["repetitions"])
        result["repeat_variation"] = {}
        for field, mean in result["repeat_metric_means"].items():
            values = [(row["metrics"][field] if field == "total_thpt_tokens_s" else row["metrics"][field]["mean"])
                      for row in result["repetitions"]]
            result["repeat_variation"][field] = {"values": values, "min": min(values), "max": max(values),
                                                  "sample_cv": statistics.stdev(values) / mean if mean else None}
    persist()
    if result["observation_errors"]:
        result.update(client_workload_verified=False, status="REFERENCE_UNSTABLE")
    return result, records

async def run(args):
    from aiohttp import web

    engine = json.loads(args.engine_config.read_text())
    # The position-routing image reuses the Step adapter under its own label.
    if runtime_version(engine) not in ("step5", "step5_position_cyclic") or args.mode != "clean":
        raise ValueError("The Step canonical consumer requires the locked clean Step adapter")
    settings = json.loads((args.trace_dir / "replay_plan.json").read_text())
    workload = canonical_workload(args.trace_dir / "workload")
    model = local_model_config(args.model_dir, "step5")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(result, records):
        (args.output_dir / "client_requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        (args.output_dir / "canonical_summary.json").write_text(json.dumps(result, indent=1))

    app = build_app(args.prefill_url, args.decode_url, "", "", args.output_dir / "proxy_requests.jsonl",
                    vllm_version="step5")
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", args.proxy_port).start()
        result, _ = await replay_repetitions(workload, args.proxy_port, model["vocab_size"],
                                             args.origin_lead_s,
                                             {"prefill": args.prefill_url, "decode": args.decode_url},
                                             settings, checkpoint=checkpoint)
        print("CANONICAL_DONE", json.dumps({"status": result["status"], "reference_admission": "UNVERIFIED"}), flush=True)
    finally:
        await runner.cleanup()
    # Auxiliary admission diagnostics do not fail or cancel the native worker lifecycle.
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    for name in ("engine-config", "trace-dir", "output-dir", "model-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("mode", "prefill-url", "decode-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument("--origin-lead-s", type=float, default=1.0)
    parser.add_argument("--startup-timeout-s", type=float)
    parser.add_argument("--drain-s", type=float)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
