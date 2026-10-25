"""H4a-5: split the H4a-4 TTFT and TPOT into parts and compare each part with native evidence.

TTFT, per formal request, in ms:
  native06   client -> proxy, proxy P service (proxy arrival -> prefill response),
             handoff (prefill response -> decode first byte), decode first byte -> client token;
  Frontier   P queue wait, P compute, P residual, KV wait, KV transfer, D first step
             (KV transfer end -> first output token, including the wait for a D step boundary).
The native P service is compared with Frontier's P parts and the native handoff with Frontier's
KV and D parts. TPOT is explained by the step table, profile02's measured decode steps by batch
size, profile02's own client TPOT, and the number of requests decoding at once over time.
profile02's traced k=1 steps split its own handoffs into the part before D's k=1 step and the rest.

Inputs are the persisted copies in inputs/ and the step-cost table; the Frontier run is argv[1].
Usage: h4a5_attribution.py <frontier run dir> <output json>
"""
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from tests.comparison.calibration.prepare_dsv4_step_cost import DECODE_TRACES, STEP, decode_steps  # noqa: E402

NATIVE = HERE / "inputs/native06/run"
PROFILE02 = HERE / "inputs/profile02"
TABLE = REPO / "data/profiling/compute/h800/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def summary(values):
    values = list(values)
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "n": len(values)}


def decoding_at_once(intervals):
    """Time-weighted mean number of requests between first and last output token."""
    edges = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    area, count, previous = 0.0, 0, edges[0][0]
    for time, step in edges:
        area += count * (time - previous)
        count, previous = count + step, time
    return {"mean": area / (edges[-1][0] - edges[0][0]), "max_observed": max(
        sum(1 for start, end in intervals if start <= t < end) for t, _ in edges)}


def native_parts():
    client = {r["request_id"]: r for r in read_jsonl(NATIVE / "client_requests.jsonl") if r["segment"] == "formal"}
    proxy = {r["request_id"]: r for r in read_jsonl(NATIVE / "proxy_requests.jsonl")}
    repeats = defaultdict(list)
    for request_id, c in client.items():
        p = proxy[request_id]
        repeats[f"repeat{c['repeat']:02d}"].append({
            "client_to_proxy": (p["proxy_arrival_monotonic"] - c["dispatch_monotonic"]) * 1e3,
            "p_service": (p["prefill_response_monotonic"] - p["proxy_arrival_monotonic"]) * 1e3,
            "handoff": (p["decode_first_byte_monotonic"] - p["prefill_response_monotonic"]) * 1e3,
            "decode_byte_to_client": (c["first_token_monotonic"] - p["decode_first_byte_monotonic"]) * 1e3,
            "ttft": c["ttft_ms"], "tpot": c["tpot_ms"],
            "decoding": (c["first_token_monotonic"], c["last_token_monotonic"]),
        })
    result = {}
    for repeat, rows in sorted(repeats.items()):
        frame = pd.DataFrame(rows)
        result[repeat] = {name: summary(frame[name]) for name in frame.columns if name != "decoding"}
        result[repeat]["decoding_at_once"] = decoding_at_once(list(frame["decoding"]))
    return result


def frontier_parts(run_dir):
    [ground_truth] = (run_dir / "metrics").rglob("metrics_ground_truth.jsonl")
    completions = {r["request_id"]: r for r in read_jsonl(ground_truth) if r["event_type"] == "request_completion"}
    metrics = pd.read_csv(ground_truth.with_name("request_metrics.csv")).set_index("Request Id")
    rows = []
    for request_id, r in completions.items():
        m = metrics.loc[request_id]
        arrival, prefill_done, first = (r[k] * 1e3 for k in ("arrived_at", "prefill_completed_at",
                                                              "first_decode_token_completed_at"))
        p_wait, p_compute = m["request_waiting_time_prefill"], m["cluster_prefill_computation"]
        rows.append({
            "p_wait": p_wait, "p_compute": p_compute, "p_residual": prefill_done - arrival - p_wait - p_compute,
            "kv_wait": m["transfer_kv_cache_request_start_ts"] - prefill_done, "kv_transfer": m["transfer_kv_cache"],
            "d_first_step": first - m["transfer_kv_cache_request_end_ts"],
            "ttft": first - arrival,
            "tpot": (r["completed_at"] * 1e3 - first) / (r["num_decode_tokens"] - 1),
            "preemptions": m["request_total_preemption_count"],
            "decoding": (first, r["completed_at"] * 1e3),
        })
    frame = pd.DataFrame(rows)
    result = {name: summary(frame[name]) for name in frame.columns if name != "decoding"}
    result["decoding_at_once"] = decoding_at_once(list(frame["decoding"]))
    return result


def profile02_steps():
    """Measured decode step interval by number of decode requests, both lanes pooled."""
    groups = defaultdict(list)
    for name in DECODE_TRACES:
        for (k, _, m, _), interval, _ in decode_steps(PROFILE02 / name):
            groups[("decode_only" if k == 0 else "first_step_after_kv", m)].append(interval)
    steps = defaultdict(dict)
    for (kind, m), intervals in sorted(groups.items()):
        steps[kind][m] = {"median_ms": statistics.median(intervals), "samples": len(intervals)}
    client = [r for r in read_jsonl(PROFILE02 / "run/client_requests.jsonl") if r["segment"] == "formal"]
    profiled = json.loads((PROFILE02 / "run/replay_summary.json").read_text())["canonical_repetitions"]["profiling"]["repeat"]
    tpot = {f"repeat{repeat:02d}{' (profiled)' if repeat == profiled else ''}":
            statistics.fmean(r["tpot_ms"] for r in client if r["repeat"] == repeat)
            for repeat in sorted({r["repeat"] for r in client})}
    return {"steps": steps, "client_tpot_mean_ms": tpot, "handoff_samples": profile02_handoff()}


def profile02_handoff():
    """Place each traced D step that computes a request's last prompt token (k=1) inside that
    request's proxy handoff. Kineto ts is microseconds after baseTimeNanoseconds on the D host;
    the proxy records time.time() on its own host, so the split carries the two hosts' clock offset.
    longest_lane_pause_ms is the longest interval between consecutive traced steps of the lane
    that overlaps the wait from prefill response to the k=1 step (null: no traced step before it)."""
    proxy = read_jsonl(PROFILE02 / "run/proxy_requests.jsonl")
    samples = []
    for name in DECODE_TRACES:
        trace = json.load(gzip.open(PROFILE02 / name))
        base = trace["baseTimeNanoseconds"] / 1e9
        annotations = {cat: sorted((base + e["ts"] / 1e6, e["name"]) for e in trace["traceEvents"]
                                   if e.get("cat") == cat and e["name"].startswith("execute_context"))
                       for cat in ("gpu_user_annotation", "user_annotation")}
        gpu = annotations["gpu_user_annotation"]
        for (start, step), (following, _) in zip(gpu, gpu[1:]):
            if STEP.fullmatch(step).group(1) != "1":
                continue
            [p] = [p for p in proxy if p["prefill_response_time"] < start < p["decode_first_byte_time"]]
            cpu_start = max(t for t, n in annotations["user_annotation"] if n == step and t <= start)
            pauses = [b - a for (a, _), (b, _) in zip(gpu, gpu[1:]) if a < start and b > p["prefill_response_time"]]
            samples.append({
                "trace": name.split("/")[2], "request_id": p["request_id"], "step": step,
                "handoff_ms": (p["decode_first_byte_time"] - p["prefill_response_time"]) * 1e3,
                "prefill_response_to_k1_cpu_start_ms": (cpu_start - p["prefill_response_time"]) * 1e3,
                "k1_cpu_start_to_gpu_start_ms": (start - cpu_start) * 1e3,
                "k1_gpu_step_ms": (following - start) * 1e3,
                "next_step_start_to_first_byte_ms": (p["decode_first_byte_time"] - following) * 1e3,
                "longest_lane_pause_ms": max(pauses) * 1e3 if pauses else None,
            })
    return samples


def main():
    run_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
    table = json.loads(TABLE.read_text())
    native, frontier = native_parts(), frontier_parts(run_dir)
    pooled = {name: statistics.fmean(native[r][name]["mean"] for r in native)
              for name in ("client_to_proxy", "p_service", "handoff", "decode_byte_to_client", "ttft", "tpot")}
    frontier_p = sum(frontier[k]["mean"] for k in ("p_wait", "p_compute", "p_residual"))
    frontier_handoff = sum(frontier[k]["mean"] for k in ("kv_wait", "kv_transfer", "d_first_step"))
    ttft_gap = frontier["ttft"]["mean"] - pooled["ttft"]
    comparison = {
        "ttft_gap_ms": ttft_gap,
        "p_side_gap_ms": frontier_p - pooled["p_service"],
        "handoff_gap_ms": frontier_handoff - pooled["handoff"],
        "client_edges_gap_ms": -(pooled["client_to_proxy"] + pooled["decode_byte_to_client"]),
        "tpot_gap_ms": frontier["tpot"]["mean"] - pooled["tpot"],
    }
    result = {
        "label": "DIAGNOSTIC", "frontier_run": str(run_dir),
        "step_table": {role: [{k: e[k] for k in ("num_prefill_requests", "num_prefill_tokens",
                                                   "num_decode_requests", "time_ms")} for e in entries]
                       for role, entries in table["roles"].items()},
        "native06": native, "native06_mean_of_repeats": pooled, "frontier": frontier,
        "comparison": comparison, "profile02": profile02_steps(),
    }
    output.write_text(json.dumps(result, indent=1) + "\n")
    for name, value in comparison.items():
        print(f"{name:22s} {value:+9.3f}")


if __name__ == "__main__":
    main()
