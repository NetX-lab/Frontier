"""Build the diagnostic V4 step-cost table (H4a) from one profiled native PD run.

Decode steps come from the torch traces of the TP0 rank of each decode DP lane: a step is the
interval from its ``execute_context_k(tokens)_generation_m(tokens)`` GPU annotation to the next
one. The prefill step is the proxy-observed prefill service time (proxy arrival to prefill
response) of the formal requests outside the repeat the run profiled. Every consumed file is checked
against the run's cloud inventory, whose own digest the download receipt records.

The table is a diagnostic input: it carries no formal-gate admission.
"""

import argparse
import bisect
import gzip
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from frontier.execution_time_predictor.dsv4_step_cost_input import QUALIFICATION, SCHEMA


STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
DECODE_TRACE = re.compile(r"torch_traces/decode/dp(\d+)/dp\1_pp0_tp0_[^/]+\.pt\.trace\.json\.gz")
RUN_FILES = ("run/client_requests.jsonl", "run/proxy_requests.jsonl")


def _require(condition, message):
    if not condition:
        raise ValueError(f"V4 step-cost preparation rejected: {message}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_files(run, names):
    """Map each consumed file to its cloud-inventory digest after checking the local copy."""
    receipt = json.loads((run / "download_receipt.json").read_text())
    _require(receipt["status"] == "HASH_IDENTICAL_DOWNLOAD", "download receipt")
    _require(sha256(run / "cloud_inventory.json") == receipt["inventory_sha256"], "inventory digest")
    inventory = {row["path"]: row["sha256"] for row in json.loads((run / "cloud_inventory.json").read_text())["files"]}
    pins = {}
    for name in names:
        _require(sha256(run / name) == inventory[name], f"local copy differs from cloud original: {name}")
        pins[name] = inventory[name]
    return receipt["cloud_root"], pins


def decode_traces(run, dp_size):
    """The TP0 trace of each decode DP lane, in lane order, as listed in the run's cloud inventory."""
    lanes = defaultdict(list)
    for row in json.loads((run / "cloud_inventory.json").read_text())["files"]:
        match = DECODE_TRACE.fullmatch(row["path"])
        if match:
            lanes[int(match.group(1))].append(row["path"])
    _require(sorted(lanes) == list(range(dp_size)) and all(len(paths) == 1 for paths in lanes.values()),
             f"one TP0 decode trace per DP lane: {dict(lanes)}")
    return [lanes[lane][0] for lane in range(dp_size)]


def topology(engine_path, manifest):
    _require(sha256(engine_path) == manifest["inputs_sha256"]["inputs/engine.json"], "engine config digest")
    engine = json.loads(engine_path.read_text())
    _require(engine["enable_expert_parallel"] and not engine["enable_prefix_caching"]
             and not engine["enable_chunked_prefill"] and engine["pipeline_parallel_size"] == 1,
             "engine feature set")
    tp, dp = engine["tensor_parallel_size"], engine["data_parallel_size"]
    return engine, {"attn_tensor_parallel_size": tp, "attn_dp": dp, "moe_tensor_parallel_size": 1,
                    "moe_expert_parallel_size": tp * dp, "num_pipeline_stages": 1}


def decode_steps(trace_path):
    """Per step: (k, k tokens, m, m tokens), interval to the next step and GPU kernel span, in ms."""
    events = json.load(gzip.open(trace_path))["traceEvents"]
    steps = sorted((e for e in events if e.get("cat") == "gpu_user_annotation"
                    and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
    kernels = sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e.get("cat") == "kernel")
    starts = [k[0] for k in kernels]
    rows = []
    for step, following in zip(steps, steps[1:]):
        window = kernels[bisect.bisect_left(starts, step["ts"]):bisect.bisect_left(starts, following["ts"])]
        _require(window, f"step without kernels in {trace_path.name}")
        composition = tuple(map(int, STEP.fullmatch(step["name"]).groups()))
        rows.append((composition, (following["ts"] - step["ts"]) / 1e3,
                     (max(k[1] for k in window) - window[0][0]) / 1e3))
    return rows


def decode_entries(run, traces, max_num_seqs):
    """Pool all lanes; a decode request count above the measured ones is a listed substitution.

    A prefill composition measured only without decode requests also prices that composition with
    decode requests, from the same measured step (listed substitution D-H200-STEP-COST-SUBSTITUTION).
    """
    groups = defaultdict(list)
    for name in traces:
        for (k, k_tokens, m, m_tokens), interval, span in decode_steps(run / name):
            _require(m_tokens == m and k_tokens >= k, f"unexpected step composition {k, k_tokens, m, m_tokens}")
            groups[(k, k_tokens, m > 0)].append((m, interval, span))
    entries = []
    for (k, k_tokens, decoding), samples in sorted(groups.items()):
        measured = sorted({m for m, _, _ in samples})
        entries.append({
            "num_prefill_requests": k, "num_prefill_tokens": k_tokens,
            "num_decode_requests": {"min": 1, "max": max_num_seqs} if decoding else {"min": 0, "max": 0},
            "time_ms": statistics.median(interval for _, interval, _ in samples),
            "samples": len(samples), "measured_num_decode_requests": measured,
            "median_gpu_span_ms": statistics.median(span for _, _, span in samples),
            "max_interval_minus_gpu_span_ms": max(interval - span for _, interval, span in samples),
        })
        if not decoding and (k, k_tokens, True) not in groups:
            entries.append({
                "num_prefill_requests": k, "num_prefill_tokens": k_tokens,
                "num_decode_requests": {"min": 1, "max": max_num_seqs},
                "time_ms": entries[-1]["time_ms"], "samples": 0, "measured_num_decode_requests": [],
                "substituted_from_num_decode_requests": {"min": 0, "max": 0},
            })
    return entries


def prefill_entry(run, prompt_tokens, repeats):
    client = {r["request_id"]: r for r in map(json.loads, (run / "run/client_requests.jsonl").read_text().splitlines())}
    proxy = {r["request_id"]: r for r in map(json.loads, (run / "run/proxy_requests.jsonl").read_text().splitlines())}
    ids = [i for i, r in client.items() if r["segment"] == "formal" and r["repeat"] in repeats]
    _require(ids and all(client[i]["num_prefill_tokens"] == prompt_tokens and proxy[i]["prefill_status"] == 200
                         for i in ids), "prefill population")
    service = [(proxy[i]["prefill_response_monotonic"] - proxy[i]["proxy_arrival_monotonic"]) * 1e3 for i in ids]
    return {"num_prefill_requests": 1, "num_prefill_tokens": prompt_tokens,
            "num_decode_requests": {"min": 0, "max": 0}, "time_ms": statistics.median(service),
            "samples": len(service), "measured_num_decode_requests": [0],
            "min_ms": min(service), "max_ms": max(service)}


def prepare_step_cost(run, engine_path, device, output):
    run = Path(run)
    manifest = json.loads((run / "run_manifest.json").read_text())
    _require(manifest["mode"] == "clean" and manifest["groundtruth_tree_dirty"] is False, "clean native run")
    engine, layout = topology(Path(engine_path), manifest)
    traces = decode_traces(run, engine["data_parallel_size"])
    cloud_root, pins = pinned_files(run, RUN_FILES + tuple(traces))
    workload = json.loads((run / "run/replay_summary.json").read_text())["canonical_repetitions"]
    profiled_repeat = workload["profiling"]["repeat"]
    repeats = [r["repeat"] for r in workload["repetitions"] if r["repeat"] != profiled_repeat]
    _require(len(repeats) == len(workload["repetitions"]) - 1, "profiled repeat")
    prompt_tokens = {r["num_prefill_tokens"] for r in map(json.loads, (run / "run/client_requests.jsonl").read_text().splitlines())}
    _require(len(prompt_tokens) == 1, "one prompt length")
    table = {
        "schema_version": SCHEMA, "qualification": QUALIFICATION, "formal_gate_admission": False,
        "model_name": engine["model_name"], "device": device, "topology": layout,
        "limits": {name: engine[name] for name in ("max_model_len", "max_num_batched_tokens", "max_num_seqs")},
        "roles": {"PREFILL": [prefill_entry(run, prompt_tokens.pop(), repeats)],
                  "DECODE": decode_entries(run, traces, engine["max_num_seqs"])},
        "sources": {"run_tag": manifest["run_tag"], "cloud_root": cloud_root,
                    "engine_config_sha256": sha256(engine_path), "files_sha256": pins,
                    "profiled_repeat": profiled_repeat, "prefill_repeats": repeats},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=1, allow_nan=False) + "\n")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="persisted run copy with run_manifest.json, run/, torch_traces/, cloud inventory and receipt")
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--device", required=True, help="Frontier replica_config_device of the profiled GPUs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = prepare_step_cost(args.run, args.engine_config, args.device, args.output)
    for role, entries in table["roles"].items():
        for e in entries:
            print(role, e["num_prefill_requests"], e["num_prefill_tokens"], e["num_decode_requests"],
                  f"{e['time_ms']:.3f} ms", "samples", e["samples"], "measured m", e["measured_num_decode_requests"])


if __name__ == "__main__":
    main()
