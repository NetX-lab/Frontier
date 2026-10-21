"""Summarize complete Step workloads with the existing strict client-population reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.comparison.calibration.client_metrics import native_client_rows
from tests.integration.step5_pd.canonical_replay import canonical_workload, formal_metrics, local_model_config
from vllm_burst_driver import prompt_token_ids


def summarize(stage: Path) -> dict:
    root = stage / "evidence"
    workload = canonical_workload(stage / "trace/workload")
    config = json.loads((stage / "config.json").read_text())
    model_config = Path(config["model_dir"]) / "config.json"
    assert hashlib.sha256(model_config.read_bytes()).hexdigest() == config["model_hashes"]["config.json"]
    vocab_size = local_model_config(model_config.parent, version="step5")["vocab_size"]
    prompt_hashes = {}
    for template in workload[1] + workload[2]:
        tokens = prompt_token_ids(template["request_id"], template["num_prefill_tokens"], vocab_size)
        assert all(0 <= token < vocab_size for token in tokens)
        prompt_hashes[template["request_id"]] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    ids = json.loads((stage / "trace/workload/request_ids.json").read_text())
    normalized = native_client_rows(root, workload, ids, source_format="step5")
    records = [json.loads(line) for line in (root / "replay/client_requests.jsonl").read_text().splitlines()]
    proxy = [json.loads(line) for line in (root / "replay/proxy_requests.jsonl").read_text().splitlines()]
    by_id = {row["request_id"]: row for row in proxy}
    assert len(by_id) == len(proxy) == len(records)
    assert set(by_id) == {row["request_id"] for row in records}
    summary = json.loads((root / "replay/canonical_summary.json").read_text())
    for row in records:
        assert row["prompt_id"] == row["template_request_id"]
        assert row["accepted_prompt_sha256"] == row["prompt_sha256"] == prompt_hashes[row["prompt_id"]]
        deltas = row["token_id_deltas"]
        assert all(0 <= token < vocab_size for delta in deltas for token in delta["token_ids"])
        times = [delta["monotonic"] for delta in deltas]
        assert all(math.isfinite(value) for value in times) and times == sorted(times)
        assert (times[0], times[-1]) == (row["first_token_monotonic"], row["last_token_monotonic"])
        assert sum(len(delta["token_ids"]) for delta in deltas) == row["generated_token_count"]
        assert by_id[row["request_id"]]["prefill_status"] == by_id[row["request_id"]]["decode_status"] == 200
    measured = [row for row in records if row["segment"] == "formal"]
    assert len(measured) == sum(len(frame) for frame in normalized.values())
    repeats = []
    for repeat in summary["repetitions"]:
        rows = [row for row in measured if row["repeat"] == repeat["repeat"]]
        actual = formal_metrics(rows)
        assert actual == repeat["metrics"]
        repeats.append({"repeat": repeat["repeat"], **actual})
    fields = ("ttft_ms", "tpot_ms", "e2e_ms")
    pooled = {}
    for field in fields:
        values = sorted(row[field] for row in measured)
        pooled[field] = {"mean": statistics.mean(values), "p90": values[math.ceil(.90 * len(values)) - 1],
                         "min": values[0], "max": values[-1]}
    phases = []
    for row in measured:
        p = by_id[row["request_id"]]
        times = [row["dispatch_monotonic"], p["proxy_arrival_monotonic"],
                 p["prefill_response_monotonic"], row["first_token_monotonic"]]
        assert times == sorted(times)
        phases.append({"request_id": row["request_id"], "repeat": row["repeat"], "ttft_ms": row["ttft_ms"],
                       "dispatch_to_proxy_ms": 1000 * (times[1] - times[0]),
                       "through_prefill_ms": 1000 * (times[2] - times[1]),
                       "after_prefill_ms": 1000 * (times[3] - times[2])})
    timestamps = [(row["dispatch_monotonic"], 1) for row in measured] + [
        (row["last_token_monotonic"], -1) for row in measured]
    active = peak = 0
    for _, change in sorted(timestamps):
        active += change
        peak = max(peak, active)
    assert active == 0
    files = [stage / "config.json", stage / "trace/workload/manifest.json", *[
        root / "replay" / name for name in ("client_requests.jsonl", "proxy_requests.jsonl", "canonical_summary.json")]]
    return {
        "stage": str(stage), "client_population_verified": True,
        "regenerated_prompt_templates": len(prompt_hashes), "vocabulary_bounds_verified": True,
        "runtime_qualification": summary["runtime_qualification"],
        "reference_admission": summary["reference_admission"],
        "requests": len(records), "warmups": len(records) - len(measured), "measured": len(measured),
        "verified_boundaries": len(summary["idle_boundaries"]), "peak_client_inflight": peak,
        "max_dispatch_lateness_s": max(row["dispatch_lateness_s"] for row in measured),
        "pooled_latencies": pooled, "repetitions": repeats,
        "mean_total_thpt_tokens_s": statistics.mean(row["total_thpt_tokens_s"] for row in repeats),
        "mean_output_thpt_tokens_s": statistics.mean(row["output_thpt_tokens_s"] for row in repeats),
        "window_weighted_total_thpt_tokens_s": sum(row["total_tokens"] for row in repeats) / sum(row["window_s"] for row in repeats),
        "ttft_over_5s": [row for row in phases if row["ttft_ms"] > 5000],
        "mean_ttft_phases_ms": {key: statistics.mean(row[key] for row in phases) for key in (
            "dispatch_to_proxy_ms", "through_prefill_ms", "after_prefill_ms")},
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "requests", "warmups", "measured", "verified_boundaries", "pooled_latencies", "peak_client_inflight",
        "mean_total_thpt_tokens_s", "mean_output_thpt_tokens_s", "ttft_over_5s")}, indent=2))
