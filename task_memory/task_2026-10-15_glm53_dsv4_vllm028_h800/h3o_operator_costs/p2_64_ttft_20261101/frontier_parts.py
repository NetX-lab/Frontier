"""Per-request Frontier P2-64 TTFT parts for B29, pass1_all and pass2_all (read-only)."""
import json, statistics
from pathlib import Path

METRICS = "metrics/deepseek_ai_deepseek_v4_flash/online_serving/a10_p2_64_operator_path"
RUNS = {
    "B29": Path("/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_k19b29"),
    "pass1_all": Path("/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/runs/pass1_all_p2_64"),
    "pass2_all": Path("/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/runs/pass2_all_p2_64"),
}


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def parts(run):
    m = run / METRICS
    done = {r["request_id"]: r for r in rows(m / "metrics_ground_truth.jsonl") if r["event_type"] == "request_completion"}
    kv = {}
    for r in rows(m / "frontier_transfer_ledger.jsonl"):
        if r["transfer_kind"] == "kv_cache":
            [rid] = r["request_ids"]
            kv[rid] = r
    out = {}
    for rid, d in done.items():
        k = kv[rid]
        out[rid] = dict(
            ttft=(d["first_decode_token_completed_at"] - d["arrived_at"]) * 1000,
            p_side=(d["prefill_completed_at"] - d["arrived_at"]) * 1000,
            kv=(k["end_ts_s"] - k["start_ts_s"]) * 1000,
            pre_kv=(k["start_ts_s"] - d["prefill_completed_at"]) * 1000,
            d_first=(d["first_decode_token_completed_at"] - k["end_ts_s"]) * 1000,
            p_replica=d.get("prefill_replica_id"),
        )
    return out


data = {name: parts(run) for name, run in RUNS.items()}
ids = sorted(data["B29"])
assert all(sorted(v) == ids for v in data.values()) and len(ids) == 128
summary = {}
for name, d in data.items():
    summary[name] = {key: statistics.mean(d[i][key] for i in ids) for key in ("ttft", "p_side", "pre_kv", "kv", "d_first")}
for name in ("pass1_all", "pass2_all"):
    delta = {key: [data[name][i][key] - data["B29"][i][key] for i in ids] for key in ("ttft", "p_side", "kv", "d_first")}
    summary[f"{name}-B29"] = {key: dict(mean=statistics.mean(v), min=min(v), max=max(v)) for key, v in delta.items()}
print(json.dumps(summary, indent=1))
print("keys sample", sorted(next(iter(rows(RUNS['B29'] / METRICS / 'metrics_ground_truth.jsonl'))).keys()))
