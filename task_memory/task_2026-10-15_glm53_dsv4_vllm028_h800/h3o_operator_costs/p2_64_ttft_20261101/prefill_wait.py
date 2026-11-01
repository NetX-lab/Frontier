"""Split Frontier P2-64 prefill side into wait-before-forward and forward, per generation (read-only)."""
import gzip, json, statistics, sys
from pathlib import Path

METRICS = "metrics/deepseek_ai_deepseek_v4_flash/online_serving/a10_p2_64_operator_path"
RUNS = {
    "B29": Path("/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_k19b29"),
    "pass1_all": Path("/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/runs/pass1_all_p2_64"),
    "pass2_all": Path("/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/runs/pass2_all_p2_64"),
}


def load(run):
    done = {r["request_id"]: r for r in map(json.loads, open(run / METRICS / "metrics_ground_truth.jsonl")) if r["event_type"] == "request_completion"}
    fw = {}
    for line in gzip.open(run / "forwards.jsonl.gz", "rt"):
        f = json.loads(line)
        if f["role"] == "prefill" and f["step_class"] == "prefill":
            for rid in f["request_ids"]:
                fw[rid] = f
    out = {}
    for rid, d in done.items():
        f = fw[rid]
        out[rid] = dict(arrival=d["arrived_at"], formed=f["formed_s"], start=f["start_s"], end=f["end_s"], done=d["prefill_completed_at"],
                        dp=f["dp"], replica=f["replica"], ahead=f["queued_ahead_at_formation"], composed=f["composed_ms"])
    return out


data = {n: load(r) for n, r in RUNS.items()}
ids = sorted(data["B29"])
for n, d in data.items():
    wait_form = [(d[i]["formed"] - d[i]["arrival"]) * 1e3 for i in ids]
    form_start = [(d[i]["start"] - d[i]["formed"]) * 1e3 for i in ids]
    fwd = [(d[i]["end"] - d[i]["start"]) * 1e3 for i in ids]
    post = [(d[i]["done"] - d[i]["end"]) * 1e3 for i in ids]
    ahead = [d[i]["ahead"] for i in ids]
    print(n, "arrival->formed", round(statistics.mean(wait_form), 3), "formed->start", round(statistics.mean(form_start), 3),
          "start->end", round(statistics.mean(fwd), 3), "end->prefill_done", round(statistics.mean(post), 3),
          "queued_ahead mean", round(statistics.mean(ahead), 3), "hist", {k: ahead.count(k) for k in sorted(set(ahead))})
    print("   formed->start quantiles", [round(q, 1) for q in statistics.quantiles(form_start, n=10)], "min", round(min(form_start), 1), "max", round(max(form_start), 1))
for n in ("pass1_all", "pass2_all"):
    moved = [(i, (data[n][i]["start"] - data["B29"][i]["start"]) * 1e3) for i in ids]
    big = [m for m in moved if abs(m[1]) > 50]
    same_lane = sum(data[n][i]["dp"] == data["B29"][i]["dp"] for i in ids)
    print(n, "requests whose prefill start moved >50 ms:", len(big), "same dp lane as B29:", same_lane, "first moved request:", min(big)[0] if big else None)
    print("   first 12 start deltas ms", [round(m[1], 2) for m in moved[:12]])
