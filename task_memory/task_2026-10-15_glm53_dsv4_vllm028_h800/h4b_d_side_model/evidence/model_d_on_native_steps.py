"""Price part d with the proposed Frontier READ rule on the native D step timeline (H200 P2-32 lifecycle run).

Rule: the READ scheduled in step i is posted when step i starts executing, which is the update of step i-1; the
transfer takes T; the first step j > i that starts executing at or after post + T detects it; the update of step j
reports it. d_model = update(j) - schedule(i). Native d = finished_recving update - schedule(i).
"""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
transfer_ms = [float(x) for x in sys.argv[2:]]
for T in transfer_ms:
    native, model, diff = [], [], []
    for f in sorted(root.glob("nodes/*/scheduler_lifecycle/*.jsonl")):
        rows = [json.loads(line) for line in f.open()]
        if rows[0]["role"] != "decode":
            continue
        sched = {r["step"]: r for r in rows if r["event"] == "schedule"}
        upd = {r["step"]: r["t"] for r in rows if r["event"] == "update"}
        read = {}
        for r in rows:
            if r["event"] == "schedule":
                for rid in r.get("recv", []):
                    read[rid] = r["step"]
            elif r["event"] == "update":
                for rid in r.get("finished_recving", []):
                    i = read.pop(rid, None)
                    if i is None or "formal" not in rid or (i - 1) not in upd:
                        continue
                    # Decoding lanes only: step i carried tokens, as in the parts analysis.
                    if not sched[i].get("tokens"):
                        continue
                    post = upd[i - 1]
                    j = i + 1
                    while upd[j - 1] < post + T / 1e3:
                        j += 1
                    d_model = (upd[j] - sched[i]["t"]) * 1e3
                    d_native = (r["t"] - sched[i]["t"]) * 1e3
                    native.append(d_native)
                    model.append(d_model)
                    diff.append(d_model - d_native)
    m = lambda xs: round(statistics.mean(xs), 2)
    print(f"T={T} ms n={len(native)} native d={m(native)} model d={m(model)} model-native={m(diff)} "
          f"model d p50={round(statistics.median(model), 2)} min={round(min(model), 2)} max={round(max(model), 2)}")
