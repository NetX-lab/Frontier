"""Split native 6144-token prefill steps (schedule -> update) by the worker execute_model record.

    python step_split.py <run artifacts/nodes dir>
"""
import glob
import json
import statistics
import sys

nodes = sys.argv[1]
rows_out = []
for sched_path in sorted(glob.glob(f"{nodes}/*/scheduler_lifecycle/*.jsonl")):
    rows = [json.loads(line) for line in open(sched_path)]
    if rows[0].get("role") != "prefill":
        continue
    node = sched_path.split("/scheduler_lifecycle/")[0]
    worker_paths = glob.glob(f"{node}/worker_dp_wave/*_tp0_*.jsonl")
    workers = [json.loads(line) for line in open(worker_paths[0])]
    execs = sorted((w for w in workers if w.get("event") == "worker_execute_model" and w.get("tokens") == 6144),
                   key=lambda w: w["t"])
    pending = {}
    for r in rows:
        if r["event"] == "schedule" and r.get("tokens") == 6144:
            pending[r["step"]] = r
        elif r["event"] == "update" and r.get("step") in pending:
            s = pending.pop(r["step"])
            if "warmup" in s["new"][0][0]:
                continue
            match = [w for w in execs if s["t"] <= w["t"] <= r["t"]]
            if len(match) != 1:
                continue
            w = match[0]
            ar = w["calls"].get("dp_cpu_all_reduce")
            mono0 = w["mono"] - w["t"]
            ar_wait = (ar["end_mono"] - ar["mono"]) * 1e3 if ar else None
            ar_end_t = ar["end_mono"] - mono0 if ar else w["t"]
            rows_out.append({
                "lane": sched_path.split("/")[-1],
                "step_ms": (r["t"] - s["t"]) * 1e3,
                "sched_to_exec": (w["t"] - s["t"]) * 1e3,
                "exec_start_to_sync": ((ar["mono"] - w["mono"]) * 1e3) if ar else None,
                "sync_wait": ar_wait,
                "sync_end_to_exec_end": (w["end_t"] - ar_end_t) * 1e3,
                "exec_ms": (w["end_t"] - w["t"]) * 1e3,
                "exec_end_to_update": (r["t"] - w["end_t"]) * 1e3,
            })
print("steps", len(rows_out))
keys = ("step_ms", "sched_to_exec", "exec_start_to_sync", "sync_wait", "sync_end_to_exec_end", "exec_end_to_update")
quick = [r for r in rows_out if r["step_ms"] < 400]
for r in quick: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
for label, sel in (("all", rows_out), ("served at once (step < 400 ms)", quick)):
    print(label, "n", len(sel))
    for k in keys:
        v = [r[k] for r in sel if r[k] is not None]
        if v:
            print(f"  {k:22s} mean {statistics.mean(v):8.2f} median {statistics.median(v):8.2f} min {min(v):8.2f} max {max(v):8.2f}")
