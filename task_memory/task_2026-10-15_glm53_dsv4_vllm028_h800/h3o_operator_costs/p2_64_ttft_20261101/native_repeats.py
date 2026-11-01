"""Native P2-64 per-repeat TTFT with and without first-contact requests, against each prediction (read-only)."""
import csv, json, statistics
from pathlib import Path

GATES = Path("/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/gates")
FIRST_CONTACT = {f"fixed-6144-1024-req128-qps1-formal-r{n:03d}" for n in (6, 17, 27, 35, 36)}
rows_dir = GATES / "pass1_all_p2_64/rows/vllm"
native = {}
for path in sorted(rows_dir.glob("*.csv")):
    rep = path.stem.rsplit("-", 1)[1]
    native[rep] = {r["request_id"]: float(r["ttft_ms"]) for r in csv.DictReader(open(path))}
assert all(len(v) == 128 for v in native.values())
pred = {}
for gen, gate in (("B29", None), ("pass1_all", "pass1_all_p2_64"), ("pass2_all", "pass2_all_p2_64")):
    if gate:
        f = GATES / gate / "rows/frontier/a10.csv"
        pred[gen] = {r["request_id"]: float(r["ttft_ms"]) for r in csv.DictReader(open(f))}
out = {}
for rep, d in native.items():
    mean_all = statistics.mean(d.values())
    kept = [v for k, v in d.items() if not (rep == "repeat01" and k in FIRST_CONTACT)]
    out[rep] = dict(mean_all=mean_all, n_all=len(d), mean_without_first_contact=statistics.mean(kept), n_kept=len(kept),
                    first_contact_ttft=[round(d[k], 1) for k in sorted(FIRST_CONTACT)] if rep == "repeat01" else [],
                    median=statistics.median(d.values()))
print(json.dumps(out, indent=1))
other = [v for rep in ("repeat02", "repeat03") for k, v in native[rep].items()]
print("repeat01 first-contact excess over same request mean in repeats02/03:",
      [round(native["repeat01"][k] - statistics.mean([native["repeat02"][k], native["repeat03"][k]]), 1) for k in sorted(FIRST_CONTACT)])
print("repeat01 non-first-contact mean minus repeats02/03 same requests:",
      round(statistics.mean(native["repeat01"][k] - (native["repeat02"][k] + native["repeat03"][k]) / 2 for k in native["repeat01"] if k not in FIRST_CONTACT), 2))
for gen, p in pred.items():
    m = statistics.mean(p.values())
    print(gen, "pred", round(m, 3), {rep: round(100 * (m - v["mean_all"]) / v["mean_all"], 3) for rep, v in out.items()},
          "repeat01 without first contact:", round(100 * (m - out["repeat01"]["mean_without_first_contact"]) / out["repeat01"]["mean_without_first_contact"], 3))
