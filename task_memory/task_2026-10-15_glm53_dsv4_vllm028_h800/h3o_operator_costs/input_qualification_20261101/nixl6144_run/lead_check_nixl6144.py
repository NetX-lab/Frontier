"""Lead's independent check of the nixl6144 download, raw records and Pi's summaries."""
import csv, hashlib, json, statistics
from collections import Counter
from pathlib import Path
from frontier.profiling.collectives.nixl_kv_read_benchmark import summarize_case

DL = Path("/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download")
RUN = Path("task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/"
           "input_qualification_20261101/nixl6144_run")
d = {"paths": json.loads((DL / "download_manifest.json").read_text())}
entry = d["paths"][0]
print("manifest entry keys:", sorted(entry) if isinstance(entry, dict) else type(entry))
bad = []
for e in d["paths"]:
    p = DL / e["path"]
    if hashlib.sha256(p.read_bytes()).hexdigest() != e["sha256"] or p.stat().st_size != e["bytes"]:
        bad.append(e["path"])
local = sorted(str(p.relative_to(DL)) for p in DL.rglob("*") if p.is_file() and p.name not in ("download.json", "download_manifest.json", "download_remote_summary.json"))
print("manifest files", len(d["paths"]), "local files", len(local), "hash/size mismatches", bad)

raw = [json.loads(l) for l in (DL / "C_nixl/nixl_reads.jsonl").read_text().splitlines() if l.strip()]
keys = Counter((r["case"], r["pass"]) for r in raw)
trials = sum(len(rd["timings"]) for r in raw for rd in r["readers"])
notes = (sum(r["notifications_received"] for r in raw), sum(r["notifications_expected"] for r in raw))
blocks = {t: sorted({r["blocks_per_request"] for r in raw if r["prompt_tokens"] == t}) for t in (6143, 6144, 6145)}
bytes_ok = all(r["bytes_per_reader"] == r["blocks_per_request"] * r["requests_in_flight"] * 1002240 for r in raw)
verified = all(rd["data_verified"] is True for r in raw for rd in r["readers"])
positive = all(t["complete_us"] > 0 for r in raw for rd in r["readers"] for t in rd["timings"])
print("records", len(raw), "distinct case/pass", len(keys), "max dup", max(keys.values()),
      "cases", len({k[0] for k in keys}), "passes", sorted({k[1] for k in keys}))
print("trials", trials, "notifications recv/exp", notes, "blocks", blocks, "bytes_ok", bytes_ok,
      "data_verified", verified, "all complete>0", positive)

csv_rows = list(csv.DictReader((DL / "C_nixl/nixl_summary.csv").open()))
mism = sum(1 for r, c in zip(raw, csv_rows) for k, v in summarize_case(r).items() if c.get(k) != str(v))
print("csv rows", len(csv_rows), "rebuild mismatches", mism,
      "copy identical", (RUN / "nixl_summary.csv").read_bytes() == (DL / "C_nixl/nixl_summary.csv").read_bytes())

def p90(v):
    s = sorted(v); return s[min(len(s) - 1, int(0.9 * len(s)))]
summ = {(r["case"], int(r["pass"])): r for r in csv.DictReader((RUN / "case_summary.csv").open())}
worst = 0.0
for r in raw:
    crit = [max(t["complete_us"] for t in rep) / 1e3 for rep in zip(*[rd["timings"] for rd in r["readers"]])]
    reader = [t["complete_us"] / 1e3 for rd in r["readers"] for t in rd["timings"]]
    s = summ[(r["case"], r["pass"])]
    for col, val in (("complete_median_ms", statistics.median(crit)), ("complete_p90_ms", p90(crit)),
                     ("complete_max_ms", max(crit)), ("reader_complete_median_ms", statistics.median(reader)),
                     ("reader_complete_p90_ms", p90(reader)), ("reader_complete_max_ms", max(reader))):
        worst = max(worst, abs(float(s[col]) - val))
    assert int(s["reader_trials"]) == len(reader)
print("case_summary rows", len(summ), "max abs diff vs lead recompute (ms)", worst)
allc = [t["complete_us"] / 1e3 for r in raw for rd in r["readers"] for t in rd["timings"]]
print("all reader trials: min %.4f median %.4f max %.4f ms" % (min(allc), statistics.median(allc), max(allc)))
v = json.loads((RUN / "verification.json").read_text())
print("verification", v["status"], [(c["name"], c["pass"]) for c in v["checks"]])
