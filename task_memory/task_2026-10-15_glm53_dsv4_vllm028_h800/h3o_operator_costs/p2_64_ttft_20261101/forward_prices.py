"""Per-class forward prices and per-scope deltas across P2-64 table generations (read-only)."""
import collections, gzip, json, statistics, sys
from pathlib import Path

RUNS = {
    "B29": Path("/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_k19b29"),
    "pass1_all": Path("/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/runs/pass1_all_p2_64"),
    "pass2_all": Path("/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/runs/pass2_all_p2_64"),
}
case = sys.argv[1] if len(sys.argv) > 1 else "p2_64"
if case == "p2_32":
    RUNS = {k: Path(str(v).replace("p2_64", "p2_32")) for k, v in RUNS.items()}


def load(run):
    groups = collections.defaultdict(list)
    for line in gzip.open(run / "forwards.jsonl.gz", "rt"):
        f = json.loads(line)
        if f["phase"] != "serving":
            continue
        groups[(f["role"], f["step_class"], f["mode"], f["local_tokens"] if f["step_class"] == "prefill" else "*")].append(f)
    return groups


data = {name: load(run) for name, run in RUNS.items()}
keys = sorted(set().union(*data.values()), key=str)
print("role/class/mode/tokens | count per gen | mean composed_ms per gen | delta vs B29")
for key in keys:
    counts = [len(data[n].get(key, [])) for n in RUNS]
    means = [statistics.mean(f["composed_ms"] for f in data[n][key]) if data[n].get(key) else float("nan") for n in RUNS]
    print(key, counts, [round(m, 4) for m in means], [round(m - means[0], 4) for m in means[1:]])
print()
for key in keys:
    if not all(data[n].get(key) for n in RUNS):
        continue
    scopes = collections.defaultdict(lambda: [0.0] * len(RUNS))
    for j, n in enumerate(RUNS):
        fs = data[n][key]
        for f in fs:
            for s, v in f["scope_ms"].items():
                scopes[s][j] += v / len(fs)
    ranked = sorted(scopes.items(), key=lambda kv: -max(abs(kv[1][1] - kv[1][0]), abs(kv[1][2] - kv[1][0])))[:8]
    print(key, "top scope deltas (B29 ms, pass1-B29, pass2-B29):")
    for s, v in ranked:
        print("   ", s, round(v[0], 4), round(v[1] - v[0], 4), round(v[2] - v[0], 4))
