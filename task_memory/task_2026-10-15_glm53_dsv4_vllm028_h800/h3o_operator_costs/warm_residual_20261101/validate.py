"""Validate the analysis arithmetic and preserved scope; no simulation rerun."""
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
s = json.load(open(out / "summary.json"))
w = json.load(open(out / "sweep_summary.json"))
checks = []
for case, n, expected in (("p2_64", 379, 720.6396803218075), ("p2_32", 384, 364.0296495311001)):
    assert s[case]["native"]["warm_n"] == n
    assert abs(s[case]["native"]["warm"]["ttft"] - expected) < 1e-7
    alloc = w["native_allocation"][case]
    assert abs(sum(v for k, v in alloc.items() if k != "ttft") - expected) < 1e-7
    for gen, d in s[case]["frontier"].items():
        assert abs(d["fifo"]["wait_error_ms"]) < 0.11
        rows = json.load(open(out / f"{case}_{gen}_rows.json"))
        assert len(rows) == 128 and len({r["request"] for r in rows}) == 128
        for r in rows:
            assert abs(r["first_decode"] - r["first_decode_queue"] - r["first_decode_compute"]) < 1e-7
        checks.append(f"{case}/{gen}: population, spans, first-decode split and FIFO <=0.11 ms PASS")
for gen, d in w["price_neighborhood_averaging"].items():
    assert 688.8 <= d["ttft"] <= 689.6
    checks.append(f"{gen}: scratch-compatible price-neighborhood TTFT {d['ttft']:.6f} ms PASS")
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.py")}
json.dump(dict(status="PASS", checks=checks, script_sha256=hashes), open(out / "validation.json", "w"), indent=2)
print("PASS:", len(checks), "case/generation checks; analysis only")
