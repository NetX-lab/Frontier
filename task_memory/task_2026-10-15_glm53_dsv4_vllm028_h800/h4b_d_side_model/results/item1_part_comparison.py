"""Compare Frontier's engine_step D-side handoff with the native lifecycle parts.

Usage: item1_part_comparison.py <task directory>

Native parts come from the lifecycle runs (decoding lanes, formal requests). Frontier
parts come from the H4a-5 attribution of the engine_step diagnostic runs: the KV
transfer runs from P end to the READ report, and the first D step from the report to
the first D token. Times are in ms.
"""
import json
import sys
from pathlib import Path
from statistics import mean

CASES = (
    # name, lifecycle requests, repetitions, engine_step attribution, default attribution
    ("native06", "h4b_lifecycle_measurement/results/h800l/lifecycle_requests.jsonl", (1, 2, 3),
     "h4a_p0_diagnostic/results/run_engine_step/h4a5_attribution.json",
     "h4a_p0_diagnostic/results/h4a5_attribution.json"),
    ("p2_32", "h200_mainline/diagnostic/results/p2_32l/lifecycle_requests.jsonl", (1, 2, 3),
     "h200_mainline/diagnostic/results/p2_32e/h4a5_attribution.json",
     "h200_mainline/diagnostic/results/p2_32d/h4a5_attribution.json"),
    ("p2_64", "h200_mainline/diagnostic/results/p2_64l/lifecycle_requests.jsonl", (2, 3),
     "h200_mainline/diagnostic/results/p2_64e/h4a5_attribution.json",
     "h200_mainline/diagnostic/results/p2_64d/h4a5_attribution.json"),
)


def native_parts(path: Path, repeats) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["repeat"] in repeats and r["lane_state"] != "idle"]
    return {
        "requests": len(rows),
        "lane_states": sorted({r["lane_state"] for r in rows}),
        "a+g": mean(r["a"] + r["g"] for r in rows),
        "b": mean(r["b"] for r in rows),
        "c+d+e": mean(r["c"] + r["d"] + r["e"] for r in rows),
        "f": mean(r["f"] for r in rows),
        "remainder a+b+g": mean(r["remainder_ms"] for r in rows),
        "handoff": mean(r["proxy_handoff_ms"] for r in rows),
        "read_transfer_max": mean(r["read_transfer_max_ms"] for r in rows),
    }


def frontier_parts(path: Path) -> dict:
    attribution = json.loads(path.read_text())
    frontier = attribution["frontier"]
    # The H4a run5 attribution predates the case-neutral key names.
    native = attribution.get("native_mean_of_repeats") or attribution["native06_mean_of_repeats"]
    return {
        "kv_transfer": frontier["kv_transfer"]["mean"],
        "d_first_step": frontier["d_first_step"]["mean"],
        "handoff": frontier["kv_transfer"]["mean"] + frontier["d_first_step"]["mean"],
        "tpot": frontier["tpot"]["mean"],
        "target_handoff": native["handoff"],
        "handoff_gap": attribution["comparison"]["handoff_gap_ms"],
        "p_side_gap": attribution["comparison"]["p_side_gap_ms"],
    }


def main() -> None:
    root = Path(sys.argv[1])
    for name, lifecycle, repeats, engine_step, default in CASES:
        n = native_parts(root / lifecycle, repeats)
        e = frontier_parts(root / engine_step)
        d = frontier_parts(root / default)
        print(f"== {name}: lifecycle repetitions {list(repeats)}, {n['requests']} requests, lane states {n['lane_states']}")
        print(f"  native lifecycle: a+g {n['a+g']:.2f}  b {n['b']:.2f}  c+d+e {n['c+d+e']:.2f}  f {n['f']:.2f}  "
              f"remainder {n['remainder a+b+g']:.2f}  handoff {n['handoff']:.2f}  READ transfer max {n['read_transfer_max']:.2f}")
        for tag, f in (("default", d), ("engine_step", e)):
            print(f"  frontier {tag:<11}: kv_transfer {f['kv_transfer']:.2f}  d_first_step {f['d_first_step']:.2f}  "
                  f"handoff {f['handoff']:.2f}  tpot {f['tpot']:.2f}  kv_transfer/tpot {f['kv_transfer'] / f['tpot']:.2f}  "
                  f"d_first_step/tpot {f['d_first_step'] / f['tpot']:.2f}")
        print(f"  target handoff (reference mean of repeats) {e['target_handoff']:.2f}; engine_step handoff gap "
              f"{e['handoff_gap']:+.2f}; P-side gap {e['p_side_gap']:+.2f}")
        print(f"  engine_step minus lifecycle: handoff {e['handoff'] - n['handoff']:+.2f}  "
              f"(b+c+d+e) vs kv_transfer {e['kv_transfer'] - n['b'] - n['c+d+e']:+.2f}  "
              f"f vs d_first_step {e['d_first_step'] - n['f']:+.2f}  a+g {-n['a+g']:+.2f}")
        print(f"  lifecycle minus target handoff {n['handoff'] - e['target_handoff']:+.2f}")


if __name__ == "__main__":
    main()
