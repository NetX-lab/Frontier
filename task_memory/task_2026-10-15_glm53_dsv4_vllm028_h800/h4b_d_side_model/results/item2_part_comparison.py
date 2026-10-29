"""Compare Frontier's depth-2 engine_step D-side handoff with item 1 and the native lifecycle parts.

Usage: item2_part_comparison.py <task directory>

Native and Frontier parts are read as in item1_part_comparison.py. The depth-2 runs
use DECODE capacity 2 (output placeholders) with nixl_read_completion='engine_step'.
Times are in ms.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from item1_part_comparison import frontier_parts, native_parts  # noqa: E402

CASES = (
    # name, lifecycle requests, repetitions, depth-1 attribution, depth-2 attribution
    ("native06", "h4b_lifecycle_measurement/results/h800l/lifecycle_requests.jsonl", (1, 2, 3),
     "h4a_p0_diagnostic/results/run_engine_step/h4a5_attribution.json",
     "h4a_p0_diagnostic/results/run_engine_step_depth2/h4a5_attribution.json"),
    ("p2_32", "h200_mainline/diagnostic/results/p2_32l/lifecycle_requests.jsonl", (1, 2, 3),
     "h200_mainline/diagnostic/results/p2_32e/h4a5_attribution.json",
     "h200_mainline/diagnostic/results/p2_32e2/h4a5_attribution.json"),
    ("p2_64", "h200_mainline/diagnostic/results/p2_64l/lifecycle_requests.jsonl", (2, 3),
     "h200_mainline/diagnostic/results/p2_64e/h4a5_attribution.json",
     "h200_mainline/diagnostic/results/p2_64e2/h4a5_attribution.json"),
)


def main() -> None:
    root = Path(sys.argv[1])
    for name, lifecycle, repeats, depth1, depth2 in CASES:
        n = native_parts(root / lifecycle, repeats)
        one = frontier_parts(root / depth1)
        two = frontier_parts(root / depth2)
        print(f"== {name}: lifecycle repetitions {list(repeats)}, {n['requests']} requests")
        print(f"  native lifecycle: b+c+d+e {n['b'] + n['c+d+e']:.2f}  f {n['f']:.2f}  a+g {n['a+g']:.2f}  "
              f"handoff {n['handoff']:.2f}")
        for tag, f in (("depth 1", one), ("depth 2", two)):
            print(f"  frontier {tag}: kv_transfer {f['kv_transfer']:.2f}  d_first_step {f['d_first_step']:.2f}  "
                  f"handoff {f['handoff']:.2f}  tpot {f['tpot']:.2f}  kv_transfer/tpot {f['kv_transfer'] / f['tpot']:.2f}  "
                  f"d_first_step/tpot {f['d_first_step'] / f['tpot']:.2f}  handoff gap {f['handoff_gap']:+.2f}  "
                  f"P-side gap {f['p_side_gap']:+.2f}")
        print(f"  depth 2 minus depth 1: kv_transfer {two['kv_transfer'] - one['kv_transfer']:+.2f}  "
              f"d_first_step {two['d_first_step'] - one['d_first_step']:+.2f}  "
              f"handoff {two['handoff'] - one['handoff']:+.2f}  (2 x tpot {2 * one['tpot']:.2f})")
        print(f"  depth 2 minus lifecycle: handoff {two['handoff'] - n['handoff']:+.2f}  "
              f"(b+c+d+e) vs kv_transfer {two['kv_transfer'] - n['b'] - n['c+d+e']:+.2f}  "
              f"f vs d_first_step {two['d_first_step'] - n['f']:+.2f}  a+g {-n['a+g']:+.2f}")
        print(f"  target handoff {two['target_handoff']:.2f}; lifecycle minus target {n['handoff'] - two['target_handoff']:+.2f}")


if __name__ == "__main__":
    main()
