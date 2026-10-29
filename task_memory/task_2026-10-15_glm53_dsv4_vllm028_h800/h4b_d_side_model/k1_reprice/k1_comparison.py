"""Compare the depth-2 diagnostics with the repriced k=1 entry against item 2 depth 2 and the native lifecycle parts.

Usage: k1_comparison.py <task directory>

D-H4B-K1-REPRICE, next action 1. Native and Frontier parts are read as in
item1_part_comparison.py. Both Frontier runs use DECODE capacity 2 and
nixl_read_completion='engine_step'; they differ only in the k=1-with-decodes DECODE entry.
The expected first D step is the decode-only entry plus the k=1-with-decodes entry, the
composition of the step that follows the READ report while a decode step runs. The gate
metrics are the client-gate readings over all repetitions. Times are in ms.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
from item1_part_comparison import frontier_parts, native_parts  # noqa: E402

CASES = (
    # name, lifecycle requests, repetitions, item 2 depth-2 run, repriced depth-2 run
    ("native06", "h4b_lifecycle_measurement/results/h800l/lifecycle_requests.jsonl", (1, 2, 3),
     "h4a_p0_diagnostic/results/run_engine_step_depth2", "h4a_p0_diagnostic/results/run_k1_reprice_depth2"),
    ("p2_32", "h200_mainline/diagnostic/results/p2_32l/lifecycle_requests.jsonl", (1, 2, 3),
     "h200_mainline/diagnostic/results/p2_32e2", "h200_mainline/diagnostic/results/p2_32_k1_reprice_depth2"),
    ("p2_64", "h200_mainline/diagnostic/results/p2_64l/lifecycle_requests.jsonl", (2, 3),
     "h200_mainline/diagnostic/results/p2_64e2", "h200_mainline/diagnostic/results/p2_64_k1_reprice_depth2"),
)


def decode_entry(table, prefill_requests, with_decodes):
    [entry] = [e for e in table["DECODE"] if (e["num_prefill_requests"], e["num_prefill_tokens"]) == (prefill_requests,) * 2
               and (e["num_decode_requests"]["max"] > 0) == with_decodes]
    return entry["time_ms"]


def expected_first_step(attribution):
    table = attribution["step_table"]
    return decode_entry(table, 0, True) + decode_entry(table, 1, True)


def gate_metrics(run):
    reading = json.loads((run / "gate/reading.json").read_text())
    return reading["verdict"], {m["metric"]: m for m in reading["metrics"]}


def main() -> None:
    root = Path(sys.argv[1])
    for name, lifecycle, repeats, item2, repriced in CASES:
        n = native_parts(root / lifecycle, repeats)
        runs = {"item 2": root / item2, "repriced": root / repriced}
        parts = {tag: frontier_parts(run / "h4a5_attribution.json") for tag, run in runs.items()}
        attributions = {tag: json.loads((run / "h4a5_attribution.json").read_text()) for tag, run in runs.items()}
        print(f"== {name}: lifecycle repetitions {list(repeats)}, {n['requests']} requests")
        print(f"  native lifecycle: b+c+d+e {n['b'] + n['c+d+e']:.2f}  f {n['f']:.2f}  a+g {n['a+g']:.2f}  "
              f"handoff {n['handoff']:.2f}")
        for tag, f in parts.items():
            first = attributions[tag]["frontier"]["d_first_step"]
            print(f"  frontier {tag:8s}: kv_transfer {f['kv_transfer']:.2f}  d_first_step mean {f['d_first_step']:.2f} "
                  f"median {first['median']:.2f} (expected {expected_first_step(attributions[tag]):.2f})  "
                  f"handoff {f['handoff']:.2f}  tpot {f['tpot']:.2f}  handoff gap {f['handoff_gap']:+.2f}  "
                  f"P-side gap {f['p_side_gap']:+.2f}  preemptions max {attributions[tag]['frontier']['preemptions']['max']:.0f}")
        old, new = parts["item 2"], parts["repriced"]
        print(f"  repriced minus item 2: kv_transfer {new['kv_transfer'] - old['kv_transfer']:+.2f}  "
              f"d_first_step {new['d_first_step'] - old['d_first_step']:+.2f}  handoff {new['handoff'] - old['handoff']:+.2f}  "
              f"tpot {new['tpot'] - old['tpot']:+.3f}  (k=1 entry "
              f"{decode_entry(attributions['repriced']['step_table'], 1, True) - decode_entry(attributions['item 2']['step_table'], 1, True):+.2f})")
        print(f"  repriced minus lifecycle: handoff {new['handoff'] - n['handoff']:+.2f}  "
              f"(b+c+d+e) vs kv_transfer {new['kv_transfer'] - n['b'] - n['c+d+e']:+.2f}  "
              f"f vs d_first_step {new['d_first_step'] - n['f']:+.2f}")
        print(f"  target handoff {new['target_handoff']:.2f}")
        for tag, run in runs.items():
            verdict, metrics = gate_metrics(run)
            cells = "  ".join(f"{metric} {m['frontier']:.3f} vs {m['vllm']:.3f} ({m['absolute_error']:+.3f}, "
                              f"{m['relative_error']:.2%}) {m['status']}" for metric, m in metrics.items())
            print(f"  gate {tag:8s} {verdict}: {cells}")


if __name__ == "__main__":
    main()
