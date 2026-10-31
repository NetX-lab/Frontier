"""Compare measured curves, exact entries and holdouts in separate interpreters."""

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path


def snapshot(root):
    sys.path.insert(0, str(root))
    from frontier.cc_backend import MeasuredCCBackend, MeasuredCCBackendConfig
    from frontier.cc_backend.backends.measured_cc_backend import logger
    from frontier.types import ClusterType
    logging.disable(logging.CRITICAL)
    result = {}
    for network in ("h200_dgx", "h800_dgx"):
        backend = MeasuredCCBackend(
            MeasuredCCBackendConfig(summary_csv_path=str(root / "data/profiling/network" / network / "summary.csv")),
            ClusterType.MONOLITHIC, network.split("_")[0], network, 32,
        )
        exact = getattr(backend, "_exact_sp_times", None)
        if exact is None:
            exact = backend._unequal_combine_times
        lane_vectors = []
        with Path(backend._source_path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["group"] != "ep" or row["op"] not in ("all_gatherv", "reduce_scatterv"):
                    continue
                tokens = tuple(int(t) for t in row["tokens_per_dp_rank"].split("/"))
                from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
                widths = sequence_parallel_moe_token_widths(tokens, int(row["group_size"]) // len(tokens))
                entry = ((row["op"], "ep", int(row["group_size"])), widths.local_tokens,
                         int(row["max_input_bytes"]))
                if entry in exact:
                    lane_vectors.append((row["op"], int(row["group_size"]), tokens))
        result[network] = dict(
            lane_vectors=sorted(set(lane_vectors)),
            points=[(key, size, time) for key, curve in sorted(backend._curves.items())
                    for size, time in zip(curve.sizes, curve.times_us)],
            calls=[(key, curve.calls) for key, curve in sorted(backend._curves.items())],
            exact=[(key, rows, size, time) for (key, rows, size), time in sorted(exact.items())],
            holdouts=[(key, row) for key, row in backend._holdouts],
        )
    print(json.dumps(result))


def read_snapshot(root):
    result = subprocess.run(["timeout", "120", sys.executable, str(Path(__file__).resolve()),
                             "--snapshot", str(root)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--base-root", type=Path, default=Path(__file__).resolve().parents[5] / "b28-base")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    if args.snapshot:
        snapshot(args.snapshot)
        return
    base, candidate = read_snapshot(args.base_root), read_snapshot(args.root)
    for network in base:
        old, new = base[network], candidate[network]
        print(network)
        assert old["calls"] == new["calls"]
        old_points = {(tuple(key), size): time for key, size, time in old["points"]}
        new_points = {(tuple(key), size): time for key, size, time in new["points"]}
        assert old_points.keys() == new_points.keys()
        changed_points = {key for key in old_points if old_points[key] != new_points[key]}
        expected_points = {((call, "ep", ep), max(size for (key, size) in old_points if key == (call, "ep", ep)))
                           for call in ("all_gatherv", "reduce_scatterv") for ep in (16, 32)} if network == "h200_dgx" else set()
        assert changed_points == expected_points, changed_points
        for key, size in sorted(changed_points):
            print(f"  curve {key} bytes={size}: {old_points[key, size]:.9f} -> {new_points[key, size]:.9f} us")
        old_exact = {(tuple(key), tuple(rows), size): time for key, rows, size, time in old["exact"]}
        new_exact = {(tuple(key), tuple(rows), size): time for key, rows, size, time in new["exact"]}
        assert old_exact.keys() <= new_exact.keys()
        changed_exact = {key for key in old_exact if old_exact[key] != new_exact[key]}
        assert changed_exact == (set(old_exact) if network == "h200_dgx" else set())
        added = new_exact.keys() - old_exact.keys()
        assert all(key[0][0] == "all_gatherv" for key in added)
        for key in sorted(changed_exact):
            print(f"  exact {key}: {old_exact[key]:.9f} -> {new_exact[key]:.9f} us")
        for key in sorted(added):
            print(f"  new exact {key}: {new_exact[key]:.9f} us")
        old_holdouts = Counter(tuple(key) for key, row in old["holdouts"])
        new_holdouts = Counter(tuple(key) for key, row in new["holdouts"])
        for key in sorted(old_holdouts.keys() | new_holdouts.keys()):
            if old_holdouts[key] != new_holdouts[key]:
                print(f"  holdouts {key}: {old_holdouts[key]} -> {new_holdouts[key]}")
        print(f"  totals: curve changes={len(changed_points)}, combine changes={len(changed_exact)}, "
              f"new dispatch entries={len(added)}, exact entries={len(old_exact)} -> {len(new_exact)}, "
              f"holdouts={sum(old_holdouts.values())} -> {sum(new_holdouts.values())}")
        dispatch = {(key[2], rows) for key, rows, size in new_exact if key[0] == "all_gatherv"}
        combine = {(key[2], rows) for key, rows, size in new_exact if key[0] == "reduce_scatterv"}
        print(f"  SP-vector coverage: dispatch={len(dispatch)}, combine={len(combine)}, "
              f"combine-only={sorted(combine - dispatch)}, dispatch-only={sorted(dispatch - combine)}")
        assert dispatch == combine
        dispatch_lanes = {(ep, tuple(lanes)) for call, ep, lanes in new["lane_vectors"] if call == "all_gatherv"}
        combine_lanes = {(ep, tuple(lanes)) for call, ep, lanes in new["lane_vectors"] if call == "reduce_scatterv"}
        print(f"  ordered raw-lane coverage: dispatch={len(dispatch_lanes)}, combine={len(combine_lanes)}, "
              f"combine-only={sorted(combine_lanes - dispatch_lanes)}, dispatch-only={sorted(dispatch_lanes - combine_lanes)}")
        print(f"  covered ordered lane vectors: {sorted(dispatch_lanes)}")
        assert dispatch_lanes == combine_lanes
        assert len(added) == (16 if network == "h200_dgx" else 9)
        assert sum(new_holdouts.values()) == (0 if network == "h200_dgx" else sum(old_holdouts.values()) - 30)
        retained = {json.dumps([key, row], sort_keys=True) for key, row in new["holdouts"]}
        assert retained <= {json.dumps([key, row], sort_keys=True) for key, row in old["holdouts"]}
    print("PASS: only the planned curve, exact-entry and holdout differences; coverage is equal")


if __name__ == "__main__":
    main()
