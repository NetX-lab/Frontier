"""Exact parsed metrics, cache names, A11 content, and independent call counts."""
import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a10"))
import compare_cache_runs as cache_compare
from a10_common import sha256


def check(plain_root, observed_root):
    plain, plain_metrics, plain_cache = cache_compare.run_paths(plain_root)
    observed, observed_metrics, observed_cache = cache_compare.run_paths(observed_root)
    comparator = cache_compare.compare
    diffs = comparator.compare_artifact_directories(
        plain_metrics, observed_metrics, cache_compare.run_substitutions(plain_root, plain),
        cache_compare.run_substitutions(observed_root, observed))
    plain_names = [row["name"] for row in cache_compare.cache_inventory(plain_cache)]
    observed_names = [row["name"] for row in cache_compare.cache_inventory(observed_cache)]
    cache_diffs = comparator.classify_cache_differences(plain_names, observed_names)
    names_equal = set(plain_names) == set(observed_names)
    with gzip.open(observed_root / "forwards.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    collectives = json.loads((observed_root / "collectives.json").read_text())["rows"]
    independent = json.loads((observed_root / "independent_counts.json").read_text())
    plain_independent = json.loads((plain_root / "independent_counts.json").read_text())
    collective_counts = Counter()
    for row in collectives:
        collective_counts[row["role"]] += row["call_count"]
    forward_counts = Counter(row["role"] for row in rows)
    unique = len({(row["role"], row["batch_id"], row["schedule_epoch"], row["stage_id"]) for row in rows})
    checks = {
        "parsed_artifacts_identical": not diffs,
        "trace_hashes_match": plain["trace"]["sha256"] == observed["trace"]["sha256"]
            and all(sha256(root / "trace.csv") == receipt["trace"]["sha256"]
                    for root, receipt in ((plain_root, plain), (observed_root, observed))),
        "cache_names_identical": not cache_diffs and names_equal,
        "both_roles_both_lanes": {(row["role"], row["dp"]) for row in rows} == {
            (role, dp) for role in ("prefill", "decode") for dp in (0, 1)},
        "piecewise_first_tail": any(row["mode"] == "PIECEWISE" and row["step_class"].startswith("first_tail") for row in rows),
        "full_multi_request_decode": any(row["mode"] == "FULL" and row["step_class"] == "decode" and row["local_requests"] > 1 for row in rows),
        "dp_dummy": any(not row["real"] for row in rows),
        "none_prefill": any(row["mode"] == "NONE" and row["step_class"] == "prefill" for row in rows),
        "every_stage_end_once": len(rows) == unique == observed["a11"]["stage_end_events"] == sum(independent["stage_end_events"].values()),
        "per_role_stage_ends": dict(forward_counts) == independent["stage_end_events"],
        "collectives_both_roles": set(collective_counts) == {"prefill", "decode"},
        "independent_collective_calls": dict(collective_counts) == independent["collective_calls"],
        "independent_plain_counts_equal": plain_independent["collective_calls"] == independent["collective_calls"] and plain_independent["stage_end_events"] == independent["stage_end_events"],
        "composed_segments_reconcile": all(abs(sum(row["segment_ms"].values()) - row["composed_ms"]) < 1e-6 for row in rows),
        "source_attention_and_prepare_present": all("dsv4_sparse_attention" in row["scope_ms"] and "dsv4_input_preparation" in row["scope_ms"] for row in rows),
        "pre_dp_scopes_retained": all(row["segment_ms"].get(name) == row["scope_ms"][name]
            for row in rows for name in ("kv_cache_block_zero", "dsv4_block_table_write") if name in row["scope_ms"])
            and all(any(name in row["scope_ms"] for row in rows) for name in ("kv_cache_block_zero", "dsv4_block_table_write")),
        "observer_hashes_match": all(sha256(observed_root / filename) == observed["a11"][key]["sha256"]
            for key, filename in (("forwards", "forwards.jsonl.gz"), ("collectives", "collectives.json"))),
    }
    seconds = {"plain": plain["run_wall_s"], "observed": observed["run_wall_s"]}
    seconds["relative_change"] = seconds["observed"] / seconds["plain"] - 1
    seconds["within_two_percent"] = seconds["relative_change"] <= .02
    return {"checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
            "run_wall_s": seconds, "artifact_differences": [vars(diff) for diff in diffs],
            "cache_differences": [vars(diff) for diff in cache_diffs],
            "cache_file_counts": {"plain": len(plain_names), "observed": len(observed_names)},
            "rows": len(rows), "unique_rows": unique, "forward_counts": dict(forward_counts),
            "forward_census": [{"role": role, "dp": dp, "mode": mode, "class": kind, "count": count}
                               for (role, dp, mode, kind), count in sorted(Counter(
                                   (row["role"], row["dp"], row["mode"], row["step_class"]) for row in rows).items())],
            "independent_counts": independent, "collective_call_counts": dict(collective_counts),
            "plain_receipt": str(plain_root / "receipt.json"), "observed_receipt": str(observed_root / "receipt.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plain", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(args.plain.resolve(), args.observed.resolve())
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["checks"], indent=1))
    raise SystemExit(0 if all(value == "PASS" for value in result["checks"].values()) else 1)
