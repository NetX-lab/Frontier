"""Exact B20 parsed simulator artifacts and predictor cache name comparison.

Usage: compare_runs.py <base-run-root> <candidate-run-root> <comparison.json>
"""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys


def compare(base, candidate):
    receipts = [json.loads((root / "receipt.json").read_text()) for root in (base, candidate)]
    if any(receipt["status"] != "FINISHED" for receipt in receipts):
        raise ValueError("both identity runs must finish")
    repo = Path(receipts[1]["repo"])
    spec = importlib.util.spec_from_file_location("b20_fidelity", repo / "tests/e2e/refactor_fidelity/compare.py")
    fidelity = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fidelity
    spec.loader.exec_module(fidelity)
    substitutions = [fidelity.substitutions_for(Path(r["repo"]), root, "metrics")
                     for root, r in zip((base, candidate), receipts)]
    metrics = [Path(r["metrics_dir"]) for r in receipts]
    differences = [asdict(item) for item in fidelity.compare_artifact_directories(
        *metrics, *substitutions)]
    names = [[item["name"] for item in r["cache_files"]] for r in receipts]
    cache_findings = [asdict(item) for item in fidelity.classify_cache_differences(*names)]
    cc_present = [(root / "cc_cache").is_dir() for root in (base, candidate)]
    cc_diff = ([asdict(item) for item in fidelity.compare_artifact_directories(
        base / "cc_cache", candidate / "cc_cache", *substitutions)] if any(cc_present) else [])
    evidence = []
    for root in (base, candidate):
        rows = [json.loads(line) for line in (root / "forward_dispatch.jsonl").read_text().splitlines()]
        prefill = [r for r in rows if r["role"] == "PREFILL" and not r["dummy"]]
        tails = [r for r in rows if r["role"] == "DECODE" and any(
            c == p - 1 and q == 1 and rid in {entry[0] for entry in r["new_rows"]}
            for c, p, q, rid in zip(r["context_lens"], r["prompt_lens"], r["query_lens"], r["request_ids"]))]
        full = [r for r in rows if r["role"] == "DECODE" and len(r["request_ids"]) > 1
                and r["dispatch"] and r["dispatch"]["runtime_mode"] == "FULL"]
        dummy = [r for r in rows if r["dummy"]]
        evidence.append(dict(root=str(root), prefill_lanes=sorted({r["lane"] for r in prefill}),
                             first_tails=len(tails), tail_modes=sorted({r["dispatch"]["runtime_mode"] for r in tails}),
                             multirequest_full_forwards=len(full), dummy_forwards=len(dummy),
                             sample_prefill=prefill[:2], sample_tail=tails[:1], sample_full=full[:1], sample_dummy=dummy[:1]))
    observer_equal = (base / "forward_dispatch.jsonl").read_text() == (candidate / "forward_dispatch.jsonl").read_text()
    exercised = all(e["prefill_lanes"] == [0, 1] and e["first_tails"] > 0 and e["tail_modes"] == ["PIECEWISE"]
                    and e["multirequest_full_forwards"] > 0 and e["dummy_forwards"] > 0 for e in evidence)
    return dict(status="PASS" if not differences and not cache_findings and not cc_diff and observer_equal and exercised else "FAIL",
                parsed_artifacts=fidelity.list_artifacts(metrics[0]),
                metrics_differences=differences, cache_name_differences=cache_findings,
                cache_files_each=[len(n) for n in names], cc_differences=cc_diff, cc_cache_produced=cc_present,
                observer_equal=observer_equal, exercised=exercised, evidence=evidence,
                trace_sha256=[r["trace_sha256"] for r in receipts],
                timing=[{k: r[k] for k in ("init_wall_s", "run_wall_s", "simulated_time_s", "peak_rss_kib")}
                        for r in receipts])


if __name__ == "__main__":
    result = compare(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "evidence"}, indent=2))
    raise SystemExit(result["status"] != "PASS")
