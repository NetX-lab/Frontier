"""Verify final DP dispatch and observer transparency on the supplementary warm run.

Usage: verify_final_dispatch.py <cold-candidate-root> <warm-proof-root> <output.json>
"""
import hashlib
import json
from pathlib import Path
import sys

from compare_runs import compare


def verify(cold, warm):
    artifact_check = compare(cold, warm)
    path = warm / "final_forward_dispatch.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    prefill = [r for r in rows if r["role"] == "PREFILL" and not r["dummy"]]
    tails = [r for r in rows if r["role"] == "DECODE" and any(
        c == p - 1 and q == 1 and rid in {entry[0] for entry in r["new_rows"]}
        for c, p, q, rid in zip(r["context_lens"], r["prompt_lens"], r["query_lens"], r["request_ids"]))]
    full = [r for r in rows if r["role"] == "DECODE" and len(r["request_ids"]) > 1
            and r["dispatch"] and r["dispatch"]["runtime_mode"] == "FULL"]
    dummy = [r for r in rows if r["dummy"]]
    lanes = sorted({r["lane"] for r in prefill})
    modes = sorted({r["dispatch"]["runtime_mode"] for r in tails})
    checks = dict(prefill_on_both_lanes=lanes == [0, 1],
                  nixl_first_tails_piecewise=bool(tails) and modes == ["PIECEWISE"],
                  multirequest_decode_full=bool(full), dp_dummy_forward=bool(dummy),
                  warm_observer_artifacts_identical=artifact_check["status"] == "PASS")
    return dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                observer_point="After native dp_dummy_forward._synchronize_dp_descriptors returns",
                observer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                final_record_count=len(rows), prefill_lanes=lanes, first_tail_count=len(tails),
                first_tail_modes=modes, multirequest_full_count=len(full), dummy_count=len(dummy),
                sample_prefill=[next(r for r in prefill if r["lane"] == lane) for lane in lanes],
                sample_first_tail=tails[:1], sample_multirequest_full=full[:1], sample_dummy=dummy[:1],
                artifact_comparison=artifact_check)


if __name__ == "__main__":
    result = verify(*(Path(p).resolve() for p in sys.argv[1:3]))
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("sample_") and k != "artifact_comparison"}, indent=2))
    raise SystemExit(result["status"] != "PASS")
