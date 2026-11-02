"""Issue the P2-32 -> P2-64 30% promotion receipt from the Q4 draft (D-P2-64-FORMAL-B).

The draft in `assessment/promotion.json` is re-checked before issue:
- the draft's predicate flags are all true and it is still a draft;
- every P2-32 relative error in `assessment/gate_readings.csv` (warm and
  all-request, three repeats, four metrics) is finite, has a non-zero native
  value and is at most 0.30, and equals the value the draft records;
- `assessment/verdict.json` records P2-32 as PASS;
- every file the draft binds by path and SHA-256 still has that hash.

The receipt is written once; an existing receipt is never overwritten.

Usage: issue_receipt.py
"""

import csv
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSESSMENT = HERE.parent / "assessment"
DRAFT = ASSESSMENT / "promotion.json"
READINGS = ASSESSMENT / "gate_readings.csv"
VERDICT = ASSESSMENT / "verdict.json"
RECEIPT = HERE / "receipt_p2_32_to_p2_64.json"
LIMIT = 0.30
METRICS = ("ttft_ms", "tpot_ms", "request_e2e_time_ms", "token_throughput_tps")
REPEATS = ("repeat01", "repeat02", "repeat03")
DRAFT_FLAGS = ("numerical_predicate_30pct", "qualified_input_predicate", "formal32_terminal",
               "formal32_runtime_coverage_verified", "promotion_predicate")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return {"path": str(path), "sha256": sha256(path)}


def bound_files(node):
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            yield node["path"], node["sha256"]
        for value in node.values():
            yield from bound_files(value)
    elif isinstance(node, list):
        for value in node:
            yield from bound_files(value)


def main():
    if RECEIPT.exists():
        raise SystemExit(f"{RECEIPT} already exists; a receipt is issued once")
    promotion = json.loads(DRAFT.read_text())
    failures = [flag for flag in DRAFT_FLAGS if promotion[flag] is not True]
    if promotion["draft_only"] is not True or promotion["issued_receipt"] is not False:
        failures.append("draft_only/issued_receipt")
    draft = promotion["draft"]

    rows = [row for row in csv.DictReader(READINGS.open()) if row["case"] == "p2_32"]
    if sorted((row["repeat"], row["metric"]) for row in rows) != sorted(
            (repeat, metric) for repeat in REPEATS for metric in METRICS):
        failures.append("gate_readings.csv does not hold exactly 3 repeats x 4 metrics for p2_32")
    draft_errors = {}
    for run, entry in draft["repetitions"].items():
        repeat = run.rsplit("-", 1)[1]
        for metric in entry["metrics"]:
            draft_errors[repeat, metric["metric"]] = metric["relative_error"]
    errors = []
    for row in rows:
        for side in ("warm", "all"):
            native, error = float(row[f"{side}_native"]), float(row[f"{side}_relative_error"])
            if not math.isfinite(error) or native == 0 or error > LIMIT:
                failures.append(f"{row['repeat']} {row['metric']} {side}: native {native}, error {error}")
            errors.append(error)
        if float(row["warm_relative_error"]) != draft_errors.get((row["repeat"], row["metric"])):
            failures.append(f"{row['repeat']} {row['metric']}: draft value differs from gate_readings.csv")

    verdict = json.loads(VERDICT.read_text())["p2_32"]["verdict"]
    if verdict != "PASS":
        failures.append(f"P2-32 verdict is {verdict}")

    bindings = sorted(set(bound_files(draft)))
    stale = [path for path, expected in bindings if sha256(path) != expected]
    failures += [f"hash changed: {path}" for path in stale]
    if failures:
        raise SystemExit("Receipt not issued:\n" + "\n".join(failures))

    receipt = {
        "receipt": "ticket 19 promotion_32_to_64, 30% launch eligibility",
        "issued_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "issuer": "Claude Code lead for YC under D-P2-64-FORMAL-B (requirements.md)",
        "from_case": "P2-H200-32 (P16+D16, TP8/DP2/EP16/PP1 per role)",
        "unlocks": {
            "case": "P2-H200-64 formal native campaign (P32+D32, TP8/DP4/EP32/PP1 per role)",
            "only": "a new campaign whose package is built after issued_utc, binds this receipt "
                    "by SHA-256 and is submitted after issued_utc",
            "resource_class": "64 H200, step_main, wholly preemptible",
            "not_covered": "exp-1025-155441-668927 (submitted 2026-10-25T15:54:41Z, before this receipt)",
        },
        "predicate": {
            "rule": "every finite relative error <= 0.30 in every required repetition, all reference "
                    "and coverage checks pass; no average across metrics or repetitions; a zero "
                    "native value is undefined and cannot qualify",
            "limit": LIMIT,
            "checked_errors": len(errors),
            "maximum_relative_error": max(errors),
            "result": True,
        },
        "draft": record(DRAFT),
        "gate_readings": record(READINGS),
        "verdict": record(VERDICT),
        "bound_files_rechecked": len(bindings),
        "binding": draft,
        "not_final_acceptance": "The final gate stays at 0.10 per metric and repeat; this receipt "
                                "is launch eligibility only.",
    }
    with RECEIPT.open("x") as stream:
        stream.write(json.dumps(receipt, indent=1) + "\n")
    print(f"ISSUED {RECEIPT} sha256={sha256(RECEIPT)} max_error={max(errors)} "
          f"errors={len(errors)} bound_files={len(bindings)}")


if __name__ == "__main__":
    sys.exit(main())
