"""Lead review of Q4: inventory integrity, verdict/promotion content and one GPU-hour recomputation."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

T = Path("task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800")
A = T / "h3o_operator_costs/formal_p2_20261102/assessment"

inventory = list(csv.DictReader(open(A / "changed_files.csv")))
bad = [r for r in inventory if hashlib.sha256(Path(r[list(r)[0]]).read_bytes()).hexdigest() != r["sha256"]]
on_disk = {str(p.resolve()) for p in A.rglob("*") if p.is_file()} - {str((A / "changed_files.csv").resolve())}
listed = {str(Path(r[list(r)[0]]).resolve()) for r in inventory}
print("inventory rows", len(inventory), "hash mismatches", len(bad), "unlisted files", sorted(on_disk - listed), "missing", sorted(listed - on_disk))

verdict = json.loads((A / "verdict.json").read_text())
print("verdict.json:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("ticket19_verdict", "verdict", "blocking_reasons")} if isinstance(v, dict) else v for k, v in verdict.items()}, indent=1)[:1500])

promotion = json.loads((A / "promotion.json").read_text())
print("promotion keys:", list(promotion))
for key in ("draft_only", "issued_receipt", "predicate_true", "predicate"):
    if key in promotion:
        print(" ", key, str(promotion[key])[:300])

receipt = json.loads((T / "h200_mainline/p2_64_reference/package_01/case/runs/groundtruth_clean/dsv4-h200-p2-64-canonical-20261025-01/platform_terminal_receipt.json").read_text())
print("receipt keys:", list(receipt)[:20])
