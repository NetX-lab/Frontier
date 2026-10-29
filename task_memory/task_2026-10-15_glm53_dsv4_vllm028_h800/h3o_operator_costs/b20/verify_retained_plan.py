"""Supplement the missing checkout fixture with a read-only check of its retained source.

Usage: verify_retained_plan.py <retained FULL_AND_PIECEWISE.json> <result.json>
This does not replace or relabel the failing required pytest check.
"""
import hashlib
import json
from pathlib import Path
import sys

import test_v4_attention_coverage as coverage


if __name__ == "__main__":
    source = Path(sys.argv[1])
    plan, _ = coverage.p1h.__wrapped__()
    raw = source.read_bytes()
    frozen = json.loads(raw)
    assert json.loads(json.dumps(plan.to_json())) == frozen
    assert (json.dumps(plan.to_json(), indent=2, sort_keys=True) + "\n").encode() == raw
    result = dict(status="PASS", source=str(source), sha256=hashlib.sha256(raw).hexdigest(),
                  bytes=len(raw), checks="same JSON equality and exact serialized bytes assertions as the missing-fixture test")
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
