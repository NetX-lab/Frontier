"""Check syntax, module size, changed-source whitespace and measured hashes.

Usage: verify_static.py <repo> <base-repo> <candidate-identity-root> <output.json>
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys


SOURCES = (
    "frontier/moe_ep_workload.py",
    "frontier/execution_time_predictor/deepseek_v4.py",
    "frontier/execution_time_predictor/dsv4_operator_predictor.py",
    "frontier/execution_time_predictor/dsv4_predictor.py",
    "frontier/execution_time_predictor/dsv4_training.py",
)
EXPECTED = {SOURCES[0], SOURCES[1], SOURCES[3]}


def verify(repo, base, identity, output):
    records = Path(__file__).resolve().parent
    paths = [repo / name for name in SOURCES]
    paths += [repo / "tests/unit/test_v4_prediction_memo.py", *sorted(records.glob("*.py"))]
    line_counts = {}
    for path in paths:
        text = path.read_text()
        ast.parse(text, filename=str(path))
        line_counts[str(path.relative_to(repo))] = len(text.splitlines())
    changed, whitespace_errors = [], []
    for name in SOURCES:
        old, new = (base / name).read_text(), (repo / name).read_text()
        if old != new:
            changed.append(name)
        for line in difflib.unified_diff(old.splitlines(), new.splitlines()):
            if line.startswith("+") and not line.startswith("+++") and line.rstrip() != line:
                whitespace_errors.append(dict(file=name, line=line))
    receipt = json.loads((identity / "receipt.json").read_text())
    hashes_match = all(hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
                       for name, digest in receipt["source_sha256"].items())
    shell_checks = {}
    for path in sorted(records.glob("*.sh")):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        shell_checks[path.name] = dict(exit_code=result.returncode, stderr=result.stderr)
    json_files = sorted(records.glob("*.json"))
    for path in json_files:
        json.loads(path.read_text())
    checks = dict(ast_parse=True, modules_at_most_2000_lines=max(line_counts.values()) <= 2000,
                  changed_source_set=set(changed) == EXPECTED, added_source_whitespace=not whitespace_errors,
                  measured_source_hashes_match=hashes_match,
                  shell_syntax=all(r["exit_code"] == 0 for r in shell_checks.values()), json_parse=True)
    result = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks, line_counts=line_counts,
                  changed_sources=changed, whitespace_errors=whitespace_errors, shell_checks=shell_checks,
                  parsed_json_files=[p.name for p in json_files])
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = verify(*(Path(arg).resolve() for arg in sys.argv[1:5]))
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "PASS")
