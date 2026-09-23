"""Compare two JUnit reports by test id: regressions, new failures, skip changes.

Usage: python compare_junit.py <before.xml> <after.xml> <output.json>
"""
import json
import sys
import xml.etree.ElementTree as ET


def outcomes(path):
    result = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        test_id = f"{case.get('classname')}::{case.get('name')}"
        tags = {child.tag for child in case}
        if "failure" in tags:
            state = "failed"
        elif "error" in tags:
            state = "error"
        elif "skipped" in tags:
            state = "skipped"
        else:
            state = "passed"
        result[test_id] = state
    return result


before, after = outcomes(sys.argv[1]), outcomes(sys.argv[2])
common = before.keys() & after.keys()
report = {
    "counts_before": {state: list(before.values()).count(state) for state in set(before.values())},
    "counts_after": {state: list(after.values()).count(state) for state in set(after.values())},
    "regressions": sorted(t for t in common if before[t] == "passed" and after[t] in ("failed", "error")),
    "now_passing": sorted(t for t in common if before[t] in ("failed", "error") and after[t] == "passed"),
    "skip_changes": sorted(t for t in common if (before[t] == "skipped") != (after[t] == "skipped")),
    "new_failures": sorted(t for t in after.keys() - before.keys() if after[t] in ("failed", "error")),
    "only_before": sorted(before.keys() - after.keys()),
    "only_after": sorted(after.keys() - before.keys()),
}
with open(sys.argv[3], "w") as handle:
    json.dump(report, handle, indent=1)
print({key: (len(value) if isinstance(value, list) else value) for key, value in report.items()})
