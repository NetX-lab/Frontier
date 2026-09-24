import sys
from pathlib import Path
P = Path(__file__).parent
sys.path.insert(0, str(P / "trees/after"))
from tests.e2e.refactor_fidelity.compare import PathSubstitution, compare_artifact_directories
same = 0; names = sorted(p.name for p in (P / "examples/after").iterdir())
for name in names:
    subs = {side: [PathSubstitution(str((P / "examples" / side / name).resolve()), "<CASE>"),
                   PathSubstitution(str((P / "trees" / side).resolve()), "<TREE>")] for side in ("before", "after")}
    runs = {side: sorted((P / "examples" / side / name / "metrics").rglob("system_metrics.json")) for side in ("before", "after")}
    assert len(runs["before"]) == len(runs["after"]) == 1, (name, runs)
    diffs = compare_artifact_directories(runs["before"][0].parent, runs["after"][0].parent, subs["before"], subs["after"])
    files = sorted(p.name for p in runs["after"][0].parent.iterdir())
    same += not diffs
    print("SAME" if not diffs else "DIFF", name, len(files), [d.as_record() for d in diffs][:2])
print(f"=== {same} of {len(names)} identical ===")
