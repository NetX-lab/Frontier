"""Rebind the original canonical comparator without changing its predicates."""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
SOURCE = "1461573de388b72450189bc0c1b88b4ac32491f0"
BASELINE = Path("/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/frozen-unit/results.xml")
ORIGINAL = Path("/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/compare_unit.py")
execution = json.loads((ROOT / "unit/execution.json").read_bytes())
previous = json.loads((BASELINE.parent / "execution.json").read_bytes())
assert execution["source_before"] == execution["source_after"]
assert execution["source_before"]["head"] == SOURCE
assert execution["source_before"]["status"] == ""
assert execution["source_before"]["submodule_status"] == ""
assert execution["source_before"]["submodule_head"] == "3dac53c544e546574cb6e8078cd14053d0916113"
assert execution["native_binary_projection"] is False
assert execution["source_binary_present"] is False
assert execution["executions"][0]["returncode"] == 1
assert execution["versions"] == previous["versions"]
assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == "bb1703a9cb5ca0d28f4fbdb732b391c8f9d38717330e375d48e6e098fe8df344"
source = ORIGINAL.read_text()
changes = {
    "BASELINE = Path('/data/ycfeng/tmp/d29_unit_parent07.xml')": "BASELINE = Path('/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/frozen-unit/results.xml')",
    "CANDIDATE = ROOT/'precommit-unit/results.xml'": "CANDIDATE = ROOT/'unit/results.xml'",
    "64371d6394026aedb9fa8d9c409a9263bfc2f23d72a063886581b61827b8b5b3": "bb1703a9cb5ca0d28f4fbdb732b391c8f9d38717330e375d48e6e098fe8df344",
    "task_id='D29-NATIVE-KEY-BYTES'": "task_id='D29-LEDGER-FINAL-FROZEN-REGRESSIONS'",
    "baseline_source='dbe19f531ff68d9edafc480418e8927c40eb1c60'": "baseline_source='376983d2ecc441d1e0a8cb8b943e65a2dcb217b5'",
    "source_dirty_before_commit=True": "source_dirty_before_commit=False",
    "Precommit source is SHA-bound but dirty;": "Final1461573d source is clean, frozen and unchanged before/after;",
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
namespace = {"__name__": "__main__", "__file__": str(Path(__file__))}
exec(compile(source, str(ORIGINAL), "exec"), namespace)
old, new = namespace["old"], namespace["new"]
for status in ("failure", "error", "skipped"):
    assert {key for key, value in old.items() if value == status} == {key for key, value in new.items() if value == status}
result = json.loads((ROOT / "unit_comparison.json").read_bytes())
result.update(source=SOURCE, versions_identical=True, retained_ids=len(old),
              failure_error_skip_sets_identical=True, native_binary_projection=False,
              original_comparator={"path": str(ORIGINAL), "sha256": hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()},
              adaptations=changes, execution={"path": str(ROOT / "unit/execution.json"), "sha256": hashlib.sha256((ROOT / "unit/execution.json").read_bytes()).hexdigest()})
(ROOT / "unit_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key:result[key] for key in ("status", "retained_ids", "candidate_counts", "raw_candidate_exit", "versions_identical", "failure_error_skip_sets_identical")}, indent=2))
