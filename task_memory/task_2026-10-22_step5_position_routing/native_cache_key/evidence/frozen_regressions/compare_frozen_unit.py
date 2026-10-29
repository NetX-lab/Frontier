"""Reuse the completed unit comparator with clean frozen inputs and outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
execution = json.loads((ROOT/'frozen-unit/execution.json').read_bytes())
assert execution['source_before'] == execution['source_after']
assert execution['source_before']['head'].startswith('376983d2')
assert execution['source_before']['status'] == ''
assert execution['executions'][0]['returncode'] == 1
source = (ROOT/'compare_unit.py').read_text()
changes = {
    "CANDIDATE = ROOT/'precommit-unit/results.xml'":"CANDIDATE = ROOT/'frozen-unit/results.xml'",
    'source_dirty_before_commit=True':'source_dirty_before_commit=False',
    "(ROOT/'unit_comparison.json')":"(ROOT/'frozen_unit_comparison.json')",
    'Precommit source is SHA-bound but dirty;':'Source376983d2is clean, frozen and unchanged before/after;',
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old,new)
exec(compile(source,str(ROOT/'compare_unit.py'),'exec'), {'__name__':'__main__','__file__':str(ROOT/'compare_frozen_unit.py')})
