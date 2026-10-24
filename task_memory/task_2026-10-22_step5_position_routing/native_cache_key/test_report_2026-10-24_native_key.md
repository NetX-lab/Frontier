# D29 native-key verification report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record the passing allocation check; implementation/runtime checks are pending. |

## Allocation check

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-preview`.
Source/records HEAD: `18cc3e9e1008d796a08f3b5637c9b10311979735`, with only the
scoped task records modified. Invocation timeout20s; small JSON/TSV reads only.

Exact command:

```sh
python3 -B - <<'PY'
import csv, json
from collections import Counter
from pathlib import Path
root = Path('task_memory/task_2026-10-22_step5_position_routing')
proposal = json.loads((root/'repairs/native_collective_key_scope_amendment.json').read_text())
rows = list(csv.DictReader((root/'repairs/evidence/named_query_scope/named_query_paths.tsv').open(), delimiter='\t'))
by_path = {row['path']: row for row in rows}
assert len(by_path) == proposal['effective_unique_paths'] == len(rows)
before = Counter()
after = Counter()
actual = Counter()
for row in rows:
    before[row['owner']] += int(row['cap'])
    after[row['owner']] += int(row['cap'])
    actual[row['owner']] += int(row['actual'])
assert dict(before) == proposal['owner_gross_ceilings'], dict(before)
assert sum(change['delta'] for change in proposal['path_ceiling_adjustments']) == 0
for change in proposal['path_ceiling_adjustments']:
    row = by_path[change['path']]
    assert int(row['cap']) == change['before']
    assert change['before'] + change['delta'] == change['after']
    assert int(row['actual']) <= change['after']
    after[row['owner']] += change['delta']
assert before == after
assert sum(proposal['path_step_ceilings'].values()) == proposal['repair_delta_ceiling'] == 3
assert proposal['source_delta_ceiling'] + proposal['test_delta_ceiling'] == 3
print(json.dumps({'verdict':'PASS_ALLOCATION_ONLY', 'allocated_paths':len(rows), 'caps_before':dict(before), 'caps_after':dict(after), 'accepted_actual':dict(actual), 'accepted_source':sum(actual.values()), 'accepted_global':sum(actual.values())+14, 'step_gross_ceiling':3, 'projected_growth':1, 'donor_actual':int(by_path['frontier/cc_backend/collective_plan.py']['actual'])}, indent=2))
PY
```

Expected:114 unique rows, unchanged owner ceilings, exact before/after values,
zero-sum allowance, donor code usage within139 and at most3gross/1net change.
Actual: **PASS_ALLOCATION_ONLY**, exit0; owner caps4404/3956/1582 and accepted
actual4261/3954/1350, source9565/global9579. The fixed doc12/gitlink2 contribution
is retained from the accepted ledger. No successor source or performance result
has been measured by this command.

## Remaining checks

RED/GREEN, focused native-schema controls, key-only timing, canonical wave,
composed source/audit checks, independent review and required regressions are
pending. Native hardware, whole-CLI and clean-E2E qualification are not performed.
