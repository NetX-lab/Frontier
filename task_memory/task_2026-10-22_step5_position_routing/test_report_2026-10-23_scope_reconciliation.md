# Final D29 design scope reconciliation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Verify the complete reviewed producer delta and parent union; no implementation or numerical qualification. |

Python3.10.6, parent feat/step5-preview. Parent parses the actual Markdown rows
under pricing6.Common/6.Model/6.Closure, preserving the previously checked
CORE/SOURCE/NATIVE rows in
`/data/ycfeng/tmp/d29_complete_scope_union_candidate01.json`.

Observed row counts and additions/deletions:

| Part | Rows | Added | Deleted | Gross |
| --- | ---: | ---: | ---: | ---: |
| Common |42|2102|237|2339|
| Model increment |15|1416|57|1473|
| Producer closure |14|708|29|737|
| Complete pricing |71 hunks /60 paths|4226|323|4549|

Other owners: CORE33/2966, SOURCE16/879, NATIVE19/1582. Before reconciliation:
139 allowance rows,107 unique paths,9976 gross. The sole fully redundant
pricing main.py allowance is30+4=34; remove it once. Retained **107 paths/9942**.
Cross-owner18 paths total2976; same-owner repeated pricing8 paths total1007;
other81 paths total5959; all three sum9942. Five newly global paths are listed
in `scope.json`; inputs.py was already CORE-owned. Sole writers: CORE16 shared
paths and pricing2. Mixed allowances remain conservative, not guessed savings.

Parent asserts the parsed figures and path identities, then saves `scope.json`.
Recheck using the same interpreter from the repository root:

```python
import json
from pathlib import Path
r = json.loads(Path('task_memory/task_2026-10-22_step5_position_routing/scope.json').read_text())
assert len({x['path'] for x in r['rows']}) == 107
assert sum(x['ceiling'] for x in r['rows']) == 9942
assert 9942 + 12 + 2 <= 10000
```

The last two quantities separately cover the existing size note and submodule
gitlink. The current pricing SHA256 is
`d5f180e6ecf6d53fe4c390ab14a46f8cffdb58168d79bc03c16786833e701723`.
Budget/ownership arithmetic PASS. All design reviews are closed; model and
scope approval remain pending. No production tests, GPU collection, installed
source verification, fitted parameter qualification or E2E gate was run here.
