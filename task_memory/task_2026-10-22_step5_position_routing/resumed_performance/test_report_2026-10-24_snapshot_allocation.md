# Snapshot allocation and intermediate receipt checks

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record parent byte-evidence checks and the zero-sum114-path allocation validation. |

## Environment and limits

Local CPU; `/data/ycfeng/envs/frontier-py310/bin/python` (CPython3.10.6),
cwd `/data/ycfeng/Frontier/.worktrees/step5-preview`. No pytest, workload,
GPU, network, source change, or native rebuild was performed by these checks.
The byte source remains delivered, not independently accepted or merged.

## Byte receipt check

The parent read the ten-path records-only commit56529fcd and verified that all
of its paths belong to `artifact_bytes_repair/`. Records were cherry-picked as
e4cb7a59. The 24 entries under `receipt.json:evidence` were streamed in1MiB
blocks through SHA256; their recorded byte sizes and hashes matched. Three
source file hashes matched both owner disk files and `git show 2f2d9339:path`.

Both original JUnit files contain78 cases, no failure/error/skip nodes, and the
same test IDs after the one documented suffix rename `current_text` to
`current_bytes`. This checks the owner's observed78PASS on each side; the
parent did not rerun or add the counts. Raw JUnit paths and all24 hashes remain
in `artifact_bytes_repair/receipt.json` relative to the task root.

Expected and observed:24/24 raw hashes,3/3 source hashes,78/78 IDs,10/10 records
paths within scope: **PASS**. Paired warm medians1.037333/0.325204s and full
208-event/ledger equality remain owner observations bound by those hashes.

## Allocation check

Exact executed command:

```bash
timeout 60s /data/ycfeng/envs/frontier-py310/bin/python - <<'PY'
import json,collections
from pathlib import Path
r=Path('task_memory/task_2026-10-22_step5_position_routing/repairs')
b=json.loads((r/'evidence/performance_scope/d29_final_budget_audit_receipt06.json').read_text())
rows={x['path']:{'owner':x['owner'],'cap':x['ceiling_after']} for x in b['rows']}
for name in ('trace_serialization','prepared_point','immutable_cost_fields','collective_artifact_bytes','run_input_snapshot'):
 d=json.loads((r/f'{name}_scope_amendment.json').read_text());delta=collections.Counter()
 for x in d.get('new_paths',[]):
  assert x['path'] not in rows;rows[x['path']]={'owner':x['owner'],'cap':x['ceiling']};delta[x['owner']]+=x['ceiling']
 for x in d['path_ceiling_adjustments']:
  y=rows[x['path']];assert y['cap']==x['before'],(name,x,y)
  assert x['before']+x['delta']==x['after'];y['cap']=x['after'];delta[y['owner']]+=x['delta']
 assert all(v==0 for v in delta.values()),(name,delta)
 assert len(rows)==d.get('effective_unique_paths',len(rows))
owner=collections.Counter();counts=collections.Counter()
for y in rows.values():owner[y['owner']]+=y['cap'];counts[y['owner']]+=1
assert dict(owner)=={'core_source':4404,'native':1582,'pricing':3956}
assert len(rows)==114 and sum(owner.values())==9942
assert sum(d['path_step_ceilings'].values())==340
assert sum(v for p,v in d['path_step_ceilings'].items() if not p.startswith('tests/'))==120
print(json.dumps({'status':'PASS','effective_paths':len(rows),'owner_caps':dict(owner),'owner_path_counts':dict(counts),'step_cap':340,'source_step':120,'test_step':220,'all_overlay_owner_deltas':0},indent=2))
PY
```

Observed:114 paths; CORE49/4404, native19/1582, PRICING46/3956; sum9942.
Every overlay's before/delta/after chain and owner-level zero sum passed.
Step ceilings sum340 (source120/tests220). **PASS**.

`repairs/evidence/run_input_snapshot/budget_before.json` is a planning
calculation on delivered2f2d9339:9377 source/9391 global. It inherits unchanged
non-superproject measurements from the accepted79 audit. This check establishes
allocation arithmetic, not final source consumption, snapshot correctness,
fresh regressions, complete workflow performance, or native qualification.

The manager currently has703 physical lines; the new12-line allowance does not
create a new over2000-line exception. Existing module-size exceptions remain
historical source-bound records and will be checked in final union verification.
