# D29-E41 final regressions completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record evidence commitb5a74416 and37-file archive verification. |
| 2026-10-23 | Archive required final-source regression results and preserved limits. |

## Overview

TaskID: D29-E41-FINAL-REGRESSIONS. Required regression gates are complete on
read-only `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`,
source `e41abc073500d61f224eda4fca1123e74e40544d`, submodule
`3dac53c544e546574cb6e8078cd14053d0916113`. No implementation edit, checkout,
full CLI/GPU run, external-network operation or further delegation occurred.

## Independent suite results

| Suite | Observed result | Gate |
| --- | --- | --- |
| strict77 final04 | 77/77 identical; run/compare/audit exit0 | PASS |
| full-unit parent04 | 8015PASS/279FAIL/32ERROR/281SKIP, raw exit1 | Raw FAIL; explicit regression comparison PASS |
| same-source py312 supplement101 | 101PASS, no FAIL/ERROR/SKIP, exit0 | PASS |
| metadata196 | 196PASS, no FAIL/ERROR/SKIP, exit0 | PASS |
| catalogue53 | 53PASS, no FAIL/ERROR/SKIP, exit0 | PASS |

Counts are not summed. strict77 uses the unchanged retained713e6a8 baseline,
scenario table and comparator. Each side has451 artifact hashes and519 cache
files independently verified; definitions, dependencies, provenance and full
clean-cache capture match. No comparison differences.

Full-unit retains all267 locked failure IDs and31 locked error IDs. Only12
missing-aiohttp failures and1 missing-Torch collection error are added; each is
explicitly covered by the new same-source101PASS supplement (12 identical HTTP
IDs and16 native module cases). No missing baseline ID or PASS-to-SKIP. The prior
final03 and fresh final04 outcome IDs/statuses match exactly. All196/53 affected
IDs also pass in full-unit. Raw exit1 and every traceback remain archived.

Both source repositories are clean before/after. A final2611-file rehash finds
no difference from the during-run inventory. The source namespace preserves
original paths and mounts all existing entries read-only; generated cache and
outputs are retained only in unique scratch.

## Deliverables

All retained files are under:
`/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_regressions/`.

- Separate `test_report_2026-10-23_{strict77_final04,unit_parent04,supplement101,metadata196,catalogue53}.md`.
- `evidence/strict77/final04_receipt.json`: fresh comparison/provenance/hash result.
- `evidence/unit_final04_comparison.json`: complete ID additions/removals,
  PASS-to-SKIP checks, explicit environment tracebacks and subset membership.
- `evidence/{unit,supplement101,metadata196,catalogue53}/`: raw XML/log/execution receipts.
- `evidence/source_final_receipt.json`, source inventory and scenario preflight.
- `evidence/scripts/`: exact executed orchestration/comparison scripts.
- `progress.md` and progress checkpoints: phase completion and task limits.

Raw artifacts and generated caches remain at:
`/data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s`.
Evidence delivery commit: `b5a74416b453288e5ca6c1fdfecad65a54128f80`,55
files confined to this record directory. All37 archived raw evidence files pass
hash/size verification. Follow-up status-only documentation records completion.

## Open work and limitations

No required regression subtask is unfinished. The historical raw unit failures,
errors and skips remain unresolved by design; this delivery establishes no new
code regression after explicit environment reconciliation, not all-green units.
Root separately owns actual full offline/online main06 and overall workflow
acceptance. Native/GPU runtime, G3 and independent E2E numerical qualification
remain outside this task. Accepted source-equivalent repair57/6 was not rerun.
