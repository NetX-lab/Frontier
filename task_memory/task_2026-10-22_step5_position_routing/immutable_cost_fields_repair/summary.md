# D29 immutable cost fields completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Parent accepts and merges the component after independent reviews and committed-source verification. |
| 2026-10-23 | Archive the scoped implementation, full differential and exact retained accounting. |

TaskID: `D29-IMMUTABLE-COST-FIELDS`.
Authority/baseline48d34106; code commit
`7540414341d5b81a86b18b9562d1506c0ba74285`.
Branch/worktree: feat/step5-d29-cost-fields,
`/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields`.

## Delivery

- Only `frontier/metrics/op_trace_utils.py` and
  `frontier/metrics/ep_wave_metrics.py`: fresh declared-field cost dictionaries
  replace the two existing asdict conversions. Exact+2/-2 each,8gross total.
- Retained1aee0af comparison stays45/22, delta0. Root-supplied owner/source/global
  totals remain unchanged; final-union accounting is parent-owned.
- Current trace+prepared baseline/candidate208actual layer60 event/proof bytes,
  full JSONL and2,309,545-byte ledger are identical. Full body SHA256
  `18d320438ee1fb5f077b12aaa83da8055b47d8cf256d88aedcca159a5d7b819c`.
- Snapshot/mapping independence and64wrong-key negatives pass. Late-JSON buffer
  restoration/3902-byte prefix unchanged. ExactAST/whitespace checks pass.
- Same affected suite98PASS/1unchanged baseline mapper failure, all99statuses and
  failure message identical. Raw suite still FAIL; comparison PASS.
- Emission-only median58.067739->51.181030ms (-11.859785%), asdict336->208 and
  deepcopy26,336->21,472. Plan/key counts remain192/64. No wholeCLI attribution.

## Artifacts and limits

All new durable records are in this directory: requirements, plan, progress,
test report, summary, `code.diff`, `receipt.json` and `evidence/`.
Full traces/ledger/log/XML and original/derived-hashed probes remain at
`/data/ycfeng/tmp/d29_immutable_cost_fields_repair/`.
No persistent new tests, helper/API/cache, component borrowing, joined write,
duplicate key encoder, suppressed guard, unrelated source edit or gitlink change.
Owner performed no fullCLI/GPU/platform/network/dependency/new-agent activity.
Parent independent Standards/Spec reviews both returned No findings. Parent
reproduced208full event/proof bytes, the full ledger and64 wrong-key negatives,
including snapshot isolation and the3902-byte error prefix on committed source.
Initial131PASS/14SKIP identified13 missingHTSIM cases; after a local build from
unchanged3dac sources, the affected file passed54/54. ID-based reconciliation
is144PASS/1existingTorchSKIP across145 unique tests; the owner98/1 comparison
is separate. `parent_acceptance.json` records all statuses and hashes.
Code is merged on feat/step5-preview; parent recovered the records after the
owner's API quota error. Final freeze/regressions and wholeCLI1024/2400s and
native qualification remain open.
