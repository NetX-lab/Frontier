# D29 plan-key reuse completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive the verified two-path implementation for primary review. |

Task `D29-PLAN-KEY-REUSE`; session `ses_e963098ecffebqCOtdysKGY6Xe`.
Branch `fix/d29-plan-key`, isolated from protected snapshot. Verified source/test commit `c4e06a61`; records are separate.

## Deliverables

- `frontier/cc_backend/collective_plan.py`: retain only the successful full canonical key string on each existing immutable plan. Public dictionaries and errors remain independent.
- `tests/unit/test_step5_moe_pricing.py`: extend the existing identity/isolation regression by11 lines. No new suite or helper.
- `task_memory/task_2026-10-22_step5_position_routing/plan_key_reuse/test_report_2026-10-24_plan_key_reuse.md`: observed validation and fidelity limits.
- `task_memory/task_2026-10-22_step5_position_routing/plan_key_reuse/evidence/`: full implementation diff, changed-file list, source/script/result hashes, exact commands, raw test/wave receipts, profiles and exact candidate output bytes.

## Observed results

Public RED1FAIL -> GREEN1PASS. Affected candidate and protected baseline each124PASS/1SKIP (torch absent).
All208 event/cost/proof/key/order values, full JSONL/ledger,64 wrong-key negatives,3902-byte error prefix and snapshot isolation PASS.
Binding+waves read once; cold/warm reads0.192 live key calls remain; actual warm encodings192->0. Numerical288, fresh proofs144, collective64 and point16 unchanged.
Warm pricing median0.161006936->0.119742442s; separate emission0.051601212->0.038155627s.
Source5gross/+3, test11gross/+11; total16gross/+14. D29 cumulative plan139/140 and test311/311; donor counts93/14/236 unchanged. Original source/input hashes and3dac pin preserved during verification.

## Open work and limits

Primary Standards/Spec reviews, parent verification, integration, final114-path accounting and broader workflow gates remain pending.
This bounded CPU-fixture check uses a reconstructed cached manager/predictor with real owner/backend wiring; it is not full Simulator/full CLI/native qualification. No diagnostic key/byte adapter, numerical/proof cache, conversion/protocol change or further delegation. H3 stays UNQUALIFIED/unallocated.
