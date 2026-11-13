# D29 Collective artifact byte comparison

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record allocated repair and its acceptance boundary. |

## [Original Request]
TaskID D29-COLLECTIVE-ARTIFACT-BYTES; session ses_e9669de64ffeH614F6iNYIqgeb.
Implement the allocated repair in sole-writer worktree
`/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes`, branch
`fix/d29-artifact-bytes`, base `2cfbb34474c1d4d31f6daf8f9c626d03ad972f58`.

Own only the backend, workload module and existing backend test listed in
`../repairs/collective_artifact_bytes_scope_amendment.json`. Step gross limits
are 12/14/10, total 36 (source 26/test 10). Retained limits against `1aee0af`
are 91/323/235; supplied baseline counts are 84/197/233. The approved two-line
PRICING transfer is already recorded; owner/global caps remain unchanged.

Both existing artifact caches compare exact current raw bytes. Read on every
query/prediction; use `json.loads(raw.decode())` only after a miss. Preserve
previous successful state after read/decode/parse errors and every placement,
source, qualification, context, buffer/count, numerical and native-key check.
No shared/stat/wave/load-time cache, API, flag, helper, omitted read/check or
numerical/proof cache. Rename representation variables accurately.

Adapt only the existing backend test read hook with its byte-to-text marker;
preserve all assertions, including two reads/one parse, same-size/restored-mtime
rewrite, repeated invalid JSON and exact original read-error identity. An old
hook failing after the source change is instrumentation evidence only.

Reuse the actual complete layer60 setup/probe from
`/data/ycfeng/tmp/d29_cost_fields_parent01` with exact original/derived hashes;
only source/cwd assertions and output destinations change. Instrument both
read methods; baseline/candidate full 208-event costs/proofs/keys must match,
with 80 current reads per 16-EP wave. Run existing backend/binding/runtime/replay
and negative/replacement seams without adding persistent tests.

All writes go to fresh scratch before CLI config parsing. Preserve original
fixture03/cache files. No full CLI/trainer/CSV scan/GPU/network/dependency work.
Probes <=60 seconds/30 GiB; pytest <=240 seconds. Local shared submodule clone
from the primary/cost-fields tree and detached `3dac53c` is allowed; no peer writes
or delegation. Commit verified code first, then these compact records. Root owns
independent review, final union accounting/regressions and full workflow.
