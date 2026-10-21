# D23-MRV2-FORWARD Progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record initial implementation and focused green check. |
| 2026-10-22 | Record real lifecycle/cache validation and broadened legacy baseline. |
| 2026-10-22 | Complete CPU verification and code commit; archive handoff contract. |

- completed: new worktree from 5c2319cd; primary approval/source inspected.
- completed: pre-prediction hold in existing stage scheduler; generic readiness
  on existing stage context; same-time existing events resume once.
- completed: one graph policy table and capture resolver; FULL_AND_PIECEWISE,
  native min-mode and local NONE/shared graph widths. No timing constants.
- completed: metadata-bearing MRv2 dummy uses shared validated descriptor and
  kind identity; actual producer features match runtime dummy.
- completed: real four-layer P/D event lifecycle (stage sync, EP owner promotion,
  restore, terminal release); dummy has no serving IDs or layer credit.
- completed: empty lane wake through real V1 engine -> dummy arrival -> stage
  event; duplicate schedule/resume and group reset; dispatched shape survives
  request preemption. FULL prompt extension changes physical phase only.
- completed: admitted real/dummy batch -> actual build_row/rows_to_dataframe ->
  shared fit/cold/warm cache/bundle/root+count exact lookup. Bidirectional missing
  kind coverage rejects. FULL/PIECEWISE/NONE exercised.
- completed: original 5c2319cd stage event substituted in memory: 12 first-call
  regressions RED, missing required descriptor before prediction.
- completed: final focused 30 PASS, affected stage 176 PASS, measured 268 PASS,
  producer 467 PASS and DP-runtime/EP/placement 55 PASS.
- completed: code commit 3cad8dbe398304d798a323389f9e3d18500a42de.
- completed: test report and parent fresh-input contract in summary.md.

First focused run: test harness lacked its workload delegate (6 failures);
second exposed missing runtime MRv2 provenance before feature validation
(6 failures). Corrected the delegate and used shared MRV2_DUMMY_ROW_IDENTITY
in runtime feature construction. Third run: 84 passed in 0.23s.
Commands/logs: `/data/ycfeng/tmp/d23_forward_first{,02,03}.log`.

Shared manager/public fixtures originally supplied no P CUDA_EVENT attention
despite FULL/piecewise policy (7 failures). Corrected required source fixtures
with actual producer MRv2 dummy kinds; did not suppress file/source validation.
One fixture write initially reused attention.csv and replaced kernel-only rows;
isolated event source directories fixed that fixture defect. Updated contracts:
46 passed in 30.98s, `/data/ycfeng/tmp/d23_forward_updated_contracts02.log`.

Final producer: 467 passed. DP-runtime/EP/placement: 55 passed in 124.09s.
Broadened stage/selector: 204 passed, 7 inherited failures. Exact same seven
cases/error messages reproduced on immutable 5c2319cd inputs worktree, no edits:
3 spec fixtures lack num_processed_tokens, 4 manager fixtures lack attention
geometry. Original and candidate XML paths:
`/data/ycfeng/tmp/d23_forward_inherited_selector_baseline.xml`,
`/data/ycfeng/tmp/d23_forward_final_stage.xml`. Comparison printed
`baseline failures 7 candidate failures 7 same cases/errors True`.

Checkpoint after first green: mainline unchanged, D1-D7 CLEAR; S1-S6 no
defect found. Remaining frontier: real completion/wakeup and fit/cache/cost,
then affected regression and parent handoff. Parent-owned actual main/native
work is not pulled into this scope.

Checkpoint after executable lifecycle/cache milestone: plan items 1-2 PASS,
item 3 TODO. D1-D7 CLEAR, S1-S6 no defect. Only remaining frontier is final
affected verification, scoped commit and fresh input handoff. Inherited selector
fixture repairs are outside this scope and remain deferred to parent.

Final checkpoint: plan items 1-3 PASS, parent full CLI/strict77/native work
explicitly outside this handoff. D1-D7 CLEAR, S1-S6 no defect. All approved core
behavior has executable CPU evidence; no inherited selector test was patched.
Read requirements.md before closure. No scheduler approximation, second queue,
new event or synthetic timing constant was introduced. Input worktree status
remains clean. Final task record commit follows the verified code commit.
