# EP Lane Ledger Component Delivery

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Archive a partial component delivery and the exact caller allocation blocker. |

## Overview

Task `D29-EP-LANE-LEDGER-IMPLEMENTATION`, session
`ses_e91675a86ffegKCQdHRPQFOypp`, branch `fix/d29-ledger-stream`, worktree
`/data/ycfeng/Frontier/.worktrees/step5-d29-ledger-stream`.

Code commit `6c2f179c9117256bb7086441b0c11a1459f48674` delivers a real
MetricsStore write-through/count/close sink, incremental actual-main EP validation
and an ordinary performance file-position digest. It preserves complete sorted
successful JSONL output and every semantic row check. The producer remains
byte-identical. This is a partial component delivery, not complete acceptance.

## Deliverables

- `frontier/metrics/metrics_store.py`: step33gross, cumulative33.
- `tests/integration/run_step5_measured_main.py`: step15gross, cumulative56.
- `tests/integration/test_step5_measured_main.py`: step62gross, cumulative320.
- `tests/performance/step5_actual06_iteration.py`: step10gross, cumulative10.
- `task_memory/task_2026-10-22_step5_position_routing/ledger_memory/test_report_2026-10-25_implementation_controls.md`.
- `task_memory/task_2026-10-22_step5_position_routing/ledger_memory/evidence/implementation_2026-10-25/`.

Total step120/135gross. All current local path caps pass. Local cumulative
increment112 projects CORE4373/source9678/global9692; independent composed
acceptance remains the primary's responsibility.

## Observed validation

Real retention and actual validator materialization controls were RED before
implementation. Final controls pass48/3SKIP and existing EP suites pass30/30.
The persistent test proves32 ordered successful rows, cumulative count, released
prior row objects, zero-row absence and no final truncation. All eight negative
and precedence cases pass. Accepted legacy writer bytes match3,528B/2rows and
the validator semantic loop AST is unchanged.

Untouched focused baseline passes87/6SKIP. Candidate whole-file integration remains
raw62PASS/4FAIL/6SKIP; the four existing cluster constructors cannot resolve an
unchanged collective-sim checkout. Exact status comparison retains all93 old IDs,
all6 skips and9 newPASS IDs; its fourPASS->FAIL setup changes remain rawFAIL.
Auxiliary launch/report/accounting failures are retained, not relabeled PASS.

## Open work

The existing performance save_state serializes MetricsStore and cannot pickle the
new open writer. An unapplied four-line existing-save/load-seam proposal costs
14gross in that path. Primary must allocate the exact2-gross transfer before it is
implemented: metrics cap38→36,performance cap12→14,total135 unchanged. No public
API/proxy or unallocated production state behavior has been added.

Resolve checkout readiness and rerun affected constructor tests, then full focused
integration. Independent clean composed audit, Standards/Spec review, final-source
freeze/full regressions and integration remain pending. No fullCLI, native, G3,
E2E, whole-runtime or full-workload RSS/storage acceptance follows from these
synthetic controls. No GPU/worker/StepMind/auth/mount/cloud/publishing action or
new required-shape invocation was performed.
