# D29 model-group reuse progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record real RED, minimal local repair and affected-suite validation. |
| 2026-10-23 | Complete committed-source pricing/profile and final per-path accounting. |

Task ID: **D29-MODEL-GROUP-REUSE**.

## completed — isolation/lifetime

Created `/data/ycfeng/Frontier/.worktrees/step5-d29-group-reuse`, branch
feat/step5-d29-group-reuse, from e41abc073500d61f224eda4fca1123e74e40544d.
Created a separate detached submodule worktree at unchanged3dac53c; no gitlink
change. Frozen integrated05 and primary remain untouched. Read primary allocation,
applicable AGENTS and construction/lifetime code; evidence is in plan.md.

## completed — actual regression RED

Added the regression before the production edit. Existing real producer/trainer
fixture supplies paired bindings/physical rows; actual query and `_step5_ffn_model`
run repeatedly over layers3/88. Gate duration/count exact proofs, identical
queries/plans/model identity, foreign native source and family negatives execute
before the count assertion. Baseline fails:24 resolutions for the first
predictor, expected1. Log `/data/ycfeng/tmp/d29_group_reuse_red01.log`.

## completed — implementation and GREEN

`MoeRoutingWorkload._step5_routed_layer_group` lazily resolves the actual immutable
tuple once per predictor, then selects the requested layer on every call. Both
existing query and FFN-binding consumers use it. No selected estimator, per-point
admission, native source/context or proof result is cached.

Regression GREEN1PASS14.64s. Extended the same regression with a foreign actual
source point and missing paired-count keys; they remain rejected after warm
group reuse. Affected six-file suite81PASS/1SKIP166.49s, no failures.
Logs `/data/ycfeng/tmp/d29_group_reuse_{green01,affected01}.log`.

Final e41-relative source17/45,test54/60,total71/105 gross. Retained actuals:
workload143/175,predictor111/128,test254/260. AST/whitespace/budget checks and
committed-candidate actual pricing/profile are PASS. CORE retained3932/4404,
global9175/10000 based on the existing final handoff receipt's owner ledger.
The retained change is59 gross rather than71 step gross because removed lines
overlap historical added hunks; both measurements are preserved in budget.json.

## completed — real pinned pricing and final handoff

Verified source commit:3763e5b87a845eac186cfd481f0b56cffb2dd045. Original diagnostic
script was copied to `/data/ycfeng/tmp/d29_group_reuse_runtime_loop.py`; only the
source directory, enforced HEAD and output revision field changed for the initial
run. No setup, selected model/point or production calls changed. Exact diff is
probe_source_pin.diff. Without reuse flag, actual pricing cold0.945020s/warm
0.369055s,88 descriptions total, GREEN. Repeated process0.996564/0.357902s, same
hash. A later observation-only explicit per-iteration counter yielded
0.919454/0.369015s with88/0 descriptions. Complete probe diff retained separately.

All cost/proof/full-plan-key payloads match the baseline SHA256d2c141f088b52063….
Profile: one group resolution1.210s,88 descriptions, across two calls; warm
pricing0.374071s. Residual backend8calls0.548s cumulative and JSON decode1654calls
0.594s are diagnostic only; neither is changed. The initial full resolution cost
is deliberately retained rather than prewarmed away. The probe's pre-existing
unprofiled group_resolutions field is0 because no pstats runs; use the explicit
88/0 description counts and profiled resolver count1, not that absent counter.

Code diff contains exactly three allocated paths,65 additions/6 deletions.
No other production/test/peer/primary/frozen edit; submodule remains clean3dac.
Primary owns independent review, final freeze, full2400s workloads/strict77/unit
and native qualification. No long CLI, GPU/platform/network or extra agent run.
