# D29 collective artifact parsing plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record conditional single-variable proof,scoped regression and separate source/records handoff. |

Dependency: immutable isolation→scratch exact-text baseline/control→decision→
actual-seam RED→minimal source repair→{regression/whole-file checks,original
real-point production loop}→budget/patch/source commit→records→parent review.
Only the affected tests and original loop are independent after implementation;
CORE group/loader work and full-run decisions remain parent-owned.

1. Isolation: create named worktree at24f970cb,read authority and current source.
   Detached local submodule3dac53c only;no external fetch,gitlink or peer change.
2. Control: new exact_text_loop.py copy,group reuse fixed in both arms,backend
   module-local json/Path instrumentation only. Preserve actual q6128+idle1,
   layer3/EP0,paired models,real Cluster/backend,all per-plan checks. Compare every
   CollectivePlanCost field plus compute proofs/times/plan keys,not duration alone.
   Record reads/decodes,time,latest pair residency and RSS. Proceed only on benefit.
3. Regression: actual writer/constructor;count only artifact decodes through module
   seam. Two equal calls must read twice/decode once. Existing rewrite case stays.
   Same-size valid update/restored mtime must change cost. Invalid JSON retried
   must throw;restoring last good text must work without reparse. Read-error object
   identity preserved. Different plan/count key must still fail on cached artifact.
4. Implementation: two instance fields and local text equality/parse update only.
   No helper/public accessor/numerical cache-key change;all downstream code intact.
5. Acceptance:46 backend cases=45 old+1 new;four affected whole files128 cases.
   Original loop rerun with observation shim but reuse arm OFF must prove source
   eliminates19 redundant decodes and preserves20 reads/full payload. No further
   broad tests after required PASS without a new change/failure.
6. Accounting: source8/15,test35/35,total43/50 churn;measure actual retained delta
   from1aee0af per owned path,not sum raw churn into owner baseline. Source<=93,
   test<=233,owner<=3956. Generate source.patch,commit source separately,then
   English records with exact commands/limits and parent-owned gates.
