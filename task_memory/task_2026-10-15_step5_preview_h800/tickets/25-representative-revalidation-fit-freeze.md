# 25: Representative revalidation and fit freeze

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | RETIRED by decision D12 (calibration scope reduction); see the retirement note below. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved correction-dependent revalidation and fit-freeze ticket. |
| 2026-10-15 | Limited operator routing to the specified StepMind operations after review. |

**What to build:** Revalidate representative acceptance with fresh affected-cell evidence after actual approved corrections, then freeze predictor and statistical fits before independent final evaluation.
When diagnosis proves no correction is required, record that conclusion and verify the qualifying representative evidence before freezing.

**Blocked by:** `24`; every actual correction identified by its diagnosis and recorded here with responsible ticket, scope, source/bundle generation and verification before eligibility. No speculative correction ID is created.

**External prerequisites:** Every actual correction must be separately scoped, completed and linked to its responsible ticket and verification; only material changes need additional approval. Alternatively record evidence that no correction is needed. Diagnosis completion alone cannot authorize a fit freeze. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Changes to shared semantics retain the original applicable D-* decision authority; diagnosis completion alone does not satisfy these prerequisites.

**Status:** RETIRED 2026-10-21 (decision D12)

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S09.
**Shared owners:** C06, existing calibration owners, and the recorded owners of actual corrections.

## Review disposition — 2026-10-18

NOT STARTED. Record actual correction blockers in this ticket and their responsible existing ticket; create additional ticket files only through the requested ticket workflow. Routine evidenced bug fixes need no extra permission. Every correction and affected representative revalidation must finish before fit freeze.

## Acceptance

- [ ] Eligibility records completion of the entire diagnosis ticket and every actual correction by responsible ticket and verified generation, with approval where materially required; alternatively, it records the evidence-backed finding that no correction is required.
- [ ] Fresh affected cells use the corrected source and bundle identities with qualified native records and frozen repeat/window rules; unchanged reference reuse, where valid, has explicit identity and applicability evidence.
- [ ] Representative cells meet each mean client TTFT, TPOT, E2E, output-only throughput, and applicable internal metric threshold of <=10%, and each eligible p95 client TTFT/ITL threshold of <=15%, without cross-cell averaging.
- [ ] Undefined zero-actual relative errors, applicable preregistered absolute budgets, missing evidence, unstable repeats, and p95 eligibility receive the frozen treatment; unresolved representative failures prevent fit freeze.
- [ ] Every code-changing correction supplies its at-least-50-scenario fidelity results and measured preregistered wall-time/memory budgets; over-budget regressions have recorded explanations and dispositions before correction acceptance.
- [ ] Predictor/statistical fits, independent input membership, source/reference identities, workload rules, and coverage are frozen before final matrix and holdout evaluation; compared native outcomes never become prediction inputs.
- [ ] Prior diagnosis and correction generations remain immutable and traceable; no speculative correction ticket is created, and no failed holdout is silently reused as unseen evaluation after refitting.
- [ ] Authorized StepMind authentication, intranet access, and RJob operations use the original personal local Python RJobBackend; ordinary local preparation, CPU checks, and public documentation reads may use local tools. At >=32 GPUs auxiliary validator errors are logged, marked unverified, and notified without stopping healthy work solely for harness failure. Native failures retain normal handling, and final verdicts require valid records.

Specification references: §§7–10; approved plan §8 correction sequencing.

## Retirement note — 2026-10-21

The separate revalidation-and-fit-freeze ticket is retired with the H800 campaign. The fit freeze still has to happen, but it is folded into ticket `28`: after the canonical case is compared and any correction is applied and re-measured, the predictor/statistical fits are frozen there before the acceptance verdict is recorded.
