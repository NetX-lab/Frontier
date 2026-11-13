# 24: H800 representative comparisons and diagnosis

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | RETIRED by decision D12 (calibration scope reduction); see the retirement note below. |
| 2026-10-20 | Recorded the D5 native sizes measured on H800. |
| 2026-10-20 | Recorded decision D5: the representative cells give the final size of the first-token and tail terms. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved representative comparison and diagnosis ticket. |

**What to build:** Compare representative H800 cells, deliver reproducible per-cell verdicts and error budgets, and identify evidence-backed, separately scoped correction needs.
The deliverable is complete comparison and diagnosis evidence, including accurate FAIL verdicts; this ticket does not implement an unknown or unbounded repair backlog.

**Blocked by:** `22` — Complete predictor bundle and end-to-end cost replay; `23` — H800 pilot and formal campaign freeze.

**External prerequisites:** Complete 22/23, qualified reference and frozen placement/metric rules are required. Compare workflow boundaries and physical costs before choosing measured corrections. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Shared semantic changes remain under the original applicable D-* records. Routine evidenced corrections use standing execution authority; a necessary material refactor or materially unresolved semantics/acceptance still requires explicit approval. A diagnosis does not authorize reference replacement.

**Status:** RETIRED 2026-10-21 (decision D12)

**Decision D5 — 2026-10-20:** The error budget reports the T1 (first-token endpoint) and T2 (recomputed tail) terms separately for every representative cell. If either term was recorded as small earlier but exceeds 1 percentage point of the native mean client TTFT or mean E2E in a representative cell, or changes a verdict, it becomes a correction under ticket 17 or 14 ([requirements.md](../requirements.md) decision D5 and [spec §8](../spec.md)).

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S09.
**Shared owners:** C06 and the existing calibration, normalizer, and gate owners.

## Review disposition — 2026-10-18

NOT STARTED. Diagnose first workflow divergence and its operator/communication/memory/CPU contribution. Valid FAIL verdicts may close diagnosis but cannot close accuracy,25 fit freeze or26 acceptance.

## Acceptance

- [ ] Frozen representative W1/W3/W4/W6 cells have reproducible clean-native and Frontier records, required repeats, source/bundle/workload identities, and actual completion/rank accounting.
- [ ] Predictive online E2E receives only preregistered inputs and independent predictors/statistics; trace-conditioned cost replay supports diagnosis but is not the predictive result.
- [ ] Every cell reports actual/predicted values and absolute/relative errors for mean client TTFT, TPOT, E2E, output-only throughput, and applicable internal metrics against <=10%, plus eligible p95 client TTFT/ITL against <=15%.
- [ ] Undefined relative error at actual zero, frozen absolute budgets where applicable, insufficient evidence, unstable reference repeats, and ineligible p95 remain explicit rather than being coerced into PASS.
- [ ] Error budgets separate semantic alignment, workflow, operator, communication, KV/tail, and approved host terms using correlated evidence; no arbitrary factor or tolerance relaxation absorbs missing work.
- [ ] Each suspected correction records the affected behavior, current owner, distinguishing evidence, minimum scope and expected outcome. Ordinary corrections use standing authority; approval is required only for a material core refactor or materially unresolved semantics/acceptance. Unresolved causes remain identified as unresolved.
- [ ] Comparisons and diagnosis close this ticket even when accuracy verdicts are FAIL; absent or invalid records remain incomplete evidence, and diagnosis completion cannot authorize the fit-freeze stage.
- [ ] No correction implementation is hidden in this diagnosis ticket. Each actual correction is separately scoped under its responsible existing ticket and linked by ticket ID, source/bundle generation and verification as a blocker of 25; additional ticket files follow the requested ticket workflow, with no speculative IDs.
- [ ] Authorized StepMind authentication, intranet access, and RJob operations use the original personal local Python RJobBackend; ordinary local preparation, CPU checks, and public documentation reads may use local tools. At >=32 GPUs auxiliary validator errors are logged, marked unverified, and notified without stopping healthy work solely for harness failure. Native failures retain normal handling, and final verdicts require valid records.

Specification references: §§7–10; approved plan §8 correction sequencing.

## Retirement note — 2026-10-21

The H800 representative-cell diagnosis is retired with the H800 campaign. Its purpose — locate discrepancies in workflow, sparse attention, MoE, graph/host overlap and PD transfer, and decide which corrections are needed — transfers to ticket `28`, which performs that diagnosis inside the single canonical 64-H200 case. The D5 error-budget rule (T1/T2 reported separately, a term above 1 percentage point becoming a correction under ticket 17 or 14) still applies there.
