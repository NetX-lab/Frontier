# 27: H200 qualification and minimum-matrix freeze

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | RETIRED by decision D12 (calibration scope reduction); see the retirement note below. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved required H200 qualification and scope-freeze ticket. |

**What to build:** Qualify actual H200 full-text native capacity and execution paths after H800 acceptance, then freeze the minimum legal profile and formal-case matrix before formal H200 execution.
Record measured resource constraints without promising an unqualified GPU count or topology.

**Blocked by:** `26` — H800 primary matrix and independent holdouts.

**External prerequisites:** H800 is primary. After26, perform the required H200 qualification and freeze its minimum legal matrix under standing in-scope resource authority and available capacity. H200 deferral during the H800 phase is not a final waiver or a promise of H20064. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
The current user grants ordinary in-scope H200 execution authority after the H800 stage; this does not supply actual H200 capacity or qualification. Unavailable resources leave required H200 work open rather than waived.

**Status:** RETIRED 2026-10-21 (decision D12)

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S10.
**Shared owners:** C01, C04, C06, and the existing qualification and campaign owners.

## Review disposition — 2026-10-18

NOT STARTED; sequenced after H800 acceptance. Existing two-node H200 environment diagnostics are not model qualification. Freeze a legal measured minimum matrix; do not invent H20064 qualification or change the target model.

## Acceptance

- [ ] Actual H200 hardware, runtime/image, metadata, backend, and placement identities are recorded independently of H800; source candidates and recipes are not represented as qualified runtime facts.
- [ ] Complete text-model capacity accounts for resolved mixed precision, all-rank parameters, initialization peaks, graph storage, workspaces, raw KV, summaries, and scratch without model reduction or native module stripping.
- [ ] Qualified H200 paths include prefill, subsequent chunks, decode graph replay, PD raw/summary/open-state integrity, tail work, target readiness, and source release on a demonstrated legal topology.
- [ ] Measured available resources and legality determine the minimum matrix; no exact count/topology or H20064 qualification is promised, inferred from H800, or substituted for missing evidence.
- [ ] Before formal execution, freeze exact cases, profile grid, legal topology, serving context, workload identities, sampling/windows, repeats/stability, metric applicability, p95 eligibility, and independent evaluation membership.
- [ ] Formal rules retain per-metric means <=10% and eligible p95 client TTFT/ITL <=15%, with zero-actual relative error undefined and applicable absolute budgets preregistered; freeze does not relax H800-approved accuracy thresholds.
- [ ] Native failures produce truthful qualification blockers and minimal evidence rather than a functional patch, disabled required path, or implicit reference switch; a needed reference change follows the existing conditional D-S5-REF authority.
- [ ] Closure supplies measured qualification and the frozen minimum scope, not formal accuracy or an H200 waiver; unqualified hardware/topology claims remain explicit.
- [ ] Authorized StepMind authentication, intranet access, and RJob operations use the original personal local Python RJobBackend; ordinary local preparation, CPU checks, and public documentation reads may use local tools. At >=32 GPUs auxiliary validator errors are logged, marked unverified, and notified without stopping healthy work solely for harness failure. Native failures retain normal handling, and final verdicts require valid records.

Specification references: §§2–3, 5–8, 10.

## Retirement note — 2026-10-21

The separate H200 qualification and minimum-matrix freeze is superseded by ticket `23`, which freezes the canonical case (workload, topology, hardware, context, runtime settings and metric contract) directly. No minimum-matrix study is performed.
