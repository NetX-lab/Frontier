# 16: Measured mixed-precision compute prediction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | D21: P0-first CPU delivery with separate hardware/scale profile identities; historical H200-only coverage and implementation-wait wording are superseded. |
| 2026-10-21 | D20 releases CPU implementation; measured consumption code is assigned to the compute/bundle owner while actual profile/native acceptance stays blocked. |
| 2026-10-20 | Recorded that trained-predictor integration still waits for the GLM milestone (decision D3). |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved measured-computation ticket. |

**What to build:** Train, load, and consume independently measured per-layer computation costs through the existing predictor path, completing full mixed-precision Step requests.
Demonstrate actual computation-cost composition from physical scopes to request execution, rather than an isolated table loader.

**Blocked by:** `10` — Full-path exclusive compute and host profiles; `15` — Complete mixed-model structural request path.

**External prerequisites:** Verified external09.full_frontier_path[GLM], complete 10/15 and the C02/C04 shared contracts are required. A bundle or config file alone does not close the GLM milestone. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Shared-contract sufficiency and any necessary material-change decision are recorded under the existing C/D owners; preparation may proceed independently.

**Status:** in-progress (D20 CPU implementation); qualified measured and whole-ticket acceptance blocked

**Current D21 scope:** D20 implementation continues P0 H80032 first, serving the existing four-track plan without new predictor infrastructure. Hardware/scale-compatible profiles and bundle identities are separate for H80032/64 and H20032/64; no timing scaling.32 uses P16/D16 TP8/DP2/EP16/PP1,64 uses P32/D32 TP8/DP4/EP32/PP1. Native/profile/predictive acceptance is not waived. Historical H200-only references or GLM CPU waits below are superseded by D20/D21; the external GLM gate remains an acceptance condition. See `../plan.md` for basic/formal releases and resources.

**Current D20 override:** implement the existing typed trainer/loader/predictor path now in `step5-measured-compute`, together with22's cache/bundle boundary. This supersedes the GLM implementation wait below only for code and CPU regression. Deliver actual fit/cache/lookup/cost replay, correct physical-scope/precision selection and explicit missing-input rejection. Real track-compatible coverage and native/predictive evidence remain required for acceptance.

**GLM gate — 2026-10-20:** Under decision D3 this ticket still waits for `09.full_frontier_path[GLM]`; only structural PD/DP/EP work in 17 and 20 was released early.

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S04, S08.
**Shared owners:** C02, C04, and the existing training, registry, and predictor owners.

## Review disposition — 2026-10-18

NOT STARTED; blocked on native profiles, complete 15 and the external GLM milestone. Operator registration and static precision metadata belong to early15; this ticket owns measured consumption, independent shape errors and full-request compute cost reconciliation.

## Acceptance

- [ ] Training, lookup, and complete-request execution read the same registry contracts for layer family, physical scope, precision, geometry, and semantic parallel domain.
- [ ] Measured costs cover the complete reachable attention, auxiliary, staging, projection, dense, routed, and shared FFN computation paths across prefill, subsequent chunks, and decode.
- [ ] Native FP8 block/scaling semantics and every resolved precision exclusion, including routed experts in layers 88–90, select matching measurements; excluded operators cannot silently consume generic FP8 rows.
- [ ] Norm/residual precision, casts, packed projections, selector work, gate work, and summary updates retain their actual fused scope ownership; each physical duration and kernel count contributes exactly once.
- [ ] Layer-to-stage-to-request computation totals reconcile against measured-scope expectations, with duration, kernel count, device gaps, and host work separately attributable rather than absorbed into a correction factor.
- [ ] Wrong identity, backend, role, topology, precision, shape, or fusion coverage produces an explicit coverage error; missing measured rows never fall back silently to another family or dummy timing mode.
- [ ] Independent shape checks record actual and predicted computation costs and errors under preregistered budgets; trace-conditioned results remain distinct from predictive online E2E acceptance.
- [ ] Cold and warm request execution agree in predictions and contract selection; changing a precision or profile identity cannot reuse an incompatible cached predictor.
- [ ] At least 50 baseline-versus-candidate scenarios preserve unchanged fidelity under specification §10; preregistered wall-time/memory budgets are measured, and over-budget results receive explanation and disposition.

Specification references: §§3–4, 7, 9–10.
