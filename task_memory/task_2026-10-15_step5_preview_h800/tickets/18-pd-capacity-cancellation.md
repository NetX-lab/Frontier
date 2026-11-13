# 18: PD capacity pressure, concurrency, and cancellation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved concurrent PD resource-conservation ticket. |

**What to build:** Extend the completed single-request handoff to concurrent requests, capacity exhaustion, cancellation, duplicate notifications, and safe page reuse.
Demonstrate complete request outcomes and resource conservation through the existing allocator and transfer owner.

**Blocked by:** `17` — Single-request PD handoff from reservation to release.

**External prerequisites:** Verified external09.full_frontier_path[GLM], complete 17 and applicable C03/D-PD semantics are required. Concurrency cannot precede a complete single-request lifecycle. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Record distinguishing native evidence and reuse sufficiency in the existing D-* record. Routine incremental changes proceed under standing execution authority; only a demonstrated material core refactor or materially unresolved semantics/acceptance requires explicit approval.

**Status:** blocked

**D31 status (2026-10-25):** Still blocked. Reopened only if the H200 runs show KV-capacity pressure or cancellation.

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S06.
**Shared owners:** C03 and the existing allocator, transfer, and request-lifecycle owners.

## Review disposition — 2026-10-18

NOT STARTED. Begin after17 and share C03 ownership. Keep source/target byte and request conservation under capacity/cancellation, including negative cases; do not add unreachable retry or duplicate-notification machinery.

## Acceptance

- [ ] Concurrent requests complete or cancel according to the approved lifecycle while retaining one source pending owner and one request-owning DP lane per live request.
- [ ] Target capacity exhaustion blocks the required reservation and READ without oversubscription; released capacity permits subsequent eligible progress with recorded expected and actual outcomes.
- [ ] Conservation checks reconcile raw KV, SWA, summary, open-region, counter, and scratch ownership across both endpoints before handoff, during pressure, and after completion or cancellation.
- [ ] Reachable cancellation points before reservation, during transfer, and around readiness release exactly the resources allowed by native-backed semantics, without leaking or prematurely recycling pages.
- [ ] Duplicate completion or release notifications cannot double-admit a request, double-free storage, or advance token/state counters twice.
- [ ] Reused pages and reset open-region state cannot expose a previous request's ownership or summary state; differing source and target page identities remain correctly associated.
- [ ] Controlled pressure cases report queueing, admission, completion/cancellation, and final balances; successful exit alone does not pass conservation checks or establish measured transfer accuracy.
- [ ] At least 50 baseline-versus-candidate scenarios preserve unchanged fidelity under specification §10; preregistered wall-time/memory budgets are measured, and every excess has an explanation and disposition.

Specification references: §§5, 9–10.
