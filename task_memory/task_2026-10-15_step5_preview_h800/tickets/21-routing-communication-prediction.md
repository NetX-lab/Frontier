# 21: Independent routing and communication prediction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved independent routing and communication ticket. |

**What to build:** Predict complete-request routing workload and TP/EP/NIXL costs using independently measured occupancy, selector locality, and communication evidence through existing workload and communication interfaces.
Demonstrate correct cost ownership for concurrency, precision, padding, and empty-lane work without conditioning predictions on the compared native run.

**Blocked by:** `11` — Independent native routing and selector statistics; `12` — TP and EP communication measurements; `13` — NIXL and shared-network measurements; `17` — Single-request PD handoff from reservation to release.

**External prerequisites:** Verified external09.full_frontier_path[GLM], the listed native measurements, complete 17 and applicable D-ROUTING/D-PD dispositions are required. Prediction must not consume evaluated native routing or timing. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Record distinguishing native evidence and reuse sufficiency in the existing D-* record. Routine incremental changes proceed under standing execution authority; only a demonstrated material core refactor or materially unresolved semantics/acceptance requires explicit approval.

**Status:** blocked

**D31 status (2026-10-25):** Routing communication prediction is checked within plan step E.

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S07.
**Shared owners:** C04 and the existing routing-workload, collective, and PD-cost owners.

## Review disposition — 2026-10-18

NOT STARTED. Reuse C04 routing/communication interfaces. Raw measurements11–13 may advance before this integration gate; pricing waits for qualified scopes and semantic sufficiency.

## Acceptance

- [ ] Independent native statistics across seeds, content families, layers, and phases drive occupancy, padding, and selector-locality features while native routing remains unchanged.
- [ ] Complete requests consume TP collectives, EP dispatch/combine/reduction, and NIXL predictions with measured payloads, participants, placement, precision, backend, and concurrency identity.
- [ ] EP-only, NIXL-only, and shared-network cases retain independently measured cost behavior; no peak-bandwidth factor, fixed 64-GPU latency, or unsupported inherited slowdown substitutes for evidence.
- [ ] Useful versus padded token work, empty-lane synchronization, and DP dummy forwards are charged correctly without fabricated effective work or forced-uniform routing.
- [ ] Wrong topology, precision, backend, payload layout, or uncovered concurrency produces an explicit rejection; physical scopes already including communication are not charged twice.
- [ ] Compared native runs and future holdouts supply no exact expert histogram, top-k list, batch composition, transfer duration, or completion time to prediction; diagnostic replay remains separately labeled.
- [ ] Independent held-out shape and communication checks report actual/predicted values and absolute/relative errors against preregistered budgets, with zero-actual relative error undefined.
- [ ] At least 50 baseline-versus-candidate scenarios preserve unchanged fidelity under specification §10; preregistered wall-time/memory budgets are measured, with explanation and disposition for every excess.

Specification references: §§4–5, 7, 9–10.
