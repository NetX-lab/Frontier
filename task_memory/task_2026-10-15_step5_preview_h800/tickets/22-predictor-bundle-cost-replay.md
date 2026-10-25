# 22: Complete predictor bundle and end-to-end cost replay

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-22 | Record actual06 failure, verified identity reuse and D26 new-fixture integration dependency. |
| 2026-10-21 | Measured integrationbc8ae27 passes323 combined/197 profiling parent checks; complete CLI acceptance remains active. |
| 2026-10-21 | Four public-manager bundles generated; actual-main mapping correction and final integration remain open after274 parent checks. |
| 2026-10-21 | D21: P0-first measured CPU integration and separate per-track bundle admission; retain external acceptance gates without reinstating the implementation wait. |
| 2026-10-21 | D20 moves from command preparation to approved predictor/cache/bundle implementation; native and whole-ticket acceptance prerequisites remain. |
| 2026-10-21 | D19 assigns task-local CPU command/input preparation while keeping full ticket and shared-code prerequisites blocked. |
| 2026-10-20 | Recorded that the predictor bundle still waits for the GLM milestone (decision D3). |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved complete measured-predictor integration ticket. |

**What to build:** Publish an isolated trained predictor bundle through the existing publisher, complete measured-predictor PD requests, and reconcile physical scopes through layer, stage, and forward costs.
Demonstrate coverage, host/communication ownership, and cross-model cold/warm identity without upgrading diagnostic replay into predictive acceptance.

**Blocked by:** `16` — Measured mixed-precision compute prediction; `18` — PD capacity pressure, concurrency, and cancellation; `19` — Graph and host scheduling for mixed arrivals; `20` — PD request placement and load feedback; `21` — Independent routing and communication prediction.

**External prerequisites:** Verified external09.full_frontier_path[GLM], complete 16/18/19/20/21 and the existing C01/C04 publication/identity contracts are required. Reuse the one producer/trainer/cache/publisher chain. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Record reuse sufficiency and evidence-backed changes through the existing C01/C04 owners and D-* records. Routine incremental changes use standing execution authority. Explicit approval is required only for a demonstrated material core refactor or materially unresolved semantics/acceptance; no duplicate decision authority is created.

**Status:** in-progress (D20 CPU implementation); formal bundle and whole-ticket acceptance blocked

**D31 status (2026-10-25):** Active within plan step E: predictor bundles built from the step-D profiles.

**Current acceptance:** actual06 still timed out after357/362 outputs. Fixed
identity source7e98e5c passes parent96 checks and complete source/ledger audit;
parent iteration7.983/8.116s remains RED, so no fullCLI claim. D26 fresh v2
fixture binds P6128/D16 and balanced source settings; oldfixture02 is immutable
and rejected by the new runner. Integrate D26/D27/identity, create a fresh
combined input generation and complete required regression before full request
acceptance. D29 orders workflow/GPU-opgap before native CPU-overhead work.

**Execution receipt:** bc8ae27 integrates compute83af0a7, PD9625984, transfer8215196, graph-policy7e41b80 and request-freeacc3a88 into the primary branch. Parent323 combined and197 profiling/timing checks pass. The public-manager fixture emits four bundles with34 artifacts each. The actual-main owner is completing offline/online6144/1024 requests, first-tail16/cached6128, READ completion/source release and final metrics. Synthetic costs do not constitute an admitted production bundle. Final-source77-case fidelity is running; the earlier comparison retains explained config/metadata differences.

**Current D21 scope:** integrate compute/PD/client measured CPU execution P0 H80032 first. The four hardware/scale tracks have distinct admitted profiles, runtime/native evidence and immutable bundle identities; no scaled times or cross-track PASS.32 total=P16/D16 TP8/DP2/EP16/PP1;64=P32/D32 TP8/DP4/EP32/PP1. Basic64 runtime can proceed now, but own readiness plus matching32 four clean errors<=30% precedes64 formal calibration; H20032 is independent of H800. Final four10% gates remain. D20 overrides historical CPU waiting/review-only paragraphs below; native/profile/external predictive acceptance stays open. Use `../plan.md` D21 graph/resources, not old allowances.

**Current D20 override:** the user explicitly releases CPU implementation from D3's GLM wait. `step5-measured-compute` owns the existing trainer/cache/publication boundary and joins16's compute consumer; `step5-measured-pd` owns lifecycle compatibility. Deliver executable fit/load/lookup/cost replay and cold/warm identity tests. Preparation-only restrictions in the D19 paragraph below are historical. Required native/profile coverage and formal predictive gates remain unfulfilled.

**D19 preparation:** `../d19_pipeline_preparation/` now owns read-only pipeline mapping and task-local CLI/configuration preparation alongside28. It verifies supported commands and names missing admitted inputs; it does not train on substitute data, publish a qualified bundle, bypass the GLM gate or modify shared code. The full ticket remains blocked.

**GLM gate — 2026-10-20:** Under decision D3 this ticket still waits for `09.full_frontier_path[GLM]`.

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S08.
**Shared owners:** C01, C04, and the existing producer, trainer, cache, publisher, and execution owners.

## Review disposition — 2026-10-18

NOT STARTED. Distinguish independent trained predictor bundle, validated trace-conditioned cost replay and coverage from predictive online E2E. No parallel cache/identity/publication mechanism; model/runtime/hardware/precision isolation is required.

## Acceptance

- [ ] Immutable bundle identity covers runtime/image, metadata, hardware, layer precision/exclusions, role/topology, graph policy, backend, layout, and physical-scope coverage through existing publication and training boundaries.
- [ ] Full 92-layer measured-predictor PD requests complete all primary text paths, including mixed attention/FFN, host work, communication, capacity, readiness, tail work, placement, and release; dummy timing mode is disabled.
- [ ] Validated observer records reconcile physical scope to layer to stage to forward, with duration, kernel count, gaps, and host/communication costs attributed exactly once and residual errors reported.
- [ ] Independent coverage includes prefill chunks, sparse/SWA and region/page/window boundaries, precision classes, graph edges, and legal placement; missing coverage, wrong identity, and duplicate fused scopes fail explicitly.
- [ ] Cold and warm loads agree, and A → B → A sequences spanning GLM, Step, and DS profiles match isolated loads without cross-model cache contamination.
- [ ] DS contract/profile isolation is checked without requiring all unplanned DS native campaigns; unqualified DS native predictions remain explicitly unverified and never borrow another model's PASS.
- [ ] Fixture replay, controlled workflow replay, trace-conditioned cost replay, and predictive online E2E inputs/results remain distinct; cost replay does not close the formal predictive gate.
- [ ] Bundle, coverage, and completed-request evidence record actual outcomes and limitations; local publication does not imply external upload, source integration, or resource authorization.
- [ ] At least 50 baseline-versus-candidate scenarios preserve unchanged fidelity under specification §10; preregistered wall-time/memory budgets are measured, and every excess has an explanation and disposition.

Specification references: §§4–5, 7, 9–10.
