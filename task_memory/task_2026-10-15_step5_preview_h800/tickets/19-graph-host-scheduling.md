# 19: Graph and host scheduling for mixed arrivals

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-23 | D30 source/CPU slice integrated4ec5b23; full ticket and fresh native counters remain open. |
| 2026-10-22 | Add approved D30 finish-sync source/observation slice independent of full mixed-arrival acceptance. |
| 2026-10-22 | Apply D29 workflow/GPU-first order and defer CPU-gap measurement. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved graph and host scheduling ticket. |

**What to build:** Complete controlled mixed-arrival PD scenarios with the native graph/eager dispatch policy, host planning reuse, held outputs, DP dummy forwards, and first-decode tail work represented by existing scheduling owners.
Expose expected versus actual request progress and cost ownership for long-prefill/short-arrival and idle-DP-lane cases.

**Blocked by:** `07` — Single-role observer correlation and perturbation; `17` — Single-request PD handoff from reservation to release.

**External prerequisites:** Verified external09.full_frontier_path[GLM], complete 07/17 and a D-ASYNC disposition are required. Preserve observed native asynchronous boundaries; do not copy native threads/IPC into Frontier. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Record distinguishing native evidence and reuse sufficiency in the existing D-* record. Routine incremental changes proceed under standing execution authority; only a demonstrated material core refactor or materially unresolved semantics/acceptance requires explicit approval.

**Status:** full mixed-arrival acceptance blocked;19.finish_sync source/CPU slice integrated at4ec5b23, fresh native observation pending.

**D31 status (2026-10-25):** The finish_sync slice is integrated. Mixed-arrival acceptance moves into plan step E.

**D30 immediate slice:** current pinned Step synchronizes on wave step1 and
each16 steps; Frontier legacy policy uses32 without the first sync. YC approved
Step-scoped configuration/scheduling alignment, focused boundaries and native
counter observation, preserving other-model behavior and legitimate dummies.
This source correction can proceed from its pinned-source/parsed-args evidence
before full07/17 and mixed-arrival acceptance; those whole-ticket prerequisites
remain. Owner `step5-finish-sync`; records in
`../../task_2026-10-22_step5_finish_sync/`. Keep native calls/returns/exceptions
and observer error isolation; no new GPU synchronization or CPU-gap campaign.

Parent acceptance: eight approved files/603 lines, independent review No findings,
420 affected checks,65 canonical checks and77 strict cases/902 artifact hashes
PASS. After the assertion-only rerun, unit failure/error IDs equal the baseline.
Merged source/tests match accepted3d6744f. These results close the code slice,
not the unchecked full-ticket criteria or live native counter qualification.

**D29 order:** workflow and GPU/operator-gap alignment precede CPU overhead/gap
measurement or correction. Do not start broad CPU-gap analysis, sampling or
suites now. Existing minimal simulator Python execution enablement is not native
CPU-overhead evidence; graph/dummy/lifecycle correctness keeps its own scope.

The October18 review was documentation-only. Current execution follows the later
explicit decisions in `../requirements.md`; whole-ticket evidence prerequisites
remain, while the approved independent D30 code slice is complete.

**Source workstreams:** S06.
**Shared owners:** C05 and the existing scheduler, held-output, stage-timing, and event owners.

## Review disposition — 2026-10-18

NOT STARTED. C05 owns shared scheduler/stage changes and serializes edits with20. Preserve native prefill pieces/eager breaks and required decode replay; a flag or capture log does not prove execution.

## Acceptance

- [ ] Long-prefill/short-arrival and mixed-length requests complete with causal admission, execution, and completion records; an idle DP lane preserves its required synchronization participation.
- [ ] Qualified decode graph replay, supported prefill captured pieces, and eager breaks select the observed dispatch; whole-prefill FULL capture and silent family substitution are not assumed.
- [ ] Planning reused across layers is charged once per actual step or metadata group, while layer-specific staging, summary updates, and kernels remain layer costs.
- [ ] Existing queues, held outputs, launch/drain behavior, and kernel-gap accounting reproduce approved ordering; no native thread, IPC, or per-kernel event framework is introduced.
- [ ] DP dummy forwards retain required collective work with zero useful payload; graph padding consumes its measured work and scratch without earning effective-token, region, or KV-frontier credit.
- [ ] First-D tail work occurs at the approved scheduling boundary and contributes once; delayed output visibility cannot be confused with completed request progress.
- [ ] Negative cases detect duplicated host planning, missing eager breaks, premature held-output release, and padded-token credit; controlled workflow replay is not presented as predictive online E2E parity.
- [ ] At least 50 baseline-versus-candidate scenarios preserve unchanged fidelity under specification §10; preregistered wall-time/memory budgets are measured, with explanation and disposition for every excess.

Specification references: §§4–5, 7, 9–10.
