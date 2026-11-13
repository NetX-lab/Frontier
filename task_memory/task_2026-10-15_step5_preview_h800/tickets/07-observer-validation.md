# 07: Single-role observer correlation and perturbation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-22 | Correct stale waiting status after accepted06.simulation_scope; native correlation and perturbation remain unfinished. |
| 2026-10-20 | Changed the native prerequisite to `06.simulation_scope` (decision D2). |
| 2026-10-19 | Completed source/design preparation and scoped parent verification; implementation/native/finalacceptance remains blocked by existing gates. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Approved planning ticket created. |

**What to build:** A validated full-role observer record set whose causal pairing, scope coverage, and measured perturbation are trustworthy for later cost replay and decision evidence, while clean runs remain the performance reference.

**Blocked by:** `03` for instrumentation_preparation (satisfied); `06.simulation_scope` for full native observer validation and whole-ticket completion (decision D2, satisfied). Only the observations a mainline ticket consumes are validated; no new observation tooling is built.

**External prerequisites:** Preparation may use03 and retained native sources; complete observer validation requires06. Native observation has no GLM barrier and does not approve a new scheduler architecture. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.

**Status:** in-progress; instrumentation preparation complete, native correlation and perturbation validation pending. The required06.simulation_scope is complete; no completed06 prerequisite remains a waiting condition. References to unfinished06 in the dated preparation records below are historical.

**D31 status (2026-10-25):** Paused by D31.4. Observer validation is not needed for the E2E comparison.

The implementation phase is active under `../requirements.md`. Routine in-scope instrumentation, incremental extensions and tests use the standing authority; preparation evidence does not satisfy native acceptance.

**Source workstreams:** S05.
**Shared owners:** C04, C05, C06.

## Review disposition — 2026-10-18

`07.instrumentation_preparation` may start after03 using the existing06 CPU metadata and shared C06/native observer seams. Define only fields required by the next full-role/PD observation and preserve hot-path behavior. Full07 remains blocked on06 and measured correlation/perturbation; fake observer tests are not native validation.

## Acceptance

- [ ] Deliver the observer validation report, correlated full-role record set, perturbation measurements, coverage/missing-evidence ledger, and asynchronous decision evidence under §§5, 7, 9, and 10.
- [ ] Separate clean, light-observer, detailed-trace, and eventual Frontier records with exact runtime/configuration/instrumentation identities; earlier unvalidated observations remain diagnostic unless explicitly validated for the intended use.
- [ ] Match requests, steps, ranks, groups, and physical scopes through stable identities and causal relationships, including empty DP-lane forwards; establish pair validity without nth-forward assumptions or DP×PP-based mode selection.
- [ ] Compare matched clean and observed repetitions under preregistered perturbation rules, recording token work, routing, dispatch, graph behavior, latency differences, and variability; pair validity is not proof that an observed run is clean performance truth.
- [ ] Demonstrate no observer-added hot-path synchronization, modified routing/dispatch, changed token work, or new graph breaks; keep diagnostic synchronization outside clean timing.
- [ ] Account for each physical scope and kernel once, with device gaps, host preparation, graph launch, and planning reuse distinguished; count planning once per actual reused metadata group rather than once per layer.
- [ ] Capture long-prefill/short-arrival, idle-lane, held-output, launch/drain, and first-decode-relevant boundaries sufficient for existing D-ASYNC review; report unsupported inferences instead of copying native threads or proposing per-kernel simulator events.
- [ ] Reject missing rank/request/scope records, ambiguous or duplicate pairs, mismatched identities, and bare cross-node clock subtraction; explicit insufficiency must prevent a valid-observer verdict and formal cost-replay use.
- [ ] Use small known fixtures only to verify correlation and normalization tools; validate actual full-role observations separately. No fixture result or trace-conditioned replay can close predictive accuracy acceptance.
- [ ] Observer-tooling changes meet §10's ≥50-scenario fidelity checks and preregistered wall-time/memory budgets with over-budget dispositions; ≥32-GPU auxiliary validator failures preserve healthy workloads, notify the user, and leave affected evidence unverified.

## Completed preparation scope — 2026-10-19

`instrumentation_preparation` source/design preparation is complete in `../observer_preparation_20261019.md`. Parent `../native_request_audit_20261019_01/preparation_document_checks_02.json` verifies its recorded source/document identities within the exact scope in `../test_report_2026-10-19_native_request.md`; no native or predictive acceptance is inferred. The document retains its original inspected snapshot, while current task status is recorded here. Whole-ticket completion still requires 06 and native correlation/perturbation. No acceptance checkbox is closed.
