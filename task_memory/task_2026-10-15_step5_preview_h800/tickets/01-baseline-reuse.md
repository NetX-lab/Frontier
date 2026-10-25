# 01: Baseline and shared-capability reuse audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-20 | Recorded the dense initialization timing excess. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Approved planning ticket created. |
| 2026-10-15 | Started the user-invoked native Grok local audit/CPU checkpoint; full-ticket acceptance remains open. |
| 2026-10-15 | Accepted checkpoint 1 after primary source/hash checks and 13 passing CPU tests; completed the source-identity criterion only. |
| 2026-10-15 | Accepted checkpoint 2: verified the 77-case capture/comparison contract and preserved external dependencies; budgets remain open. |
| 2026-10-15 | Primary completed compatibility mapping, evidence-rejection dispositions and prerequisite accounting; validated the existing CPU timing/RSS runner. |
| 2026-10-15 | Closed after measured budget preregistration, independent artifact checks and phase reviewer acceptance. |

**What to build:** An auditable baseline and reuse dossier that identifies the actual shared capabilities, admissible prior evidence, missing prerequisites, and preregistered comparison budgets before Step execution changes begin.

**Blocked by:** None (can start immediately once execution is authorized).

**External prerequisites:** The baseline audit is complete. Future changes use the current task authority and verified baseline scope; no model-support or native-performance claim follows from this ticket. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.

**Status:** done

**Review note — 2026-10-20:** Dense initialization rose from 4.198711431 s to 4.299223691 s (+2.394%), above the frozen 1.941% budget; the disposition is `UNVERIFIED_UNRESOLVED_MEASUREMENT_CONDITIONS`. It is re-measured with the next code change that touches initialization.

Audit execution is complete. Later tickets retain their evidence and semantic prerequisites under the user's continuous-execution authorization.

**Source workstreams:** S00.
**Shared owners:** C01–C06, through the existing T00–T11 owners.

## Review disposition — 2026-10-18

DONE: the baseline/reuse dossier and original budgets are accepted. Later dense initialization sentinel failures do not erase the original baseline; retain their UNVERIFIED dispositions and use matched conditions in affected future verification.

## Acceptance

- [x] Deliver the baseline reuse dossier with independently checkable source identities, evidence identities, owner assignments or explicitly unresolved owners, comparison coverage, and prerequisite dispositions; follow specification §§1, 9, and 10.
- [x] Record the actual source HEAD and dirty state, selected stacked base, source-reviewed main, and historical review identity separately; identify shared changes actually present rather than infer integration from branch ancestry.
- [x] Map each proposed reuse to the existing single owner and distinguish interface, semantic, wrapper, and timing-table compatibility; identify minimum missing capabilities without introducing parallel allocators, schedulers, publishers, or verdict engines.
- [x] Admit reused evidence only with exact source, environment, configuration, input, output, and applicability identities; mark mismatches and absent artifacts unverified instead of treating source citations as runtime results.
- [x] Establish the applicable existing unit/integration, public architecture smoke, and E2E-gate baseline using new authorized runs or precisely matching reusable records; report individual failures and structural-only evidence honestly.
- [x] Record baseline results for at least 50 concrete supported scenarios spanning dense/MoE, offline/online, lengths, counts, QPS, feature toggles, and models; define numeric-result and key-intermediate comparisons for every future code-changing step.
- [x] Preregister simulator wall-time and memory regression budgets before comparisons; retain measurement methodology and require measured deltas, explanations, and recorded dispositions for every over-budget regression. Planning does not invent budget values.
- [x] Preserve the inherited GLM full-path milestone and required qualified cross-model evidence as explicit dependencies; distinguish their recorded evidence from Step baseline checks and do not mark missing inherited results PASS.
- [x] Demonstrate rejection of fixture-only, dummy-timing-only, wrong-source, and incomplete evidence as native or predictive baseline PASS; source model fixtures remain expectations, not runtime constants.
- [x] Identify Step execution authorization and unresolved native identities as missing prerequisites; distinguish preparation from the future GLM full-path barrier applicable to production simulator integration.

## Execution evidence

Checkpoint 1 is accepted within its local audit scope. `source_map.md` separates all source identities and verifies that core source is unchanged from the historical baseline. `reuse_inventory.md` records six inherited documents and one manifest without admitting historical timing data. `test_report_2026-10-15_ticket01_checkpoint1.md` records the executor receipt and independent primary verification: 13 CPU tests passed, all seven reference hashes matched, and primary document corrections were applied.

Checkpoint 2 independently reproduced the 77-case capture (451 artifact hashes; 7 trained/70 dummy), 173 non-pass classifications, and 12 structural smoke outcomes. The primary reran the gate/placement tests: 83 passed. `baseline_audit.md` records case settings and intermediate artifacts; `test_report_2026-10-15_ticket01_checkpoint2.md` records commands and acceptance. The existing comparator requires exact equality after limited path normalization for future behavior-preserving changes.

All ten criteria are complete. Phase reviewer `ses_ebfba4f19ffe4q64MTbzEQ09Jv` accepted the dossier with no actionable finding; primary re-read requirements before closure. `reuse_inventory.md` sections 9–10 record compatibility and evidence dispositions. The original suite argv remains missing and is excluded from exact-replay claims. Fresh applicable documentation/integration baseline: 19 passed, 3 explicitly retained README failures, both PDD cases complete (2/2 and 8/8). See `test_report_2026-10-15_baseline_admission.md` for independent checks.

The primary also validated the existing CPU measurement runner on dense and MoE pilot cases: both completed 8/8 requests and recorded positive run time and RSS. Pilot evidence is in `test_report_2026-10-15_cpu_measurement_pilot.md`. These two single measurements establish runner availability, not repeat noise or performance budgets. Native identity/resources remain unqualified; no Step-native scenario requirement is added.

The subsequent authorized Grok checkpoint collected two warm-ups and ten measured attempts. Primary independently verified 72 hashes and all summaries; `performance_budgets.md` freezes case-specific measured-envelope review budgets before candidate comparisons, with explicit dense sensitivity limits and additional-coverage obligations. CPU/GPU measurements are authorized by the user's continuous-execution invocation. Actual native qualification and the GLM full-path milestone remain unfulfilled.
