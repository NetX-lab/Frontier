## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Created the current W00–W11 status index, separating observed validation, accepted semantic corrections, hardware limits and the open D02 decision. |

# Current remediation status

This index supersedes historical completion/audit status elsewhere in this task. The active specification is `Frontier_PR33_New_Execution_Plan_2026-09-16_EN.md`. Production source is `c9f8f904e3550c11aad3cc5d851d75d648cef6e1`; pinned main is `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. Later task-document changes do not imply a new source run. Original plans and historical test evidence are preserved.

**Current disposition: implementation, final-source functional validation and hunk review delivered; performance evidence and local handoff delivered. D02 is explicitly accepted for this measured scope.** No native hardware PASS, production-profile timing parity, remote publication or merge is claimed.

## Package status

| Package | Implementation | Verification | Readability / ownership cleanup |
| --- | --- | --- | --- |
| W00 | Completed: frozen baseline and initial paired campaign | 69 focused PASS; 18 paired executions; exact source/environment retained | Original471 hunks catalogued; current675 hunks/106 production files reconciled |
| W01 | Completed: authoritative model topology/family/shape | 51 focused PASS; actual MLA case in eight-case campaign | Existing binder and cached model specs reused |
| W02 | Completed: shared automatic capacity and atomic supported lifecycle | Capacity/OOM/rollback and three-request normal Simulator checks PASS; D03 resolved | Constructor-owned slot state; allocation before queue mutation |
| W03 | Completed: physical single-layer values and finalized ordered stages | Arithmetic/isolation/MTP tests PASS; final homogeneous optimization132 PASS | Ambiguous first-layer facade, sentinels and repeated snapshots removed |
| W04 | Completed: stage-local attention reuse and bounded immutable EP workloads | 21 controlled subprocesses; real hybrid attention-bypass median+22.34%; exact outputs | Reflection/repr fallback removed; independent routing identity retained |
| W05 | Completed: whole-campaign preflight, timer/sample/cleanup contracts | 111 focused PASS; native capabilities individually SKIP | Existing timer/store and context management reused |
| W06 | Completed: canonical per-field path precedence and strict metadata | 51 focused PASS | Manager, direct predictor and training share existing boundary |
| W07 | Completed: adjacent profiling, replay, layout and byte contracts | CPU contracts PASS; ten real native adapter nodes collected, hardware SKIP | All profiling hunks reviewed; actual backend identity and helpers reused |
| W08 | Completed: stage/physical-lane metrics and requested-output behavior | Independent mixed-layer and EP phase/barrier oracles PASS; real exported artifacts verified | Existing component/family/operator maps; focused86-line EP reporter |
| W09 | Completed: GDN selected identity, task completeness and atomic publication | 108 focused PASS; real CPU fit/save/fresh-load/exact/non-exact checks | Canonical task definition and existing atomic pickle utility reused |
| W10 | Completed, lanes individually classified | CPU3587PASS/18knownFAIL/25SKIP;8syntheticPASS;58dummy raw24PASS/34classified differences;19nativeSKIP | No candidate-only failure/new skip;574 fidelity and106 synthetic artifacts unchanged after final optimization |
| W11 | Completed: RCA, hunk review and local handoff; D02 accepted |18 unprofiled runs; run ratios1.775045x/1.679023x/0.978816x; user accepted measured scoped residual |675 current +471 original hunks reviewed; large-module split decisions recorded |

## Decisions and evidence boundaries

- **D01 resolved for uniform layer/stage semantics.** The user accepted the isolated physical-layer corrections. The dummy campaign retains raw comparator differences: same wrong `/32` normalization also occurred in PDD, and the planned reporting migration removes duplicate layer multipliers and adds actual EP lane records. These are independently verified implementation corrections under §6, not a tolerance waiver. All unclassified discrepancies remain blocking.
- **D03 resolved.** Keep admission, continuation, completion and failure cleanup. No request-level cancellation API exists; `StageExecutionContext.cancel(ticket)` is not one. A new request cancellation API is explicitly deferred.
- **D02 resolved.** The user explicitly accepted the final measured residual and retained the current contract. No universal budget, native result, new representation change, publication or merge approval is implied.
- Native timer/GDN/MoE/collective/SGLang execution is hardware-limited. Collected tests invoke real wrappers when capable; CPU SKIP is not native correctness evidence.
- E2E profiles are explicitly synthetic. The eight-case CPU campaign establishes constructor/trainer/predictor/scheduler/export integration and independent numerical ownership, not production latency or benchmark parity.

## Final performance observation

Small dense median run14.149→25.225ms; longer dense29.309→49.268ms; representative MoE9.730→9.524s. Median paired ratios are1.775045x,1.679023x and0.978816x. Repeated homogeneous snapshots and block sums were removed; actual per-layer identities and finalized publication remain. The full distributions, separate init/process/RSS/CPU figures, causal limits and representation alternative are in `performance_rca.md`. User acceptance is explicit and limited to the measured residual; it was not inferred from near1x process ratios.

## Current evidence entry points

- `validation_manifest.json`: source, environment, exact commands, statuses, artifact identities.
- `test_report_2026-09-16_w10_cpu.md`: full CPU node/cause comparison.
- `test_report_2026-09-16_w10_fidelity.md`: 58-case raw comparison and independent arithmetic classification.
- `test_report_2026-09-16_w10_nondummy_acceptance.md` and `w10_nondummy_report.md`: eight non-dummy cases and seven main controls.
- `performance_rca.md` and `w04_performance_review.md`: separated initial/final pairing, causal ablations and cost limits.
- `changed_hunk_review.md`, `hunk_review_predictors.md`, `hunk_review_runtime.md`, `hunk_review_attention_training.md`, `hunk_review_shared_contracts.md`, `w07_sglang_review.md`: complete production review and responsibility decisions.
- `review_reassessment_2026-09-16.md`: all38 original findings, R01–R14 and N01–N09 dispositions.

## Remaining actions

None for the authorized local delivery. D01, D02 and D03 are resolved. Further representation optimization is not selected; no new implementation direction is inferred.

No further code or verification action is pending for the delivered candidate. No new unexplained issue remains. Native execution and production-profile parity remain explicitly unverified future capabilities.

No new unexplained production defect is established at this checkpoint. Eighteen pre-existing CPU failures remain visible in the final full run and have been matched by node/cause. The baseline has19 failures; one existing profiling documentation assertion now passes. Twenty-five unit skips are identical by node. Remote publication and merge are outside this remediation authorization.
