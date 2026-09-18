## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Added the subsequent PR33 review-remediation archive and current evidence. |
| 2026-09-17 | Archived completed diff-driven cleanup, final verification, performance measurements and explicit limits. |

# Current PR33 Review Remediation

This section supersedes the historical cleanup completion below for the subsequent request to fix the d43ae932 review. All R01–R10 corrections and C01/C02 improvements are implemented in separately verified commits. C03 is a local PR-description handoff; no remote publication is included.

The user's explicit R01-A decision preserves the selected scheduler. A mixed batch uses prefill estimators for all GDN timing with a RuntimeWarning, while actual request phases, physical features, KV ownership and recurrent-state slots remain intact. The co-location limitation and lack of native mixed timing accuracy are documented. Native producer/training mixed rows remain rejected; full-attention still needs matching mixed profiles.

Production/test checkpoint: `35ac95eb231aa3990a25cd49e24b62b034bda3a3`. Reviewed source: `d43ae93240444bd4eff9bd99f296d2e751370514`. Preservation baseline: `c288a19f59bec09529ee18d782fa57218da2c781`. Frozen main remains `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`; accepted main-to-candidate D01 changes must not be confused with cleanup parity. Subsequent changes are documentation only.

## Deliverables Inventory

All report paths below are relative to `task_memory/task_2026-09-17_candidate_quality_cleanup/` in the active worktree.

| Item | Commit | Delivered correction | Report |
| --- | --- | --- | --- |
| R01 | 9f5c0c45 | Authorized mixed-to-prefill approximation, warning, docs, real concurrency/ownership regression | test_report_2026-09-17_r01_approximation.md |
| R02 | 20b85ce7 | Canonical runtime attention-family binding for hybrid ROCm full-attention | test_report_2026-09-17_r02_binding.md |
| R03 | 2030968d | Immutable estimator generations with atomic manifest publication | test_report_2026-09-17_r03_publication.md |
| R04 | 336bdaf0 | Producer/import/runtime ragged-feature consistency | test_report_2026-09-17_r04_features.md |
| R05 | e7ccd04c | Receiving-model normalized dtype validation | test_report_2026-09-17_r05_dtype.md |
| R06 | 6e61f99f | Effective timer-owner provenance validation before native setup | test_report_2026-09-17_r06_timer_owner.md |
| R07 | b3fc348f | Canonical shell output paths and executed postflight coverage | test_report_2026-09-17_r07_launchers.md |
| R08 | 360c54d6 | Explicit native ROCm backend and phase coverage in recipe | test_report_2026-09-17_r08_backend_recipe.md |
| R09 | 30ce7b28 | Eight-process TP8 launch in three docs, separate TP1 example | test_report_2026-09-17_r09_distributed_recipe.md |
| R10 | af4fb055 | Single normalization of replay workload iterables | test_report_2026-09-17_r10_iterables.md |
| C01 | 71c0f679 | Capacity check without diagnostic owner tuple allocation | test_report_2026-09-17_c01_capacity.md |
| C02 | 35ac95eb | Demand-gated reporting payloads, including summary-only consumers | test_report_2026-09-17_c02_reporting.md |
| Final verification | documentation archive | Commands, baseline classification, artifact/numeric/timing evidence | test_report_2026-09-17_review_final.md; review_final_evidence.json |
| C03 | documentation archive | Current source/baseline/evidence/scope in reviewable local PR body | pr33_description.md |

`requirements.md` preserves the raw requests and R01-A decision; `plan.md`, `progress.md`, `review_remediation_decisions.md` and `refactoring_record.md` retain implementation order, superseded proposals, failures and corrections. The mixed-batch documentation is in `docs/training/README.md` and `docs/profiling/ROCM_MI355X.md`; command copies are in `docs/profiling/README.md` and `docs/cli/README.md`.

## Validation Status

- Full unit: **3928 PASS / 18 existing FAIL / 25 existing SKIP**, 167.43 s. Failed/skipped node sets exactly match the previous frozen validation. Failure messages also match after replacing only the scratch-directory prefix. The suite is not all green.
- Real non-dummy Simulator: **9 PASS**, 81.28 s. New concurrency case observes 14 batches, including two mixed batches; four requests complete with stable/reused/released GDN ownership. Eight existing cases preserve 90 stable artifacts, a symmetric 106-file inventory and 16 nonempty supplemental JSON pairs.
- Frozen-candidate fidelity: **58 PASS / 0 FAIL**, 304 artifacts, 122 completed requests per side. Zero drift across 902628 finite numeric leaf pairs; maximum absolute and relative differences are both 0.0.
- C01/C02 focused allocation/call evidence: no owner-tuple access during admission; disabled reporting reduces the deterministic completion fixture from two predictor calls to one and one full-stage allocation to zero, preserving event/wall/model timing and the metrics callback.
- Final isolated timing: **18 successful measurements**, all nine event/completion-preserving pairs. Median paired Simulator.run changes: **−12.07% / −8.84% / −0.62%** for small dense / longer dense / representative MoE; subprocess changes **−3.62% / −6.90% / −2.92%**. Small-dense attempt 2 is 16.62% slower and remains in the evidence. Three reporting-disabled dummy pairs do not establish general or native performance.

## Open Items / Future Extensions

In-scope code corrections remaining: **none**. New unresolved code regressions: **none observed**. The existing 18 baseline failures remain classified in the final report. Native ROCm/CUDA/SGLang/GDN execution, TP8 hardware validation and production-profile numerical accuracy are not established by CPU contracts or synthetic fixtures. Related output-gate variant interchangeability remains unverified and was not generalized speculatively. The user-authorized mixed prefill approximation remains a documented accuracy limitation.

The PR description is prepared locally. External PR publication, push and merge have not been performed. Broader performance studies and native campaigns require their own environment and scope. Historical evidence below is preserved without relabeling it as a run of the corrected source.

# Historical Cleanup Through d43ae932

## Task Overview

Completed the dependency-ordered review and implementation requested for the Frontier feature worktree: runtime -> metrics -> profiling -> training/model management/configuration. The objective was explicit invariants, clear ownership and reuse of existing policy, not cosmetic formatting or a repository-wide rewrite.

- Frozen main and merge-base: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Frozen pre-cleanup candidate: `c288a19f59bec09529ee18d782fa57218da2c781`.
- Final production source: `5cb8794f49fb3e27e84316a3f87ebadbcb9d1574`; final test-only isolation correction: `ff2b6cdf`.
- Final paired measurement checkpoint: `567742f3df4cdc818400d00c0575426d66d0dee3`, with the same production/test source. Subsequent final archive changes are documentation-only.
- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
- Audited all 106 changed paths under `frontier/` (105 Python files and one preserved README): 38 cleaned, 68 inspected and intentionally retained. Linked hunk/caller reasoning exists for every path. The initial branch diff also includes tests, assets and historical records; the 106 count specifically describes production-area paths.
- Completed code substeps were committed individually after their required verification. No push, merge, history rewrite, worktree removal, README edit or external publication was performed.

## Delivered implementation

1. Runtime construction and ownership: removed duplicate predictor construction, missing-member recovery on valid objects, unused aggregate timing state and private one-layer width parameters. Memory, GDN guards and batch features use model/Batch/Request contracts directly. GDN slot allocation uses a heap while preserving minimum-free-ID selection and lifecycle cleanup.
2. Canonical policy: architecture identity, attention family/schema/tasks, supplied-model capabilities, routing integerization, measurement path substitution and GDN gate defaults reuse their existing owners. No new local model/family classifier or duplicate default was added.
3. Metrics and traces: removed orphaned private overrides and test-only reporting helpers; retained real Stage/scalar and lane/barrier distinctions. Phase-local trace context avoids repeated per-operator work. Positive residual metadata is a separately diagnosed correctness restoration, described below.
4. Profiling: reused admitted ROCm phase plans and canonical model parsing, centralized exact native-type declarations in the existing registry, and made experimental replay arguments/metadata explicit. Standard profiling does not depend on experimental SGLang code.
5. Training/configuration: required registries and topology members are constructor invariants; valid lightweight fixtures replace incomplete objects. Dataset-only model absence, sparse path overrides, explicit selector precedence, artifact identities and native lifecycle optionals remain supported. Unread trainer DataFrame state was removed.

The seven already-large critical modules have cleanup-first findings and concrete split analyses. Broad module extraction remains outside this task; no extraction is silently represented as implemented.

## Deliverables Inventory

Paths below are exact and relative to the worktree root above.

| Deliverable | Path |
| --- | --- |
| User intent and behavior boundaries | `task_memory/task_2026-09-17_candidate_quality_cleanup/requirements.md` |
| Completed dependency plan | `task_memory/task_2026-09-17_candidate_quality_cleanup/plan.md` |
| Execution history, failures, RCA and decisions | `task_memory/task_2026-09-17_candidate_quality_cleanup/progress.md` |
| Every changed production path, rank, disposition and evidence link | `task_memory/task_2026-09-17_candidate_quality_cleanup/inventory.md` |
| Requested seven-column refactoring table and retained-pattern table | `task_memory/task_2026-09-17_candidate_quality_cleanup/refactoring_record.md` |
| Independent Standards review | `task_memory/task_2026-09-17_candidate_quality_cleanup/standards_review.md` |
| Independent Spec review | `task_memory/task_2026-09-17_candidate_quality_cleanup/spec_review.md` |
| Attention/binding review | `task_memory/task_2026-09-17_candidate_quality_cleanup/attention_review.md` |
| Metrics/ownership review | `task_memory/task_2026-09-17_candidate_quality_cleanup/metrics_review.md` |
| Cleanup-first large-module analysis | `task_memory/task_2026-09-17_candidate_quality_cleanup/large_module_review.md` |
| Residual metadata discrepancy and controls | `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p2_residual_metadata.md` |
| Profiling broader regression and native limits | `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p3_integrated.md` |
| Training artifact and cross-load preservation | `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p4_gdn.md` |
| Training/configuration broader regression | `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p4_integrated.md` |
| Final correctness commands, outcomes and baseline classification | `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p5_final.md` |
| Final performance command, all samples and interpretation | `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p5_timing.md` |

The inventory links the remaining focused reports, without duplicating their evidence here. Original reviewer proposals are historical; the refactoring record and implementation reports supply their final dispositions. Durable numerical summaries, sample tables, commands and selected errors are in the reports; raw logs/caches remain under `/data/ycfeng/tmp` at the recorded paths.

## Validation Status

Environment: Python 3.12.3 at `/data/ycfeng/tmp/quality-review-env/bin/python`, uv venv with same-interpreter system packages, no conda activation. Focused checks and before/after comparisons accompanied each logical code change. Final checks used frozen production source.

| Gate | Observed outcome |
| --- | --- |
| Full unit | **3813 PASS / 18 FAIL / 25 SKIP**, 132.23 s. The exact 18 failed nodes match both frozen candidate and main. All 25 final skipped nodes also skip on frozen candidate. Baseline had 3587 PASS, so the cleanup adds 226 passing parameterized checks. The suite is not all green. |
| Reporting-enabled non-dummy acceptance | **8 PASS**, 68.21 s. 90 stable artifacts, the bidirectional 106-file metric inventory and 16 explicitly nonempty acceptance/summary pairs match frozen candidate. |
| Final simulator fidelity | **58 PASS / 0 FAIL**, 304 artifacts, 122 completed requests per revision. 902,628 finite numeric leaf pairs contain zero unequal values; maximum absolute/relative difference is 0.0. These include numeric metadata, not that many independent predictions. |
| P3 broader CPU profiling regression | **721 PASS / 5 SKIP / 2 known FAIL**. The failures are main/frozen-equivalent stage2 CLI missing-FlashInfer paths, not new profiling regressions. |
| P4 broader training/config regression | **545 PASS / 0 FAIL / 0 SKIP**, 56.12 s, after correcting the new test module's global-state leak. No production guard was weakened. |
| GDN artifact comparison | 48 byte-identical estimators plus eight identical manifests; 32 cross-loads preserve 192 exact prediction dictionaries. |
| Final paired performance | **18 successful runs**, three baseline/candidate pairs for each case, all event/completion counts equal. No concurrent task-owned test load. See below for scope. |

Median paired `Simulator.run()` changes are **-13.46% small dense, -2.00% longer dense, -1.46% representative MoE**. Corresponding subprocess changes are **-4.89%, -3.12%, -2.52%**. Each percentage is the median of paired ratios, not a ratio of marginal medians. All samples are retained; the slower MoE baseline sample was not excluded. The small dense path shows a consistent measured reduction; the longer dense/MoE changes are modest and three pairs do not establish general statistical significance.

The performance harness is sequential online PDD with a dummy predictor and reporting disabled on both revisions. It does not measure reporting-enabled or native-device speed. The separate non-dummy campaign keeps reporting enabled and verifies its artifacts; no simulation or metrics semantics were dropped to obtain a speedup.

### Explicit discrepancies and resolutions

- **Candidate residual metadata bug:** six frozen-candidate trace cases failed because fully resolved residual metadata was treated as extra tensor identity. Three scalar main controls established the supported path. The local emitter now distinguishes resolved metadata from additive physical-layer identity, with no error suppression or hard-coded operator escape. This is a documented correctness restoration, not claimed equality with the broken candidate cases.
- **New test fixture state leak:** initial P4 combined run had three GDN failures because dense config tests left global `IS_MOE=False`. A two-node reproducer established the cause. The new fixture uses the existing `global_vars.reset_global_vars` setup/teardown; production invariants remain unchanged. Corrected broader and final unit checks close the issue.
- **Other transient investigation failures:** import ordering, incomplete doubles, environment setup and mistaken characterization expectations are recorded with their causes/corrections in progress and focused reports. No expected numerical output was changed to hide a discrepancy.

## Open Items / Future Extensions

- Pending in-scope implementation: **none**. Consequential unresolved semantic decisions: **none**. No grill-me decision gate was required.
- Existing 18 main-equivalent failures remain outside scope: ten legacy optimizer/debug-path contracts, five public documentation contracts, two MHA/MQA stage2 CLI dependency failures and one MLA trace-builder dependency failure. README and missing legacy assets were not rewritten as part of cleanup. The exact node sets and known causes are documented in the final correctness report.
- Native CUDA/ROCm/SGLang/GDN kernel accuracy, production-profile prediction accuracy and live vLLM parity were not established by these CPU/synthetic checks. They require appropriate native environments, hardware and datasets in a separate scoped task.
- Broad extraction of the seven pre-existing oversized modules and statistically stronger reporting-enabled performance campaigns are possible future work, not hidden unfinished cleanup requirements.

Planning-with-files maintained the existing task directory; code-review provided independent Standards/Spec evidence; codebase-design guided invariant/ownership simplification; diagnosing-bugs guided the fixture-state RCA. User scope and preserved behavior took precedence over generic skill layouts or wider workflow suggestions.
