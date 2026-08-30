## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-08-29 | Checkpoint 50: reviewed and verified global layer identity propagation at the ReplicaStageScheduler PP boundary; the new RED test and mixed-layer scheduler regressions pass. |
| 2026-08-30 | Checkpoint 49: independently reproduced and fixed the pure-MoE inactive dense-contract leak; resolver/registry matrix passes `64/64`. |
| 2026-08-29 | Checkpoint 48: reviewed ParamCounter enum TP dispatch and non-uniform MoE/PP layer-map boundary; strict identity/counting matrix passes `70/70`. |
| 2026-08-29 | Checkpoint 47: reviewed the sklearn MoE routed-width migration; profile-owned filtering and runtime features pass `101/101` focused regressions. |
| 2026-08-29 | Checkpoint 46: reviewed typed manager alias ownership and layernorm-context isolation; `105/105` focused regressions pass. |
| 2026-08-29 | Checkpoint 45: reviewed typed-width dataframe filtering and cache separation; registry-scoped alias handling preserves legacy memory behavior, with `152` focused regressions passing. |
| 2026-08-29 | Checkpoint 44: independently reviewed the predictor typed-contract consumer; RED TP1 was corrected to dense TP8, routed TP1/EP8, and shared TP8 with `161` focused regressions passing. |
| 2026-08-29 | Checkpoint 42: recorded Option-2 dedicated typed-registry approval and verified the RED implementation entry point; dense-18432 profiling remains separately gated. |

## Checkpoint 49

- **Target Component/Phase:** Profile-owned layer-contract activation in
  `ModelArchitectureProfile.resolve_layer_contract`
- **Reviewer Agent Identity:** `/root` with independent finding from
  `/root/option2_architecture_review`
- **Inspected Artifacts:** `frontier/model_architectures.py`,
  `tests/unit/test_model_architecture_registry.py`, the pure-MoE profile
  activation list, and the direct resolver probe.
- **Identified Issues/Anomalies:** A pure-MoE config materialized only routed
  and shared contracts, but an operator-only `mlp_up_proj` query still selected
  the inactive dense contract and fell back to the legacy routed width. This
  could misbind manager/predictor lookups without a concrete `layer_id`.
- **Remediation/Verification Code Actions Taken:** Added one final
  profile-owned activation check after contract selection. Added a regression
  asserting the pure-MoE query raises an explicit inactive-contract error.
  The focused registry file reports `64 passed in 6.22s`; no data,
  manifest, README, worker, or remote state changed.

## Checkpoint 50

- **Target Component/Phase:** Global layer identity propagation at the
  `ReplicaStageScheduler.predict_and_create_stage()` boundary
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:**
  `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py`,
  `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`, the new
  `tests/unit/test_replica_stage_scheduler_layer_identity.py`, existing
  mixed-layer scheduler tests, and fresh pytest/compile/diff-check output.
- **Identified Issues/Anomalies:** Before the change, a regular PP stage read
  its configured layer count but always passed `layer_id=0` and no complete
  identity tuple. The failure was directly reproduced for stage `1`, where the
  expected global range is `4..7`.
- **Remediation/Verification Code Actions Taken:** Reused the existing stage
  bounds helper; regular stages now pass `layer_id=4` and
  `layer_ids=(4, 5, 6, 7)` in the focused case. PD-AF single-layer branches
  retain their existing `num_layers=1` and scalar handling. The new test
  reports `1 passed`, the existing focused scheduler subset reports `5 passed,
  135 deselected`, and `py_compile` plus `git diff --check` exit `0`. Predictor
  API/reconstruction propagation remains open for the next checkpoint.

## Checkpoint 48

- **Target Component/Phase:** ParamCounter typed TP dispatch and MoE layer-map
  counting under pipeline parallelism
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `frontier/utils/param_counter.py`,
  `frontier/model_architectures.py`, `frontier/config/model_config.py`, the
  existing PP stage-boundary helper, and
  `tests/unit/test_param_counter_typed_layer_contract.py` plus the focused
  ParamCounter/registry matrix
- **Identified Issues/Anomalies:** TP dispatch compared enum display strings;
  the MoE/PP path rounded a model-wide ratio and could report a count that no
  pipeline stage owned. The current API has no stage index, so an uneven map
  cannot be represented exactly by one returned count.
- **Remediation/Verification Code Actions Taken:** Switched TP selection to
  `TensorParallelMode` identity. Added exact per-stage counting from the
  existing layer-ID contract, strict ID validation, and fail-fast admission for
  non-uniform maps or partial counts without IDs. The focused matrix reports
  `70 passed in 6.54s`; Step3 and uniform/pure-model totals remain unchanged.

## Checkpoint 46

- **Target Component/Phase:** Profile-owned typed prediction-manager
  compatibility fixes (alias binding and layernorm training context)
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/shared_prediction_model_manager.py`,
  `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`,
  `frontier/model_architectures.py`, registered operator families/bindings,
  `tests/unit/test_shared_prediction_model_manager_typed_contract.py`,
  `tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py`,
  `tests/unit/test_operator_query_tp_consumers.py`, and the focused pytest
  output (`105 passed in 4.85s`)
- **Identified Issues/Anomalies:** An unqualified MEMORY `add` alias maps to
  both residual profiling operators and must not enter the profile-owned FFN
  typed resolver. The FFN `training_context` also carried a typed contract
  into `post_attention_layernorm`, which could corrupt cache identity or make
  legacy layernorm retrieval ambiguous.
- **Remediation/Verification Code Actions Taken:** Manager resolution now
  determines family ownership from the declarative operator registry before
  scoped binding; non-owned aliases use the existing legacy TP resolver while
  true typed-family ambiguity remains fail-fast. Layernorm training receives a
  copy of the base context without `layer_contract`, with an explicit test
  assertion. The focused matrix passes `105/105`; no data, manifest, README,
  worker, or remote state changed.

## Checkpoint 43

- **Target Component/Phase:** Option-1 typed-layer ownership and source-migration entry gate
- **Reviewer Agent Identity:** `/root`, `/root/option2_architecture_review`
- **Inspected Artifacts:** `requirements.md`, `plan.md`, `design.md`, `harness.md`,
  `frontier/model_architectures.py`, the runtime/profiling model loaders,
  operator registry, predictor consumer, and focused RED test output
- **Identified Issues/Anomalies:** Historical records used option-2 wording for
  ownership, but the latest maintainer decision selects option 1. The live RED
  test still observes TP `1` for a Step3 dense `mlp_*` query where Attention TP
  `8` is required.
- **Remediation/Verification Code Actions Taken:** Reconciled the current
  authority in task docs without touching source, accepted data, manifests,
  README files, untracked files, workers, or remote refs. Authorized the next
  bounded step: add the typed contract to the existing architecture profile
  registry, then migrate consumers through RED/GREEN checks.

## Checkpoint 44

- **Target Component/Phase:** `SklearnExecutionTimePredictor` typed-layer TP consumer
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `frontier/model_architectures.py`,
  `frontier/operators/binding.py`, `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`,
  the Step3 typed contract tests, existing FFN/operator regressions, and the
  fresh pytest/compile output
- **Identified Issues/Anomalies:** The pre-change predictor used the generic
  model-level MoE branch and returned TP `1` for a mixed Step3 dense query.
  A naive family-name branch would violate the registry ownership decision;
  the resolver also had to preserve the PD-AF `DECODE_FFN` local FFN field.
- **Remediation/Verification Code Actions Taken:** Bound operator identity
  through `bind_operator_query`, selected only profile-declared typed families,
  resolved width/TP/EP through `resolve_layer_contract`, and kept all other
  operators on the existing path. Added routed/shared/DECODE_FFN tests. The
  focused matrix passed `161/161`; `py_compile` exited `0`. No CSV, manifest,
  README, worker, or remote state changed.
| 2026-08-29 | Added Checkpoint 41: independently reproduced the Option-2 mixed-layer TP and dense-width failures, corrected the implementation verdict to BLOCK, and recorded the shared-contract ownership gate. |
| 2026-08-28 | Reviewed the documentation-only continuation: six-model measurement scope is closed, option A is selected but not implemented, and option B plus the Step3 option-2 resolver are deferred. |
| 2026-08-28 | Added the option-2 RCA re-review, compatibility assessment, and staged implementation boundary before source edits. |
| 2026-08-28 | Reviewed the current merge-snapshot contract rerun: `47` passed and `10` failed on the validator's retired `simulation`/`uniform_random` routing vocabulary; no unresolved index entries exist, but `MERGE_HEAD` and cached whitespace keep the branch out of PR review. |
| 2026-08-28 | Corrected the prior merge-conflict wording: the current index has no unresolved entries or conflict markers; the remaining merge gate is the active `MERGE_HEAD` plus cached whitespace, while three conflicts are present only in the clean merge-tree preview. |
| 2026-08-28 | Reviewed the fresh direct validator matrix; 6/6 contracts and profiles pass, 2/6 runtime artifacts pass, and three metadata mismatches reinforce the cross-module PR blocker. |
| 2026-08-28 | Reviewed the live merge whitespace gate: `git diff --check` exits 2 on 12 conflict markers plus staged trailing whitespace; committed ref-range whitespace remains clean but is insufficient for PR readiness. |
| 2026-08-28 | Direct validator construction exposed a routing vocabulary mismatch (`simulation` rejected by the canonical resolver); classified it as a live, pre-CSV contract blocker pending the in-progress migration. |
| 2026-08-28 | Re-ran focused contracts under the live merge state: collection exits 2 on the unresolved `merge_profile_csv_contexts.py` marker, while the independent parallel-semantics suite passes 4/4; kept the verdict BLOCK. |
| 2026-08-28 | Reviewed the fresh branch-local contract test: 43 pass and 10 fail because the H200 validator still uses `data_parallel_size` after the resolver migrated to `num_replicas`; classified as an additional PR blocker. |
| 2026-08-28 | Added the corrected isolated latest-main consumer recheck; snapshot cwd/import provenance is now explicit and the BLOCK verdict remains unchanged. |
| 2026-08-27 | Added an independent isolated-matrix review and reconciled the historical narrow count (64) with 96 unique gating-op failures plus 48 repeated shared-FFN status entries. |
| 2026-08-27 | Reviewed the latest-main compatibility checkpoint; producer data passes, but MoE context admission, Step3 identity, mixed-attention publication, and merge conflicts keep the branch out of PR scope. |
| 2026-08-27 | Reviewed the second-session H200 continuation audit; all six profile/runtime lanes pass, full-tree counts are `56` CSVs/`14,064` rows, and no worker allocation is justified. |
| 2026-08-27 | Reviewed the fresh current-validator recheck; all six frozen H200 profile/runtime lanes pass and no new worker allocation is needed. |
| 2026-08-26 | Reviewed the post-r19 remaining-model audit and confirmed that available H200 capacity does not correspond to an unfinished frozen-scope measurement. |
| 2026-08-26 | Reviewed Mixtral r19 profiling, independent frozen validation, formal `2+2` E2E, six-model aggregate, and worker release; confirmed publication remains gated. |
| 2026-08-26 | Reviewed r14 terminal profiling evidence, confirmed the `/mnt/host0` cache permission root cause, verified the r15 writable-cache probe, and confirmed that exact H200 capacity still blocks allocation. |
| 2026-08-25 | Reviewed r12 exact H200 capacity gate and confirmed Mixtral remains blocked with no worker; deferred Step3 source/topology changes remain untouched. |
| 2026-08-25 | Reviewed the r9/r10/r11 Mixtral continuation attempts and reconciled the authoritative no-worker/capacity-blocked state; Step3 changes remain deferred. |
| 2026-08-25 | Reviewed the refreshed r7 checkpoint and current worktree state; confirmed Mixtral is the sole active incomplete lane and Step3 semantic changes remain deferred. |
| 2026-08-25 | Reviewed the Mixtral r7 worker/cache launch gate and confirmed it is the sole active incomplete-model lane; Step3 source changes remain deferred. |
| 2026-08-25 | Reviewed the six-model inventory and deferred Step3 decision; confirmed Mixtral is the sole incomplete formal profiling/E2E lane and no frozen artifacts were changed. |
| 2026-08-24 | Reviewed five-model accepted-staging aggregate revalidation; all 46 CSVs and 11,720 physical rows pass the frozen manifest. |
| 2026-08-24 | Reviewed the fixed Step3 accepted-data E2E retry; confirmed preflight PASS, deterministic parameter-memory admission failure, and the maintainer decision gate before Mixtral. |
| 2026-08-24 | Reviewed Step3 worker-local-cache profiling, producer/independent validation, exact row and timing metrics, and the transition to Step3 E2E/Mixtral. |
| 2026-08-24 | Reviewed qwen3-a3b frozen-manifest validation, formal 2+2 E2E evidence, and terminal worker cleanup; all current gates pass. |
| 2026-08-24 | Historical checkpoint: reviewed the successful H200 retry allocation while qwen3-a3b profiling was still in progress. |
| 2026-08-24 | Reviewed the H200 live quota attempt; predict-only capacity and live quota diverged, and no worker or profiling process was allowed to start. |
| 2026-08-24 | Reviewed accepted-staging revalidation, focused pytest, README gate, and read-only archive/coverage manifests; all local gates pass while H200 capacity remains blocked. |
| 2026-08-24 | Reviewed the completed Qwen formal E2E and fresh H200 predict-only retry; confirmed the worker blocker is external and live profiling was correctly suppressed. |
| 2026-08-23 | Added the second-model formal profiling/E2E and validator-repair checkpoint. |
| 2026-08-23 | Reviewed the H200 preemption evidence and replacement allocation gate. |
| 2026-08-23 | Added the H200 runtime-contract and attention-dedup checkpoint review. |
| 2026-08-23 | Added the six-model configuration and H200 environment checkpoint. |
| 2026-08-23 | Initialized the checkpoint review ledger. |

# Review

## Checkpoint 45

- **Target Component/Phase:** `SklearnExecutionTimePredictor` typed FFN width
  filtering and training dataframe cache
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`,
  `frontier/model_architectures.py`, `frontier/operators/binding.py`, the
  profile-owned operator families, `tests/unit/test_operator_query_tp_consumers.py`,
  and the fresh RED/GREEN logs
- **Identified Issues/Anomalies:** The legacy loader selected one
  `mlp_hidden_dim` for all linear operators, so a mixed Step3 dense query could
  discard its `18432` rows. The first typed implementation also surfaced the
  unrelated memory `add` alias as an ambiguity because it attempted an
  unscoped binding for every operator.
- **Remediation/Verification Code Actions Taken:** Added contract-aware width
  selection and a cache key containing the resolved typed contract. Family
  ownership now comes from the registered profile family/operator table before
  scoped binding; non-owned aliases continue through the existing resolver and
  typed-family ambiguity remains fail-fast. The predictor file and combined
  contract matrix pass (`47` and `152` tests respectively); compile and
  whitespace checks pass. No profiling data, manifest, README, worker, or
  remote state changed.

## Checkpoint 47

- **Target Component/Phase:** `SklearnMoEExecutionTimePredictor` routed-width
  consumer migration
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  `frontier/model_architectures.py`, the registered MoE operator family,
  `tests/unit/test_sklearn_moe_typed_contract.py`, existing MoE predictor
  regressions, and the direct RED/GREEN outputs.
- **Identified Issues/Anomalies:** Dataset validation and
  `_build_moe_load_imbalance_features()` read the model-wide
  `mlp_hidden_dim`. The RED fixture used `mlp_hidden_dim=9999` and a registered
  routed width of `5120`; the unmodified validator raised before TP/EP
  coverage checks because it required `expert_hidden_dim=9999`.
- **Remediation/Verification Code Actions Taken:** Added one profile-owned
  resolver that binds `moe_grouped_gemm` and returns the routed
  `ResolvedLayerContract`; both consumers use `effective_ffn_width`, while
  no-profile lightweight fixtures preserve the legacy field. The focused
  consumer test passes `2/2`, and the combined MoE predictor matrix passes
  `101/101` in `4.85s`. No model-name branch, scaling factor, data rewrite,
  accepted CSV change, or remote action was introduced.

## Checkpoint 40

- **Target Component/Phase:** Frozen-scope H200 measurement closure and
  aggregate/Step3 source-contract deferral
- **Reviewer Agent Identity:** `/root` (fresh direct verification)
- **Inspected Artifacts:**
  `/data/ycfeng/tmp/frontier_h200_remaining_model_audit_20260826.json`, the
  six accepted staging inventories, `frontier/execution_time_predictor/
  sklearn_moe_execution_time_predictor.py`,
  `frontier/entities/execution_time.py`,
  `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`, and the
  current Git refs/index
- **Identified Issues/Anomalies:** The frozen scope is complete (`6` models,
  `56` CSVs, `14,064` physical rows, `6` runtime reports, `6` completed
  requests), but the approved aggregate A guard is absent: the resolver has no
  explicit multi-layer identity argument and returns the model-level MoE result
  for `num_layers != 1`. `ExecutionTime` and the scheduler aggregate call also
  lack `layer_ids`. The Step3 strict-vLLM option-2 resolver is likewise not in
  the production tree. These are source-contract gaps, not missing measurement
  lanes.
- **Remediation/Verification Code Actions Taken:** Added
  `test_report_2026-08-28_option_a_b_deferral_and_measurement_closure.md` and
  synchronized the task documents. Marked option A as the selected future
  fail-fast contract, recorded its identity-free and dense-`18432` limitations,
  and deferred option B and the Step3 resolver to a separately approved later
  version. Preserved all measured data, manifests, remote refs, and untracked
  files; performed no production or remote change.

## Checkpoint 39

- **Target Component/Phase:** Current merge-snapshot branch-local contract
  verification and PR-readiness gate
- **Reviewer Agent Identity:** `/root` (fresh direct verification)
- **Inspected Artifacts:** Focused pytest matrix, validator caller
  `tests/performance/profiling/validate_h200_six_model_non_dummy_e2e.py:123-215`,
  routing resolver `frontier/moe_routing_runtime.py:24-35`, `MERGE_HEAD`,
  index state, synchronized refs, and H200 staging inventory
- **Identified Issues/Anomalies:** The command completed with `47 passed` and
  `10 failed` in `133.00s`. All ten failures happen during model-contract
  construction because the validator passes `simulation` by default (and one
  explicit test passes `uniform_random`), while the resolver accepts only
  `balanced`, `random`, `skewed`, and `zipf`. No CSV timing or shape assertion
  ran for those cases. The index has `0` unmerged entries, but `MERGE_HEAD`
  remains and `git diff --cached --check` reports `12` trailing-whitespace
  findings.
- **Remediation/Verification Code Actions Taken:** Classified the result as an
  integration/API blocker rather than a profiling-data defect. Preserved all
  accepted staging trees, frozen manifest, canonical files, README files,
  untracked `=10.1`, and remote refs. Did not edit source, resolve merge
  content, commit, push, create a PR, add remote comments, or re-profile.
  Final review status remains **BLOCK / REQUEST CHANGES**.

## Checkpoint 38

- **Target Component/Phase:** Live merge index whitespace and conflict gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `git diff --check`, `git diff --cached --check`,
  `MERGE_HEAD`, the three `UU` paths, and
  `/data/ycfeng/tmp/h200_live_merge_diff_check_20260828.log`.
- **Identified Issues/Anomalies:** The live checks exit `2`: 12 conflict
  markers remain in the three unmerged files, and the cached latest-main
  changes contain trailing whitespace in `base_cluster_scheduler.py`. The
  committed `origin/main...HEAD` comparison is clean but does not cover this
  merge state.
- **Remediation/Verification Code Actions Taken:** Classified the live merge
  as not reviewable/PR-ready and recorded the exact failure. No conflict or
  whitespace edit was made.

## Checkpoint 37

- **Target Component/Phase:** Validator routing-mode contract in the live
  branch snapshot
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `tests/performance/profiling/validate_h200_six_model_non_dummy_e2e.py:123-127,212-215`,
  `frontier/moe_routing_runtime.py:24-35`, and the persisted direct probe log
  `/data/ycfeng/tmp/h200_branch_contract_default_routing_recheck_20260828.log`.
- **Identified Issues/Anomalies:** The validator's default
  `moe_routing_mode='simulation'` is rejected by the current resolver, which
  accepts `balanced`, `random`, `skewed`, or `zipf`; contract construction
  raises before any CSV assertion. An explicit `balanced` probe constructs all
  six contracts, isolating the issue to the caller vocabulary.
- **Remediation/Verification Code Actions Taken:** Recorded this as an
  additional branch-integration blocker and kept the no-PR verdict. No source
  or data change was made by this review; the in-progress migration requires
  a later full-suite recheck.

## Checkpoint 36

- **Target Component/Phase:** Live merge-state contract collection and PR
  readiness
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `MERGE_HEAD`, the three unmerged paths, the focused
  pytest collection traceback, `tests/unit/test_parallel_semantics.py`, and
  the latest-main/H200 audit artifacts.
- **Identified Issues/Anomalies:** The combined focused command exits `2`
  before test execution because `tests/e2e/operator_parity/merge_profile_csv_contexts.py:10`
  contains an unresolved `<<<<<<< ours` marker. The independent resolver
  suite passes `4/4`; this is a merge-state failure, not evidence of malformed
  H200 measurements. Latest-main context/config admission and the absent
  tracked H200 payload remain independent blockers.
- **Remediation/Verification Code Actions Taken:** Recorded the collection
  failure with its exact command and preserved the **BLOCK / REQUEST CHANGES**
  verdict. No conflict resolution, source edit, data publication, PR, push,
  or remote review comment was performed.

## Checkpoint 35

- **Target Component/Phase:** Approved option-2 Step3 semantic repair and
  latest-main merge boundary
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `data/config/models/step3-moe-noquant.json`,
  `frontier/config/model_config.py`, `frontier/model_architectures.py`,
  `frontier/operators/{spec.py,families.py,binding.py}`, linear/MoE profiling
  producers, predictor/training/cache/trace consumers, the accepted Step3
  CSV/tree, runtime trace, synchronized `main`, and the focused validator
  output
- **Identified Issues/Anomalies:** The loader collapses dense and routed FFN
  dimensions; profiling rows therefore use `5120` for all Step3 linear shapes.
  Dense `mlp_*` runtime lookup selects TP1 instead of the declared Attention
  TP domain. The validator caller still uses `data_parallel_size` after the
  resolver API moved to `num_replicas`. These are contract/root-cause issues,
  not an OOM tuning problem. The merge is also paused at three conflict paths.
- **Remediation/Verification Code Actions Taken:** The maintainer-approved
  option 2 is recorded: retain 61 layers, use strict vLLM semantics, add a
  shared registry-backed resolver, preserve latest-main runtime/routing/EP/
  cache/MTP logic, and avoid hard-coded branches or scaling patches. No source
  or data was changed during the review. Implementation proceeds with merge
  resolution, RED regressions, then one resolver migration and direct runtime
  verification.

## Checkpoint 34

- **Target Component/Phase:** Branch-local H200 contract validator and
  parallel-semantics API compatibility
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:**
  `tests/performance/profiling/validate_h200_six_model_non_dummy_e2e.py:143-147`,
  `frontier/config/parallel_semantics.py:80-86`,
  `tests/unit/test_parallel_semantics.py`, and the fresh focused pytest output
- **Identified Issues/Anomalies:** The validator calls
  `resolve_frontier_parallelism_mapping` with `data_parallel_size`, while the
  current resolver accepts `num_replicas`. The focused command returns
  `43 passed, 10 failed in 9.28s`; all ten failures happen during contract
  construction before any CSV assertions. This is a branch-local incomplete
  API migration, independent of the H200 measurements and in addition to the
  latest-main consumer blockers.
- **Remediation/Verification Code Actions Taken:** Reproduced the failure,
  verified the four direct resolver tests pass with the new keyword, and
  recorded the root cause in the task issue/report files. No source, merge
  conflict, accepted data, manifest, README, untracked file, or remote state
  was changed. Verdict remains **BLOCK / REQUEST CHANGES**.

## Checkpoint 33

- **Target Component/Phase:** Corrected latest-main consumer recheck and
  import-provenance audit
- **Reviewer Agent Identity:** `/root/main_schema_audit/code_review_lane/runner_probe`
- **Inspected Artifacts:** Latest-main snapshot
  `/data/ycfeng/tmp/frontier_main_audit_20260827`, recheck script
  `/data/ycfeng/tmp/h200_origin_main_context_recheck_20260828.py`, isolated
  output `/data/ycfeng/tmp/h200_origin_main_context_recheck_20260828_isolated.json`,
  accepted H200 profile roots, and the latest-main model/config and MoE
  context implementations
- **Identified Issues/Anomalies:** An earlier probe launched from the profiling
  worktree could resolve the branch's `frontier` namespace package despite a
  snapshot `PYTHONPATH`; that invocation is not independent latest-main
  evidence. The corrected snapshot-cwd invocation imports the byte-identical
  latest-main module and still reports `96` unique gating-op failures (`32`
  each for Qwen3-235B-A22B, qwen3-a3b-30b-moe, and Mixtral), all caused by a
  request for `standalone_legacy` while accepted rows expose
  `direct`/`prefill_warmed` (or `direct`). Latest-main still lacks the Step3
  model config.
- **Remediation/Verification Code Actions Taken:** Recorded the corrected
  command, script/output checksums, import boundary, and numeric results in
  `test_report_2026-08-28_latest_main_isolated_consumer_recheck.md`. Kept the
  H200 accepted trees, frozen manifest, README files, source, untracked files,
  and remote state unchanged. The independent review result remains
  **BLOCK / REQUEST CHANGES**; no PR, push, remote comment, merge, canonical
  publication, or re-profiling was performed.

## Checkpoint 32

- **Target Component/Phase:** Independent latest-main H200 consumer matrix and
  count reconciliation
- **Reviewer Agent Identity:** `/root/main_schema_audit/code_review_lane/architect_lane/main_schema_audit`
- **Inspected Artifacts:** Fresh `origin/main` ref comparison, isolated
  latest-main snapshot `/data/ycfeng/tmp/frontier_main_audit_20260827`,
  accepted H200 CSV roots, corrected matrix JSON/log,
  `moe_gating_runtime.py`, shared prediction manager, model-config lookup,
  attention dataset contract, and `git merge-tree` output
- **Identified Issues/Anomalies:** `main` and `origin/main` both resolve to
  `a24cedabedcc7bd374073fd508dcf770c860ede5` with `0/0` divergence. Producer
  validation is `6/6` (`56` CSVs, `14,064` rows), but the latest-main matrix
  reports `96` unique independent gating-op admissions rejected by the
  `standalone_legacy` context (`3` MoE models x `16` TP/EP pairs x `2` ops).
  The matrix also records `48` repeated shared-FFN training statuses, giving
  `144` context-dependent status entries when downstream repeats are counted;
  the historical narrow probe's `64` is a smaller subset. `step3-moe-noquant`
  is absent from latest-main config, standard attention omits TP1 true-mixed
  rows (`0` versus `24` in combined input), and three merge conflicts remain,
  including the protected profiling README.
- **Remediation/Verification Code Actions Taken:** Re-vetted the independent
  report against the JSON and source-level contracts, corrected all current
  task-document count references, and kept source, accepted staging,
  canonical CSVs, frozen manifest, README files, untracked files, and remote
  refs unchanged. The branch remains `BLOCK / REQUEST CHANGES`; no PR, push,
  remote review comment, merge, or re-profiling was executed.

## Checkpoint 31

- **Target Component/Phase:** Latest-main H200 compatibility and PR gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Fresh `origin/main` ref comparison, frozen H200
  manifest and SHA-256, six accepted profile trees, six formal `2+2` runtime
  reports, latest-main MoE consumer probe, latest-main attention contract and
  H200 attention CSV row counts, Step3 model-config lookup, merge-tree output,
  README baseline, and the dedicated compatibility test report
  `test_report_2026-08-27_latest_main_compatibility_audit.md`.
- **Identified Issues/Anomalies:** `main` and `origin/main` are synchronized,
  and the branch producer/frozen validators pass (`56` CSVs, `14,064` rows),
  but latest-main rejects `96` unique MoE gating admissions in the corrected
  isolated matrix because it requests `standalone_legacy` while the data
  exposes `direct`/`prefill_warmed`. The historical narrow probe reports `64`;
  the matrix also has `48` repeated shared-FFN status failures.
  Latest-main also lacks `step3-moe-noquant.json`. Standard attention files
  parse but contain `0` TP1 true-mixed rows, while combined inputs contain
  `24`; publication therefore needs the documented supplement merge. The
  merge-tree preview has three content conflicts. The branch matches its
  recorded README baseline, but the target comparison contains a `13`-line
  `frontier/profiling/README.md` difference; resolving that conflict would
  change a protected README and violate the byte-identical gate.
- **Remediation/Verification Code Actions Taken:** Re-vetted the probe and
  row-count evidence independently, recorded the root causes and exact
  adaptation scope in the compatibility report and task docs, and left source,
  accepted staging, canonical CSVs, frozen manifest, README files, and
  preserved untracked files unchanged. The controlled data-only PR gate is
  closed; no push, PR, remote review comment, merge, or re-profiling was run.

## Checkpoint 30

- **Target Component/Phase:** Fresh H200 remaining-measurement recheck
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Frozen manifest and SHA-256, all six accepted
  profile leaf directories, all six formal runtime `offline_batch` leaf
  directories, current repository validator output, validation log, Git
  status, and namespace-scoped H200/H800 RJob/Replica queries
- **Identified Issues/Anomalies:** No formal profile or runtime E2E gap
  remains. Step3 still has the previously recorded strict vLLM semantic
  mismatch, which is deliberately deferred and is not a missing measurement.
- **Remediation/Verification Code Actions Taken:** Ran both current
  validators for all six models with exit code `0`; observed `1/1` requests
  and fixed `2+2` token counts for every runtime; confirmed the aggregate
  remains `56` CSVs and `14,064` rows and the manifest hash is unchanged;
  ran the full frozen-file helpers over `56/56` CSVs with
  `FULL_RECHECK_EXIT=0`; confirmed zero task-owned H200/H800 workers; added
  the dated test report.
  No source, accepted data, frozen manifest, README, or preserved untracked
  file changed.

## Checkpoint 29

- **Target Component/Phase:** Post-r19 remaining H200 measurement audit
- **Reviewer Agent Identity:** `/root`, `/root/capacity_gate`
- **Inspected Artifacts:** Frozen manifest and SHA-256, six accepted profile
  roots, aggregate revalidation JSON, six formal runtime JSON reports,
  remaining-model audit JSON, Git state, and namespace-scoped RJob/Replica
  inventory
- **Identified Issues/Anomalies:** No formal profiling or runtime-report gap
  remains in the six-model frozen scope. Step3 remains surface-pass and strict
  semantic-deferred because dense `mlp_*` resolves through TP1; that is a
  shared predictor contract defect rather than missing measurement data.
- **Remediation/Verification Code Actions Taken:** Revalidated `6/6` formal
  profiling (`56` CSVs, `14,064` rows), confirmed all six runtime reports are
  present and `PASS`, confirmed the manifest hash is unchanged, and kept the
  control plane worker-free. No source, accepted data, frozen manifest,
  README, or preserved untracked file changed.

## Historical/Superseded Checkpoint 2026-08-25 — deferred semantic issue and continuation audit

- **Target Component/Phase:** Step3 semantic blocker and six-model H200
  continuation gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Frozen manifest and SHA-256, five accepted H200
  staging roots, formal runtime report paths, Mixtral formal launcher roots,
  Step3 option-A semantic audit, and namespace-scoped control-plane output
- **Identified Issues/Anomalies:** The independent monolithic predictor maps
  Step3 dense-boundary `mlp_*` to TP1, so the option-A request-level PASS is not
  strict vLLM semantic acceptance. The requested cross-cutting fix is deferred.
  `mixtral_8x7b_moe` has no formal profile tree or formal E2E report.
- **Remediation/Verification Code Actions Taken:** Updated task docs and
  persisted the inventory evidence; preserved all source/profile/frozen
  artifacts; scheduled Mixtral as the next lane under the existing H200 gates.

## Checkpoint 1

- **Target Component/Phase:** Task recovery and isolated worktree setup
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** PR #21 task records, profiling workflow skill, GPU
  worker handbooks, Git worktree state, README baseline hashes
- **Identified Issues/Anomalies:** Historical task records are not present in
  the PR #21 commit tree; the primary checkout is dirty and unsuitable for new
  changes.
- **Remediation/Verification Code Actions Taken:** Read records by absolute
  path, created a clean linked worktree from `f8fea750`, and recorded nine
  README hashes.

## Checkpoint 2

- **Target Component/Phase:** Six-model config and H200 capability preparation
- **Reviewer Agent Identity:** `/root/pr21_postrepair_codespec`,
  `/root/pr21_postrepair_claude`, `/root`
- **Inspected Artifacts:** Six requested model JSONs, pinned Step3 and Mixtral
  Hugging Face sources, current `ModelConfig` loader, attention family binding,
  linear-op profiling plan, H200 predict-only output, worker environment log
- **Identified Issues/Anomalies:** Exact `step3-moe-noquant.json` was absent;
  the old alias used an outdated context; Mixtral had stale test-fixture
  provenance; the final TP axis was not frozen; an unrelated empty `=10.1`
  file appeared in the worktree.
- **Remediation/Verification Code Actions Taken:** Added the normalized exact
  Step3 config, pinned and corrected Mixtral provenance, validated all six
  configs for TP `1/2/4/8`, launched a persistent H200 worker, verified the
  CUDA 12.8 environment, kept the unrelated file outside commits, and started
  real per-model dual-measurement capability smokes.

## Checkpoint 3

- **Target Component/Phase:** H200 runtime readiness and exact attention
  workload generation
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** H200 runner and validator, model contract registry,
  generated Frontier `config.json`, request/system metrics, stage ledger,
  op traces, attention sampling generator, and focused unit tests
- **Identified Issues/Anomalies:** The two largest models exceeded single-stage
  physical capacity; a one-token prefill missed the positive prefill predictor
  contract; standard attention emitted `50` rows for `47` structural
  workloads; the shell runner initially duplicated the Python model registry.
- **Remediation/Verification Code Actions Taken:** Selected divisible `PP2`
  contracts for the two large models, fixed requests at `2+2 token`, made the
  runner consume the Python registry, validated two non-dummy fixture runs,
  and applied first-occurrence stable dedup after validity filtering. Focused
  results are `6 passed`, `47 passed`, and `git diff --check` PASS.

## Checkpoint 4

- **Target Component/Phase:** H200 formal collection recovery and naming
  compatibility gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `frontier/moe_gating_runtime.py`, focused alias
  tests, the retained H200 worker control log, partial
  `llama3.3-70b` output, and the replacement H200 predict-only log
- **Identified Issues/Anomalies:** The canonical/legacy naming contract is
  intact and `25` focused tests pass. The retained worker was platform
  preempted during the second model, leaving `118/284` attention samples and
  no complete status artifact.
- **Remediation/Verification Code Actions Taken:** Kept
  `direct`/`prefill_warmed` as producer values, kept warned
  `standalone_legacy`/`prefill_hot` aliases with the explicit future-removal
  comment, quarantined the partial output from publication, and passed a new
  8-GPU H200 predict-only request with five eligible nodes.

## Checkpoint 5

- **Target Component/Phase:** `llama3.3-70b` formal H200 collection,
  runtime-slice validator repair, and per-model E2E
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Replacement RJob and replica state, all four
  producer logs, eight accepted CSVs, exact staging validator JSON, E2E
  preflight/runtime JSON, stage ledger, op trace, focused unit tests, and Git
  diff
- **Identified Issues/Anomalies:** The old partial run was not publishable;
  fused add/norm and replicated-op split semantics caused false validator
  rejection; whole-file E2E preflight needed explicit TP/EP/context runtime
  selection.
- **Remediation/Verification Code Actions Taken:** Reran the model from zero on
  the replacement worker, validated `1,288` physical rows, filtered preflight
  rows through the runtime contract, added regression cases, recorded commit
  `7d1be838`, and completed a `2+2` PP2 non-dummy E2E with exact numeric
  evidence.

## Checkpoint 6

- **Target Component/Phase:** Cross-session handoff synchronization
- **Reviewer Agent Identity:** `/root`, `/root/pr21_postrepair_codespec`
- **Inspected Artifacts:** Current branch and HEAD, worktree status, frozen
  manifest, accepted H200 staging trees, formal E2E runtime JSONs, H200/H800
  RJob and replica queries, README baseline hashes, and all task-memory files
- **Identified Issues/Anomalies:** Plan and notes still marked the formal Qwen
  E2E as pending; issues retained resolved manifest/timestamp entries and one
  legacy H800 context value; summary still described the initial task state.
- **Remediation/Verification Code Actions Taken:** Updated the handoff records
  to `3/6` accepted H200 profiles and `3/6` formal accepted-data E2E runs,
  recorded the exact branch/HEAD and accepted-data paths, distinguished worker
  process termination from control-plane RJob finish time, and recorded the
  fresh-worker gate for `qwen3-a3b-30b-moe`.

## Checkpoint 7

- **Target Component/Phase:** Qwen3-235B-A22B formal accepted-data E2E and
  repeated next-model H200 capacity gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Qwen accepted assembly and validator JSON, formal
  runtime report, prepared `qwen3-a3b-30b-moe` launcher, fresh H200
  predict-only log, and current RJob state
- **Identified Issues/Anomalies:** Qwen3-235B-A22B formal E2E passes; no H200
  worker is currently schedulable for the next model. The wrapper's zero exit
  code does not override the body-level `no machine available` failure.
- **Remediation/Verification Code Actions Taken:** Verified `1/1` request
  with TTFT `64.99044638502953 ms`, TPOT `14.827130135476638 ms`, E2E
  `79.81757652050617 ms`; reran the exact 8-GPU predict-only request, stored
  its log at `/data/ycfeng/tmp/frontier_h200_qwen3_a3b_predict_20260824.log`,
  and suppressed any live launch. The log SHA-256 is
  `4b6a7871ca9b971b8b6431dde02855cf94a3490521a3f4ece2c18abeb03f59fa`.

## Checkpoint 8

- **Target Component/Phase:** Post-handoff validation and active-data
  governance preparation
- **Reviewer Agent Identity:** `/root`, `/root/h200_scheduler_check`,
  `/root/status_audit`
- **Inspected Artifacts:** Frozen-manifest validator output for the three
  accepted H200 staging trees, focused contract pytest log, nine README hash
  results, `git diff --check`, strict audit JSON, archive/coverage manifests,
  and independent H200 control-plane queries
- **Identified Issues/Anomalies:** Bare system Python initially failed pytest
  collection because `PYTHONPATH` was unset; H200 exact gang scheduling still
  returns semantic `no machine available`; no active H200/H800 RJob or replica
  exists. The audit identifies `37` direct archive files and `21` auxiliary
  supplements requiring deterministic merge before archive.
- **Remediation/Verification Code Actions Taken:** Recorded the verified
  `PYTHONPATH=$PWD` recipe in the shared environment handbook; reran focused
  tests with `34 passed in 8.77s`; revalidated `1,288`, `1,288`, and `3,400`
  accepted physical rows; confirmed `9/9` README hashes and clean whitespace;
  generated and validated `58` archive and `64` coverage manifest rows without
  mutating active data. Independent scheduler checks confirmed the live
  launcher remains gated and the stopped historical worker was not accessed.

## Checkpoint 9

- **Target Component/Phase:** Fresh H200 allocation gate for
  `qwen3-a3b-30b-moe`
- **Reviewer Agent Identity:** `/root`, `/root/h200_capacity_check`
- **Inspected Artifacts:** Predict-only log, live RJob YAML, queue annotations,
  live control log, replica listing, and prepared model launcher
- **Identified Issues/Anomalies:** Predict-only returned one physical H200
  candidate (`8 GPU`, `174 CPU`, `1884.6 GiB`), but the exact live RJob stayed
  `Pending` because `infer_af_test` reported `H200=0` quota remaining.
- **Remediation/Verification Code Actions Taken:** Waited through the bounded
  live allocation window, confirmed `0` replicas and `0` profiling processes,
  persisted the logs and hashes, and kept the launcher gated. No stopped
  historical worker received `brainctl exec`.

## Checkpoint 10

- **Target Component/Phase:** Fresh H200 worker and qwen3-a3b formal retry
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Exact predict-only log, environment smoke log,
  RJob/replica state, retry control log, attention and linear CUDA_EVENT logs,
  frozen manifest, and both old failed-attempt evidence trees
- **Identified Issues/Anomalies:** Capacity recovered and the new worker is
  healthy. The retry has completed attention `284/284` and linear CUDA_EVENT
  `76/76`; MoE CUDA_EVENT `direct` is still running, so no accepted directory,
  status artifact, or E2E result exists yet. Generic model-architecture
  warnings remain expected and match the independent warning audit.
- **Remediation/Verification Code Actions Taken:** Kept the live RJob
  `frontier-h200-qwen3-a3b-retry-20260824-1423` and profiling exec running,
  isolated output under the new retry root, and recorded the exact resource,
  environment, progress, and historical root-cause evidence in the task
  records. No canonical CSV or README was changed.

## Checkpoint 11

- **Target Component/Phase:** `qwen3-a3b-30b-moe` accepted profiling,
  non-dummy E2E, and H200 worker release
- **Reviewer Agent Identity:** `/root`, `/root/qwen_warning_audit`
- **Inspected Artifacts:** Frozen manifest
  `h200_exact_manifest_frozen_v3.json`, accepted `status.json` and
  `validation.json`, all ten accepted CSVs, independent profile preflight,
  E2E `preflight.json` and `runtime.json`, `request_metrics.csv`,
  `frontier_stage_batch_ledger.jsonl`, `op_traces.jsonl`, runtime log, and
  final RJob/replica YAML
- **Identified Issues/Anomalies:** Generic architecture fallback warnings,
  missing linear record-function `add`, and the vLLM fused-MoE default-config
  warning are expected non-fatal conditions. Auxiliary attention CSVs contain
  non-runtime replicated slices; the validator correctly selects TP1/EP2
  runtime tuples. No traceback, OOM, duplicate conflict, or non-finite target
  timing appeared in the accepted slices.
- **Remediation/Verification Code Actions Taken:** Confirmed all eight lanes
  complete with `3,400` accepted physical rows and frozen-validator `PASS`;
  ran the registry-derived `2+2` non-dummy E2E and confirmed `1/1` request,
  TTFT `30.99188063610683 ms`, TPOT `4.0792620915476 ms`, E2E
  `35.07114272765443 ms`, residual `0.0 ms`, two ledger rows, and 27 finite
  op-trace events. After evidence persistence, terminated only the current
  detached launcher and verified RJob/replica `Succeeded`. The stopped
  historical RJob was not accessed, and canonical CSV/README state is
  unchanged.

## Checkpoint 12

- **Target Component/Phase:** `step3-moe-noquant` H200 profiling and staging
  validation
- **Reviewer Agent Identity:** `/root`, `/root/h200_scheduler_check/step3_monitor`
- **Inspected Artifacts:** Worker-local-cache launcher and logs, Step3
  `status.json`, `validation.json`, accepted ten-CSV tree, frozen manifest,
  independent validation JSON, RJob/replica state, and fatal-signature scan
- **Identified Issues/Anomalies:** The first attempt's NFS Ninja metadata race
  remained in the preserved failed tree. Structural non-applicable NaN/zero
  timing cells appeared in auxiliary slices; a generic all-column positive
  check was rejected in favor of the module-aware validator contract.
- **Remediation/Verification Code Actions Taken:** Confirmed the retry used a
  worker-local cache and completed all seven lanes with exit code `0`; reran
  `_load_manifest`, `_validate_attention_files`, `_validate_linear_file`, and
  `_validate_moe_file`; verified `10` CSVs, `2,344` physical rows, zero
  duplicate feature rows, no infinite/negative timing values, and manifest
  SHA-256 `4df580ca1e30a007f45aeed4eb9f5d43593cbab49e59194ecabf8c5996ce8098`.
  Canonical data and README files remain unchanged; Step3 E2E is the next gate.

## Checkpoint 13

- **Target Component/Phase:** Fixed Step3 accepted-data `prefill=2,
  decode=2` non-dummy E2E retry
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Fresh E2E command output, preflight JSON, runtime
  log, accepted Step3 profile tree, active RJob/replica YAML in namespace
  `shai-core`, and H200 GPU utilization through active-replica `brainctl exec`
- **Identified Issues/Anomalies:** Preflight passes, but runtime admission
  fails before request execution with
  `parameter_memory_per_device_bytes=351956369408` and
  `requested_memory_bytes=136257837465`; no TTFT/TPOT/E2E metrics exist.
  The same values appeared in the earlier attempt, establishing a
  reproducible blocker. The worker remains `Running/Ready` but idle with all
  eight H200 GPUs at `0%` utilization and `1 MiB` used.
- **Remediation/Verification Code Actions Taken:** Re-ran the exact fixed
  contract into a fresh persistent output tree, preserved fail-fast runtime
  admission without scaling or fallback, recorded the numeric delta and log
  hash in `test_report_2026-08-24_step3_moe_noquant_non_dummy_e2e_retry.md`,
  and gated topology changes and Mixtral launch on explicit maintainer
  disposition. No canonical CSV, source code, README, or stopped historical
  worker was changed or accessed.

## Checkpoint 14

- **Target Component/Phase:** Five-model H200 accepted-staging aggregate
  revalidation
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Frozen manifest, all five accepted profile roots,
  validator calls for both measurement families, structured aggregate JSON,
  and generic-warning log
- **Identified Issues/Anomalies:** Generic architecture fallback warnings were
  emitted during model loading; they are non-fatal and do not alter validator
  results. The first attempt mixed warnings into stdout, so its JSON was not
  treated as evidence.
- **Remediation/Verification Code Actions Taken:** Re-ran the same validator,
  wrote JSON directly from the bounded command, parsed it successfully, and
  confirmed `5/5` PASS, `46` CSVs, and `11,720` physical rows. Evidence SHA-256
  is `32cbbfa08ba3cc1cf11f42a96ca53cbb70ed16d278e248eb293b6db682e89975`.
  Canonical CSVs and README files remain unchanged.

## Checkpoint 15

- **Target Component/Phase:** Step3 strict vLLM parallel-topology review gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Read-only vLLM 0.10.2 `ParallelConfig` and
  `FusedMoEParallelConfig` sources, Frontier
  `resolve_frontier_parallelism_mapping`, Step3 model config, accepted
  profiling CSV coverage, `ParamCounter` calculations, and the fixed E2E
  admission log
- **Identified Issues/Anomalies:** The fixed `PP1/AT1/ADP2/MT1/EP2` contract
  needs `351,956,369,408 B` per device against a
  `136,257,837,465 B` budget. Strict vLLM EP-on semantics expose EP as a
  boolean and flatten `TP*DP` into the expert domain; therefore the prior
  Frontier-specific `AT4/ADP2/MT2/EP4` factorization is not a vLLM-equivalent
  setting.
- **Remediation/Verification Code Actions Taken:** Recomputed four feasible
  8-GPU candidates with `PP1`, `TP*DP=8`, and EP-on. Options
  `TP8/DP1`, `TP4/DP2`, `TP2/DP4`, and `TP1/DP8` map respectively to
  `AT/ADP/MT/EP = 8/1/1/8`, `4/2/1/8`, `2/4/1/8`, and `1/8/1/8`; all pass
  parameter-only admission, with margins of `51.077`, `44.614`, `31.687`,
  and `5.834 GiB`. Option `TP4/DP2` is recommended, pending maintainer
  approval. No E2E contract, shell, validator, CSV, worker, or Mixtral state
  changed.

## Checkpoint 16

- **Target Component/Phase:** Approved Step3 option-A runtime retry gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** User-approved topology request, vLLM 0.10.2
  parallel semantics snapshot, Step3 accepted profile coverage, current
  validator/shell registry, and bounded H200 control-plane queries
- **Identified Issues/Anomalies:** The prior Step3 worker is now `Stopped` and
  its replica is absent. Option A has a static parameter shard of
  `81,413,799,936 B` (`75.823 GiB`) against the recorded
  `126.9 GiB` budget, leaving `51.077 GiB`; no A runtime has run yet. The
  layer64 plus PP path would alter model topology and remains a conditional
  fallback rather than a current implementation.
- **Remediation/Verification Code Actions Taken:** Recorded option A as the
  sole next retry (`PP1/TP8/DP1/EP-on` -> `AT8/ADP1/MT1/EP8`, one replica),
  preserved the accepted tree and manifest, and kept layer64 plus PP gated on
  an observed A admission/runtime failure. No stopped historical worker was
  queried with `brainctl exec`.

## Checkpoint 17

- **Target Component/Phase:** Mixtral formal profiling continuation and
  worker-local cache gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Frozen manifest and SHA-256, r7 launcher, r7
  predict-only evidence, RJob/replica YAML, worker-local cache probes,
  environment log, runner log, and live Attention CUDA_EVENT process tree
- **Identified Issues/Anomalies:** r5 ended by platform preemption and r6 was
  cancelled by an outer timeout while its detached launch client remained
  attached. The current r7 run is live but has not produced terminal
  validation artifacts; generic Mixtral architecture warnings are non-fatal.
- **Remediation/Verification Code Actions Taken:** Created a fresh r7 output
  root and worker-local cache, verified root and UID `10250` write access,
  started the frozen launcher, and persisted control PID/logs under
  `/data/ycfeng/tmp`. No source, accepted CSV, frozen manifest, README, or
  `=10.1` file changed. The next gate is terminal r7 status followed by the
  independent preflight, frozen validator, and formal `2+2` E2E.

## Checkpoint 18

- **Target Component/Phase:** Mixtral r7 live continuation and model-scope
  audit
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** `brainctl version`, namespace-scoped RJob/Replica
  YAML, active worker process tree, `nvidia-smi`, Attention CUDA_EVENT log,
  frozen manifest, repository model-config inventory, and H200/H800 directory
  names
- **Identified Issues/Anomalies:** Intermediate `status.json` and
  `validation.json` are absent because the runner is still in its first lane;
  this is not a terminal failure. The repository has model configurations
  outside the six-model frozen scope, so a repository-wide count must not be
  used as an implicit sampling request.
- **Remediation/Verification Code Actions Taken:** Confirmed r7 remains the
  sole active H200 worker, observed Attention progress `69/284` with
  `13,864-18,575 MiB` per GPU and no fatal signatures, and recorded the
  out-of-scope configurations without launching them. No source, CSV,
  manifest, README, or `=10.1` file changed.

## Checkpoint 19

- **Target Component/Phase:** Current Mixtral r7 continuation and Step3
  deferral record
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Current branch/HEAD and untracked-file list, frozen
  manifest, r7 RJob/Replica YAML, all available r7 lane logs, formal accepted
  profile roots, and formal E2E report paths
- **Identified Issues/Anomalies:** Mixtral r7 remains non-terminal: CUDA_EVENT
  lanes are complete, Attention KERNEL_ONLY is `47/284` at the latest read,
  and launcher/status artifacts are not yet present. Step3 option A remains
  request-level PASS but strict semantic FAIL because dense `mlp_*` used TP1
  profile rows.
- **Remediation/Verification Code Actions Taken:** Preserved the sole r7
  worker, recorded the current state, and deferred all Step3 predictor,
  registry, and layer64+PP changes. Confirmed the only remaining frozen-scope
  model work is Mixtral profiling followed by validation and `2+2` E2E.

## Checkpoint 20

- **Target Component/Phase:** Mixtral r7 live profiling continuation
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Named RJob/replica state, active replica process
  tree, H200 `nvidia-smi` output, all four existing r7 lane logs, terminal
  artifact paths, and fatal-marker scan
- **Identified Issues/Anomalies:** The runner is still non-terminal. Attention
  KERNEL_ONLY is `89/284`; `launcher_*_status.txt`, `status.json`,
  `validation.json`, and the accepted tree are not present yet. No fatal
  profiling signature was observed.
- **Remediation/Verification Code Actions Taken:** Continued the sole r7
  worker without restart or duplicate allocation, recorded the current
  progress and resource values, and kept Step3 predictor/topology work,
  canonical publication, legacy migration, and consumer admission deferred.

## Checkpoint 21

- **Target Component/Phase:** Mixtral r7 interruption diagnosis
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** RJob and Replica YAML, cleanup annotations, finish
  timestamps, persistent launcher/runner/lane logs, and fatal-marker scan
- **Identified Issues/Anomalies:** The best-effort H200 allocation was
  preempted by the platform GPU-utilization cleanup policy at `20:37:37 HKT`
  while Attention KERNEL_ONLY was `165/284`;
  terminal status and accepted-tree artifacts were never written. No
  profiling-level fatal marker was found.
- **Remediation/Verification Code Actions Taken:** Classified r7 as partial
  preemption evidence, prohibited further `brainctl exec` against its stopped
  replica, and recorded the fresh predict-only plus isolated retry sequence.

## Checkpoint 22

- **Target Component/Phase:** First post-preemption H200 retry capacity gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Predict-only raw log and SHA-256, prediction RJob
  and Replica lookup, stopped r7 RJob state, and task-owned staging roots
- **Identified Issues/Anomalies:** The exact 8-GPU H200 request returned
  semantic `no machine available` after the preemptible r7 worker was
  reclaimed; no new allocation was created.
- **Remediation/Verification Code Actions Taken:** Preserved the raw failure
  log, kept the retry launcher unstarted, and scheduled a later repeated
  predict-only gate without touching the stopped r7 replica.

## Checkpoint 23

- **Target Component/Phase:** Repeated post-preemption H200 capacity check
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Second predict-only raw log/hash, namespace RJob and
  Replica inventory, and persistent r7 partial tree
- **Identified Issues/Anomalies:** The scheduler still reports semantic `no
  machine available` after the wait; no new worker exists.
- **Remediation/Verification Code Actions Taken:** Preserved the second
  failure evidence, kept all retry launchers stopped, and deferred allocation
  until a later successful capacity gate.

## Checkpoint 24

- **Target Component/Phase:** Third post-preemption H200 capacity gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Third predict-only log, prediction-name lookup,
  namespace RJob/Replica inventory, and all r7 partial artifacts
- **Identified Issues/Anomalies:** `no machine available` persists after three
  bounded checks; no new worker exists and Mixtral cannot proceed.
- **Remediation/Verification Code Actions Taken:** Escalated the capacity
  blocker, stopped further allocation and profiling actions, and retained the
  exact retry boundary for the next authorized continuation.

## Checkpoint 25

- **Target Component/Phase:** Remaining-model audit and Mixtral r9/r10/r11
  continuation gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Frozen manifest and SHA-256, all five accepted H200
  trees, formal E2E report paths, r7/r9/r10 RJob and Replica states, r9/r10
  persistent logs, r11 predict-only log, and the three preserved untracked
  worktree files
- **Identified Issues/Anomalies:** The six-model audit leaves only
  `mixtral_8x7b_moe` incomplete (`5/6` formal profiling, `4/6` strict E2E).
  r9 was stopped in queue with `Insufficient resource`; r10 reached a Ready
  H200 replica but the sleep-worker/attached-exec lifecycle ended before any
  terminal profiling artifact; r11 returned semantic `no machine available`.
  Step3's option-A surface PASS remains strict-semantic FAIL because dense
  `mlp_*` used TP1, and its source/topology changes are deferred.
- **Remediation/Verification Code Actions Taken:** Kept r7/r9/r10 partial
  outputs outside accepted data, did not run `brainctl exec` against stopped
  replicas, recorded the exact r10/r11 hashes and commands, and synchronized
  `plan.md`, `issues.md`, `notes.md`, `progress.md`, `summary.md`, and the
  Mixtral blocker report. No source, canonical CSV, README, frozen manifest,
  or preserved untracked file changed. The next retry must pass semantic
  predict-only and run the launcher directly as the worker command.

## Checkpoint 26

- **Target Component/Phase:** Mixtral r12 H200 capacity gate and continuation
  boundary
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Exact r12 predict-only log and SHA-256, named RJob
  and Replica lookups, frozen manifest, five accepted profile trees, formal
  runtime inventory, and preserved untracked worktree files
- **Identified Issues/Anomalies:** Scheduler returned `no machine available`
  for the exact 8-GPU `infer_af_test` request. The wrapper returned `0`, but
  both prediction objects were `NotFound`; no task-owned H200/H800 worker is
  active. Mixtral remains the only incomplete model.
- **Remediation/Verification Code Actions Taken:** Added the dedicated r12
  test report and synchronized `requirements.md`, `plan.md`, `issues.md`,
  `notes.md`, `progress.md`, and `summary.md`. Kept Step3 source/registry/
  layer64+PP work deferred; did not modify source, CSVs, README files, frozen
  manifest, or preserved untracked files.

## Checkpoint 27

- **Target Component/Phase:** Mixtral r14 environment diagnosis and r15 retry
  capacity gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** r14 RJob/Replica terminal JSON, launcher status,
  environment log, frozen-runner `status.json`, Attention CUDA_EVENT traceback,
  prior successful workspace-NFS launcher, r15 launcher/hash, CPU-side
  FlashInfer cache probe, exact predict-only poll log, and namespace worker
  inventory
- **Identified Issues/Anomalies:** r14 reached `Succeeded` but its first and
  only command exited `1` before profiling because FlashInfer attempted to
  create `/mnt/host0/frontier_mixtral_jit_retry_20260826_r14`, which is not
  writable by the worker user. The corrected r15 cache probe passes locally,
  but 17 exact `8 GPU / 64 CPU / 409600 MiB` predict-only attempts still return
  semantic `no machine available`; no r15 worker exists.
- **Remediation/Verification Code Actions Taken:** Prepared the isolated r15
  launcher with all JIT caches under writable `/data/ycfeng/tmp`, verified
  `bash -n`, mode `755`, launcher SHA-256, and `FLASHINFER_IMPORT=PASS` plus
  `CACHE_WRITE=PASS`. Preserved the frozen manifest, accepted trees, README
  files, deferred Step3 source/topology decisions, and stopped-replica safety
  boundary. No `brainctl exec` was sent to r7/r9/r10 or any stopped replica.

## Checkpoint 28

- **Target Component/Phase:** Mixtral r19 formal profiling, accepted-data E2E,
  and six-model H200 completion gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** r19 launcher log and status, accepted Mixtral CSV
  tree, producer `status.json` and `validation.json`, frozen manifest and
  SHA-256, independent frozen revalidation JSON, independent preflight and
  runtime JSON reports, runtime config/ledger/op trace, six-model aggregate
  revalidation, and RJob/Replica release state
- **Identified Issues/Anomalies:** No profiling or runtime correctness issue
  remains for Mixtral. The generic model-architecture fallback warnings are
  non-fatal and the runtime validator reports the required registry-derived
  topology. Step3 remains surface-pass/strict-vLLM-semantic-deferred, and
  canonical publication, legacy migration, and consumer admission are still
  open.
- **Remediation/Verification Code Actions Taken:** Confirmed all six r19
  commands exited `0` without timeout; independently validated `10` CSVs and
  `2,344` physical rows against the frozen manifest; validated `1/1` E2E
  request with TTFT `24.30746322544336 ms`, TPOT `8.854987900952366 ms`, E2E
  `33.16245112639572 ms`, two ledger rows, and `27` op-trace events; persisted
  the new test report; released the completed r19 worker; and left source,
  README, manifest, accepted trees, and preserved untracked files unchanged.

## Checkpoint 31

- **Target Component/Phase:** User-requested H200 continuation measurement audit
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Frozen manifest and SHA-256, six accepted profile
  leaves, six formal runtime leaves, final validator log
  `/data/ycfeng/tmp/frontier_h200_remaining_measurement_recheck_20260827_session2_final.log`,
  SHA-256
  `4283831a05c555e46c155f027a9941835846c5dacf17bf6d0806c3998b512701`,
  aggregate staging revalidation JSON, and namespace-scoped RJob/Replica
  listings
- **Identified Issues/Anomalies:** The first two exploratory summary wrappers
  failed only in local aggregation: one expected helper keys that are not
  returned, and one compared the runtime contract slice with the full accepted
  tree. The underlying six profile/runtime validator calls all returned
  `PASS`. A separate shell quoting mistake expanded a backtick and ran
  no-target `brainctl exec`; it failed before selecting a pod and did not
  access a stopped replica.
- **Remediation/Verification Code Actions Taken:** Re-ran the corrected
  validator with explicit assertions. It returned
  `RECHECK_SESSION2_FINAL_EXIT=0`, covering six models, `32` contract-slice
  CSVs / `9,504` slice rows, and `56` full-tree CSVs / `14,064` full rows;
  every runtime report has `1/1` request with `prefill=2` and `decode=2`.
  Confirmed zero task-owned H200/H800 RJobs and Replicas, left all source,
  accepted data, canonical CSVs, README files, frozen manifest, and preserved
  untracked files unchanged, and kept Step3 strict semantic work deferred.

## Checkpoint 34

- **Target Component/Phase:** Approved Option 2 Step3 RCA and source-change gate
- **Reviewer Agent Identity:** `/root`
- **Inspected Artifacts:** Step3 JSON config, runtime/profiling model-config
  loaders, operator family registry/binding, predictor and shared-manager TP
  consumers, `ParamCounter`, accepted Step3 linear CSV metadata, and the
  focused mixed-layer RED test
- **Identified Issues/Anomalies:** The defect is a missing typed binding between
  layer kind, width source, and operator TP domain. The runtime loader collapses
  dense `18432` and routed/shared `5120` into one `mlp_hidden_dim`; the
  model-level `is_moe` branch then sends dense `mlp_*` to TP1. The RED test
  reproduces actual TP `1` versus expected TP `8`. No evidence supports changing
  the official `61` layers or modifying measured CSV values.
- **Remediation/Verification Code Actions Taken:** Approved the staged
  registry-backed resolver boundary: preserve `61` layers and existing runtime
  paths, add fail-fast dimension/layer/divisibility validation, migrate callers
  one family at a time, and keep accepted H200 trees, frozen manifest, README
  files, untracked `=10.1`, remote refs, and re-profiling state unchanged. The
  implementation starts with validator caller vocabulary/API cleanup and
  focused RED coverage.

## Checkpoint 35

- **Target Component/Phase:** Mixed-layer aggregate interface decision
- **Reviewer Agent Identity:** `/root` (maintainer decision recorded)
- **Inspected Artifacts:** `frontier/entities/execution_time.py` aggregate
  representation, scheduler layer-ID propagation, Step3 layer map and typed
  vLLM topology evidence, plus the previously reviewed option-A/option-B
  designs.
- **Identified Issues/Anomalies:** A scalar component vector multiplied by
  `num_layers` cannot represent a mixed dense/routed/shared FFN stage without
  layer identity. An identity-free aggregate can therefore produce a plausible
  but semantically wrong timing. Exact expansion remains a broader interface
  migration, and accepted Step3 data still lacks dense `18432` rows.
- **Remediation/Verification Code Actions Taken:** Recorded the maintainer's
  selection of option A: mixed multi-layer FFN calls require explicit
  `layer_id`/`layer_ids` and fail fast on missing or invalid identity; pure-model
  and attention-only paths remain compatible. Recorded option B and A's known
  limitations as dedicated future work. No production or accepted-data edit
  occurred during this checkpoint.

## Checkpoint 41

- **Target Component/Phase:** Independent Option-2 Step3 RCA re-review and
  pre-implementation gate
- **Reviewer Agent Identity:** `/root` (fresh direct verification)
- **Inspected Artifacts:** Current branch `HEAD=df499454ecd657f176f746046d2a404ec6a82d3`,
  synchronized `main`/`origin/main`, Option-2 requirements, Step3 JSON config,
  runtime/profiling `ModelConfig` loaders, operator registry and TP modes,
  independent predictor and layer-aware MoE predictor paths, aggregate/trace/
  `ParamCounter` consumers, accepted Step3 linear CSVs, and the focused RED
  test output
- **Identified Issues/Anomalies:** The checked-in config declares dense
  `intermediate_size=18432` and routed/shared `5120` over 61 layers, but both
  loaders expose only `mlp_hidden_dim=5120`. The predictor's model-level MoE
  branch maps dense `mlp_*` to `moe_tp=1`; the RED test observes `1` versus the
  strict Option-2 requirement `8`. Each accepted linear measurement family has
  `76` rows at width `5120` and zero rows at `18432`. A surface `2+2` E2E pass
  therefore cannot establish strict semantic correctness.
- **Remediation/Verification Code Actions Taken:** Rebuilt the causal chain
  from raw artifacts, checked topology-only and data-corruption alternatives,
  and rejected TP-only, layer64/PP, scaling, copied-row, and model-name fixes.
  Added the complete evidence ledger and corrected plan to
  `test_report_2026-08-29_option2_rca_review.md`; synchronized task docs; kept
  source, accepted CSVs, frozen manifest, README files, untracked `=10.1`,
  workers, and remote state unchanged.
- **Verdict:** **BLOCK / REVISE.** Option 2 remains the selected semantic
  direction, but implementation crosses a shared typed-layer contract and
  requires a maintainer decision on profile-owned versus dedicated registry
  ownership. Dense-`18432` H200 profiling also needs separate authorization.
  Pure dense/pure MoE and downstream consumer compatibility remain unknown
  until the approved migration and focused GREEN checks run.
