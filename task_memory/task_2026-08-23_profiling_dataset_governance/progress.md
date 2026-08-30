## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-08-29 | Completed the ReplicaStageScheduler global layer-identity propagation sub-step: PP stage calls now carry the profile-relevant global layer interval; focused RED/GREEN and mixed-layer regressions pass. |
| 2026-08-30 | Closed the profile-resolver activation gap: inactive dense contracts now fail fast for pure-MoE operator queries; the registry regression matrix passes `64/64`. |
| 2026-08-29 | Removed ParamCounter's unverified MoE/PP ratio estimate: explicit layer IDs are counted per stage and non-uniform maps fail fast; focused matrix passes `70/70`. |
| 2026-08-29 | Completed the ParamCounter typed-contract sub-step: TP dispatch now uses `TensorParallelMode` identity; RED/GREEN regression and 69-test focused matrix pass. |
| 2026-08-29 | Completed the sklearn MoE predictor consumer migration: routed dataset filtering and runtime load-imbalance features now resolve width through the profile-owned contract; focused consumer matrix passes `101/101`. |
| 2026-08-29 | Recorded the manager alias-ownership and layernorm-context fixes; focused typed-contract regressions pass `105/105`. |
| 2026-08-29 | Completed the predictor typed-width sub-step: mixed dense/shared training now resolves profile-owned widths and separates same-TP cache entries; `152` focused regressions pass. |
| 2026-08-29 | Completed the first Option-1 consumer sub-step: the sklearn predictor now resolves typed dense/routed/shared TP through the profile-owned contract; RED `1` vs `8` became GREEN with `161` focused tests passing. |
| 2026-08-29 | Maintainer selected Option-2 dedicated typed `LayerContract` ownership; synchronized docs and entered RED source implementation with the existing TP1-vs-TP8 failure preserved. |

## Verification record - ReplicaStageScheduler global layer identity 2026-08-29

- **Motivation:** The first runtime identity boundary still replaced every
  ordinary pipeline stage's global layer range with `layer_id=0` and did not
  expose the range to the predictor. A PP stage beyond stage zero could
  therefore resolve a local layer as the wrong global typed contract.
- **Expectation:** Reuse the existing
  `BaseClusterScheduler.get_pipeline_stage_layer_bounds()` helper, pass the
  contiguous global tuple as `layer_ids`, and retain its first element as the
  scalar compatibility `layer_id`. Keep PD-AF `DECODE_ATTN` and `DECODE_FFN`
  calls single-layer and unchanged.
- **Method:** Added a minimal RED test with stage `1` and four layers, then
  changed only `ReplicaStageScheduler.predict_and_create_stage()` to build the
  global interval for regular stages and forward it through the optional
  predictor kwargs. Ran:

  ```bash
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
    python3 -m pytest -q -p no:cacheprovider \
    tests/unit/test_replica_stage_scheduler_layer_identity.py
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
    python3 -m pytest -q -p no:cacheprovider \
    tests/unit/test_mixed_layer_decode_ffn_scheduling.py \
    -k 'replica_stage_scheduler or dense_ffn_batch_stage or post_routing_batch_stage'
  env PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
    frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py
  git diff --check -- frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py
  ```

- **Result:** **PASS.** The identity test reports `1 passed`; the focused
  mixed-layer scheduler subset reports `5 passed, 135 deselected`; compile and
  whitespace checks exit `0`. The recorded predictor call is
  `num_layers=4`, `layer_id=4`, `layer_ids=(4, 5, 6, 7)`. No accepted CSV,
  frozen manifest, README, worker, remote state, or protected untracked file
  changed. Predictor implementations and reconstruction paths remain the next
  source sub-step.

## Verification record - ParamCounter typed TP dispatch 2026-08-29

- **Motivation:** The profile-owned ParamCounter resolver still selected TP
  domains by comparing `TensorParallelMode.value` strings. That duplicated the
  registry's semantic enum contract and could silently misroute a domain when a
  display label changes.
- **Expectation:** Dense, routed, and shared parameter calculations select the
  declared `TensorParallelMode` member by identity while preserving all legacy
  parameter totals.
- **Method:** Added a focused RED test that temporarily changes the
  `ATTENTION_TP` backing label while configuring a Step3 counter with replica
  attention TP `1` and requesting dense TP `8`. The pre-fix path raised the
  stale-domain conflict (`8 != 1`). Replaced the string comparisons with
  `TensorParallelMode` identity checks, then ran:

  ```bash
  timeout 120s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
    /usr/bin/python3 -m pytest -q -p no:cacheprovider \
    tests/unit/test_param_counter_typed_layer_contract.py \
    tests/unit/test_param_counter_share_expert_memory_contract.py \
    tests/unit/test_model_architecture_registry.py
  ```

- **Result:** **PASS.** The focused matrix reports `69 passed in 6.41s`;
  the identity regression resolves dense TP `8`, and the existing Step3
  totals remain unchanged. `py_compile` and `git diff --check` also pass. No
  accepted CSV, frozen manifest, README, worker, untracked protected file, or
  remote reference changed.

## Verification record - ParamCounter MoE layer-map/PP boundary 2026-08-29

- **Motivation:** `_get_num_moe_layers_per_pipeline_stage()` used a rounded
  model-wide MoE ratio whenever PP was greater than one. A partial map can
  place different numbers of MoE layers in otherwise equal-sized stages, so
  the rounded value did not describe any actual device.
- **Expectation:** Count explicit `get_moe_layer_ids()` entries in each PP
  stage, accept the result only when the existing one-count API is valid for
  every stage, and fail fast for a non-uniform map or a partial count without
  layer IDs. Preserve all pure-dense, pure-MoE, and uniform-map totals.
- **Method:** Added a RED fixture with eight layers, PP2, and MoE IDs
  `0,1,2,3,4`; the old implementation returned the rounded ratio `2` despite
  actual stage counts `(4, 1)`. Replaced the ratio branch with strict ID
  validation, exact stage counting, and a descriptive stage-aware boundary.
  Ran:

  ```bash
  timeout 120s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
    /usr/bin/python3 -m pytest -q -p no:cacheprovider \
    tests/unit/test_param_counter_typed_layer_contract.py \
    tests/unit/test_param_counter_share_expert_memory_contract.py \
    tests/unit/test_model_architecture_registry.py
  ```

- **Result:** **PASS.** The matrix reports `70 passed in 6.54s`. The
  non-uniform map now raises `ValueError` with stage counts `(4, 1)`; the
  existing Step3 dense/routed/shared totals and pure-model regressions remain
  unchanged. No accepted CSV, frozen manifest, README, worker, protected
  untracked file, or remote reference changed.

## Verification record - typed manager alias and layernorm-context fixes 2026-08-29

- **Motivation:** The first profile-owned typed resolver pass exposed two
  compatibility regressions: the unqualified MEMORY `add` alias was treated
  as an FFN-family ambiguity, and the FFN typed training context leaked into
  `post_attention_layernorm`.
- **Expectation:** Non-FFN aliases retain the legacy replicated-TP path,
  genuine typed-family ambiguity remains fail-fast, and layernorm training
  keeps a contract-free identity while typed FFN models remain isolated.
- **Method:** Reviewed registered operator-family ownership before binding
  aliases, changed the manager to scope binding only for profile-owned
  families, copied the base training context for layernorm, and added focused
  regressions covering `MONOLITHIC`, `PREFILL`, and `DECODE_FFN` plus the
  typed ambiguity path. Ran:

  ```bash
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
    python3 -m pytest -q -p no:cacheprovider \
    tests/unit/test_shared_prediction_model_manager_typed_contract.py \
    tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py \
    tests/unit/test_operator_query_tp_consumers.py \
    tests/unit/test_moe_share_expert_operator_families.py
  ```

- **Result:** **PASS.** The focused matrix reports `105 passed in 4.85s`.
  `add` resolves through the legacy path for all three cluster types, typed
  family ambiguity still raises, and the layernorm context contains no
  `layer_contract`. No profiling data, manifest, README, worker, or remote
  state changed.

## Implementation checkpoint - option-1 ownership confirmed 2026-08-29

- **Motivation:** Resume the approved source-contract repair after the
  maintainer confirmed option 1 and prevent historical option-2 ownership
  wording from steering implementation toward a second registry.
- **Expectation:** Keep `ModelArchitectureProfile` as the ownership seam,
  preserve the official Step3 61-layer identity and immutable H200 evidence,
  and expose the typed layer binding through one reusable resolver.
- **Method:** Re-read the task requirements, plan, design, harness, and current
  call graph. Re-ran the focused RED regression:
  `test_mixed_moe_dense_ffn_uses_attention_tp_domain`.
- **Result:** RED remains valid and fails at the real predictor consumer with
  actual TP `1` versus expected Attention TP `8`. No production source,
  accepted CSV, frozen manifest, README, untracked file, worker, or remote ref
  was changed in this checkpoint. Documentation now marks option 1 as the
  current authority; older option-2 entries remain historical evidence.
- **Next action:** Add the profile-owned typed contract and migrate consumers
  in bounded RED/GREEN sub-steps. Dense `18432` profiling and strict Step3 E2E
  remain separately gated.

## Verification record - predictor typed contract consumer 2026-08-29

- **Motivation:** Remove the confirmed production semantic defect where a
  mixed Step3 dense `mlp_*` query used MoE TP `1` instead of Attention TP `8`.
- **Expectation:** Use the existing operator binding and profile-owned typed
  contract to resolve dense, routed, and shared domains without model-name
  branches, while preserving the PD-AF `DECODE_FFN` role mapping and all
  non-typed operator paths.
- **Method:** Added the routed/shared and DECODE_FFN regression cases to
  `tests/unit/test_operator_query_tp_consumers.py`. Connected
  `SklearnExecutionTimePredictor` to `ModelArchitectureProfile.resolve_layer_contract`
  only for profile-declared operator families; left attention/memory/MTP
  queries on the existing resolver. Ran the focused five-file command and
  `py_compile`.
- **Result:** **PASS.** The pre-change RED observed TP `1` vs expected `8`.
  The post-change matrix reports `161 passed in 6.66s`. Direct typed results
  are dense width `18432`/TP `8`, routed width `5120`/TP `1`/EP `8`, and
  shared width `5120`/TP `8`; DECODE_FFN preserves local FFN TP `8`.
- **Remediation/Verification Code Actions Taken:** Added the helper and
  focused tests, recorded the reproducible report in
  `test_report_2026-08-29_predictor_typed_contract.md`, and kept accepted
  CSVs, the frozen manifest, README files, untracked files, workers, and
  remote refs unchanged. The next sub-step is the shared prediction-manager
  migration.

## Verification record - predictor typed width consumer 2026-08-29

- **Motivation:** The predictor's linear dataframe loader still filtered every
  operator with the model-wide `mlp_hidden_dim`, which discards dense Step3
  rows when routed and dense layers use different widths.
- **Expectation:** Resolve the width from the profile-owned typed contract for
  each registered FFN family, keep same-TP dense/shared frames separate, and
  preserve the replicated memory alias path.
- **Method:** Added RED tests for mixed dense-width loading and same-TP cache
  separation. Extended `SklearnExecutionTimePredictor` with a contract
  resolver, contract-aware loader arguments, and a `(TP, contract)` dataframe
  cache key. Rechecked the `add` alias through a registry ownership test after
  the first implementation exposed an ambiguity regression.
- **Result:** **PASS.** The RED loader failed with the missing
  `operator_name` argument; the post-fix predictor file reports `47 passed`,
  and the combined architecture/manager matrix reports `152 passed`. Observed
  widths are dense `18432`, routed `5120`, and shared `5120`; memory `add`
  remains TP `1`.
- **Remediation/Verification Code Actions Taken:** Recorded the evidence in
  `test_report_2026-08-29_predictor_typed_width.md`. Kept accepted CSVs,
  manifests, README files, protected untracked files, workers, and remote
  refs unchanged. The next active sub-step remains the shared prediction
  manager's typed width/TP/signature migration.
| 2026-08-29 | Re-ran the independent Option-2 RCA checks: mixed dense FFN still resolves to TP1, both accepted Step3 linear families lack dense-18432 rows, and the source change is blocked pending shared-contract ownership and profiling authorization. |
| 2026-08-28 | Closed the six-model measurement-gap audit and superseded the implementation-start wording: option A is selected as a future fail-fast contract, option B and the Step3 option-2 resolver remain deferred, and only documentation changed in this checkpoint. |
| 2026-08-28 | Re-reviewed the approved option-2 RCA against the merged code and began the staged contract repair with a test-first boundary. |
| 2026-08-28 | Freshly reran the focused contract matrix in the current merge snapshot: `47` passed and `10` failed before CSV assertions because the validator default `simulation` routing vocabulary is rejected by the current resolver; no source, CSV, staging, or remote state changed. |
| 2026-08-28 | Corrected the merge-state interpretation: `git ls-files -u` and conflict-marker search are clean; only the active `MERGE_HEAD`, three merge-tree preview conflicts, and 12 cached trailing-whitespace findings remain as integration gates. |
| 2026-08-28 | Added a fresh direct six-model validator recheck: all contracts and profile trees pass, while runtime metadata passes only 2/6 under the current `num_replicas`/routing contract; no artifact or source was changed. |
| 2026-08-28 | Live `git diff --check` exits 2 on 12 conflict markers and staged trailing whitespace; separated the clean ref-range check from the failing merge-worktree gate. |
| 2026-08-28 | Direct contract construction exposed a second live API mismatch: validator default `moe_routing_mode=simulation` is rejected by the canonical routing resolver; recorded the failure and log hash without changing source or data. |
| 2026-08-28 | Fresh focused pytest collection stops at the unresolved `merge_profile_csv_contexts.py` conflict marker (`SyntaxError`, exit 2); the standalone parallel-semantics suite still passes 4/4. |
| 2026-08-28 | Fresh verification found a branch-local validator/API mismatch (`data_parallel_size` versus `num_replicas`): 43 focused tests pass and 10 fail before CSV assertions; recorded as an additional PR blocker without changing source or accepted data. |
| 2026-08-28 | Fresh target-timing staging audit passes (`56` CSVs/`14,064` rows, zero active timing or metadata failures); live `MERGE_HEAD` and three `UU` paths keep PR blocked. |
| 2026-08-28 | Re-ran the latest-main compatibility gates: refs remain synchronized, all six H200 profile preflights pass, the independent context probe reports 96 unique gating-op failures, Step3 config is absent upstream, and three merge conflicts keep the branch at BLOCK / REQUEST CHANGES. |
| 2026-08-27 | Rechecked latest-main ref distance and Git-tree payload: `main == origin/main`, the branch is `173` commits behind/`11` ahead, and no H200 CSV is tracked in the branch tree; PR publication remains blocked. |
| 2026-08-27 | Reconciled the latest-main MoE failure counts: 64 belongs to the historical narrow probe; the isolated full matrix has 96 unique gating-op failures, with 48 repeated shared-FFN statuses recorded as downstream manifestations. |
| 2026-08-27 | Added the non-acceptance legacy-label compatibility-export probe: `6,336` MoE rows changed only in the context column, and detached latest-main consumer slices passed after the temporary conversion and Step3 config injection. |
| 2026-08-27 | Added the attention mixed-coverage result to the latest-main audit: standard files parse but have zero TP1 true-mixed rows; combined files have 24. |
| 2026-08-27 | Audited the frozen H200 data against freshly fetched `origin/main`; producer validation passes, latest-main MoE admission fails on context names, Step3 config is missing upstream, and no PR was created. |
| 2026-08-27 | Re-ran the user-requested H200 continuation audit; separated the `32`-file contract slice from the complete `56`-file accepted trees and recorded final exit `0`. |
| 2026-08-26 | Re-audited the frozen H200 scope after the capacity update; no formal profiling or runtime E2E gap remains, so no new worker was allocated and docs moved to publication gates. |
| 2026-08-26 | Completed Mixtral r19 profiling and formal runtime E2E, independently revalidated the six-model H200 aggregate, released the worker, and recorded the remaining publication gates. |
| 2026-08-26 | Added independent r16 exact predict-only evidence and confirmed r14 resource release; Mixtral remains capacity-blocked after the cache fix. |
| 2026-08-26 | Recorded Mixtral r14's FlashInfer cache permission root cause, validated the writable-NFS r15 launcher, and preserved the current exact predict-only capacity failures. |
| 2026-08-25 | Ran r12 exact H200 predict-only; scheduler returned `no machine available`, so Mixtral remains blocked and no worker was created. |
| 2026-08-25 | Reconciled the live r9/r10/r11 Mixtral retry outcomes and closed the current continuation checkpoint as capacity/lifecycle blocked. |
| 2026-08-25 | Completed a third post-preemption H200 predict-only check; `no machine available` persists, so Mixtral execution is escalated as a capacity blocker. |
| 2026-08-25 | Repeated the H200 Mixtral predict-only gate after a resource-release wait; semantic result remains `no machine available`, with no RJob/Replica created. |
| 2026-08-25 | Ran the fresh H200 Mixtral retry predict-only gate after r7 preemption; scheduler returned `no machine available`, so no retry worker was created. |
| 2026-08-25 | Recorded the r7 platform preemption at Attention KERNEL_ONLY `165/284`; no profiling fatal error was found, and the next action is a fresh H200 predict-only gate before retry. |
| 2026-08-25 | Refreshed the Mixtral r7 live checkpoint: CUDA_EVENT lanes remain complete, Attention KERNEL_ONLY advanced to `89/284`, and the worker remains healthy; no Step3 source change was made. |
| 2026-08-25 | Updated the live Mixtral r7 checkpoint: CUDA_EVENT lanes are complete, Attention KERNEL_ONLY is progressing, and Step3 source changes remain deferred. |
| 2026-08-25 | Rechecked the live Mixtral r7 worker and six-model scope; r7 remains healthy at Attention CUDA_EVENT `69/284`, while the deferred Step3 source changes remain untouched. |
| 2026-08-25 | Recorded the r7 Mixtral H200 retry with a pre-created worker-local cache; profiling is active and Step3 source/topology work remains deferred. |
| 2026-08-25 | Recorded the deferred Step3 semantic issue and audited all six frozen models; Mixtral is the only model without formal profiling and formal E2E, so its lane is resumed. |
| 2026-08-25 | Audited the fresh strict vLLM option-A E2E: request-level runtime passed, but monolithic independent predictor used TP1 for dense `mlp_*`; recorded semantic gate and approval requirement. |
| 2026-08-25 | Maintainer approved Step3 option A (`PP1/TP8/DP1/EP-on` -> `AT8/ADP1/MT1/EP8`) for the next runtime retry; layer64 plus PP is conditional on A failure. |
| 2026-08-24 | Completed the read-only vLLM EP-on topology review for Step3; calculated four strict 8-GPU candidates, verified accepted TP/EP slices, and kept the shared E2E contract unchanged pending maintainer approval. |
| 2026-08-24 | Retried the exact H200 8-GPU predict-only recipe twice; direct exit-code evidence and semantic scheduling passed with 8 candidates and no new RJob/Replica, while the existing Step3 worker remains reserved behind the topology gate. |
| 2026-08-24 | Completed a bounded namespace-scoped Step3 control-plane recheck; RJob/replica remain Running/Ready with restart 0, owner-scoped jobs are terminal except Step3, and cluster-wide listing is Forbidden. |
| 2026-08-24 | Extended accepted-staging frozen-manifest revalidation to five H200 models; all 46 CSVs and 11,720 physical rows pass. |
| 2026-08-24 | Rechecked the Step3 control-plane state after the retry request; the sole H200 worker remains idle and no new H200/H800 lane is available, so the topology decision gate remains open. |
| 2026-08-24 | Retried the fixed Step3 accepted-data 2+2 E2E; preflight passed and the runtime reproduced the parameter-memory admission blocker before request execution. |
| 2026-08-24 | Confirmed Step3 MoE CUDA_EVENT direct `528/528` and advanced the active lane to Attention KERNEL_ONLY `17/284` with zero fatal errors. |
| 2026-08-24 | Reconciled the Step3 context count with the frozen manifest and producer rule; Step3 is direct-only by design, while prefill_warmed remains a qwen3_moe-only lane. |
| 2026-08-24 | Completed a systematic audit of the slow-looking Step3 MoE interval; source and prior accepted logs confirm expected multiprocessing/CUDA scheduling, so no restart or fix was applied. |
| 2026-08-24 | Resumed the active Step3 retry, recorded Linear `76/76` completion and MoE `direct` progress, and confirmed no fatal log signatures. |
| 2026-08-24 | Resumed the single Step3 worker-local-cache retry; verified PID/process liveness through the active replica and prepared an unstarted worker-local-cache Mixtral launcher. |
| 2026-08-24 | Diagnosed the Step3 FlashInfer NFS-JIT race and passed a four-input local-cache attention smoke before the formal retry. |
| 2026-08-24 | Re-ran the exact H200 Step3 predict-only gate, allocated a fresh worker, passed the corrected environment smoke, and started isolated Step3 profiling. |
| 2026-08-24 | Revalidated all four accepted H200 staging trees under the frozen manifest; 36 CSVs and 9,376 physical rows pass. |
| 2026-08-24 | Completed qwen3-a3b-30b-moe H200 profiling validation and formal 2+2 non-dummy E2E; released the retry worker and persisted all evidence. |
| 2026-08-24 | Added the read-only canonical audit evidence report with inventory, timing, duplicate, gating, YAML, and manifest findings; canonical data remains untouched. |
| 2026-08-24 | Recorded a semantic H200 predict-only candidate followed by a live `infer_af_test` quota rejection; no worker or profiling process started. |
| 2026-08-24 | Generated and validated the exact archive and coverage manifests without mutating active data; all 58 archive sources and 64 coverage sources exist with zero collisions. |
| 2026-08-24 | Re-ran the nine-file README hash gate from the primary Frontier root after correcting the relative-path invocation; all files match, and `git diff --check` is clean. |
| 2026-08-24 | Revalidated all three accepted H200 staging trees with the frozen-manifest validator and ran the focused contract pytest with `PYTHONPATH=$PWD`; both gates pass. |
| 2026-08-24 | Added an independent control-plane recheck: all historical H200 jobs are terminal, no H800 jobs are present, and a uniquely named predict-only retry again returned `no machine available`. |
| 2026-08-24 | Repeated the exact H200 predict-only gate after the session handoff; the scheduler still returned `no machine available`, and no worker or profiling artifact was created. |
| 2026-08-24 | Re-ran the exact 8-GPU H200 predict-only gate for `qwen3-a3b-30b-moe`; semantic result remains `no machine available`, so no live worker was created. |
| 2026-08-24 | Completed Qwen3-235B-A22B accepted-data preflight and matching 2+2 non-dummy E2E; reconciled the replacement worker as stopped. |
| 2026-08-23 | Recorded Qwen partial-lane reuse audit, the unique replacement H200 worker, and targeted KERNEL_ONLY recovery. |
| 2026-08-23 | Recorded H200 worker preemption, preserved the partial second-model output as failure evidence, and passed the replacement predict-only gate. |
| 2026-08-23 | Recorded the first formal H200 model revalidation, validator root-cause repair, 2+2 E2E result, and second-model launch. |
| 2026-08-23 | Completed and verified the registry-backed canonical naming and warned legacy-alias implementation. |
| 2026-08-23 | Confirmed option A and started the warned legacy-alias TDD checkpoint. |
| 2026-08-23 | Froze the H200 numeric envelope, reconciled worker cleanup state, and opened the evidence-backed gating-context naming decision. |
| 2026-08-23 | Closed the H200 E2E runtime contract and standard-attention stable-dedup checkpoints. |
| 2026-08-23 | Normalized the missing Step3 config, corrected Mixtral provenance, and brought a persistent 1-GPU H200 worker to a validated profiling environment. |
| 2026-08-23 | Confirmed the six H200 models and started config, worker, and per-model E2E preparation lanes. |
| 2026-08-23 | Halted H200 execution before model sampling because the model set was still provisional. |
| 2026-08-23 | Switched execution to the H200-only lane and deferred H800 by maintainer instruction. |
| 2026-08-23 | Recorded the H800 quota blocker and stopped all execution at the GPU allocation gate. |
| 2026-08-23 | Re-ran the strict audit at the clean head and confirmed current H800 token/routing gaps. |
| 2026-08-23 | Initialized execution and created the clean data-governance worktree. |

# Progress

## Verification record - live merge whitespace gate 2026-08-28

- **Motivation:** Verify the actual index/worktree that a future merge or PR
  would use, rather than relying only on the committed ref-range comparison.
- **Expectation:** Both working-tree and cached diffs should be free of conflict
  markers and whitespace errors before merge completion.
- **Method:** Ran `git diff --check` followed by `git diff --cached --check`
  and persisted the output at
  `/data/ycfeng/tmp/h200_live_merge_diff_check_20260828.log`.
- **Result:** **FAIL, exit 2.** The worktree reports 12 leftover conflict
  markers across the three `UU` paths. The cached diff reports trailing
  whitespace in `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`
  at lines `1008, 1013, 1017, 1023, 1026, 1030, 1034, 1040, 1044, 1065,
  1088, 1099`. The earlier `git diff --check origin/main...HEAD` remains
  clean because it compares committed refs and does not inspect the live merge
  index.
- **Remediation/Verification Code Actions Taken:** Recorded the layered
  whitespace result and kept the merge/PR gate closed. No conflict resolution,
  whitespace cleanup, source edit, data publication, or remote action was
  performed.

## Verification record - routing-mode contract gate 2026-08-28

- **Motivation:** Check the current validator contract directly after the
  `num_replicas` caller edit appeared in the shared worktree.
- **Expectation:** `build_model_contract()` should reach profile assertions
  using the routing vocabulary exported by the current resolver.
- **Method:** Ran a bounded Python probe that calls
  `build_model_contract('llama3.1-8b')` with its default routing mode and
  persisted the output at
  `/data/ycfeng/tmp/h200_branch_contract_default_routing_recheck_20260828.log`
  (SHA-256
  `cef046eccc6581768526067ddfaf3a392f3283c559d92004748330a4a77b2633`).
- **Result:** **FAIL before CSV assertions.** The validator default is
  `moe_routing_mode='simulation'`, while
  `frontier/moe_routing_runtime.py:24-35` accepts canonical distribution
  values `balanced`, `random`, `skewed`, and `zipf`; the call raises
  `ValueError: Unsupported moe_routing_distribution_type='simulation'`.
  Passing `moe_routing_mode='balanced'` reaches and constructs all six model
  contracts, so the failure is a caller-vocabulary migration gap in the live
  snapshot, not malformed H200 data.
- **Remediation/Verification Code Actions Taken:** Recorded the exact
  failure and retained the no-PR boundary. Re-run the full contract suite
  only after the in-progress source migration settles; no source, accepted
  staging tree, canonical CSV, manifest, README, untracked file, or remote
  state changed in this check.

## Verification record - merge-conflict collection gate 2026-08-28

- **Motivation:** Re-run the focused branch contracts against the live
  `MERGE_HEAD` state before using them as a PR baseline.
- **Expectation:** The focused contract modules should collect; the
  independent parallel-semantics module should remain runnable.
- **Method:** Ran the combined focused command with
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD` and then ran the standalone
  `tests/unit/test_parallel_semantics.py` command.
- **Result:** The combined command exits `2` during collection because
  `tests/e2e/operator_parity/merge_profile_csv_contexts.py:10` still contains
  the unresolved `<<<<<<< ours` marker. The independent command exits `0`
  with `4 passed in 0.90s`. The prior `43 passed, 10 failed` result remains
  historical evidence from before the conflict became syntactically active;
  neither result invalidates accepted CSV rows, while the live merge state
  prevents a complete branch contract run.
- **Remediation/Verification Code Actions Taken:** Recorded the exact
  collection failure and retained the no-PR boundary. No source, accepted
  staging tree, canonical CSV, manifest, README, untracked file, or remote
  state changed.

## Review and RCA record - approved option 2 2026-08-28

- **Motivation:** The maintainer selected option 2 after the latest-main
  compatibility review and requested an evidence-based RCA before source
  changes.
- **Expectation:** The repair must preserve Step3's official 61-layer model,
  use strict vLLM parallel semantics, share one registry-backed resolver across
  producers and consumers, and avoid hard-coded model branches or scaling
  patches.
- **Method:** Re-read the task requirements and design records; inspected the
  Step3 JSON, `BaseModelConfig` loader, operator-family registry, linear/MoE
  profiling producers, predictor/training/cache/trace callers, and the latest
  runtime trace. Reproduced the branch-local validator/API mismatch and
  compared the branch against synchronized `main`.
- **Result:** The RCA is confirmed. `step3-moe-noquant.json` declares
  `intermediate_size=18432`, `moe_intermediate_size=5120`,
  `share_expert_dim=5120`, and `num_hidden_layers=61`; the loader currently
  stores only the MoE width (`5120`) as `mlp_hidden_dim`, causing dense layers
  to select the routed-expert shape. The accepted Step3 linear CSV consequently
  has 76 rows all at `n_expanded_embd=5120`; the runtime trace selects TP1 for
  dense `mlp_*` where strict vLLM semantics require Attention TP8. The focused
  validator also reports `43 passed, 10 failed` because its caller still sends
  `data_parallel_size` to a resolver that accepts `num_replicas`. No source,
  CSV, manifest, README, untracked file, or remote state was changed in this
  review step.
- **Remediation/Verification Code Actions Taken:** Recorded the approved
  scope and execution order in `requirements.md`, `plan.md`, `issues.md`,
  `review.md`, and `design.md`. The next sub-step is merge resolution followed
  by RED tests.

## Verification record - fresh branch-local contract test 2026-08-28

- **Motivation:** Confirm that the current profiling branch can execute its own
  H200 contract validator before using its results as a PR baseline.
- **Expectation:** The validator and its parallel-semantics helper should share
  one callable signature, and all focused contract tests should complete.
- **Method:** Ran
  `timeout 300s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 -m pytest -q -p no:cacheprovider tests/unit/test_moe_gating_runtime_context_aliases.py tests/unit/test_operator_parity_profile_context_merge.py tests/unit/test_h200_six_model_non_dummy_e2e_contract.py`.
  Independently ran `tests/unit/test_parallel_semantics.py` and inspected the
  resolver and validator call sites.
- **Result:** **FAIL for branch-local verification.** The focused run reports
  `43 passed, 10 failed in 9.28s`. Every failure raises
  `TypeError: resolve_frontier_parallelism_mapping() got an unexpected keyword
  argument 'data_parallel_size'` at
  `tests/performance/profiling/validate_h200_six_model_non_dummy_e2e.py:143-147`.
  The resolver now accepts `num_replicas` at
  `frontier/config/parallel_semantics.py:80-86`; the four direct resolver tests
  pass, so this is an incomplete caller migration. The failures occur before
  CSV row assertions and do not change the independent frozen-data result.
- **Remediation/Verification Code Actions Taken:** Added the failure to
  `issues.md`, `review.md`, and the final verification report. Kept source,
  accepted H200 trees, frozen manifest, README files, untracked files, and
  remote state unchanged. A cross-module API repair remains outside this audit
  boundary and requires maintainer approval.

## Historical/Superseded checkpoint - Mixtral r7 and deferred Step3 2026-08-25 20:05 +08:00

- **Motivation:** Record the maintainer-requested Step3 deferral and the
  current state of the only incomplete frozen-scope model before waiting for
  terminal profiling artifacts.
- **Expectation:** No Step3 source, topology, registry, accepted CSV, frozen
  manifest, README, or preserved untracked file changes occur; Mixtral r7
  remains the sole active model lane.
- **Method:** Queried the namespace-scoped RJob/Replica, inspected the
  persistent r7 logs, scanned fatal signatures, and compared the six frozen
  model identities with accepted profile roots and formal E2E report paths.
- **Result:** **IN PROGRESS.** HEAD is
  `9b24c7c0c11cea991277c749e44c9d6df2a6e2e5`; the worktree has exactly three
  untracked files: `=10.1`,
  `outputs/metrics/step3_moe_noquant/offline_batch/run_2026_08_25_13_16_18_964917/config.json`,
  and `tests/performance/profiling/launch_step3_a_e2e_worker.sh`. The r7 RJob
  `frontier-h200-mixtral-20260825-r7` is `Running/Ready` on
  `gpu-h200-0234.lgcm.sh.istep.fun`. Attention CUDA_EVENT is `284/284`,
  Linear CUDA_EVENT is `76/76`, MoE CUDA_EVENT direct is `528/528`, and
  Attention KERNEL_ONLY is `29/284` at the latest read. Its launcher,
  `status.json`, and `validation.json` are not terminal yet. Fatal-marker
  scan count is `0`. The six-model audit identifies Mixtral as the only
  incomplete profiling/E2E lane; Step3 remains surface PASS / strict semantic
  FAIL and is deferred to a later task.

## Validation record - exact H200 predict-only retry 2026-08-24 22:06 +08:00

- **Motivation:** Recheck H200 scheduling with the complete verified resource
  recipe after the Step3 E2E admission blocker, without creating a duplicate
  worker.
- **Expectation:** The semantic predict-only result reports eligible H200
  capacity, uses `8 GPU / 64 CPU / 409600 MiB` and `--backoff-limit=1`, and
  creates no RJob or Replica.
- **Method:** Ran the bounded `/kubebrain/rlaunch --predict-only` command
  named `frontier-h200-predict-retry-20260824-2206` in namespace `shai-core`
  with output redirected directly so the tool result captured the scheduler
  command's exit code rather than a pipe consumer's exit code.
  Persisted the raw output at
  `/data/ycfeng/tmp/frontier_h200_predict_retry_20260824_2206.log` and then
  queried the prediction name and the existing Step3 RJob read-only.
- **Result:** **PASS.** The wrapper returned `0` and the semantic output
  listed `8` H200 candidates (`8 GPU` each; CPU values `174/174/174/174/174/
  142/134/134`, memory `1884.6/1884.6/1884.6/1884.6/1884.6/1634.6/1738.1/
  1738.1 GiB`). The prediction RJob and Replica were both `NotFound`; no new
  allocation or profiling process started. The existing Step3 worker remains
  `Running/Ready` and reserved until the topology disposition. The preceding
  same-parameter 22:02 check returned the same candidate count and allocation
  result. Report: `test_report_2026-08-24_h200_predict_only_retry_2202.md`.

## Status

- **Completed:** recovered prior profiling audit and GPU-gate evidence.
- **Completed:** read the profiling workflow and H800/H200 worker recipes.

## Review record - Step3 strict vLLM topology options 2026-08-24

- **Motivation:** The fixed Step3 E2E uses `PP1`, Attention `TP1/DP2`, and
  MoE `TP1/EP2`; its deterministic parameter shard is
  `351,956,369,408 B` against a `136,257,837,465 B` H200 budget. The
  maintainer requested vLLM parallel semantics with TP and EP enabled before
  another runtime attempt.
- **Expectation:** Enumerate every one-node 8-GPU factorization
  `TP*DP=8` under vLLM `enable_expert_parallel=true`, map it to Frontier's
  `AT/ADP/MT/EP` fields, and verify parameter memory plus accepted CSV axes
  without changing source or launching a worker command.
- **Method:** Read the pinned vLLM 0.10.2 source snapshot, used
  `resolve_frontier_parallelism_mapping()` and `ParamCounter` on the checked-in
  Step3 config, and inspected the immutable accepted tree at
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_step3_moe_noquant_2ecc496b_20260824_retry_localcache_1807/step3-moe-noquant/accepted/step3-moe-noquant`.
  The four candidates were `TP8/DP1`, `TP4/DP2`, `TP2/DP4`, and `TP1/DP8`,
  all with effective EP8 and `PP1`.
- **Result:** **PASS / review gate remains open.** Parameter shards are,
  respectively, `81,413,799,936 B` (margin `51.077 GiB`),
  `88,353,800,192 B` (margin `44.614 GiB`),
  `102,233,800,704 B` (margin `31.687 GiB`), and
  `129,993,801,728 B` (margin `5.834 GiB`). Attention and Linear files cover
  TP `{1,2,4,8}`; each MoE measurement file covers all 16 TP/EP pairs and
  has 33 direct rows for the required strict `MT1/EP8` slice. The analysis is
  recorded in `step3_topology_analysis.md`. No E2E contract, CSV, worker, or
  Mixtral launcher changed or ran.

## Operational checkpoint - Step3 retry resumed 2026-08-24 18:33 +08:00

- **Motivation:** Continue the interrupted H200 collection without creating a
  duplicate runner or losing the exact cache-race mitigation.
- **Expectation:** The existing Step3 process remains the only profiling lane,
  writes to its new staging root, and advances without the shared-NFS Ninja
  failure; the next Mixtral launcher is prepared but remains stopped.
- **Method:** Read the persistent Attention CUDA_EVENT log and queried the
  active replica read-only. Verified PID `8693`, its Attention child, and
  worker-local `ninja`/`nvcc` processes. Created and syntax-checked the
  unstarted launcher at
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_mixtral_8x7b_moe_2ecc496b_20260824_retry_localcache_20260824/launch_model.sh`.
- **Result:** **IN PROGRESS.** Step3 Attention reached `114/284`; the RJob
  `frontier-h200-step3-20260824-1815` remains `Running/Ready`, with no
  `Traceback`, Ninja error, CUDA OOM, or `Killed` marker. The Mixtral launcher
  passes `bash -n`, has SHA-256
  `b543536b2e121545c3de0c4b27b993be7740cec0236383bd5e8ec85fbbdfbcd1`, and
  has not created a model output directory.

## Validation record - Step3 H200 allocation and launch 2026-08-24 17:15 +08:00

- **Motivation:** Resume the next confirmed H200 model only after the exact
  frozen resource envelope passes a semantic capacity check and a real worker
  reaches `Ready`.
- **Expectation:** `infer_af_test` with H200, 8 GPUs, 64 CPUs, and 409600 MiB
  reports eligible nodes; the new worker exposes eight H200 devices and the
  pinned profiling stack; the Step3 launcher writes only to its own staging
  root.
- **Method:** Ran the bounded exact predict-only request with
  `--gpu=8 --cpu=64 --memory=409600 --predict-node-num=10`. The wrapper exit
  was `0` and the semantic output listed `10` H200 candidates, each with
  `8 GPU`, `174 CPU`, and `1884.6 GiB`. Launched RJob
  `frontier-h200-step3-20260824-1815`; it transitioned from `Pending` with two
  transient no-machine messages to `Running/Ready` on replica
  `frontier-h200-step3-20260824-1815-cfcd2084`. Ran two corrected bounded
  environment probes, then executed the independent Step3 launcher copied
  from the verified frozen runner.
- **Result:** The first probe failed because it did not `cd` to the source
  checkout, so the relative model-config lookup could not find
  `step3-moe-noquant.json`. The second probe reached the full stack but used a
  nonexistent diagnostic attribute (`ModelConfig.model_name`). The final
  probe passed: Python `3.12.3`, torch `2.8.0+cu128`, CUDA `12.8`, vLLM
  `0.10.2`, FlashInfer `0.3.1.post1`, CUDA available, device count `8`, and
  Step3 `model_type=step3_text`, `num_experts=48`, `num_experts_per_tok=3`.
  The Step3 producer is now running Attention CUDA_EVENT under a separate
  staging root; no accepted CSV, canonical CSV, README, or legacy move has
  occurred.
- **Persistence:** Predict-only log SHA-256
  `d26a57031a44acb76daf699853afb626897512fd0ebac3687ae8758c0d4ef669`;
  worker launch log SHA-256
  `442339b4f59aac6510a6fb21c92adab99f8bcca6e672c06616f0c606b181bc60`;
  final environment log SHA-256
  `f1dca19d551b72ef23a0576ca72bdd41743f1c86c89d395c9e1da0b41f5681b1`;
  Step3 launcher SHA-256
  `1fbac8952ba4ef957ad8b86a467193626c26e1261147e1b245c05e436c800877`.

## Validation record - fresh H200 live quota gate 2026-08-24 13:11 +08:00

- **Motivation:** Recheck the next-model allocation after the session resumed;
  the prepared `qwen3-a3b-30b-moe` launcher must remain gated until a real
  eight-GPU worker exists.
- **Expectation:** The exact H200 `infer_af_test` request first reports
  schedulable capacity, then creates a Ready worker before any profiling
  process starts.
- **Method:** Ran the bounded predict-only request with `--gpu=8`,
  `--cpu=64`, `--memory=409600`, `--predict-node-num=10`, and then attempted a
  detached live RJob with the same resource envelope. Queried the new RJob
  YAML and replica list read-only; never executed against the stopped
  historical worker.
- **Result:** Predict-only reported candidate
  `gpu-h200-0703.lgcm.sh.istep.fun` (`8 GPU`, `174 CPU`, `1884.6 GiB`). The
  live RJob `frontier-h200-qwen3-a3b-20260824-1311` stayed `Pending` and was
  stopped after `119` seconds. Queue evidence reported `Insufficient GPU
  quota` and `Queue remaining: H200=0`; replica count and profiling process
  count were both `0`. The launcher therefore did not run. Full numeric and
  hash evidence is in
  `test_report_2026-08-24_h200_qwen3_a3b_live_quota_gate.md`.

## Validation record - focused contract regression recheck 2026-08-24

- **Motivation:** Confirm that the documentation-only handoff updates and
  current shared worktree state did not regress the existing profiling
  contracts.
- **Expectation:** The established alias, six-model E2E contract, and output
  contract tests remain green under the documented local import path.
- **Method:** Ran
  `timeout 120s env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 pytest -q
  -p no:cacheprovider tests/unit/test_moe_gating_runtime_context_aliases.py
  tests/unit/test_h200_six_model_non_dummy_e2e_contract.py
  tests/unit/test_profiling_output_contract.py`.
- **Result:** **PASS.** `34 passed in 4.88s`, exit code `0`.
- **Completed:** created the clean worktree at PR #21 head `f8fea750`.
- **Completed:** recorded hashes for all nine README files.
- **Completed:** selected canonical MoE gating-context values `direct` and
  `prefill_warmed`.
- **Completed:** registry-backed `direct` / `prefill_warmed` canonical output,
  temporary `standalone_legacy` / `prefill_hot` aliases, visible removal
  warnings, legacy CSV/suffix normalization, and unknown-value fail-fast.
- **Completed:** generated and froze the H200 exact manifest with canonical
  context labels.
- **Blocked:** H800 `codesign` GPU quota is full; predict-only reports
  `129/128`, so the required parallel real-GPU collection cannot start.
- **Confirmed:** six H200 model identities are fixed by the maintainer.
- **Completed:** all six requested JSON identities now resolve exactly; the new
  `step3-moe-noquant` identity uses the official 65,536-token Step3 text
  contract.
- **Completed:** H200 predict-only and the real-worker environment smoke.
- **Completed:** six-model non-dummy E2E runner/validator contract; the two
  large-model fixtures pass with registry-selected `PP2` and fixed `2+2 token`
  requests.
- **Completed:** standard-attention stable dedup; the bounded H200 manifest
  basis now emits `47/47` unique rows instead of `50/47`.
- **Completed:** all `224/224` H200 axis-legality cells and all `32/32`
  six-model capability cells.
- **Completed:** Qwen3 prefill-context smoke, `8/8` cells at source
  `08b70647`.
- **Completed:** numeric profiling envelope frozen at `10,656` logical rows.
- **Confirmed:** every per-model E2E uses the minimum legal `2+2 token`
  request.
- **Completed:** formal profiling and matching `2+2` non-dummy E2E for
  `llama3.1-8b` and `llama3.3-70b`.
- **Completed:** `Qwen3-235B-A22B` targeted recovery. Five reused lanes passed
  the direct reuse audit; Linear KERNEL_ONLY completed `76/76`; both MoE
  KERNEL_ONLY contexts completed `528/528`; assembly contains `10` accepted
  CSVs and `3,400` physical rows.
- **Completed:** Qwen3-235B-A22B accepted-data preflight and matching
  non-dummy E2E. The fixed request used `prefill=2`, `decode=2`, PP2,
  ATTN TP1/DP2, and MoE TP1/EP2.
- **Observed:** The formal accepted-data E2E for Qwen3-235B-A22B completed
  `1/1` request with TTFT `64.99044638502953 ms`, TPOT
  `14.827130135476638 ms`, and E2E `79.81757652050617 ms`.
- **Blocked:** The replacement RJob
  `frontier-h200-qwen-resume-20260823-1` is now `Stopped` with
  `finishTime=2026-08-23T21:53:15Z`; no live H200 worker remains for the next
  model.
- **Deferred:** H800 execution until the maintainer explicitly resumes it.
- **Completed:** `qwen3-a3b-30b-moe` formal profiling. All eight profiling
  lanes completed (`284/284` attention, `76/76` linear, and `528/528` for
  each MoE context in both measurement families); the accepted tree contains
  `10` CSVs and `3,400` physical rows and passes the frozen-manifest validator.
- **Completed:** `qwen3-a3b-30b-moe` accepted-data preflight and matching
  non-dummy E2E. The fixed request used `prefill=2`, `decode=2`, PP1,
  ATTN TP1/DP2, and MoE TP1/EP2 with the analytical backend.
- **Observed:** The formal qwen3-a3b E2E completed `1/1` request with TTFT
  `30.99188063610683 ms`, TPOT `4.0792620915476 ms`, E2E
  `35.07114272765443 ms`, and residual `0.0 ms`. The stage ledger has `2`
  rows (`0.64566418 ms` prefill and `0.09501359 ms` decode); the op trace has
  `27` finite-duration events from `0.0007184533333333333` to
  `1.72269355 ms`.
- **Completed:** The retry RJob and replica now report `Succeeded` after the
  launcher status was written; no active H200/H800 worker remains. The old
  stopped RJob `frontier-h200-qwen-resume-20260823-1` was not accessed.
- **Pending:** `step3-moe-noquant` and `mixtral_8x7b_moe` profiling,
  validation, and 2+2 E2E; then deterministic publication, legacy migration,
  consumer admission, and sub-step commits.

## Modification record — isolated task baseline

- **Motivation:** Keep dataset governance independent from the dirty primary
  checkout and from open PR #20/#21 worktrees.
- **Expectation:** Start from the exact locally approved PR #21 head with no
  inherited file changes.
- **Method:** Verified `.worktrees` is ignored, created
  `data/profiling-governance-h800-h200-20260823` from `f8fea750`, and captured
  SHA-256 hashes for all README files.
- **Result:** The new worktree reports a clean branch at `f8fea750`; nine README
  baseline hashes are stored under `/data/ycfeng/tmp`.

## Validation record — clean-head strict audit and grid probe

- **Motivation:** Reproduce the prior audit against the only worktree that will
  publish the governed datasets.
- **Expectation:** Confirm the exact active/archive counts and identify missing
  rows using the current PR #21 sampling functions rather than old assumptions.
- **Method:** Ran
  `/data/ycfeng/tmp/frontier_profiling_dataset_strict_audit_20260823.py` with
  the clean worktree as both code root and active data root; directly called
  `get_num_tokens_to_profile()` and inspected H800 feature-axis distributions
  with pandas.
- **Result:** `104/104` CSVs parsed; `37` files require archive; `46` direct
  files pass current admission; `21` files are supplements. Current
  `get_num_tokens_to_profile(128)` returns `19` points and
  `get_num_tokens_to_profile(64)` returns `11` points. Existing small H800
  linear files contain only `8` powers-of-two points, and existing H800 MoE
  files contain only `7` powers-of-two points. All five H800 MoE models contain
  only `uniform_topk`; the current runtime-default
  `standard_fused_topk` rows are absent.

## Design record — current option-A tuple basis

- **Motivation:** Freeze a meaningful bounded envelope that matches the current
  PR #21 generators and real runtime selectors.
- **Expectation:** Required tuples cover automatic endpoint-inclusive axes,
  while legacy extra points remain usable and do not force remeasurement.
- **Method:** Re-activated Serena on the clean worktree, read the current
  attention/MoE generators and predictor filters, and directly enumerated the
  `max_seq_len=max_model_len=128` attention grid.
- **Result:** Per TP and measurement family, the current attention envelope has
  `50` standard rows and `24` automatic true-mixed rows. The Qwen3-MoE runtime
  can select both `standalone_legacy` and `prefill_hot` gating contexts;
  other model families require `standalone_legacy`. Runtime mode
  `simulation` selects `standard_fused_topk`; uniform modes select
  `uniform_topk`.

## Validation record — H800 predict-only quota gate

- **Motivation:** Confirm that a real H800 worker can be allocated before
  starting any kernel smoke or profiling collection.
- **Expectation:** The full-shape `codesign` predict-only request passes quota
  validation and returns at least one eligible H800 candidate.
- **Method:** Ran a bounded one-GPU request with
  `--charged-group=codesign --private-machine=group --positive-tags=h800
  --backoff-limit=1`; inspected both process status and semantic output.
- **Result:** **FAIL / BLOCKED.** The output reports
  `gpu : 129/128; current value + has used value: 129; total value: 128`.
  The wrapper returned exit code `0`, so semantic output is required for the
  verdict. No live RJob, replica, GPU environment capture, or profiling row
  was created. Evidence is stored under
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h800/`.

## Modification record — six-model config normalization

- **Motivation:** The exact requested `step3-moe-noquant.json` was absent, the
  closest local alias declared an outdated 8,192-token context, and the local
  Mixtral file still described itself as a non-production test fixture.
- **Expectation:** Every requested model name resolves directly through the
  current loader with official structural values, BF16 precision, and no
  hidden alias substitution.
- **Method:** Downloaded the pinned Step3 `config.json`,
  `configuration_step3.py`, and `modeling_step3.py` through the company HTTPS
  proxy; normalized the official Step3 `text_config`; downloaded and compared
  the pinned official Mixtral config; recorded source revisions and checksums
  in each normalized JSON.
- **Result:** `step3-moe-noquant` resolves as profile `step3_text`, context
  `65536`, Q/KV heads `64/1`, head dimension `256`, experts/top-k `48/3`,
  BF16, and quant signature `none`. `mixtral_8x7b_moe` resolves as profile
  `generic`, context `32768`, Q/KV heads `32/8`, experts/top-k `8/2`, BF16,
  and quant signature `none`. Direct loader assertions passed, and the focused
  existing tests report `8 passed`.

## Validation record — persistent H200 environment

- **Motivation:** Confirm the real H200 worker can execute the exact CUDA
  profiling stack before spending time on six-model collection.
- **Expectation:** A persistent worker reaches Ready and imports the pinned
  torch/vLLM/FlashInfer stack against the CUDA 12.8 toolkit.
- **Method:** Launched
  `frontier-h200-env-20260823-1723` with quota group `infer_af_test`, one H200,
  eight CPUs, and 65,536 MiB memory; executed the environment probe through
  `brainctl exec`.
- **Result:** Worker `frontier-h200-env-20260823-1723-cfcd2084` is Running on
  `gpu-h200-0745`. Observed H200 memory `143771 MiB`, driver `570.124.06`,
  compute capability `9.0`, Python `3.12.3`, torch `2.8.0+cu128`, vLLM
  `0.10.2`, FlashInfer `0.3.1.post1`, and nvcc `12.8.61`; CUDA availability is
  true.

## Modification record — H200 non-dummy E2E runtime contract

- **Motivation:** The single-stage runtime fixture exceeded physical weight
  capacity for the two largest models and a one-token prefill could not close
  the attention predictor's positive-prefill contract.
- **Expectation:** Derive PP from one model registry, exercise a legal
  positive prefill/decode request, and fail when produced runtime artifacts
  depart from that contract.
- **Method:** Added `num_pipeline_stages` to `ModelContract`; selected `PP2`
  for `llama3.3-70b` and `Qwen3-235B-A22B`; fixed the request at `2+2 token`;
  added a serial runner, config validation, and bounded Random Forest settings.
- **Result:** Focused tests report `6 passed`; shell syntax and two-model
  dry-run pass. Non-dummy fixture runs complete `1/1` requests for both large
  models. `llama3.3-70b` reports TTFT `41.70383707320797 ms`, TPOT
  `37.54113845333334 ms`, and E2E `79.24497552654131 ms`;
  `Qwen3-235B-A22B` reports TTFT `47.10755882697941 ms`, TPOT
  `14.657960619999798 ms`, and E2E `61.76551944697921 ms`.

## Modification record — standard-attention stable dedup

- **Motivation:** Chunked-prefill and full-prefill generators produced three
  duplicate structural workloads, inflating every exact manifest and GPU run.
- **Expectation:** Emit one row per structural workload while preserving
  deterministic first-occurrence order and explicit invalid-input failures.
- **Method:** Added a local seen-set after the existing
  `AttentionInput.is_valid()` gate and retained the first valid tuple.
- **Result:** Focused tests report `47 passed`. The direct probe changed from
  `50 generated / 47 unique` to `47 generated / 47 unique`, with `7` prefill
  and `40` decode rows. Explicit KV `128` at `max_model_len=128` still raises
  `ValueError`.

## Validation record — H200 envelope freeze

- **Motivation:** Formal profiling needs one explicit option-A contract with
  measured H200 legality.
- **Expectation:** Every selected TP/EP and measurement lane emits finite
  positive timing rows for all applicable model/module combinations.
- **Method:** Ran the six-model `32`-cell capability smoke, the `224`-cell
  TP/EP axis-legality matrix, and an `8`-cell Qwen3 prefill-context smoke.
- **Result:** Capability is `32/32` PASS; axis legality is `224/224` PASS with
  `130` CUDA_EVENT rows and `130` KERNEL_ONLY rows; the Qwen3 prefill-context
  smoke is `8/8` PASS. The frozen envelope contains `3,408` Attention, `912`
  Linear, and `6,336` MoE logical rows.

## Operational record — H200 worker cleanup reconciliation

- **Motivation:** Apply the maintainer's instruction to keep one H200 worker
  and remove the duplicate allocation.
- **Expectation:** One eight-GPU H200 worker remains available for formal
  profiling.
- **Method:** Queried the two historical RJobs and replicas directly, then
  probed the current running H200 worker and its GPU/process state.
- **Result:** The duplicate `frontier-h200-profile-20260823-1807` is already
  absent. The requested original `frontier-h200-profile-20260823-1744` had
  already entered `Stopped` at `2026-08-23 19:56:40 +08:00`. The current
  replacement `frontier-h200-prefill-hot-20260823-202452` is Running with
  `8/8` idle H200 GPUs and remains retained for the formal lane.

## Decision record — MoE gating-context canonical names

- **Motivation:** The old values imply unsupported semantics: the direct path
  is not a legacy implementation, and the warmed path describes execution
  context rather than a transformed gating input.
- **Expectation:** New metadata uses clear stable names while existing users
  receive an actionable migration signal instead of an immediate failure.
- **Method:** Selected option A: `direct` and `prefill_warmed`; retained
  `standalone_legacy` and `prefill_hot` as temporary aliases that emit visible
  warnings; kept unknown values fail-fast.
- **Result:** The design gate is closed. Production changes wait for focused
  tests to demonstrate the missing alias/canonical contract first.

## Modification record — canonical naming and warned compatibility aliases

- **Motivation:** Formal H200 sampling must emit stable canonical metadata
  while current users and frozen README commands still need a guided migration
  path.
- **Expectation:** New producers, CLI defaults, pseudo-model names, merge
  outputs, examples, and E2E fixtures use `direct` / `prefill_warmed`; both
  legacy values and `__prefill_hot` remain readable with visible warnings;
  unknown values remain fail-fast.
- **Method:** Added a declarative registry in
  `frontier/moe_gating_runtime.py`; derived aliases, implementation metadata,
  suffixes, and CLI values from it; routed all three training/prediction
  consumers through central helpers; normalized legacy CSV rows during
  filtering and deterministic merge; updated every non-README producer,
  example, focused document, and affected test.
- **Result:** The RED baseline reported `10` expected failures, followed by
  `2` integration RED failures and one fail-fast RED. Final focused validation
  reports `97 passed`; Python compilation, five shell syntax checks,
  `git diff --check`, canonical example dry-run, and all nine README SHA-256
  checks pass. Legacy direct probes emit two visible `FutureWarning` messages
  and return canonical metadata.

## Validation and repair record — first formal H200 model

- **Motivation:** Close the exact-manifest evidence chain for
  `llama3.1-8b` before advancing the six-model lane.
- **Expectation:** Both measurement families produce all eight canonical CSVs,
  and the validator checks only timing slices the producer and runtime
  actually use.
- **Method:** Ran the frozen-manifest runner on eight H200 GPUs. The original
  validator failed after successful collection because it required
  `time_stats.add.median` for an RMSNorm fused-add model and required
  replicated operations on TP `2/4/8` rows even though the producer moves
  those measurements into deduplicated TP=1 rows. Added focused RED tests,
  aligned validation with the existing E2E contract and producer split
  semantics, and revalidated the unchanged accepted CSVs.
- **Result:** Original profiling commands all completed successfully:
  attention CUDA_EVENT `807.610 s`, linear CUDA_EVENT `413.519 s`,
  attention KERNEL_ONLY `721.177 s`, and linear KERNEL_ONLY `393.008 s`.
  Revalidation reports `8` CSVs and `1,288` physical rows. Attention contains
  `188` standard rows, `96` true-mixed rows, and `284` combined rows per
  measurement family. Linear contains `76` rows per family and `27`
  applicable positive timing slices. Focused unit validation reports
  `10 passed`.

## Validation record — llama3.1-8b non-dummy E2E

- **Motivation:** Exercise the completed profiling dataset through the actual
  non-dummy simulator path before the next model completes.
- **Expectation:** One request with `prefill_tokens=2` and `decode_tokens=2`
  completes using the formal H200 CSVs and produces complete runtime artifacts.
- **Method:** Ran
  `tests/performance/profiling/run_h200_six_model_non_dummy_e2e.sh` for
  `llama3.1-8b` under a 2 GiB CPU-master memory scope while
  `llama3.3-70b` profiling ran independently on the H200 worker.
- **Result:** **PASS.** Preflight selected `47` TP1 attention rows and `19`
  TP1 linear rows per measurement family. Runtime completed `1/1` request with
  TTFT `17.614207923 ms`, TPOT `4.740640000 ms`, E2E
  `22.354847923 ms`, `2` stage-ledger rows, and `20` op-trace events.

## Operational record — second formal H200 model

- **Motivation:** Keep the GPU lane occupied while the first model's CPU E2E
  closes independently.
- **Expectation:** `llama3.3-70b` runs against the same frozen manifest,
  source commit, CUDA 12.8 environment, and persistent JIT cache.
- **Method:** Started the per-model frozen-manifest runner on the retained
  eight-GPU H200 worker and captured the active worktree diff hash alongside
  the frozen source commit.
- **Result:** Profiling is in progress. The CPU E2E finished without GPU
  contention, and the H200 lane remains the only active profiling worker.

## Operational record — H200 worker preemption and replacement gate

- **Motivation:** Resume the six-model H200 lane after the platform terminated
  the retained worker during `llama3.3-70b` collection.
- **Expectation:** Preserve the old partial output for audit evidence, reject it
  from publication, and allocate one fresh H200 worker with the same resource
  envelope before restarting the model from zero.
- **Method:** Inspected the worker control log and partial output tree. The
  worker reported `Error: signal: terminated` and `ExitCode=-1`; the partial
  attention run stopped at `118/284` and produced no accepted CSV or
  `status.json`. Ran the replacement `--predict-only` request with
  `infer_af_test`, H200, 8 GPUs, 64 CPUs, 409600 MiB, and backoff limit `1`.
- **Result:** The old partial tree remains under
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_2ecc496b_20260823_222643/`
  as failed evidence. Predict-only passed with five eligible 8-GPU H200
  nodes; a new worker and a new staging root are required before collection
  resumes.

## Validation record — llama3.3-70b formal profiling and non-dummy E2E

- **Motivation:** Replace the platform-preempted partial second-model run with
  a complete frozen-manifest dataset and exercise that exact dataset through
  the non-dummy simulator.
- **Expectation:** Produce all eight dense-model CSVs in both measurement
  families, validate `1,288` physical rows, and complete one legal `2+2`
  request with the registry-selected `PP2` topology.
- **Method:** Allocated the replacement RJob
  `frontier-h200-formal-replacement-20260823-2359`, reran the model from zero
  in a fresh staging root, then ran the E2E under a 2 GiB CPU-master memory
  scope against the accepted CSV directory.
- **Result:** **PASS.** Profiling produced `188` standard attention, `96`
  true-mixed attention, `284` combined attention, and `76` linear rows per
  measurement family. The E2E completed `1/1` request with TTFT
  `46.176956782693495 ms`, TPOT `37.570378453333326 ms`, E2E
  `83.74733523602681 ms`, `4` stage-ledger rows, and `42` op-trace events.

## Modification record — H200 runtime-slice validator repair

- **Motivation:** The original exact validator treated fused add/norm and
  producer-deduplicated replicated operations as missing data, while the E2E
  preflight examined whole multi-TP CSVs instead of the runtime-consumed
  slice.
- **Expectation:** Validate exactly the rows the producer emits and the
  simulator consumes, while retaining fail-fast behavior for a missing
  required TP/EP/gating-context slice.
- **Method:** Filtered dense runtime inputs to TP1; filtered MoE inputs to
  TP1/EP2 and all registry-required gating contexts; excluded fused standalone
  `add` and replicated non-TP1 operations from required timing slices; added
  focused regression cases.
- **Result:** Fresh verification reports `25 passed in 5.96s`; compilation,
  shell syntax, and `git diff --check` pass. Commit `7d1be838` records the
  focused repair.

## Operational record — third formal H200 model

- **Motivation:** Keep the H200 lane active after the second model's CPU E2E
  closed.
- **Expectation:** Profile `Qwen3-235B-A22B` from the unchanged frozen
  producer commit and collect both `direct` and `prefill_warmed` MoE contexts.
- **Method:** Created a local execution clone under `/data/ycfeng/tmp` with
  detached HEAD `2ecc496b`, applied only the verified validator repair as an
  uncommitted patch, and started the frozen-manifest runner on the retained
  eight-GPU worker.
- **Result:** Profiling is in progress. Attention CUDA_EVENT admitted all
  `284` manifest cases after physical-capacity filtering and initialized all
  eight H200 devices. Two launch attempts failed before sampling because the
  clone lacked the untracked manifest and a worker-owned staging directory was
  not CPU-writable; both root causes were corrected before the active run.

## Operational record — Qwen targeted KERNEL_ONLY recovery

- **Motivation:** The platform stopped the formal worker after five Qwen lanes
  had completed, while Linear KERNEL_ONLY stopped at `57/76` and both MoE
  KERNEL_ONLY lanes had not started.
- **Expectation:** Reuse only closed, exact-manifest lanes; collect only the
  three missing lanes on one idle eight-GPU H200 worker; assemble and validate
  the full dataset without modifying the original partial tree.
- **Method:** Audited the nine completed raw CSVs for exact row counts,
  measurement metadata, canonical gating contexts, and duplicate feature
  keys; allocated
  `frontier-h200-qwen-resume-20260823-1`; launched sequential Linear,
  MoE `direct`, and MoE `prefill_warmed` KERNEL_ONLY commands into a new
  supplement staging directory.
- **Result:** The reuse audit passed for `2,268` physical rows. Linear
  KERNEL_ONLY completed `76/76` in about `490` seconds and produced
  `linear_op_kernel_only.csv`. Both MoE KERNEL_ONLY contexts completed
  `528/528`; supplement staging contains `1,132` rows. The full assembly
  contains `10` accepted CSVs and `3,400` physical rows, and the frozen
  validator reports `PASS`.

## Validation record — Qwen3-235B-A22B matching non-dummy E2E

- **Motivation:** Exercise the newly assembled Qwen3 H200 data through the
  actual simulator consumer before accepting the model's profiling result.
- **Expectation:** The six-file preflight passes for the runtime-consumed
  TP1/EP2 slices, and one fixed `2+2` request completes with finite positive
  metrics, positive stage-ledger model time, and positive operation traces.
- **Method:** Ran
  `tests/performance/profiling/run_h200_six_model_non_dummy_e2e.sh` for
  `Qwen3-235B-A22B` with the accepted profile directory
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/qwen_assembled_accepted_20260823`,
  output/report roots under
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/e2e_qwen_resume_20260824`,
  Python `3.12.3`, and
  `timeout 1800s systemd-run --user --scope -p MemoryMax=2G --quiet`.
- **Result:** **PASS.** Preflight selected `47` TP1 attention rows per
  measurement family (`7` prefill and `40` decode), `19` TP1 linear rows per
  family, and `66` TP1/EP2 MoE rows per family across the required routing
  contexts. Runtime completed `1/1` request with TTFT
  `64.99044638502953 ms`, TPOT `14.827130135476638 ms`, and E2E
  `79.81757652050617 ms`; the equality residual
  `E2E - (TTFT + TPOT)` is `0.0 ms`. The stage ledger has `4` rows with
  positive model times (`0.692593219`, `0.691374765`, `0.194144145`, and
  `0.193034918 ms`), and the op trace has `56` events with durations from
  `0.0009369066666666667` to `4.634825246874999 ms`.

## Blocking record — H200 worker unavailable for qwen3-a3b-30b-moe

- **Motivation:** Resume the next formal H200 model only after confirming that
  the requested eight-GPU worker can be scheduled.
- **Expectation:** The bounded H200 `--predict-only` request reports eligible
  capacity before any live allocation or profiling process starts.
- **Method:** Ran the verified `infer_af_test` H200 recipe with
  `--gpu=8 --cpu=64 --memory=409600 --backoff-limit=1` and
  `--predict-node-num=10`, then checked current RJob/replica state.
- **Result:** **BLOCKED.** The command printed `no machine available`; the
  wrapper returned exit code `0`, which is treated as semantic failure.
  No `qwen3-a3b-30b-moe` profiling process or output artifact was created.
  The existing launcher remains ready under
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_qwen3_a3b_30b_2ecc496b/launch_model.sh`.

## Handoff record — documentation and execution-state synchronization

- **Motivation:** Prepare a reliable handoff for a new session after the
  Qwen3-235B-A22B profiling and formal runtime validation completed.
- **Expectation:** Every task-memory record reflects the current Git state,
  persistent artifacts, control-plane state, accepted-model count, and exact
  next command boundary.
- **Method:** Rechecked the worktree branch and HEAD, verified the frozen
  manifest hash, inspected the three accepted H200 profile trees and formal
  runtime JSONs, queried H200/H800 RJobs and replicas, and rechecked the
  nine-file README baseline from the repository root.
- **Result:** **PASS.** The handoff is:
  - task memory:
    `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/profiling-data-governance-20260823/task_memory/task_2026-08-23_profiling_dataset_governance/`
  - worktree:
    `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/profiling-data-governance-20260823`
  - branch:
    `data/profiling-governance-h800-h200-20260823`
  - HEAD:
    `7d1be8382fd807f31ee19cb5d866f95a141d83c3`
  - accepted H200 profiling:
    `3/6` models (`llama3.1-8b`, `llama3.3-70b`,
    `Qwen3-235B-A22B`)
  - formal accepted-data E2E:
    `3/6` models, each with `prefill=2` and `decode=2`
  - next executable boundary:
    rerun the H200 `infer_af_test` 8-GPU predict-only gate; after semantic
    success, run
    `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_qwen3_a3b_30b_2ecc496b/launch_model.sh`.
  - control plane:
    no active H200 or H800 RJob/replica.

## Validation record — repeated H200 predict-only gate 2026-08-24

- **Motivation:** Resume the next H200 model only after a fresh scheduler
  capacity check, while preserving the verified eight-GPU resource envelope.
- **Expectation:** The exact `infer_af_test` request either reports at least one
  eligible H200 node or stops before live allocation with an explicit
  scheduler failure.
- **Method:** Ran the handbook-backed command with RJob name
  `frontier-h200-qwen3-a3b-20260824`, namespace `shai-core`, H200 tag,
  `--gpu=8`, `--cpu=64`, `--memory=409600`, `--predict-node-num=10`, and
  `--backoff-limit=1`; captured
  `/data/ycfeng/tmp/frontier_h200_qwen3_a3b_predict_20260824.log`.
- **Result:** **BLOCKED.** Scheduler output is `no machine available`.
  The launcher returned `PREDICT_EXIT=0`, but the semantic classifier treats
  the body failure as authoritative. Log SHA-256 is
  `4b6a7871ca9b971b8b6431dde02855cf94a3490521a3f4ece2c18abeb03f59fa`.
  No live RJob, profiling process, or output artifact was created.

## Validation record — resumed H200 predict-only gate 2026-08-24 12:50 +08:00

- **Motivation:** Recheck H200 capacity after a new session resumed the task;
  the next model must remain at the scheduler gate until an eligible worker
  is reported.
- **Expectation:** The exact `infer_af_test` request reports an eligible H200
  node before any live allocation or profiling process starts.
- **Method:** Ran the handbook-backed request with RJob name
  `frontier-h200-qwen3-a3b-resume-20260824`, namespace `shai-core`,
  `--gpu=8`, `--cpu=64`, `--memory=409600`, `--predict-node-num=10`, and
  `--backoff-limit=1`; persisted the output at
  `/data/ycfeng/tmp/frontier_h200_qwen3_a3b_predict_20260824_resume.log`.
- **Result:** **BLOCKED.** The sole scheduler message was
  `no machine available`; the wrapper exit code was `0`, which is treated as
  semantic failure. The log SHA-256 is
  `31c25f492780ce3ea7829596c3009a4844b9ac58ef6641d775acd1d0f82fef72`.
  No live RJob, replica, profiling process, or new output artifact exists.

## Validation record — independent H200 control-plane recheck 2026-08-24 12:53 +08:00

- **Motivation:** Independently verify that the scheduler failure was not
  caused by a stale RJob name or an unobserved live replica.
- **Expectation:** A unique predict-only name produces the same semantic
  capacity result, and follow-up control-plane queries show no RJob or
  replica for that name.
- **Method:** Queried all `frontier-h200*` and `frontier-h800*` RJobs and
  replicas without executing against any stopped worker. Then ran the unique
  request `frontier-h200-qwen3-a3b-recheck-20260824-125342` with the frozen
  eight-GPU resource envelope. Persisted log:
  `/data/ycfeng/tmp/frontier_h200_qwen3_a3b_predict_recheck_20260824_85c6wP.log`.
- **Result:** **BLOCKED.** All historical H200 jobs were terminal, no H800
  jobs were present, and the unique request printed `no machine available`.
  The wrapper exit code was `0`, semantic status `FAIL`; log SHA-256 is
  `1bffde855e088b0a6804ead4bee57774df06035e1bc2f980bfef92ecb7dc4806`.
  Follow-up queries returned no RJob and `NO_REPLICA`; no profiling process or
  output artifact was created.

## Validation record — accepted H200 staging revalidation 2026-08-24 12:58 +08:00

- **Motivation:** Confirm that the three accepted staging trees remain
  directly eligible for future canonical publication while GPU collection is
  externally blocked.
- **Expectation:** The frozen manifest validator accepts every required
  attention, linear-op, and MoE key, metadata value, measurement type, and
  positive finite target timing slice.
- **Method:** Ran the repository validator functions against the accepted
  directories for `llama3.1-8b`, `llama3.3-70b`, and `Qwen3-235B-A22B` with
  `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1`; persisted JSON evidence at
  `/data/ycfeng/tmp/frontier_h200_accepted_staging_revalidation_20260824.json`
  (SHA-256
  `899d744fc475e15b4c1e4b755ff8608c155db9f8042f74a2683e3ac42a387288`).
- **Result:** **PASS.** The three models contain `8`, `8`, and `10` accepted
  CSVs, respectively, with `1,288`, `1,288`, and `3,400` physical rows. All
  attention standard/true-mixed keys match; linear target slices are `27/27`,
  `27/27`, and `15/15` per measurement family; Qwen MoE contexts each contain
  `528` rows for `direct` and `prefill_warmed`, with finite positive target
  minima.

## Validation record — focused contract pytest 2026-08-24

- **Motivation:** Verify the registry-backed gating-context aliases and the
  six-model non-dummy E2E contract after the handoff.
- **Expectation:** The focused unit tests pass without changing source files.
- **Method:** The first collection attempt used the bare system interpreter
  and failed with `ModuleNotFoundError: No module named 'frontier'`. After
  consulting the repository environment handbook, reran with
  `PYTHONPATH=$PWD`:
  `timeout 120s env PYTHONPATH="$PWD" pytest -q -p no:cacheprovider
  tests/unit/test_moe_gating_runtime_context_aliases.py
  tests/unit/test_h200_six_model_non_dummy_e2e_contract.py
  tests/unit/test_profiling_output_contract.py`.
  Log:
  `/data/ycfeng/tmp/frontier_profiling_governance_focused_pytest_20260824_pythonpath.log`.
- **Result:** **PASS.** `34 passed in 8.77s`, exit code `0`; the initial
  environment-only collection failure is resolved by the documented
  `PYTHONPATH` setup.

## Validation record — repository README and whitespace gate 2026-08-24

- **Motivation:** Confirm that the task left the protected README surface and
  tracked source whitespace unchanged.
- **Expectation:** All nine baseline README hashes match and `git diff --check`
  reports no whitespace errors.
- **Method:** Ran
  `sha256sum -c /data/ycfeng/tmp/frontier_profiling_governance_readme_baseline_20260823.sha256`
  from `/data/ycfeng/stepfun-performance-optimization/Frontier` so the
  baseline's `.worktrees/...` relative paths resolve correctly, then ran
  `git diff --check` in the task worktree.
- **Result:** **PASS.** All `9/9` README files report `OK`; `git diff --check`
  emits no diagnostics. The only worktree status entry remains the explicitly
  preserved untracked file `=10.1`.

## Modification record — governance manifests 2026-08-24

- **Motivation:** Prepare the active-data cleanup and publication steps while
  H200 capacity remains externally blocked.
- **Expectation:** Produce an exact, reviewable archive plan and a coverage
  inventory without moving files or publishing unvalidated rows.
- **Method:** Derived `archive_manifest.csv` and `coverage_manifest.csv` from
  `strict_audit_before.json`, the 38 admitted H800 CSVs, and the three
  accepted H200 staging trees; then checked source existence, taxonomy,
  duplicate destinations, and recorded row counts.
- **Result:** **PASS.** The archive manifest has `58` rows (`37` direct
  `git_mv`, `21` `merge_then_git_mv`); the coverage manifest has `64` rows
  (`38` H800 direct, `26` H200 staging). All sources exist and collision count
  is `0`. No active-data mutation was performed.

## Audit record — canonical data evidence report 2026-08-24

- **Motivation:** Close the remaining read-only canonical-data audit sub-step
  and leave a reproducible evidence package before the H200 worker becomes
  available again.
- **Expectation:** The report records the complete CSV/YAML inventory,
  strict-admission result, metadata/path/schema anomalies, raw timing counts,
  duplicate conflicts, gating aliases, manifest hashes and coverage gaps, and
  separates Evidence, Inference and Unknown without changing active data.
- **Method:** Reconciled the strict-audit JSON/Markdown, direct CSV/YAML
  probes, archive/coverage manifest checks, and accepted H200 staging/E2E
  artifacts. Wrote the report to
  `/data/ycfeng/tmp/frontier_canonical_audit_20260824.md` and verified its
  frozen-manifest hash reference against the checked-in file.
- **Result:** **PASS.** The report is `302` lines / `22,546` bytes with
  SHA-256 `c7e8c8d0efb56bd9ace260599605c1068ffbc80cdfb61052dc1aa863c289b3d8`.
  It records `104` CSVs / `97,671` rows, `46` strict current-direct files,
  `37` archive-required files, `170,493` blank timing cells, `753,352`
  non-positive timing cells, `55` duplicate files, and the eight RTX coverage
  gaps. `git diff -- data/profiling/compute` is clean and there are no
  untracked files under that tree; the only worktree status entry remains the
  preserved untracked `=10.1`.

## Validation record — fresh H200 worker and qwen3-a3b retry 2026-08-24

- **Motivation:** Resume the fourth confirmed H200 model after the scheduler
  recovered capacity, while preserving the frozen manifest and all prior
  failed-attempt evidence.
- **Expectation:** The exact `infer_af_test` envelope allocates one 8-GPU H200
  replica, the environment smoke passes, and the isolated launcher advances
  through the manifest lanes without overwriting the old output directory.
- **Method:** Ran the bounded predict-only request with `8 GPU / 64 CPU /
  409600 MiB`, then created RJob
  `frontier-h200-qwen3-a3b-retry-20260824-1423` and executed the mechanical
  retry launcher with a new `RUN_ROOT`. Persisted control, environment, and
  raw logs under `/data/ycfeng/tmp`.
- **Result:** **IN PROGRESS / GATE PASS.** Predict-only returned `8` H200
  candidates (log SHA-256
  `34bdeb437f0743aa69e36e9b068c2c319aeb3f440c0f700e6d019a870bd938ba`). The
  replica is `frontier-h200-qwen3-a3b-retry-20260824-1423-cfcd2084` on
  `gpu-h200-0949.lgcm.sh.istep.fun`, with eight `NVIDIA H200` GPUs and
  `143771 MiB` per GPU. Python `3.12.3`, torch `2.8.0+cu128`, vLLM `0.10.2`,
  FlashInfer `0.3.1.post1`, and CUDA availability all passed. Attention
  CUDA_EVENT completed `284/284` and wrote the standard, true-mixed, and
  combined CSVs; linear CUDA_EVENT completed `76/76` and wrote
  `linear_op.csv`. MoE CUDA_EVENT `direct` is now running. No traceback,
  cgroup OOM, or platform termination has appeared.
- **Historical root-cause record:** The prior `exit 137` was an outer
  `timeout 300s` interrupt of detached `rlaunch` (control-plane RJob exited
  `0`, no `OOMKilled`); the original launcher then fail-fast rejected its
  existing model output directory with `FileExistsError`. Both evidence trees
  remain untouched.

## Validation record - qwen3-a3b profiling completion 2026-08-24

- **Motivation:** Close the fourth confirmed H200 model after the retry
  launcher completed, and establish that its accepted data satisfies the
  frozen manifest before runtime consumption.
- **Expectation:** All eight dual-measurement lanes finish without a fatal
  worker error; the accepted tree contains the exact manifest coverage,
  canonical metadata, and finite positive timing values in every runtime
  slice.
- **Method:** Read the retry launcher status, `status.json`, and
  `validation.json`; independently scanned the accepted tree with the frozen
  manifest validator and the registry-aware profile preflight. Preserved the
  output under
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_qwen3_a3b_30b_2ecc496b_retry_20260824_1423/`.
- **Result:** **PASS.** The launcher exit code is `0`, and all eight lanes
  completed: Attention CUDA_EVENT `284/284`, Linear CUDA_EVENT `76/76`, MoE
  CUDA_EVENT `direct=528/528` and `prefill_warmed=528/528`, with matching
  KERNEL_ONLY counts. The accepted tree has `10` CSVs and `3,400` physical
  rows (`188/96/284` standard/true-mixed/combined attention rows per
  measurement family, `76` linear rows, and `1,056` MoE rows per family).
  `status.json.status` and `validation.json.status` are both `PASS`. The
  frozen manifest SHA-256 is
  `4df580ca1e30a007f45aeed4eb9f5d43593cbab49e59194ecabf8c5996ce8098`.
  The minimum positive timing values are Attention CUDA_EVENT prefill
  `0.0550719983875751 ms`, decode `0.0405119992792606 ms`; Attention
  KERNEL_ONLY prefill `0.00944 ms`, decode `0.0064639999999999 ms`; Linear
  CUDA_EVENT `0.0175519995391368 ms`, KERNEL_ONLY `0.00176 ms`; and MoE
  CUDA_EVENT `0.0248799994587898 ms`, KERNEL_ONLY `0.00435 ms`.
- **Known warnings:** Generic architecture fallback, missing linear
  record-function `add`, and the vLLM fused-MoE default-config warning remain
  non-fatal and are recorded in the validation report.

## Validation record - qwen3-a3b accepted-data 2+2 non-dummy E2E 2026-08-24

- **Motivation:** Verify that the newly accepted qwen3-a3b CSVs are directly
  consumable by the current non-dummy Frontier runtime before advancing to the
  fifth model.
- **Expectation:** The registry-derived contract uses H200, co-location,
  analytical communication, PP1, ATTN TP1/DP2, MoE TP1/EP2, and exactly one
  fixed `prefill=2`, `decode=2` request. Preflight and runtime validation must
  pass, with one completed request and a zero E2E residual.
- **Method:** Ran the bounded runner with `PYTHON_BIN=python3`,
  `PROFILE_ROOT` set to the accepted parent directory, and outputs under
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/e2e_qwen3_a3b_20260824/`:
  `timeout 1800s systemd-run --user --scope -p MemoryMax=2G --quiet -- env
  PROFILE_ROOT=.../accepted OUTPUT_ROOT=.../runs REPORT_ROOT=.../reports
  RUN_ID_PREFIX=qwen3_a3b_h200_e2e_20260824 PYTHON_BIN=python3 bash
  tests/performance/profiling/run_h200_six_model_non_dummy_e2e.sh --model
  qwen3-a3b-30b-moe`. The runner first executed preflight and then
  `frontier.main`; it validated the persisted metrics, stage ledger, and op
  trace.
- **Result:** **PASS.** Preflight and `validate-run` both exited `0`. Runtime
  metrics report `1/1` completed request, TTFT
  `30.99188063610683 ms`, TPOT `4.0792620915476 ms`, and E2E
  `35.07114272765443 ms`; `35.07114272765443 -
  (30.99188063610683 + 4.0792620915476) = 0.0 ms`. The stage ledger contains
  `2` rows with model times `0.64566418 ms` (prefill) and `0.09501359 ms`
  (decode). The op trace contains `27` finite durations, minimum
  `0.0007184533333333333 ms`, maximum `1.72269355 ms`. The runtime report
  records `standard_fused_topk`, ATTN DP `2`, and MoE EP `2` through the
  generated configuration.
- **Persistence:** Runtime report SHA-256
  `9e875b4d832a86dda6d9bfae82f6375afff449990d6d80aad87b2fb5e0624fd3`;
  preflight report SHA-256
  `1ec5012eadf035011d74c5e39e3119c8be9ae8d27734a65d8550602b76c16bff`;
  runtime log SHA-256
  `4c0bd7726675c87aaba3cd7ae7c3323e65f532177c7cbedc9ccc18cbb170efa4`.

## Operational record - qwen3-a3b worker release 2026-08-24

- **Motivation:** Release the eight-GPU H200 worker after the launcher and
  downstream E2E evidence were persisted, so the task leaves no active GPU
  allocation.
- **Expectation:** The detached `brainctl rjob launch` exits cleanly and the
  control plane converges to terminal RJob/replica states without deleting
  evidence or touching the stopped historical worker.
- **Method:** Sent SIGTERM to the current detached launcher PID only after
  `launcher_qwen3-a3b-30b-moe_status.txt` reported `exit_code=0`; queried the
  named RJob and replica read-only with bounded `brainctl get` calls.
- **Result:** The local launcher exited, the replica transitioned to
  `Succeeded`, and the RJob converged to `Succeeded`. No active H200/H800
  worker remains; all accepted CSVs and reports stay on `/data/ycfeng`. The
  stopped historical RJob `frontier-h200-qwen-resume-20260823-1` received no
  `brainctl exec`.

## Validation record - four-model accepted staging revalidation 2026-08-24

- **Motivation:** Extend the prior three-model aggregate check to include the
  newly completed qwen3-a3b tree before treating the H200 staging checkpoint
  as `4/6` accepted.
- **Expectation:** Every accepted tree passes both measurement-family checks
  against the frozen manifest, with exact CSV and physical-row counts.
- **Method:** Reused the repository's `_validate_attention_files`,
  `_validate_linear_file`, and `_validate_moe_file` functions in a bounded
  inline Python command. Persisted JSON evidence at
  `/data/ycfeng/tmp/frontier_h200_accepted_staging_revalidation_20260824_all4.json`.
- **Result:** **PASS.** `llama3.1-8b` and `llama3.3-70b` each have `8` CSVs
  and `1,288` rows; `Qwen3-235B-A22B` and `qwen3-a3b-30b-moe` each have `10`
  CSVs and `3,400` rows. The aggregate is `36` CSVs and `9,376` physical
  rows. Evidence SHA-256 is
  `bcdd05f2a66212508c61a164f86146fda1919d84572ab353e562d9de39d737dc`.

## Diagnostic record - Step3 FlashInfer JIT retry preparation 2026-08-24

- **Motivation:** The first Step3 attempt failed at attention CUDA_EVENT
  `53/284` with FlashInfer/Ninja reporting `opening build log: No such file or
  directory`; a root-cause check was required before retrying.
- **Expectation:** A minimal true-mixed attention case succeeds when its JIT
  cache is isolated from the shared NFS cache, proving the model, CUDA, and
  kernel path are usable and separating the failure from model semantics.
- **Method:** On the active H200 replica
  `frontier-h200-step3-20260824-1815-cfcd2084`, ran the real attention entry
  point with `--num_gpus 1`, TP `1`, head dim `256`, four standard/true-mixed
  inputs, and `FLASHINFER_WORKSPACE_BASE=/mnt/host0/frontier_step3_jit_diag_20260824/flashinfer_workspace`.
  Output was persisted under
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/step3_jit_serial_diag_20260824/`.
  The failed NFS target was also inspected: its `.ninja_log` was absent at the
  failure point and appeared again after a concurrent Ninja repair, while the
  profiling source uses eight spawned processes sharing one NFS cache.
- **Result:** **PASS / root cause isolated.** The local-cache smoke exited `0`
  after `143` seconds and produced `attention.csv` (3,444 bytes),
  `attention_true_mixed.csv` (2,232 bytes), and `attention_combined.csv`
  (4,427 bytes). The evidence supports a first-use concurrent NFS JIT metadata
  race, not an unsupported Step3 operator or CUDA environment failure. The
  formal retry uses a new output root and worker-local JIT cache.

## Operational record - Step3 retry resume checkpoint 2026-08-24

- **Motivation:** Resume the interrupted session without duplicating the
  formal runner or allocating another H200 worker.
- **Expectation:** The named Step3 RJob remains the sole active lane; completed
  module outputs remain intact; the next MoE context advances without a fatal
  error.
- **Method:** Queried the named RJob and replica with bounded read-only
  `brainctl get` calls, inspected the persistent staging tree and module logs,
  and used read-only `brainctl exec` only on the active Step3 replica to check
  PID `8693`, its MoE child, and the eight H200 devices.
- **Result:** **IN PROGRESS.** RJob
  `frontier-h200-step3-20260824-1815` is `Running` and replica
  `frontier-h200-step3-20260824-1815-cfcd2084` is `1/1 Running`. Attention
  CUDA_EVENT is complete at `284/284`; Linear CUDA_EVENT is complete at
  `76/76`; MoE CUDA_EVENT `direct` advanced from `33/528` to `66/528` during
  the checkpoint. `prefill_warmed` and KERNEL_ONLY remain pending, and no
  `status.json` or `validation.json` has been written. The error scan found
  no fatal exception or CUDA/Ninja failure signature.

## Diagnostic record - Step3 MoE direct interval audit 2026-08-24

- **Motivation:** Progress slowed near `198/528` while the active replica
  showed low GPU utilization and spawned children in `D` state; distinguish a
  normal long CUDA task from a deadlock before touching the worker.
- **Expectation:** Source-level scheduling and a completed model's logs should
  explain the pause, or the evidence should identify a reproducible hang.
- **Method:** Read the current MoE multiprocessing implementation, inspected
  the active replica's runner/child states and wait channel, scanned all Step3
  logs for fatal signatures, and compared progress timestamps with the
  completed qwen3-a3b direct log.
- **Result:** **PASS / expected behavior.** The implementation uses one
  `ProcessPoolExecutor(max_workers=1)` per GPU and advances `tqdm` only after a
  future completes. Step3 advanced from `198/528` to `231/528` during the
  audit; the error count stayed `0`. The qwen3-a3b reference shows the same
  first-sample `~160 s` and grouped-progress pattern. No fix, restart, or
  worker control action was taken.

## Contract record - Step3 gating-context clarification 2026-08-24

- **Motivation:** The handoff summary incorrectly implied that Step3 needed
  both `direct` and `prefill_warmed` MoE contexts; the active runner advanced
  to the next measurement method after `direct`.
- **Expectation:** The frozen manifest, producer, and runtime validator agree
  on the exact Step3 context set before accepting its output.
- **Method:** Read the frozen manifest's model entry, traced
  `_gating_contexts()` in `generate_h200_exact_manifest.py`, and checked the
  shared `should_enable_prefill_warmed_moe_gating_contract()` registry rule
  used by the E2E validator.
- **Result:** **PASS / clarified.** Step3 is `model_type=step3_text` with
  `gating_runtime_contexts=["direct"]`, MoE target rows `1,056` per
  measurement family, and no `prefill_warmed` requirement. Only qwen3_moe
  enables the second context. The manifest hash remains unchanged.

## Operational record - Step3 CUDA_EVENT direct completion 2026-08-24

- **Motivation:** Capture the first completed Step3 MoE measurement family and
  the runner's exact next phase.
- **Expectation:** Direct MoE reaches the manifest target and the runner
  advances to the next method without creating an extra Step3 context.
- **Method:** Read the direct log tail, output file metadata, active child
  command, and all current log error signatures.
- **Result:** **PASS / IN PROGRESS.** MoE CUDA_EVENT direct reached `528/528`
  at 19:50:07 HKT and wrote a 366,419-byte `moe.csv`. The runner started
  Attention `record_function`; that lane reached `17/284` by 19:53:59 HKT.
  No fatal exception signature was observed.

## Validation record - Step3 H200 profiling and independent frozen-manifest validation 2026-08-24

- **Motivation:** Close the Step3 profiling gate after the worker-local
  FlashInfer cache retry and establish an independent evidence chain before
  starting its runtime E2E.
- **Expectation:** The single manifest runner completes both measurement
  families with the frozen Step3 direct-only MoE context; producer and
  independent validators agree on exact metadata, coverage, and timing
  invariants without modifying canonical data.
- **Method:** Kept RJob
  `frontier-h200-step3-20260824-1815` / replica
  `frontier-h200-step3-20260824-1815-cfcd2084` as the only active lane. Read
  `status.json` and `validation.json`, scanned all six logs for fatal
  signatures, computed CSV row counts and duplicate feature keys, verified the
  frozen manifest SHA-256, and reran `_load_manifest`,
  `_validate_attention_files`, `_validate_linear_file`, and `_validate_moe_file`
  from the frozen execution clone at HEAD `2ecc496b`.
- **Result:** **PASS.** The runner ended at `2026-08-24T21:26:32+0800` with
  exit code `0` for all seven commands. Attention is `188` standard + `96`
  true-mixed + `284` combined per measurement family; Linear is `76`; MoE is
  `528` direct rows per family. The accepted tree contains `10` CSVs and
  `2,344` physical rows. Producer and independent validators both report
  `PASS`; duplicate feature rows are `0`, no timing value is infinite or
  negative, and the manifest SHA-256 is
  `4df580ca1e30a007f45aeed4eb9f5d43593cbab49e59194ecabf8c5996ce8098`.
  Numeric minima are CUDA_EVENT Attention prefill/decode
  `0.0553280003368854/0.0414399988949298 ms`, Linear
  `0.0068320001009851 ms`, MoE `0.0242880005389451 ms`; KERNEL_ONLY minima
  are `0.010016/0.006718 ms`, `0.00152 ms`, and `0.005822 ms` respectively.
  Evidence is persisted at
  `/data/ycfeng/tmp/frontier_step3_independent_validation_20260824.json` and
  the task report
  `test_report_2026-08-24_step3_moe_noquant_profiling_validation.md`.

## Validation record - Step3 accepted-data 2+2 E2E retry 2026-08-24 21:36 +08:00

- **Motivation:** Re-run the exact fixed Step3 runtime contract after the
  first admission failure, using a fresh persistent output tree and leaving
  the accepted profiling tree immutable.
- **Expectation:** Preflight remains `PASS`; the runtime either completes the
  one-request `prefill=2, decode=2` case or emits the same explicit admission
  error, which determines whether the blocker is reproducible.
- **Method:** Ran the bounded command from the new test report with
  `PP1`, Attention `TP1/DP2`, MoE `TP1/EP2`, H200, analytical communication,
  and a 2 GiB CPU-master memory scope. Outputs are under
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/e2e_step3_retry_20260824_2215/`.
- **Result:** **BLOCKED / REPRODUCED.** Preflight returned `status=PASS` for
  the six runtime-required CSVs; the accepted tree contains 10 CSVs and 2,344
  physical rows. Frontier exited `1` before
  request execution with `FRONTIER_MEMORY_OOM`:
  `parameter_memory_per_device_bytes=351956369408`,
  `requested_memory_bytes=136257837465`, and
  `non_kv_cache_overhead_bytes=0`. The excess is `215698531943` bytes and the
  shard is approximately `2.58x` the requested budget. No TTFT, TPOT, E2E,
  or residual metric exists for this run. The retry report is
  `test_report_2026-08-24_step3_moe_noquant_non_dummy_e2e_retry.md`; runtime
  log SHA-256 is
  `c4c59ed3022ebc701a56a6c21f872d03b08e6cab38f5efedbd978aecda8ddd57`.

## Operational record - Step3 control-plane retry 2026-08-24

- **Motivation:** Confirm that the retry request did not leave a stale or
  duplicate GPU lane and establish whether the existing H200 worker can be
  reused after the deterministic Step3 admission failure.
- **Expectation:** A namespace-scoped read-only query identifies the named
  RJob/replica state and shows whether any additional H200/H800 RJob is active;
  the stopped historical RJob remains untouched.
- **Method:** Ran bounded `brainctl -n shai-core get rjob` and `get replica`
  queries under a 2 GiB systemd scope. Queried only
  `frontier-h200-step3-20260824-1815` and its Ready replica, plus the RJob
  inventory. No `brainctl exec` call was made.
- **Result:** **PASS / gate remains open.** The named RJob is `Running` with
  `active=1`; its replica is `Running`, `Ready`, on
  `gpu-h200-0753.lgcm.sh.istep.fun`, and its command is `sleep 21600`. The
  owner-scoped inventory contains no other active H200/H800 RJob. Cluster-wide
  listing is forbidden for this identity, so this does not claim that other
  users have no active GPU jobs. The worker is reusable after an approved
  topology decision, while the existing PP1 / Attention TP1/DP2 / MoE TP1/EP2
  contract remains blocked by the recorded parameter-memory admission failure.

## Validation record - five-model accepted staging revalidation 2026-08-24 21:43 +08:00

- **Motivation:** Extend the previous four-model aggregate check to the
  accepted Step3 tree before any publication or legacy migration decision.
- **Expectation:** All five accepted H200 roots pass both frozen measurement
  families with exact CSV and physical-row counts; the evidence file parses as
  JSON and remains separate from canonical data.
- **Method:** Ran the bounded inline validator against the five accepted roots
  using the frozen manifest and wrote structured evidence to
  `/data/ycfeng/tmp/frontier_h200_accepted_staging_revalidation_20260824_all5.json`.
  Generic architecture warnings were captured separately in
  `/data/ycfeng/tmp/frontier_h200_accepted_staging_revalidation_20260824_all5.log`.
- **Result:** **PASS.** `llama3.1-8b` and `llama3.3-70b` each have `8` CSVs
  and `1,288` rows; `Qwen3-235B-A22B` and `qwen3-a3b-30b-moe` each have `10`
  CSVs and `3,400` rows; Step3 has `10` CSVs and `2,344` rows. The aggregate
  is `46` CSVs and `11,720` physical rows. Evidence SHA-256 is
  `32cbbfa08ba3cc1cf11f42a96ca53cbb70ed16d278e248eb293b6db682e89975`.

## Static validation record - Step3 candidate topology 2026-08-24

- **Motivation:** Test whether the recommended alternative topology is
  accepted by the existing Frontier configuration and whether its required
  profiling slices are present before presenting the design fork.
- **Expectation:** A CPU-only `ReplicaConfig`/`ParamCounter` check accepts the
  shared-domain topology, and the immutable Step3 accepted CSVs contain the
  corresponding TP/EP rows with module-applicable positive targets.
- **Method:** Constructed Step3 configs for the current and candidate layouts
  with `PYTHONPATH=$PWD`, then inspected the accepted CSV metadata. The
  candidate checked was PP1, Attention TP4/DP2, MoE TP2/EP4, with 8 devices.
- **Result:** **PASS / design fork remains gated.** The current layout reports
  `351,956,369,408` bf16 parameter bytes per device; the candidate reports
  `88,353,800,192` bytes and satisfies the shared-domain equation. Step3
  attention and linear files contain TP values `{1,2,4,8}`; both MoE files
  contain `33` direct rows for TP2/EP4, so the candidate has profiling
  coverage. This static check did not alter source, CSVs, or runtime state.
## Validation record - Step3 strict vLLM option-A semantic audit 2026-08-25

- **Motivation:** Determine whether the fresh option-A runtime PASS is a
  valid strict vLLM mixed-layer result before opening the authorized layer64
  plus PP fallback.
- **Expectation:** A strict result must map dense boundary `mlp_*` to attention
  TP8, shared-expert `share_expert_*` to MoE TP1, and routed MoE operations to
  MoE TP1/EP8 while preserving the finite 2+2 request metrics.
- **Method:** Inspected the exact worker launcher and runtime JSON/logs,
  replayed the predictor TP lookup on CPU with the Step3 model configuration,
  compared op-trace durations against TP1/TP8 rows in the immutable accepted
  `linear_op.csv`, and checked the registry family bindings.
- **Result:** **SURFACE PASS / SEMANTIC FAIL.** The worker completed `1/1`
  request with TTFT `61.64956658853185 ms`, TPOT `13.888752592678598 ms`, E2E
  `75.53831918121044 ms`, two ledger rows, and `30` op-trace events. However,
  monolithic construction passes `model_manager=None`, and the independent
  predictor resolves both `mlp_*` and `share_expert_*` to TP1. The dense trace
  values exactly match TP1 profile rows. The accepted profile tree and frozen
  manifest remain unchanged. Strict acceptance and the next worker run wait
  for explicit approval of a family-aware shared-predictor fix; layer64 plus
  PP remains conditional on a corrected A admission/runtime failure.

## Historical/Superseded continuation record - six-model inventory audit and Step3 deferral 2026-08-25

- **Motivation:** The proposed family-aware predictor change spans the
  independent predictor, shared manager, validator, and regression contract.
  The maintainer requested that this change be recorded and deferred while any
  incomplete model profiling or runtime E2E work continues.
- **Expectation:** The frozen manifest's six models reconcile to accepted
  profiling roots and formal `prefill=2, decode=2` runtime reports. A model is
  considered incomplete when its formal accepted tree or formal E2E artifact is
  absent; capability smoke and fixture data do not close that gate.
- **Method:** Read the frozen manifest and SHA-256, counted CSVs and corrected
  physical rows in each accepted staging root, inspected formal runtime report
  paths, checked the two prepared Mixtral launcher directories, and ran
  namespace-scoped read-only control-plane queries. Evidence is persisted at
  `/data/ycfeng/tmp/frontier_model_inventory_audit_20260825.txt` and
  `test_report_2026-08-25_model_inventory_and_deferred_step3.md`.
- **Result:** **PASS / continuation identified.** The manifest contains six
  models and remains SHA-256
  `4df580ca1e30a007f45aeed4eb9f5d43593cbab49e59194ecabf8c5996ce8098`.
  Five accepted roots are complete: `llama3.1-8b` and `llama3.3-70b` each
  have `8` CSVs and `1,288` physical rows; `Qwen3-235B-A22B` and
  `qwen3-a3b-30b-moe` each have `10` CSVs and `3,400` rows; Step3 has `10`
  CSVs and `2,344` rows. The first four have formal accepted-data E2E PASS;
  Step3 has only a surface PASS with strict semantic FAIL. Both formal Mixtral
  directories contain only `launch_model.sh`; its capability/axis smoke and
  fixture outputs are not formal profiling. Mixtral is therefore the only
  incomplete profiling and E2E model, and its lane is resumed. No source,
  canonical CSV, frozen manifest, README, or `=10.1` file was changed.

## Validation record - Mixtral r7 launch and cache gate 2026-08-25

- **Motivation:** Continue the only incomplete formal model lane after the
  six-model audit, while preserving the maintainer's decision to defer the
  cross-cutting Step3 predictor and topology changes.
- **Expectation:** A fresh `infer_af_test` H200 worker accepts the frozen
  launcher, both root and UID `10250` can write the worker-local cache, and the
  manifest runner starts without changing accepted trees or the frozen hash.
- **Method:** Queried RJob `frontier-h200-mixtral-20260825-r7` and replica
  `frontier-h200-mixtral-20260825-r7-cfcd2084`, verified node
  `gpu-h200-0234.lgcm.sh.istep.fun` with 8 H200 GPUs, 64 CPU, and 409600 MiB,
  created `/mnt/host0/frontier_mixtral_jit_retry_20260825_r7` as root, and
  verified writes as `i-fengyicheng` (`uid=10250`). Started the isolated
  launcher at
  `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_mixtral_8x7b_moe_2ecc496b_20260825_retry_localcache_r7/launch_model.sh`
  with the frozen manifest and source checkout unchanged.
- **Result:** **IN PROGRESS / GATE PASS.** Launcher SHA-256 is
  `a4aba2f521546b521d2dc6f946ef69f09157bc6a74af539a62bfbc7a67002bdd`;
  manifest SHA-256 remains
  `4df580ca1e30a007f45aeed4eb9f5d43593cbab49e59194ecabf8c5996ce8098`.
  The persistent control log is
  `/data/ycfeng/tmp/frontier_h200_mixtral_exec_r7.log` and the worker PID is
  recorded in `/data/ycfeng/tmp/frontier_h200_mixtral_exec_r7.pid`.
  The runner entered Attention CUDA_EVENT (`284` cases); the initial state
  showed 8 live child processes and no `Traceback`, CUDA OOM, permission, or
  Ninja error. Formal validation and E2E remain pending until the runner
  writes its terminal status.

## Live continuation record - Mixtral r7 control-plane recheck 2026-08-25

- **Motivation:** Verify the apparent missing local status files against the
  worker control plane before classifying the formal lane as failed or starting
  another worker.
- **Expectation:** The existing r7 replica remains the sole active H200 lane,
  its manifest runner remains alive, and persistent logs show forward progress
  without a fatal profiling signature.
- **Method:** Ran `brainctl version` (`v2.12.0-alpha.4.1328`), queried the
  namespace-scoped RJob and replica, and used `brainctl exec` only against the
  still-Running r7 replica. Inspected worker PIDs, `nvidia-smi`, the Attention
  log, and terminal artifact presence.
- **Result:** **PASS / IN PROGRESS.** RJob
  `frontier-h200-mixtral-20260825-r7` is `Running`; replica
  `frontier-h200-mixtral-20260825-r7-cfcd2084` is `Ready` on
  `gpu-h200-0234.lgcm.sh.istep.fun` with 8 H200 GPUs, 64 CPU, and 409600 MiB.
  Worker PIDs `396` (launcher), `414` (frozen-manifest runner), and `504`
  (Attention CUDA_EVENT) are alive. Progress advanced from `48/284` to
  `69/284` by `2026-08-25T18:35:37+08:00`; GPU memory was
  `13,864-18,575 MiB` per device and no OOM, traceback, permission, or Ninja
  error was observed. `status.json` and `validation.json` do not exist yet,
  so the profile remains unaccepted and no E2E was started. The old stopped
  `frontier-h200-qwen-resume-20260823-1` was not queried.

## Live continuation record - Mixtral r7 progress refresh 2026-08-25 20:20 +08:00

- **Motivation:** Keep the task memory aligned with the active profiling
  process while terminal artifacts remain pending.
- **Expectation:** The existing r7 worker remains the sole active H200 lane,
  the current KERNEL_ONLY command continues to make progress, and no fatal
  profiling signature appears.
- **Method:** Queried the named RJob and replica with bounded read-only
  `brainctl get` calls, used `brainctl exec` only on the active r7 replica to
  inspect the runner and child process, read all persistent lane logs, and
  rescanned the output tree for terminal artifacts and fatal markers.
- **Result:** **IN PROGRESS / HEALTHY.** RJob
  `frontier-h200-mixtral-20260825-r7` is `Running`; replica
  `frontier-h200-mixtral-20260825-r7-cfcd2084` is `1/1 Running` on
  `gpu-h200-0234.lgcm.sh.istep.fun`. Attention CUDA_EVENT is `284/284`,
  Linear CUDA_EVENT is `76/76`, and MoE CUDA_EVENT direct is `528/528`.
  Attention `record_function` KERNEL_ONLY advanced from `76/284` to
  `89/284` during the observation window. The runner PID is `414` and the
  active Attention child is `54857`; GPU memory is approximately
  `13,864-18,587 MiB` per H200 out of `143,771 MiB`. No `Traceback`, CUDA OOM,
  Ninja error, permission failure, kill, or segmentation fault was found.
  `launcher_*_status.txt`, `status.json`, `validation.json`, and the accepted
  tree remain absent, so Mixtral remains unaccepted and its E2E remains
  pending. No source, CSV, manifest, README, or preserved untracked file was
  changed.

## Interruption record - Mixtral r7 platform preemption 2026-08-25 20:37 +08:00

- **Motivation:** Classify the unexpected r7 stop before deciding whether the
  only incomplete frozen-scope model needs a controlled retry.
- **Expectation:** The control plane identifies the stop reason, persistent
  logs preserve the partial evidence, and no profiling-level fatal marker
  appears.
- **Method:** Queried the named RJob and replica YAML with bounded read-only
  commands, inspected the platform cleanup annotations and finish time, read
  the launcher control log and persistent lane logs, and scanned all Mixtral
  logs for fatal signatures.
- **Result:** **PREEMPTED / RETRY REQUIRED.** RJob
  `frontier-h200-mixtral-20260825-r7` and replica
  `frontier-h200-mixtral-20260825-r7-cfcd2084` entered `Stopped` at
  `2026-08-25T12:37:37Z` (`20:37:37 HKT`). The replica reports
  `last-clean-policy-execute-method=toPreemptibled` and the job is marked
  best-effort/preemptible. The control-plane event records a
  `CleanPolicyTriggered` GPU-utilization cleanup followed by `Preempted` on
  `gpu-h200-0234`; this is resource reclamation rather than a process error.
  Attention CUDA_EVENT, Linear CUDA_EVENT, and MoE
  CUDA_EVENT direct had completed (`284/284`, `76/76`, `528/528`); Attention
  KERNEL_ONLY stopped at `165/284`. No OOM, traceback, Ninja error, permission
  failure, kill marker, or segmentation fault appears in the persistent logs.
  No `status.json`, `validation.json`, launcher terminal status, or accepted
  tree was produced. The partial r7 output remains immutable evidence; do not
  run `brainctl exec` against this stopped replica.

## Blocking record - Mixtral retry H200 capacity 2026-08-25 20:40 +08:00

- **Motivation:** Verify that a fresh worker can be allocated after the r7
  platform preemption before creating another profiling RJob.
- **Expectation:** The exact verified `infer_af_test` H200 request reports an
  eligible node; a semantic scheduler failure creates no RJob or Replica.
- **Method:** Ran `/kubebrain/rlaunch --predict-only` with unique name
  `frontier-h200-mixtral-predict-20260825-r8`, `8 GPU`, `64 CPU`, `409600 MiB`,
  `--predict-node-num=10`, and `--backoff-limit=1`. Persisted the raw output
  and then queried the prediction name and the stopped r7 job read-only.
- **Result:** **BLOCKED / RETRY LATER.** The command returned wrapper exit
  `0` but the semantic body was `no machine available`; the prediction RJob
  and Replica are `NotFound`. Log:
  `/data/ycfeng/tmp/frontier_h200_mixtral_predict_retry_20260825_r8.log`
  (SHA-256 `da02f4b3fc5c49cbe82bb757cdc6c0d3c689f72c0eb991af2306d7688b3a23e5`).
  No new worker or profiling process exists. Retry the same predict-only gate
  after the platform releases the preempted H200 allocation.

## Blocking record - repeated Mixtral retry capacity gate 2026-08-25 20:45 +08:00

- **Motivation:** Recheck H200 availability after a bounded wait following the
  r7 GPU-utilization preemption.
- **Expectation:** The same exact resource envelope either reports eligible
  H200 capacity or fails semantically without allocating a worker.
- **Method:** Ran a second unique `/kubebrain/rlaunch --predict-only` request
  named `frontier-h200-mixtral-predict-20260825-r8b` with the frozen
  `infer_af_test`, H200, `8 GPU`, `64 CPU`, `409600 MiB` envelope, then listed
  namespace RJobs/Replicas read-only.
- **Result:** **BLOCKED.** The wrapper exit code is `0`, but the scheduler body
  remains `no machine available`; no prediction RJob or Replica exists. Raw
  log: `/data/ycfeng/tmp/frontier_h200_mixtral_predict_retry_20260825_r8b.log`
  (SHA-256 `14f327856333f4dc5ece9efb7ff867d38a34f9f45c9d9b3656d928064a644700`).
  No new profiling process started.

## Blocking record - third Mixtral retry capacity gate 2026-08-25 20:50 +08:00

- **Motivation:** Make one final bounded capacity check after the two prior
  post-preemption failures before escalating the unavailable H200 resource.
- **Expectation:** The verified H200 request reports eligible capacity or fails
  semantically without creating a worker.
- **Method:** Ran unique predict-only name
  `frontier-h200-mixtral-predict-20260825-r8c` with `infer_af_test`, H200,
  `8 GPU`, `64 CPU`, `409600 MiB`, `--predict-node-num=10`, and
  `--backoff-limit=1`; queried no stopped replica and checked for new RJob /
  Replica creation.
- **Result:** **BLOCKED / ESCALATED.** The scheduler body remains
  `no machine available` (wrapper exit `0`), and no prediction RJob/Replica or
  profiling process exists. Log:
  `/data/ycfeng/tmp/frontier_h200_mixtral_predict_retry_20260825_r8c.log`.
  Three consecutive post-preemption capacity checks have failed; stop further
  execution until H200 capacity is restored or the maintainer directs a new
  resource plan.

## Historical/Superseded scope audit record - repository model inventory 2026-08-25

- **Motivation:** Separate the frozen six-model deliverable from unrelated
  model configurations before deciding whether another profiling lane exists.
- **Expectation:** The frozen manifest remains the authoritative collection
  scope; configurations outside it are reported as out of scope rather than
  sampled implicitly.
- **Method:** Counted `data/config/models/*.json` and inspected the H200/H800
  compute directory names, then reconciled the result with the frozen manifest
  and accepted staging inventory.
- **Result:** **PASS.** The repository contains 22 model configuration files,
  while the frozen H200 scope contains exactly six identities. Within that
  scope, Mixtral is the only model lacking formal accepted profiling and formal
  runtime E2E. The H200 canonical directory is still unpublished in this
  worktree; its accepted trees remain in `/data/ycfeng/tmp` pending the later
  publication gate. The seven existing H800 model directories and the other
  16 config files are outside this task's confirmed six-model lane and remain
  untouched.

## Validation record - Mixtral r9/r10 retry lifecycle 2026-08-25 21:04 +08:00

- **Motivation:** Continue the only incomplete frozen-scope model after a
  semantic capacity result, while preserving r7 as immutable partial evidence.
- **Expectation:** A fresh H200 worker either reaches a terminal frozen-manifest
  result or exposes a control-plane failure without contaminating accepted data.
- **Method:** Ran exact H200 predict-only `r9` with
  `infer_af_test`, H200, `8 GPU`, `64 CPU`, `409600 MiB`,
  `--predict-node-num=10`, and `--backoff-limit=1`; the scheduler listed two
  candidates. Submitted RJob `frontier-h200-mixtral-20260825-r9`, which was
  stopped in queue with `Insufficient resource` before replica creation.
  After a fresh predict-only PASS window, submitted RJob
  `frontier-h200-mixtral-20260825-r10` on the same exact resource shape.
  The r10 replica became Ready on `gpu-h200-0078.lgcm.sh.istep.fun` with
  eight H200 GPUs. A short environment/cache probe passed, and the launcher
  was started through detached `brainctl exec` from the sleep-worker shell.
- **Result:** **FAIL / LIFECYCLE BLOCKED.** r9 produced no worker. r10's RJob
  later became `Succeeded`, but no `launcher_mixtral_8x7b_moe_status.txt`,
  `status.json`, `validation.json`, accepted directory, or profiling CSV was
  produced. The only runner output was the generic architecture warning:
  `model_architectures.py:407 ... model_arch=generic`. The r10 launch log is
  `/data/ycfeng/tmp/frontier_h200_mixtral_launch_r10.log` (SHA-256
  `bd1da05e07f9cfa57b85b3881eaf8e04c4aebbc5cb362b575cf8547aaae225e3`), and
  the r10 runner log is under the isolated staging root. The sleep-worker plus
  detached-exec lifecycle is not an acceptable long-running profiling mode;
  the next retry must run the launcher directly as the worker command.

## Blocking record - Mixtral r11 capacity gate 2026-08-25 21:05 +08:00

- **Motivation:** Recheck H200 capacity immediately after the r10 lifecycle
  failure before creating another worker.
- **Expectation:** The exact resource shape reports an eligible node or fails
  semantically without creating a RJob/Replica.
- **Method:** Ran `/kubebrain/rlaunch --predict-only --name=frontier-h200-mixtral-predict-20260825-r11 --namespace=shai-core --charged-group=infer_af_test --private-machine=group --positive-tags=h200 --gpu=8 --cpu=64 --memory=409600 --predict-node-num=10 --backoff-limit=1 -- bash -lc true` and queried the prediction name read-only.
- **Result:** **BLOCKED.** Scheduler body is `no machine available`; wrapper
  exit code is `0`; prediction RJob and Replica are `NotFound`. Persistent log:
  `/data/ycfeng/tmp/frontier_h200_mixtral_predict_retry_20260825_r11.log`
  (SHA-256 `c10202deccb63d8c111b10f57aa4bb24b7a5936f160c2897650708e2d12f42c9`).
  No task-owned H200/H800 worker is active. Stop further allocation until a
  later semantic predict-only PASS or an approved resource-plan change.

## Historical/Superseded checkpoint - remaining-model audit 2026-08-25 21:07 +08:00

- **Motivation:** Answer the continuation request with an evidence-backed
  scope check after the failed retry attempts.
- **Expectation:** Every frozen model maps to an accepted profile tree and a
  formal `prefill=2, decode=2` E2E report, or the missing lane is named
  explicitly; out-of-scope repository configs are not scheduled.
- **Method:** Reconciled the frozen manifest SHA-256, five accepted trees,
  formal runtime report paths, r7/r9/r10/r11 control-plane states, and the
  preserved worktree state.
- **Result:** **PASS / BLOCKED CONTINUATION.** Frozen scope is exactly six
  models. Formal profiling is `5/6` with `46` accepted CSVs and `11,720`
  physical rows; strict accepted-data E2E is `4/6`. Mixtral is the only
  incomplete profiling/E2E lane. Step3 remains surface PASS / strict semantic
  deferred. No source, canonical CSV, README, frozen manifest, or preserved
  untracked file changed.
## Validation record - Mixtral r12 capacity gate 2026-08-25 21:17 HKT

- **Motivation:** Recheck whether H200 capacity had recovered before
  continuing the only incomplete frozen-scope model, while respecting the
  deferred Step3 source decision.
- **Expectation:** The exact `infer_af_test` 8-GPU request either returns an
  eligible node or fails semantically without creating a worker.
- **Method:** Ran the bounded exact predict-only command for
  `frontier-h200-mixtral-predict-20260825-r12` with H200, 8 GPU, 64 CPU,
  409600 MiB, `--predict-node-num=10`, and `--backoff-limit=1`. Queried the
  named RJob and Replica read-only after the command. No `brainctl exec` was
  sent to a stopped worker.
- **Result:** **BLOCKED.** Scheduler output is `no machine available`; the
  wrapper and shell statuses are both `0`, so the semantic scheduler message is
  authoritative. The prediction RJob and Replica are `NotFound`. Log:
  `/data/ycfeng/tmp/frontier_h200_mixtral_predict_retry_20260825_r12.log`
  (SHA-256
  `2ca673019ffa43b1f299bbff7345134c9365cb1783e3b30edb4469e529c5f9f0`).
  Mixtral remains `0` accepted CSV / `0` formal rows and has no formal 2+2 E2E;
  frozen-scope completion remains `5/6` profiling and `4/6` strict E2E.
- **Next boundary:** Keep all accepted trees and the frozen manifest immutable;
  do not allocate a worker until a later semantic predict-only PASS. After
  capacity returns, use a new isolated staging/cache root and run the launcher
  directly as the worker command.

## Validation record - Mixtral r14 cache diagnosis and r15 capacity gate 2026-08-26 13:37 HKT

- **Motivation:** Continue the only incomplete frozen-scope model after the
  H200 capacity gate recovered, while preserving the exact profiling contract
  and avoiding the failed sleep-worker lifecycle.
- **Expectation:** r14 should execute all frozen commands on an 8-GPU H200
  worker; if it failed, the terminal log should identify the first failing
  boundary. A corrected launcher should pass local cache validation and wait
  for a semantic exact predict-only PASS before allocation.
- **Method:** Inspected r14 `status.json`, launcher status, environment and
  Attention CUDA_EVENT log, then compared the cache setup with the earlier
  successful workspace-NFS launcher. Confirmed r14 RJob/Replica reached
  `Succeeded` on `gpu-h200-0949.lgcm.sh.istep.fun`. Created
  `/data/ycfeng/tmp/frontier_mixtral_h200_launcher_20260826_r15.sh` with all
  JIT caches below `/data/ycfeng/tmp/frontier_mixtral_jit_retry_20260826_r15`,
  ran `bash -n`, set mode `755`, and ran a CPU-side `import flashinfer` plus
  directory write probe. Re-ran the exact predict-only shape (`H200`, `8 GPU`,
  `64 CPU`, `409600 MiB`, `infer_af_test`, `--predict-node-num=10`) 17 times
  with unique names and no live allocation.
- **Result:** **FAIL / ENVIRONMENT ROOT CAUSE, THEN CAPACITY BLOCKED.** r14
  stopped at the first command with
  `PermissionError: [Errno 13] Permission denied:
  '/mnt/host0/frontier_mixtral_jit_retry_20260826_r14'`; it produced no
  profiling row, `validation.json`, or accepted CSV. The CPU cache probe
  passed (`FLASHINFER_IMPORT=PASS`, `CACHE_WRITE=PASS`). r15 launcher SHA-256
  is `94e2ac33e79075194ecdf37fdc0b38613d6bc8cccfc93fc6250b0f30a883f310`.
  All 17 exact predict-only attempts returned semantic `no machine available`
  (wrapper exit `0`), so no r15 RJob/Replica exists. Poll log SHA-256:
  `ab902cc5cdf51cf1c52396a1fb6f88640a8c0d085f6a9bf00641e0239d8bbefb`.
- **Next boundary:** Keep r15 unallocated until a later semantic PASS. Then
  run it directly as the sole worker command, validate six terminal commands,
  and perform the independent frozen validator before Mixtral E2E.

## Validation record - Mixtral r19 completion 2026-08-26 16:30 HKT

- **Motivation:** Complete the only remaining frozen-scope profiling and
  runtime lane after H200 capacity became available, while preserving the
  frozen manifest and the deferred Step3 decision.
- **Expectation:** A single corrected-cache H200 worker would finish all six
  frozen commands, produce an accepted Mixtral tree, pass independent file and
  runtime validators, and leave no task-owned worker allocated.
- **Method:** Ran RJob `frontier-h200-mixtral-20260826-r19` on
  `gpu-h200-0735.lgcm.sh.istep.fun` with `8 x H200`, `64 CPU`, and
  `409600 MiB`. Rechecked the producer status and validation artifacts,
  reloaded the frozen manifest and SHA-256, called the repository's attention,
  linear, and MoE validators for both measurement families, ran independent
  preflight and `validate-run`, then sent SIGTERM to the local detached
  launcher after all evidence was persisted.
- **Result:** **PASS.** All six profiling commands have `exit_code=0` and
  `timed_out=false`. The accepted tree contains `10` CSVs and `2,344`
  physical rows. Attention coverage is `188/96/284` (standard/true-mixed/
  combined) per family, linear coverage is `76` per family, and MoE coverage
  is `528` direct rows per family. Minimum positive timings are Attention
  CUDA_EVENT prefill/decode `0.0550080016255378/0.040608000010252 ms`,
  Attention KERNEL_ONLY `0.009377/0.006431 ms`, Linear CUDA_EVENT
  `0.015360000077635 ms`, Linear KERNEL_ONLY `0.0021275 ms`, MoE CUDA_EVENT
  `0.0240799998864531 ms`, and MoE KERNEL_ONLY `0.0032 ms`.
- **Runtime evidence:** The independent report is `PASS` with `1/1` request,
  `prefill=2`, `decode=2`, TTFT `24.30746322544336 ms`, TPOT
  `8.854987900952366 ms`, E2E `33.16245112639572 ms`, zero E2E residual,
  two stage-ledger rows, and `27` positive-duration op-trace events. The
  registry-derived topology is vLLM `TP1/DP2/EP-on`, Frontier attention
  `TP1/DP2`, and MoE `TP1/EP2`.
- **Aggregate:** Independent six-model staging validation is `PASS` with
  `56` CSVs and `14,064` physical rows. Formal accepted-data E2E is `5/6`;
  Step3 remains surface-pass/strict-vLLM-semantic-deferred by maintainer
  decision. Canonical publication, legacy migration, and consumer admission
  remain pending.
- **Worker state:** The named r19 RJob and Replica transitioned to `Stopped`
  after release. No stopped historical replica was accessed with
  `brainctl exec`; no source, README, frozen manifest, accepted tree, or
  preserved untracked file changed.

## Validation record - post-r19 remaining H200 measurement audit 2026-08-26

- **Motivation:** The maintainer reported available H200 capacity and asked to
  complete any remaining measurement tasks after the Mixtral r19 release.
- **Expectation:** A fresh inventory would identify every frozen model lacking
  formal profiling or a formal `prefill=2, decode=2` runtime report before any
  worker allocation.
- **Method:** Re-read the frozen manifest, aggregate staging validator,
  remaining-model audit JSON, all six runtime report paths, manifest hash, Git
  state, and namespace-scoped RJob/Replica inventory. The direct commands and
  numeric acceptance criteria are captured in
  `test_report_2026-08-26_remaining_h200_measurement_audit.md`.
- **Result:** **PASS / no measurement gap.**
  `missing_formal_profile_models=[]` and `missing_runtime_e2e_models=[]`;
  formal profiling is `6/6` with `56` accepted CSVs and `14,064` physical
  rows; all six runtime reports are present with `1/1` completed requests;
  the frozen manifest SHA-256 remains
  `4df580ca1e30a007f45aeed4eb9f5d43593cbab49e59194ecabf8c5996ce8098`; and
  no task-owned H200/H800 worker exists. Step3 remains surface-pass with
  strict semantic acceptance deferred, so an unchanged rerun would not close
  its shared predictor contract defect.
- **Remediation/Verification Code Actions Taken:** Synchronized task docs,
  preserved all accepted trees and untracked files, and kept source,
  canonical CSVs, README files, and the frozen manifest unchanged. No
  `brainctl exec` was sent to a stopped replica.

## Validation record - fresh H200 remaining-measurement recheck 2026-08-27

- **Motivation:** Revalidate the maintainer's report of available H200 capacity
  against the actual frozen six-model scope before starting another worker.
- **Expectation:** Every accepted profile and formal `prefill=2,
  decode=2` runtime artifact should pass the current repository validators; a
  new worker should be scheduled only if a model-level gap appears.
- **Method:** Ran the current `validate_profile_directory` and
  `validate_runtime_artifacts` helpers sequentially for all six frozen models
  from the CPU master (Python `3.12.3`, `PYTHONPATH=$PWD`). Used the exact
  accepted profile leaf directories and runtime `offline_batch` leaf
  directories recorded in the new test report. Queried the namespace-scoped
  H200/H800 RJob and Replica inventory after validation.
- **Result:** **PASS / no remaining measurement lane.** All six profile
  contracts passed and all six runtime contracts passed. Runtime metrics were
  finite and positive for every model, with exactly `1/1` completed request
  and `prefill=2/decode=2`. The frozen aggregate remains `56` accepted CSVs
  and `14,064` physical rows; the manifest SHA-256 remains
  `4df580ca1e30a007f45aeed4eb9f5d43593cbab49e59194ecabf8c5996ce8098`.
  A second full frozen-file pass covered all `56/56` CSVs and returned
  `FULL_DYNAMIC_RECHECK_EXIT=0` (log SHA-256
  `5c09683d4ebb973cd9d81b76758a6419d1f488a07980be81e27ed73f886e2d01`).
  Namespace queries returned no task-owned H200/H800 RJob or Replica.
- **Remediation/Verification Code Actions Taken:** Added
  `test_report_2026-08-27_h200_remaining_measurement_recheck.md` and kept all
  source files, accepted trees, canonical CSVs, README files, frozen manifest,
  and preserved untracked files unchanged. Step3 strict vLLM semantic work
  remains deferred as previously approved.

## Validation record - second-session H200 continuation audit 2026-08-27 16:51 HKT

- **Motivation:** Re-run the requested remaining-measurement check in the
  current session after H200 capacity was reported available, using the
  current validator implementation and the frozen six-model scope.
- **Expectation:** Every model would pass both profile and formal runtime
  validation; the complete accepted trees would retain `56` CSVs and `14,064`
  physical rows; a new worker would be justified only by a missing model-level
  contract.
- **Method:** Ran the six-model `validate_profile_directory` and
  `validate_runtime_artifacts` helpers with `PYTHONDONTWRITEBYTECODE=1`,
  `PYTHONPATH=$PWD`, and a `300s` timeout. The final output is
  `/data/ycfeng/tmp/frontier_h200_remaining_measurement_recheck_20260827_session2_final.log`.
  Its SHA-256 is
  `4283831a05c555e46c155f027a9941835846c5dacf17bf6d0806c3998b512701`.
  The helper-selected contract slice was counted separately from all CSVs in
  each accepted leaf. Namespace-scoped RJob/Replica inventory was read
  without control actions.
- **Result:** **PASS / no remaining measurement lane.** All six profile and
  runtime validators returned `PASS`; each runtime report has `1/1` completed
  request with `prefill=2` and `decode=2` and finite positive metrics. The
  contract slice is `32` CSVs / `9,504` rows, while the full accepted trees are
  `56` CSVs / `14,064` rows. Final assertion output is
  `RECHECK_SESSION2_FINAL_EXIT=0`. The namespace query shows zero task-owned
  H200/H800 RJobs and zero Replicas.
- **Remediation/Verification Code Actions Taken:** Corrected two local
  aggregation assumptions exposed by exploratory wrappers (missing helper
  summary keys and slice-versus-tree row totals); the corrected wrapper passed
  without changing source, accepted data, canonical CSVs, README files, the
  frozen manifest, or preserved untracked files. An incidental shell quoting
  error invoked `brainctl exec` with no target; it failed at argument parsing
  and contacted no stopped replica or altered control-plane state. Step3
  strict-vLLM predictor/topology work remains deferred.

## Historical/Superseded validation record - latest-main H200 compatibility audit 2026-08-27

- **Motivation:** Determine whether the completed H200 data can be merged into
  the current remote `main` with a controlled data-only adaptation before any
  push or PR operation.
- **Expectation:** Freshly fetched `origin/main` should match local `main`, all
  six frozen profile/runtime lanes should remain complete, and the latest-main
  consumers should admit the producer metadata without a cross-module change.
- **Method:** Ran `timeout 60s git fetch origin main`, then ran the read-only
  script `/data/ycfeng/tmp/audit_h200_latest_main_compatibility_20260827.py`
  with Python `3.12.3` and pandas `3.0.3`. The exact output is persisted at
  `/data/ycfeng/tmp/h200_latest_main_compatibility_audit_20260827.log`; the
  existing consumer probe is
  `/data/ycfeng/tmp/h200_origin_main_consumer_probe_20260827_v2.log`.
- **Result:** **PASS for synchronization and producer data; FAIL for direct
  latest-main admission.** `main` and `origin/main` both resolve to
  `a24cedabedcc7bd374073fd508dcf770c860ede5` (`0/0` divergence). The frozen
  six-model inventory has no missing profile or runtime model; accepted trees
  total `56` CSVs and `14,064` physical rows, and each runtime report has
  `1/1` request with `prefill=2` and `decode=2`. Branch CSVs use canonical
  contexts `direct` and `prefill_warmed`, while latest-main defaults to
  `standalone_legacy`/`prefill_hot`; the historical narrow consumer probe
  records `64` context admission failures. The corrected isolated full matrix
  `/data/ycfeng/tmp/h200_origin_main_consumer_matrix_20260827_isolated.json`
  (SHA-256
  `82eb638685f2f1b2e785f02ce107bf1c20ebae20877f9fd054ccdfdf5b5c8fd7`)
  records `96` unique independent gating-op failures (`3` MoE models x `16`
  TP/EP pairs x `2` gating ops); the same JSON has `48` repeated shared-FFN
  training statuses, so counting every context-dependent status gives `144`
  entries rather than `96` unique gating admissions. `step3-moe-noquant.json` exists on the branch but not in
  `origin/main`, producing one explicit unknown-model failure. A merge-tree
  preview exits `1` with conflicts in the sklearn MoE predictor,
  `frontier/profiling/README.md`, and the operator-parity merge utility. The
  branch changes `34` files (`4,509` insertions, `247` deletions), so the
  adaptation is a cross-module contract migration rather than a CSV-only
  adjustment.
- **Attention coverage note:** Latest-main accepts the standard attention
  files because their marker columns are present, but each standard file has
  `47` TP1 rows and `0` true-mixed rows; the paired combined input has `71` TP1
  rows and `24` true-mixed rows. The documented supplement merge remains a
  publication gate to prevent silent coverage loss.
- **Remediation/Verification Code Actions Taken:** Added the dedicated
  compatibility test report, recorded the exact evidence and decision in the
  task docs, and kept source, accepted trees, canonical CSVs, frozen manifest,
  README files, and preserved untracked files unchanged. No remote push, PR,
  review comment, merge, or re-profiling was executed because the PR-ready
  condition is not met. The next decision is a maintainer checkpoint for a
  separate contract migration, a scoped data publication, or an approved
  latest-main reprofile.

## Historical/Superseded verification record - documentation sync and final compatibility gate 2026-08-27

- **Motivation:** Close the latest-main review checkpoint with fresh evidence
  after synchronizing the task records, while preserving the accepted H200
  artifacts and the maintainer's no-PR boundary.
- **Expectation:** The ref comparison, frozen profile/runtime validators,
  attention row audit, README hashes, whitespace check, and preserved-file
  inventory would all confirm the recorded state without changing source or
  control-plane resources.
- **Method:** Re-ran `timeout 60s git fetch origin main`, the read-only audit
  `/data/ycfeng/tmp/audit_h200_latest_main_compatibility_20260827.py`, the six
  `validate_profile_directory`/`validate_runtime_artifacts` calls with
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD`, the attention CSV row-count
  probe, `sha256sum -c
  /data/ycfeng/tmp/frontier_profiling_governance_readme_baseline_20260823.sha256`,
  and `git diff --check`.
- **Result:** **PASS for the recorded gates; PR gate remains blocked.**
  `main` and `origin/main` are both
  `a24cedabedcc7bd374073fd508dcf770c860ede5` with `0/0` divergence. All six
  profile and runtime validators returned `PASS`; every runtime report has
  `1/1` request with `prefill=2` and `decode=2`; aggregate coverage remains
  `56` CSVs and `14,064` rows. Standard attention files have `47` TP1 rows and
  `0` true-mixed rows, while combined files have `71` TP1 rows and `24`
  true-mixed rows. README hashes are `9/9` `OK`, `git diff --check` is clean,
  and `=10.1`, the Step3 runtime config, and the Step3 launcher remain
  present. The latest-main audit still reports the historical narrow `64` count;
  the corrected isolated full matrix reports `96` unique gating-op failures
  (or `144` status entries including repeated shared-FFN failures), one missing
  Step3 config, and three merge-tree conflicts, so no remote action or
  re-profiling is authorized by the current gate.
- **Remediation/Verification Code Actions Taken:** Updated
  `summary.md`, `review.md`, `plan.md`, `issues.md`, `notes.md`, and this
  report with the compatibility and attention findings. No source, canonical
  CSV, accepted staging tree, frozen manifest, README, untracked file, or
  worker state was modified.

## Verification record - exploratory latest-main compatibility export 2026-08-27

- **Motivation:** Determine whether the latest-main failures are caused by the
  measured values themselves or by the branch's renamed MoE context contract,
  without mutating accepted data or source.
- **Expectation:** A detached export that restores the legacy context labels
  and supplies the missing Step3 config should pass the latest-main low-level
  loader/training slices if the physical measurements remain compatible.
- **Method:** Compared the accepted trees with
  `/data/ycfeng/tmp/h200_latest_main_data_only_probe_20260827` and recorded the
  field-level result in
  `/data/ycfeng/tmp/h200_latest_main_data_only_diff_20260827.log`. The temporary
  export rewrote only `gating_runtime_context` for four MoE models (eight CSVs,
  `6,336` rows), then ran the detached latest-main context and full probes
  against `/data/ycfeng/tmp/frontier_main_data_only_config_probe_20260827`.
- **Result:** **PASS for the bounded technical probe; NOT a publication pass.**
  Every changed file reported only the context column difference, with all
  shape, TP/EP, measurement, and timing fields unchanged. The detached full
  probe reports `PASS` for all six model configs, attention standard/combined
  loader slices, and the selected independent/shared MoE training slices. The
  original latest-main probe still records the historical narrow `64` context
  admission failures; the corrected isolated full matrix records `96` unique
  gating-op failures (with `48` repeated shared-FFN statuses) and one missing
  Step3 config because it consumes the unconverted branch data.
- **Remediation/Verification Code Actions Taken:** Recorded the probe as an
  exploratory compatibility-export option in the latest-main audit and kept
  the canonical/accepted trees, frozen manifest, source files, README files,
  and preserved untracked files unchanged. The export requires an explicit
  provenance manifest, naming policy, complete runtime/cache admission, and
  merge-conflict resolution; it does not make this branch PR-ready.

## Validation record - latest-main branch-state recheck 2026-08-27 18:55 HKT

- **Motivation:** Independently verify whether a PR made from the current
  branch would actually carry the accepted H200 measurements after the latest
  `main` synchronization.
- **Expectation:** `main` and `origin/main` should remain synchronized; the
  branch payload should expose the canonical H200 CSVs if the branch is ready
  for publication.
- **Method:** Ran `git fetch origin main`, then recorded
  `git rev-parse main origin/main HEAD`,
  `git rev-list --left-right --count main...origin/main`,
  `git rev-list --left-right --count origin/main...HEAD`,
  `git merge-base origin/main HEAD`, and
  `git ls-tree -r --name-only HEAD data/profiling/compute/h200`. The exact
  output is `/data/ycfeng/tmp/h200_branch_latest_main_state_20260827_session3.log`
  (SHA-256
  `2015288305449a72fe4bf76afbebc51995476ecd8553feca0989f611d9e6ba7b`).
- **Result:** **PASS for ref synchronization; FAIL for branch payload
  readiness.** `main` and `origin/main` are both
  `a24cedabedcc7bd374073fd508dcf770c860ede5` with `0/0` divergence. The
  profiling branch is `173` commits behind and `11` commits ahead of
  `origin/main` (common ancestor `8cc267ead605276090345a788eabbc60f526f8a0`),
  and its Git tree contains zero H200 CSV files. The accepted `56`-CSV,
  `14,064`-row H200 trees remain staging-only under `/data/ycfeng/tmp`.
- **Remediation/Verification Code Actions Taken:** Recorded the payload and
  ancestry evidence, kept staging and frozen artifacts unchanged, and held
  PR/push/review/merge/re-profiling actions behind the maintainer checkpoint.

## Verification record - latest-main recheck 2026-08-28

- **Motivation:** Reconfirm the latest-main compatibility decision after a
  fresh remote fetch and ensure that no new H200 measurement gap or branch
  payload has appeared.
- **Expectation:** The synchronized refs, six accepted profile/runtime lanes,
  latest-main consumer behavior, and merge feasibility should reproduce the
  recorded gate without changing protected artifacts.
- **Method:** Ran `timeout 60s git fetch --prune origin main`, the read-only
  audit `/data/ycfeng/tmp/audit_h200_latest_main_compatibility_20260827.py`,
  six branch preflight validators, the independent latest-main context probe
  `/data/ycfeng/tmp/h200_origin_main_context_recheck_20260828.py`, the three
  focused contract test files, `git diff --check`, and
  `git merge-tree --write-tree --messages origin/main HEAD`.
- **Result:** **PASS for data and branch-local tests; PR gate remains
  blocked.** `main` and `origin/main` are both
  `a24cedabedcc7bd374073fd508dcf770c860ede5` with `0/0` divergence. All six
  preflights pass (`56` CSVs, `14,064` rows), while the latest-main context
  probe rejects `32` gating-op slices in each of three MoE models (`96`
  unique failures) and cannot load `step3-moe-noquant`. Standard attention
  remains `0` true-mixed rows versus `96` in combined files. The branch is
  `173` behind/`11` ahead, tracks `0` H200 CSV files, and has three merge-tree
  conflicts. Focused tests report `33 passed in 11.11s`; no remote or accepted
  data mutation occurred.
- **Remediation/Verification Code Actions Taken:** Added
  `test_report_2026-08-28_latest_main_recheck.md` and retained the
  no-PR/no-push/no-reprofile checkpoint. No source, README, staging tree,
  canonical CSV, manifest, or preserved untracked file was changed.

## Verification record - isolated latest-main consumer provenance recheck 2026-08-28

- **Motivation:** Confirm that the latest-main consumer failure is reproduced
  by the latest-main implementation itself, rather than by accidental import
  of the profiling branch from the worktree cwd.
- **Expectation:** Running from the detached latest-main snapshot with an
  explicit snapshot `PYTHONPATH` should import the snapshot module and retain
  the previously observed context/config admission result.
- **Method:** Ran the recheck from
  `/data/ycfeng/tmp/frontier_main_audit_20260827` with
  `PYTHONPATH=/data/ycfeng/tmp/frontier_main_audit_20260827`. Compared the
  imported `frontier/moe_gating_runtime.py` with the snapshot and persisted
  the output at `/data/ycfeng/tmp/h200_origin_main_context_recheck_20260828_isolated.json`
  (SHA-256 `369b20b5134519391ffa25cf65d97dfe74618402a16d3c7b9ba77f418b95d910`).
- **Result:** **PASS for provenance correction; compatibility gate remains
  blocked.** The snapshot module is byte-identical and defaults to
  `standalone_legacy`. The isolated run reproduces `96` unique gating-op
  failures (`32` per Qwen3-235B-A22B, qwen3-a3b-30b-moe, and Mixtral) and the
  missing latest-main Step3 config. The accepted H200 trees and frozen
  manifest remain unchanged.
- **Remediation/Verification Code Actions Taken:** Added
  `test_report_2026-08-28_latest_main_isolated_consumer_recheck.md` and
  `review.md` Checkpoint 33. Marked the earlier worktree-cwd invocation as
  non-independent evidence and retained the corrected isolated artifact as
  authoritative. No source, CSV, README, remote ref, PR, merge, or
  re-profiling action occurred.

## Verification record - clean isolated probe artifact 2026-08-28

- **Motivation:** Make the fresh isolated probe output machine-readable while
  preserving the runner's original stdout and its architecture warnings.
- **Expectation:** Extracting only the JSON array from the immutable raw log
  should yield valid JSON with the same six-model and `96`-failure totals.
- **Method:** Kept the raw rerun at
  `/data/ycfeng/tmp/h200_origin_main_context_recheck_20260828_isolated_rerun.json`
  and used the documented `awk` boundary (`[` through before
  `total_unique_gating_failures`) to write
  `/data/ycfeng/tmp/h200_origin_main_context_recheck_20260828_clean.json`.
  `python3 -m json.tool` then parsed the clean file.
- **Result:** **PASS.** The clean artifact parses as JSON, contains `6` model
  records, totals `96` unique gating failures, and records Step3 config
  `FAIL`. Raw and clean SHA-256 values are
  `d54956c3cd9f6ada649d5316b9b17898182fda8c1fa43639302d6cf360340625` and
  `750fbdc84ae521b2ea1b6c61626a19545f61e3f886dba9ba4efa02b21a4be456`.
- **Remediation/Verification Code Actions Taken:** Updated the isolated
  consumer report to distinguish raw versus clean evidence. No source,
  accepted CSV, manifest, README, or remote state changed.

## Verification record - current merge snapshot contract rerun 2026-08-28

- **Motivation:** Recheck the branch-local contract tests after the latest
  main synchronization and distinguish an integration/API failure from a
  malformed H200 measurement.
- **Expectation:** The focused contract matrix should construct all six model
  contracts and exercise the frozen profile fixtures; any failure should name
  the current caller/consumer contract that prevents admission.
- **Method:** From the profiling worktree, ran:

  ```bash
  timeout 180s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
    python3 -m pytest -q -p no:cacheprovider \
    tests/unit/test_moe_gating_runtime_context_aliases.py \
    tests/unit/test_operator_parity_profile_context_merge.py \
    tests/unit/test_h200_six_model_non_dummy_e2e_contract.py \
    tests/unit/test_parallel_semantics.py
  ```

  Environment: `/usr/bin/python3` (Python `3.12.3`), pandas `3.0.3`.
  The read-only state checks in the same session recorded
  `main == origin/main == a24cedabedcc7bd374073fd508dcf770c860ede5`, a
  branch distance of `173` behind/`11` ahead, zero unmerged index entries,
  and `12` staged trailing-whitespace findings from `git diff --cached --check`.
- **Result:** **FAIL / integration blocker.** Pytest completed with `47 passed,
  10 failed` in `133.00s`. Every failure occurs while constructing
  `ModelContract`, before profile-row assertions. The common exception is:
  `ValueError: Unsupported moe_routing_distribution_type='simulation'.
  Expected 'balanced', 'random', 'skewed', or 'zipf'.` One explicit test also
  passes `uniform_random`, which the resolver likewise rejects. The failure
  is in the validator caller vocabulary, not in timing values or CSV shape.
  `git ls-files -u` is empty, but `MERGE_HEAD` remains present, so the merge
  still requires a commit and the cached whitespace gate remains open.
- **Remediation/Verification Code Actions Taken:** Recorded the fresh output
  in `test_report_2026-08-28_merge_snapshot_contract_rerun.md`. Preserved the
  accepted H200 staging trees, frozen manifest, canonical data, README files,
  untracked `=10.1`, and remote state. No source edit, merge commit, push, PR,
  remote review comment, or re-profiling was performed.

## Verification record - Option 2 RCA re-review and implementation start 2026-08-28

- **Motivation:** Reconfirm that the approved Step3 fix addresses the measured
  semantic failure at its source and remains compatible with the merged
  latest-main runtime contracts before changing production code.
- **Expectation:** The source inspection and RED regression should identify a
  single missing mixed-layer contract, with no evidence requiring a layer-count
  change, a model-name branch, a timing scale factor, or mutation of accepted
  H200 data.
- **Method:** Inspected `data/config/models/step3-moe-noquant.json`,
  `frontier/config/model_config.py`,
  `frontier/profiling/common/model_config.py`, the operator family registry,
  predictor/shared-manager consumers, `frontier/utils/param_counter.py`, and
  the focused consumer test. The current RED command is:

  ```bash
  env PYTHONPATH="$PWD" pytest -q \
    tests/unit/test_operator_query_tp_consumers.py::test_mixed_moe_dense_ffn_uses_attention_tp_domain \
    -p no:cacheprovider
  ```

- **Result:** **RCA confirmed; Option-2 direction recorded, implementation not
  authorized in this checkpoint.** The
  Step3 config has `num_hidden_layers=61`, dense layers `0,1,2,3,60`, routed
  layers `4..59`, dense width `18432`, routed/shared width `5120`. The runtime
  loader collapses every MoE layer to `mlp_hidden_dim=5120` at
  `frontier/config/model_config.py:590-602`; the profiling config exposes only
  one `mlp_hidden_dim` at `frontier/profiling/common/model_config.py:22-79`.
  The registry declares `FFN_TP` for dense/shared-expert operators and
  `MOE_TP` for routed operators, but the predictor consumer still uses the
  model-level MoE flag. The focused RED assertion fails with actual TP `1`
  versus expected TP `8`, proving the defect is the missing layer/family-aware
  binding rather than a runtime scheduler, routing, EP, or profiling-value
  defect.
- **Self-check:** Option 2 preserves official layer count and all existing
  runtime paths; it extends the existing registry rather than introducing a
  parallel classifier; it fails fast for unknown layer IDs, missing dimensions,
  and invalid TP/EP divisibility; it preserves pure dense/pure MoE behavior;
  and it leaves accepted H200 CSVs and the frozen manifest immutable. The
  accepted Step3 CSV has only `5120` linear-width rows, so the repair will
  expose the missing dense-`18432` coverage as a data gap instead of fabricating
  rows. Work proceeds in the recorded order:
  `validator caller contract -> RED resolver matrix -> config/profiling fields
  -> predictor/training/cache/trace/ParamCounter consumers -> focused matrix`.

## Verification record - Option 2 independent RCA re-review 2026-08-29

- **Motivation:** Re-check the approved Option-2 diagnosis from raw current
  code, configuration, accepted CSV metadata, and a focused regression before
  any shared production contract is changed.
- **Expectation:** The review should distinguish the parameter-memory topology
  concern from the semantic lookup defect, prove whether a TP-only edit is
  sufficient, and identify any missing measurement tuple without mutating
  accepted data.
- **Method:** Used the current worktree at `HEAD=df499454ecd657f176f746046d2a404ec6a82d3`
  with `/usr/bin/python3` Python `3.12.3`. Re-read the Option-2 requirements,
  model/config loaders, operator registry, predictor consumers, layer-aware MoE
  path, aggregate path, and accepted Step3 profile tree. Ran:

  ```bash
  timeout 180s python3 -m pytest \
    tests/unit/test_operator_query_tp_consumers.py::test_mixed_moe_dense_ffn_uses_attention_tp_domain \
    -q -p no:cacheprovider
  ```

  and direct probes for `BaseModelConfig.create_from_name`,
  `profiling.common.ModelConfig.from_model_name`, and both accepted linear
  CSVs.
- **Result:** **BLOCK / REVISE.** The RED test fails with actual TP `1` versus
  expected TP `8`. Both loaders report `num_layers=61`, MoE IDs `4..59`,
  `mlp_hidden_dim=5120`, and `share_expert_dim=5120`, despite the checked-in
  dense `intermediate_size=18432`. `linear_op.csv` and
  `linear_op_kernel_only.csv` each contain `76` rows with width set
  `['5120']` and `0` rows at `18432`. The evidence proves a typed
  layer/width/TP contract gap and rules out a predictor-only or timing-scaling
  patch as a complete fix.
- **Self-check:** The corrected plan retains 61 layers and the existing
  registry, routing, EP, cache, MTP, and scheduler paths; it rejects model-name
  branches, copied rows, synthetic metadata, and layer64/PP fallback. Pure
  dense/pure MoE compatibility remains unverified until migration tests run.
  The source change crosses shared interfaces, so the escalation gate remains
  active. No source, accepted CSV, manifest, README, untracked `=10.1`, GPU
  worker, or remote state changed.
- **Next gate:** Obtain maintainer confirmation for the typed-layer contract
  owner/API and separate authorization for dense-`18432` H200 profiling. Then
  execute the report's RED -> resolver -> consumer migration -> GREEN -> strict
  E2E sequence.

## Decision record - mixed-layer aggregate policy A 2026-08-28

- **Motivation:** Close the remaining aggregate design fork before the shared
  Step3 contract migration and preserve an auditable boundary for work that
  cannot be implemented without a wider representation change.
- **Expectation:** The selected policy should stop incorrect mixed-stage
  predictions at the aggregate seam, preserve pure-model behavior, and state
  exactly what option B would change in a later version.
- **Method:** Reviewed `ExecutionTime`'s scalar-layer aggregate, the scheduler's
  existing per-layer `layer_id` propagation, the typed Step3 layer map, and the
  two reviewed designs. Recorded the maintainer's selection of A in
  `requirements.md`, `design.md`, `plan.md`, and `issues.md`.
- **Result:** **DECISION RECORDED.** Mixed multi-layer FFN calls will require
  explicit `layer_id`/`layer_ids` and fail fast when absent or invalid. Pure
  dense, pure MoE, and attention-only aggregates keep their current behavior.
  Option B (exact aggregate from complete layer identity or PP bounds) is
  deferred. A's identity-free limitation and the missing Step3 dense `18432`
  coverage remain open issues; no production source, accepted CSV, manifest,
  README, remote ref, or preserved untracked file changed in this documentation
  step.
- **Remediation/Verification Code Actions Taken:** Added the decision and
  future-work record to the task documents. The next code action is a minimal
  RED matrix for layer identity, typed dimensions, and TP/EP domains.

## Documentation-only continuation checkpoint - measurement closure and deferral 2026-08-28

- **Motivation:** Continue the requested H200 work while recording the selected
  aggregate option A, the deferred option B design, and the unresolved Step3
  strict-vLLM semantics without claiming an unimplemented source fix.
- **Expectation:** A fresh inventory check should show no remaining formal
  profiling or ordinary runtime E2E lane; source inspection should distinguish
  the approved contracts from code that is actually present.
- **Method:** Ran the reproducible inline Python probe recorded in
  `test_report_2026-08-28_option_a_b_deferral_and_measurement_closure.md` under
  Python `3.12.3`. Read the immutable remaining-model audit, inspected the
  resolver AST, `ExecutionTime` constructor, and scheduler aggregate call, and
  rechecked `main`/`origin/main` and the worktree index.
- **Result:** **PASS for measurement closure, BLOCKED for source-contract
  completion.** The inventory reports six models, empty missing-profile and
  missing-runtime lists, `56` CSVs, `14,064` physical rows, six runtime
  reports, and six completed requests. The resolver has no explicit
  multi-layer identity argument and still returns the model-level MoE result
  for `num_layers != 1`; `ExecutionTime` and the scheduler aggregate call have
  no `layer_ids`. Step3 strict semantic acceptance remains deferred.
- **Remediation/Verification Code Actions Taken:** Updated the task documents
  to mark the option-2 resolver and option-A production guard as deferred source
  work, documented option A's identity-free limitation and missing dense
  `18432` coverage, and preserved option B as a later exact-aggregation task.
  No production source, accepted CSV, frozen manifest, README, remote ref, or
  preserved untracked file changed.

## Verification record - sklearn MoE typed routed-width consumer 2026-08-29

- **Motivation:** The MoE predictor still used `model_config.mlp_hidden_dim`
  for both dataset contract filtering and runtime load-imbalance features. A
  mixed-layer model can expose a different routed width, so this path could
  reject valid rows or compute a training/runtime feature mismatch.
- **Expectation:** A profile-backed MoE config resolves the routed contract
  through `moe_grouped_gemm`, selects its effective width for
  `expert_hidden_dim`, and preserves legacy fixtures that do not expose an
  architecture profile.
- **Method:** Added
  `tests/unit/test_sklearn_moe_typed_contract.py` with a Step3-like config
  (`mlp_hidden_dim=9999`, `routed_mlp_hidden_dim=5120`) and two direct cases:
  dataset filtering and runtime feature construction. The RED run failed at
  `sklearn_moe_execution_time_predictor.py:1205` with the observed requirement
  `expert_hidden_dim=9999` and no matching rows. The minimal fix adds
  `_resolve_routed_moe_layer_contract()` and reuses the existing
  `ModelArchitectureProfile.resolve_layer_contract()` resolver in both
  consumers. Verification command:

  ```bash
  timeout 240s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
    python3 -m pytest -q -p no:cacheprovider \
    tests/unit/test_sklearn_moe_typed_contract.py \
    tests/unit/test_moe_share_expert_operator_families.py \
    tests/unit/test_moe_finite_prediction_lookup.py \
    tests/unit/test_typed_ep_predictor_contract.py \
    tests/unit/test_predictor_effective_tokens.py
  ```

  Environment: `/usr/bin/python3` (Python `3.12.3`), pandas `3.0.3`.
- **Result:** **PASS.** The focused consumer file reports `2 passed`; the
  combined regression command reports `101 passed in 4.85s`. The routed
  contract resolves to width `5120`, and runtime features report
  `expert_hidden_dim=5120` with `model_expansion_ratio=5.0` for hidden width
  `1024`. Legacy no-profile predictor fixtures remain green. Accepted CSVs,
  frozen manifests, workers, README files, and protected untracked files were
  unchanged.
- **Remediation/Verification Code Actions Taken:** Recorded this sub-step in
  `test_report_2026-08-29_sklearn_moe_typed_contract.md` and left the next
  pending consumer as the shared-manager cluster model-contract view and its
  `add` alias path.
