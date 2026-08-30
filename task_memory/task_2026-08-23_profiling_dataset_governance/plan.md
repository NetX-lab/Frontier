## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-08-29 | Completed the first runtime identity propagation boundary: ReplicaStageScheduler now derives global PP layer IDs and forwards them to multi-layer predictor calls; focused regressions pass. |
| 2026-08-29 | Completed the predictor typed-width sub-step with registry-owned dense/routed/shared filtering and per-contract dataframe caching; `152` focused regressions pass. |
| 2026-08-29 | Completed the predictor consumer sub-step: profile-owned typed dense/routed/shared TP resolution is GREEN with `161` focused tests; manager/config/training/cache/trace/runtime migrations remain pending. |
| 2026-08-29 | Recorded confirmation of option 1: extend the existing architecture profile registry with a reusable typed layer contract and keep a second global registry out of scope. |
| 2026-08-29 | Historical wording superseded: the source-contract phase uses profile-owned option 1 while keeping dense-18432 profiling and publication separately gated. |
| 2026-08-29 | Completed the independent Option-2 RCA re-review and recorded the maintainer-approved dedicated typed-layer ownership; source execution is in progress while dense-18432 profiling remains separately gated. |
| 2026-08-28 | Closed the frozen-scope measurement audit and changed the Step3/aggregate source-contract phase to documentation-only deferral: option A is selected, option B and A's unresolved limits are reserved for a later version. |
| 2026-08-28 | Confirmed aggregate option A and recorded its fail-fast boundary, known limitations, and deferred exact option-B design. |
| 2026-08-28 | Added the direct validator recheck gate: current contracts/profile trees pass 6/6, while runtime metadata admission passes 2/6 and remains part of the cross-module migration. |
| 2026-08-28 | Started the approved option-2 implementation after evidence-backed RCA: resolve Step3 layer identity, dimensions, and TP domains through the shared registry while preserving 61 layers and strict vLLM semantics. |
| 2026-08-27 | Reconciled the latest-main admission metric: the narrow historical probe reports 64 failures, while the isolated full matrix reports 96 unique gating-op failures plus 48 repeated shared-FFN statuses. |
| 2026-08-27 | Reconciled the remote PR facts (`#17/#20/#21` merged), recorded the final compatibility-audit rerun, and kept the branch at the no-PR checkpoint. |
| 2026-08-27 | Closed the latest-main compatibility review as a no-PR outcome: producer validation passes, while consumer admission and complete attention publication require a separately approved cross-module migration. |
| 2026-08-27 | Added the latest-main compatibility gate; the six-model H200 data is complete but consumer admission requires a separate cross-module contract migration. |
| 2026-08-26 | Closed the frozen-scope H200 measurement phase after a fresh no-gap audit; publication and consumer admission are now the next executable gates. |
| 2026-08-26 | Completed Mixtral r19 H200 profiling, independent frozen-manifest revalidation, formal `2+2` E2E, six-model staging revalidation, and worker release; canonical publication remains the next gate. |
| 2026-08-26 | Diagnosed Mixtral r14 as a worker-cache permission failure, prepared r15 with a writable workspace-NFS cache, and re-established the exact predict-only gate before any new allocation. |
| 2026-08-25 | Ran the r12 exact H200 predict-only gate; scheduler still reports `no machine available`, so Mixtral remains blocked and docs are synchronized without source changes. |
| 2026-08-25 | Reconciled the remaining-model audit: Mixtral is the only incomplete frozen-scope model; r7 was preempted, r9 stopped in queue, r10 ended without terminal profiling artifacts, and r11 capacity gating failed. |
| 2026-08-25 | Synchronized the live r7 checkpoint: HEAD `9b24c7c`, three preserved untracked files, and Mixtral KERNEL_ONLY profiling remains the only active model lane. |
| 2026-08-25 | Started the Mixtral r7 formal profiling continuation after the inventory audit; Step3 predictor and layer64+PP changes remain deferred. |
| 2026-08-25 | Deferred the cross-cutting Step3 predictor fix by maintainer request, completed the six-model inventory audit, and advanced the next executable lane to Mixtral profiling. |
| 2026-08-25 | Recorded the option-A semantic audit: surface request execution passed, but the monolithic independent predictor used TP1 for dense boundary `mlp_*`; a family-aware predictor approval gate now precedes the corrected A rerun and any layer64+PP fallback. |
| 2026-08-25 | Applied the approved Step3 retry order: strict vLLM option A first; layer64 plus PP remains conditional on an A admission/runtime failure. |
| 2026-08-24 | Added the strict vLLM EP-on Step3 review gate and four 8-GPU candidate mappings; topology contract changes remain pending maintainer approval. |
| 2026-08-24 | Re-ran the fixed Step3 accepted-data E2E; preflight passed, runtime admission reproduced the parameter-memory blocker, and Mixtral launch is gated on maintainer disposition. |
| 2026-08-24 | Completed Step3 H200 profiling, independent frozen-manifest validation, and the 2,344-row accepted staging tree; the next gate is Step3 2+2 E2E alongside Mixtral profiling. |
| 2026-08-24 | Closed qwen3-a3b profiling, frozen-manifest validation, and formal 2+2 E2E; released the completed H200 worker and advanced the lane to Step3. |
| 2026-08-24 | Resumed the H200 lane after a successful exact predict-only gate and started the live qwen3-a3b retry on an 8-GPU worker; profiling is in progress. |
| 2026-08-24 | Added the live H200 quota-gate result: predict-only returned a candidate, while the exact RJob remained queued with `H200=0` and produced no worker. |
| 2026-08-24 | Generated and validated archive/coverage manifests as a read-only preparation step; active-data migration remains gated after six-model collection. |
| 2026-08-24 | Independent control-plane recheck confirmed no active H200/H800 worker and repeated the semantic H200 capacity failure. |
| 2026-08-24 | Repeated the H200 predict-only gate after handoff; the scheduler still reports no eligible machine, so the next-model launch remains pending. |
| 2026-08-24 | Repeated the H200 predict-only gate for `qwen3-a3b-30b-moe`; the scheduler still reports no eligible machine, so the live launch remains blocked. |
| 2026-08-24 | Reconciled the formal Qwen E2E completion, current worker absence, and exact cross-session resume order. |
| 2026-08-24 | Synchronized the plan to the three-model H200 checkpoint, ended-worker state, and cross-session resume order. |
| 2026-08-23 | Resolved the naming gate with option A and inserted a TDD compatibility-alias implementation checkpoint. |
| 2026-08-23 | Froze the H200 numeric envelope and inserted the gating-context naming decision before exact-manifest generation and formal profiling. |
| 2026-08-23 | Completed the H200 runtime-contract and attention-dedup prerequisites before freezing exact manifests. |
| 2026-08-23 | Completed six-model config normalization and moved H200 capability smoke ahead of the final TP/EP envelope decision. |
| 2026-08-23 | Replaced the provisional H200 scope with the confirmed six-model profiling and per-model E2E pipeline. |
| 2026-08-23 | Reordered execution to complete H200 first and defer H800 until explicit maintainer notice. |
| 2026-08-24 | Closed Qwen3-235B-A22B formal E2E and paused the next H200 model at the semantic no-machine predict-only gate. |
| 2026-08-23 | Created the execution plan for active-data cleanup and parallel H800/H200 profiling. |

## Current execution authority (2026-08-29)

Option 1 is the active ownership decision: extend the existing
`ModelArchitectureProfile`/`ModelArchitectureRegistry` with the typed layer
contract and one resolver. Historical option-2 labels in older checkpoints
describe the same semantic need but are superseded as an ownership choice.
The source-contract phase is authorized; accepted H200 data, frozen manifests,
README files, and remote state remain immutable and dense-`18432` profiling
remains separately gated.

# Plan

## Scope

Govern `data/profiling/compute` as the directly usable public surface, archive
nonconforming data under `data/profiling/legacy/compute`, and add validated
H800/H200 datasets for explicitly declared option-A envelopes.

## Dependency Map

`clean worktree -> task baseline -> six-model config/capability audit -> {attention stable dedup, non-dummy E2E contract} -> freeze numeric envelope -> gating-context naming decision -> central registry -> freeze exact manifest -> llama3.1 profiling -> {llama3.1 E2E, llama3.3 profiling} -> llama3.3 E2E -> Qwen3-235B profiling/recovery/assembly -> Qwen3-235B accepted-data E2E -> fresh H200 worker -> qwen3-a3b profiling -> qwen3-a3b staging validation -> {qwen3-a3b E2E, step3 profiling} -> step3 staging validation -> step3 semantic audit -> mixtral profiling -> mixtral staging validation -> mixtral E2E -> H200 publication audit -> latest-main compatibility audit -> documentation-only closure -> maintainer checkpoint`

The caller migration, profile-owned typed Step3 resolver, and layer-aware
aggregate guard are the active source-contract phase after the maintainer's
option-1 ownership decision. Dense-18432 profiling and canonical publication
remain separately gated.

- The H800 lane stays deferred until the maintainer explicitly resumes it.
- Legacy migration and RTX cleanup remain pending after the H200 measurement
  path unless they are required for H200 publication.
- H200 writes only to its dedicated staging tree until validation passes.
- Each completed model's E2E may run alongside the next model's profiling.
- Unsupported quantization or operator structure records a failed model result
  and advances the profiling lane to the next model.
- The current worktree is at HEAD `df499454ecd657f176f746046d2a404ec6a82d3`.
  Preserve the three untracked files `=10.1`,
  `outputs/metrics/step3_moe_noquant/offline_batch/run_2026_08_25_13_16_18_964917/config.json`,
  and `tests/performance/profiling/launch_step3_a_e2e_worker.sh`.
- No task-owned H200 or H800 worker is active after the completed r19 release.
  Mixtral r7 is `Stopped` after
  platform preemption; r9 is `Stopped` in the queue before replica creation;
  r10 reached a Ready H200 replica but ended without a profiling terminal
  artifact; r11 and r12 predict-only returned `no machine available`; r14
  reached `Succeeded` but failed before the first workload because FlashInfer
  could not create its `/mnt/host0` cache directory; r19 completed all six
  commands and was released after validation. Do not run
  `brainctl exec` against any stopped replica, especially
  `frontier-h200-qwen-resume-20260823-1` or the r7 replica.
- The current control plane has no idle Step3 worker. Keep the Step3 accepted
  profile tree immutable while the approved source-contract repair proceeds.

## Phases

### 3A. **In progress — profile-owned typed-layer and aggregate option-A source-contract repair**

- Preserve branch canonical MoE context/Step3 identity behavior and
  latest-main runtime/routing/EP/cache/MTP behavior during the source migration.
- Keep the validator `num_replicas` migration and the RED regressions as local
  review evidence; do not treat them as a completed production contract change.
- Extend the existing `ModelArchitectureProfile` and
  `ModelArchitectureRegistry` with a reusable declarative typed layer contract
  and migrate each consumer through its existing extension point. Do not add a
  second global layer registry.
- Keep `num_layers=61`, avoid model-name conditionals and scaling factors, and
  leave accepted H200 staging immutable when that future task starts.
- Keep aggregate option A as the selected target contract: mixed multi-layer
  FFN calls require explicit `layer_id`/`layer_ids` and must fail fast when the
  caller omits or misstates them. Implement this guard as its own source
  sub-step; preserve pure-model and attention-only behavior.
- Record option B as deferred future work: exact mixed-stage aggregation from
  complete layer IDs or PP bounds, with coordinated `ExecutionTime`, scheduler,
  trace, predictor, training/cache, and `ParamCounter` changes.

Dependency for this source phase: `latest-main merge resolution -> RED regressions -> typed LayerContract registry -> runtime/profiling loaders -> predictor/training/cache/trace/ParamCounter migration -> aggregate guard -> GREEN regressions`. Dense-18432 profiling and strict E2E follow only after coverage authorization.

Current implementation checkpoint (2026-08-29): the worktree is at
`3a11bbabc3daa1f94e5f64ff1492cd3dc222f890`; `git ls-files -u` is empty and no
`.git/MERGE_HEAD` exists. Existing validator/test edits and untracked
`=10.1`, Step3 output, and launcher remain preserved. Accepted H200 CSVs and
the frozen manifest remain immutable. Option-1 profile-owned contract
ownership is confirmed; the profile-owned contract and predictor TP/width
consumer are now GREEN (`152` focused tests in the current matrix). Existing validator/test edits and
untracked artifacts remain preserved. Option A's fail-fast guard and option
B's exact aggregate remain separate substeps.

Current independent review checkpoint (2026-08-29): the Option-2 causal chain
is reproduced with a failing mixed-layer TP contract (`1` observed versus `8`
required) and a direct profile scan (`0` dense `18432` rows in either linear
measurement family). The maintainer has now selected the profile-owned typed
contract owner/API; source implementation is **AUTHORIZED, RED NEXT**. Dense H200
profiling remains separately gated. The review report is
`task_memory/task_2026-08-23_profiling_dataset_governance/test_report_2026-08-29_option2_rca_review.md`.

Source dependency is:
`contract ownership decision -> RED matrix -> profile-owned typed resolver -> consumer
migration -> GREEN regressions -> coverage admission -> authorized profiling
-> strict Step3 E2E`. No layer64/PP fallback, data conversion, or remote action
is part of this checkpoint.

Completed source sub-step: predictor typed consumer. The next active sub-step
is the shared prediction-manager TP/width/signature migration, followed by the
runtime/profiling config field propagation.

1. **Completed — isolated baseline**
   - Start from PR #21 local head `f8fea750`.
   - Create branch `data/profiling-governance-h800-h200-20260823` in
     `.worktrees/profiling-data-governance-20260823`.
   - Record README hashes and verify the source branch is clean.
2. **Completed — H200 contract and exact manifest**
   - Re-run the strict CSV audit in the clean worktree.
   - Completed: resolved all six exact model identities.
   - Completed: downloaded the official Step3 text source through the company
     HTTPS proxy and added the normalized
     `data/config/models/step3-moe-noquant.json`.
   - Completed: replaced the stale Mixtral test-fixture provenance with the
     pinned official source and corrected its upstream router auxiliary-loss
     metadata.
   - Completed: validated every configuration with the current `ModelConfig`
     loader, attention binding, and linear-op profiling plan.
   - Completed: removed duplicate standard-attention structural workloads
     while preserving first-occurrence order and explicit invalid-value
     fail-fast behavior.
   - Completed: added the registry-derived six-model non-dummy E2E contract
     and validated the two large-model `PP2` fixture paths.
   - Completed: real H200 dual-measurement capability and axis-legality smokes
     admit TP `1/2/4/8` and MoE EP `1/2/4/8`.
   - Completed: froze the numeric envelope at Attention `128`, Linear `128`,
     MoE `64`, both measurement families, `standard_fused_topk`, uniform load,
     and seeds `0/1/2`.
   - Completed: selected `direct` and `prefill_warmed` as the canonical MoE
     gating runtime-context values.
   - Completed: added focused RED tests and implemented a single registry that
     keeps `standalone_legacy` and `prefill_hot` as warned temporary aliases.
   - Completed: froze
     `h200_exact_manifest_frozen_v3.json` with SHA-256
     `4df580ca1e30a007f45aeed4eb9f5d43593cbab49e59194ecabf8c5996ce8098`.
3. **Completed — H200 six-model collection and per-model E2E measurement scope**
   - Completed: bounded H200 `--predict-only`.
   - Completed: environment, kernel, capability, axis-legality, and Qwen3
     prefill-context smokes on real H200 GPUs.
   - Completed: `llama3.1-8b` profiling validation and formal accepted-data
     `2+2` E2E.
   - Completed: `llama3.3-70b` profiling validation and formal accepted-data
     `2+2` E2E.
   - Completed: `Qwen3-235B-A22B` targeted recovery, deterministic assembly,
     and exact validation with `10` accepted CSVs and `3,400` physical rows.
   - Completed: `Qwen3-235B-A22B` formal accepted-data `2+2` E2E against
     `qwen_assembled_accepted_20260823`.
   - Completed: a fresh eight-GPU H200 worker passed the exact
     `infer_af_test` predict-only and environment smoke gates. The live RJob is
     `frontier-h200-qwen3-a3b-retry-20260824-1423` on
     `gpu-h200-0949.lgcm.sh.istep.fun`.
   - Completed: the isolated `qwen3-a3b-30b-moe` retry under
     `/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_qwen3_a3b_30b_2ecc496b_retry_20260824_1423/`.
     All eight lanes completed, the accepted tree passed the frozen-manifest
     validator with `10` CSVs and `3,400` physical rows, and the matching
     `prefill=2, decode=2` E2E passed.
   - Completed: `step3-moe-noquant` profiling under the worker-local JIT cache;
     all lanes passed the frozen-manifest validator with `10` CSVs and `2,344`
     physical rows. Step3 uses the manifest's direct-only MoE context.
   - Completed: the accepted-data Step3 E2E was retried with the exact fixed
     contract. Preflight passed, but runtime admission failed before request
     execution with parameter shard `351956369408` bytes versus budget
     `136257837465` bytes. Evidence is in
     `test_report_2026-08-24_step3_moe_noquant_non_dummy_e2e_retry.md`.
   - Approved retry order: run strict vLLM option A, `PP1/TP8/DP1` mapped to
     `AT8/ADP1/MT1/EP8`, on a fresh 8-GPU H200 worker. Keep the accepted
     profile tree immutable. The first A surface run completed, but its
     monolithic independent predictor resolved dense-boundary `mlp_*` through
     TP1 instead of attention TP8, so it is not strict-semantic acceptance.
     Obtain approval for the family-aware predictor correction, rerun A, and
     only if the corrected A run fails admission or runtime design and validate
     the explicitly authorized `num_layers=64` plus PP fallback. The other
     reviewed candidates remain diagnostic alternatives and are not scheduled.
   - Deferred by maintainer request: the family-aware independent-predictor
     correction and all layer64+PP changes. The current task records the
     mismatch and leaves the accepted Step3 profile tree immutable.
   - Completed: `mixtral_8x7b_moe` formal profiling on r19 with all six frozen
     commands passing, followed by independent frozen-manifest validation.
     The accepted tree contains `10` CSVs and `2,344` physical rows. The
     matching non-dummy `prefill=2, decode=2` E2E also passed with `1/1`
     request, and the worker was released after the evidence was persisted.
     The r7/r9/r10 partial attempts and r14 cache failure remain historical
     evidence only and were not merged.
   - On unsupported quantization or operator structure, preserve the exact
     error and advance to the next model.
   - Current completion: all `6/6` frozen-scope profiling datasets pass the
     independent validator (`56` accepted CSVs, `14,064` physical rows).
     Formal accepted-data E2E is `5/6`: llama3.1-8b, llama3.3-70b, Qwen3,
     qwen3-a3b, and Mixtral pass the fixed `2+2` contract. Step3 remains a
     surface E2E pass with strict vLLM semantic acceptance deferred because
     dense `mlp_*` resolves through TP1; its predictor, registry, layer-count,
     and PP changes remain outside this task.
   - Completed: a fresh remaining-model audit found
     `missing_formal_profile_models=[]` and `missing_runtime_e2e_models=[]`.
     No additional H200 allocation is needed for the frozen measurement scope.
4. **Blocked — H200 publish/admission pending latest-main compatibility**
   - Completed staging validation for all six accepted H200 trees: `56` CSVs
     and `14,064` physical rows pass the frozen manifest; the aggregate evidence
     is `/data/ycfeng/tmp/frontier_h200_accepted_staging_revalidation_20260826_all6.json`.
   - Remaining checks: schema, identity metadata, measurement types, timing
     NaN/finite values, duplicate conflicts, legal TP/EP values, exact
     manifest coverage, and current consumer admission during publication.
   - Current result: producer/schema and frozen coverage checks pass, but
     latest-main admission is blocked by the MoE context contract and missing
     Step3 identity. The corrected isolated matrix reports `96` unique
     gating-op failures (the historical narrow probe reports `64`; `48`
     additional shared-FFN status entries repeat the same mismatch). Standard
     attention files also omit true-mixed rows until the documented supplement
     merge is performed.
   - Fresh direct validator recheck: all six contracts and profile trees pass,
     while only `2/6` historical runtime artifacts satisfy the current
     metadata contract. Qwen3-235B-A22B, qwen3-a3b-30b-moe, and
     mixtral_8x7b_moe retain `cluster_config.num_replicas=1` where the current
     registry requires `2`; Step3 retains `moe_routing_mode='simulation'`
     where the explicit current contract uses `balanced`. Preserve these
     artifacts as provenance and resolve the contract before publication.
   - Merge accepted staging rows deterministically into canonical CSVs only
     after the compatibility decision is approved.
   - Keep rejected worker output outside the active dataset.
5. **Pending — active-data cleanup**
   - **Completed preparation:** generated and validated the exact `git mv`
     archive manifest and H800/H200 coverage manifest; execute the moves only
     after the six-model publication gate.
   - Deterministically merge RTX mixed/true-mixed supplements into canonical
     attention files.
   - `git mv` A100, A40, A800, and H100 active SKU trees into
     `data/profiling/legacy/compute/`.
   - `git mv` superseded H800 combined and RTX auxiliary CSVs into matching
     legacy paths.
   - Verify active filenames and tuple completeness.
6. **Deferred — H800 real-GPU collection**
   - Keep H800 stopped until the maintainer explicitly reports availability.
   - Resume with the recorded `codesign` predict-only gate before allocation.
7. **Pending — repository gates and commits**
   - Verify README hashes are unchanged.
   - Record the remote state of PR #17/#20/#21; as of this checkpoint all
     three are already merged into `main`.
   - Run `git diff --check` and focused profiling tests.
   - Commit one completed sub-step at a time.
   - Stop before push/PR creation and request maintainer approval. The current
     compatibility audit failed the controlled data-only PR gate, so no remote
     action is scheduled.
8. **Completed — cross-session handoff**
   - Synchronize all task docs to the actual artifacts and control-plane state.
   - Record the absolute task-memory/worktree paths and exact resume order.
9. **Blocked pending maintainer decision — latest-main compatibility**
   - Completed the ref synchronization and read-only compatibility audit.
   - Keep the six accepted H200 trees and frozen manifest immutable.
   - Resolve the canonical-versus-legacy MoE context contract, missing Step3
     config, and three merge-tree conflicts in a separately approved task, or
     approve a fresh profiling campaign against latest main.
   - Hold remote push, PR creation, review comments, merge, and re-profiling
     until the maintainer selects one of those paths.

## Acceptance Criteria

- Active `data/profiling/compute` contains only current canonical filenames and
  directly consumable option-A datasets.
- Every archived file appears under
  `data/profiling/legacy/compute/<sku>/<model>/`.
- H800/H200 published CSVs cover every tuple in their recorded manifests.
- Critical timing columns contain zero missing or non-finite values.
- Duplicate feature keys contain zero conflicting timing rows.
- Current consumers admit every published dataset tuple selected for release.
- Worker logs identify the exact GPU model, environment, command, and output
  paths.
- Every README hash matches the baseline.
- The current remote state for PR #17/#20/#21 is recorded before any remote
  operation; all three are merged as of 2026-08-27.
- No remote push or new PR occurs without explicit maintainer approval.

## Errors Encountered

| Date | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-08-23 | A relative read from the PR #21 worktree could not find task records because those records live in the primary checkout. | 1 | Switched to absolute paths in the primary checkout and recovered the required records. |
| 2026-08-23 | Initial Serena symbol reads targeted the primary checkout instead of the new worktree and therefore showed pre-PR #21 attention behavior. | 1 | Activated `.worktrees/profiling-data-governance-20260823` as the Serena project, repeated the symbol reads, and used only the repeated results for manifest decisions. |
| 2026-08-23 | A token-grid probe called `get_attention_batch_sizes_to_profile()` without its required arguments and raised `TypeError`. | 1 | Enumerated the full attention generator with all required arguments and used that output for the coverage basis. |
| 2026-08-23 | H800 `codesign` predict-only failed its quota gate at `129/128`, while the `rlaunch` wrapper still returned exit code `0`. | 1 | Treated the semantic failure text as authoritative, skipped live allocation, preserved the worker logs, interrupted the other active lanes, and escalated the blocker before further execution. |
| 2026-08-23 | H200 execution was resumed before the provisional model set had received explicit maintainer confirmation. | 1 | Interrupted the H200 lane before model sampling, marked the model list provisional, and added an explicit confirmation gate. |
| 2026-08-24 | Task records still described Qwen recovery as running and the formal E2E as pending. | 1 | Reconciled Git, NFS artifacts, the current date, and `brainctl` state; recorded Qwen profiling and formal E2E as complete, and recorded the absence of any active H200 worker. |
| 2026-08-24 | A fresh H200 predict-only retry still reported `no machine available` while the wrapper returned `0`. | 1 | Recorded the semantic failure as authoritative, kept the live launch suppressed, and persisted the log under `/data/ycfeng/tmp`. |
| 2026-08-24 | The fixed Step3 accepted-data E2E reproduced `FRONTIER_MEMORY_OOM` before request execution. | 2 | Preserved the fail-fast runtime behavior, recorded exact parameter/budget bytes and a fresh report, and gated topology changes and Mixtral launch on maintainer disposition. |
| 2026-08-28 | The first final-state verification probe concatenated a string and integer while printing the unmerged-entry count and exited with `TypeError`. | 1 | Re-ran the same read-only checks with f-string formatting; the corrected probe passed and confirmed the manifest hash, protected files, synchronized refs, and zero unmerged entries. |
