## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-16 | Added the current W00–W11 source-bound status index; retained older execution/audit statements as historical evidence. |
| 2026-09-15 | Archived the independent v1.1 completion audit, exact evidence limits, dependency-ordered TODOs, and decision status; no source/test changes were made by the audit. |
| 2026-09-15 | Bound final R11/R12 and broad CPU evidence to candidate `b8cecf53`; recorded D02 residual cost and D03 cancellation decision gates. |
| 2026-09-15 | Archived final verification: 220 focused tests passed with clean compileall, diff check, worktree status, and parsed co-author trailer. |
| 2026-09-15 | Archived direct actual-replica-ID routing coverage in `7b6a3eb1`; trailer parsing confirms `powderluv` attribution. |
| 2026-09-15 | Archived the post-routing-fix full CPU unit regression: 3276 passed with the same 19 baseline failures and no candidate-only failure. |
| 2026-09-15 | Archived the monolithic MoE routing-identity repair, explicit manager device-event paths, and 219-test focused CPU verification. |
| 2026-09-15 | Archived the persistent real CPU hybrid Simulator E2E evidence, GDN schema correction, and its prepared-predictor/model-manager boundary. |
| 2026-09-15 | Archived the final pushed head `578785bb`, whitespace hygiene verification, and unchanged full-unit baseline comparison. |
| 2026-09-14 | Archived the completed selective PR #31 integration, final Review 2 evidence, baseline comparison, attribution status, and publication boundary. |

# Task Summary

## Current W00–W11 remediation checkpoint — 2026-09-16

The current authoritative status is [current_status.md](current_status.md), bound to production commit `c9f8f904`. The final source has **3587 CPU PASS / 18 matched baseline FAIL / 25 unchanged SKIP**, **8 synthetic non-dummy PASS**, and **58 dummy cases with raw24PASS/34independently classified corrections**. The final optimization preserves574/574 fidelity artifacts and106/106 non-dummy artifacts. All471 original and675 current production hunks are reviewed. Native acceptance remains19 hardware SKIPs. D01 uniform timing and D03 existing lifecycle decisions are recorded; **D02 measured residual is explicitly accepted for this delivery**. Final18 unprofiled executions are complete: median paired run ratios1.775045x/1.679023x/0.978816x (small dense/long dense/MoE). The local evidence handoff is complete; D01, D02 and D03 are resolved.

Older audit/execution sections, source/runtime counts, remote-head statements and open-item lists are **historical snapshots**. The current remediation checkpoint and archive use the linked index; historical evidence is preserved and is not rebound to the current source.


## Runtime-first remediation archive — current delivery

### Task Overview

Implemented the user-assigned W00–W11 plan in the existing worktree through production commit `c9f8f904e3550c11aad3cc5d851d75d648cef6e1`, preserving source attribution and the original plans. D01 retains uniform physical-layer/stage semantics and independent arithmetic verification; D03 retains existing request lifecycle scope. Implementation and final functional verification are delivered. **The authorized local delivery is complete; D02 was explicitly accepted for the measured scope.**

### Deliverables Inventory

Paths below are relative to this task directory unless they begin with `frontier/` or `tests/`.

- `current_status.md`, `validation_manifest.json`: one current package index and final source/environment/command/node/artifact evidence.
- `review_reassessment_2026-09-16.md`, `changed_hunk_review.md`, `hunk_review_predictors.md`, `hunk_review_runtime.md`, `hunk_review_attention_training.md`, `hunk_review_shared_contracts.md`, `w07_sglang_review.md`: all38 original findings, R01–R14/N01–N09,471 original hunks and675 current hunks across106 production files, with ownership/split decisions.
- `test_report_2026-09-16_w10_cpu.md`, `w10_final_cpu_comparison.json`: exact baseline-relative full CPU failures, collection and skips.
- `test_report_2026-09-16_w10_fidelity.md`, `w10_fidelity_classification.json`: all58 raw statuses and independent stage/lane/request/batching oracles.
- `test_report_2026-09-16_w10_nondummy_acceptance.md`, `w10_nondummy_report.md`, `w10_nondummy_final_artifact_comparison.json`, `w10_nondummy_baseline_comparison.json`: eight real synthetic-profile constructor/export cases and seven main controls.
- `performance_rca.md`, `w04_performance_review.md`, `w00_paired_samples.json`, `w11_intermediate_paired_samples.json`, `w11_final_paired_samples.json`, `w11_profile_counts.json`: initial/intermediate/final pairing, controlled cache/snapshot ablations and D02 proposal.
- `frontier/entities/execution_time.py`, `frontier/entities/stage_execution_time.py`: one-layer numerical scope, finalized isolated payloads and explicit ordered stage ownership. Existing component maps, family/operator registries and the homogeneous constructor are reused.
- `frontier/metrics/ep_wave_metrics.py`: focused actual-lane projection at shared phase barriers. The existing stage ledger contract remains intact; new actual-lane records use a separate requested ledger.
- `tests/integration/test_pr33_nondummy_acceptance.py`, `tests/integration/test_pr33_native_profiling_acceptance.py`, `tests/performance/measure_pr33_paired.py`, `tests/performance/measure_pr33_cache_ablation.py`: reusable final acceptance and controlled measurement entrypoints.

Verified implementation commits: `41777755` topology; `2e8ab0e1` capacity/lifecycle; `b07f653d` campaign/timer; `5df2e65e` canonical paths; `d3178bfd` artifact boundary; `c1e924db`/`a677e535` profiling/replay; `65aaf2c9` slot-owner cleanup; `5a1cc2af` native lanes; `f9099f85` integrated runtime/reporting; `d514f417` non-dummy acceptance; `5738ecb3` controlled performance checks; `c9f8f904` homogeneous aggregation. `10326ce4` preserves the initial freeze. Applicable commits retain `Co-authored-by: powderluv <74956+powderluv@users.noreply.github.com>`.

### Validation Status

- Final CPU: **3587 PASS /18 known baseline FAIL /25 unchanged SKIP**,3630 collected,95.29s. Baseline3144PASS/19FAIL/25SKIP; one baseline documentation assertion now passes. No candidate-only failure or new skip. All retained causes/node identities and fixture migrations are recorded.
- Synthetic non-dummy: **8 PASS**,35.43s;7/7 request and7/7 system baseline comparisons MATCH;106/106 artifacts unchanged after final optimization. Oracles verify actual layer IDs, physical operation totals, EP lanes and GDN capacity/state behavior.
- Dummy fidelity: **24 raw PASS /34 raw FAIL**,116 successful simulator processes.12 reporting-only cases preserve request/system values;22 disaggregated MoE cases have the source-proven dummy normalization correction.127 candidate stage rows,130 baseline stage rows,11264 lane rows/4064 waves and176 request-role sums pass independent arithmetic.574/574 artifacts match the preceding candidate/baseline campaign byte for byte.
- Native: **19 SKIP** on this CPU host;5 timer,4 AMD GDN,10 profiling/replay/collective lanes remain real runnable entrypoints. No CUDA/ROCm/AMD hardware correctness claim.
- Performance:18 final unprofiled runs, same event/request/token workloads and output flags. Small dense14.149→25.225ms, longer dense29.309→49.268ms, MoE9.730→9.524s; median paired ratios1.775045x/1.679023x/0.978816x. D02 is explicitly accepted; process-wide ratios are not substituted for run-phase cost.
- Source review:471 original and675 current production hunks reconciled. Final shared-stage optimization independently reviewed with no concrete issue;132 focused tests plus final integrated campaigns pass within the classifications above.

### Open Items / Future Extensions

There is no unresolved current decision. The user explicitly selected **D02 option1**, accepting the measured scoped residual with the existing uniform contract. Further representation optimization was not selected. The concrete alternatives and costs remain in `performance_rca.md`. A new request-cancellation API is deferred under D03. Native GPU execution and production-profile timing/benchmark parity remain unverified and require their corresponding hardware/data. Pre-existing missing assets/dependencies and unrelated documentation assertions remain visible; no unrelated fixes were added. This local remediation was not pushed or merged.

## Historical selective-integration archive

## Task Overview

Selective integration of the useful, reviewable portions of PR #31 into the `feature-amd-sglang-gdn` worktree, following the read-only execution plan at `/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md`. The integration covers MI355X metadata, hybrid GDN/full-attention semantics, standard ROCm profiling contracts, selected vLLM/AITER and RCCL producers, and isolated experimental SGLang tooling while preserving existing dense/MoE simulator behavior.

The source plan was not modified. Its final SHA-256 is `17365fe6a96cecf162d63f450fac3fd83bd91caaa2abf87f04c81131a08afd66`.

## Deliverables Inventory

### Runtime and model semantics

- `frontier/attention/gdn/` — immutable per-layer GDN schedule/configuration, feature construction, fixed-state memory, lifecycle slots, and unsupported-feature guards.
- `frontier/attention/model_binding.py`, `frontier/attention/families.py`, `frontier/attention/ops.py`, and `frontier/attention/profiling_mapping.py` — explicit attention-family registry, per-layer binding, GDN family metadata, and runtime-family resolution for hybrid cache/layout helpers.
- `frontier/config/model_config.py`, `frontier/config/config.py`, `frontier/config/device_sku_config.py`, `frontier/config/node_sku_config.py`, `frontier/config/quantization_manager.py`, `frontier/config/utils.py`, and `frontier/types/*sku_type.py` — MI355X platform metadata, Qwen3.8 hybrid configuration, quantization metadata, and safe serialization/configuration guards.
- `frontier/entities/execution_time.py`, `frontier/entities/stage_execution_time.py`, `frontier/execution_time_predictor/`, `frontier/metrics/metrics_store.py`, and `frontier/metrics/op_trace_utils.py` — identity-bearing per-layer timing, ordered stage aggregation, hybrid dispatch, stage-aware metrics/traces, and one-time stage-owned accounting.
- `frontier/execution_time_predictor/gdn_predictor.py` — complete phase-qualified GDN structured operator schema, with explicit zero values for inactive phase cores required by trace consumers.
- `frontier/utils/param_counter.py`, `frontier/scheduler/utils/memory_planner.py`, `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`, and `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py` — schedule-aware memory/capacity accounting and fail-fast GDN runtime boundaries.
- `frontier/moe_ep_workload.py` and the standard/disaggregation MoE predictors — one deterministic routing-ratio helper shared by both predictor paths.
- `frontier/simulator.py`, `frontier/execution_time_predictor/random_forrest_execution_time_predictor.py`, and `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` — process-global monolithic Replica IDs are propagated into shared MoE routing maps with strict shape validation.

### Standard profiling and training

- `frontier/profiling/common/accelerator.py`, `cuda_timer.py`, `device_timer.py`, `timer_stats_store.py`, `constants.py`, `model_config.py`, and `vllm_compat.py` — CPU-safe accelerator discovery, strict `DEVICE_EVENT`, preserved CUDA timing behavior, and current vLLM compatibility helpers.
- `frontier/profiling/gdn/` and `frontier/training/gdn_trainer.py` — standard vLLM GDN producer/schema, six phase-qualified estimators, manifest identity, and dataset fingerprint validation.
- `frontier/profiling/attention/backends/vllm_rocm_attention_wrapper.py` — explicit `VLLM_ROCM` full-attention producer with lazy runtime imports.
- `frontier/profiling/moe/` — standard fused-MoE/AITER/MXFP4 path with CPU-safe packed-layout planning and explicit quantization metadata.
- `frontier/profiling/collectives/` — standalone NCCL/RCCL collective profiler with dtype-aware byte accounting and deterministic CSV output.

### Experimental tooling

- `frontier/profiling/experimental/sglang/` — isolated dense/GDN/attention/MoE primitive contracts, HIP graph replay planning, routed replay with shared routing semantics, and Kineto GDN trace summaries.
- Experimental outputs remain outside the standard `gdn.csv` training discovery path and are labeled with their experimental measurement source.

### Documentation and tests

- `docs/cli/README.md`, `docs/profiling/README.md`, `docs/profiling/ROCM_MI355X.md`, `docs/training/README.md`, and `examples/profiling/README.md` — release-facing ROCm/GDN workflow, standard versus experimental boundaries, and canonical output taxonomy.
- `tests/fixtures/pr31_hybrid/` — synthetic CPU hybrid GDN CSV/model fixture with explicit test-only provenance.
- `tests/unit/test_gdn_*`, `test_hybrid_runtime_family.py`, `test_metrics_stage_execution_time.py`, `test_collectives_*`, `test_moe_mxfp4_increment10.py`, `test_moe_shared_routing_helper.py`, `test_sglang_*`, `test_vllm_rocm_attention_wrapper_increment9.py`, and related compatibility tests — focused acceptance coverage for each planned increment.
- Durable task records: `requirements.md`, `plan.md`, `progress.md`, `review.md`, `issues.md`, and all `test_report_*.md` files in `task_memory/task_2026-09-14_pr31_selective_integration_v2/`.

### Commit lineage and attribution

The branch contains independently reviewable selective commits from `0989f0ed` through `578785bb`, including the final docs commit `7937e534`, compatibility fix `ffb9feea`, and whitespace hygiene commit `578785bb`. Selective integration commits retain `Co-authored-by: powderluv <74956+powderluv@users.noreply.github.com>` wherever the implementation was extracted or materially adapted from PR #31. The earlier `e8ac59ae` rename commit has no trailer; no history rewrite was performed. The original PR #31 remains unchanged.

## Validation Status

Environment used for CPU validation: Python 3.12.3, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, Plotly 6.8.0, and PyTorch 2.5.1+cu124. `vllm`, `sglang`, and `aiter` are unavailable in this environment.

- Review 2 focused suite: **56 passed**.
- Existing homogeneous execution/metrics/MLA/MoE/FFN suite: **91 passed**.
- Profiling examples documentation contract: **15 passed**.
- Final full unit suite before the routing repair at pushed HEAD `578785bb`: **3274 passed, 19 failed, 25 skipped, 576 warnings** in 70.74 seconds. Post-fix full unit suite: **3276 passed, 19 failed, 25 skipped, 576 warnings** in 74.32 seconds. The 19 failure names exactly match baseline: one missing `frontier.config_optimizer` module, nine missing debug E2E shell assets, two stale release-example documentation contracts, four existing MLA/MHA/MQA analysis-builder contracts, and three stale top-level PDD documentation contracts. No candidate-only failure remains. The post-fix log is `/data/ycfeng/tmp/pr31-full-unit-20260915-routingfix.log`.
- `python -m compileall -q frontier tests`: PASS.
- `git diff --check`: PASS.
- Non-dummy dense CSV simulator smoke: PASS; request metrics and system JSON were written under `/data/ycfeng/tmp/pr31-final-dense-20260914/qwen2_dense_test/offline_batch/final_dense_csv/`.
- Non-dummy MoE CSV simulator smoke: PASS; request metrics and system JSON were written under `/data/ycfeng/tmp/pr31-final-moe-20260914/qwen3_30b_a3b_tiny/offline_batch/final_moe_csv/`.
- Profiling wrapper dry-runs (`profile_linear_op.sh`, `profile_attention_chunked_prefill.sh`, `profile_moe.sh`), GDN CLI help, and collective CLI help: PASS.
- The final full-unit log is preserved at `/data/ycfeng/tmp/pr31-final-unit-578785bb.log`.
- Persistent hybrid CPU Simulator E2E: **PASS — 4 passed in 3.54s on the post-fix rerun** (the fixed-basetemp evidence report records the prior 3.52s run). The test produces one completed 18-token request, two stage-ledger rows, and per-layer op traces carrying both GDN and dense attention identities. GDN training/artifact loading is real; unrelated FFN/communication timings are deterministic test hooks, and the simulator receives the prepared predictor through a test-local registry seam. This is control-flow/accounting evidence rather than production-data latency evidence.
- Hybrid production constructor/routing closure: **PASS** — real synthetic standard-profile manager construction passed `1` test in `4.43s` and a follow-up rerun in `4.60s`; direct actual-ID routing coverage passed `1` test in `2.58s`; manager path contract passed `1` test in `2.73s`; the complete focused command passed **219 tests in 13.54s**. The routing map keys matched the process-global `Cluster.replicas` keys, and all three device-event derivatives were present in the path contract.
- Final verification command including the direct actual-ID test: **220 passed in 15.05s**; `compileall` and `git diff --check` exited `0`; worktree was clean at remote HEAD `7b6a3eb1`.
- `SKIP: AMD/MI355X hardware unavailable`: no ROCm `DEVICE_EVENT` runtime, vLLM HIP/AITER/MXFP4 execution, RCCL process-group run, SGLang/HIP graph execution, AMD benchmark comparison, or groundtruth parity was performed or claimed.

## Open Items/Future Extensions

- New remote PR [#33](https://github.com/NetX-lab/Frontier/pull/33) is open from `feature-amd-sglang-gdn` to `main`. The branch must remain unmerged until the user provides separate explicit merge authorization.
- Run native ROCm/vLLM/AITER/MXFP4/RCCL/SGLang and benchmark/groundtruth checks when an AMD/MI355X worker becomes available.
- Run a production-data hybrid simulator request/metrics case with complete model-specific standard attention and MoE CSVs when they are available; the synthetic CPU production-constructor E2E now proves manager/predictor construction, while the persistent full-flow case still uses a prepared-predictor seam for unrelated operator timings.
- Existing baseline failures remain outside this task scope and are recorded in `test_report_2026-09-14_final_review2.md` and `issues.md`.

## v1.1 exact-HEAD validation checkpoint

The current source evidence is bound to candidate `b8cecf53f8b81ea8380238971277ba94c6fe4961`, with clean baseline
`0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. The final full unit run is **3309 passed, 19 baseline failures, 25 skipped, 576 warnings**; the corrected concurrent broad groups add **1234 passed and 19 skips** across the required attention, GDN, scheduler/memory, stage/metrics, MoE/parallel, timer/measurement, SGLang, and non-dummy/golden surfaces. The clean 58-case baseline/candidate matrix is **58 passed, 0 failed** at this SHA.

The final paired unprofiled R12 artifact contains 18/18 successful runs. The historical approximately 9.42x `Simulator.run()` regression is not reproduced. The measured repeated stage aggregation overhead was removed in `b8cecf53`; dense `sim_wallclock_s` residuals remain 1.504303x and 1.407252x, and are awaiting D02 user disposition. D03 remains open because no reachable request-level cancellation path exists in the current production call graph. AMD/MI355X and visible NVIDIA runtime checks are explicit SKIPs. See `test_report_2026-09-15_r11_r12_final.md` for raw evidence paths and exact commands.

## Independent v1.1 completion audit — 2026-09-15

The report filenames retain the pre-existing task handoff date `2026-09-16`;
the audit execution date is September 15, 2026.

The authoritative audit is
`completion_audit_2026-09-16.md`, with execution evidence in
`test_report_2026-09-16_completion_audit.md`. Its conclusion is
**substantially incomplete**: R01–R05, R07–R12, and R14 are PARTIAL; R06 and
R13 are NOT DONE; no R package is COMPLETE. The 69-test focused CPU command
passed, and the 58-case dummy/fidelity artifact passed, but those results do
not replace the missing real MLA, automatic GDN/lifecycle, unified ownership,
non-dummy/golden, GPU-entry, performance-RCA, and cleanup evidence.

The final source evidence remains bound to `b8cecf53`; the remote PR remains at
`69d09305`, 15 commits behind the local candidate. This audit changed only task
documentation. Historical review/test records remain preserved and are not
reinterpreted as new source runs.
