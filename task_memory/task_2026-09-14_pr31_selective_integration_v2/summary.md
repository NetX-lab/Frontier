## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
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
