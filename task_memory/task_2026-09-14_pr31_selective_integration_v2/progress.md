## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Prepared the 25 local task documents, including two independent reviews, for explicitly requested publication to PR #33. |
| 2026-09-15 | Fixed stage aggregate cache invalidation after timing payload mutation; the focused stage/metrics/operator suite passed 52 tests. |
| 2026-09-15 | Final verification rerun including the direct actual-ID contract: 220 focused tests passed; compileall, diff check, clean status, and trailer inspection passed. |
| 2026-09-15 | Added a direct actual-replica-ID routing contract test and committed it as `7b6a3eb1` with a parsed `powderluv` co-author trailer. |
| 2026-09-15 | Completed the post-routing-fix full CPU unit regression: 3276 passed, 19 unchanged baseline failures, 25 skipped, 576 warnings; no candidate-only failure. |
| 2026-09-15 | Closed the focused-suite monolithic MoE routing identity regression; real Replica IDs now flow into hybrid predictor maps, manager path contracts include device-event derivatives, and 219 focused tests pass. |
| 2026-09-15 | Fixed the GDN structured operator schema so inactive prefill/decode phase keys are explicitly zero; compileall, diff checks, and the focused hybrid E2E rerun passed. |
| 2026-09-15 | Persisted and passed the real CPU hybrid Simulator/Replica/MemoryPlanner E2E test, including request completion, metrics, stage ledger, and per-layer GDN/dense trace identities. |
| 2026-09-15 | Reran the full CPU unit suite at pushed HEAD `578785bb`; 3274 passed, 19 baseline failures, 25 skipped, and commit-range diff check passed. |
| 2026-09-15 | Closed the current-HEAD SGLang graph replay tuple-contract review concern; no additional code change was required. |
| 2026-09-14 | Completed Review 2, final CPU validation, documentation closure, and PR #33 publication; AMD checks remain explicit skips. |
| 2026-09-14 | Initialized execution records; baseline inspection is in progress. |
| 2026-09-14 | Completed Increment 6 implementation and CPU verification; 19 baseline unit failures remain unchanged. |
| 2026-09-14 | Completed Increment 12 shared MoE routing helper and CPU verification; baseline replay collection error reproduced unchanged. |
| 2026-09-14 | Completed Increment 8 CPU-safe GDN trainer/predictor pipeline, CLI documentation, and focused verification. |
| 2026-09-14 | Completed the Increment 14A/B CPU hybrid dispatch checkpoint: fresh GDN artifact load, per-layer family resolution, ordered stage identities, and compatibility regression. |
| 2026-09-14 | Committed the verified Increment 14A/B per-layer dispatch checkpoint; production-data hybrid simulator E2E remains open. |
| 2026-09-14 | Implemented Increment 7 standard vLLM GDN producer/CLI with explicit phases, carried-state planning, and CPU contract tests; AMD runtime remains unavailable. |
| 2026-09-14 | Implemented Increment 9 explicit VLLM_ROCM attention backend and generic current-vLLM compatibility/discovery paths; CPU contracts pass and AMD runtime remains unavailable. |
| 2026-09-14 | Implemented Increment 10 current-vLLM functional MoE fallback, online MXFP4/AITER path, physical layout contract, and explicit MoE metadata; CPU contracts pass and AMD runtime remains unavailable. |
| 2026-09-14 | Implemented Increment 11 standalone NCCL/RCCL collective runner, dtype-aware byte accounting, local multiprocessing path, and output schema; CPU contracts pass and AMD runtime remains unavailable. |

# Progress

Status: completed for the scoped selective integration; PR #33 is open and the branch remains unmerged pending explicit user authorization.

## 2026-09-14 — Review 2, final CPU validation, and documentation closure

- Fixed the only candidate regression found by the final full-unit comparison: `frontier.profiling.common.model_config.ModelConfig.get_attention_family()` now preserves the module-level homogeneous `bind_attention_family` seam for MLA and dense profiling callers, while hybrid GDN schedules continue through `resolve_runtime_attention_family()`. The fix is in commit `ffb9feea` and carries the `powderluv` co-author trailer.
- Added the missing modification-history block to `examples/profiling/README.md`, restoring the existing profiling documentation contract without changing the top-level `README.md`.
- Review 2 focused suite passed **56/56**; existing homogeneous metrics/MLA/MoE/FFN suite passed **91/91**; profiling examples documentation contract passed **15/15**.
- Final full unit command returned **3274 passed, 19 failed, 25 skipped, 576 warnings** in 70.02 seconds. The 19 failures exactly match the recorded baseline test names: one missing config-optimizer module, nine missing debug E2E assets, two stale release-example documentation contracts, four existing MLA/MHA/MQA analysis-builder contracts, and three stale top-level PDD documentation contracts. No candidate-only failure remains.
- `python -m compileall -q frontier tests` and `git diff --check` passed. The source plan remains outside the worktree and was not modified; its final SHA-256 is recorded in `test_report_2026-09-14_final_review2.md`.
- The already completed non-dummy dense and MoE CSV simulator smokes produced request metrics and system JSON under `/data/ycfeng/tmp/pr31-final-dense-20260914` and `/data/ycfeng/tmp/pr31-final-moe-20260914`. Profiling wrapper dry-runs, GDN CLI help, and collective CLI help also passed.
- Standard ROCm/vLLM/AITER/MXFP4/RCCL/SGLang GPU execution, AMD benchmark comparison, and groundtruth parity remain **SKIP: AMD/MI355X hardware unavailable**.
- Pushed `feature-amd-sglang-gdn` to `origin` and created new PR [#33](https://github.com/NetX-lab/Frontier/pull/33) against `main`. No merge, rebase, branch deletion, or modification of PR #31 was performed.
- Current exact-HEAD code re-review confirmed the SGLang graph replay normalization blocker is closed: all builder tuple shapes are normalized by `_make_replay_call()`, and the Increment 13 seven-test suite passes. This is a review finding closure, not new code.

Report: `test_report_2026-09-14_final_review2.md`.

## 2026-09-15 — Increment 14A/B persistent CPU Simulator E2E

- Persisted the previously temporary hybrid probe in `tests/unit/test_gdn_hybrid_e2e_increment14ab.py::test_hybrid_gdn_real_simulator_cpu_e2e`.
- The test trains and freshly loads the real `GDNTrainer`/`GDNPredictor`, constructs a real `ReplicaConfig` and `Simulator`, executes one 16-prefill/2-decode request through the scheduler, MemoryPlanner, EP event/barrier path, continuation, and metrics writer, and uses deterministic hooks only for unrelated dense/MoE FFN and communication timings.
- The test enables per-layer op traces and asserts both `gated_delta_net` and `dense_attention` identities, GDN layers `{0, 1, 2, 4, 5, 6}`, dense layers `{3, 7}`, ordered stage-ledger rows, and `total_requests == completed_requests == 1` with 18 committed tokens.
- Exact command: `PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR python -m pytest tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider`.
- Result: **4 passed in 3.71s**. The run produced `request_metrics.csv`, `system_metrics.json`, `frontier_stage_batch_ledger.jsonl`, `monolithic_operation_metrics.csv`, and `op_traces.jsonl` under the pytest temporary output. A standalone reproducible artifact run is recorded in `test_report_2026-09-15_hybrid_simulator_e2e.md`.
- The first decode retry exposed a test-seam bug: the fake internal predictor queried GDN attention for `include_attention=False` post-attention FFN lanes. The seam now returns zero attention with the complete structured GDN operator schema, and the real simulator completes. This is a test correction; no production runtime workaround was added.
- `SKIP: AMD/MI355X hardware unavailable`; this CPU evidence proves control flow and accounting only, not ROCm timing, AMD benchmark parity, or groundtruth parity.

## 2026-09-15 — GDN structured operator schema correction

- Root cause: `GDNPredictor.predict_operator_times()` returned only the active phase's core operator. Prefill omitted `gdn_core_decode`, and decode omitted `gdn_core_prefill`; stage trace consumers require the complete structured GDN schema.
- Correction: `predict_attention_time()` now inserts the inactive phase core key with value `0.0`, preserving the active phase timing and avoiding trace-schema failures.
- Verification: `python -m compileall -q frontier tests` passed; `git diff --check` passed; the fixed-basetemp focused command passed **4/4 in 3.54s**. The earlier persistent artifact run remains documented at 3.52s in `test_report_2026-09-15_hybrid_simulator_e2e.md`.
- Scope: this is a production predictor contract fix surfaced by the CPU Simulator E2E. It does not add an AMD runtime claim; `SKIP: AMD/MI355X hardware unavailable` remains in force.

## 2026-09-14 — Increment 11 standalone RCCL profiler

- Updated collective input/grid handling with explicit FP16/BF16/FP32 precision validation and complete-node world-size checks.
- Updated `GraphedCollective` and `CollectiveWrapper` to pass the requested tensor dtype and derive byte size from `Tensor.element_size()` rather than a fixed FP16 multiplier.
- Updated the collective entrypoint with lazy Ray import, local single-node multiprocessing, NCCL backend initialization (RCCL on ROCm), deterministic output path, and flattened CSV writer.
- No simulator communication model, `ExecutionTime` estimate, or `network_device=mi355x_ubb` auto-enable behavior was changed.
- Collective and accelerator verification: **29 passed**. CLI help, compileall, and diff checks passed.
- `SKIP: AMD/MI355X hardware unavailable`; no RCCL process-group or GPU timing run was possible.

Report: `test_report_2026-09-14_increment11.md`.

## 2026-09-14 — Increment 10 standard vLLM/AITER MoE and MXFP4

- Updated `frontier/profiling/moe/moe_impl.py` with current vLLM fused-MoE import fallbacks and standalone-safe `ReplicatedLinear` construction.
- Updated `frontier/profiling/moe/moe_vllm_kernel.py` with the functional `fused_experts` API, `vllm_config_context`, current vLLM online MXFP4 `FusedMoEFactory`/`OnlineQuantizationConfig` path, ROCm AITER backend checking, and FP8/MXFP4 mutual exclusion.
- Added CPU-safe `plan_mxfp4_weight_layout()` for packed FP4, group32, and E8M0 shape metadata. The planner validates dimensions before GPU allocation; the vLLM quantization method performs actual packing on supported hardware.
- Updated `MoEWrapper` to derive MXFP4 from `PrecisionType.FP4`, route grouped GEMM through `vllm_aiter_mxfp4`, and emit `moe_quantization_mode` metadata. No independent MXFP4 CLI flag was added; model quantization metadata remains the source of truth.
- Focused Increment 10 plus existing MoE metadata/governance/accelerator suite: **63 passed**. Targeted compileall and `git diff --check` passed.
- `SKIP: AMD/MI355X hardware unavailable`; BF16 functional fused experts, online MXFP4 packing, AITER execution, ROCm `DEVICE_EVENT`, and GPU timings remain unverified.

Report: `test_report_2026-09-14_increment10.md`.

## 2026-09-14 — Increment 14A/B early CPU hybrid checkpoint

- Added per-layer attention-family resolution to the full-attention branch of `SklearnExecutionTimePredictor.predict_attention_layer_time()`. GDN layers continue to use the artifact-backed `GDNPredictor`; full-attention layers resolve their own dense `LayerAttentionSpec` without invoking the homogeneous whole-model binder.
- Removed the Increment 8 temporary skip that prevented attention estimator training for hybrid models. The shared manager now trains dense full-attention estimators for the full-attention layers, while `_is_mla_family()` explicitly keeps Qwen3.5 hybrid configurations out of the MLA-only path.
- Attached resolved `global_layer_id`, `attention_family_id`, and `attention_variant_id` to MoE `ExecutionTime` results. `StageExecutionTime.from_execution_time()` preserves an existing source identity when expanding compatibility records.
- Added `tests/unit/test_gdn_hybrid_e2e_increment14ab.py`. It runs the real `GDNTrainer`, loads artifacts through a fresh `GDNPredictor` and fresh model-manager seam, checks phase-qualified GDN values and ignored `gdn_layer_e2e=99.0`, validates one-token continuation as prefill, and executes all eight synthetic layers through `SklearnMoEExecutionTimePredictor` with exact `G G G A G G G A` identities and ordered stage aggregation.
- Focused hybrid/manager/stage/predictor suite: **76 passed**. Extended dense/MoE/disaggregation/MLA/PD predictor suite: **116 passed**. Single-case two-worktree fidelity: **1 passed, 0 failed**, with request/metrics artifacts under `/data/ycfeng/tmp/pr31-inc14ab-fidelity-20260914-run2`.
- `python -m compileall -q frontier tests` and `git diff --check` are required next before committing this increment. This checkpoint uses deterministic CPU hooks for unrelated FFN/communication components; a full production hybrid simulator request/metrics run with model-specific standard attention/MoE CSVs remains a follow-up within Increment 14A/B.
- `SKIP: AMD/MI355X hardware unavailable`; no ROCm profiling, AMD `DEVICE_EVENT` writer execution, benchmark parity, or groundtruth parity is claimed.

Commit: `feat: connect hybrid per-layer attention dispatch` (candidate SHA recorded after commit).

## 2026-09-14 — Increment 7 standard vLLM GDN producer

- Added `frontier/profiling/gdn/vllm_wrapper.py` and `main.py` by selectively adapting the PR's vLLM Qwen GDN module path. Runtime imports remain lazy and standard ROCm profiling requires `DEVICE_EVENT`.
- Extended `GDNProfileInput` with explicit `logical_phase`, `prefill_mask`, `physical_batch_size`, `state_init_mode`, and reserved-null-page state IDs. A one-token continuation constructed through `prefill()` remains `PREFILL`; same-batch mixed execution is rejected before wrapper work.
- Added carried-state preparation: positive-prefix workloads are primed outside the timed region, snapshotted, and restored before warmups/timed decomposition/e2e runs. The wrapper retains vLLM's null page zero convention and aggregates TP samples by per-sample rank maximum.
- Added CPU contracts in `tests/unit/test_gdn_profiler_cpu_increment7.py`, covering phase semantics, state metadata, mixed rejection, CLI planning, strict ROCm event mapping, and optional-dependency-safe imports.
- Verification: **28 passed** across the Increment 7 contracts plus existing GDN semantic/trainer/hybrid suites; `python -m compileall -q frontier tests` and `git diff --check` passed; CLI `--help` imports without vLLM.
- The wrapper's real vLLM/HIP execution, decomposition equality, carried-state GPU continuity, and DEVICE_EVENT row production are **SKIP: AMD/MI355X hardware unavailable**. Synthetic BF16 weights are provenance metadata only and do not establish checkpoint timing or benchmark/groundtruth parity.

## 2026-09-14 — Increment 9 standard ROCm compatibility and VLLM_ROCM attention

- Added the explicit `AttentionBackend.VLLM_ROCM` selector and lazy backend construction. The vLLM ROCm wrapper preserves the PR's native `RocmAttentionImpl` metadata/slot mapping path and rejects same-batch prefill/decode in the standard entry point before timing/output.
- Added `frontier/profiling/common/vllm_compat.py`; RMSNorm and RoPE wrappers now tolerate current standalone vLLM context/signature changes while retaining CPU-safe import behavior. Generic linear/MoE launchers use the shared accelerator visibility/discovery helpers, and Qwen3.5 RMSNorm compatibility is explicit.
- Verification: **51 passed** across ROCm sequence planning, accelerator discovery, attention/profiling contracts, GDN producer/trainer, and existing attention/MoE metadata suites; compilation and diff checks passed. One stale release README contract failure remains baseline-only and is recorded in `test_report_2026-09-14_baseline.md`.
- Source lineage: the ROCm wrapper and compatibility changes are selectively adapted from PR #31; candidate ROCm execution, current vLLM kernel behavior, and DEVICE_EVENT rows are **SKIP: AMD/MI355X hardware unavailable**. No CUDA producer behavior was intentionally changed.

## 2026-09-14 — Increment 6 GDN state, memory, and runtime guards

- Added fixed TP-aware `GatedDeltaNetStateLayout` accounting for convolution and recurrent state, with speculative-token sizing excluded from the initial public API.
- Added simulator-only `GatedDeltaNetStateSlotManager` ownership tokens for admission/running, waiting retention, resume reuse, and completion/cancellation release. The manager stores request/slot identity only and never tensor state.
- Added schedule-aware attention parameter accounting in `ParamCounter`, including GDN projection/conv sharding, FP32 `A_log`, per-stage GDN/full-attention composition, and resident-stage selection. D57 retains the 2-byte MLP/MoE/MTP approximation with an explicit MXFP4 comment and ordinary capacity/OOM checks.
- Updated `MemoryPlanner` to reserve fixed GDN state for `max_num_seqs` and divide KV capacity only by full-attention layers in the resident stage. Existing non-GDN paths remain on their prior parameter/KV policy.
- Added fail-fast GDN guards for prefix cache, P-to-D, speculative/MTP execution, state-dropping preemption, PP>1, EP>1, attention-DP>1, and cross-node execution. Waiting retention is explicitly accepted. Wired guards into configuration initialization, analytical P-to-D transfer, and vLLM preemption before request mutation.
- Added private-field filtering in `dataclass_to_dict` so lazy layer-spec caches do not leak into JSON serialization.
- Focused Increment 6/compatibility suite: **82 passed**. `python -m compileall -q frontier tests` and `git diff --check`: PASS.
- Fresh full unit regression: **3206 passed, 19 failed, 25 skipped, 576 warnings**. The 19 failures match the baseline categories (missing config-optimizer/debug/analysis assets and stale release docs); no Increment 6 failure remains.
- Real Qwen3.8 CPU structural/capacity check: 92 layers, 69 GDN, 23 full-attention; PP4 stage counts `((18, 5), (17, 6), (17, 6), (17, 6))`; modeled TP1/PP4 capacity raised the expected `FrontierMemoryOOMError` under a 1 GiB budget after applying the D57 approximation.
- Detailed evidence: `test_report_2026-09-14_increment6.md`.
- `SKIP: AMD/MI355X hardware unavailable`; no ROCm/MI355X profiling or benchmark/groundtruth parity is claimed.

## 2026-09-14 — Increment 12 shared MoE routing helper

- Added `generate_moe_routing_ratios()` to `frontier/moe_ep_workload.py` and routed both standard and disaggregation MoE predictors through it.
- Preserved RNG construction (`default_rng(seed + layer_id)`), expert iteration order, normalization, supported distributions, and downstream integerization. The helper does not observe a live router or change model expert-count/top-k configuration.
- Added CPU unit coverage for distributions, determinism, normalization, layer-specific random variation, invalid inputs, and standard/disaggregation equivalence.
- Focused checks passed: **9**, **62**, and **94** tests in the three suites; exact old-formula comparison passed for **48/48** cases; compile and diff checks passed.
- The baseline replay test still fails during collection with the same import-cycle error on both candidate and detached `0d1b87a4`, so it remains a pre-existing issue.
- Evidence: `test_report_2026-09-14_increment12.md`.
- `SKIP: AMD/MI355X hardware unavailable`; no AMD profiling or parity claim.

Status: Increment 12 implementation and verification complete; Increment 8 is now recorded below and remains pending its source commit.

## 2026-09-14 — Increment 8 standard GDN training and prediction

- Added the shared `GDNBatchFeatures` contract with physical batch/query/statefulness features. Decode excludes history/context length, and same-batch prefill plus decode is rejected.
- Added `GDNTrainer` with six phase-qualified estimators, identity and runtime-stack filtering, exact-row metadata, deterministic single-row fixture support, and a `gdn_manifest.json`. `gdn_layer_e2e` is ignored as a diagnostic-only target.
- Added artifact-backed `GDNPredictor` with full identity validation, exact lookup first, estimator extrapolation warnings without clipping or TP scaling, and structured `AttentionOperatorTimes` output.
- Added the direct `python -m frontier.training.cli gdn` dispatch and public `gdn_input_file` config template. Updated `docs/training/README.md` with the standard GDN CSV, `DEVICE_EVENT`, manifest, and experimental-artifact boundary.
- Fixture CLI output: `/data/ycfeng/tmp/pr31-inc8-gdn-final-20260914`; all six estimator artifacts and `gdn_manifest.json` were created. Exact fixture outputs were prefill `0.11/0.22/0.33` ms and decode `0.04/0.05/0.06` ms.
- Fresh focused suite: **33 passed**. `python -m compileall -q frontier tests`, `git diff --check`, and `python -m frontier.training.cli gdn --help`: PASS.
- Detailed evidence: `test_report_2026-09-14_increment8.md`.
- `SKIP: AMD/MI355X hardware unavailable`; synthetic `DEVICE_EVENT` proves schema/identity plumbing only, not AMD timing.

Status: Increment 8 implementation and focused verification complete; commit pending. Increment 14A/B is next: complete small hybrid CPU E2E through full-attention artifact loading, per-layer dispatch, stage aggregation, request completion, and metrics.

## 2026-09-14 — Initialization

- Worktree: `feature-amd-sglang-gdn`.
- Candidate HEAD at start: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2` (`Make test harness roots configurable`).
- `origin/main` currently points at the same SHA.
- Source PR reference available locally as `origin/pr-31` at `215a80a204ecdd8fc50275b5c48f14a86cedaab5`.
- No tracked or untracked source modifications were present at initialization.
- The user-provided plan is outside the worktree and remains read-only.
- The repository's `.gitignore` excludes `task_memory/`; these records are intentionally local-only.

Pending: run baseline environment/test checks, complete Review 1 inventory, and execute Unit 1 onward.

## 2026-09-14 — Increment 0 baseline

- `python -m compileall -q frontier tests`: PASS.
- Environment: Python 3.12.3, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, Plotly 6.8.0, PyTorch 2.5.1+cu124; `vllm`, `sglang`, and `aiter` are unavailable.
- `python -m pytest tests/unit -q -p no:cacheprovider`: **3144 passed, 19 failed, 25 skipped** in 87.22s. Existing failures are recorded in `test_report_2026-09-14_baseline.md`; they are unrelated missing debug/config-optimizer/analysis assets and existing documentation contracts.
- `python tests/integration/run_scheduler_refactor_fidelity.py --baseline "$PWD" --candidate "$PWD" --output <fresh-dir> --case co-location_offline_dense_model_basic_short --workers 1`: PASS, `1 passed, 0 failed`.
- `bash examples/profiling/smoke_simulator_dense_csv.sh` with an isolated metrics root: PASS; non-dummy qwen2 dense request completed and metrics CSV/JSON artifacts were written.
- `bash examples/profiling/smoke_simulator_moe_csv.sh` with an isolated metrics root: PASS; non-dummy Qwen3 tiny MoE request completed and metrics CSV/JSON artifacts were written.
- Review 1 search found the existing attention binding/operator/training paths were homogeneous-family consumers, with no hybrid resolver yet. The detailed inventory is in `review.md`.

## 2026-09-14 — Increment 1 naming cleanup

- Renamed internal `LinearAttentionImplementation`/`LinearAttentionProfile`/`linear_attention` references to `AttentionLinearOpImplementation`/`AttentionLinearOpProfile`/`attention_linear_ops` across model architecture, operator binding/contracts, profiling, training, predictor policy, parity oracle, and tests.
- No compatibility aliases were introduced. No upstream config literal was changed because the current tree had no such literal in the touched internal paths.
- `python -m compileall -q frontier tests`: PASS.
- Focused rename and binding suite (`test_model_architecture_registry.py`, `test_operator_parity_op_family_coverage_oracle.py`, `test_profiling_confirmation_attention.py`, `test_attention_tp_effective_mapping.py`): **84 passed** in 6.44s.
- `rg -n --glob '*.py' 'linear_attention|LinearAttention' frontier tests`: no matches.
- `git diff --check`: PASS.

## 2026-09-14 — Review 1 inventory completed

- Re-read the actual attention binding, memory/KV, operator/profiling, training/model-manager, simulator, and parameter/memory consumers from the current HEAD.
- Classified each listed site as `PER_LAYER`, `HOMOGENEOUS_ONLY`, or `UNSUPPORTED_HYBRID_PATH` in `review.md`.
- Established the extraction seam: the existing typed operator/family registry and model-config resolver are the safe semantic boundary; scheduler/memory migration remains pending until the per-layer contract exists.

## 2026-09-14 — Increment 2 GPU platform and MI355X

- Added MI355X and MI355X_UBB enum/config entries, explicit `gpu_platform` metadata, the supplemental `data/config/device/mi355x.json`, and CPU-only accelerator discovery/binding helpers.
- Added deterministic tests for all known GPU SKU platforms, MI355X topology, unknown SKU failure before discovery, visibility parsing, AMD/NVIDIA discovery fallbacks, and mismatch warnings that preserve the selected backend.
- AMD/MI355X hardware execution remains `SKIP: AMD/MI355X hardware unavailable`; no host SKU inference or communication-backend rewrite was added.
- Verification is recorded in `test_report_2026-09-14_increment2.md`.

## 2026-09-14 — Increment 3 DEVICE_EVENT measurement family

- Added `MeasurementType.DEVICE_EVENT`, strict profile-method/platform validation, a lazy platform-neutral `DeviceTimer`, and separate `_device_event` output filenames.
- Preserved the existing `CudaTimer` singleton behavior, including KINETO, RECORD_FUNCTION, CUDA_EVENT, and PERF_COUNTER paths; added materialized timing helpers to `TimerStatsStore`.
- Added separate shared-manager `device_event` registries and ROCm device-based event-family selection while leaving CUDA selectors and external family keys unchanged for CUDA configurations.
- CPU/mocked timer, method mapping, output-path, registry-separation, manager-selection, and legacy focused tests are recorded in `test_report_2026-09-14_increment3.md`.
- No DEVICE_EVENT writer execution was claimed on AMD; it remains `SKIP: AMD/MI355X hardware unavailable`.

## 2026-09-14 — Increment 2/3 commits and post-commit verification

- Increment 2 committed as `0989f0ed feat: add MI355X platform metadata and discovery` with `powderluv` co-author attribution.
- Increment 3 committed as `49fcd9b8 feat: add strict DEVICE_EVENT timing family` with `powderluv` co-author attribution.
- Fresh post-commit focused regression: **131 passed** across accelerator, timer, measurement-family, profiling-contract, attention-binding, operator-oracle, and model-architecture suites.
- Fresh `python -m compileall -q frontier tests`: PASS; fresh `git diff --check`: PASS; working tree is clean for tracked source changes.

## 2026-09-14 — Increment 4 hybrid attention/GDN semantic core

- Added the Qwen3.5-only hybrid resolver under `frontier/attention/gdn/`, with immutable `LayerAttentionSpec` values and explicit global-layer binding. The legacy whole-model binder now rejects hybrid schedules to prevent first-layer or majority-family fallback.
- Registered the `gated_delta_net` family with `FIXED_STATE` memory semantics, prefill/decode-only GDN operators, and a raw feature schema. Existing Qwen3-Next `linear_*` metadata remains dense unless the explicit Qwen3.5 profile identity is present.
- Added Qwen3.8 model metadata and Quark FP4 selective operation metadata. The architecture profile is marked experimental and declares current runtime limitations for prefix caching, speculative decoding/MTP, GDN P-to-D, PP/EP/attention-DP, same-batch mixed execution, and MXFP4 memory approximation.
- Added CPU-safe `GDNProfileInput`, fixed-state layout accounting, and a synthetic eight-layer fixture documented as test-only. The CSV now contains complete min/max/mean/median/std/count timing columns and validates through the shared attention schema validator.
- Added `tests/unit/test_gdn_semantic_core.py` covering Qwen3.8 pins, the eight-layer schedule, legacy dense/MLA behavior, Qwen3-Next non-activation, fixed-state/KV-helper guards, input phase guards, fixture schema, and optional-stack-safe imports.
- Verification: `python -m compileall -q frontier tests` PASS; focused combined suite **134 passed**; `git diff --check` PASS; selective quantization smoke PASS (`moe_grouped_gemm=FP4`, attention/shared expert `BF16`). Full evidence is in `test_report_2026-09-14_increment4.md`.
- `SKIP: AMD/MI355X hardware unavailable` for ROCm runtime profiling and real `DEVICE_EVENT` writer execution.

Pending: commit Increment 4 after final diff review, then implement Increment 5 unified per-layer `ExecutionTime`/ordered `StageExecutionTime` migration with CPU E2E numerical regression checks.

## 2026-09-14 — Increment 5 unified per-layer execution timing

- Added identity-bearing single-layer `ExecutionTime` records and ordered `StageExecutionTime` aggregation. Stage ownership covers pipeline communication, CPU overhead, terminal work, and draft proposer; layer records cover attention, norms, residuals, FFN/MoE, and TP/DP/EP work.
- Migrated dense, MoE, and disaggregation public stage predictors to return `StageExecutionTime`; MoE materializes one record per global layer.
- Added immutable attention query caching with complete workload/runtime identity keys and deep-copy cache hits. MoE routing and layer-specific work remain outside the cache.
- Focused Increment 5 suite: 89 passed; post-regression focused checks: 45 passed; disaggregation/EP/PD-AF regression: 93 passed; communication regression: 29 passed. `compileall` and `git diff --check` passed.
- Two-worktree fidelity against clean Increment 4 SHA `57da36835051d4ec5b512abaa28103e54fd4e380`: 5 passed, 0 failed; fresh rerun artifacts are under `/data/ycfeng/tmp/pr31-inc5-fidelity-20260914-final-1789374105` across co-location dense/MoE, sequential PDD dense, and sequential PD-AF dense/MoE.
- Three paired wall-time attempts per side all completed with 2/2 requests and 104 events. Median `init_s` ratio was 0.999589; median `total_proc_s` ratio was 1.035132; candidate event-loop wall time was 0.086823318 s versus 0.009217162 s baseline. This is recorded as an observed small CPU measurement, not a hard performance gate.
- Resolved a metrics adapter double-counting regression found by fidelity: private per-layer compatibility fields now expose first-layer raw values while explicit stage aggregates remain summed.
- Evidence report: `test_report_2026-09-14_increment5.md`.

Status: Increment 5 implementation and verification complete; commit pending. Increment 6 is next after the commit.

## 2026-09-14 — Increment 13 experimental SGLang primitives/replay/trace

- Added `frontier/profiling/experimental/sglang/` with lightweight package initialization and selected dense, packed GDN, attention, and MoE primitive contracts/builders.
- Added independent `graph_replay.py` planning/artifact helpers and lazy HIP graph replay; no whole-model capture, routing observer, runtime-cost contract, or standard `DEVICE_EVENT` producer was retained.
- Added `routed_moe_replay.py` with Frontier helper reuse, explicit expert-count JSON, deterministic top-k reconstruction, domain/count/conservation checks, and lazy AITER replay.
- Added `gdn_trace.py` with strict `.trace.json`/`.gz` parsing, known anchors, GDN layer-count checks, runtime stack provenance, and experimental summary CSV/JSON (`measurement_source=sglang_kineto_trace`, `evidence_kind=in_situ_kernel_sum`).
- Added `tests/unit/test_sglang_experimental_increment13.py`; Increment 13 plus Increment 10/11 regression: **18 passed**. Compileall and diff checks passed.
- `SKIP: AMD/MI355X hardware unavailable`; no SGLang/AITER/HIP graph runtime, RCCL, benchmark parity, or groundtruth parity claim.

Report: `test_report_2026-09-14_increment13.md`.

## 2026-09-15 — Hybrid production constructor routing identity and DEVICE_EVENT path closure

- Focused-suite ordering exposed a real identity mismatch in the new monolithic hybrid constructor: `Replica.id` is process-global, so the scheduler used an ID such as `1` while `_build_shared_routing_details()` had materialized only local `range(replica_count)` key `0`.
- Added an optional `actual_replica_ids` contract to `SklearnMoEExecutionTimePredictor`, validated its type/cardinality/uniqueness, and used the supplied IDs for monolithic routing maps. `Simulator` now passes the actual monolithic cluster replica keys; the Random Forest wrapper forwards them for MoE predictors while preserving the dense predictor signature.
- The manager path contract now asserts derived `compute_device_event_input_file`, `attention_device_event_input_file`, and `moe_device_event_input_file` values. Empty fallback paths remain empty, and explicit config fields remain supported.
- Production constructor E2E: **1 passed in 4.43s**. Routing helper contract: **1 passed in 2.60s**. Manager path contract: **1 passed in 2.73s**.
- Full focused command across timer, GDN, hybrid E2E, predictor budget/attention, MoE matrix, and measurement-family suites: **219 passed in 13.54s**.
- Direct actual-ID routing contract: **1 passed in 2.58s**; production constructor rerun after the added assertion: **1 passed in 4.60s**.
- `SKIP: AMD/MI355X hardware unavailable`; this closes CPU routing identity and constructor control flow only. It does not establish ROCm timing, AMD benchmark parity, RCCL, AITER/MXFP4, SGLang, or groundtruth parity.

Status: the focused CPU acceptance sub-step is complete. The remaining production-data limitation and all AMD runtime checks remain documented open items.

## 2026-09-15 — Post-fix full CPU unit regression

- Re-ran the complete unit suite after the monolithic routing identity repair and manager path-contract update.
- Exact command: `PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR python -m pytest tests/unit -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr31-full-unit-20260915-routingfix`.
- Result: **3276 passed, 19 failed, 25 skipped, 576 warnings** in **74.32s**. The captured log is `/data/ycfeng/tmp/pr31-full-unit-20260915-routingfix.log`.
- The 19 failing test names match the recorded baseline set exactly: one missing `frontier.config_optimizer`, nine missing debug E2E shell assets, two stale release-example documentation contracts, four existing MLA/MHA/MQA analysis-builder contracts, and three stale top-level PDD documentation contracts.
- No candidate-only unit failure remains. This is a repository-level CPU regression result; unavailable AMD/ROCm execution remains an explicit skip.

## Final verification checkpoint — 2026-09-15

- Final focused command including the new direct actual-replica-ID contract: **220 passed in 15.05s**.
- `python -m compileall -q frontier tests && git diff --check`: exit `0`.
- Worktree status: clean and synchronized with `origin/feature-amd-sglang-gdn` at `7b6a3eb1`.
- Latest commit trailer inspection: `Co-authored-by: powderluv <74956+powderluv@users.noreply.github.com>` parsed by Git.

## 2026-09-15 — Documentation publication preparation

- The user requested committing the uncommitted documentation and pushing it to the existing PR.
- Confirmed PR #33 is open with head branch `feature-amd-sglang-gdn`; local HEAD and the remote branch both pointed to `7b6a3eb192e7d911219fd942fbc55121c4b0ea54` before this publication.
- Ordinary Git status was clean because `.gitignore` excludes `task_memory/`. The publication scope is the 25 Markdown files in this task directory, explicitly staged without modifying the ignore policy or including other task directories.
- Preserve `review_report_2026-09-15_134541_HKT_7b6a3eb1_root.md` and `review_report_2026-09-15_branch_audit.md` independently. Their findings qualify earlier completion statements; publication does not resolve or supersede those findings.
- No implementation changes, test runs, simulation runs, or PR merge are part of this documentation step.
