## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Completed post-fix full CPU unit regression; 19 baseline failures remain unchanged and no candidate-only failure was found. |
| 2026-09-15 | Resolved focused-suite monolithic MoE routing identity drift and updated the device-event training-path contract evidence. |
| 2026-09-15 | Closed the temporary hybrid probe gap with a persistent real CPU Simulator E2E test; recorded the remaining prepared-predictor/model-manager boundary. |
| 2026-09-15 | Confirmed final pushed HEAD `578785bb` retains the same 19 baseline full-unit failures after whitespace hygiene. |
| 2026-09-15 | Closed the re-review tuple-contract concern in experimental SGLang graph replay at the current exact HEAD. |
| 2026-09-15 | Recorded PR #33 creation and confirmed that merge remains gated by explicit user authorization. |
| 2026-09-14 | Closed the final candidate regression and recorded Review 2, baseline failures, attribution limits, and hardware skips. |
| 2026-09-14 | Created the issue ledger; no implementation blocker is established yet. |
| 2026-09-14 | Recorded Increment 6 regression fixes, baseline comparison, and the unchanged AMD hardware boundary. |
| 2026-09-14 | Recorded Increment 12 routing-helper verification and baseline replay collection comparison. |
| 2026-09-14 | Recorded Increment 8 artifact-backed GDN training/prediction results and the remaining hybrid E2E seam. |
| 2026-09-14 | Recorded Increment 14A/B early CPU hybrid dispatch verification and the remaining production-data limitation. |
| 2026-09-14 | Recorded Increment 7 standard vLLM GDN producer and the explicit AMD runtime verification boundary. |
| 2026-09-14 | Recorded Increment 9 VLLM_ROCM backend and generic compatibility verification; one stale documentation test remains baseline-only. |

# Issues and Decisions

## Open

- AMD/MI355X hardware is unavailable in the current execution context. All candidate ROCm/DEVICE_EVENT runtime checks must remain explicit SKIP unless a permitted remote run becomes available.
- The local Python environment inventory still needs a bounded command; the initial combined probe did not produce a complete module table before output collection ended.
- Increment 14A/B's persistent CPU Simulator test uses the real GDN trainer/predictor, Replica, MemoryPlanner, scheduler/events, continuation, metrics, and op traces, with deterministic hooks for unrelated FFN/communication components. The simulator receives a prepared predictor through a test-local registry seam; the fresh model-manager load is covered separately. A production-data hybrid request/metrics E2E still needs model-specific standard attention and MoE CSVs; synthetic CPU evidence must not be presented as AMD timing or benchmark parity.
- Increment 7's vLLM wrapper is source-derived and CPU-import safe, but its vLLM/HIP execution and DEVICE_EVENT writer are not reverified without MI355X. The carried-state implementation is retained for later AMD execution; CPU tests cover planning and fail-fast semantics only.
- Increment 9 reuses the existing shared accelerator helper for linear/MoE launcher visibility and adds no backend auto-selection. The candidate has no ROCm runtime evidence; the standard VLLM_ROCM wrapper remains source-retained plus CPU metadata tests.
- Increment 10 has no AMD runtime evidence: current vLLM functional fused experts, online MXFP4 packing, and AITER backend selection are source-integrated and CPU-contract tested only. The model quantization metadata path derives `PrecisionType.FP4`; there is no standalone `--use_mxfp4` flag. `SKIP: AMD/MI355X hardware unavailable`.
- Increment 11 has no GPU collective evidence: the standalone runner is source-integrated with NCCL/RCCL selection and dtype-aware bytes, but Ray/ROCm hardware is unavailable. `SKIP: AMD/MI355X hardware unavailable`.
- The post-fix full CPU unit run reports **3276 passed, 19 failed, 25 skipped, 576 warnings**. The 19 failures match the recorded baseline names and categories exactly: one missing `frontier.config_optimizer`, nine missing `tests/debug/e2e-level/monolith_mode` assets, two stale release-example documentation contracts, four existing MLA/MHA/MQA analysis-builder contracts, and three stale top-level PDD documentation contracts. These failures are outside this selective integration scope.
- Production-data hybrid simulation with complete model-specific standard attention/MoE CSVs is not available in this CPU environment. The synthetic profile fixture now exercises the real manager constructor and the production constructor E2E passes; the persistent request/metrics E2E still uses a prepared-predictor seam for unrelated operator timings. This does not establish AMD timing, benchmark parity, or groundtruth parity.
- GitHub's contributor API may not immediately display `powderluv`; commit-level attribution is preserved on the selective integration commits through `Co-authored-by: powderluv <74956+powderluv@users.noreply.github.com>`. Commit `e8ac59ae` predates the attribution pass and has no trailer; rewriting history is intentionally out of scope.

## Resolved

- **Monolithic MoE routing replica identity mismatch:** `Replica.id` is process-global, while the shared monolithic routing map had been keyed by local `range(cluster_num_replicas)` values. Focused-suite ordering therefore produced scheduler lookups such as `target_replica_id=1` against a map containing only key `0`. `Simulator` now passes actual cluster replica keys through the Registry; the MoE predictor validates and materializes those IDs, while direct predictor tests retain local keys when no IDs are supplied. Production constructor E2E, routing identity, and the full 219-test focused suite pass.

- **Incomplete manager device-event path contract:** `get_training_file_paths()` now returns explicit derived or configured `compute_device_event_input_file`, `attention_device_event_input_file`, and `moe_device_event_input_file` values. Empty fallbacks remain empty, and the path contract test covers the expected taxonomy.

- **Incomplete GDN structured operator schema:** `GDNPredictor.predict_attention_time()` previously emitted only the active phase core key, so op-level trace consumers failed when they enumerated the missing inactive phase key. The predictor now emits both `gdn_core_prefill` and `gdn_core_decode`, assigning `0.0` to the inactive phase while preserving active timing. The focused four-test hybrid suite, compileall, and diff checks pass.

- **Temporary hybrid Simulator probe gap:** The initial probe failed on decode EP lanes because its fake internal predictor called the GDN attention estimator when `include_attention=False`, yielding `query_len must be positive`. The persisted test now returns zero attention with all required GDN operator keys for FFN-only lanes, trains/loads a fresh GDN artifact, executes the real Simulator path, and verifies request/metrics/ledger/trace outputs. The remaining prepared-predictor injection is documented as a test seam because production standard attention/MoE CSVs are unavailable.

- **Hybrid full-attention binder failure:** the hybrid path previously reached the homogeneous whole-model binder after the GDN branch. `SklearnExecutionTimePredictor` now resolves `bind_layer_attention(model_config, layer_id)` for full-attention layers, and MoE execution results carry the resolved layer identity. The eight-layer CPU stage checkpoint passes with exact `G G G A G G G A` order.
- **Hybrid model-manager attention skip:** the temporary GDN guard prevented dense full-attention estimator training. The guard was removed; Qwen3.5 hybrid models are excluded only from the MLA-only path and are now eligible for dense full-attention model training.
- **Lightweight predictor compatibility:** a test double without `_gdn_predictor` raised during the new branch. Safe `getattr(..., None)` access restores the existing mock contract without changing production dispatch.
- **Profiling family-binding seam regression:** the hybrid runtime resolver replaced the profiling module's imported `bind_attention_family`, breaking an existing MLA contract test that patches that symbol. `ModelConfig.get_attention_family()` now uses the local binder for homogeneous configurations and the runtime resolver only when the resolved layer schedule contains GDN. MLA, hybrid, and profiling contract tests pass.
- **Missing examples profiling history:** `examples/profiling/README.md` lacked the repository-required modification-history block. The block was restored without changing the top-level README, and the profiling documentation contract passes.

- Use the current main SHA as the baseline because `origin/main` and the branch HEAD match at initialization.
- Use `origin/pr-31` only as a read-only source reference; do not cherry-pick it.

## Increment 5 resolved issue

- **Metrics double-counting during first fidelity run:** `StageExecutionTime.__getattr__()` originally summed private per-layer fields. The legacy metrics adapter interprets those fields as one layer, so the MoE case emitted a 32x `attention_all_reduce_time`. The adapter contract was preserved by returning first-layer raw values for private compatibility names and retaining explicit stage aggregates for stage-aware consumers. The corrected case and the complete five-case fidelity rerun passed.

## Increment 5 observation

- The paired three-attempt CPU wall-time check completed successfully on both revisions. Candidate `sim_wallclock_s` median was 0.086823318 s versus 0.009217162 s baseline, while `total_proc_s` median was 2.280853388 s versus 2.203441358 s. This is an observed cost for the new per-layer path, not a blocker or a benchmark parity claim.

## Increment 6 resolved issues

- **Mock scheduler preemption regression:** The new GDN preemption guard assumed `_replica_config` existed on every scheduler object. Existing tests construct lightweight objects without that field. The guard now reads the optional config safely and still rejects initialized GDN state-dropping preemption before request mutation. Focused preemption tests pass.
- **Legacy predictor sentinel regression:** Increment 5's stage wrapper rejected non-`ExecutionTime` sentinels used by admission-focused tests. Production helper results remain typed `ExecutionTime` and are wrapped; legacy test doubles pass through at the predictor boundary. MoE EP admission tests pass.
- **GDN registry expectation drift:** Registering the execution-enabled GDN family intentionally expanded the attention-family views. The registry unit contract now includes that family; no execution-enabled family is silently omitted.

## Increment 6 baseline comparison

- Fresh full unit regression reports `3206 passed, 19 failed, 25 skipped`. The 19 failures are the same baseline categories: missing `frontier.config_optimizer`, missing debug e2e scripts, three analysis builder failures, and stale release documentation contracts. They are environment/repository baseline conditions and remain outside Increment 6 scope.
- Real Qwen3.8 capacity checking reaches the D57 2-byte approximation and raises the ordinary modeled OOM under a deliberately small 1 GiB budget. This is expected behavior and is not a blocker.

## Increment 12 resolved issue

- **Baseline replay collection error:** `tests/unit/test_moe_ep_baseline_replay.py` fails before test execution because importing `frontier.operators.families` re-enters `frontier.config` while the family module is partially initialized. The same `get_operator_family` import error and `step3_text` unknown-family error reproduce on candidate and detached `0d1b87a4`. It is therefore a baseline/environment issue; Increment 12 does not add a fallback or expand into an unrelated import-cycle refactor.

- Increment 13 CPU contracts and source-boundary checks pass. Native SGLang/AITER builders, HIP graph capture, and kernel timing remain unexecuted: `SKIP: AMD/MI355X hardware unavailable`.
- The experimental trace importer intentionally writes only `gdn-trace-summary.csv/json` in a caller-selected directory. It does not validate or emit standard GDN training CSVs; `GDNTrainer` must not discover these artifacts.
- The earlier graph-replay review concern about Dense/GDN/Attention/MoE builder tuple lengths is resolved at `frontier/profiling/experimental/sglang/graph_replay.py:205-240`. `_make_replay_call()` provides the common `(fn, reset, check, backend, ...)` contract used by `profile_graph()`; current exact-HEAD re-review passed the 7-test Increment 13 suite. Native HIP execution remains unavailable.

## Review 2 closure

- MetricsStore's StageExecutionTime handling, hybrid runtime-family selection, SGLang replay tuple/repetition validation, and GDN dataset fingerprint checks are resolved and covered by the final focused suites.
- The bounded Python inventory is complete: Python 3.12.3, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, Plotly 6.8.0, PyTorch 2.5.1+cu124; `vllm`, `sglang`, and `aiter` are unavailable.
- No approval is pending for local code/docs or verification. PR #33 is open after the authorized push; merge remains explicitly prohibited until the user gives separate authorization.
