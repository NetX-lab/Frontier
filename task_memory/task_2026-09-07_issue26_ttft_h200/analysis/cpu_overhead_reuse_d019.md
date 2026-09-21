## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Independently inspected historical CPU measurement/replay code and current hooks; specified a conditional CPU phase after CUDA-event-span correction. |

# D019 CPU-overhead component reuse review

## Verdict and scope

**ACCEPT the staged reuse direction; REVISE any plan to copy the historical timing rows or enable CPU overhead immediately.** The repository already contains the CSV, validation and predictor mechanisms needed for a bounded future phase. Historical “CPU” helpers also contain residual allocation, clipping and shape-copying approximations that do not independently measure missing CPU critical-path time.

This is a source/document review, not a new measurement or numerical validation. D019 explicitly places CUDA-event-span RCA and correction before CPU integration. No GPU/network operation, profiling execution, test execution, production edit, CPU-enable change or workflow redesign was performed. Only this document is owned by this review.

Paths below use these exact roots:

- **H** = `/data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908`, branch `v0.3-hopper-testbed`, HEAD `9fd7fea584dba998d3b8c12af7233c6f0c7deeed`.
- **HT** = `H/task_memory/task_2026-04-19_frontier_vllm_v1_accuracy_matrix`.
- **A** = `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`, the active Frontier worktree.
- **T** = `A/task_memory/task_2026-09-07_issue26_ttft_h200`.
- **V** = `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`, diagnostic vLLM branch `feature/frontier-comparison-instrumentation`, inspected baseline `8453dd342c6aa2721aaf4b410998aab2f38bc2ec`. Line numbers identify the inspected source and may move in the forthcoming phase-1 edits.

The local reference is outside `tmp`; `T/analysis/historical_accuracy_matrix_review.md` records its location. HT uses `README.md`, `task_plan.md`, `notes.md`, `progress.md`, `issues.md`; guessed `requirements.md`/`summary.md` are absent. Its README excludes raw `artifacts/` and generated `profiling/` from the curated checkout. Historical vLLM gitlink `1109c4f16e2e4c565fd5c85564f2a6eacc164393` is not initialized here. Consequently historical hook placement cannot be certified solely from the old script's field names. Current V hook placement is locally inspectable, so no network access is needed to prepare this plan.

## Independent evidence ledger

| Claim | Label | Direct evidence and implication |
| --- | --- | --- |
| A has a reusable CPU profiling module already. | Evidence | Byte comparison of all 13 Python files under `H/frontier/profiling/cpu_overhead/` with A found 12 identical. Only `backends/vllm_mapping.py` differs: A adds a 1e-9 ms floating-point epsilon and emits `measurement_type=CUDA_EVENT`. No module copy is necessary. |
| The vLLM backend itself measures vLLM CPU phases. | Evidence: false | `backends/vllm_backend.py:VllmCpuOverheadProfilerBackend.start/create_runner` loads pre-collected JSON/JSONL and returns a normalized row. It performs no live vLLM execution. |
| The replay CLI accepts the current 4096/0 prefill identity unchanged. | Evidence: false | `create_runner` matches fixed defaults `num_prefill_tokens=256`, `num_decode_tokens=3*batch_size`, `scheduling_mode=sync`; its public runner arguments have no explicit token-shape fields. JSON normalization supports explicit shapes, but CLI runner selection does not. |
| Historical test success establishes physical CPU attribution. | Evidence: false | Inspected tests validate fixtures, arithmetic, schema, row selection and a stub estimator. None proves current DP2 H200 critical-path attribution. Historical status audit explicitly keeps overhead/scale surfaces case-local. |
| Current first-forward residual can be added as CPU. | Unknown / not established | Existing `T/analysis/first-batch-op-rca/timing_contract.md` separates official TTFT, CUDA-event span and kernels. Model-span host submission gaps and DP/collective waits may already be included; remaining unmatched operators and instrumentation perturbation are not CPU measurements. |
| Existing CPU CSV injection preserves route/admission timing automatically. | Evidence: false | `A/frontier/entities/time_components.py:OverheadTime.simulated_total_time` sums stage overhead; `ExecutionTime.total_time` adds it to model time. This interface does not by itself create separate frontend route/enqueue/snapshot events. |

## Historical measurement and test components: actual behavior

All paths in this section are relative to H unless prefixed HT.

| Component / symbol | Actual inputs and operation | Reuse decision |
| --- | --- | --- |
| `HT/scripts/run_vllm_colo_clean.py:_local_container_command` (around lines 495–570) | Enables Frontier batch instrumentation; optional `--enable-schedule-diagnostics` exports `VLLM_FRONTIER_SCHED_LOG_PATH`, `VLLM_FRONTIER_SCHED_DECISION_LOG_PATH`, `VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH`, plus trace warmup control. Splits historical head/headless logs. | Reuse explicit mode/manifest/namespace controls, through the current verified H200 runner. Do not run its two-host Docker/SSH lifecycle wholesale. Its historical “clean” label includes synchronization-bearing instrumentation. |
| `HT/scripts/materialize_vllm_schedule_gap_cpu_overhead_csv.py:_materialize_lane_rows/_aggregate_rows` | Consecutive representative batch completion timestamps minus the current batch CUDA-event duration give an inter-batch gap. Scheduler start/end walltime is matched by ordered request IDs and scheduled-token vector. PP first-rank preprocess timestamps provide a median per batch ID. It writes residual `max(gap - schedule - preprocess, 0)` into `ray_comm_time_mean`; Sample/output fields are zero. | CSV aggregation and source/sample-count metadata are useful examples. Residual attribution is not a measured Ray or CPU phase. Missing schedule matches are silently skipped; negative gaps are clipped. Current per-worker/DP/iteration/request-progress identities must replace the head/headless and repeated-shape matching assumptions. |
| `HT/scripts/materialize_vllm_batch_gap_cpu_overhead_csv.py:_representative_rows/_materialize` | Picks latest timestamp for duplicate batch IDs in each lane, then assigns all positive completion-gap-minus-CUDA time to `ray_comm_time`. | Cadence diagnostic only. Workload idle, IPC, queue waits, synchronization, logging and asynchronous overlap are unresolved. No direct import as CPU training truth. |
| `tests/analysis/online_colocation_llama32_1b_v1/measure_cpu_overhead.py:main` | Calls next-scheduler-start minus current-start “wall”, subtracts batch CUDA duration, assumes iteration ID equals batch ID; uses list index for metadata. Last step adds an explicit 2 ms estimate. Drops the first iteration as warmup. | Not suitable for this DP2 case. Preserve as historical exploration only; do not reuse its 2 ms estimate, ID assumptions, warmup rule, or clean-run-minus-diagnostic-run CPU claim. |
| `tests/analysis/ttft_llama31_8b/measure_vllm_prefill_cpu_overhead.py:parse_prefill_iteration_timing_stats_ms` | Parses iteration start/decision/end; derives scheduler wall, pure-schedule and tail durations. Compact single schedule events use next-step timestamp as a proxy end. | Useful structured parsing/explicit-clock patterns. Current `decision` means multiple per-request decisions, and compact cadence is not scheduler method CPU duration. Reconfirm producer contract before reusing “pure”/“tail” labels. |
| Same file: `extract_batch_scope_trace_stats` (around line 471), `_measure_case` | Selects `frontier_batch_<id>` user annotation, sums overlapping host/GPU annotations by name and derives `max(host annotation duration - GPU annotation duration,0)`. Collects CUDA-runtime categories and compares instrumented vs clean TTFT. | Reuse phase inventory and explicit perturbation reporting ideas. Name-based duration subtraction is not interval union, thread accounting or causal overlap removal. Nested/partial-overlap intervals and cross-run subtraction must not become additive CPU truth. |
| Same file: `_dedup_host_overhead_with_scheduler_tail` (750), `compute_python_residual_ms`, `_build_contract_preview_row` (957) | `scheduler_tail_clip` proportionally scales all host op totals to a tail budget. Residual budget can be TTFT extra or scheduler wall; the preview places “Python residual” in `ray_comm_time_mean`. Preview only maps Preprocess, Sample, Postprocess. | Do not reuse clipping or residual fitting as measurement. Inspect Bookkeep separately; omission cannot imply zero. |
| Same file: `_build_direct_sampled_contract_rows` (1058) | Deduplicates batch signatures, then copies one `base_row` timing vector to every signature. A minimum of two signatures is a shape-count check. | Cannot establish two independently sampled shapes; do not expand first-prefill timing into mixed/decode rows. |
| `HT/scripts/materialize_case_cpu_overhead_variant.py` | Filters phase, optionally adds PP boundary means to a selected target (default `ray_comm_time_mean`). | Phase filtering may be adapted. PP boundary allocation is irrelevant to current PP1 and is not a CPU attribution method. |
| `HT/scripts/scale_cpu_overhead_rows.py:ScaleRule/_scale_row` | Applies user-provided scales to exact batch-size/prefill/decode keys, defaulting to scheduler, preprocess and Ray fields. | Do not import historical scaling rules or introduce shape-specific fit coefficients. |
| `frontier/profiling/cpu_overhead/benchmark_runner.py:BenchmarkRunner` | Live backend uses **Sarathi** `LLMEngine`, dummy weights, 256 prompt tokens, output length `3*batch_size`, its CPU metrics and step-loop residual for Ray. | Not a vLLM V1 H200 DP2 direct measurement. No Sarathi installation or live profiler launch is necessary for phase-2 reuse. |

HT `runs/status_audit/mainline_post_a24_current_state_20260427T233223+0800.md` explicitly states that case-local overhead/scales must not be promoted without multi-case evidence. We preserve the useful mechanisms and this limitation, rather than treating historical accepted leaves as fresh numeric truth.

### Tests actually inspected

- `tests/comparison/chunked_prefill_online/test_materialize_vllm_schedule_gap_cpu_overhead_csv.py`: synthetic 10 ms batch completion gap and 3 ms CUDA duration, 0.5 ms schedule and 1 ms preprocess expect 5.5 ms Ray residual. It proves the subtraction implementation, not the residual's cause.
- `tests/unit/test_cpu_overhead_vllm_backend.py`: native fields map Preprocess to prepare-inputs and Postprocess + Bookkeep to process-outputs; negative residual rejection and identity/duplicate-row behavior.
- `tests/unit/test_cpu_overhead_pipeline_replay_consistency.py`: contract→native JSON→backend round trip on 256 / 3*batch-size fixtures. It intentionally replaces schedule mean with schedule median through the builder.
- `tests/unit/test_measure_cpu_overhead_direct_sampling_n2.py`: verifies copied rows have distinct signatures and rejects fewer than two; does not sample either signature.
- `tests/integration/test_cpu_overhead_minimal_loop.py:test_cpu_overhead_csv_to_predictor_training_and_injection`: synthetic CSV, model/TP filtering, fake lookup estimator and injected getter values. Good integration seam; not model-accuracy proof.
- `tests/integration/test_chunked_prefill_cpu_overhead_e2e_smoke.py`: runs the replay CLI on synthetic 256 / 3*batch-size inputs and verifies CSV output.
- `tests/profiling/a800/llama3.1-8b/run_cpu_overhead_replay_backend.sh`: direct CSV→JSON→CLI→CSV consistency wrapper. Hard-coded A800-era paths, TP2/4/request-count combinations and duplicate selection by TP/batch size require adaptation; this wrapper is conversion validation, not fresh measurement.

These tests were read, not executed in this review. Future tests should target the actual 4096/0 and selected same-case mixed/decode identities, with independently measured rows when those phases are integrated.

## Current vLLM host boundaries and overlap ownership

Inspected `V/vllm/v1/worker/gpu_model_runner.py:GPUModelRunner.execute_model`:

| Region | Current boundary | Correct ownership constraint |
| --- | --- | --- |
| Scheduler | `scheduler.py:Scheduler.schedule`, start event around 368 and end around 973. Decision logger uses `time.time()`; compact schedule also exports a monotonic endpoint. | Method wall duration can include diagnostic logging and bookkeeping. Pair within the same scheduler/DP process; individual per-request decisions are not one global “schedule end.” |
| Preprocess annotation | Around 2272: `_update_states`, possible `prepare_inputs_event.synchronize()`, `_prepare_inputs`, `_preprocess`, graph-dispatch selection. | Host wall includes any synchronization and GPU submission work. PP boundary preprocess timestamps begin only around 2291, after state update/event wait, and end around 2312 before graph dispatch. They are a narrower interval than the annotation. |
| DP padding | `get_dp_padding` around 1927. | Current enforce-eager exits with `(0,None)`; do not assign the non-eager metadata collective to current preprocessing. Current forward-context metadata synchronization is a different path. |
| Forward CUDA events | Start around 2409, before entering `set_forward_context`; end around 2453 after model/context exit. | CUDA-event span can include GPU stream idle while host performs forward setup/launches and waits for peers. Do not subtract kernels and call the entire difference CPU, or add those in-span delays a second time. Phase 1 must resolve this contract first. |
| Forward torch annotation | `frontier_batch_<id>` around 2417 is entered **after** `set_forward_context` and encloses Forward/model only. | Historical trace parser restricted to this annotation cannot recover current Preprocess/Postprocess/Sample/Bookkeep, which are outside it. Missing phase annotations are missing evidence, not zero overhead. |
| Postprocess | Around 2500, after forward end; includes hidden-state indexing, `model.compute_logits`, possible PP behavior and grammar masks. | This contains GPU work as well as CPU; missing logits must remain a GPU op, not be folded into CPU. Current PP1 does not require importing historical PP boundary additions. |
| Sample | Around 2545, `_sample`. | May run GPU sampling kernels. Distinguish GPU work, host submission and synchronization; the field name `sampler_e2e` is not a guarantee of pure CPU. |
| Bookkeep | Around 2548, `_bookkeeping_sync`. | May include GPU→CPU completion waits and CPU output/state handling. The native normalizer includes it, whereas the old preview omits it; future accounting must identify its actual first-token contribution. |
| Engine output / API stats | Worker result→core update/output publication→API output handling→official first-token stats. | Only the missing critical-path portion before official TTFT completion is eligible. Work after that endpoint and client-side delivery do not belong to the primary server metric. |
| Route→enqueue / waiting | Existing route/snapshot/enqueue diagnostics. | Not an unconditional CPU bucket. Waiting can reflect engine busy/dummy/collective scheduling. Do not place it in stage occupancy merely because it happens on a host clock. |

The future accounting ledger needs **same run + host/process + DP/TP/PP + scheduler iteration + worker batch + request/token/progress composition + named monotonic endpoints**, retaining raw measured intervals. Use a host-clock anchor only if comparing host and GPU absolute timestamps; event duration alone does not supply one. Keep annotation nesting, GPU dependencies and other-rank waits explicit. Across parallel ranks, critical-path joining is not summing rank durations, and the minimum rank duration is not a measured uncontended wire or CPU cost.

## Existing Frontier training and insertion path

1. `A/frontier/profiling/cpu_overhead/schema.py` identity is model, batch size, TP, prefill tokens, decode tokens, scheduling mode; precision is required metadata. Raw same-run DP/PP/worker/CPU provenance must remain in a sidecar before aggregation, because these are not schema identity fields.
2. `backends/vllm_mapping.py:normalize_vllm_v1_native_record` maps preprocess directly, sums postprocess + bookkeep, and derives Ray as `step_wall_time_ms - (schedule + preprocess + sampler + postprocess + bookkeep)`. It **does not subtract model CUDA time**. Feeding a true complete GPU-inclusive iteration wall here would label its unremoved GPU time Ray. The old contract builder instead manufactures `step_wall_time_ms` from five selected CPU/residual columns; its round-trip passes do not validate physical iteration time.
3. `validation.py:validate_cpu_overhead_dataframe` and `SklearnExecutionTimePredictor._load_cpu_overhead_df` (1470) already load validated CSV. The inspected load filter selects model and attention TP; phase2 should supply one frozen precision/scheduling/host configuration per case file instead of mixing configurations the predictor cannot distinguish.
4. `_train_cpu_overhead_models` (3411) trains five outputs, using median for schedule/sampler/prepare/process and mean for Ray. Features are batch size, prefill tokens and decode tokens; an exact lookup is persisted. Runtime `_get_cpu_overhead_features` (6788) uses the actual batch token split. Explicit 4096/0 rows can therefore enter this existing CSV path without the replay CLI's fixed-shape runner.
5. `OverheadTime.simulated_total_time` (time_components.py:641) adds these components once per stage; `ExecutionTime.total_time` (1479) converts milliseconds to seconds and adds model time. It is not per-layer overhead. For shared MoE forwards, verify source-batch ownership and completion timing with the current D017 runtime before enabling any nonzero values.
6. The current `T/config/frontier_uniform_candidate.json` explicitly has `random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling=true`. Leave it unchanged during D019. A's native normalizer attaches `measurement_type=cuda_event` as profile-family metadata; that label must not be interpreted as proof that CPU fields were physically measured by CUDA events.

A minimal future adapter can emit a case-local, validated CPU CSV directly from proved disjoint components, preserving original intervals and coverage in a sidecar. It should not require replacing the profiling backend or extending a shared runner contract merely to import 4096/0. If no observed component belongs to Ray, report the evidence gap explicitly; a required numeric column does not authorize populating it from unexplained latency.

## Corrected conditional phase-2 plan

Dependency: **phase-1 complete operator mapping/shape checks → CUDA-event-span RCA and scoped repair → fresh same-case validation → CPU boundary/data audit → {minimal measurement adapter and schema validation, existing injection-path verification} → reviewed CPU integration → fresh clean E2E comparison**.

1. **Close phase 1 first.** Keep compute/memory, communication, launch/idle and instrumentation perturbation separately evidenced. Missing op pairs and the insufficient multi-batch collector coverage remain open until their own gates pass. No CPU constant may stand in for them.
2. **Freeze a component ownership table from the repaired version.** Start with the first formal 4096-token prefill; include only additional same-case batch identities needed to support subsequent runtime injection. Record scheduler, preprocess, forward, logits/sample, bookkeep and API endpoint boundaries, the owning process/rank and whether a segment is already represented in a profile, queue interval or collective. Resolve stage occupancy versus endpoint-only work before changing shared workflow semantics.
3. **Collect only missing direct boundaries in a separately labeled diagnostic run once that phase is authorized.** Reuse current identity, warmup namespace and low-overhead logger controls; add narrowly scoped same-clock endpoints if existing hooks cannot establish a needed phase. Do not activate full torch profiling by default merely to reproduce a historical script's shape. Validate perturbation against a matching clean/batch control.
4. **Materialize measured rows, not residual targets.** Per identity, retain sample counts, timing provenance and the disjoint critical-path definition. Do not scale to fit TTFT, clip negative differences into “measured zero,” copy one shape's timing to another, or sum nested annotations/ranks. Retain unresolved intervals outside the training truth.
5. **Reuse the existing CSV→predictor→stage seam.** Validate first-prefill lookup and non-duplication with a small adaptation of `test_cpu_overhead_minimal_loop.py`; separately verify mixed/decode source ownership only where the final integration uses it. CSV validity and process exit are insufficient: check actual stage duration/completion and official request metrics after fresh execution.
6. **Integrate only reviewed missing work.** Preserve D006 official TTFT and internal queue→prefill diagnostics. Use the existing model only for components its insertion point actually represents. If measured route/output overlap requires a shared workflow change, prepare that concrete design for YC rather than silently using stage-overhead fields as a surrogate.

Minimum future components: (a) current vLLM identity/endpoints with narrowly scoped missing host-phase intervals; (b) one case-local materializer + provenance/coverage sidecar; (c) existing CPU schema/validation and exact-shape predictor path; (d) targeted identity/overlap/injection checks and a fresh E2E comparison. A new profiling framework, Sarathi runs, historical PP2 matrix replay, historical scale tuning, generalized async workflow refactor and historical raw-data download are not needed to prepare this bounded phase.

## Verification record and open items

Read-only checks performed: actual HT documents and scripts above; named test bodies; current V producer scopes; current A schema/mapping/training/runtime insertion; direct file-content comparison. Example reproducible source checks:

```bash
git -C /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908 rev-parse HEAD
git -C /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908 status --short
diff -u /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908/frontier/profiling/cpu_overhead/backends/vllm_mapping.py /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/frontier/profiling/cpu_overhead/backends/vllm_mapping.py
sed -n '175,320p' /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908/task_memory/task_2026-04-19_frontier_vllm_v1_accuracy_matrix/scripts/materialize_vllm_schedule_gap_cpu_overhead_csv.py
sed -n '1058,1106p' /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908/tests/analysis/ttft_llama31_8b/measure_vllm_prefill_cpu_overhead.py
sed -n '2248,2565p' /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/v1/worker/gpu_model_runner.py
sed -n '3411,3463p' /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/frontier/execution_time_predictor/sklearn_execution_time_predictor.py
```

Observed result: **component discovery and source review complete**; historical test results are not re-certified, no new CPU milliseconds are reported, and phase-2 numerical/integration acceptance is **NOT RUN**. Read errors for nonexistent guessed historical filenames were resolved by following the actual curated layout. No runtime/environment failure was encountered.

Open: phase-1 repair/validation, current host-phase coverage and perturbation, interval/overlap attribution, exact per-shape fresh CPU inputs, and correct source-batch/runtime ownership before enabling CPU. Historical raw runs and exact old vLLM producer source are absent locally; they are not prerequisites for the proposed fresh-current-version phase, and old numbers remain excluded.
