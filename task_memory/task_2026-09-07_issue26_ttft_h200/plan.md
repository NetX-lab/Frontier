## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Recorded profiler failures, standalone validation and queued same-node normal/skip replay. |
| 2026-09-10 | Recorded approved single-call AR bypass and standard-scale gate. |
| 2026-09-10 | Recorded successful standard H800 replay reproduction and corrected baseline provenance. |
| 2026-09-13 | Added parallel clean PPLX boundary and native naive Kineto profiling lanes under the ten-warmup/1100-row gate. |
| 2026-09-13 | Closed the PPLX clean-boundary lane and native CUDA-only Nsight artifact lane; retained Nsight category timing as diagnostic because it remained above the clean reference. |
| 2026-09-10 | Recorded the ten-warmup H800 bounded formal-span result and its validation limits. |
| 2026-09-08 | Updated cross-session handoff and live candidate continuation state. |
| 2026-09-08 | YC deferred naive protocol modeling as optional; resumed D019 with ideal communication. |
| 2026-09-08 | YC approved D019 execution and proven gated SiLU/reduction repairs; A/B/C started. |
| 2026-09-08 | Proposed D019 complete op mapping and CUDA-first correction with parallel ownership; pending YC review. |
| 2026-09-08 | Delivered parallel RCA and fresh H200 supplements; retained unresolved numerical attribution and production design decisions. |
| 2026-09-08 | Completed the authorized shared-forward design and specified source-local attention and request-level verification. |
| 2026-09-08 | Recovered proxy configuration and launched fresh replay-02 after confirming old job termination. |
| 2026-09-08 | Separated RR correctness from weighted-load policy alignment; recorded replay uncertainty. |
| 2026-09-08 | Recorded confirmed communication/proxy decisions and completed H200 runtime/backend checks. |
| 2026-09-07 | Initialized the fresh H200 single-case calibration task. |

## Current execution — 2026-09-10 same-node ABBA

`{canonical_source_and_chain_audit, standalone_analyzer_validation} -> same_allocation(normal_1 -> skip_1 -> skip_2 -> normal_2) -> four_arm_drain_identity_and_scale_review -> paired_delta -> conditional_completion_queue_capture -> conditional_reconciliation`

The audit and analyzer validation run in parallel; root serializes all GPU work and owns docs. First two prerequisites completed; RJob yc26-h800-paired-normal-skip-20260910-01 queued. Each arm uses full standard clean and batch replay, 3x100 warmup plus100formal. Report four-rank median/P90/max/spread per arm and two pair deltas; two runs per mode are limited replication, not a reliable population P90. CUDA/op gate and clean E2E/CPU integration remain downstream.

# Plan

## Active checkpoint — D019 executing with ideal communication; D020 deferred optional

### 2026-09-13 completion status — PPLX and native CUDA-only profiling

Both authorized H200 lanes have completed their required artifact gates. PPLX
runs `yc26-h200-pplx-clean-20260913-03` and `-05` are platform `Succeeded`
with 10 drained warmups, 100 formal requests, 1100 rows, validated first
formal identity/predicates, and complete DP0 TP0–TP3 event rows. Their
`VLLM_ALL2ALL_BACKEND=pplx`, `VLLM_MOE_DP_CHUNK_SIZE=4096` boundary medians
are `182.860077` and `183.436272 ms`.

Native run `yc26-h200-nsys-cuda-only-20260913-04` is platform `Succeeded`
with the same 10/100/1100 client gate and complete Nsight CUDA/SQLite
artifacts. After removing the generic `collective` classifier token, the
four inferred DP0 device windows are `106.824398/102.003857/99.196101/101.641481 ms`;
the category medians are compute `33.320378 ms`, communication `59.362114 ms`,
memory `2.209333 ms`, all activity `94.891825 ms`, and idle `4.918915 ms`.
The first formal client-to-token marker path nevertheless spans `11,925.042 ms`,
showing that host marker time and the Nsight trace clock cannot be substituted
for one another. Nsight and Kineto remain diagnostic and do not provide a
clean 79 ms decomposition; no Frontier correction or clean/diagnostic
reconciliation follows. A lower-perturbation retry is optional and requires a
new scoped decision.

### 2026-09-13 parallel clean-span investigations

The active order is:

`{PPLX_boundary_replay, naive_Kineto_capture} -> {identity_and_warmup_validation, trace-window/category_validation} -> clean-span reports -> CUDA gap interpretation`

Both lanes use H200 `step_main + h200 + num_gpu_blocks_override=310809`, the
fixed 4096-prefill/1024-output case, and ten drained 100-request warmup rounds
followed by 100 formal requests. The PPLX lane uses the minimal source patch
`f025cc30` and records one selected first-formal CUDA event envelope per DP0
TP rank with one post-end synchronization. The naive lane uses the clean source
with `VLLM_ALL2ALL_BACKEND=naive`, all Frontier instrumentation disabled, and
vLLM's Kineto profiler only. A trace decomposition is accepted only after
formal client identity, exact first-forward predicates, trace count and real
kernel category inspection pass; profiler spans remain diagnostic and cannot
replace the clean 78--79 ms reference. GPU allocations are serialized by the
root coordinator even though source review and parser validation proceed in
parallel.

### 2026-09-13 execution result

The PPLX launch/analyzer issue is resolved for the completed artifact: the
analyzer now selects the unique validated DP lane instead of assuming DP0, and
the PPLX boundary-only replay has ten warmups, 1100 client rows and four
selected TP rows on DP1. Its median is `542.338012695 ms`; this fused protocol
remains a separate diagnostic result.

The naive Kineto run produced the complete ten-warmup/100-formal artifact and
passed independent identity, trace, marker and category checks. The platform
RJob is `Failed` only because the worker's final row-count Bash expression was
invalid after data collection; the expression is fixed in commit `f1f5ba74`.
The run-03 formal DP0 marker median is `112.731364746 ms` with compute,
communication, memory and idle unions recorded in
`test_report_2026-09-13_naive_kineto_run03.md`. Because Kineto perturbs launch
and event timing, this is diagnostic decomposition only and cannot replace the
clean `78.118782043--79.307357788 ms` span. No Frontier correction or
reconciliation follows from either lane. Optional PPLX chunk-size attribution
is deferred until a new scoped decision.

## Synchronization-anchored post-MoE AR RCA phase — 2026-09-09

The current RCA must start at the nearest completed synchronization boundary for each TP rank. Source inspection establishes that naive MoE `combine` uses a DP group (`[0,4]`, `[1,5]`, `[2,6]`, `[3,7]` for DP2/TP4), while post-MoE AR is the first TP-wide collective (`[0,1,2,3]` or `[4,5,6,7]`). Therefore combine completion cannot be treated as a TP-wide barrier. The user requested direct evidence for the cause of recurring late TP participant submission; rank-local post-MoE scope alone is insufficient.

Execution dependency and parallelism:

`{A_combine_contract, B_ar_bypass_design, C_timeline_observer, D_rank_input_audit} -> review_measurement_matrix -> {E_baseline_scalar_skip_AB, F_sync_anchored_timeline} -> cross-check_bypass_delta_and_late_rank_cause -> YC_review_before_any_production_correction`

Lanes A-D run in parallel with disjoint ownership. GPU experiments E-F are serialized by the root scheduler on the existing H200 allocation. All diagnostic vLLM edits live in `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`; the frozen Frontier checkout and production code remain untouched.

### Lane A — combine boundary contract (completed read-only)

Verify process groups, rank mapping, tensor shape and asynchronous enqueue semantics from `all2all.py`, `parallel_state.py`, `cuda_communicator.py` and `pynccl.py`. Acceptance is a source-backed statement of which ranks are synchronized and what completion means. A DP-pair all-reduce is recorded as the boundary for each rank; TP-wide alignment is not assumed.

### Lane B — diagnostic AR bypass A/B (design completed; execution pending)

Use one env-gated diagnostic mode at a time: `normal` (full payload), `scalar_sync` (one-element TP all-reduce preserving participant rendezvous), and `skip` (no TP collective, timing-only and semantically invalid). All ranks must select the same mode. Record mode, source diff, backend, scope counts, shapes and hang status. Restrict `skip` to a bounded first-forward or selected layer; never use its output for accuracy or production latency.

### Lane C — synchronization-anchored timeline observer

At `reduce_output()` measure per layer/rank: DP combine return, host pre-AR call, CUDA event at combine return and pre-AR, TP AR event end, and next operation start. Do not synchronize each scope. Use NVTX/CUPTI/Nsight only for a bounded corroboration run if event/host timestamps cannot distinguish queued device work from host scheduling.

### Lane D — rank input and routing audit

Capture local expert token counts, route counts, grouped-GEMM shapes, combine shape, post-MoE input shape and TP/DP rank identity for the same layer/batch. This tests whether recurring late participant submission follows real rank-dependent work instead of nominal uniform-routing labels.

### Measurement decision table

For every matched layer, compute `combine_end`, `ar_submit`, `ar_kernel_start/end`, `next_op_start`, `ar_start - combine_end`, and cross-TP arrival/scope spread. Interpret only with the A/B controls:

1. `normal` large spread + `scalar_sync` large spread: payload is not the primary cause; inspect post-combine arrival, stream queue or host scheduling.
2. `normal` large spread + `scalar_sync` small spread: payload/backend scheduling materially contributes; inspect effective NCCL/custom-AR backend and message bytes.
3. `skip` preserves the pre-AR skew: skew is upstream of the TP AR. `skip` removes the skew and outer span: the AR is on the critical path, but this does not identify why the late participant was late.
4. A late participant's `combine_end`, host pre-AR timestamp, and local route/GEMM shape must be compared directly. Scope inversion alone is not causal evidence.

The RCA closes only when these observations identify one of `DP-pair completion skew`, `host submission delay`, `queued stream/device work`, or `collective payload/backend cost`, with a matched A/B result and explicit unresolved limitations. Until then no Frontier operator correction, TP-rank-specific term, or clean/diagnostic span reconciliation is authorized.

Cross-session continuation: read `handoff.md` first. The last candidate query completed and is verified (54.914898417ms, CUDAgateFAIL); no jobs remain. Review its evidence before another launch. An isolated context-pinned dataset is diagnostic-only; general producer/consumer context semantics require a concrete user-reviewed design. The latest handoff supersedes historical pending-approval and job-status language below.

D019 supersedes the prior implication that all first-batch operators were compared. Current status: first-forward mapping inventory COMPLETE (19 comp rows with explicit missing/fused terms and separate communication); gated activation/reduction source repair and exact-data predictor integration PASS; existing TP4 primitive calibration/holdout and scoped integration PASS; integrated59.190354384ms CUDA gate FAIL; profiling/runtime context RCA INCOMPLETE; first-forward CUDA-event-span gate FAIL; clean TTFT acceptance remains the prior FAIL (19.673373%), not a fresh repaired result. A complete inventory or isolated check is not a completed numerical calibration phase.

YC approved this plan and demonstrated activation/reduction coverage repairs on 2026-09-08. A/B/C implementation and fresh GPU validation are now authorized; CPU integration remains conditional on the CUDA gate. Historical pending-review language below records the submitted proposal. Preserve the existing task layout, checkout and single4096/1024case, H200step_main, image, uniform routing, prefixOFF, collective_sim/htsim with nvlink_analytic and official server TTFT target.

### Outcome and order

First close the first formal forward's non-communication compute/memory operations and validate existing ideal collective costs under explicit shape/rank/timing identities. D020 explicitly defers matching the naive protocol; retain that approximation in the reported residual. Then verify the repaired CUDA-event span with a fresh run. Only after that gate consider how independently measured CPU overhead should enter Frontier workflow and the official-TTFT boundary. The target is the measured forward stream elapsed span; CUDA events can include host-induced device gaps and collective waits, so do not rename it pure kernel time.

`YC_review -> freeze_first_forward_contract -> {A_vLLM_compute_memory, B_Frontier_profiles_predictor, C_existing_collective_costs} -> integrate_scoped_op_repairs -> fresh_first_forward_CUDA_validation -> fresh_clean_E2E_checkpoint -> D_CPU_overhead_workflow_design -> reviewed_CPU_integration -> fresh_full_case_validation`

A/B/C source analysis and CPU work can run in parallel. All A/B/C GPU collection, including B exact-shape profiling, is scheduled by root in the same approved H200 runtime in separate modes, serialized on a shared allocation; they may use separate allocations only when explicitly scheduled with nonoverlapping GPUs. Predictor training and Frontier replay run on CPU master after input validation. D execution depends on the CUDA gate; only D source/component inventory is prepared now.

### First-forward contract

- Anchor to formal request0/4096real prefill tokens, original1024decode workload, actual8H200 identities, TP4/DP2/EP8, eager and uniform routing. This is one logical forward across participants, not whichever local batch ID happens to match.
- Record the participating idle DP lane's1dummy token and resulting4097global MoE tokens, expert assignments, local EP shapes and padding. Do not force a4096-to4096shape match by dropping runtime work.
- Use a bounded first-forward diagnostic trace; extend existing request/batch/rank and forward metadata only where necessary to join dummy and real participants. No full100request event-system redesign is required to investigate this forward.
- Define non-comm device work to include compute kernels, memory-bound kernels, copies, packing/permutation, reduction and memset. Define collective kernels separately; account communication buffer preparation without counting it both as non-comm work and again inside an inclusive communication parent.
- Keep same-generation measurement families separate: CUDA-event spans versus correlated kernel activity. Map parent/child scopes and rank intervals explicitly. Use the audited15Frontier comp labels,32vLLMcomp/mem families and6comm/staging families as the initial checklist. Every device activity must be assigned once or appear in a named uncovered/excluded list; each Frontier alias must be marked instead of double-counted. Inventory completeness with a declared missing counterpart does not pass numerical closure. K kernel activity is coverage evidence and must not be compared numerically with F event-based predictions.

### Parallel lane A — vLLM compute/memory reference and timing coverage

Owner: vLLM/operator measurement agent. Existing seams: diagnostic vLLM op logger and bounded GPU-model-runner selection; tests/e2e/issue26_first_batch_op_rca_{vllm,kernels,communication}.py and the established H200 worker. Coordinate edits to gpu_model_runner/logger through one owner; C requests needed comm hooks from A rather than modifying the same file independently.

1. Inventory every first-forward device kernel/copy/memset and its source scope, including embedding masking/gather, QKV/QK copies and normalization, RoPE, KV save, attention setup/index/output-scale, output projection, residual/norm fusion, router/topk, shuffle, both MoE GEMMs, gated activation, MoE sum and final norm. Explicitly identify operations outside the forward event boundary (logits/sample etc.) rather than mix them into its total.
2. Make the first-forward capture complete and bounded. The already complete first profile is useful for inventory; do not require repairing repeated later profiler startup merely to collect another unrelated batch. Validate CUDA launch/device correlations and expected layer/op counts for every required participant. If the selected first profile is incomplete, investigate capture lifecycle before using any values.
3. Obtain isolated, low-density event measurements for the missing comparable scope groups, including pure output-projection compute separately from its nested TP reduction. Reuse the existing allowlist/logger; do not add per-op synchronizations or subtract values from unrelated runs to manufacture a missing scope.
4. First use existing current-task fresh reference/profile evidence where implementation, identity and context match. Collect batch-only references around new probes and repeat same-case first-forwards only when changed scope/implementation/shape or unresolved measurement spread requires it. Existing exact CSV measurements are S_existing; an independently needed repeat is S_fresh. Do not require a new measurement for every unchanged op. Reuse completed400request control logic unless a reviewed bounded worker is necessary. Do not run dense per-op instrumentation over all100formal requests.
5. Produce full mapping tables and a critical-path accounting check: non-overlapping device activity union, collective intervals, gaps and event-boundary residual. Never sum parallel ranks or sum inclusive parents with their children. A host submission gap already inside the event span stays inside this gate; naming it host-related does not postpone it to the later CPU add-on phase.

Acceptance: every reached first-forward comp/mem op has an explicit counterpart or identified missing model term; all scoped time values have proven identity, positive finite duration and source/measurement-family provenance; pure-comp rows exclude communication; no silent unaccounted device work. Instrumented measurements with material perturbation remain diagnostic and cannot size a production repair on their own.

### Parallel lane B — Frontier profile coverage and prediction error

Owner: predictor/profiling agent. Own the existing profiling/predictor modules selected after the coverage audit, plus a bounded test under tests/; do not edit vLLM instrumentation or collective backend files.

1. Extract the exact predictor query used by each first-forward op: rows/context, token counts, M/N/K, dtype/layout, TP/EP/local experts, routing/padding and hot/standalone profile context. Audit the actual selected rows, not unrelated sparse columns or legacy contexts. The current predictor prefers exact feature matches before RF fallback; several first4096linear/attention predictions already match selected CSV rows. Record which path each query takes before blaming interpolation or retraining.
2. For each material discrepancy compare three quantities independently: P=current model prediction, S=direct Frontier profiling at the exact same implementation/shape, V=matching vLLM operation measurement. P-S identifies model fit/coverage errors only under identical implementation; S-V exposes implementation/shape/measurement differences. Neither a sparse CSV nor a missing4097row alone proves a large prediction error.
3. If exact-shape rows are missing or insufficient, collect only reachable missing shapes and necessary nearby points using the existing H200 profiling wrappers. Preserve declared profile context, merge through the existing mechanism, invalidate only the relevant trained cache, and retrain on CPU master. Do not reuse historical numerical data or add a case-specific timing factor.
4. If source coverage or kernel selection differs, correct that implementation/mapping before collecting replacement profiles and retraining. Do not densify a wrong-kernel dataset and call the problem solved.

Acceptance: material per-op discrepancies have P/S/V evidence or an explicit unresolved counterpart; coverage fixes demonstrate prediction improvement on the actual query and a repeat/holdout at the same reachable shape. Profile sparsity, mapping errors and implementation differences remain distinct causes. The current MoE producer separately prepares sampled expert inputs for shuffle/GEMM and deterministic uniform_topk gating; observed CSV local counts4081/4058/4073 and nonzero CV conflict with treating routing_assignment_policy metadata as a guarantee of uniformly balanced GEMM workload. Correct and verify actual routing_inputs through the existing profiling seam before merely adding nominal-token rows. Publish absolute and signed relative gaps, not only aggregate cancellation.

### Parallel lane C — communication protocol and backend timing

Owner: communication agent. Own collective protocol/backend accounting and focused tests; request diagnostic scope changes through A. Retain collective_sim/htsim and nvlink_analytic.

1. Map actual vLLM naive DP broadcast(hidden/router), DP combine reduction/slice, post-combineTP4reduction, attentionTP4reduction and embedding reduction to Frontier's operation/group/message representation. Explain which idealized Frontier step performs the same mathematical aggregation before adding or replacing a primitive.
2. Measure the actual group, dtype, message size and implementation with lower-density in-context events plus matching collective microbenchmarks where needed. Report rank arrival skew and wait effects; neither maximum nor minimum in-context elapsed time is automatically pure wire cost.
3. Audit the selected analytical algorithm, bytes, steps and NVLink parameters, including the existing50us-per-step allreduce launch overhead. The observed17.899440ms prediction versus5.537248–5.708768ms diagnostic is a reason to investigate, not a universal scaling factor.
4. Under D020, apply only source/evidence-backed timing-parameter corrections through existing backend mechanisms. Protocol changes are deferred optional work. Re-run the same communication shapes and the whole forward.

Acceptance: comm rows are separate from comp/mem; runtime and simulated mathematical operations, groups and byte volumes are matched or their abstraction is explicitly approved. No added post-MoE TP term double-counts an existing combine. Measured wait is not fitted into a link-bandwidth constant.

### Integration and CUDA gate

Root owns integration, review, shared task records and fresh run orchestration. Integrate one verified code sub-step at a time; commit after its direct checks pass. A reviewer checks the full op mapping, no double-counting, and the P/S/V causal conclusion before numerical acceptance.

Proposed first-forward acceptance uses the existing10%calibration target for the aligned forward span, reported alongside repeat spread and individual material op discrepancies. Passing the total through cancellation does not close known material op defects. Re-run with fresh relevant caches and separate batch-only versus clean artifacts. The current65.790077/80.335617ms comparison and19.673373%meanofficialTTFT are the baseline, not a new acceptance result.

The original request prioritizes this first4096prefill forward. Three arbitrary DP-local selected batches do not establish global phase coverage. Full three-batch/mixed/decode operator qualification remains a later check inside this same case, not a reason to divert the current first-forward repair into a broad matrix.

### Later lane D — CPU-overhead reuse and workflow integration

After the CUDA gate, inspect the new clean official-TTFT result and use independently identified outside-forward work to decide CPU integration. Reuse the historical task's verified CPU profiling/CSV/predictor interfaces where they fit current vLLM; refresh H200 measurements and bind rows to actual model/phase/batch identities. A JSON adapter with a fixed different token shape is not a valid4096prefill replay. Source review is recorded in analysis/cpu_overhead_reuse_d019.md. Reuse tests/unit/test_cpu_overhead_vllm_backend.py and tests/integration/test_cpu_overhead_minimal_loop.py at the real future shape; the latter is an injection seam test, not physical timing validation. Historical measure_vllm_prefill_cpu_overhead.py and materialize_vllm_schedule_gap_cpu_overhead_csv.py supply parsing examples only where current hooks actually fit. Current forward annotation excludes surrounding host phases; native step_wall-to-Ray residual mapping does not subtract model CUDA and cannot consume whole GPU-inclusive iteration time as CPU truth.

Separate CPU work outside the CUDA markers from host submission gaps already included inside them. Logits and sampling can execute GPU kernels after the current forward-end event; inventory those separately when closing officialTTFT and never relabel their CUDA work as CPU overhead. Establish schedule, preprocess, sample/output/postprocess and required dispatch portions as appropriate to the existing producer. Deduct only proven overlap by interval identity; do not relabel an unexplained batch gap as measured CPU/Ray transport, copy one timing across shapes, or scale total CPU time to fit TTFT.

Only then design the minimal route/input/full-step handoff needed for the measured portions. The already demonstrated route/enqueue inversion remains a known workflow issue. Keep the existing proposal for later review; do not advance a broad input-drain/dummy lifecycle refactor before completing the operator gate. Preserve official TTFT origin and per-request prefill completion as separate recorded endpoints.

Acceptance: new CPU rows pass producer/shape/phase identity checks and are consumed by the current predictor; exclusive/overlapping intervals are documented; integration changes the correct DES event boundary rather than postprocessing a metric. Fresh full100request4096/1024case validates DP, admission/KVprogress and officialTTFT, with other existing metrics reported independently.

### Scope decisions submitted to YC

1. Approve the sequence A/B/C -> CUDA gate -> CPU/workflow phase; retain present runtime/initial-state settings while collecting comparable first-forward evidence.
2. The previous gated-SiLU repair is explicitly deferred. A complete compute-path correction may require replacing the current missing activation/reduction work and regenerating profiles. This plan proposes reviewing those exact demonstrated coverage repairs together; approval of this plan should explicitly state whether the gated-SiLU deferral is lifted. If it remains deferred, the known coverage error remains open and full operator correction cannot be declared complete by ignoring it.
3. Do not expand to other prefill lengths, H800, a historical test matrix, arbitrary three-local-batch gates, full100op tracing, or a general CPU profiler rewrite. No such expansion is needed to start A/B/C.

## Historical D018 checkpoint — partial diagnostic findings, superseded by D019

D017 implementation and fresh 100-request run are complete: all requests finish with exact token conservation, but numerical calibration remains open (raw TTFT 105.443668418 vs 131.268637180 ms; 78/100 same DP and 5/100 same complete prefill membership). YC requested two RCA lanes and supplemental tests in parallel; both now have source-backed findings and fresh supplemental evidence. See analysis/parallel_rca_summary.md. First-divergence workflow RCA passes bounded controls; operator numerical attribution remains qualified because protocol, shape and instrumentation differences prevent an exact clean gap budget.

recover_frozen_case -> {workflow_source_and_counterfactuals, first_batch_timing_and_op_coverage, minimum_route_instrumentation} -> diagnostic_preflight -> isolated_H200_supplements -> {route_snapshot_enqueue_join, operator_and_kernel_comparison} -> combined_RCA_and_next_decision. CPU source/control analysis and probe preparation run in parallel; GPU modes use disjoint outputs and execute sequentially within one allocation.

Observed full-scope instrumentation inflated the first forward by35.875ms and produced rank-dependent collective waiting. Add an independent communication allowlist run using the existing logger interface, in parallel with kernel tracing, to compare both AR families with lower probe density. This is an authorized supplemental measurement, not a production correction.

Acceptance: reconstruct observed DP choices from actual snapshots; locate the first admission divergence and distinguish proven causes from timing-feedback possibilities. Explain official TTFT, CUDA-event forward span, and kernel time as separate boundaries. Produce per-op signed/absolute/relative gaps for the first qualified formal batch, identify missing terms and global EP/shape limitations, and avoid claiming the full three-batch operator gate without qualifying evidence. New measurements use the same approved H200 configuration. No CPU residual constant or production repair before causal evidence.

## Pre-design checkpoint — preserved evidence

Completed: fresh replay-02 clean400/formal100 and batch400/formal100 validated,8H200 workers,24uniform-router checks; official server TTFT mean131.26863718032837ms. RR and snapshot policy are committed and focused checks pass.

Observed failure: the full predictor-timed Frontier run exits1 with no pending events and unfinished requests. First3requests reproduce a mixed-phase EP synchronization stall at0.29563780122719624s. Same group5 splits into decode step240/lane0 and prefill step241/lane1. Same-phase controls emit a wave; mixed-phase control does not. The minimized2request4096/8control completes;3request4096/8stillfails.

Pending: YC's decision on extending the repair to shared mixed-phase forward synchronization, then scoped implementation/regression, fresh full100request4096/1024CPU run, batch and metric comparison. Proposal: analysis/mixed_phase_forward_stall_rca.md. This changes the shared forward protocol beyond D016 RR/load snapshot; no implementation or temporary substitute is applied. Raw original failure and selected reproductions are preserved. First formal batch composition already matches:DP0request0/4096 on both sides; no complete numerical Frontier result exists.


YC approved both RR continuity repair and a distinct vLLM load-balancing policy with snapshot semantics. Execute RR regression -> scoped fix -> focused validation -> commit; then implement the load-state/snapshot policy and DES integration -> focused source-equivalence and event-flow validation -> commit. Recover H200 job visibility in parallel with local work, validate complete clean/batch artifacts, then run the current case on CPU master. Approval is already provided for the proposed shared interface and config/registry changes. The earlier pending-review statements below are historical.

RR_repair_and_commit -> vllm_snapshot_policy_and_commit -> fresh_CPU_case -> batch_and_metric_analysis.
H200_visibility_recovery -> complete_clean_and_batch_validation -> fresh_CPU_case.

2026-09-08 follow-up: YC requested actual vLLM DP policy research before choosing a repair. Source-method verification proves weighted load routing differs from both persistent RR and current LOR. Prior RR repair remains a valid standalone correctness proposal, not sufficient evidence of vLLM parity. Recommend a distinct registered vLLM internal-load-balancing policy with separate waiting/running counts and explicit snapshot semantics; shared-interface/config changes require YC's policy decision. No production change has been applied.

At 03:40–03:43 UTC, replay artifacts remained at 232 completed warmup requests, last updated 03:31 UTC. Platform status queries failed with Bad Gateway and EOF. Clean400/formal100 validation and batch replay completion are not established. Preserve the job/artifacts; recover platform visibility before considering an execution retry.

YC approved the current4096/1024 historical-control replay. Harness substeps are verified and committed897d2482 (isolated batch mode, dispatch records, GPU orchestration) and91f7147a (batch identity extraction). H200 job yc26-h200-historical-replay-20260908-01 is scheduled on gpu-h200-0844; fixed-image startup is in progress. It executes fresh uniform clean400requests followed by isolated batch/scheduler diagnostic400requests. Operator/routing probes stay disabled in this diagnostic. Current Frontier remains the active issue26 branch.

Observed blocking correctness issue for Frontier comparison: RoundRobinClusterScheduler resets DP selection on each incremental schedule call. Current uniform run's1,839ledger rows are lane0; direct singleton/burst reproduction differs [0,0,0,0,0,0] versus [0,1,0,1,0,1]. D016 proposes a scoped cumulative-counter correction while retaining round-robin policy. Human review is pending. First-request TTFT impact and exact vLLM placement are not established by this bug.

fresh_vllm_clean -> validate_400_rows_and_new_arrivals -> {fresh_vllm_batch_diagnostic, D016_review_and_scoped_fix} -> fresh_CPU_Frontier_replay -> first_batch_and_all_prefill_membership_comparison -> official_TTFT_TPOT_E2E_throughput_analysis.

GPU reference collection and D016 review are independent. Do not run another known-defective Frontier baseline or claim batch equality before resolving the demonstrated correctness issue and examining current diagnostic rows. D013 activation repair remains deferred. Existing baseline103.510401225ms versus127.380511761ms (18.739217017%) is historical within this task, not this new replay's result.

## Current checkpoint — 2026-09-08

- D001–D004, D006–D008 are confirmed and carried forward; D005 is deferred. D009 is now pending explicit YC approval for dispatch-global routing counts; no extension is applied. Historical checkpoint sections below preserve earlier states and are superseded by this checkpoint.
- Completed: fresh clean official server TTFT mean115.982880592ms;82 linear/2026 attention/774 merged MoE rows; both diagnostic modes validated on all8 workers; exact24-cache cleanup.
- In progress: repair the newly observed collective_sim execution failure, then obtain the100-request Frontier baseline and qualify formal operator comparisons. The full128-expert routing vector and MoE operator coverage require further evidence/design review.
- Backend execution repair passed26 checks and is committed69764e50; no numerical calibration gate has passed. Fresh H200 generation03 completed100requests; baseline98.783926557ms vs115.982880592ms, unadjustedrelativegap14.828872974%. D009 shared routing-record extension awaits YC; it is independent of the running baseline preparation.

## Routing alignment checkpoint — supersedes earlier routing status

D010 confirms groundtruth routing-distribution equality as the objective. Capability inspection is complete: no current co-location trace reader; reuse the existing materializer, which consumes static replica/layer expert ratios. D009 dispatch-only proposal is being revised to retain both dispatch and source-DP populations. No routing implementation/configuration has changed.

routing_interface_audit (completed) -> source_and_dispatch_record_design_agreement (pending) -> fresh_routing_capture_and_identity_validation -> batch_population_import_design -> scoped_implementation -> fresh_single_case_validation

The first isolated formal 4096-token prefill is the initial validation anchor inside the existing 4096/1024 case. It is not evidence of whole-run routing equality. Aggregate per-batch counts alone cannot be arbitrarily repartitioned if Frontier batches differ; close that join before extending the claim to all 100 requests. No new delegated work is authorized by this checkpoint.

## Scope and dependencies

context_recovery -> environment_and_semantics -> case_freeze -> {fresh_H200_profiling, fresh_vllm_clean_and_diagnostics} -> fresh_frontier -> TTFT_comparison -> RCA -> scoped_repair -> fresh_validation

Fresh profiling and vLLM evidence collection use isolated H200 allocations. Bounded source reviews and artifact analysis run in parallel with the primary execution lane under the current delegation authorization. Formal GPU execution depends on resolving material settings. Only pf4096_dc1024 is in scope.

## Steps

1. completed: Recover configuration, runtime/image, prior failures, current main contracts, and H200 step_main access; verify NV18/NVSwitch topology and rebuild collective-sim in the actual image.
2. in-progress: D001 collective_sim and D002 nvlink_analytic confirmed. D003 canonical clean endpoint and D004 disabled prefix caching are approved. D006 selects official server TTFT; defer D005. Preserve the endpoint experiment, use official E2E records, and freeze runtime semantics/receipts.
3. in-progress: Fresh H200 profiles, official clean vLLM, and D007 operator/routing worker-identity validation are complete. Frontier generation02 completed fresh training but failed in the first decode EP wave when htsim_runner rejected tensor_bytes. The zero-payload runner boundary is corrected and committed69764e50 after26 passing checks; Fresh H200 generation03 and100-request artifactvalidation are complete. Routing/operator comparability remains pending.
4. in-progress:100formalrequests joined and unadjustedbaseline compared(14.828872974%gap). Diagnoseworkflow/operator/routingcauses and independently accountmissingcritical-pathwork underD006. D009globalroutingcountsdecisionremains pending.
5. pending: Apply the justified scoped correction, commit verified code sub-steps, rerun this case, and archive the result.

## Acceptance

Current main base; H200 step_main only; no historical measurements or trained caches; complete per-request provenance; no warmup contamination; clean metrics separated from diagnostics; observed TTFT mean and <=10% error gate; evidence-backed repair and fresh verification. Preserve unrelated work and README files. Broader code changes and unresolved key choices require the applicable approval gate.


## D011 active checkpoint — supersedes trace-import plan

confirmed_uniform_knob -> {fresh_uniform_vllm_clean, fresh_uniform_moe_profiles, CPU_runtime_and_runner_preparation} -> CPU_Frontier_with_matching_runtime_profiles_and_fresh_queue_trace -> request_comparison -> scoped_RCA

The three sibling preparations run concurrently. Final paired Frontier run depends on new clean queue arrivals as well as complete profile inputs; using an old run arrival trace would require a separate diagnostic label and cannot close this pair. D012 shared runtime-selection extension is pending; trace importer and D009 counts are deferred feature work.


## Uniform rerun checkpoint

Completed: fresh uniform vLLM clean400requests/100formal, servermean127.380511761ms; uniform MoE774rows; fresh queue-arrival trace; CPU fullruntime/backend/runner. Pending: D012 design agreement and implementation, isolated selector/sharedmodel verification, CPU numerical run and fresh TTFT comparison. Candidate config is prepared at config/frontier_uniform_candidate.json with the proposed not-yet-supported field and must not be launched before D012. No further GPU allocation is needed for the prepared Frontier rerun.

## D012 confirmed execution checkpoint

YC approved explicit runtime selection and shared model identities. Implementation and 74 focused checks are complete. Prior pending-D012 statements are superseded. verified_code -> commit -> fresh_CPU_uniform_run -> 100_request_validation -> official_TTFT_comparison -> scoped_RCA. Current candidate config is supported and approved. D009/D010 trace import remains deferred by D011.


## D020 optional communication contract (deferred by YC)

Source/math RCA and independent review: analysis/d019-communication.md and analysis/d019-communication-review.md. A/B measurement and analysis continue under D019. YC has explicitly deferred the new shared communication-runtime/configuration implementation; it is optional future work.

Historical proposal, not active implementation scope: explicit runtime selector preserving ideal default, this case vllm_naive; physical DP populations in existing EPWaveInputs drive both routing and communication while dummy stays outside Request/KV progress; use existing five phases in actual local-router -> broadcast -> global-topk/shuffle/expert -> DP-combine -> TP order; native same-node broadcast under nvlink_analytic uses its appropriate analytical capability without invoking an unsupported htsim broadcast, while existing collectives retain their current backend path. Multi-node native broadcast is explicitly unsupported, not silently approximated. No general traced-routing import or new scheduling event hierarchy.

The earlier shared schema/descriptor gate is resolved by YC’s deferral. Do not implement this proposal or substitute P2P/allgather in the active plan. Reopening it requires an explicit future decision and its own physical/mixed/request invariant tests.


### D020 current execution route

YC explicitly retains ideal communication and defers the complete vllm_naive supplement. The prior pending-design gate is resolved by deferral; it no longer blocks D019. DAG: {fresh V groups, corrected MoE data/P integration, existing-collective measurement audit} -> complete qualified comp/mem and comm mapping -> scoped evidence-backed corrections within ideal -> fresh first-forward CUDA result -> fresh clean E2E -> conditional CPU/workflow. Protocol discrepancy remains a declared approximation; do not claim exact protocol parity or silently compensate it with a fitted constant. All other frozen settings and official TTFT boundary remain unchanged.

## Active checkpoint — H800 ten warmups, 2026-09-10

completed 10-warmup run -> completed bounded identity/drain and formal span statistics -> pending qualified standard normal baseline -> pending normal/skip paired runs -> secondary scalar diagnosis -> CUDA completion attribution. Detailed result: analysis/h800-ar-176000-normal-10warm-01/report.md. Ten warmups do not restore the normal scale. Read-only harness/source comparison and evidence reporting may run in parallel; GPU scheduling remains centralized.

## Active checkpoint — standard H800 reproduction complete, 2026-09-10

completed standard H800 replay -> completed400-row/identity/drain gates -> completed first-formal comparison (77.483665ms vs80.654881ms median) -> pending normal/skip paired causal analysis -> secondary scalar/completion attribution -> CUDA closure. The current reproduction request is complete. GPU tests remain centralized; no additional run launched during validation.

## Current execution — approved minimal bypass, 2026-09-10

Verified normal reproduction -> {single-call source preparation, independent read-only source audit (parallel)} -> preflight -> one centrally scheduled H800 standard replay -> 400-row/drain/identity validation -> DP0 TP0-3 first-formal statistics -> paired normal/skip repeats -> CUDA completion attribution. This supersedes the older bounded skip and boundary-logging execution recipe. General profiling and CPU integration remain downstream.

## Minimal bypass checkpoint — 2026-09-10 complete

Source preparation, runtime asset restoration, standard clean/batch replay and numerical identity/drain validation completed. Bypass median77.935920715ms vs normal77.483665466ms (+0.583678%). Next unresolved causal work: same-node repeated normal/skip pairing -> completion-based collective/queue attribution -> conditional clean/diagnostic reconciliation. Scalar remains secondary; general profiling and CPU corrections remain downstream.
