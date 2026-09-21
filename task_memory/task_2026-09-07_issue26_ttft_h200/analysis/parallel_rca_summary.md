## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | D019 review clarifies that full operator coverage, numerical correction and forward-span repair remain incomplete. |
| 2026-09-08 | Integrated parallel workflow counterfactuals, native metric boundaries, and first-forward operator diagnostics. |

# D018 parallel RCA

D019 correction: the completed items below are bounded diagnostic findings and collection checks. Complete comp/mem/comm correspondence, numerical operator correction and CUDA-event-span repair are INCOMPLETE. The revised CUDA-first plan is in ../plan.md; it awaits YC review before execution.

Status: workflow first-divergence analysis complete; operator semantics, instrumentation perturbation and24selected communication rank-batches analyzed. Communication serving completes400requests and passes the independent100formal/8worker batch identity check. This report does not accept numerical calibration or claim full 100-request closed-loop parity.

## Frozen scenario and evidence generations

Only the approved 4096-prefill/1024-decode case is used: Qwen3-30B-A3B, BF16 dummy weights, TP4/DP2/PP1/EP8, eight H200 charged to step_main, uniform MoE routing, prefix caching off, chunked prefill off, eager FLASHINFER, QPS2, client seed20260908, server seed0, three drained 100-request warmups and100formal requests. Frontier retains collective_sim/htsim and intra-node nvlink_analytic, with CPU overhead prediction disabled. Current-task operator profiles are freshly collected; historical versions are not numerical inputs.

- Clean E2E: runs/h200-historical-replay-02/clean/runtime/clean, vLLM46f7b179. This is the official server metric reference.
- Complete numerical Frontier: production8ba22b49; analysis/cpu-shared-forward-01 and analysis/shared-forward-numerical-01. Later D018 changes affect diagnostics only.
- Fresh route/batch and operator diagnostics: analysis/h200-rca-01/runtime, diagnostic vLLM8453dd342, gpu-h200-0844. Batch/route and CUDA-event modes each complete400requests. Record-function mode fails on its third selected DP0 profile; the complete first profile remains diagnostic evidence.
- Communication supplement: analysis/h200-rca-comm-01/runtime/operators, same diagnostic commit/image, gpu-h200-0761. Only the existing three AR scopes are enabled;400requests complete,24selected rank-batches and2328scope rows pass integrity checks. No production implementation is changed.

Every diagnostic mode remains separate from clean E2E. Different runs cannot be subtracted to measure CPU overhead.

## 1. DP ownership and admission

The first demonstrated DP swap has a workflow boundary cause that does not require accumulated operator error. vLLM chooses an engine in the frontend before sending ADD; QUEUED is emitted later when EngineCore adds the request to its scheduler. Frontier currently treats that downstream QUEUED timestamp as an external arrival and makes another routing decision there. EngineCore can defer draining input until its current step ends, so QUEUED itself depends on execution progress.

Same-run route/snapshot/enqueue records join400/400requests, including100formal requests with exact per-TP prefill membership. Feeding each actual vLLM snapshot into Frontier's selector reproduces400/400choices;399snapshot/local-increment transitions are reconstructed after the initial observed state. This verifies the selection rule under supplied state, not autonomous simulator state equivalence.

The first close pair establishes the ordering mechanism directly:

| Boundary | client5 | client6 |
| --- | --- | --- |
| Frontend routing | First; counts[[0,3],[0,2]], DP1 | 2.023212ms later; counts[[0,3],[1,2]], DP0 |
| Engine QUEUED | Second | 0.048529ms earlier than client5 |

Two actual DES controls preserve all predictor inputs and change only the observed arrival boundary. Enqueue-input matches6/8DPowners and5/7prefill member/token multisets; route-input matches8/8 and7/7. Counts at the disputed pair are identical between controls. This isolates the first swap from a necessary operator-duration correction.

Admission remains independently misaligned. In the batch containing request1's prefill, request0 has4096prior scheduled tokens in vLLM,4097in the enqueue-input simulation, and4098in the route-input simulation. These are cumulative scheduled model tokens, not measured KV occupancy or a generated-token counter. Matching request labels is insufficient; replacing the sole arrived_at timestamp with route time is not a complete repair.

Source and same-run timing explain why the second boundary matters: Frontier schedules again at its modeled batch-end event; vLLM continues through logits, sampling, bookkeeping, scheduler/output/load updates, optional DP synchronization and input draining. The fresh first schedule-to-next-schedule interval is83.452619ms, while the CUDA forward span is80.335617ms. Their3.117002ms difference includes work outside the measured forward and is not exclusively CPU time.

The first formal request also sees stale frontend counts despite drained warmup clients. Fresh records show no warmup request scheduled after the first formal route; this excludes unfinished warmup requests in those schedules. They do not directly establish whether a dummy GPU round was in flight at that instant. The core's distributed idle/finish protocol needs its own execution identity to settle that remaining question.

Full100request feedback decomposition remains open. Later execution-time errors can alter load publication, input visibility and admission; the prefix control does not attribute every later mismatch. See dp-workflow-rca/rca.md, fresh_boundary_controls.json and independent_review.md for evidence and limits.

## 2. First-forward and TTFT boundaries

Official server TTFT uses the recorded process_inputs arrival timestamp through frontend receipt of the first engine output. It includes relevant host processing, transport, waiting, scheduling, preparation, forward, sampling and output handling within those endpoints. It is not a CUDA-only metric.

The instrumented batch CUDA events surround self.model after preprocessing and before logits/sampling, including forward-context setup and its DP metadata coordination. Their elapsed span can include GPU idle caused by host submission and collective waiting between markers. A correlated kernel trace removes idle intervals but can still include collective spin/wait execution. Frontier's operator profiling inputs also use CUDA events; disabling its CPU overhead predictor does not imply all predicted durations are pure kernel activity.

The clean run permits an independent native endpoint reconstruction without cross-run subtraction:

| Clean metric | First request (ms) | Mean100requests (ms) |
| --- | ---: | ---: |
| Official server TTFT | 121.484280 | 131.268637 |
| First scheduled to engine first-token endpoint | 81.157911 | 88.392300 |
| Remaining portions of the official interval | 40.326369 | 42.876338 |

The last row combines arrival-to-schedule and engine-to-frontend portions. It is not a measured CPU constant. The separate fresh route run observes39.147422ms from first route to enqueue, but that different run cannot numerically decompose the clean40.326369ms.

First Frontier forward prediction is65.790077ms; fresh batch-only vLLM CUDA span is80.335617ms. The absolute difference is14.545540ms, or18.105967% relative to vLLM. No earlier formal request's operator-error accumulation is required for this discrepancy. Warmup/runtime state and operation/communication contracts remain relevant.

## Per-operator evidence and communication contract

The detailed signed/absolute/relative comparisons are in first-batch-op-rca/operators/op_gap_table.csv. Scope normalization retains48layers, removes nested EP dispatch/combine double-counting, splits router projection from top-k selection, includes Q/K normalization with input projection, and compares attention output projection together with its nested TP reduction.

Source inspection proves a communication implementation mismatch: vLLM naive dispatch broadcasts hidden states and router logits across DP peers; combine performs DP all-reduce and slices the local range; reduce_results then executes an additional TP4all-reduce. Its misleading scope name is expert_parallel_allreduce. Frontier predicts ideal EP8dispatch/combine and records zero post-combine cost for this wave. The absent explicit primitive and differing collective protocol are established; their clean latency contribution requires qualified timing. Frontier's idealized combine may implement part of the same mathematical aggregation through another protocol, so simply adding a TP term without mapping that protocol could double-count.

The first real lane processes4096tokens while the idle DP lane executes1dummy token. In the current eager source, no max-DP padding is selected, so vLLM's global MoE input is4097tokens and32776top-k assignments; Frontier models4096and32768. The extra assignments can cross a grouped-GEMM padding boundary, so their possible cost is not bounded by1/4096. Current rank timing does not demonstrate that EP0's extra padded work is the dominant gap. Across the four complete firstbatch TP traces, gated SiLU kernels total7.062557–7.070356ms, moe_sum kernels2.709863–2.723883ms and the two grouped GEMMs10.143596–10.270075ms. These compute families are substantially more stable across ranks than collective elapsed time. The current Frontier profiling loop omits real gated SiLU and moe_sum computation; the measured kernel totals are diagnostic evidence, not additive clean-TTFT corrections. Gated-SiLU production correction remains deferred by YC.

Full CUDA-event instrumentation increases the first measured forward from batch-only80.335617ms to116.210144ms across separate diagnostic runs. The instrumented extra TP all-reduce totals37.563ms on TP0 but5.653ms on TP2, indicating rank-dependent waiting. It cannot be used as a37.563ms production correction.

The complete first kernel profile supplies direct mechanism evidence: every DP0TP rank has3077launches with corresponding device activity. TP1 has94.431ms cumulative GPU idle, including74.126ms of accumulated gaps where the next kernel launch has not started (largest single idle gap0.382ms, not one74ms stall); TP0/TP2 spend65.824/66.155ms in the extra TP reduction, compared with4.619ms on TP1. Host submission delay on one rank can therefore appear as long collective GPU activity on peers under profiling. This evidence explains why kernel-active time alone is insufficient to infer wire cost.

## Lower-density communication measurement

The97-scope firstbatch communication supplement retains48attentionTP reductions,48extraMoETP reductions and1embedding reduction. All24selected rank-batches preserve the expected scope counts. The first fourTP forward spans are86.412033–86.443520ms; TP0 is6.076416ms (7.56%) above the separate batch-only reference, substantially less run/probe sensitivity than the full-op measurement but still not clean execution.

| Firstbatch aggregate | Frontier prediction (ms) | vLLM TP0/1/2/3 CUDA-event scopes (ms) |
| --- | ---: | --- |
| Attention TP all-reduce,48layers | 17.899440 | 5.537248 /5.649952 /5.708768 /5.644864 |
| Extra post-combine MoE TP all-reduce,48layers | No explicit term | 4.792544 /25.384416 /28.413280 /27.393568 |
| Embedding TP all-reduce,once | No explicit term | 0.218336 /0.301152 /0.275616 /0.296320 |

Attention TP cost is overpredicted relative to every measured rank in this diagnostic, while the naive MoE protocol contains runtime stages not explicitly represented by Frontier. The minimum observed MoE TP elapsed time is not a clean wire-time estimate: it may still include wait and host submission gaps. No ranks are summed, no cross-run operator totals are assembled into a fitted clean critical path, and no28msmissing-cost claim is made.

These opposing discrepancies, compute-family omissions and measurement sensitivity prevent an exact additive decomposition of the14.545540ms first-forward difference. The RCA establishes specific workflow/protocol/coverage causes and bounds the available timing evidence; a production numerical correction still requires aligned protocol accounting and a fresh clean rerun.

## Failures and limits

- Record-function collection fails on selected DP0batch4736: all four TP ranks lack the first5device correlations, including embedding all-reduce, in the original Chrome trace. The collector's zero-duration rejection is correct. Batch4735 also has two missing correlations on TP1/TP2; only the all-rank-complete firstbatch4734is used. CUPTI startup/buffer root cause is not established.
- The bounded Frontier first-op probe was interrupted after complete first-wave evidence because the existing sequential time-limit guard is unreachable. Its exit130is not a completed E2E run. This unrelated simulator defect is recorded for later work.
- Runtime metadata stays disabled because existing model calls emit metadata outside the required active scope. Actual shapes are established from source and available traces; missing runtime metadata is not claimed present.
- Neither the firstbatch evidence nor the per-DP first-three selection establishes the full three-aligned-global-forward operator gate.

## Remaining numerical acceptance and next design boundary

The latest complete clean comparison remains Frontier105.443668ms versus vLLM131.268637ms meanTTFT: absolute error25.824969ms, relative error19.673373%, exceeding the10%target. D018 does not alter that acceptance result.

A source-grounded next workflow design must distinguish frontend routing, engine input visibility and full-step completion rather than overload one arrival timestamp. A communication repair must express the actual supported primitive sequence through the selected backend rather than add a fitted residual. No CPU residual constant, idealized replacement backend or speculative production patch is applied in this RCA.
