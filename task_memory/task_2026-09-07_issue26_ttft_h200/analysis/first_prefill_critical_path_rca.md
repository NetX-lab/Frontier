## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-11 | Consolidated existing H200 first-prefill critical-path evidence while paired H200 normal/skip run is queued. |

# First-prefill CUDA batch span: evidence-based composition

The existing clean H200 diagnostic batch is batch 4250, DP0, 4096 prefill and zero decode. Outer spans are 86.412033, 86.437950, 86.414658 and 86.443520 ms for TP0--TP3. The four-rank spread is 0.031487 ms, so the ~20 ms Frontier gap cannot be explained by TP-rank completion skew in this boundary.

The same batch's inclusive communication scopes are:

| TP | TP allreduce | attention post-proj TP AR (48) | post-MoE EP/TP AR (48) | Sum of logged comm scopes | Outer span | Outer minus comm sum |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.218336 | 5.537248 | 4.792544 | 10.548128 | 86.412033 | 75.863905 |
| 1 | 0.301152 | 5.649952 | 25.384416 | 31.335520 | 86.437950 | 55.102430 |
| 2 | 0.275616 | 5.708768 | 28.413280 | 34.397664 | 86.414658 | 52.016994 |
| 3 | 0.296320 | 5.644864 | 27.393568 | 33.334752 | 86.443520 | 53.108768 |

The communication values are inclusive rank-local CUDA event scopes. They are not additive across TP ranks and do not equal critical-path duration. The outer span is a cross-rank completion boundary. A short TP0 post-MoE scope can include a late enqueue relative to peers; scope duration alone does not measure its wait contribution.

The older kernel-correlated H200 trace provides the missing structural evidence. For one first-formal batch, device envelope was approximately149--153 ms while rank-local device active union ranged54.768--143.430 ms; within-envelope idle was2.98--94.43 ms and `next_launch_not_started_ms` was2.98--74.13 ms. The trace explicitly classifies delayed next launch as absent command submission, with synchronization/thread scheduling/waits still possible. This proves that sums of observed device activities omit queue gaps. It does not identify which host call or synchronization caused each gap.

The operator table also shows a large attention post-proj and post-MoE rank spread in a different instrumented run, while the clean outer spans stayed nearly equal. Therefore the previous hypothesis “one 20 ms TP AR event directly adds 20 ms to the batch” is unsupported. The observable critical path is: ordered layer work -> DP combine completion -> rank-local host enqueue -> TP collective participation -> next layer launch -> final completion. Current logs capture operator scopes and outer completion, but lack paired host-call timestamps and collective completion event IDs. That is the specific missing measurement needed for causal closure.

## Current RCA boundary

Established:

1. Frontier's modeled 59.190354 ms is 18.928--20.117 ms below clean vLLM batch-only references 78.119--79.307 ms in the integrated H200 calibration. This is a real outer-span gap under the retained ideal Frontier communication abstraction.
2. A TP post-MoE scope can differ by ~20 ms while outer spans differ by <0.04 ms; rank-local scope sums therefore cannot be summed or mapped directly to outer critical-path time.
3. Device-active work has large inter-launch idle gaps; the gap is partly scheduling/queue/wait behavior or uninstrumented work, not necessarily missing FLOPs.
4. Existing operator measurements are from different diagnostic runs and include instrumentation perturbation; they cannot supply an additive 20 ms decomposition.

Unresolved and required next measurement: same-run, same-request host/cuda timeline around every 48-layer `combine_return -> tp_ar_call -> tp_ar_completion -> next_layer_start`, including both DP lanes. The measurement must be capture-only, with normal and minimal skip on the same H200 allocation, and report event IDs and host monotonic timestamps. It must not be used as a production accounting correction until paired outer span and completion evidence closes.
