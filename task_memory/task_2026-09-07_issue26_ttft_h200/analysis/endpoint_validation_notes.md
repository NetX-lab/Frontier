## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded endpoint validation coverage and the remaining long-run clock check. |

# Endpoint validation scope

The eight-H200 standalone probe validates 512 records, anchor intervals, independent short-run clock brackets, and per-call overhead. The exact 4096/1024 A/B run validates real request state, natural output synchronization, TP4 request identity, warmup barriers, and observed probe-on/off request latency.

The initial anchor width is not a bound on subsequent GPU/host clock drift. A final independent idle clock bracket must therefore be captured after the complete workload and compared with the original CUDA elapsed-time mapping before admitting canonical TTFT as clean evidence. Do not silently relabel the initial uncertainty as a whole-run bound.

Inspected shutdown seam: vLLM `WorkerProc.worker_main` always invokes `worker.shutdown()` in its finally clause after the busy loop; `gpu_worker.Worker.shutdown` calls the model-runner shutdown path. A final recorder clock check can run there outside the measured request stream, and its retained record can be required by the analyzer. This is a bounded continuation of approved D003, not a TTFT boundary change. Apply only after the currently executing A/B run completes so both sides retain one unchanged source patch.

The completion of a TP4 request is the maximum forward-end completion among that request's four TP ranks. Queue arrival is the engine's monotonic QUEUED event. The normalized join uses the exact client response ID followed by the single-prompt engine suffix `-0`, verified in serving_completion.py. A group must be exactly ranks 0–3 or ranks 4–7, with one record per rank.

A single sequential A/B pair reports observed variation, not a causal bound on instrumentation overhead. The short GPU probe and source-level absence of a new per-batch synchronization supply additional evidence; investigate material observed A/B variation before claiming low overhead.
