## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded the inspected endpoint mismatch and pending D003 proposal. |

# D003: Clean Canonical TTFT Producer

Status: D003 approved by YC. Scoped implementation and H200 validation are authorized; no validation success is implied.

Observed source boundaries:

- Frontier AGENTS.md canonical contract: queue-visible arrival -> prefill completion. The documented historical batch-log reconstruction uses the vLLM forward completion boundary.
- /data/ycfeng/tmp/vLLM-BS/vllm/v1/engine/output_processor.py::_build_frontier_metrics exports stats.first_token_latency, measured at frontend receipt. It cannot establish the requested earlier endpoint.
- vllm/v1/worker/gpu_model_runner.py records its existing batch timestamp after model forward and an instrumentation-induced torch.cuda.synchronize(). It precedes compute_logits, sampling and bookkeeping. That logger is not clean unperturbed timing.
- vllm/v1/metrics/stats.py maintains monotonic queued_ts, scheduled_ts and first_token_ts. EngineCoreOutputs.timestamp is generated after scheduler output processing, so first_token_ts is later than the forward endpoint too.
- GPUModelRunner._bookkeeping_sync already synchronizes the sampled token copy for the non-async, non-speculative route selected by this case. This provides a candidate natural completion point for reading earlier CUDA timing events without inserting a new per-batch global synchronize.

Recommendation presented to YC: retain the original canonical endpoint and extend the existing E2E recorder minimally to export queue-visible arrival and prefill forward completion. Record only required endpoints; keep per-op and CPU probes off. Verify GPU/host clock alignment, formal request identity, and probe overhead on the exact H200 eager route before admitting results. The event-to-host-clock mapping and extension's exact file-level design remain implementation work after agreement, not verified behavior.

Rejected substitutions: frontend first-token latency and EngineCore first-token timestamps include later work; silently substituting either changes the metric contract. Existing instrumented batch logs remain useful diagnostics but do not establish an unperturbed clean metric.

Question sent through the interactive surface: preserve the original endpoint and add validated minimal E2E recording (recommended), or hold the measurement extension for YC to specify another boundary. The extension affects measurement data contracts and is gated by workspace Approval Gate items 4 and 5. Backend/topology verification proceeded independently and passed.
