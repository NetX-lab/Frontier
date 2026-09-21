## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded the source-proven diagnostic identity gap and a scoped proposal. |

# D007 proposal: identifiable diagnostic records

Status: approved by YC (D007). Clean measurement and fresh profiling continue independently.

Observed in selected clean vLLM commit 46f7b179f:
- GPUModelRunner creates a process-local batch counter starting at zero. Its batch log includes request IDs but no DP/TP/PP identity.
- FrontierCudaEventOpLogger writes batch_id and pp_rank only, without request IDs or DP/TP identity. All eight workers receive the same log pathname.
- FrontierMoeRoutingLogger includes ep_rank and layer_name, but cannot be unambiguously joined to the rankless batch/op records using batch_id alone.
- No rank-specific filename mechanism was found in the worker, executor, logger, or recovered runner.

Proposal: extend the existing diagnostic loggers in vllm/v1/utils.py and their existing GPUModelRunner integration. Record explicit DP/TP/PP identity, isolate files per worker identity, and join local batch IDs through the batch request-ID list. Use existing group/config identities rather than parsing worker names or timestamps. Preserve official request metrics and inference behavior. Validate all eight identities and unique first-formal-batch joins on H200 before consuming operator timing.

Eliminated options: pooled batch_id-only logs lose rank ownership; timestamp proximity is not a unique identity; selecting one rank cannot explain an eight-rank critical path; a new tracing framework duplicates existing loggers.

Additional existing-path choices: operator timing and routing count capture must run separately. Routing capture copies tensors to CPU from inside the moe_gating timing scope, which contaminates that op duration. Per-scope timing without runtime_meta is supported; enabling runtime_meta with Qwen3's before-scope metadata emission fails the current logger contract. Derive initial shape identity from batch counts and model configuration, and mark unavailable shapes as missing rather than inventing metadata.

Module boundary: utils.py is 1258 lines; GPUModelRunner is 4192 lines in an external vLLM checkout. The proposed integration only supplies identity to existing tracing hooks. A wholesale model-runner cleanup would modify the reference runtime beyond this measurement task. Keep new logging mechanics in the existing utility module; defer unrelated model-runner extraction.

Acceptance: clean logging remains official; eight unique DP/TP worker files; batch/op joins unique within each worker; no phase/rank inferred from timestamps; routing capture does not contribute to operator or E2E timing evidence.
