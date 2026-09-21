## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Established pinned runtime timing boundaries, corrected nested scope ownership, and qualified fresh event/kernel measurements. |

# First-batch timing contract

Observed source facts (diagnostic vLLM baseline 361d941c97fcec52e544f74b7ab91c54192de9c9; source-compatible bounded selection 1404bd8d4):

- `vllm/v1/worker/gpu_model_runner.py` records CUDA start after `_preprocess`, immediately before forward contexts and `self.model`; it records CUDA end immediately after `self.model` and before postprocess/logits/sample. `batch_execution_time_ms` is `start.elapsed_time(end)` after synchronization. It is a forward stream elapsed interval, not the sum of CUDA kernel active durations.
- CPU preparation before the start event and postprocess after the end event are outside this interval. The start precedes `set_forward_context`, including CPU-side DP metadata synchronization, so the interval is broader than only `self.model()`. Host launch delays that leave the GPU stream idle between the two events, CUDA waits and collective waits may contribute to its elapsed interval. The CPU time of a region is not automatically additive to CUDA time because CPU launch work can overlap device execution.
- `FrontierCudaEventOpLogger.scope()` in `vllm/v1/utils.py` uses the same CUDA-event mechanism in `cuda_event/default` mode. The alternative `kernel_only` mode synchronizes before every scope and after collectives, substantially perturbing execution; it is not selected for this RCA.
- `record_function` mode profiles CPU/CUDA activities, exports a per-batch Chrome trace, and reports correlated CUDA kernel durations. It introduces profiler overhead, so its whole-batch duration is a diagnostic measurement and cannot replace clean E2E or the batch-only baseline.

Official server TTFT (clean checkout `/data/ycfeng/tmp/vLLM-BS`):

- `vllm/v1/engine/processor.py:342–345` sets `arrival_time=time.time()` inside `process_inputs`, after parameter validation and before subsequent input processing, when the caller has not supplied a timestamp. This is not the HTTP socket arrival or engine queue timestamp.
- `vllm/v1/engine/async_llm.py:442–448` receives `EngineCoreOutputs` and constructs `IterationStats`; `vllm/v1/metrics/stats.py:114` captures `iteration_timestamp=time.time()`.
- `stats.py:139–143` computes `first_token_latency=iteration_timestamp-arrival_time` on the first output. `output_processor.py:271` exports this as `ttft_ms`.
- Therefore official TTFT includes host-side processing and communication after its arrival boundary, engine queue/scheduling time, input preparation, GPU forward, sampling/output handling up to output receipt. It excludes client networking and host work before the recorded arrival boundary. It does not measure pure CUDA time or merely prefill completion.

Known comparability risks to preserve:

- The prior 77.1250228881836 ms batch-only CUDA event and 121.48427963256836 ms clean official TTFT came from different runs. Their difference is not a measured CPU overhead.
- First formal request has no preceding formal op-error accumulation, so later request DP assignment divergence cannot explain its first-forward duration error. Warmup may affect device/runtime state, and CPU launch/collective/op modeling still require measurements.
- Qwen3 `moe_gating` is used twice per layer: one scope wraps router linear projection (`models/qwen3_moe.py:184`), the other wraps selection (`layers/fused_moe/layer.py:1477`). Scope sequence is not identical to layer index for this operator. Frontier retains separate linear and routing predictions.
- `moe_grouped_gemm` includes W1, gated activation and W2; `moe_sum` is outside this parent scope. `input_layernorm` and `post_attention_layernorm` may contain nested `add`; summing parent and child would double-count.
- Existing runtime metadata logging is incompatible with some model call sites: Qwen3 records layernorm meta before entering its scope, while the per-scope logger requires an active matching scope. Keep `VLLM_FRONTIER_RUNTIME_META_ENABLED=0`; do not silently claim metadata coverage.

Bounded supplement: each DP worker's first three real formal forwards, TP peers select from identical local request IDs. Other forwards retain batch logging but have no op logger. Dummy DP forwards are not profiled by the existing path. The selection changes no model/collective calls. Different DP local batch counters do not prove shared global-forward identity; source request membership and token/KV state must establish each comparison.

Additional scope checks, corrected against the fresh runtime rows: `qwen3_moe.py:382` wraps row-parallel `o_proj` in `attn_post_proj`, but `linear.py:1306` names its child scope `attn_post_proj_tp_allreduce`, which is absent from the default scope list. The 48 attention TP reductions therefore have no separate CUDA-event rows. Compare the observed parent against Frontier's projection **plus** attention TP prediction, and use kernel traces for finer separation. `vocab_parallel_embedding.py:451` invokes the sole `tensor_parallel_allreduce` row (scope sequence 0). The first analyzer expected 49 generic AR rows and failed its count assertion; inspecting the runtime and scope-name resolver corrected that assumption without changing vLLM. Q/K RMS normalization is already inside `attn_pre_proj` (`qwen3_moe.py:306–319`).

EP dispatch and combine each use two nested scopes with identical names: the MoE layer outer scope and `parallel_state.py:846/859` inner scope. Each has 96 rows over 48 layers. The outer scope receives even sequence numbers, the inner receives odd numbers and finishes first. Keep raw rows, aggregate only the outer 48. Counting all 96 would double-count the communication.

The naive MoE collective protocol differs from ideal EP all-to-all. `all2all.py:28–55` broadcasts local hidden states and router logits across DP peers; `combine():57–66` performs DP all-reduce and slices the local range. `fused_moe/layer.py:1852–1861` then applies an additional TP all-reduce because `reduce_results=True` and EP>1. This extra runtime scope is named `expert_parallel_allreduce`, despite using the TP group. Frontier's first-wave model reports dispatch/combine EP phases and zero post-combine cost. Source semantics alone do not quantify the production time of this difference.

The Frontier profile datasets also use CUDA events (`frontier/profiling/common/cuda_timer.py:54/93`), so operator predictions can already contain host-induced GPU gaps. Disabling Frontier's CPU overhead predictor does not make its entire predicted latency a pure CUDA-kernel sum. Adding every observed CPU interval would risk double-counting.
