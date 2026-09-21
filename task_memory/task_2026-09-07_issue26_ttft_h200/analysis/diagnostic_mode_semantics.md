## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-10 | Audited the diagnostic normal/scalar_sync/skip implementation and recorded its evidence and limits. |

# Diagnostic post-MoE TP all-reduce modes

Source audited: `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab/vllm/model_executor/layers/fused_moe/layer.py`, `maybe_all_reduce_tensor_model_parallel()` and its `reduce_output()` caller.

## Exact behavior

- `normal`: calls `tensor_model_parallel_all_reduce(final_hidden_states, record_scope_name=op_name)`. With `ep_size > 1`, `op_name` is `expert_parallel_allreduce`; otherwise it is `moe_tensor_parallel_allreduce`. The full `final_hidden_states` payload is submitted to the TP process group and the reduced result is returned. This is the only mode preserving normal post-MoE numerical semantics.
- `scalar_sync`: allocates `final_hidden_states.new_zeros(1)`, all-reduces that one-element tensor over the same TP group (scope suffix `_diag_scalar`), discards the scalar result, then returns `final_hidden_states.clone()`. It preserves a TP collective participant/rendezvous and its framework/runtime path while removing the full hidden-state payload. The returned value is local, unreduced data, so it is a timing diagnostic only.
- `skip`: performs no TP all-reduce and returns `final_hidden_states.clone()`. It removes the post-MoE TP collective entirely. The clone still introduces a local device operation and preserves tensor shape, but no cross-rank synchronization remains. Its output is also local/unreduced and therefore cannot be used for accuracy or clean latency calibration.

The mode is read from `VLLM_FRONTIER_DIAG_MOE_AR_MODE`; invalid values raise `ValueError` immediately. If `use_pplx_kernels`, `use_deepep_ht_kernels`, or `use_deepep_ll_kernels` is active, `maybe_all_reduce_tensor_model_parallel()` returns the input before this mode switch because those combine kernels already reduce outputs; these diagnostic modes therefore do not instrument that path. The audited runs set `VLLM_ALL2ALL_BACKEND=naive`, so the mode switch was reached.

## Boundary marker meaning

`combine_return` is a host timestamp immediately after `get_ep_group().combine(states)` returns. `tp_ar_call` is immediately before entering the helper; `tp_ar_return` is immediately after it returns. The logger records process-local `time.perf_counter_ns()` plus rank, DP/TP/EP rank, layer, shape, dtype, and mode. No CUDA event or request_id is recorded.

Thus `combine_return -> tp_ar_call` measures host-visible elapsed time after Python `combine()` returns; it is not proof that the DP combine CUDA kernel completed. `tp_ar_call -> tp_ar_return` measures the host-visible helper scope, including wrapper/enqueue, possible blocking or stream dependencies, and in `normal` the collective-related runtime effects. It is not guaranteed to equal physical NCCL kernel duration. In `skip`, this interval is clone-path overhead and must not be labeled AR duration.

## Observed shape and interpretation limits

The first rows of H800 normal/scalar/skip all report `shape=[16384,2048]`, `numel=33,554,432`, `dtype=torch.bfloat16` (about 67 MiB payload). This confirms that normal is a substantial payload collective; scalar uses one element; skip uses no collective. H800 first-request summaries nevertheless show normal per-rank host medians around 0.56–0.69 ms for `combine_return -> tp_ar_call` and 0.76–0.98 ms for `tp_ar_call -> tp_ar_return`, with layer-level cross-rank arrival spreads up to tens of ms. Scalar/skip retain similarly large spreads. These rows establish ordering and the payload-vs-rendezvous diagnostic distinction, but they cannot identify whether skew comes from DP completion, host scheduling, queued device work, or NCCL kernel execution without CUDA/NCCL completion timestamps.

The H800 first DP0 prefill batch spans were approximately 24.1 s (`normal`), 23.8–23.9 s (`scalar_sync`), and 18.6–18.7 s (`skip`); the H200 run was approximately 18.0 s, 17.9–18.0 s, and 19.6 s respectively. These are independent instrumented runs and are diagnostic only; their differences cannot be assigned as pure AR cost. `skip` also changes subsequent tensor values and execution behavior, so it cannot serve as a clean model result.
