## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-09 | Recorded read-only Lane D audit of post-MoE TP AR shapes, DP/EP token boundaries, and existing runtime hooks. |

# Lane D: Post-MoE TP AR input and routing shape audit

This is a read-only source/artifact audit. It does not modify the shared vLLM checkout or Frontier production code.

## Direct source evidence

- `vllm/distributed/parallel_state.py:1100-1174` defines the layout as `ExternalDP x DP x PP x TP`. For world size 8 with DP2/PP1/TP4, the TP groups are `[0,1,2,3]` and `[4,5,6,7]`; the DP groups are `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`; the EP group is each flattened DP×TP group (`[0..7]`).
- `vllm/model_executor/layers/fused_moe/config.py:267-301` flattens DP and TP into EP when expert parallelism is enabled. With DP2/TP4, `FusedMoEParallelConfig` uses local `tp_size=1`, `ep_size=8`, and `ep_rank=dp_rank*4+tp_rank`. The later `tensor_model_parallel_all_reduce()` still resolves the global TP group through `get_tp_group()`.
- `vllm/model_executor/layers/fused_moe/layer.py:1790-1808` performs EP dispatch; `:1852-1862` performs `get_ep_group().combine(states)` and then `maybe_all_reduce_tensor_model_parallel(states)`. There is no model-local operation between the combine return and the post-MoE AR call.
- `vllm/distributed/device_communicators/all2all.py:46-66` implements the naive path. Dispatch creates a buffer sized by `cu_tokens_across_dp_cpu[-1]`, broadcasts each DP slice through `self.dp_group`, and combine performs `self.dp_group.all_reduce(hidden_states)` before slicing `[start:end]`. This synchronizes DP peers only; it does not synchronize the TP peers in `[0,1,2,3]` or `[4,5,6,7]`.
- `vllm/forward_context.py:72-116` obtains per-DP token counts by CPU-group all-reduce, computes cumulative boundaries, and stores them in `DPMetadata`. `all2all.py:60-66` uses those boundaries for the post-combine slice. The output shape is therefore the local DP slice length, with the common hidden dimension.
- `vllm/model_executor/layers/fused_moe/layer.py:1599-1613` records the post-MoE operation as `expert_parallel_allreduce` when `ep_size>1`, then calls `tensor_model_parallel_all_reduce(final_hidden_states, ...)`. For this BF16 model (`hidden_size=2048`), a 4096-token slice is `[4096,2048]` and is about 16 MiB; a one-token lockstep slice is `[1,2048]`.
- `vllm/model_executor/models/qwen3_moe.py:118-155` configures 128 global experts, top-k 8, and the FusedMoE with `reduce_results=True`; `:176-189` flattens model input to `[num_tokens, hidden_size]` before the FusedMoE call. The active model has no shared experts (`num_shared_experts=0` in the existing RCA report), so the single `reduce_output(final_hidden_states)` path applies.

## Existing hooks and their limits

- `vllm/v1/utils.py:622-765` (`FrontierMoeRoutingLogger.log_routing`) records `num_tokens`, routing top-k, EP rank, total routed tokens, and per-local-expert counts. Its `topk_ids` are read after routing and expert-map filtering; this is a useful local grouped-GEMM workload signal, but it does not record `final_hidden_states.shape` or DP slice boundaries.
- `vllm/distributed/communication_op.py:15-35` records only static collective metadata (`collective_domain`, group name/rank/world size). The existing `record_frontier_op_meta()` path in `vllm/v1/utils.py:395-424` can carry shape metadata while a scope is active, but current operator rows therefore have no input shape.
- The diagnostic checkout already contains `VLLM_FRONTIER_DIAG_MOE_AR_MODE={normal,scalar_sync,skip}` at `layer.py:1611-1632`. This changes payload/synchronization behavior only for timing diagnostics and does not expose shape by itself.

## Existing artifact evidence

`task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-diagnostics-03/runtime/routing/server.routing.*.jsonl` shows rank-dependent local routing work while the dispatched population is shared per DP forward. For `batch_id=0`, `model.layers.0.mlp.experts` reports:

| DP rank | TP rank | `num_tokens` after dispatch | local routed tokens | local expert CV |
|---:|---:|---:|---:|---:|
| 0 | 0 | 4097 | 4176 | 0.1243 |
| 0 | 1 | 4097 | 4083 | 0.0778 |
| 0 | 2 | 4097 | 4296 | 0.0734 |
| 0 | 3 | 4097 | 4081 | 0.0551 |
| 1 | 0 | 12289 | 12308 | 0.0617 |
| 1 | 1 | 12289 | 12034 | 0.0584 |
| 1 | 2 | 12289 | 12118 | 0.0572 |
| 1 | 3 | 12289 | 11952 | 0.0645 |

For `batch_id=1`, DP local populations flip: DP0 has `num_tokens=12289`, while DP1 has `num_tokens=4100`. The `num_tokens` field is the post-dispatch population; it can include a one-token lockstep dummy when the peer DP rank is idle. The route log does not prove post-combine output shape because it is emitted before combine.

The existing operator JSONL rows under `analysis/h200-rca-01/runtime/operators/` contain timing/count/worker metadata but no shape or `numel` fields. Thus the current artifacts cannot directly verify the post-combine AR payload shape.

## Evidence-based conclusion

For one DP lane, all four TP ranks execute the same scheduler token slice and the same post-combine slice boundaries. The post-MoE TP AR payload shape is therefore equal across TP ranks within that lane and is determined by that lane's DP token count and `hidden_size=2048`; it does not vary by TP rank based on the current code. The observed 5–38 ms per-rank CUDA-event scope differences cannot be attributed to different AR payload sizes without a future run showing divergent `cu_tokens_across_dp_cpu` or `states.shape` within one DP lane. They remain consistent with arrival/queue/participant-wait redistribution.

The per-EP-rank routed-token counts are genuinely rank-dependent (for example, 4081–4296 in DP0 batch 0), so they are a valid candidate contributor to local expert execution and arrival skew. The current routing records do not establish how much of that skew survives the DP combine boundary.

## Lowest-perturbation next measurement

A fresh diagnostic forward should capture one row per layer and rank containing:

1. `dp_rank`, `tp_rank`, `ep_rank`, `batch_id`, `layer_name`;
2. `cu_tokens_across_dp_cpu` and the derived `[start,end)` local slice;
3. `states.shape`, dtype, and `numel` immediately after `get_ep_group().combine(states)`;
4. the same shape/numel at entry to `tensor_model_parallel_all_reduce`;
5. local routed-token counts already available from `FrontierMoeRoutingLogger`.

The minimal source seam is `FusedMoE.reduce_output()` immediately after `get_ep_group().combine(states)` and before `maybe_all_reduce_tensor_model_parallel(states)`. Dynamic shape fields can be attached through the existing active-scope metadata mechanism used by `communication_op.py`; avoid `torch.cuda.synchronize()` or host waits. A separate explicit JSONL diagnostic row is preferable if metadata consistency across repeated scopes would make shape metadata unwieldy.
