## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-11 | Audited diagnostic vLLM MoE execution path and existing stage hooks. |

# MoE subphase code audit

This is a read-only audit of `/data/ycfeng/tmp/issue26-vllM-diagnostics-ar-ab`
(actual directory name: `issue26-vllm-diagnostics-ar-ab`), currently at
`eb4c9a1394ef136f134b5a36538847ec01679c68`.

## Execution path and available scopes

The Qwen3 MoE path starts in
`vllm/model_executor/models/qwen3_moe.py:177-193`:

1. `Qwen3MoeSparseMoeBlock.forward()` records `moe_gating` around the
   replicated gate linear.
2. `self.experts(...)` enters `FusedMoE.forward_impl()` in
   `vllm/model_executor/layers/fused_moe/layer.py:1818`.
3. With DP=2 and `VLLM_ALL2ALL_BACKEND=naive`,
   `do_naive_dispatch_combine` is true (`layer.py:1830-1835`). The dispatch
   calls `GroupCoordinator.dispatch()` and is wrapped by
   `expert_parallel_alltoall_dispatch` (`layer.py:1843-1849`).
4. The local envelope `local_moe_apply` wraps `quant_method.apply()`
   (`layer.py:1851-1887`).
5. The combine calls `get_ep_group().combine()` and is wrapped by
   `expert_parallel_alltoall_combine` (`layer.py:1898-1907`).
6. `maybe_all_reduce_tensor_model_parallel()` then performs the post-MoE
   all-reduce. For naive EP it uses `expert_parallel_allreduce` because
   `ep_size > 1` (`layer.py:1599-1625`). Diagnostic modes `normal`,
   `scalar_sync`, and `skip` only alter this last all-reduce.

The naive DP communicator is in
`vllm/distributed/device_communicators/all2all.py:17-58`:

- dispatch broadcasts each DP rank's token slice to the other DP rank and
  returns the concatenated buffer; with the formal case this creates the
  observed `[4097, 2048]` local input from `[4096, 1]` DP token counts;
- combine all-reduces that buffer across the DP group and slices the local
  range. This is a DP collective and is distinct from the following TP/EP
  all-reduce.

## Exact local compute decomposition

For the non-modular `fused_experts_impl()` path in
`vllm/model_executor/layers/fused_moe/fused_moe.py:1570-1806`, the following
record scopes already exist:

- `moe_gating`: routing/top-k selection in `layer.py:1478-1516` (the Qwen gate
  linear is a separate scope in `qwen3_moe.py`, so the expected count is two
  per decoder layer);
- `moe_shuffling`: `moe_align_block_size(...)`, which sorts token IDs and
  computes block padding (`fused_moe.py:1716-1720`);
- `moe_grouped_gemm`: parent envelope around W1 GEMM, activation, and W2 GEMM
  (`fused_moe.py:1721-1777`);
- `moe_grouped_gemm_w1`: first expert grouped GEMM;
- `moe_activation`: fused SiLU-and-mul (or GELU) kernel;
- `moe_grouped_gemm_w2`: second expert grouped GEMM;
- `moe_sum`: `ops.moe_sum(...)`, the routed output reduction into the token
  output (`fused_moe.py:1798-1801`).

These scopes are nested. `moe_grouped_gemm` is an envelope and must not be
added to W1 + activation + W2. The non-overlapping local compute estimate is
`moe_shuffling + moe_grouped_gemm_w1 + moe_activation +
moe_grouped_gemm_w2 + moe_sum`, with `moe_gating` kept separate.

The current `issue26_h200_diagnostics_worker.sh` selects these scopes in two
different modes:

- `compute_moe`: `moe_gating,moe_shuffling,moe_grouped_gemm,moe_sum`;
- `compute_detail`: `...moe_grouped_gemm_w1,moe_activation,moe_grouped_gemm_w2`.

Thus no existing single run records all non-overlapping MoE child scopes. The
two runs also have different profiler overhead and cannot be summed as if they
were the same execution. A minimal next diagnostic is one standard warmup run
whose scope list contains the union
`moe_gating,moe_shuffling,moe_grouped_gemm_w1,moe_activation,moe_grouped_gemm_w2,moe_sum`
(optionally keep the parent envelope only as a consistency check).

## Routing and token-population observability

`vllm/v1/utils.py:563-765` already logs per-layer routing data when
`VLLM_FRONTIER_MOE_ROUTING_LOG_PATH` is set. Each record includes
`num_tokens`, `router_topk`, `global_num_experts`, `local_num_experts`,
`ep_rank`, `ep_size`, `total_routed_tokens`, `per_expert_tokens`,
`min_load_ratio`, `max_load_ratio`, `load_imbalance_cv`, utilization, entropy,
and Gini. For an `expert_map`, counts are converted to local expert IDs before
logging. This is enough to test whether rank-local grouped GEMM work differs by
expert population; the current uniform-routing flag does not prove equal local
expert counts.

The routing record does not expose `num_tokens_post_padded` from
`moe_align_block_size`. That value controls grouped-GEMM block count and is the
most useful missing routing-to-work quantity. It can be added with a narrow
metadata hook immediately after `moe_align_block_size` in `fused_moe.py`, but
the hook should be diagnostic-only and record the selected batch/layer plus
`tokens_in_chunk`, `top_k`, block size, `num_tokens_post_padded`, and the local
expert counts. No synchronization is required.

## Recommended evidence collection

1. Run one H200 standard warmup-chain diagnostic with the union of child MoE
   scopes above and retain `expert_parallel_alltoall_dispatch/combine` and
   `expert_parallel_allreduce` as separate outer scopes. Do not add nested
   parent durations to child sums.
2. Join the selected `batch_id` and layer sequence across DP0 TP0--TP3 using
   `server.ops.*.jsonl`, `server.batch.*.jsonl`, and routing JSONL. Compare
   local child-scope durations with `per_expert_tokens` and padded-token
   metadata when available.
3. If child durations remain rank-skewed after matching token populations,
   use a single `torch.profiler` record-function capture for the selected
   formal batch. The existing profiler path is configured in
   `vllm/v1/utils.py:340-455`; it exports a Chrome trace with the exact nested
   scope hierarchy. Treat that run as diagnostic because profiler/export and
   final synchronization perturb the clean span.
4. Keep `local_moe_apply` and collective completion events as envelopes for
   critical-path ordering. Use the child scopes to explain local work; never
   infer a correction by summing rank-local inclusive `tp_ar` durations.

## Evidence limits

The current completion event hook in `vllm/v1/frontier_trace.py` measures the
whole `quant_method.apply()` envelope and does not distinguish routing,
grouped GEMM, activation, or routed reduction. The current CUDA-event logger
(`vllm/v1/utils.py:230-470`) records scope-level device intervals, which may
include queued work and host-induced gaps in `scope_mode=default`. The parent
`moe_grouped_gemm` interval overlaps its child intervals. Therefore the
existing local-stage RCA cannot identify the missing ~20 ms source until the
child scopes and routing population are joined on the same first formal batch.
