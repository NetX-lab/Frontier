## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Audited default CUDA_EVENT scope nesting, Frontier ledger mappings, and communication stream dependencies for the frozen Qwen3 H200 case. |

# Operator Scope Contract

Scope: Qwen3-30B-A3B, BF16, eager execution, FLASHINFER attention, naive all-to-all, attention DP2/TP4, MoE EP8/TP1, PP1. Source versions: Frontier `74b1800e9562d5ddfab3a0c091ca300b4dcb9473`; diagnostic vLLM `361d941c97fcec52e544f74b7ab91c54192de9c9`. This document records inspected execution boundaries and candidate comparison groups, not numerical parity or root-cause attribution. No source code or probes were changed during this audit.

## Scope and ledger mapping

The default logger selects 25 names, exactly as declared in [DEFAULT_FRONTIER_CUDA_EVENT_OP_SCOPES](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/v1/utils.py:42). Selection does not expand aliases. Frontier's additive ledger is built in [MetricsStore._build_frontier_stage_batch_component_ledger](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/frontier/metrics/metrics_store.py:3856). Ledger values cover the stage's layers; do not multiply them by 48 again.

| Recorded vLLM scope | Frontier `component_ledger_ms` comparison group | Boundary and accounting rule |
| --- | --- | --- |
| `attn_pre_proj` | `attention_pre_proj_time` | QKV projection plus Q/K normalization; retain the whole scope. [Attention source](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/models/qwen3_moe.py:306). |
| `attn_rope` | `attention_rope_execution_time` | Rotary embedding scope. [Source](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/models/qwen3_moe.py:361). |
| `attn_kv_cache_save` | `attention_kv_cache_save_execution_time` | Cache write precedes the attention kernels; it is a sibling of prefill/decode. [Source](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/v1/attention/backends/flashinfer.py:861). |
| `attn_prefill`, `attn_decode` | `attention_prefill_execution_time`, `attention_decode_execution_time` | Separate sequential branches; a mixed local batch may contain both. [Prefill](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/v1/attention/backends/flashinfer.py:913), [decode](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/v1/attention/backends/flashinfer.py:1008). |
| `attn_post_proj` | `attention_post_proj_time + attention_all_reduce_time` | Includes O projection and its TP4 reduction. The child is named `attn_post_proj_tp_allreduce`, which is absent from the default list. Default logs cannot split this parent into projection and communication. [Parent](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/models/qwen3_moe.py:382), [child name and call](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/linear.py:1304). |
| `input_layernorm` | `attn_norm_time`, grouped with the preceding layer's `add_ffn_residual_time` where residual exists | Layer 0 has standalone normalization; subsequent layers perform fused residual-add/RMSNorm. The nested `add` measures the same fused call. A full-stage comparison must preserve the terminal-layer boundary described below. [Source](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/models/qwen3_moe.py:453). |
| `post_attention_layernorm` | `mlp_norm_time + add_attn_residual_time` | Fused residual-add/RMSNorm. Count the parent once; do not add the nested `add` duration. [Source](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/models/qwen3_moe.py:552). |
| `add` | No independently measured add-only value | These default scopes enclose the same fused calls as their layernorm parents. Parent minus child is not an isolated normalization kernel measurement. |
| `moe_gating`, even/odd `scope_seq` | Even: `moe_gating_linear_time`; odd: `moe_gating_routing_topk_time` | In this fixed path, each layer first computes local router logits, then performs TopK after naive dispatch. They are separate sibling scopes and may be summed after mapping. Their token domains differ. [Linear](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/models/qwen3_moe.py:184), [TopK](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe/layer.py:1477). |
| `moe_shuffling` | `moe_shuffling_time` | `moe_align_block_size`, separate from grouped GEMM; sum invocations if the runtime chunks the input. [Source](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe/fused_moe.py:1716). |
| `moe_grouped_gemm` | Candidate: `moe_grouped_gemm_time`; internal contract differs | Runtime scope includes W1, SiLU-and-multiply, and W2. The Frontier profiling helper omits that activation and copies a slice before W2. This needs separate analysis before numerical comparison; see below. |
| `expert_parallel_alltoall_dispatch`, `expert_parallel_alltoall_combine` | Candidate combined group: `expert_parallel_communication_time` | Each collective has two nested scopes with the same name. Retain one boundary, preferably the outer scope for workflow accounting. Actual primitives are DP2 broadcasts and DP2 all-reduce; Frontier currently models EP8 all-to-all, so name equality does not establish primitive/domain parity. |
| `expert_parallel_allreduce` | Candidate MoE-output reduction group; no established exact ledger match | Despite its name, this invokes the attention TP4 group after combine. Do not map it to Frontier's EP8 all-reduce registry entry or assume equivalence to the MoE-TP1 `mlp_all_reduce_time`. [Name selection](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe/layer.py:1599), [actual TP group](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/distributed/communication_op.py:15). |
| `tensor_parallel_allreduce` | No dedicated input-embedding reduction field in the current ledger | The single scope before layer 0 comes from `VocabParallelEmbedding`; it is not the attention O-projection reduction. [Source](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/vocab_parallel_embedding.py:433). |

The six `attn_mla_*` names, `moe_tensor_parallel_allreduce`, and `kv_p2p_send`/`kv_p2p_recv` are in the default list but are not active in this Qwen3/EP8/PP1 path. Their absence is not missing coverage for this case. `moe_expert`, `moe_grouped_gemm_w1`, `moe_grouped_gemm_w2`, and the named attention TP-reduction child are absent from the default list, although corresponding wrappers exist in source.

## Nested scopes and uncovered work

For each layer, [FusedMoE.forward_impl](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe/layer.py:1804) opens the dispatch parent, then [GroupCoordinator.dispatch](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/distributed/parallel_state.py:839) opens its same-named child. [FusedMoE.reduce_output](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe/layer.py:1852) and [GroupCoordinator.combine](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/distributed/parallel_state.py:854) do the same for combine. Each name's sequence is allocated at entry and records are queued at exit: outer `2L`, inner `2L+1`, written inner first. Adding both duplicates one communication interval. The sequence rule is specific to this inspected source/path and must be checked against actual formal scope coverage before use.

The output projection includes its unselected communication child. Layernorm parents contain `add`. The two `moe_gating` calls are siblings, not nested. `moe_shuffling` and `moe_grouped_gemm` are also siblings; their encompassing `moe_expert` wrapper is unselected.

The runtime [grouped-GEMM body](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe/fused_moe.py:1721) contains `silu_and_mul` at line 1747. Frontier's [_run_fused_moe_iteration](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/frontier/profiling/moe/moe_vllm_kernel.py:269) executes W1, selects the first half of its output and calls `contiguous()`, then executes W2. This is an observed profiling-boundary mismatch, not a measured explanation of the TTFT gap. The runtime `ops.moe_sum` at [line 1797](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe/fused_moe.py:1797) is outside the grouped-GEMM scope and has no selected default scope.

Input embedding lookup/masking and the final fused residual/RMSNorm at [Qwen3MoeModel.forward](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/models/qwen3_moe.py:623) are not covered by selected compute scopes. The final FFN residual is consumed by that final normalization; it does not create a 96th `add` record. Logits/sampling occur after GPUModelRunner finishes these model-forward loggers. A deduplicated scope sum therefore remains a partial forward decomposition; subtracting it from clean official TTFT does not yield CPU overhead.

## Communication event and stream semantics

In `scope_mode=default`, [FrontierCudaEventOpLogger.scope](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/v1/utils.py:348) records start/end events on the current stream, with no per-scope device synchronization. The end-of-batch synchronization makes events readable; it does not insert missing cross-stream dependencies into earlier scopes. Durations include work and waits ordered between those events, including possible device idle time while the host enqueues the next operation. They are not isolated kernel-active time or separately measured CPU service time.

For this actual naive path, cross-stream communication completion is ordered back into the measured stream:

- Dispatch uses [NaiveAll2AllManager.naive_multicast](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/distributed/device_communicators/all2all.py:28), then [GroupCoordinator.broadcast](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/distributed/parallel_state.py:455). PyTorch 2.8.0's synchronous-default `broadcast` calls `work.wait()`; ProcessGroupNCCL first orders its stream after the current stream, and `WorkNCCL::synchronizeStream` blocks the current stream on the NCCL completion event. Thus the enclosing default CUDA_EVENT interval includes the broadcast's required cross-stream wait and peer arrival delay, not merely its launch. Sources: [PyTorch broadcast](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/distributed_c10d.py#L2785), [ProcessGroupNCCL stream synchronization](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L775).
- Combine's DP all-reduce and explicit TP reductions pass through [CudaCommunicator.all_reduce](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/distributed/device_communicators/cuda_communicator.py:105). The PyNCCL path uses the current stream directly at [pynccl.py:125](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/distributed/device_communicators/pynccl.py:125). The custom all-reduce path launches its buffer copy and kernel on `getCurrentCUDAStream()` at [custom_all_reduce.cu:62](/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/csrc/custom_all_reduce.cu:62). The default event boundary therefore encloses these ordered operations. The selected implementation can depend on payload size; this source audit does not assert which implementation every formal row took.

This conclusion is limited to inspected communication paths with established stream dependencies; it does not imply that default events cover arbitrary asynchronous work on every CUDA stream. Both PyTorch source reads used the company HTTP proxy and the pinned `v2.8.0` upstream tag corresponding to the observed runtime package version.

## Current-run schema/count confirmation

Read only the first complete observed batch of `runs/h200-diagnostics-03/runtime/operators/server.ops.dp0.tp0.pp0.jsonl`, stopping at the next batch boundary with a 6,000-line cap. No timing values were extracted. This warmup sample confirms names, local sequence order, and counts; it is not formal numerical evidence.

| Observed warmup scope | Count in batch 0 |
| --- | ---: |
| `tensor_parallel_allreduce` | 1 |
| `input_layernorm`, `post_attention_layernorm` | 48 each |
| `attn_pre_proj`, `attn_rope`, `attn_kv_cache_save`, `attn_prefill`, `attn_post_proj` | 48 each |
| `add` | 95 |
| `moe_gating` | 96 |
| `expert_parallel_alltoall_dispatch`, `expert_parallel_alltoall_combine` | 96 each |
| `moe_shuffling`, `moe_grouped_gemm`, `expert_parallel_allreduce` | 48 each |

Observed sequence prefixes: gating `0,1,2,3,4,5`; dispatch/combine `1,0,3,2,5,4`. Formal request identity, phase coverage, routing alignment, and numerical comparisons remain separate required checks. No historical measurements were consumed.
