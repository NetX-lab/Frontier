## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-12 | Defined the H200 warmup-normalized all2all backend research and clean-span experiment. |

# H200 all2all backend research and test plan

## Objective

Determine whether a vLLM all2all implementation other than `naive` has both
communication semantics and clean first-formal CUDA batch-span behavior closer
to Frontier's current ideal EP dispatch/combine abstraction. This is a bounded
research experiment; it does not authorize changing Frontier's communication
model or replacing the current canonical vLLM reference.

## Fixed case and evidence boundary

- H200 `step_main`, `h200`, eight GPUs, `num_gpu_blocks_override=310809`.
- Qwen3-30B-A3B dummy weights, BF16, eager, FlashInfer attention,
  TP4/DP2/PP1/EP8, uniform routing, prefix caching OFF, chunked prefill OFF.
- Ten fully drained 100-request warmup replays followed by 100 formal requests.
- Standard chain: `issue26_h200_replay_worker.sh` -> uniform preflight ->
  official clean worker -> batch diagnostics worker.
- Clean batch-only first formal batch is selected by request identity
  `cmpl-pf4096_dc1024:0-0`, one request, 4096 prefill tokens, zero decode,
  DP0 TP0--TP3. Report median, P90, rank max, and rank spread.
- Clean official request metrics remain separate from batch-only CUDA spans.
  Instrumented operator/full diagnostic spans are diagnostic only.

## Source hypotheses

| Backend | vLLM source behavior | Relation to Frontier | Test decision |
| --- | --- | --- | --- |
| `naive` | DP broadcasts for hidden/router payloads, DP combine all-reduce plus slice, then TP4 post-MoE all-reduce | Primitive graph differs from ideal EP8 all-to-all; canonical reference | Re-run as warmup-10 baseline |
| `pplx` | PPLX fused prepare/finalize and all2all kernels; post-MoE reduction is handled by the fused path | EP all2all is structurally closer, but implementation and payload packing differ | Run only if `pplx_kernels` is available and startup succeeds |
| `deepep_high_throughput` | DeepEP high-throughput buffer dispatch/combine and fused expert path | EP participant graph is closer to Frontier's ideal EP phase, with runtime overlap and kernel-specific protocol | Run only if `deep_ep` and its CUDA dependencies initialize |
| `deepep_low_latency` | DeepEP low-latency buffer dispatch/combine and fused expert path | Same EP graph, different queue/protocol objective and likely different span | Run only if `deep_ep` and its CUDA dependencies initialize |

The source selector and capability checks are in
`vllm/distributed/device_communicators/cuda_communicator.py:84-103`,
`vllm/distributed/device_communicators/all2all.py:17-66, 74-281`, and
`vllm/model_executor/layers/fused_moe/config.py:360-400`.

## Execution order

1. Verify the active worker syntax, ten-warmup client/analyzer contract, and
   backend selector override on CPU.
2. Submit a small H200 capability probe that imports `pplx_kernels` and
   `deep_ep` in the pinned GPU image. It emits no model timing and is not used
   as a parity result.
3. Run one isolated ten-warmup `naive` standard replay to establish a
   warmup-normalized H200 reference.
4. For each capability-positive alternative, run the same standard chain in a
   separate output directory with only `ISSUE26_ALL2ALL_BACKEND` changed.
5. Validate complete drain, identities, source commit, backend manifest,
   first-formal predicates, and the four-rank clean batch statistics.
6. Rank candidates by semantic closeness and independently report span values.
   Do not apply a Frontier change or reconcile diagnostic spans from these
   results alone.

## Decision criteria

- A candidate is **unsupported** when its optional library is absent or server
  initialization fails before formal requests; preserve the failure log.
- A candidate is **timing-valid** only after all ten warmups, 100 formal rows,
  identity PASS, complete drain, and first-formal predicates pass.
- A candidate is **semantically closer** only when source inspection confirms
  an EP all2all participant graph and payload path closer to Frontier's ideal
  EP abstraction. A lower clean span alone is insufficient.
- No candidate closes the Frontier CUDA gate or authorizes communication
  correction. The measured `+9.740 ms` compute excess remains a separate
  attribution lane.
