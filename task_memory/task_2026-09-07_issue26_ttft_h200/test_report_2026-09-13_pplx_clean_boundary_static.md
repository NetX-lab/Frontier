## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the uninstrumented PPLX clean-boundary worker and static validation. |

# PPLX clean-boundary worker static validation

## Execution

The existing PPLX compatibility checkout is
`/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt`. Its clean-boundary
support is committed at `448f2b65e7679ae7490114ad382b6ba79becb3c3` and the
checkout is clean. The worker is
`tests/e2e/issue26_h200_pplx_clean_boundary_worker.sh`.

Static checks run from the Frontier worktree:

```bash
bash -n tests/e2e/issue26_h200_pplx_clean_boundary_worker.sh
python -m py_compile tests/e2e/issue26_pplx_boundary_analysis.py
git diff --check
```

All checks passed. No GPU command was launched by this report.

A fake-CUDART smoke test also passed: a nonmatching warmup context produced no
record, while a matching first-formal context produced exactly one per-rank
record with the expected request and token predicates and one synchronize call.

## Intended execution contract

The worker keeps the frozen H200 case: `step_main`, GPU selector `h200`,
`num_gpu_blocks_override=310809`, Qwen3-30B-A3B dummy weights, TP4/DP2/PP1/EP8,
BF16, eager, FlashInfer, uniform routing, prefix caching OFF, chunked prefill
OFF, 4096-prefill/1024-output. It requires ten complete drained 100-request
warmup replays and 100 formal requests (1100 client rows), then validates the
first formal request `cmpl-pf4096_dc1024:0-0` with batch size 1, 4096 prefill
tokens, and zero decode tokens.

The worker sets `VLLM_MOE_DP_CHUNK_SIZE=4096` by default. vLLM's default is 256
(`vllm/envs.py:959-965`), and PPLX's chunked path iterates from zero through
`max_tokens_across_dispatchers` in that quantum (`layer.py:1793-1806`). A
4096-token first prefill therefore uses one PPLX wave instead of sixteen. This
is a scoped runtime-setting correction for the PPLX experiment; it does not
change Frontier production modeling.

Frontier instrumentation and all per-op, scheduler, routing, completion, and
batch loggers are disabled. The source records one independent CUDA-event pair
around the selected `model.forward`, performs one post-end synchronization to
read the event, and writes per-rank records. This is the minimum event-based
boundary needed to obtain a batch CUDA span; it is separate from the full
diagnostic outer span and from Nsight capture. The analyzer derives the actual
selected DP lane and requires complete TP0--TP3 coverage.

## Acceptance criteria

The GPU run is admissible only when ten warmups are drained, 100 formal rows
are complete, the identity validator/analyzer passes, and one selected DP lane
contains all TP0--TP3 first-formal records. The resulting median, P90, rank max,
and rank spread must be reported from those four CUDA-event values. A measured
value remains a PPLX clean-boundary event envelope and must not be used to fit
Frontier predictors, communication costs, CPU add-ons, or clean/diagnostic
reconciliation without separate review.
