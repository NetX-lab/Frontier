## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-09 | Recorded rank0 post-MoE TP AR stability RCA and the fresh H200 rerun submission. |

# TP0 post-MoE TP AR rank-stability RCA

## 1. Test Script Information

Existing diagnostic parser and artifacts:

- Parser: `tests/e2e/issue26_first_batch_op_rca_communication.py`
- Reduced communication artifacts: `analysis/h200-rca-comm-01/runtime/operators/`
- Full operator artifacts: `analysis/h200-rca-01/runtime/operators/`
- Independent formal operator artifacts: `analysis/formal-operators-03/`
- Source checkout: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`, commit `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`

Reproducible reduced-run extraction:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  tests/e2e/issue26_first_batch_op_rca_communication.py \
  --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-comm-01/runtime/operators \
  --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/first-batch-op-rca/communication/summary.json
```

Environment: CPU conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13 for parsing. The source artifacts were collected on one eight-H200 worker in `step_main`, with TP4/DP2/EP8, BF16, eager execution, FLASHINFER, uniform routing, prefix caching OFF, no chunked prefill, and 4096 prefill / 1024 decode settings.

A fresh repeat was authorized and submitted with the same settings:

```text
RJob: yc26-h200-rank-stability-20260909-01
Launcher: analysis/h200-rank-stability-01/launch.sh
Selection: communication
Output: analysis/h200-rank-stability-01/runtime/
```

The platform accepted the job, but the job remained `Pending` because the `step-main-default` queue reported `H200=0` remaining quota. No new GPU row was produced at the time of this report; the existing completed runs provide the RCA evidence.

## 2. Validation Criteria

The RCA tests the following claims:

1. Every TP rank records the same number of post-MoE AR calls for an aligned batch.
2. A shorter CUDA-event scope indicates a later participant arrival / less collective wait, rather than additional work performed by that rank.
3. If TP0 is a rank-intrinsic critical path, it should remain the slow or fast outlier across independent runs, DP lanes, and batch phases.
4. The model source should expose a rank0-only operation if rank0 performs extra model work.

The observed outer batch span is reported separately from per-rank scope sums. Ranks are parallel participants; their scope sums are never added together.

## 3. Test Results and Evidence

### Collection integrity

The reduced communication parser returned `PASS_COLLECTION_DIAGNOSTIC_ONLY`. For every selected rank/batch, the records contain 48 `expert_parallel_allreduce` calls, 48 attention TP AR calls, and one embedding TP AR call. The full and formal operator validations also retain 48 post-MoE AR rows per TP rank.

### Rank ordering across batches and runs

Post-MoE AR sums in milliseconds are:

| Run / batch | TP0 | TP1 | TP2 | TP3 | Smallest rank |
| --- | ---: | ---: | ---: | ---: | --- |
| Reduced DP0 / 4250 | 4.792544 | 25.384416 | 28.413280 | 27.393568 | TP0 |
| Reduced DP0 / 4251 | 4.809696 | 29.016576 | 26.142208 | 26.617312 | TP0 |
| Reduced DP0 / 4252 | 2.390848 | 38.431008 | 33.919232 | 34.384512 | TP0 |
| Full RCA DP0 / 4706 | 37.563168 | 37.882560 | 5.652832 | 37.671584 | TP2 |
| Full RCA DP0 / 4707 | 31.821504 | 36.966688 | 5.638912 | 30.183008 | TP2 |
| Full RCA DP0 / 4708 | 2.326208 | 9.975808 | 6.916928 | 9.592064 | TP0 |
| Formal DP0 / 3851 | 5.249856 | 29.211584 | 37.162848 | 20.706624 | TP0 |
| Formal DP0 / 3852 | 18.546336 | 8.274688 | 18.176064 | 18.475808 | TP1 |
| Formal DP0 / 3853 | 2.078240 | 28.840032 | 34.368352 | 25.727456 | TP0 |

The reduced run has a within-run TP0 pattern for its three DP0 batches, but this pattern does not persist in the full RCA run or the independent formal run. Across the available DP1 batches, the smallest rank also changes, including TP3, TP0, and TP2. Therefore “TP0 is stably late” is false as a rank-intrinsic statement. “TP0 was the late participant in reduced DP0 batches 4250–4252” is true as a local observation.

The corresponding reduced batch 4250 outer spans were 86.412033, 86.437950, 86.414658, and 86.443520 ms for TP0–TP3, a spread of 0.031487 ms despite a 23.620736 ms spread in post-MoE scope sums. Full RCA batch 4706 likewise ended at 116.210144–116.414497 ms while its post-MoE sums ranged from 5.652832 to 37.882560 ms. This is the signature of collective wait relocation, not rank-local extra work.

### Source RCA

`parallel_state.py:1060-1180` constructs the layout `ExternalDP x DP x PP x TP`. For DP2/PP1/TP4, the TP groups are `[0,1,2,3]` and `[4,5,6,7]`; world rank0 is DP0/TP0/EP0, without a leader-only model stage.

`qwen3_moe.py` runs the same decoder/MoE path on all TP ranks. The configured Qwen3 MoE model has `num_shared_experts=0`; its projections use `bias=False`. The only generic TP rank0 bias special case in `linear.py` is therefore inactive for this model. Embedding and final norm also execute across the TP ranks.

`fused_moe/layer.py:1795-1862` orders dispatch, expert compute, DP combine, and then the TP all-reduce recorded as `expert_parallel_allreduce`. The scope starts after the preceding work. Its elapsed CUDA event includes participant arrival, NCCL submission, device queue gaps, and collective waiting. A short scope means that rank entered near the end of the collective; it does not mean that rank performed more work inside the scope.

The full record-function first batch supports this interpretation: post-MoE AR sums were TP0/1/2/3 = 65.823843/4.619317/66.155092/38.273636 ms, while grouped-GEMM and gating work stayed close across ranks. The rank with the short AR changes with the batch and instrumentation path.

## RCA conclusion

TP0's late post-MoE AR submission is a real observation for the reduced DP0 run, and it persists across its three selected batches. It is not a stable TP0/rank0 property: independent full-scope, formal, DP1, and kernel traces move the short/late rank to TP1, TP2, or TP3. No rank0-only model operation or extra operator count exists in the active Qwen3 path. The supported cause is per-batch participant-arrival skew caused by the preceding EP routing/dispatch/expert execution and host/device launch queue timing, followed by NCCL wait redistribution. The exact split between EP token-load skew and host scheduling remains unmeasured in a clean forward.

This evidence does not authorize adding a 20–30 ms TP0 term to Frontier. The post-MoE scope is an inclusive participant-wait measurement, and summing ranks would double-count the same collective. It also does not close the CUDA operator gate or justify clean/diagnostic span reconciliation as a production correction. A future targeted run should add per-rank routing counts, preceding MoE scope endpoints, host launch timestamps, and device start/end timestamps in the same low-perturbation forward; the queued repeat is intended to provide the rank-stability portion when H200 quota becomes available.
