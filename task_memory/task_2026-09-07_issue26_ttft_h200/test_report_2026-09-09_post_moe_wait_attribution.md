## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-09 | Recorded the per-rank pre-MoE versus post-MoE AR wait attribution RCA. |

# Post-MoE TP AR wait attribution RCA

## Test Script Information

The analysis reads the validated CUDA-event operator rows from:

- `analysis/h200-rca-01/runtime/operators/`
- `analysis/formal-operators-03/`
- `analysis/h200-rca-comm-01/runtime/operators/`

The independent verification command was:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import json, glob, math
from collections import defaultdict
from statistics import mean

def inspect(pattern):
    rows = defaultdict(lambda: defaultdict(float))
    for path in glob.glob(pattern):
        for line in open(path):
            row = json.loads(line)
            key = (row["batch_id"], row["tp_rank"])
            if row["op_name"] in {
                "moe_gating", "expert_parallel_alltoall_dispatch",
                "moe_shuffling", "moe_grouped_gemm",
                "expert_parallel_alltoall_combine",
            }:
                rows[key]["pre"] += row["cuda_time_ms"]
            if row["op_name"] == "expert_parallel_allreduce":
                rows[key]["ar"] += row["cuda_time_ms"]
    values = [(x["pre"], x["ar"]) for x in rows.values()
              if "pre" in x and "ar" in x]
    xbar = mean(x for x, _ in values)
    ybar = mean(y for _, y in values)
    corr = sum((x - xbar) * (y - ybar) for x, y in values) / math.sqrt(
        sum((x - xbar) ** 2 for x, _ in values)
        * sum((y - ybar) ** 2 for _, y in values))
    print(len(values), corr)

inspect("task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-01/runtime/operators/server.ops.dp*.pp0.jsonl")
inspect("task_memory/task_2026-09-07_issue26_ttft_h200/analysis/formal-operators-03/server.ops.dp*.pp0.jsonl")
PY
```

Environment: CPU conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13. Source timing rows were collected on H200 `step_main` with TP4/DP2/EP8, BF16, eager, FLASHINFER, uniform routing, prefix caching OFF, and the fixed 4096-prefill case.

## Validation Criteria

The check tests whether a rank with more pre-MoE work tends to enter the shared post-MoE collective later and therefore records a shorter post-MoE CUDA-event scope. It also checks whether rank-local scope differences predict the outer batch span.

The expected evidence is:

- negative correlation between pre-MoE work and post-MoE AR scope;
- equal post-MoE call counts across TP ranks;
- nearly equal outer spans despite large rank-local AR-sum differences;
- no single 20 ms post-MoE kernel event.

## Results and Evidence

The independent calculation returned:

```text
h200-rca-01:       24 rank/batch samples, Pearson(pre, AR) = -0.7603954471
formal-operators:  24 rank/batch samples, Pearson(pre, AR) = -0.4868734905
```

These are diagnostic inclusive CUDA scopes, so the correlations establish ordering evidence rather than pure compute attribution.

For full RCA batch 4706:

| TP rank | pre-MoE scope sum | post-MoE AR sum |
| --- | ---: | ---: |
| TP0 | 48.692 ms | 37.563 ms |
| TP1 | 48.493 ms | 37.883 ms |
| TP2 | 75.664 ms | 5.653 ms |
| TP3 | 48.966 ms | 37.672 ms |

TP2 has approximately 27 ms more pre-MoE scope than its peers and approximately 32 ms less post-MoE AR scope. Its later arrival explains why the other ranks observe long collective scopes while TP2 observes almost the pure per-call collective duration.

For formal batch 3851, the late participant changes:

| TP rank | pre-MoE scope sum | post-MoE AR sum |
| --- | ---: | ---: |
| TP0 | 66.231 ms | 5.250 ms |
| TP1 | 46.415 ms | 29.212 ms |
| TP2 | 40.115 ms | 37.163 ms |
| TP3 | 51.841 ms | 20.707 ms |

Here TP0 has the largest pre-MoE scope and the shortest post-MoE scope. This is a batch-specific late-arrival pattern, not a rank0-only path.

The same collective's outer span remains shared. For reduced communication batch 4250:

| TP rank | post-MoE AR sum | outer batch span |
| --- | ---: | ---: |
| TP0 | 4.792544 ms | 86.412033 ms |
| TP1 | 25.384416 ms | 86.437950 ms |
| TP2 | 28.413280 ms | 86.414658 ms |
| TP3 | 27.393568 ms | 86.443520 ms |

The AR-sum spread is `23.620736 ms`, while the outer-span spread is only `0.031487 ms`. The early ranks' waits overlap the late rank's preceding work and therefore are not additional serial batch time.

## Causal conclusion

For a layer `l`, let `A[r,l]` be rank `r`'s arrival/submission time, `K[l]` the collective completion time, and `S[r,l]`/`E[r,l]` the CUDA-event boundaries. The diagnostic scope is approximately:

```text
scope[r,l] = E[r,l] - S[r,l] ≈ K[l] - A[r,l]
```

The physical collective work is close to common across the TP ranks. A rank that reaches the call early waits for the latest participant, so its scope is long. The rank that reaches the call late has a short scope. The shared collective completion, and therefore the forward critical path, is controlled by the latest participant at each layer:

```text
critical_end[l] ≈ max_r(A[r,l]) + collective_kernel[l]
```

The long scope observed by early ranks is concurrent with the late rank's pre-MoE work. Therefore:

- the late rank's preceding routing/dispatch/expert/queue delay can **drag the outer batch span**;
- the 20–30 ms difference between rank-local AR sums is **not an additional 20–30 ms to add to the outer span**;
- summing all TP rank scopes would double-count one shared collective.

The current evidence ranks the sources as follows:

1. **High confidence:** participant arrival and collective wait are the direct cause of the rank-local scope spread.
2. **Medium-high confidence:** batch-specific pre-MoE routing/dispatch/expert work and host/device enqueue scheduling select the late participant. The negative correlations and the batch 4706 / 3851 examples support this.
3. **Medium confidence:** expert token-load skew contributes to pre-MoE variation. Exact per-rank route counts are not present in the reduced communication run, so this portion is not closed.
4. **Low confidence:** a fixed TP0-only CPU or model operation. Source and cross-run evidence contradict this.

## Remaining measurement gap

The current artifacts cannot uniquely split a late arrival into expert token-load imbalance, CPU thread scheduling, host enqueue delay, and device queue delay. A targeted fresh measurement should record, for every layer and TP rank in the same low-perturbation forward:

- local expert/token counts and grouped-GEMM input shape;
- host enqueue timestamp;
- CUDA-event start/end;
- NCCL kernel start/end from CUPTI or an equivalent device activity trace;
- the shared collective completion boundary.

This measurement is for causal attribution only. It does not justify a rank-specific Frontier correction or a scalar post-MoE additive term.
