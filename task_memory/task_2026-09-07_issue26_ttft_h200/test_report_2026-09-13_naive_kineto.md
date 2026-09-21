## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the ten-warmup H200 native/naive Kineto decomposition and corrected first-formal window selection. |

# H200 native/naive Kineto diagnostic decomposition

## Execution

The run used the fixed H200 calibration case:

- cluster/quota/tag: `step_main` / `h200`;
- one 8-GPU worker, `num_gpu_blocks_override=310809`;
- Qwen3-30B-A3B dummy model, TP4/DP2/EP8, BF16, FLASHINFER, eager;
- 4096 prefill / 1024 output, uniform routing, prefix caching OFF,
  chunked prefill OFF;
- `VLLM_ALL2ALL_BACKEND=naive`;
- Frontier operator, batch, communication and scheduling instrumentation OFF;
- ten fully drained 100-request warmup replays, followed by 100 formal
  requests (1100 client rows);
- vLLM's built-in `torch.profiler` (Kineto) started after warmup drain and
  stopped when formal request `pf4096_dc1024:0` emitted its first token.

The persistent run directory is
`analysis/naive-profiler-20260913/run-02/`.  The exact client and profile
records are in `runtime/client.jsonl`, `runtime/phase_records.json`, and
`torch-profiler/*.pt.trace.json.gz`.  The pinned clean vLLM source is recorded
in `vllm_commit.txt` (`46f7b179fd3bf42b9616dc4670cba419afdb2085`).

The parser was run with:

```bash
python -m py_compile \
  task_memory/task_2026-09-07_issue26_ttft_h200/analysis/naive-profiler-20260913/parse_kineto_trace.py

python task_memory/task_2026-09-07_issue26_ttft_h200/analysis/naive-profiler-20260913/parse_kineto_trace.py \
  --trace-dir task_memory/task_2026-09-07_issue26_ttft_h200/analysis/naive-profiler-20260913/run-02/torch-profiler \
  --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/naive-profiler-20260913/kineto_breakdown_run02_prefill.json \
  --phase-records task_memory/task_2026-09-07_issue26_ttft_h200/analysis/naive-profiler-20260913/run-02/runtime/phase_records.json \
  --client-jsonl task_memory/task_2026-09-07_issue26_ttft_h200/analysis/naive-profiler-20260913/run-02/runtime/client.jsonl
```

The independent launch/correlation audit is persisted in
`analysis/naive-profiler-20260913/kineto_launch_correlation_run02.json`.

## Criteria and evidence

| Criterion | Evidence | Result |
| --- | --- | --- |
| Ten warmup drains | `phase_records.json` contains `warmup:0` through `warmup:9`, each with `completed_requests=100` and non-overlapping phase intervals | PASS |
| Formal drain and row count | `phase_records.json` records 100 formal completions; worker completion record reports `total_rows=1100` and `profile_stop_seen=true`; `client.jsonl` has 1100 rows | PASS |
| First formal identity | `first_formal_first_token` records `pf4096_dc1024:0`; parser uses this wall timestamp and the profile start to bound selection | PASS |
| GPU trace availability | 8 per-rank GPU traces plus one `async_llm` trace; all four DP0 traces contain model markers and positive kernel events | PASS |
| Target first-formal prefill selection | DP0 TP0--TP3 each select a 48-layer group with attention-marker median 172.825--177.681 us; DP1 groups are rejected as no formal prefill window | PASS with explicit non-target exclusion |
| Parser syntax and persisted output | `py_compile` passes and `kineto_breakdown_run02_prefill.json` contains four selected DP0 rows | PASS |
The DP1 traces are not part of the target batch statistic.  Their marker index
48 corresponds to a concurrent decode group (approximately 8 us attention
markers), while the first DP1 prefill group occurs later than the target
request's first-token boundary.  The parser records these four traces with
`status=NO_FORMAL_PREFILL_WINDOW`; a cross-DP marker occurrence index therefore
does not establish batch identity.

## DP0 target window breakdown

The parser selects the first complete 48-layer prefill marker window after the
profile start and before the first formal token.  `comp`, `comm`, and `mem`
are per-category interval unions of GPU activities in that window.  `idle` is
the gap between the selected window and the union of all selected GPU kernel or
memory activities.

| DP0 rank | marker span (ms) | compute union (ms) | communication union (ms) | memory union (ms) | all-activity union (ms) | idle/non-kernel (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP0 | 111.735588 | 28.881344 | 21.550196 | 3.888993 | 54.320533 | 57.415055 |
| TP1 | 108.395569 | 27.774529 | 68.066995 | 3.704426 | 99.545950 | 8.849619 |
| TP2 | 107.340562 | 27.450432 | 69.627183 | 3.630506 | 100.708120 | 6.632441 |
| TP3 | 106.682713 | 27.519764 | 67.907767 | 3.636302 | 99.063832 | 7.618881 |

Across DP0 TP0--TP3, marker-window span statistics are:

```text
median = 107.868065 ms
P90    = 110.733582 ms
max    = 111.735588 ms
spread = 5.052875 ms
```

The accepted clean native/naive batch-only references are 78.118782043 ms and
79.307357788 ms.  The Kineto marker-window result is therefore a diagnostic
106.68--111.74 ms window, not a clean latency result.  It is not valid to
subtract the clean reference from these profiled values to fit a model.

## Launch/correlation evidence

The marker-window communication union differs sharply by rank, but the launch
ownership audit finds the same logical operation population on all four ranks:

| TP rank | CPU CUDA launches | matched GPU kernels | matched NCCL all-reduce kernels | all-reduce duration sum (ms) | launch-to-kernel-start median / P90 / max (us) | launch-owned GPU span (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP0 | 3064 | 3064 | 144 | 12.972966 | 180.773 / 194.274 / 199.808 | 111.937212 |
| TP1 | 3064 | 3064 | 144 | 61.101563 | 1611.461 / 2691.641 / 3404.144 | 112.115226 |
| TP2 | 3064 | 3064 | 144 | 63.028059 | 2255.400 / 3651.674 / 4452.682 | 112.137711 |
| TP3 | 3064 | 3064 | 144 | 61.313837 | 2055.089 / 4156.219 / 4982.619 | 112.018683 |

The all-reduce count is identical at 144 per rank.  The communication duration
difference is consequently not evidence that TP0 skipped layers or had a
smaller payload.  In the marker slice, many TP1--TP3 NCCL kernels include
longer launch-to-start and peer/ring wait behavior, while TP0's corresponding
kernels are short.  When GPU activities are assigned by their CPU launch
correlations, their tails extend beyond the CPU marker end and all four ranks
have approximately the same 112 ms execution envelope.

The trace also exposes the native naive protocol's activity shape.  The target
window contains repeated
`ncclDevKernel_Broadcast_RING_LL` and
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` kernels.  This is consistent with
the vLLM source's `NaiveAll2AllManager`:

- `vllm/distributed/device_communicators/all2all.py:28-55` performs two DP
  multicast sequences (hidden states and router logits), each iterating over
  DP ranks and calling `broadcast`;
- `all2all.py:57-66` performs a DP `all_reduce` for combine;
- `vllm/model_executor/layers/fused_moe/layer.py:1852-1861` then invokes the
  EP combine and the independent tensor-parallel reduction when
  `reduce_results` is enabled.

Thus the observed NCCL count is the same logical model work on every TP rank,
while rank-local kernel intervals include different queue and collective
completion placement.

## Interpretation and limits

The measured evidence supports the following bounded conclusion:

1. There is no observed hidden 20 ms model operation that exists only on TP0 or
   that is missing from the other ranks. All four target ranks have the same
   launch and all-reduce counts, and their launch-owned GPU envelopes are
   nearly equal.
2. The apparent TP0 communication deficit in the CPU-marker slice is a window
   and queue-alignment effect. A rank can record short NCCL kernel durations
   when its peer/stream work has already aligned, while another rank's NCCL
   kernel spends more time in launch/collective completion waiting. The latter
   does not create a second serial term to sum across ranks.
3. The 30--33 ms increase over the accepted clean reference cannot be assigned
   entirely to communication. Kineto itself, marker capture, queued device
   work, and boundary choice contribute to the diagnostic envelope. A clean
   CUPTI/nsys run without profiler-induced activity is required before any
   pure-kernel/idle decomposition can be used quantitatively.

The run does not close the CUDA gate, prove clean communication reconciliation,
or authorize a Frontier predictor/protocol correction. DP1 concurrent decode
traces and the `async_llm` trace are retained for provenance but excluded from
the formal first-prefill breakdown.
