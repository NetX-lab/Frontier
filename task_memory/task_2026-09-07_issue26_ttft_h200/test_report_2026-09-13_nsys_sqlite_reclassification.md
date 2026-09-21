## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Corrected Nsight SQLite communication classification and reprocessed run04. |

# Nsight SQLite category reclassification

The rank-aware SQLite parser in `tests/e2e/issue26_nsys_sqlite_breakdown.py`
previously matched the generic word `collective`. FlashInfer's attention kernel
name contains `CollectiveEpilogue`; that string identifies an attention
epilogue implementation and does not establish a communication collective.
The generic match was removed. Communication remains classified by explicit
NCCL and collective operation names such as `nccl`, `all-reduce`,
`broadcast`, `cross_device_reduce`, and point-to-point names.

## Verification

Commands:

```bash
python -m py_compile tests/e2e/issue26_nsys_sqlite_breakdown.py
git diff --check
python tests/e2e/issue26_nsys_sqlite_breakdown.py \
  --sqlite task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-cuda-only-20260913-run04-postprocess/first_formal.sqlite \
  --api-trace task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-cuda-only-20260913-run04-postprocess/stats_api_trace_cuda_api_trace.csv \
  --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-cuda-only-20260913-run04-postprocess/nsys_sqlite_rank_breakdown_reclassified.json
```

All commands passed. The code change is committed as `8d5dfc5f`.

## Reprocessed run04 result

The parser still uses per-process `cuProfilerStart` to `cudaProfilerStop`
windows and reports interval unions. It is diagnostic Nsight evidence; it
does not replace the accepted clean native 78--79 ms boundary.

| Nsight device | Compute pure union (ms) | Communication pure union (ms) | Memory pure union (ms) | Activity envelope (ms) | Idle/non-activity (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 35.519020 | 17.580257 | 2.272524 | 105.741408 | 50.369607 |
| 1 | 33.800442 | 60.052668 | 2.172709 | 101.189260 | 5.163441 |
| 2 | 32.840313 | 58.671559 | 2.245958 | 98.432219 | 4.674389 |
| 3 | 32.726633 | 61.517546 | 2.114272 | 100.694972 | 4.336521 |

Removing the false generic `collective` match moves approximately 4.5 ms of
FlashInfer attention epilogue work from communication to compute on each rank.
The per-rank communication/idle redistribution remains: device 0 has a short
communication union and a large idle interval, while devices 1--3 have roughly
58--62 ms communication and 4--5 ms idle. These are interval placements inside
a profiler window and cannot be summed across ranks.

The activity envelopes remain approximately 98--106 ms, above the accepted
clean reference. The difference is therefore a profiler/capture-boundary
effect or queued work and remains diagnostic. No Frontier predictor,
communication backend, production accounting, or clean/diagnostic
reconciliation was changed.

