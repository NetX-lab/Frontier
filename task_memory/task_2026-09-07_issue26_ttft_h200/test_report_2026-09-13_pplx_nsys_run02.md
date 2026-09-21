## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Persisted and audited the H200 PPLX Nsight run-02 artifacts, including process-local interval unions and protocol-kernel aggregates. |

# H200 PPLX Nsight diagnostic — run 02

## Outcome

The PPLX Nsight run completed the required data collection after ten drained
warmup replays. The persisted RJob is recorded as terminal `Failed`, but the
client, first-formal capture markers, `.nsys-rep`, and all postprocess CSVs are
present. The failure is treated as a platform/execution-status failure; the
artifact gate is evaluated separately and passes its data predicates.

The capture does **not** provide a clean PPLX batch CUDA span. The process-local
activity envelopes with an observed stop are `675.202--693.996 ms` (their API
start-to-stop windows are `675.986--694.920 ms`), while the accepted
native/naive clean reference remains `78.118782043--79.307357788 ms`.
The PPLX boundary-only replay remains a separate `542.338012695 ms` event
result. Nsight profiling, the PPLX dispatch/combine protocol, queued device
work, and the `cudaProfilerStop` drain all affect the diagnostic windows.
These values are therefore suitable for protocol-population and queue evidence
only; they must not be used for Frontier predictor fitting, communication
correction, CPU add-on, or clean/diagnostic span reconciliation.

## Execution

The run used the frozen H200 case:

- RJob: `yc26-h200-pplx-nsys-profiler-20260913-02`;
- cluster/quota: `step_main`, GPU tag `h200`;
- 8 H200 GPUs, TP4 / DP2 / PP1 / EP8;
- Qwen3-30B-A3B-Instruct-2507 dummy weights;
- BF16, `FLASHINFER`, eager execution;
- uniform routing, prefix caching OFF, chunked prefill OFF;
- `num_gpu_blocks_override=310809`;
- 4096-prefill / 1024-output;
- PPLX all-to-all backend (`VLLM_ALL2ALL_BACKEND=pplx`);
- vLLM diagnostic source `/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt`, commit `3e2fc3159b8145ba632c378522e361ff2430b39f`;
- `VLLM_FRONTIER_INSTRUMENTATION=0`, with Frontier per-op, batch, scheduler,
  routing, CUDA-event, and record-function loggers disabled;
- Nsight Systems 2025.6.3 target/host runtime, CUDA and NCCL tracing,
  `cudaProfilerApi` capture, no CPU sampling;
- ten complete drained warmup replays of 100 requests, then 100 formal
  requests.

The worker is
`tests/e2e/issue26_h200_nsys_profiler_worker.sh`. Its capture hook starts
`cudaProfilerStart` immediately before the selected first-formal model forward
and invokes `cudaProfilerStop` immediately after that forward returns. The
PPLX runtime was supplied through the verified overlay
`/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03` and NCCL/NVSHMEM library
paths under
`/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/`.

The selected run directory is:

```text
task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-nsys-profiler-20260913-run02/
```

The postprocess directory is:

```text
task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-nsys-profiler-20260913-run02-postprocess/
```

Nsight reports were regenerated from the persisted `.nsys-rep` with the
version-matched host importer. The command shape was:

```bash
POST=task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-nsys-profiler-20260913-run02-postprocess
NSYS_ROOT=/data/ycfeng/tmp/issue26-nsys-runtime-20260913-01-host
export LD_LIBRARY_PATH="$NSYS_ROOT/host-linux-x64:$NSYS_ROOT/target-linux-x64:${LD_LIBRARY_PATH:-}"
NSYS="$NSYS_ROOT/target-linux-x64/nsys"
for REPORT in cuda_gpu_trace cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum cuda_api_trace; do
  "$NSYS" stats --report "$REPORT" --format csv \
    --output "$POST/stats" --force-overwrite=true \
    "$POST/first_formal.nsys-rep" \
    > "$POST/stats_${REPORT}.log" 2>&1
done
```

The process-local interval output was produced from the persisted SQLite
trace with
`analysis/pplx-nsys-profiler-20260913-run02-postprocess/analyze_pplx_per_process.py`.
It groups CUPTI kernel and memcpy intervals by process, clips them to each
process's `cuProfilerStart` through `cudaProfilerStop` API-start window, and
computes interval unions. PPLX dispatch/combine/triton names were explicitly
classified into the communication/protocol lane for this table because they
are the PPLX all-to-all dispatch/combine path.

## Data and identity gate

The independent gate artifact is
`analysis/pplx-nsys-profiler-20260913-run02-postprocess/run02_gate_audit.json`.
The observed predicates are:

| Criterion | Observed | Result |
| --- | ---: | --- |
| Complete warmup replays | 10 × 100 requests | PASS |
| Warmup client rows | 1000 | PASS |
| Formal requests | 100 | PASS |
| Total client rows | 1100 | PASS |
| Formal prompt/completion | 4096 / 1024 for every formal row | PASS |
| First formal request | `pf4096_dc1024:0` | PASS |
| `phase_records.json` | 10 warmup phases + 1 formal phase, each 100 complete | PASS |
| Capture start marker | present | PASS |
| Capture stop marker | present | PASS |
| Nsight report | `first_formal.nsys-rep`, 12,496,496 bytes | PASS |
| Nsight stats | GPU trace, kernel sum, memory sum, API sum and API trace | PASS |
| RJob terminal phase | `Failed` | FAIL (platform status; artifact data present) |

The persistent run does not include a narrower platform failure line for the
terminal `Failed` status. The status is retained as a failure and is not
rewritten to `Succeeded`; the data predicates above are independently
verifiable from the saved client, phase, marker and trace artifacts.

## Capture window and API evidence

The trace-domain control markers are:

| Marker | Trace-domain time |
| --- | ---: |
| profiler start | 22.109715 ms |
| profiler stop API invocation | 824.758035 ms |
| control interval | 802.648320 ms |

`cuda_gpu_trace.csv` contains 284,314 input rows; 276,436 rows remain after
clipping and no event is unknown to the postprocess classifier. The trace
contains four H200 device streams. This control interval is a profiler window,
not a clean batch boundary.

The Nsight CUDA API aggregate provides a direct warning about the stop boundary:

| API | Calls | Inclusive host API time |
| --- | ---: | ---: |
| `cudaProfilerStop` | 3 | 112.881237 ms |
| `cudaStreamSynchronize` | 4 | 78.486681 ms |
| `cudaLaunchKernel` | 230,412 | 1,259.354948 ms |
| `cudaLaunchCooperativeKernel` | 18,385 | 143.396922 ms |
| `cudaMemcpyAsync` | 21,897 | 135.645395 ms |

These API totals are inclusive across captured processes. In particular,
`cudaProfilerStop` and stream synchronization can wait for queued work and
must not be added to a single-rank batch span.

## Process-local diagnostic decomposition

Only three of four captured processes have an observed stop API. The fourth
process (PID 1172) started at 22.112817 ms but has no stop, so no bounded
category row is generated for it. The persisted result is marked
`PARTIAL_STOP_COVERAGE` in
`pplx_per_process_classified.json`.

The process labels in the saved server log are only `VLLM::Worker_DP`; a
reliable PID-to-DP/TP mapping was not persisted. PID values below are therefore
process identities, not TP rank claims. Each value is an interval union within
that process's own profiler API window. `Idle` is the gap inside the activity
envelope after subtracting the all-kernel union.

| Process | API window (ms) | Compute union (ms) | Communication/protocol union (ms) | Memory union (ms) | All-kernel union (ms) | Activity envelope (ms) | Idle (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PID 1168 | 694.920 | 63.620 | 215.267 | 21.094 | 299.982 | 693.996 | 394.014 |
| PID 1170 | 682.208 | 61.872 | 551.502 | 20.733 | 634.107 | 681.307 | 47.200 |
| PID 1173 | 675.986 | 61.082 | 555.737 | 20.494 | 637.313 | 675.202 | 37.889 |
| PID 1172 | unavailable (stop missing) | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

Compute and memory unions are close across the three bounded processes,
whereas protocol union and idle time trade off strongly. This is evidence that
PPLX queued protocol work is placed differently relative to each process's
capture boundary. It does not show a missing TP operation and does not permit
summing the three process communication values as serial latency.

## PPLX protocol kernel population

The kernel-sum report is an inclusive aggregate across captured devices and
processes. The largest PPLX protocol entries are:

| Kernel family | Instances | Inclusive total (ms) | Median (us) | Maximum (ms) |
| --- | ---: | ---: | ---: | ---: |
| `dispatchKernel` (main variant) | 6,128 | 1,918.894 | 318.609 | 1.205 |
| `dispatchKernel` (alternate variant) | 6,128 | 67.869 | 12.560 | 0.016 |
| `batched_triton_kernel` | 12,258 | 1,262.891 | 100.864 | 0.154 |
| `combineKernel` | 6,129 | 909.533 | 160.672 | 0.300 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 196 | 180.649 | 326.529 | 127.002 |

For context, `act_and_mul_kernel` contributes 207.558 ms across 6,129
instances. The protocol totals are inclusive and overlap in time across
streams, devices and processes. They cannot be converted into a serial
first-forward duration by addition. The large number of dispatch/combine and
batched-triton instances does, however, identify the PPLX fused protocol as a
material source of captured work that is absent from Frontier's ideal EP
abstraction.

## Interpretation and limits

1. **What the run establishes.** The PPLX path is executable with the verified
   Python and NCCL/NVSHMEM overlays, and the first-formal identity/capture gate
   is complete. PPLX dispatch/combine kernels dominate the captured kernel
   population. The per-process table shows similar compute/memory work but a
   large communication-versus-idle redistribution.

2. **Why this is not a clean span.** The bounded Nsight windows are roughly
   `8.5--8.9×` the accepted 78--79 ms native clean span. The profiler changes
   host launch and queue placement, while `cudaProfilerStop` and stream
   synchronization drain queued work. The process 1172 stop is missing, and
   process IDs cannot be mapped to TP ranks. These limits prevent a clean
   PPLX-versus-native latency comparison.

3. **What cannot be inferred.** Inclusive kernel totals must not be summed over
   ranks. The tables do not isolate pure kernel time from host launch delay,
   stream wait, first-use/JIT cost, or PPLX tuning fallback. They do not prove
   that one specific 20 ms collective is missing from Frontier.

4. **Calibration consequence.** Keep the accepted native clean reference
   (`78.118782043--79.307357788 ms`) and the PPLX boundary-only value
   (`542.338012695 ms`) separate from this Nsight diagnostic. No Frontier
   predictor, communication backend, operator accounting, CPU overhead or
   clean/diagnostic reconciliation is changed from this run.

## Persisted artifacts

- Run data: `analysis/pplx-nsys-profiler-20260913-run02/`;
- postprocess report: `analysis/pplx-nsys-profiler-20260913-run02-postprocess/`;
- per-process classification:
  `pplx_per_process_classified.json`;
- API-window decomposition:
  `nsys_breakdown_api_window.json`;
- machine-readable gate:
  `run02_gate_audit.json`;
- reproducibility hashes:
  `artifact_sha256sums.txt`;
- process-window helper:
  `analyze_pplx_per_process.py`.
