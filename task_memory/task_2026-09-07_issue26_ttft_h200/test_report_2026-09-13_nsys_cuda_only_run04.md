## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the completed H200 native/naive CUDA-only Nsight run and rank-aware decomposition. |

# H200 native/naive CUDA-only Nsight run04

## Execution

The run used the approved H200 standard recipe:

- RJob: `yc26-h200-nsys-cuda-only-20260913-04`
- Platform status: `Succeeded`
- Cluster: `step_main`, positive tag `h200`
- Allocation: 8 GPUs, 64 CPUs, 409600 MiB
- Model: Qwen3-30B-A3B-Instruct-2507 dummy weights
- Parallelism: TP4 / DP2 / PP1 / EP8
- BF16, eager execution, FLASHINFER attention
- `num_gpu_blocks_override=310809`, block size 16
- Uniform MoE routing; prefix caching and chunked prefill disabled
- Workload: 4096 prefill tokens / 1024 output tokens
- vLLM source: `/data/ycfeng/tmp/vLLM-BS@0f34fb271fd66d7dd84201ebdd4722781f829390`
- `VLLM_ALL2ALL_BACKEND=naive`
- `VLLM_FRONTIER_INSTRUMENTATION=0`; all Frontier batch/operator/scheduler/routing loggers unset

The client completed ten drained warmup replays of 100 requests each and then
100 formal requests. `runtime/client.jsonl` contains 1100 rows. The first
formal request is `pf4096_dc1024:0`; `phase_records.json` contains
`warmup:0` through `warmup:9` and the formal phase. The Nsight report and
SQLite export are present:

```text
analysis/nsys-cuda-only-20260913-run04/nsys/first_formal.nsys-rep
analysis/nsys-cuda-only-20260913-run04-postprocess/first_formal.sqlite
```

## Capture and analysis

The worker used Nsight Systems with CUDA activity tracing only:

```text
--capture-range=cudaProfilerApi
--capture-range-end=stop
--trace=cuda
--sample=none
--cpuctxsw=none
--cuda-memory-usage=false
--cuda-event-trace=false
--flush-on-cudaprofilerstop=false
--cuda-trace-scope=process-tree
```

The independent Issue 26 hook called `cudaProfilerStart` immediately before
the selected first-formal model forward and `cudaProfilerStop` immediately
after that forward returned. No per-operator Frontier instrumentation was
enabled. Communication classification in the SQLite parser is based on CUDA
kernel names (`ncclDevKernel_*`, cross-device and broadcast/all-reduce names);
it does not use NCCL API tracing.

The raw `cuda_gpu_trace` timestamps are in Nsight's session/trace clock, while
the control files contain host wall-clock epoch nanoseconds. In this capture,
the SQLite session metadata reports
`TARGET_INFO_SESSION_START_TIME.utcEpochNs=1789259786290725865` and
`systemClockNs=3935027087492773`. Treating the raw CSV and host markers as the
same domain (`timestamp_offset_ns=0`) produces an invalid alignment and a
zero-row/incorrect window. The session-offset postprocess is retained at
`analysis/nsys-cuda-only-20260913-run04/nsys_breakdown_session_offset.json`;
the rank-aware parser instead uses the CUDA API trace's process-local
`cuProfilerStart`/`cudaProfilerStop` timestamps, which are already in the
same Nsight trace domain as the SQLite kernel records.

The rank-aware parser was run as follows:

```bash
python tests/e2e/issue26_nsys_sqlite_breakdown.py \
  --sqlite \
  task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-cuda-only-20260913-run04-postprocess/first_formal.sqlite \
  --api-trace \
  task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-cuda-only-20260913-run04-postprocess/stats_api_trace_cuda_api_trace.csv \
  --output \
  task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-cuda-only-20260913-run04-postprocess/nsys_sqlite_rank_breakdown.json
```

The parser selected the four DP0 processes from profiler-start membership and
mapped each process to its CUDA device through the CUDA-context table.
The persisted SQLite rows expose process/globalPid and device identity, but do
not carry a durable TP rank label; the device-0..3 ordering is therefore an
inferred DP0 TP0..TP3 correspondence and is reported with that limitation.

## Criteria and evidence

The data and artifact gate passed:

| Criterion | Result |
| --- | --- |
| Ten complete warmup replays | PASS |
| 100 requests per warmup and full drain | PASS |
| 100 formal requests | PASS |
| Client rows | PASS, 1100 |
| First formal request identity | PASS, `pf4096_dc1024:0` |
| Nsight `.nsys-rep` | PASS |
| SQLite export and parser | PASS |
| DP0 TP0--TP3 process/device rows | PASS, with one bounded stop fallback |

The accepted clean native reference remains `78.118782043--79.307357788 ms`.
The rank-aware Nsight windows and category unions are:

| CUDA device / inferred DP0 process | Window (ms) | Compute union (ms) | Communication union (ms) | Memory union (ms) | All-kernel union (ms) | Idle/non-activity (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / PID 1103 | 106.824398 | 35.519020 | 17.580257 | 2.272524 | 55.371801 | 50.369607 |
| 1 / PID 1105 | 102.003857 | 33.800442 | 60.052668 | 2.172709 | 96.025819 | 5.163441 |
| 2 / PID 1107 | 99.196101 | 32.840313 | 58.671559 | 2.245958 | 93.757830 | 4.674389 |
| 3 / PID 1108 | 101.641481 | 32.726633 | 61.517546 | 2.114272 | 96.358451 | 4.336521 |

Across the four selected device/process rows, the aggregate diagnostics are:

| Quantity | Median (ms) | P90 (ms) | Rank max (ms) | Rank spread (ms) |
| --- | ---: | ---: | ---: | ---: |
| Bounded profiler window | 101.822669 | 105.378236 | 106.824398 | 7.628297 |
| Compute pure-kernel union | 33.320378 | 35.003447 | 35.519020 | 2.792387 |
| Communication pure-kernel union | 59.362114 | 61.078083 | 61.517546 | 43.937289 |
| Memory-kernel union | 2.209333 | 2.264554 | 2.272524 | 0.158252 |
| All-kernel/activity union | 94.891825 | 96.258661 | 96.358451 | 40.986650 |
| Idle/non-activity remainder | 4.918915 | 36.807757 | 50.369607 | 46.033086 |

These category values are interval unions within each device window. They are
not a serial sum: communication, compute, and memory may overlap, and the
communication/idle spread is largely a placement redistribution. The table is
therefore a device-local diagnostic decomposition suitable for identifying
candidate residuals, not a replacement for the clean batch span.

## Marker-time anomaly

The external client markers span `11,925.042 ms` from formal-start marker to
the first formal first-token marker. The client record for
`pf4096_dc1024:0` reports `client_ttft_ms=11832.188371`. Nsight's CUDA API
trace, however, records the process-local `cudaProfilerStart`/`cudaProfilerStop`
window at approximately `102.0--106.8 ms`; the capture artifacts contain no
11.9-second GPU activity interval. This means the host/client marker path and
the profiler trace domain are not interchangeable. The long host interval is
retained as an execution anomaly and is excluded from the CUDA category
breakdown; it is a strong reason the run cannot be called a clean 79 ms
measurement.

The diagnostic window range is therefore `99.196101--106.824398 ms`, above
the accepted clean range. Device 0 shows approximately 17.58 ms of classified
communication and 50.37 ms of idle/non-activity, while devices 1--3 show
approximately 58--62 ms of communication and 4--5 ms of idle. This is a
rank-local redistribution in the captured trace; it is not valid to sum these
communication values across ranks.

The API trace recorded `cudaProfilerStart` for PIDs 1103, 1105, 1107 and 1108.
It recorded stops for PIDs 1103, 1105 and 1107; PID 1108 had no stop row. The
parser uses the earliest observed process stop (`192.079010 ms`) as a bounded
fallback for PID 1108. This missing stop is an artifact limitation, not
evidence that TP3 completed at that time.

## Interpretation and limits

The run establishes a complete low-overhead CUDA activity artifact and shows
that native/naive execution contains a large NCCL/cross-device kernel
population. It does not provide a clean 79 ms decomposition: Nsight capture,
CUDA profiler control and stop handling perturb scheduling and can drain
queued work. The 99--107 ms windows must therefore remain diagnostic.

The category values are per-device interval unions. They cannot be added over
TP ranks, and the kernel-name classifier is heuristic. The missing stop row
and earliest-stop fallback prevent a strict four-rank common-window claim.
Consequently this run does not authorize a Frontier predictor/backend change,
an inferred fixed 20 ms communication operator, CPU overhead addition, or
clean/diagnostic span reconciliation. The clean `78--79 ms` reference must
continue to come from the standard uninstrumented replay.
