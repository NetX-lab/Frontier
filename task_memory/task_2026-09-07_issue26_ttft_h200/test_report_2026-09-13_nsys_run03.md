## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded native H200 Nsight run-03 gate, post-client shell failure, and trace-domain kernel decomposition. |

# Native H200 Nsight run-03

## Execution

The run used the existing H200 `step_main + h200` recipe:

- 8 H200 GPUs, 64 CPUs, 409600 MiB;
- `num_gpu_blocks_override=310809`;
- Qwen3-30B-A3B-Instruct-2507 dummy weights;
- TP4 / DP2 / PP1 / EP8, BF16, FLASHINFER, eager execution;
- uniform routing, prefix caching OFF, chunked prefill OFF;
- 4096-prefill / 1024-output;
- vLLM source `/data/ycfeng/tmp/vLLM-BS`, commit `0f34fb271fd66d7dd84201ebdd4722781f829390`;
- `VLLM_FRONTIER_INSTRUMENTATION=0` and no Frontier per-op, batch, routing, or record-function logging;
- Nsight Systems 2025.6.3.541 with CUDA/NCCL tracing and `cudaProfilerApi` capture;
- ten drained 100-request warmup replays followed by 100 formal requests.

The reproducible launch is:

```bash
bash task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-profiler-20260913-run03/launch.sh
```

The worker completed the client and capture phases. Because the worker-created `nsys/` directory is root-owned, summary reports were generated in the user-owned postprocess directory with:

```bash
POST=task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-profiler-20260913-run03-postprocess
NSYS_ROOT=/data/ycfeng/tmp/issue26-nsys-runtime-20260913-01-host
export LD_LIBRARY_PATH="$NSYS_ROOT/host-linux-x64:$NSYS_ROOT/target-linux-x64:${LD_LIBRARY_PATH:-}"
NSYS="$NSYS_ROOT/target-linux-x64/nsys"
cp task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-profiler-20260913-run03/nsys/first_formal.nsys-rep "$POST/first_formal.nsys-rep"
for REPORT in cuda_gpu_trace cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum cuda_api_trace; do
  "$NSYS" stats --report "$REPORT" --format csv --force-export=true \
    --output "$POST/stats" --force-overwrite=true \
    "$POST/first_formal.nsys-rep" > "$POST/stats_${REPORT}.log" 2>&1
 done
```

## Gate result

The data gate passed:

- `runtime/client.jsonl`: 1100 rows;
- `runtime/phase_records.json`: ten `warmup:*` phases with 100 completed requests each and one formal phase with 100 completed requests;
- every warmup phase completed before the next phase began;
- first formal request identity: `pf4096_dc1024:0`;
- `nsys-control/start` and `nsys-control/stop` are present;
- `nsys/first_formal.nsys-rep` exists and is 1,994,203 bytes;
- `runtime/server.log` contains `Capture range started in the application` and `Capture range ended in the application`.

The RJob phase is `Failed`, but the failure occurred after these artifacts were written. The captured worker log is:

```text
tests/e2e/issue26_h200_nsys_profiler_worker.sh: line 152: ttp://127.0.0.1:8000: No such file or directory
```

`phase_records.json` proves that the client itself completed the ten warmups and the formal 100-request drain. The failure is therefore a post-client shell/worker status defect; it is recorded as FAIL for the platform execution status and does not invalidate the already-written capture artifact. A future rerun must repair the shell invocation before claiming a `Succeeded` RJob.

## Trace-domain decomposition

`cuda_gpu_trace.csv` contains 24,052 GPU activity rows on four H200 devices. Nsight timestamps are process-relative (the first activity is about 101 ms), while the filesystem markers contain wall-clock `time_ns`; they cannot be clipped directly without a clock-domain conversion. The trace itself is bounded by the application profiler API. The postprocess report therefore uses the trace-domain interval from the earliest `cuProfilerStart` call (`100,157,909 ns`) to the earliest `cudaProfilerStop` invocation (`200,984,879 ns`) and records the full activity envelope separately.

The report is:

```text
task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-profiler-20260913-run03-postprocess/nsys_breakdown_api_window.json
```

The per-device table below reports interval unions. `compute`, `communication`, and `memory` are pure unions within one device; they are not sums of kernel durations and must not be summed across TP ranks. `idle` is the gap inside that device's activity envelope. Classification uses explicit CUDA memcpy/memset rules and operation-name heuristics for communication/copy kernels.

| Nsight device | Compute union (ms) | Communication union (ms) | Memory union (ms) | All-activity union (ms) | Activity envelope (ms) | Idle inside envelope (ms) | Category overlap (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA H200 (0) | 23.566 | 80.553 | 2.027 | 98.693 | 99.514 | 0.821 | 7.452 |
| NVIDIA H200 (1) | 23.423 | 74.954 | 2.118 | 93.041 | 93.839 | 0.798 | 7.454 |
| NVIDIA H200 (2) | 23.017 | 76.497 | 2.165 | 97.394 | 99.504 | 2.110 | 4.286 |
| NVIDIA H200 (3) | 37.433 | 50.414 | 2.606 | 67.962 | 71.217 | 3.255 | 22.491 |

For reference, the un-clipped activity envelopes are 132.731, 137.046, 142.650 and 104.266 ms for devices 0--3. The difference is caused by `cudaProfilerStop` draining queued work after the model-forward boundary and by profiler-induced scheduling overhead.

The operation-level summary gives the concrete communication population:

| Kernel | Total instances | Total duration across devices (ms) | Median (us) | Maximum (ms) |
| --- | ---: | ---: | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 756 | 294.974 | 97.232 | 28.344 |
| `vllm::cross_device_reduce_1stage<bf16,4>` | 354 | 155.411 | 5.568 | 28.460 |
| `ncclDevKernel_Broadcast_RING_LL` | 1470 | 126.579 | 10.368 | 0.721 |

The corresponding compute kernels are `fused_moe_kernel` (736 instances, 79.427 ms total) and `act_and_mul_kernel` (368 instances, 53.071 ms total). Device-to-device memcpy activity is 1,457 calls and 3.138 ms total; host-to-device memcpy is 56 calls and 0.143 ms total.

## Interpretation and limits

This run is valuable evidence about the population and shape of the native communication path: 189 all-reduce-like kernels per device (756 total), approximately 88 cross-device-reduce kernels per device, and individual collective kernels with 28 ms maxima. It supports communication/queue work as a material residual candidate relative to Frontier's ideal EP/AR accounting.

It does **not** produce a valid 78--79 ms clean-span decomposition. The accepted clean native references remain 78.118782043--79.307357788 ms. The Nsight capture's per-device activity envelopes are 71--100 ms before the stop-drain tail and 104--143 ms including it. Thus the table is a low-perturbation diagnostic decomposition, not a replacement clean metric. The report has no reliable DP/TP mapping beyond Nsight device identity, and the communication values must not be added across devices. No Frontier predictor, communication backend, operator mapping, CPU accounting, or clean/diagnostic reconciliation was changed from this result.
