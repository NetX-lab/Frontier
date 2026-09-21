## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded independent Nsight stats export and profiler-window evidence for native run-03. |

# Native Nsight run-03 manual stats export

## Execution

The H200 native/naive Nsight RJob `yc26-h200-nsys-profiler-20260913-03` used the standard `step_main + h200 + num_gpu_blocks_override=310809` recipe and the gated source `vLLM-BS@0f34fb271fd66d7dd84201ebdd4722781f829390`. It completed 10 drained 100-request warmups and 100 formal requests (1100 client rows), wrote the first-formal control markers for `pf4096_dc1024:0`, and produced:

```text
task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-profiler-20260913-run03/nsys/first_formal.nsys-rep
```

The worker later failed while invoking the client command (`ttp://127.0.0.1:8000` shell error) before its automatic stats export. The report was copied to a writable temporary directory and processed with Nsight Systems 2025.6.3 host tools:

```text
/data/ycfeng/tmp/issue26-nsys-run03-manual-stats/
```

## Criteria

- Preserve the raw `.nsys-rep` unchanged.
- Verify the report can be imported with the matching host `QdstrmImporter`.
- Export `cuda_gpu_trace`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, and `cuda_api_sum`.
- Treat all interval accounting as diagnostic; clean 78–79 ms batch span remains the acceptance metric.

## Evidence

PASS for import/export:

- Nsight version: `2025.6.3.541-256337736014v0`.
- `.nsys-rep` size: `1,994,203` bytes.
- SQLite import succeeded; SQLite size `5,619,712` bytes.
- `stats_cuda_gpu_trace.csv`: `6,773,850` bytes, 24,053 rows.
- `cuda_gpu_kern_sum.csv`: top totals include NCCL all-reduce `294,973,604 ns` (756 instances), `cross_device_reduce_1stage` `155,410,645 ns` (354 instances), NCCL broadcast `126,579,124 ns` (1,470 instances), and `fused_moe_kernel` `79,426,792 ns` (736 instances).
- `cuda_gpu_mem_time_sum.csv`: device-to-device memcpy `3,137,740 ns`; host-to-device memcpy `143,425 ns`.
- `cuda_api_sum.csv`: `cudaProfilerStop` 3 calls totaling `119,128,366 ns`; `cudaStreamSynchronize` 4 calls totaling `110,502,444 ns`; `cudaLaunchKernel` 18,462 calls totaling `87,168,837 ns`.

The four `cuProfilerStart` calls occurred at trace times `100,157,909`, `100,234,908`, `106,340,588`, and `128,939,568 ns`. Three `cudaProfilerStop` calls were observed at `200,984,879`, `205,516,630`, and `206,750,833 ns` with end times near `244 ms`; one expected stop was absent from the runtime table. A synthetic global trace window `[100,157,909, 244,236,792)` is therefore retained only to inspect category unions:

| Device | Compute union | Communication union | Memory union | All activity union | Idle remainder |
| --- | ---: | ---: | ---: | ---: | ---: |
| H200 (0) | 33.319 ms | 101.732 ms | 2.910 ms | 127.424 ms | 5.307 ms |
| H200 (1) | 33.088 ms | 96.178 ms | 3.034 ms | 121.764 ms | 15.283 ms |
| H200 (2) | 33.004 ms | 97.627 ms | 3.277 ms | 126.127 ms | 16.523 ms |
| H200 (3) | 52.987 ms | 71.420 ms | 3.722 ms | 96.105 ms | 8.161 ms |

Classification uses operation-name heuristics and the synthetic API-derived trace window; device labels do not prove DP/TP identity. The global window is approximately 144.079 ms and is not the clean 78–79 ms batch boundary. The absence of one stop record and profiler/queue effects prevent using this table for predictor fitting or Frontier correction.

## Limitations

The platform job is recorded as harness-failed after its core artifacts were produced. A follow-up run with the quoted URL argument is available through worker commit `54d4538b` if a clean platform-pass is required. No Frontier production code or communication model was changed from this evidence.
