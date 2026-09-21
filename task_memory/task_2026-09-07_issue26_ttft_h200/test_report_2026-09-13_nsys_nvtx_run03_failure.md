## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the completed-workload NVTX process-tree capture failure for run03. |

# H200 NVTX Nsight run03 failure

## Execution

- RJob: `yc26-h200-nsys-nvtx-20260913-03`
- Cluster and selector: `step_main + h200`
- Allocation: 8 GPUs, 64 CPUs, 409600 MiB
- Workload: 4096-prefill / 1024-output, TP4 / DP2 / PP1 / EP8, BF16,
  eager, FlashInfer, uniform routing, prefix caching OFF, chunked prefill OFF
- KV budget: `num_gpu_blocks_override=310809`
- vLLM source: `/data/ycfeng/tmp/vLLM-BS@0f34fb271fd66d7dd84201ebdd4722781f829390`
- Backend: `VLLM_ALL2ALL_BACKEND=naive`
- Worker: `tests/e2e/issue26_h200_nsys_nvtx_worker.sh`
- Run directory:
  `task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-nvtx-20260913-run03/`

Nsight was launched with the process-tree NVTX trigger:

```text
--capture-range=nvtx
--nvtx-capture=issue26_root_formal_forward
--capture-range-end=stop
--trace=cuda,nvtx
--cuda-trace-scope=process-tree
```

The target process-tree watcher pushed `issue26_root_formal_forward` after the
formal start marker and popped it after the first-formal first-token marker.
The vLLM process-local shim also loaded successfully, but Frontier per-op and
diagnostic instrumentation remained disabled.

## Criteria and evidence

| Criterion | Result | Evidence |
| --- | --- | --- |
| Ten complete warmup replays | PASS | `runtime/client.log`, `warmup:0` through `warmup:9` |
| 100 requests per warmup | PASS | `runtime/client.log` and phase records |
| 100 formal requests and full drain | PASS | `runtime/client.log`, formal record |
| Total client rows | PASS, 1100 | `runtime/client.jsonl` |
| First formal identity | PASS, `pf4096_dc1024:0` | `runtime/client.log` |
| Formal start/stop markers | PASS | `nsys-control/start`, `nsys-control/stop` |
| Process-tree watcher | PASS | `nvtx-watcher-status.json` |
| Nsight report | **FAIL** | `nsys/` contains only `version.txt` |

The client log ends with:

```text
{"phase": "formal", "completed_requests": 100, ...}
{"formal_requests": 100, "warmup_replays": 10, "total_rows": 1100,
 "profile_stop_seen": true}
```

The watcher status is:

```json
{"status":"PASS","range":"issue26_root_formal_forward"}
```

Nsight's server log contains:

```text
The target application terminated. One or more process it created re-parented.
Waiting for termination of re-parented processes.
Processing events...
Generated:
	No reports were generated
```

It contains no `Capture range started in the application.` or
`Capture range ended in the application.` message. The run therefore produced
no `.nsys-rep`, SQLite database, or stats CSV.

## Interpretation and limits

This is a profiler/harness failure, not a vLLM latency result. The complete
1100-row replay proves the standard workload drained, but it supplies no valid
CUDA composition. The accepted clean native reference remains
`78.118782043--79.307357788 ms`; this run has no timing value to compare with
that reference.

Run02 and run03 both completed the same workload and both failed before report
generation. The repeated absence of Nsight capture-start/end messages localizes
the issue to NVTX trigger delivery or process-scope handling. The process-tree
watcher status alone does not prove that Nsight observed the range. No further
identical NVTX retry should be treated as progress without changing and first
validating the capture mechanism. Existing CUDA-only run04 and Kineto run03
remain diagnostic windows above the clean scale and cannot be reconciled into
a clean 79 ms decomposition.
