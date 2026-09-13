## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-13 | Recorded the completed-workload NVTX capture failure for run02. |

# H200 NVTX Nsight run02 failure

## Execution

- RJob: `yc26-h200-nsys-nvtx-20260913-02`
- Cluster and selector: `step_main + h200`
- Allocation: 8 GPUs, 64 CPUs, 409600 MiB
- Workload: 4096-prefill / 1024-output, TP4 / DP2 / PP1 / EP8, BF16,
  eager, FlashInfer, uniform routing, prefix caching OFF, chunked prefill OFF
- KV budget: `num_gpu_blocks_override=310809`
- vLLM source: `/data/ycfeng/tmp/vLLM-BS@0f34fb271fd66d7dd84201ebdd4722781f829390`
- Backend: `VLLM_ALL2ALL_BACKEND=naive`
- Worker: `tests/e2e/issue26_h200_nsys_nvtx_worker.sh`
- Run directory:
  `task_memory/task_2026-09-07_issue26_ttft_h200/analysis/nsys-nvtx-20260913-run02/`

The worker was launched with Nsight Systems capture switches:

```text
--capture-range=nvtx
--nvtx-capture=issue26_formal_forward
--capture-range-end=stop
--trace=cuda,nvtx
--cuda-trace-scope=process-tree
```

## Criteria and evidence

The standard replay predicates completed:

| Criterion | Result | Evidence |
| --- | --- | --- |
| Ten complete warmup replays | PASS | `runtime/client.log`, `warmup:0` through `warmup:9` |
| 100 requests per warmup | PASS | `runtime/phase_records.json` |
| 100 formal requests and full drain | PASS | `runtime/client.log`, formal record |
| Total client rows | PASS, 1100 | `runtime/client.jsonl` |
| First formal identity | PASS, `pf4096_dc1024:0` | `runtime/client.log` |
| NVTX shim import | PASS | `nvtx-shim/status.json` |
| Nsight report | **FAIL** | `nsys/` contains only `version.txt` |

The vLLM client log ends with:

```text
{"phase": "formal", "completed_requests": 100, ...}
{"formal_requests": 100, "warmup_replays": 10, "total_rows": 1100,
 "profile_stop_seen": true}
```

The control markers exist and the shim reports:

```json
{"capture_class":"Issue26NvtxCapture",
 "range":"issue26_formal_forward",
 "source_module":"vllm.v1.issue26_nsys_capture",
 "status":"PASS"}
```

The Nsight server log contains only:

```text
The target application terminated. One or more process it created re-parented.
Waiting for termination of re-parented processes.
Processing events...
Generated:
	No reports were generated
```

It does not contain `Capture range started in the application.` or
`Capture range ended in the application.`. Therefore the NVTX trigger was not
observed by Nsight, even though the run-local shim imported successfully.

The platform RJob phase is `Failed` because the worker requires a nonempty
`.nsys-rep` after cleanup. No `.nsys-rep`, SQLite export, or Nsight stats were
produced.

## Interpretation and limits

This is a harness/profiler failure, not a vLLM latency result. The completed
client rows establish that the replay itself drained, but they do not establish
the timing or composition of the first formal CUDA span. The accepted clean
native reference remains `78.118782043--79.307357788 ms`; no value from this
run may be compared with it or used for Frontier correction.

The absence of Nsight's capture-start/end messages localizes the failure to
NVTX trigger delivery or process-scope capture. A follow-up retry must first
make the trigger observable (for example, validate NVTX capture across the
vLLM worker process tree or use an explicitly bounded alternate trigger) and
must produce a nonempty report before any CUDA decomposition is attempted.
