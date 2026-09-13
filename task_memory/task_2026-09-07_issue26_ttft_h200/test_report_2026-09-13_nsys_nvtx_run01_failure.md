## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-13 | Recorded the first NVTX worker launch failure and correction. |

# NVTX Nsight launch run-01 — failure report

## Execution

- GPU allocation: H200, `step_main`, selector `h200`, 8 GPUs, 64 CPUs,
  409600 MiB, `num_gpu_blocks_override=310809`.
- Predict-only command:

  ```bash
  /kubebrain/rlaunch --predict-only --charged-group=step_main \
    --private-machine=group --positive-tags=h200 --gpu=8 --cpu=64 \
    --memory=409600 --predict-node-num=10 --backoff-limit=1 -- bash -lc 'true'
  ```

  It returned seven available H200 nodes and exit code `0`.  The captured
  output is `analysis/nsys-nvtx-20260913-run01/predict_only.log`.
- RJob: `yc26-h200-nsys-nvtx-20260913-01`.
- Launch command: `analysis/nsys-nvtx-20260913-run01/launch.sh`.
- Worker: `tests/e2e/issue26_h200_nsys_nvtx_worker.sh` at initial commit
  `f04ec19a`.
- Source verification: `/data/ycfeng/tmp/vLLM-BS` at
  `0f34fb271fd66d7dd84201ebdd4722781f829390`.

## Criteria and result

The intended criteria were the standard ten drained warmup replays, 100
formal requests, 1100 client rows, and a nonempty NVTX-triggered `.nsys-rep`.
The RJob was scheduled to `gpu-h200-0052` and then entered `Failed` before the
vLLM server started.  No client request or formal batch was executed, so no
latency or profiler value is reported.

The worker's embedded shim did load successfully:

```json
{"capture_class":"Issue26NvtxCapture",
 "range":"issue26_formal_forward",
 "status":"PASS"}
```

The server log contains the complete failure:

```text
option '--env-var' cannot be specified more than once
usage: nsys profile [<args>] [application args]
```

Nsight's help requires one `--env-var` argument with comma-separated
assignments (`A=B,C=D`).  The worker initially supplied separate arguments for
`LD_LIBRARY_PATH` and `PYTHONPATH`.  This was corrected in commit `acbcba19`
by combining both assignments.  A replacement RJob must use the corrected
worker; run-01 remains a harness failure and is excluded from all profiling
analysis.
