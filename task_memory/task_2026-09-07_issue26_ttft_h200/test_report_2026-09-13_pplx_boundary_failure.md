## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the failed PPLX boundary startup and its missing runtime-overlay root cause. |

# PPLX boundary replay startup failure

## Execution

- RJob: `yc26-h200-pplx-boundary-20260913-01`
- Command: `bash tests/e2e/issue26_h200_pplx_boundary_worker.sh task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-boundary-20260913-01`
- Cluster: H200, `step_main`, one 8-GPU worker, `num_gpu_blocks_override=310809`
- Source commit: `cf1ef5de9c0c45aedec4cf9d22c8eb56caf8bf0b`
- Intended PPLX overlay: `/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03`
- Intended dependency path: `/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/python/nvidia/nccl/lib:/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/python/nvidia/nvshmem/lib`

The RJob reached `Running` on 2026-09-12T19:17:42Z and finished `Failed` at
2026-09-12T19:19:11Z. The environment probe completed and recorded H200/NVLink
evidence before vLLM startup. No client requests, warmups, formal rows, or
boundary timing rows were produced.

## Criteria

The run would be admissible only after ten drained 100-request warmups, 100
formal requests, 1100 client rows, identity validation, and four DP0 TP0--TP3
first-formal boundary rows. Since the server failed before health readiness,
all timing criteria are `FAIL/PENDING`; no clean span value is reported.

## Evidence and root cause

`runtime/server.log:251` and repeated worker traces show:

```text
AssertionError: pplx_kernels not found. Please follow ... install pplx_kernels.
```

The failure occurs in `PPLXAll2AllManager` during model-parallel group
initialization (`vllm/distributed/device_communicators/all2all.py:78`), before
model execution. The RJob launch annotation contains the source path but no
`ISSUE26_OPTIONAL_PYTHONPATH` or PPLX NCCL/NVSHMEM library path. Consequently
the vLLM process imports the pinned source, but Python cannot discover the
PPLX package. This is a launch-environment omission, not a PPLX capability or
kernel timing failure.

The corrected capability probe independently confirms that the overlay is
valid on H200: `test_report_2026-09-13_pplx_capability_probe_rerun.md` records
all eight ranks importing and initializing the intranode PPLX handle with the
Gloo metadata group.

## Recovery

Do not rerun capability or instrumented replay. Submit one fresh boundary-only
RJob after explicitly exporting:

```text
ISSUE26_OPTIONAL_PYTHONPATH=/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03
ISSUE26_EXTRA_LD_LIBRARY_PATH=/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/python/nvidia/nccl/lib:/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/python/nvidia/nvshmem/lib
ISSUE26_LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.1
ISSUE26_DIAGNOSTIC_VLLM_SOURCE=/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt
ISSUE26_DIAGNOSTIC_VLLM_COMMIT=cf1ef5de9c0c45aedec4cf9d22c8eb56caf8bf0b
ISSUE26_WARMUPS=10
```

The retry must use a new output directory and preserve the same H200
`step_main + h200 + 310809` configuration. Only a successful standard replay
with 1100 client rows and four selected boundary rows can provide the requested
PPLX clean span.

