## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded failed H200 PPLX intranode capability probe and evidence-only Gloo metadata-group correction. |

# PPLX capability probe — H200 2026-09-13

## Execution

RJob: `yc26-h200-pplx-capability-20260913-01` (H200, `step_main`, eight GPUs,
image digest `sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`).
The probe used the PPLX overlay `/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03`
and vLLM source `/data/ycfeng/tmp/issue26-vllm-diagnostics-deepep-20260912`.

The worker set `PYTHONPATH=$OVERLAY:$VLLM_SOURCE`, imported `pplx_kernels`,
initialized an NCCL world process group, registered a process group for PPLX's
C++ binding, and attempted an eight-rank `AllToAll.intranode` handle followed by
an inter-rank barrier and handle destruction.

## Criteria

The probe passes only when all eight ranks complete PPLX import/custom-op
registration, c10d metadata-group resolution, CUDA IPC handle construction,
barrier, and destruction. A failure is classified by the first failed phase;
it is not reported as a generic runtime unsupported result.

## Evidence

Status: **FAIL — process-group backend mismatch in probe setup**.

`preflight.json` confirms the wheel is importable and the custom op is present:

```json
{"find_spec": true, "origin": "/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03/pplx_kernels/__init__.py"}
```

All reached `pplx_import` and `dist_initialized` on NVIDIA H200. The first
failed operation was `AllToAll.intranode(...)` on ranks 0, 2, 3, 5, 6, and 7;
ranks 1 and 4 were terminated by torchrun after the first failures. Every
persisted failure status reports:

```text
RuntimeError: No backend type associated with device type cpu
```

The traceback points to `pplx_kernels/all_to_all.py:107` and the C++ binding's
`DistributedTorch::allToAllImpl`. That binding constructs CPU tensors and calls
`group->alltoall_base`; the probe had registered the NCCL world group under
`"default"`, so NCCL rejected CPU tensors. This establishes a probe setup error,
not a PPLX build or runtime capability failure.

## Evidence-only correction prepared

The probe source now creates `dist.new_group(backend="gloo")`, registers that
CPU metadata group as `"default"` for PPLX resolution, and uses it for the
metadata barrier. The NCCL world group remains initialized for CUDA device
selection. The status output is also rank-specific (`rank-<rank>-status.json`)
to avoid eight-rank write races.

The corrected source passes local Python syntax compilation and `git diff --check`.
No GPU rerun or formal PPLX replay was submitted from this correction.

## Limits and next action

The failed run proves import and custom-op registration only. It does not prove
or disprove PPLX intranode handle construction. A rerun of the same capability
probe with the Gloo metadata-group setup is required before any formal replay.
