## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded successful H200 PPLX intranode capability rerun with Gloo metadata group. |

# PPLX capability probe rerun — H200

## Execution

- RJob: `yc26-h200-pplx-capability-20260913-02`
- Cluster/quota: H200, `step_main`, single node, eight GPUs
- Node: `gpu-h200-0246.lgcm.sh.istep.fun`
- Image: `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`
- PPLX overlay: `/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03`
- vLLM source: `/data/ycfeng/tmp/issue26-vllm-diagnostics-deepep-20260912`
- Preflight: H200 `step_main+h200`, 8 GPU, 64 CPU, 409600 MiB; ten candidate nodes returned.

The corrected worker used `pplx_capability_probe_20260913.py`: NCCL world group
for CUDA initialization plus a Gloo process group registered as `"default"` for
PPLX's CPU metadata `alltoall_base` exchange. It created an intranode PPLX
handle with `max_num_tokens=8`, `num_experts=128`, `experts_per_token=8`,
`world_size=8`, `dp_size=4`, and `hidden_dim=2048` / BF16 payload bytes 4096.

## Criteria

PASS requires all eight ranks to import the PPLX custom op, initialize NCCL and
Gloo groups, construct the intranode CUDA IPC handle, complete the Gloo barrier,
and destroy the handle, with `torchrun.exit_code=0`.

## Evidence

Status: **PASS**.

- RJob phase: `Succeeded`; worker command exit code: `0`.
- `preflight.json`: `find_spec=true`; origin is the intended PPLX overlay.
- All rank status files `rank-0-status.json` through `rank-7-status.json` are
  `{"status":"PASS", "world_size":8}`.
- Every rank log contains the ordered phases:
  `process_start -> pplx_import -> dist_initialized (device_name=NVIDIA H200, metadata_backend=gloo) -> handle_initialized -> barrier PASS -> handle_destroyed PASS`.
- No exception or failed phase occurred.

The previous `-01` failure (`No backend type associated with device type cpu`)
was therefore confirmed as a probe setup error caused by registering NCCL for
PPLX's CPU metadata exchange. The corrected probe now demonstrates import,
custom-op registration, c10d Gloo metadata-group resolution, CUDA IPC handle
construction, synchronization, and teardown on all eight H200 ranks.

## Limits and next action

This is a capability probe only. It launches no MoE dispatch/combine kernel and
provides no clean batch CUDA span. Formal PPLX replay remains a separate job and
must use the standard warmup/drain/identity chain after coordinator scheduling.
