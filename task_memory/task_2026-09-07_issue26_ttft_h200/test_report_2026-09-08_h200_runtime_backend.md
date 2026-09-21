## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded actual H200 topology, exact image runtime, and in-image backend verification. |

# H200 Runtime and Collective Backend Verification

## Execution

Worktree: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907

Worker script: tests/e2e/issue26_h200_backend_worker.sh, sourcing tests/e2e/issue26_h200_environment_probe.sh. Verified code committed as 591c5f05. The actual run used the same file contents before the commit; its HEAD field therefore records preceding commit 3319bc00.

The company proxy recipe was obtained from http://deploy.i.shaipower.com/httpproxy. The inherited localhost proxy returned HTTP 502; direct access to this endpoint alone bootstrapped the required company configuration. Subsequent platform operations used proxy.i.shaipower.com:3128, preserving company no_proxy and adding localhost. Older proxy-unset commands are historical, not the current recipe.

Exact submitted launcher:

```bash
#!/usr/bin/env bash
set -euo pipefail
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" NO_PROXY="$no_proxy,127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
exec /kubebrain/rlaunch --detach --name yc26-h200-backend-20260908-0010 --charged-group=step_main --private-machine=group --positive-tags=h200 --gpu=8 --cpu=64 --memory=409600 --backoff-limit=1 --max-wait-duration=2h --enable-sshd=false --image hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc --volume /data:/data --workdir /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907 -- bash tests/e2e/issue26_h200_backend_worker.sh /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-h200-backend-runtime-01
```

The launcher ran independently of interactive sessions:

```bash
systemd-run --user --unit=yc26-h200-backend-launcher-01 --property=MemoryMax=2G --property=StandardOutput=append:/data/ycfeng/tmp/issue26-h200-network/launcher-backend-01.log --property=StandardError=append:/data/ycfeng/tmp/issue26-h200-network/launcher-backend-01.log /bin/bash /data/ycfeng/tmp/issue26-h200-network/launch-backend-01.sh
```

Python: /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python, conda environment vllm-bs-0.10.2, version 3.10.16. Torch 2.8.0+cu128, FlashInfer 0.3.0, CUDA toolkit V12.8.93, driver 570.124.06. vLLM source overlay: /data/ycfeng/tmp/vLLM-BS, commit c169f48fa0ce16455f64101bc4366b23b1652a43. The pinned image is recorded in the launcher. Collective-sim uses main's gitlink b8518afcc310f0fe0e3ce52ba6b4f0bf57a3be04.

The script invokes the real NVIDIA binary at /usr/local/nvidia/bin/nvidia-smi, rebuilds with make -B -j8 -C "$REPO_ROOT/frontier/cc_backend/backends/collective-sim/sim", runs ldd on sim/datacenter/htsim_ndp, and executes the embedded Frontier backend check. Shell and embedded Python syntax checks, git diff --check, and subsequent live execution all passed.

## Criteria and Evidence

| Check and passing condition | Observed result |
| --- | --- |
| Eight actual H200 GPUs on step_main | PASS, gpu-h200-0019; 143771 MiB / 150754820096 bytes per GPU |
| Fully available intra-server NVLink fabric | PASS, nvidia-smi shows NV18 for all off-diagonal GPU pairs; NVML independently finds 18 active links/GPU, NVSwitch remote type 2, four switch PCI addresses, and 56/56 directed peer statuses OK (0) |
| Actual computation on all GPUs | PASS, each ones(32,32) matrix product returned 32.0; H200_RUNTIME_PROBE_PASS |
| Fresh build in the exact worker image | PASS, make -B -j8 completed, ldd dependencies resolve; existing compiler warnings retained |
| Actual Frontier -> collective_sim/htsim calls | PASS, nvlink_analytic TP4 allreduce and EP8 alltoall returned finite positive times; H200_BACKEND_RUNTIME_PASS |
| Launcher completed | PASS, user systemd service ExecMainStatus=0 |

For 16777216 bytes, the fresh structural predictions are TP4 allreduce=0.3729050666666667 ms and EP8 alltoall=0.044277955555555554 ms. These are not measured NCCL latencies or calibrated parameters. Ground-truth communication latency, absolute/relative prediction error, and formal TTFT error are unavailable. No formal profiling/TTFT evidence has been generated or imported from historical runs.

## Resolved Failures

The image contains a zero-byte /usr/bin/nvidia-smi, shadowing the injected binary and silently emitting nothing. Explicitly selecting /usr/local/nvidia/bin/nvidia-smi produced the required inventory/topology/NVLink evidence. Probe-03 stopped before worker execution and its interactive launcher disappeared; the stopping actor was not independently established. Independent user systemd services subsequently completed worker runs. Probe-02's Git ownership admission failure was corrected using the task-local GIT_CONFIG_GLOBAL with a trusted-worktree entry, verified under image Git 2.34.1.

## Durable Artifacts and Limits

- runs/h200-h200-backend-runtime-01/environment.log: complete inventory, topology, NVLink/NVML, runtime, compute, ldd, and backend output.
- runs/h200-h200-backend-runtime-01/backend_build.log: full in-image build log.
- runs/h200-h200-backend-runtime-01/backend_smoke.json: fresh structural predictions.
- analysis/h200_topology.json: compact topology from independent probe-05.
- runs/h200-environment-probe-02/environment.log: retained Git failure.

The doubled h200 prefix in the backend output directory is the actual submitted path and is intentionally retained. The initial report-generation command used the wrong path and failed before writing; this report uses the verified existing path.

D003 clean canonical TTFT producer remains awaiting YC's decision. Full case semantics, fresh profiling, clean/diagnostic vLLM, Frontier caches/results, comparison, RCA, repair and validation remain pending. Physical topology is established for the observed node; future workers must pass inventory checks too.
