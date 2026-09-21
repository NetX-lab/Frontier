## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-07 | Recorded fresh backend build and structural validation. |

# Collective Backend Build and Structural Validation

## Execution

Repository: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907

Submodule: frontier/cc_backend/backends/collective-sim at b8518afcc310f0fe0e3ce52ba6b4f0bf57a3be04, the current main gitlink. Initialized and freshly built; no historical prediction dataset or binary was copied.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
timeout -k 5s 120s git submodule update --init --recursive frontier/cc_backend/backends/collective-sim
mkdir -p /data/ycfeng/tmp/issue26-h200-build
cd frontier/cc_backend/backends/collective-sim/sim
env TMPDIR=/data/ycfeng/tmp/issue26-h200-build make -j8 > /data/ycfeng/tmp/issue26-h200-build/build.log 2>&1
```

Build exit code: 0. The complete build log is retained in backend_build.log. Existing compiler warnings include queue.cpp missing returns and eqds.cpp potentially uninitialized variables; no source was changed to suppress them. The required sim/datacenter/htsim_ndp exists and all ldd dependencies resolve on the CPU host. Container ABI compatibility still requires the H200 image probe; rebuild inside that runtime if its libc/libstdc++ is incompatible.

Topology checks use conda environment dev-vidur-v03-hopper-e2e, Python 3.13.13 (not the future GPU vllm-bs-0.10.2 runtime):

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_cc_backend_replica_local_layout.py tests/unit/test_parallel_semantics.py -q -p no:cacheprovider
```

Result: 9 passed in 20.25s. These checks detect incorrect outer-Replica multiplication and attention-DP/MoE-EP domain mapping. They establish structural mapping only.

## Direct backend invocation

The following reproduces the four calls. Use fresh runner output paths for a subsequent check.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
env TMPDIR=/data/ycfeng/tmp/issue26-h200-build PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PYTHON'
from frontier.cc_backend.cc_backend_config import CollectiveSimCCBackendConfig
from frontier.cc_backend.backends.collective_sim_cc_backend import CollectiveSimCCBackend
from frontier.types import ClusterType
for model in ('legacy_fabric', 'nvlink_analytic'):
    config = CollectiveSimCCBackendConfig(
        cluster_servers=1, cluster_gpus_per_server=8, parallel_tp=4, parallel_dp=2,
        runtime_num_replicas=1, runtime_num_pipeline_stages=1,
        runtime_attn_tensor_parallel_size=4, runtime_attn_dp=2,
        runtime_moe_tensor_parallel_size=1, runtime_moe_expert_parallel_size=8,
        intra_server_model=model,
        runner_out_dir='/data/ycfeng/tmp/issue26-h200-build/' + model,
        runner_end_us=100000, runner_stop_on_finished=True,
    )
    backend = CollectiveSimCCBackend(config, ClusterType.MONOLITHIC, 'h200', 'h200', 8)
    print(model, 'ATTN_TP', backend.predict_allreduce(16777216, 4, comm_domain='ATTN_TP'))
    print(model, 'MOE_EP', backend.predict_all_to_all(16777216, 8, comm_domain='MOE_EP'))
PYTHON
```

| Intra-server model | TP4 allreduce predicted ms | EP8 alltoall predicted ms |
| --- | ---: | ---: |
| legacy_fabric | 0.0524851 | 0.0 |
| nvlink_analytic | 0.3729050666666667 | 0.044277955555555554 |

Input: one server, eight GPUs, 16,777,216 bytes. Actual direct-call outputs are recorded in backend_smoke.json. Process exit 0 and all four calls returned finite nonnegative values. This detects runner import, executable launch, and scenario contract failures; those checks PASS.

The zero EP8 result identifies an unusable default for this single-node communication path. nvlink_analytic supplies nonzero intra-server timing, but its 450 GB/s, 0.8 efficiency, 0.5 us latency, and 50 us per-step allreduce launch overhead are uncalibrated defaults. None is a new H200 measurement. Physical topology and fresh collective timing evidence are pending. Ground-truth actual values, absolute errors, and relative errors are unavailable for these structural diagnostics; this is not a simulation-error or TTFT parity report.

## Prepared H200 probe

Script: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_h200_environment_probe.sh

```bash
bash -n tests/e2e/issue26_h200_environment_probe.sh
git diff --check
```

Both checks passed. Code was committed as 65c90aed, followed by a scoped CUDA library-path correction. The script has not run on a GPU worker. It records GPU identity, memory, driver, topology, NVLink status, nvcc, runtime versions, and eight-device compute assertions. It requires a new task-local output directory and uses the recovered image runtime without altering platform NCCL settings.

## H200 probe submission receipt

Prediction returned two H200 nodes with eight GPUs each using the following command:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy /kubebrain/rlaunch --predict-only --charged-group=step_main --private-machine=group --positive-tags=h200 --gpu=8 --cpu=64 --memory=409600 --backoff-limit=1 --predict-node-num=3 -- bash -lc 'true'
```

The first launcher used an outer 50-second timeout and was interrupted while pulling the image. It sent a stop request at 2026-09-07T15:33:32Z. This is a launcher failure, not a GPU/runtime failure. The stopped RJob yc26-h200-probe-20260907-1534 is retained. The corrected submission omits that timeout:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy /kubebrain/rlaunch --detach --name yc26-h200-probe-20260907-1536 --charged-group=step_main --private-machine=group --positive-tags=h200 --gpu=8 --cpu=64 --memory=409600 --backoff-limit=1 --max-wait-duration=2h --enable-sshd=false --image hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc --volume /data:/data --workdir /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907 -- bash tests/e2e/issue26_h200_environment_probe.sh /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-environment-probe-02
```

Accepted at 2026-09-07T15:34:27Z, assigned gpu-h200-0742.lgcm.sh.istep.fun. At 15:40:51Z it remained Starting/ContainerCreating, pulling the pinned image; worker restartCount=0, no image ID yet. Probe script has not executed. Do not infer runtime PASS from accepted submission or prediction.

## Probe-02 failure and probe-03 continuation

Probe-02 image started at 2026-09-07T15:41:34Z; environment.log captured the worker hostname followed by `fatal: detected dubious ownership in repository`. Result: FAIL for the provenance step, not a GPU compatibility verdict. No nvidia-smi output was generated because Git preceded it. The worker image did not accept the command-line safe.directory setup; its Git version is not yet observed.

Correction commit: 07bdc8d4. The probe now records GPU inventory/topology first, then uses an isolated task-temporary GIT_CONFIG_GLOBAL and an explicit trusted-worktree entry. `bash -n` and `git diff --check` both passed. No shared Git config was modified on the CPU host.

Probe-03 uses the same exact launch command above with job name `yc26-h200-probe-20260907-1544` and fresh output `runs/h200-environment-probe-03`. Accepted at 15:43:33Z; assigned `gpu-h200-0128.lgcm.sh.istep.fun`. At 15:44:39Z it was pulling the image. The first two jobs remain retained; resume by checking this third job, not by submitting a duplicate.
