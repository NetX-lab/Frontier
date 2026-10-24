# D29 copied runtime environment verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | Recover the interrupted environment copy, separate the CPython base/dependency prefixes, and verify the unchanged strict probe locally and in the published image. |

PASS for CPU dependency imports and tiny binary numerical operations in both environments. This is not worker, full-workload, native vLLM, GPU or E2E acceptance. Production source remains `2ca5e57113b0bc99ab84e926e5bb492b3670072e`; the current non-task tree equals that source and frozen `1461573de388b72450189bc0c1b88b4ac32491f0`. The four ledger/copy/init worktrees are clean, their heads are ancestors of the primary, and the historical 89-row census is reused.

The original runtime's symlink target `/usr/bin/python3.10` is absent from the published image. A new copied dependency prefix was therefore prepared at `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025`. The first copy hit its 60-second cap (exit124); only 98 missing and one incomplete task-owned package files were completed. The retained completion receipt verifies 11,371 files / 386,015,193 bytes with inventory SHA256 `628b0758e492799670f3821bd81a5402792466cb5c3a95693c8b1c7d8c591ed8` and binary SHA256 `dbb04ace02303fa5cb3a6d1aa0e7a180d8dd23d3257cc6a0b155d0ee219432a2`. This inventory was not rescanned as another copy operation.

Both original copied-prefix probes then failed with `ModuleNotFoundError: No module named 'numpy'`. Inspected Debian `/usr/lib/python3.10/site.py:365` tests whether `sys.base_prefix != sys.prefix` before adding `site-packages`. The copied prefix had equal prefixes and searched `dist-packages`, while the complete dependencies were in `lib/python3.10/site-packages`. The three hypotheses before repair were prefix/search configuration, image dynamic-library compatibility after search is corrected, and copy completeness. The retained complete inventory rules out the last; successful strict image imports now also rule out a missing OS library for the exercised modules.

The repair adds a distinct base prefix `/data/ycfeng/envs/frontier-runtime-base-py310-d29-20261025`, copies only the identical CPython binary, shares the existing copied stdlib through one symlink, and writes the standard three-field `pyvenv.cfg` into the dependency prefix. The standalone base prefix is verified before writing that config. No existing copied file, original environment or dependency package is modified or installed. Configuration SHA256 is `16d91e196c87c1859a55ecd629ecefaaa806a1b1f2181b98ffd0f16a81d2af89`.

The original strict `python_runtime_probe.py` remains byte-identical (SHA256 `5669d6b0f64b40abbff96efbd431305983cb8e3db81c3e87f0f962b81b37e350`). Both affected probes rerun it unchanged. Expected: Python3.10.6, the configured dependency/base prefixes, all package versions/origins fixed, real Simulator/config imports from the current worktree, and exact small NumPy/SciPy/pandas/sklearn operation assertions. Actual: every assertion PASS; local wall1.857779s and image wall2.892711s. Probe process peak RSS is178,016,256B /183,410,688B respectively, measured by `getrusage`; these are import-probe values only. Configured scope/container memory is32,212,254,720B; cgroup peak and full-workload peak are not measured.

Versions remain Python3.10.6, NumPy2.2.6, pandas2.3.3, SciPy1.15.3, scikit-learn1.7.2, Plotly7.1.0, fasteners0.20, ddsketch3.0.1 and pytest9.1.1. The canonical environment was created with uv0.11.14 according to its existing pyvenv.cfg; no conda environment is used in these commands. The image is pinned to `hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee`, and the Docker checks use no network or GPU and a read-only source mount.

Exact execution results and commands are below. Raw stdout/stderr/execution JSON and scripts are retained under `resumed_execution/evidence/execution_preparation_2026-10-25/`. Initial absent generic `python` metadata invocation exit127 is an invocation-choice error; the specified interpreter is used thereafter. Successful image environment inspection alone never qualified the copied runtime; both original import failures remain retained.

| Execution | Raw exit | Wall seconds | Evidence |
| --- | --- | --- | --- |
| image_environment_probe | 0 | 15.837891 | image_environment_probe.stdout / .stderr / .execution.json |
| python_runtime_copy | 124 | 60.011371 | python_runtime_copy.stdout / .stderr / .execution.json |
| python_runtime_completed | 0 | 6.197543 | python_runtime_completed.stdout / .stderr / .execution.json |
| canonical_runtime_probe | 0 | 1.842964 | canonical_runtime_probe.stdout / .stderr / .execution.json |
| copied_runtime_local_probe | 1 | 0.101312 | copied_runtime_local_probe.stdout / .stderr / .execution.json |
| copied_runtime_image_probe | 1 | 2.516382 | copied_runtime_image_probe.stdout / .stderr / .execution.json |
| python_runtime_configured | 0 | 0.109191 | python_runtime_configured.stdout / .stderr / .execution.json |
| configured_runtime_local_probe | 0 | 1.857779 | configured_runtime_local_probe.stdout / .stderr / .execution.json |
| configured_runtime_image_probe | 0 | 2.892711 | configured_runtime_image_probe.stdout / .stderr / .execution.json |

All source regressions, reviews, composed audit and production guard are inherited from their unchanged frozen source, not rerun. Usage stays CORE4377/4404, PRICING3955/3956, native1350/1582, source9682/9942 and global9696/10000. Required-shape usage stays6/6; new pricing-wave, full CLI and StepMind/GPU operations are zero. The next actionable work is the joint pricing/emission/flush script and actual worker/cloud/lifecycle conditions. Extra pricing calls require a concrete finite plan against the existing execution bounds; no call is added by this environment repair.

image_environment_probe:

```sh
timeout 60s docker run --rm --network none --read-only --memory 30g --cpus 4 --tmpfs /scratch:rw,size=256m -v /data/ycfeng:/data/ycfeng:ro -w / -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR=/scratch --entrypoint python3 hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/image_environment_probe.py
```

python_runtime_copy:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_execution_preparation_20261025_01 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/prepare_python_runtime.py
```

python_runtime_completed:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_execution_preparation_20261025_01 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/complete_python_runtime.py
```

canonical_runtime_probe:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-preview PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_execution_preparation_20261025_01 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/python_runtime_probe.py /data/ycfeng/envs/frontier-py310
```

copied_runtime_local_probe:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-preview PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_execution_preparation_20261025_01 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/python_runtime_probe.py /data/ycfeng/envs/frontier-runtime-py310-d29-20261025
```

copied_runtime_image_probe:

```sh
timeout 60s docker run --rm --pull=never --network none --read-only --memory 30g --cpus 4 --tmpfs /scratch:rw,size=256m -v /data/ycfeng:/data/ycfeng:ro -w /data/ycfeng/Frontier/.worktrees/step5-preview -e PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-preview -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR=/scratch -e XDG_CACHE_HOME=/scratch/cache -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 --entrypoint /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/python_runtime_probe.py /data/ycfeng/envs/frontier-runtime-py310-d29-20261025
```

python_runtime_configured:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_execution_preparation_20261025_01 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/configure_python_runtime.py
```

configured_runtime_local_probe:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-preview PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_execution_preparation_20261025_01 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/python_runtime_probe.py /data/ycfeng/envs/frontier-runtime-py310-d29-20261025
```

configured_runtime_image_probe:

```sh
timeout 60s docker run --rm --pull=never --network none --read-only --memory 30g --cpus 4 --tmpfs /scratch:rw,size=256m -v /data/ycfeng:/data/ycfeng:ro -w /data/ycfeng/Frontier/.worktrees/step5-preview -e PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-preview -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR=/scratch -e XDG_CACHE_HOME=/scratch/cache -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 --entrypoint /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/python_runtime_probe.py /data/ycfeng/envs/frontier-runtime-py310-d29-20261025
```
