# D24 native package offline verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded exclusive package, actual import origins, native pins, tests and parent-only live boundaries. |

## Scope and environment

Task `D24-NATIVE-STAGE`. Source checkout `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`, branch `feat/step5-d24-stage`, source HEAD `ebd5baf61757d48ff4ca6287903a01e6828c25f4` (clean at build). Candidate `/data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage`. No production code changes.

Builder/test/static interpreter `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python **3.12.13**, Torch **2.9.1+cu128**, pytest **9.0.2**. No conda activation. Dry-run interpreter `/data/ycfeng/tmp/stepmind-env/bin/python`; its exact version is in `offline_receipt.json`. `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`; scratch writes remain under `/data/ycfeng/tmp`.

## Executed commands

Working directory for these commands: `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`.

```bash
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d24_stage/build_capture.py

env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage/verify_r2_stage.py --stage-dir /data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage --output /data/ycfeng/tmp/step5_d24_native_20261022/capture02/offline_verification.json

env TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_native_pd_r2_verdict.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py tests/unit/test_step5_http_capture.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/step5_d24_native_20261022/capture02/pytest_temp --junitxml=/data/ycfeng/tmp/step5_d24_native_20261022/capture02/capability_tests.xml > /data/ycfeng/tmp/step5_d24_native_20261022/capture02/capability_tests.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/tmp/stepmind-env/bin/python -B /data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage/submit_native_pd.py --config /data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage/config.json --launcher-scope /data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage/launcher_scope.py --dry-run > /data/ycfeng/tmp/step5_d24_native_20261022/capture02/stepmind_dryrun.json
```

Additional direct Python CPU checks used the same 3.12.13 interpreter, cwd `/data/ycfeng/tmp/step5_d24_native_20261022/capture02`, with the same no-bytecode/no-user-site environment. They compiled every staged `*.py` with `compile(..., 'exec')`, ran `bash -n stage/submit.sh` and `bash -n -c <exact spawn_arguments.command>`, reproduced `build_stage.preflight(config, model_dir)`, and spawned isolated `python -I -B -c` import checks from both candidate cwd and source cwd. Both import censuses matched, every loaded task module resolved beneath the stage and its frozen tool hash, and no vLLM module was imported. Exact resolved paths and hashes are retained in `package_static.json`.

Staged capability checks reused existing test functions, substituting only the module under test with the actual staged module:

- `test_step5_native_reference.test_first_tail_values_use_actual_and_full_local_worker_callers`: 3 lengths × 8 ranks × actual/reference = **48** cases; same original CPU fixture, copies and assertions. `staged_collector.json`.
- `test_step5_http_capture.exchange`: two requests on one connection, healthy then injected reused EOF; preserve bodies/metadata, HTTP200/502, fresh/reused trace, response-header phase and symbolic traceback. `staged_http.json`.
- `test_step5_native_forward_census.{host_runtime,worker_runtime}` fixtures and `test_startup_worker_execute_is_excluded_without_losing_later_serving`: successful/failed compilation = **2** cases; actual locked `StepTimingCollector.record_batch` method body in CPU fixture. `staged_census.json`.

`ResourceConfig` was imported directly from `/data/ycfeng/steptron/steptron/exp/base_exp.py` with `PYTHONPATH=/data/ycfeng/steptron`, constructed from the exact dry-run arguments, and `sanity_check()` called. A Python audit hook rejected `socket.connect`, `socket.bind`, `socket.getaddrinfo`, `subprocess.Popen`, `os.system` and credential-file reads. No `steptron.utils.stepmind*` module was imported; no backend/client constructed. The two existing RDMA defaults, `fail_policy=Any`, no restart and 32 non-preemptible GPUs were asserted. This establishes local class validation only, not an API payload or live permission. `resourceconfig_offline.json`.

## Results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Fresh builder, Git/source binding | Owned checkout, committed implementation bytes | 16 actual module origins, 21 copied Git files; no root/cwd shadowing | PASS |
| Original capture01 | All manifest/model/native/settings bytes unchanged | Exact run definition, engine, resource arguments and original manifest matched | PASS |
| Native pins | Exact original50, exported source unchanged | 50/50 pins equal original and local locked source | PASS |
| Stage preflight | Complete immutable diagnostic settings | 50 tools, 163 requests, 14 waves, 8 content, 155 readiness-only | PASS |
| Python/bash | No compilation or syntax failure | 25 source files and both configured shell inputs passed | PASS |
| Existing scoped suite | All selected tests pass | **590 passed in43.25s**, no skips/failures | PASS |
| Actual staged collector | Bounded values across ranks/callers | 48/48, exact BF16 raw/uint8 mean bytes, later hash-only, original copy/fence counts | PASS |
| Actual staged HTTP/startup | Diagnostics active without behavior repair | Healthy/reused-EOF and2 startup cases passed | PASS |
| Resource class / submit dry-run | Offline, no API/auth/backend | Exact arguments and RDMA defaults passed | PASS |
| Budget | Original90min/48GPUh | 4×8GPUs×5400s/3600=48GPUh | PASS |
| Live/native outcomes | Actual creator/mount/GPU/source/content/witnesses | Not run in this subtask | UNVERIFIED |

One initial build attempt failed **before creating a candidate** because the new task-local `__main__` script was incorrectly included in the Git implementation-origin census. The census was restricted to implementation modules and the next build passed; no production/original-stage edit. The staged HTTP502 is an intentional fault-injection result.

The upper bound for added raw/mean hex payload remains **259,072,000 bytes** including cancel/reuse controls, versus **185,212,928 bytes** for three normal pairs. Existing 1GiB per-file/per-request,600s post-return diagnostic caps,7168/8192/128 settings and the workload deadline remain unchanged. These are bounds, not measured native artifact sizes.

## Verification limits and parent handoff

No GPU, platform/API, credentials, SSH, cloud read/write, submission or stop was performed. All>=32GPU auxiliary validation remains a soft diagnostic. Native failures retain their own handling; validator failure must not stop healthy work. The frozen stage is diagnostic-only and not clean-timing eligible. Independent state mismatch, pressure, pending READ abort and physical overlap are not waived.

Before one parent submission: verify fresh personal reservations/capacity policy and creator `i-fengyicheng`; confirm exact locked image digest, source Git blobs/import origins after integration, unused generation/cloud output, local NFS source/target equal candidate workspace, four8GPU H800 replicas, `codesign`, non-preemptible and both RDMA limits. Parent must recompute identity if imported code changes; use a distinct new generation instead of modifying this seal.

Parent-only launch (not executed):

```bash
setsid bash /data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage/submit.sh > /data/ycfeng/tmp/step5_d24_native_20261022/capture02/launcher.log 2>&1
```

This wrapper runs fresh exclusive presubmit verification, sources the existing personal local StepMind environment, then invokes the staged submitter with `/data/ycfeng/tmp/stepmind-env/bin/python`. Maintain the local launcher for the unchanged native wait. Do not rerun when a submission-attempt marker exists or the job ID is unknown. Use existing personal Python inspections by exact returned job name; no guessed job name or standalone CLI. If stopping is separately authorized, follow the parent runbook and `rjob_stop.sh` rule; this slice supplies no stop action.

After launch, parent checks actual installed pins/import origins, H800 UUID/role/DP/TP census, startup-versus-serving rows, actual raw/mean contents, independent references, native pressure/abort/overlap conservation, native I/O/business outputs, all replica/container exits and terminal status. Absent witnesses stay UNVERIFIED; propose numerical agreement separately. No submitted marker or job exists for capture02 at sealing.
