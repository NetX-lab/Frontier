# D24 native staging completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archived sealed offline-ready D24 capture02, validation boundaries and parent-only launch handoff. |

## Outcome

Task **D24-NATIVE-STAGE** is complete: a fresh diagnostic package is sealed and offline-ready at `/data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage`. No production code changed. All original capture01 stage/evidence bytes remain immutable. Native execution and judgment are exclusively parent-owned.

The existing canonical G3 builder was used from `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`, branch `feat/step5-d24-stage`, implementation source `ebd5baf61757d48ff4ca6287903a01e6828c25f4`. Sixteen actual imported module origins were bound to this checkout and Git blobs, and21 copied implementation files matched Git. The existing canonical package marker prevents installed regular `tests` shadowing; isolated imports from two cwd values matched the staged files and pins. The configured runtime/workload/resource definition is byte-equivalent to capture01; current HTTP/startup/reference diagnostic source improvements are active.

## Exact identities and deliverables

| Item | Path / SHA-256 |
| --- | --- |
| Stage configuration | `/data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage/config.json` — `6b5186548c791dccf9f2dd11203c48fbeb5d92f63dfa6aec44e89bb835b56680` |
| Sealed manifest | `/data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage_manifest.json` — `822f715925d23c8dabd015baaea6d025cfd4f1868a194e03d5caf19525affba3` (56 entries) |
| Source origins | `stage/source_origins.json` — `07359f45fc8af4a627bddecb9ede2655696cd43d9e7af5f00fca24ce0fd6d39d` |
| Exact50 native pins | `capture02/installed_source_pins.txt` — `724d5dfefab52fe9b82624306170d0583966cb94469a97cc9610eda1e7faa2ca` |
| Reference collector | `stage/r2_reference.py` — `30fbfd0a9eb440cc5dd42e3fc550eff2c2f1782fabdbce1bea92212bca8f7698` |
| Observation collector | `stage/r2_observation.py` — `774280921282530c3df3eb22d8f20435d8b99766ffe412f7cafac20a045e5b81` |
| Startup/runtime collector | `stage/r2_runtime.py` — `b71bc59620648ea2d575d1c32906b3da7b682ba2353efd56a0614f1c2900326a` |
| HTTP diagnostics | `stage/tests/comparison/calibration/pd_proxy.py` — `1720b0e8145ab0fcd1ff19c716e59490f18591188cf50308a2a43e6ce61650ce` |
| Canonical workload manifest | `5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3` |
| Canonical workload SHA256SUMS | `594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9` |

Retained task directory: `/data/ycfeng/Frontier/.worktrees/step5-d24-stage/task_memory/task_2026-10-22_step5_d24_stage/`. It contains requirements/plan/progress/summary, the bounded builder command, build/offline/resource/static/staged-capability receipts, sealed manifest copy, test log/JUnit and `test_report_2026-10-22_offline_stage.md`. Exact command receipt and source identities are in `offline_receipt.json` and the test report.

## Observed validation

- **PASS:** original run definition, engine, resources, model metadata and50 exported/installed native pins unchanged; native source root `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`.
- **PASS:** existing offline stage verifier:50 tool files,50 native pins,12 observation/14 schema source pins,163 requests in14 diagnostic waves,8 content/155 readiness-only requests,three independent local references and128 unchanged pressure attempts.
- **PASS:**590 existing scoped tests in43.25s, Python3.12.13 / Torch2.9.1+cu128 / pytest9.0.2. Additional existing fixtures exercised actual staged source:48 first-tail caller/rank/length cases,2 startup success/failure cases,healthy and reused-EOF HTTP cases.
- **PASS:**25 staged Python files compile, wrapper/worker bash syntax, configured node argv/environment reproduction and isolated staged-module origins from both cwd values.
- **PASS:** actual offline `ResourceConfig` construction/sanity check with network/process/credential-read audit boundaries; both RDMA defaults, `fail_policy=Any`, no automatic restart preserved. No backend/client constructed. Actual StepMind Python3.10.6 dry-run passed without SDK/authentication calls.
- **PASS:**H800/codesign32 non-preemptible P16/D16 TP8/DP2/EP16/PP1;7168/8192/128,0.92 memory utilization,FULL_AND_PIECEWISE/256 graphs,unchunked/native dummy/FP8 exclusions/MTP-off unchanged.5400s×32GPUs=48GPUh.

Initial build attempt01 failed before candidate creation because the new task-local script was included in Git-module census; restricting the census to committed implementation modules fixed it. No production change. Injected HTTP502 remains expected evidence, not a suppressed error.

## Parent-only next actions and limits

No submitted marker, submission-attempt marker or job exists for capture02. No GPU/platform/API/authentication/cloud/SSH/submission/stop action occurred. Loopback HTTP fixture tests were local CPU verification only. No further delegation, no push, no D25 repetition, no CPU actual-main or pressure-capacity repair was performed.

The frozen output destination is a string only: `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d24_capture02`. Expected creator `i-fengyicheng`; local source/target mount `/data/ycfeng/tmp/step5_d24_native_20261022/capture02`; cloud mount declaration `juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp`. All live identities, native raw values, pressure/abort/overlap, G3 qualification and formal timing remain **UNVERIFIED**. State numerical policy is unchanged; any tolerance needs separate approval. Auxiliary diagnostics stay soft and native I/O/failure semantics remain intact.

1. Parent rechecks current imported-code Git blobs/origins after integration, fresh personal reservations and unused generation. If any imported source changed, recompute identity and build a distinct new generation; never change this sealed package.
2. Parent revalidates live image/creator/local NFS source/resource shape/RDMA defaults, then launches once:

   ```bash
   setsid bash /data/ycfeng/tmp/step5_d24_native_20261022/capture02/stage/submit.sh > /data/ycfeng/tmp/step5_d24_native_20261022/capture02/launcher.log 2>&1
   ```

   The wrapper invokes a fresh exclusive presubmit check, the existing personal local StepMind environment and the staged submitter. Keep the launcher alive. Do not retry when an attempt marker exists or job identity is unknown.
3. Parent inspects actual loaded source and GPU/rank census, startup/runtime rows, bounded raw/mean contents, independent reference and native pressure/READ-abort/overlap conservation, business outputs and all replica/container terminal states using the returned exact job name. Missing witnesses stay UNVERIFIED; numerical-rule selection remains a separate user decision.

Implementation source remains ebd5baf. The task-record commit is recorded in `progress.md`; no source-code integration is required from this branch.
