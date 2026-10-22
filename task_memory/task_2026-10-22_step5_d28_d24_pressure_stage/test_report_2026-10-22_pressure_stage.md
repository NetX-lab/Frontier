# D28 pressure-stage verification

## Modification History

| Date | Change |
| --- | --- |
|2026-10-22|Recorded committed code, golden/native failure seam, exclusive capture03 and explicit routing blocker.|

## Environment and scope

Owned repo `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`, branch `feat/step5-d24-stage`; base `cac54074aff3ff2a536aa242630892d0948fabc5`; code commit `6e73330c2d04af382cf65a2c689c5401ca202485`. Four changed code/test files only, listed in `changed_files.txt`, with exact diff `code_diff.patch`. No native source/simulator/recorder/gate/CPU-stream edits.

Builder/tests/static: `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13, Torch2.9.1+cu128, pytest9.0.2. Payload: `/data/ycfeng/tmp/stepmind-env/bin/python`, Python3.10.6. No conda activation. Scratch `/data/ycfeng/tmp`; no GPU/platform/API/auth/cloud/SSH/client/backend construction.

## Exact commands and observed outcomes

Working directory `/data/ycfeng/Frontier/.worktrees/step5-d24-stage` unless noted. The 711-test check ran against the exact bytes subsequently committed as6e73330. No simulator/native engine or CPU-overhead run occurred.

```bash
env TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_native_pressure_observation.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_native_pd_r2_verdict.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py tests/unit/test_step5_http_capture.py tests/unit/test_dp_placement_harness.py tests/unit/test_step5_canonical_native.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/step5_d28_affected03 --junitxml=/data/ycfeng/tmp/step5_d28_affected03.xml > /data/ycfeng/tmp/step5_d28_affected03.log 2>&1
```

Expected all selected checks pass; observed **711 passed in122.89s**, no skipped/failed tests. Includes no-override golden control, P byte-equivalent argv, exact D-only32000, unchanged reference/base engine/workload/resource settings, rejected role/count/definition drift, existing content/HTTP/startup/native worker checks.

New `test_step5_native_pressure_observation.py` executes exact pinned `KVCacheManager.allocate_slots` method body with CPU fault injection at coordinator return. Five cases cover native RUNNING demand4/free3 after retirement, WAITING watermark, reserved capacity, scheduled=False and pre-retirement full-sequence admission rejection. The real method executes its own comparison and returns None; observer captures exact injected native-return demand/free pair and preserves native no-grant behavior. This is **fault injection at the native call seam**, not a full allocator/engine/GPU execution; it introduces no demand formulas, GPU reads or sync. Parent's independent full native CPU allocator results are referenced unchanged in `parent_pressure_evidence.json`; they are not relabeled as this task's native execution.

```bash
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d28_d24_pressure_stage/build_capture.py

env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage/verify_r2_stage.py --stage-dir /data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage --output /data/ycfeng/tmp/step5_d24_native_20261022/capture03/offline_verification02.json

env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/tmp/stepmind-env/bin/python -B /data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage/submit_native_pd.py --config /data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage/config.json --launcher-scope /data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage/launcher_scope.py --dry-run > /data/ycfeng/tmp/step5_d24_native_20261022/capture03/stepmind_dryrun.json
```

Builder PASS with16 actual imported origins and21 copied Git files bound to6e73330. Final offline verifier PASS:51 tool pins,50 unchanged installed native pins,12 observation/14 schema pins,163 requests/14 waves/8 content/155 readiness-only,three independent local references,128 pressure attempts; original 6144/1024,7168/8192/128,graph/native runtime and5400s bounds remain. Additional tool entry is truthful `routing_status.json`, not a routing code modification. Routing effective state is UNVERIFIED, not claimed from declared config.

Direct static Python checks (same3.12.13/no-user-site/no-bytecode environment, cwd capture03) reproduced `build_stage.preflight` exactly, compared every normalized argv with capture02 and called `node_plan` to retain each expected effective engine in `package_static.json`. Both P argv omit override and match golden byte-for-byte after path normalization; both D argv contain exactly one flag/value32000 and otherwise match golden. Base `engine.json` deliberately has no override so P is not mislabeled; role map in config/run controls existing node projection. Reference requests share the same D engine; no reference-specific pool.

All25 staged Python files compiled with `compile(...,'exec')`; `bash -n stage/submit.sh` and `bash -n -c <exact spawn_arguments.command>` passed. Isolated `/data/ycfeng/envs/asyncopd-py312/bin/python -I -B -c <worker/replay import census> <stage>` subprocesses from both capture03 and source cwd resolved all task modules beneath stage and expected tool hashes; no vLLM import. `staged_imports.json` retains exact paths/hashes.

Offline resource check used `PYTHONPATH=/data/ycfeng/steptron` and3.12.13: imported actual ResourceConfig from `steptron/exp/base_exp.py`, constructed exact dry-run arguments and called sanity_check with an audit hook rejecting socket connect/bind/DNS, process creation and credential-file reads. Both existing RDMA resources, fail_policyAny, no restart and32 non-preemptible GPUs passed. No `steptron.utils.stepmind` modules/client/backend imported. `resourceconfig_offline.json`.

Actual frozen staged modules reused existing tests:48 raw/mean caller/rank/length cases,2 startup cases with success/native compilation failure,2 healthy/reused-EOF HTTP cases,5 native method-body pressure cases all PASS. Modules and checks recorded in `staged_capabilities.json`. HTTP502 is intentional injected EOF, not a swallowed native error.

## Failures retained and diagnosis

- Focused01:148 passed,2 failures from test-owned missing nested directory and missing override lookup ValueError before presence assertion. Corrected precise setup/assertion order; existing/native behavior unchanged. JUnit retained.
- Affected02:60s timeout near completion; no PASS claimed. Affected03 used180s and finished711 checks. No workload ceiling change.
- Composite staged01 passed source bytes until final import-origin assertion; earlier test imports cached repository `pd_proxy` under same alias. Composite02 removed only that process-local alias before stage-first import, asserted every origin and passed. First fixtures retained in scratch. No stage/core/native source alteration or retry policy change.
- Before sealing only: metadata was corrected from an inferred effective routing claim to original declared routing/effective UNVERIFIED. Final verifier02/config/manifest reflect corrected bytes; prior receipt01 is scratch historical, not final acceptance.

## Routing steering and limits

Read-only native routing probe command and exact hashes/assignments are in `native_routing.md`/`native_routing_receipt.json`. Native352 experts/top_k8 built-ins are `uniform_random` and `normal_routing`.44 tokens/352 assignments yield min0/max4 instead of exact1;6144 tokens yield106–173 instead of exact balance. `balanced` is unsupported. Normal Step factory and mega branch construct the router; monolithic kernels can bypass select_experts. No speculative flag, custom registration, EPLB substitution or backend change was applied.

**Pressure local stage acceptance PASS; balanced-aligned submission BLOCKED.** Native GPU queuing/preemption/actual decode None/READ correlation/pressure timing/ownership/overlap/state agreement/business outputs/terminal validation remain UNVERIFIED. Same exact numerical criteria, bit-exact wire-content requirement and native failure handling continue. All>=32GPU auxiliary checks soft gate. Capture02 and original packets/pins/receipts remain immutable and unsubmitted; capture03 has no launcher/attempt/submitted marker or job.

Parent must resolve balanced route separately and use a fresh generation for any post-seal routing/source change. Then rebind integrated imported-code origins/committed bytes, check personal creator/live local NFS mounts/image/RDMA/32H800 codesign5400s, and inspect actual effective P/D argv/pools and native pressure/reference witnesses. Do not launch capture03 as balanced calibration. Workflow→GPU/operator gap→CPU overhead last; no CPU overhead suite is introduced here.
