# Profile03 collection repair CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded unchanged-source RED, repaired GREEN, final frozen suites, budgets and explicit existing matrix binding. |

## Scope and environment

Task: `W128-PROFILE03-COLLECTION`; worktree `/data/ycfeng/Frontier/.worktrees/v4-profile03-collection`, branch `v4-profile03-collection`, base `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`. Only two allocated source files changed. Root independent review/integration and actual image/package/GPU verification remain outside this report.

Executable: `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13 / Clang22.1.3; no conda activation. Linux5.10.25 / glibc2.35. Explicit environment:

```text
PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site
PYTHONDONTWRITEBYTECODE=1
TMPDIR=/data/ycfeng/tmp
FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69
```

Raw logs, JUnit XML and temporary fixture outputs: `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE03-COLLECTION/`. Selected evidence is retained in this task directory. Worker tests use process/platform/filesystem/time stand-ins; existing HTTP suites use loopback-only local servers. No external network, GPU, Docker, StepMind, authentication, SSH, cloud access or package generation was performed. No DP-owner changes were inspected or edited.

## Exact commands

Baseline worker and manifest commands are retained verbatim in `baseline_commands.json`. The unchanged-worker RED command is retained in `red_command.json`; it used:

```bash
PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/ycfeng/tmp \
FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 \
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest \
tests/unit/test_native_pd_worker.py -q -p no:cacheprovider \
--basetemp=/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE03-COLLECTION/red_worker_temp
```

Every final argv, UTC timestamp, exit code, XML/log hash, count and source hash is in `final_checks.json`. Each suite ran in a separate process, at most three concurrently. No combined-process fake-module contamination was introduced. Final worker argv:

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_pd_worker.py \
-q -p no:cacheprovider \
--junitxml=/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE03-COLLECTION/frozen_test_native_pd_worker.xml \
--basetemp=/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE03-COLLECTION/frozen_worker_temp
```

The other final commands use `tests/unit/<suite>.py`, `frozen_<suite>.xml` and `frozen_<suite>_temp` under the same root; their full argv is retained, not inferred from this template.

## Baseline and RED

Baseline worker19PASS and manifest14PASS. Production source before/after RED was unchanged SHA256 `9ec701ec12f7a5da0f442f994a353f98c5ae1d2880b9feca38106ea536fd5cab`, identical to the root's original lifecycle controls. RED exit1:19PASS/16FAIL, raw original `red_worker.log` retained here.

| Control | RED observation | GREEN expectation and actual |
| --- | --- | --- |
| Role values7/5 or prefill-only7 | All four values stayed3 | 7/7/5/5 or7/7/3/3; input JSON and four paths unchanged. PASS |
| Invalid mapping/type/role/count | Six cases did not raise ValueError | Six cases raise the specific role-iteration ValueError. PASS |
| Natural headless marker20, stop10, peer60 | Marker missing at short wait deadline | Wait0→20, marker true, native0, signalNone, worker0. PASS |
| Active headless marker20 | Marker missing at short wait deadline | Wait0→20, marker before stop call, native0/SIGINT, worker0. PASS |
| Marker61 or no marker | Existing soft missing observation; timing field absent | Wait0→60, soft missing marker, worker/native0; active SIGINT or natural signalNone. PASS |
| Headless native23 before marker | Existing native error outcome; timing absent | Wait0→0, native23, worker1, signalNone; marker remains softly unverified. PASS |
| Stale listing20, four direct paths visible | Wait20 + fake stop3, not expected0+3 | Quorum0→0; all four known members retained despite stale listing. PASS |
| Peer host3 appears20 | Existing wait20, timing absent | Quorum0→20, all four retained, worker/native0. PASS |
| Peer host3 never appears | Existing deadline/native outcome, timing absent | Quorum0→60, worker1/failed sentinel; three retained members, native0. PASS |
| Native stop exception | Additional preservation control | The same injected RuntimeError propagates; timing does not suppress it. PASS |

The3-second native stop is solely a test stand-in for timing separation, not a production constant, model cost or stop-bound claim. Normal/error native outcomes also retain the existing admission-return4, early-native23, shutdown-native23, manifest/report soft-error and staged-child controls. Existing ordinary `shutdown_order` dictionary equality still passes.

Profiler overrides use only the existing absolute local Torch `NativeServeProfile` path. Empty/missing role overrides preserve max_iterations; CUDA and URI profiler configuration remain unchanged even with a role override. Each planned profiler dictionary is copied; no later package512/64 or records600000 literal was added to production or sealed inputs.

## Final frozen suites

Passing condition: actual exit0, expected scenario assertions, zero errors/failures/skips, immutable before/after source hashes. Counts below are separate, not summed.

| Suite | Actual tests | Exit | Errors / failures / skips | Result |
| --- | ---: | ---: | --- | --- |
| test_native_pd_worker | 36 | 0 | 0 / 0 / 0 | PASS |
| test_native_pd_manifest | 14 | 0 | 0 / 0 / 0 | PASS |
| test_native_serve_profile | 33 | 0 | 0 / 0 / 0 | PASS |
| test_native_streaming_replay | 39 | 0 | 0 / 0 / 0 | PASS |
| test_native_cold_start | 18 | 0 | 0 / 0 / 0 | PASS |
| test_native_pd_canonical_http | 1 | 0 | 0 / 0 / 0 | PASS |
| test_calibration_pd_harness | 68 | 0 | 0 / 0 / 0 | PASS |
| test_dp_placement_harness | 75 | 0 | 0 / 0 / 0 | PASS |

An initial eight-suite batch had passing XML summaries but its final snapshot assertion failed after the owner added two test assertions while it ran. Its individual exits were not retained; they are not inferred. `issues.md` and `verification_interruption.json` preserve the original aggregate exit1 and traceback. Both sources were then frozen and all affected suites rerun to obtain the final actual exits above. This was an auxiliary verification-record failure, not a production or business-test failure.

## Budgets, source hashes and matrix

| Source | Additions | Deletions | Gross / allocation |
| --- | ---: | ---: | --- |
| tests/comparison/calibration/native_pd_worker.py | 24 | 6 | 30 / 80 |
| tests/unit/test_native_pd_worker.py | 121 | 8 | 129 / 180 |
| Total | 145 | 14 | 159 / 260 |

Final worker SHA256: `59cdd58d9fce0d3aa432caeace4411d8fef074f59857260e81c75d3441bcfa1b`.
Final tests SHA256: `84fa7a4be7a8a957eb62671e652d6e00ff48d153c92671bea7a53fe6528cea4c`.
`source_manifest.json` binds the final commit, exact two-file gross budget, source hashes and native stop source identity; `source_changes.diff` is the exact diff from the allocated base.

Archive checks preserve raw evidence: the initial staged whitespace command exited2 on two original pytest `E` console lines in `red_worker.log`. That log is byte-identical to its original temporary output. Only the raw log and separately base-validated unified diff are excluded from ordinary staged whitespace checks; all maintained documents and source remain checked. The original diagnostic and exact-byte validations are in `issues.md`/`artifact_checks.json`.

`matrix_binding.json` explicitly reuses the existing77-case matrix through primary `profile01/stream_probe/matrix_binding.json`. Comparison SHA256 `7a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6`; complete77-case comparison, all77 identical, all missing/failure lists empty. `frontier`, `data`, `examples`, `tests/e2e/refactor_fidelity`, `tests/integration/canonical_case` tree identities match capture `4a1b53e6033134c59b4561306eb2d586bd3d8300`, with no working-tree differences in these domains. No new capture.

## Verification limits

The separate `shutdown_timings` start/end values are process-local monotonic diagnostics. Existing `shutdown_order` meaning and post-stop marker are preserved. `stop_server` remains unchanged: three timed SIGINT/SIGTERM/SIGKILL attempts then an unbounded final wait. Finite peer wait does not guarantee head completion, and local clocks do not prove cross-host order.

These CPU controls do not establish cloud listing causality, live GPU shutdown, native profiler coverage guarantees, stream/DP pairing or kernel costs. No CPU-overhead interpretation or auxiliary hard gate was introduced. Root review, integration and actual image/package/GPU verification remain pending outside this slice.
