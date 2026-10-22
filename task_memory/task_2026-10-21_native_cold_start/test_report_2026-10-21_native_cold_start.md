# Native cold-start test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded RED/GREEN, focused regressions and immutable-input checks. |
| 2026-10-22 | Recorded P1 RED/GREEN, complete predicates and latest focused rerun. |

## Environment and exact final command

Host `kun-workspace-vgen2`; cwd `/data/ycfeng/Frontier/.worktrees/native-cold-start`; Python `/data/ycfeng/envs/asyncopd-py312/bin/python` version 3.12.13; simulator subprocess `/data/ycfeng/envs/frontier-py310/bin/python` version 3.10.6. Existing environments only; no conda activation/package installation.

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_native_cold_start.py tests/unit/test_native_streaming_replay.py tests/unit/test_native_pd_worker.py tests/unit/test_native_pd_manifest.py tests/unit/test_calibration_pd_harness.py tests/unit/test_native_groundtruth_check.py tests/unit/test_native_pd_canonical_http.py tests/unit/test_canonical_client_metrics.py --basetemp=/data/ycfeng/tmp/native_cold_start_final_03 --junitxml=/data/ycfeng/tmp/native_cold_start_final_03.xml
```

Expected: healthy captured replay and successor new proof pass; missing/wrong/nonzero/existing-client/stale/failed-warmup cases remain unverified; native exceptions and later phases continue; existing client metrics unchanged. Initial actual: **241 passed / 21 skipped, 113.62 s, exit0**; this omitted worker/lifecycle/terminal assertions and is not full-checker admission proof. Latest P1 rerun: **241 passed / 21 skipped, 132.05 s, exit0**. Nineteen skips: unavailable approved C26 source fixture; two: unavailable legacy checker baseline. Retained latest JUnit: `focused_regressions.xml`. Captured replay/verdicts: `captured_response_replay.json`, `verification_evidence.json`.

## P1 exact reproduction and verification

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_native_cold_start.py::test_successor_full_checker_rebinds_pinned_launch_receipts --basetemp=/data/ycfeng/tmp/native_cold_start_p1_red --junitxml=/data/ycfeng/tmp/native_cold_start_p1_red.xml
```

Added `native_workers` PASS before fixing successor. Actual RED:1 failed in2.58s; `FAIL: UNVERIFIED: ValueError: Head readiness endpoint mismatch: host0`; request/idle PASS. Exact retained result: `p1_predicates_red.json`, JUnit `p1_red.xml`.

Changed only the task-local successor and existing test. The source-pinned `dedicated_services` validator still checks every readiness field against launch/node/manifest/endpoint evidence; the worker predicate compares the full validated receipt instead of the old two-field object only for new proof. No metadata-validation projection/fallback. Legacy exact equality remains unchanged.

The same targeted command with `native_cold_start_p1_green` basetemp/XML was1 PASS in18.88s. All16 missing/wrong values for eight readiness fields rejected request/idle and worker predicates; both strict legacy controls match frozen checker.

Latest full focused command:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_native_cold_start.py tests/unit/test_native_streaming_replay.py tests/unit/test_native_pd_worker.py tests/unit/test_native_pd_manifest.py tests/unit/test_calibration_pd_harness.py tests/unit/test_native_groundtruth_check.py tests/unit/test_native_pd_canonical_http.py tests/unit/test_canonical_client_metrics.py --basetemp=/data/ycfeng/tmp/native_cold_start_p1_final --junitxml=/data/ycfeng/tmp/native_cold_start_p1_final.xml
```

Result241 PASS/21 SKIP,132.05s. `p1_review_receipt.json` records every predicate's expected/actual status: seven PASS, `rank_runtime_graph` FAIL (`KeyError: 'vllm/v1/cudagraph_dispatcher.py'`), overall FAIL. These are fixture results only, never a native qualification claim.

## RED and iterations

Captured-response RED used Python3.12 `importlib.util.spec_from_file_location` on actual `pd_replay.py`, patched only `aiohttp.ClientSession` to return `/data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_snapshot_20261021a/{prefill,decode}_metrics.txt`, and called `asyncio.run(observe_idle({'prefill':'http://prefill','decode':'http://decode'},2,.01,.001))`. Actual: `verified=False`; both roles raise `Missing native running/waiting/KV gauge for a declared lane`; assertion that approved cold start should pass fails. Initial durable test separately fails on the not-yet-implemented dedicated-service seam.

Iterations are in `progress.md`. Initial interpreter attempts found no `python` alias, no pytest in the observer venv, and an installed `tests` package shadowing this repo in `oneshot-opd-py312`. A temporary in-process namespace bridge allowed early RED/GREEN without changing packages. The broader attempt `/data/ycfeng/tmp/native_cold_start_final_01.log` reports 12 failures/14 errors/21 skips due to fresh subprocesses retaining that unrelated installed namespace. Switching to existing `asyncopd-py312` (no shadowing package) and using the existing py310 simulator interpreter fixes the test environment; no production fallback was added. Raw attempts remain under `/data/ycfeng/tmp/native_cold_start_*`.

Focused iterations: 36 PASS / 9 FAIL; then 59 PASS / 1 FAIL; source-specific fixture fixes described in progress. Final targeted successor: 18 PASS in19.94s. Final suite above supersedes those attempts.

## Identity and limits

`source_identity.json`: canonical exact-byte comparisons against `git show 68f891b:<path>`; root checker fixed SHA256; native collector compared to `git show e55c83c:<path>`; AST successor diff restricted to the two native-boundary functions; five production/test files,495 cumulative changed lines. Raw `cumulative_code_vs_parent.diff`, `p1_fix.diff` and `p1_red.xml` preserve exact Git/JUnit output, including blank-context spaces; source/document whitespace checks exclude these three raw evidence artifacts. `checker_successor.diff` is the frozen-to-current narrow delta.

All observed numbers are CPU harness verification, not native GPU correctness, performance, or a Frontier-vLLM accuracy result. No predicted/actual simulation comparison was performed. The old actual V4's first failure and 431 completions are read-only observations, not replay qualification.
