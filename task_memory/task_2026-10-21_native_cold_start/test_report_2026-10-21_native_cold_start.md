# Native cold-start test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded RED/GREEN, focused regressions and immutable-input checks. |

## Environment and exact final command

Host `kun-workspace-vgen2`; cwd `/data/ycfeng/Frontier/.worktrees/native-cold-start`; Python `/data/ycfeng/envs/asyncopd-py312/bin/python` version 3.12.13; simulator subprocess `/data/ycfeng/envs/frontier-py310/bin/python` version 3.10.6. Existing environments only; no conda activation/package installation.

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_native_cold_start.py tests/unit/test_native_streaming_replay.py tests/unit/test_native_pd_worker.py tests/unit/test_native_pd_manifest.py tests/unit/test_calibration_pd_harness.py tests/unit/test_native_groundtruth_check.py tests/unit/test_native_pd_canonical_http.py tests/unit/test_canonical_client_metrics.py --basetemp=/data/ycfeng/tmp/native_cold_start_final_03 --junitxml=/data/ycfeng/tmp/native_cold_start_final_03.xml
```

Expected: healthy captured replay and successor new proof pass; missing/wrong/nonzero/existing-client/stale/failed-warmup cases remain unverified; native exceptions and later phases continue; existing client metrics unchanged. Actual: **241 passed / 21 skipped, 113.62 s, exit0**. Nineteen skips: unavailable approved C26 source fixture; two: unavailable legacy checker baseline. Retained JUnit: `focused_regressions.xml`. Captured replay/verdicts: `captured_response_replay.json`, `verification_evidence.json`.

## RED and iterations

Captured-response RED used Python3.12 `importlib.util.spec_from_file_location` on actual `pd_replay.py`, patched only `aiohttp.ClientSession` to return `/data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_snapshot_20261021a/{prefill,decode}_metrics.txt`, and called `asyncio.run(observe_idle({'prefill':'http://prefill','decode':'http://decode'},2,.01,.001))`. Actual: `verified=False`; both roles raise `Missing native running/waiting/KV gauge for a declared lane`; assertion that approved cold start should pass fails. Initial durable test separately fails on the not-yet-implemented dedicated-service seam.

Iterations are in `progress.md`. Initial interpreter attempts found no `python` alias, no pytest in the observer venv, and an installed `tests` package shadowing this repo in `oneshot-opd-py312`. A temporary in-process namespace bridge allowed early RED/GREEN without changing packages. The broader attempt `/data/ycfeng/tmp/native_cold_start_final_01.log` reports 12 failures/14 errors/21 skips due to fresh subprocesses retaining that unrelated installed namespace. Switching to existing `asyncopd-py312` (no shadowing package) and using the existing py310 simulator interpreter fixes the test environment; no production fallback was added. Raw attempts remain under `/data/ycfeng/tmp/native_cold_start_*`.

Focused iterations: 36 PASS / 9 FAIL; then 59 PASS / 1 FAIL; source-specific fixture fixes described in progress. Final targeted successor: 18 PASS in19.94s. Final suite above supersedes those attempts.

## Identity and limits

`source_identity.json`: canonical exact-byte comparisons against `git show 68f891b:<path>`; root checker fixed SHA256; native collector compared to `git show e55c83c:<path>`; AST successor diff restricted to the two native-boundary functions; five production/test files, 453 changed lines. `git diff --check` passes.

All observed numbers are CPU harness verification, not native GPU correctness, performance, or a Frontier-vLLM accuracy result. No predicted/actual simulation comparison was performed. The old actual V4's first failure and 431 completions are read-only observations, not replay qualification.
