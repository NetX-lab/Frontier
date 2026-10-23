# Root verification of repaired native collector

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded eight independent suite runs on fixed f8621a8; integration and final source review follow below. |

## Source and environment

Owner worktree: `/data/ycfeng/Frontier/.worktrees/v4-native-serve-profile`.
Source: `f8621a8258ffa445abfb51f5ea73258ad271494e`.
Python: `/data/ycfeng/envs/asyncopd-py312/bin/python`,3.12.13; pytest9.0.2.
Environment: `PYTHONPATH=.`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`; pinned native checkout is `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. R4 checker resolves from the current tested worktree and is SHA-bound.

Command:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/glm53_dsv4/verify_native_collector_source.py --commit f8621a8258ffa445abfb51f5ea73258ad271494e --collector-count 28 --output /data/ycfeng/tmp/glm53_dsv4/root_native_collector_f8621a8
```

The verifier compares all tested files to committed Git blobs, checks unchanged FullModelInventory and frozen R4 bytes, runs each suite in its own process, stores exact argv/timestamps/log/XML digests, and rechecks source after execution. Its script digest is in the resulting receipt.

## Observed results

Each suite had zero errors, failures and skipped tests. Counts are separate and never summed.

| Suite under tests/unit/ | Expected | Actual | Result |
| --- | --- | --- | --- |
| test_native_serve_profile.py |28|28|PASS|
| test_native_serving_inventory.py |28|28|PASS|
| test_native_v2_serving_inventory.py |45|45|PASS|
| test_native_reader_binding.py |135|135|PASS|
| test_native_pd_worker.py |16|16|PASS|
| test_dp_placement_harness.py |75|75|PASS|
| test_calibration_pd_harness.py |68|68|PASS|
| test_native_pd_manifest.py |14|14|PASS|

Receipt: `/data/ycfeng/tmp/glm53_dsv4/root_native_collector_f8621a8/receipt.json`.
Run interval:2026-10-23T06:20:10Z–06:21:25Z. Each suite's JSON/log/XML remains beside the receipt.

Owner RED/GREEN controls and final source digests are in `../collector/repair_receipt.json`. They record the five finding repairs; the root suite above independently executes the final regression module. Installed-image31/31source-symbol evidence is the owner's CPU/static control, not root execution or actual GPU hook qualification.

## Boundaries

Current result is source/CPU validation. No driver, GPU allocation, per-kernel cost or serving acceptance is supplied. Profile output remains UNVERIFIED; actual GPU stream mapping, PDL correlation, compile/breakable survival, captured-segment ownership and CUPTI restart need future runtime evidence. Original d94 root preflight error is retained in `review_2026-10-23_initial.md`; no tests ran in that failed attempt.
