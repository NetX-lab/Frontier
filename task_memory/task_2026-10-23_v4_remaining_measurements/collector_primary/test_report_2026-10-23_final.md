# Root verification of repaired native collector

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded eight independent suite runs on fixed f8621a8; integration and final source review follow below. |
| 2026-10-23 | Recorded merged64f3c950 suite rerun, joint94 equality, explicit77matrix binding and CPU acceptance. |

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

## Final integration checks

Both independent re-reviews closed all five unique findings without a new finding (`review_2026-10-23_final.md`). Source integrated at `64f3c9509b09279f321d237e3a95cbe904c6dd22` on the main task branch. All five source paths remain byte-identical to the reviewedf8621a8 blobs.

From `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, same environment:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/glm53_dsv4/verify_native_collector_source.py --commit 64f3c9509b09279f321d237e3a95cbe904c6dd22 --collector-count 28 --output /data/ycfeng/tmp/glm53_dsv4/root_native_collector_merged_64f3c950
```

PASS,2026-10-23T06:33:31Z–06:34:58Z. Each of the eight table rows above reproduced its expected count, again with zero failures/errors/skips. These are separate repeated suite observations, not an added pass total. Receipt/log/XML directory: `/data/ycfeng/tmp/glm53_dsv4/root_native_collector_merged_64f3c950/`; each XML digest was independently rehashed after completion.

Joint automatic-input argv, invoked under `env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode` with Python3.10.6:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/check_dsv4_measured_scopes.py --accepted-prefill /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json --prefill-package /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced --attention-input /data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_candidate02 --output /data/ycfeng/tmp/glm53_dsv4/root_input_collector_64f3c950.json
```

PASS: candidate admission hash unchanged;94positive scopes, serving_admission=false, entire parsed receipt exactly equals `/data/ycfeng/tmp/glm53_dsv4/root_attention_input_merged_732406aa_scopes.json`. The isolated input's accepted source hashes also remain unchanged.

Default matrix: `merged_matrix_binding.json` verifies all simulator/data/examples/fidelity-driver bytes unchanged from4a1b53e6 and binds the existing77/77comparison. The six non-record changes are the five H paths plus standalone `tests/analysis/v4_kernel_coverage.py`; no native-collector/replay imports were found in `frontier/` or fidelity drivers. No new capture, native-performance comparison or runtime qualification is asserted.

Final disposition: `primary_acceptance.json`, **ACCEPTED_CPU_COLLECTOR_COMPONENT**. Source/receipt/evidence digests were verified before its exclusive creation. An initial partial-line documentation patch failed matching before applying; it was replaced with exact-context hunks, with no code or evidence change.

## Handoff document verification

Root inline verification on the integrated worktree passed59checks (`handoff_verification.json`): accepted source/evidence hashes, all owner repair-evidence hashes, exact scope and separate counts, current/historical status separation, modification-history/conflict checks, retained original request, G2 retention boundary and absence of code changes. The merged suite receipt is retained byte-for-byte as `merged_suite_receipt.json`; raw XML/logs stay at their recorded scratch paths.

The independent W128-H-CLOSURE-DOCS writer synchronized exactly five parent milestone documents, then reported205/205focused checks and85/85post-write checks, plus whitespace exit0. Those numbers are writer evidence, not additional root test executions. Root inspected the complete five-document diff: current H acceptance, historical markers, original tails, G2 boundaries and artifact links are consistent; no source or approval change. Writer stopped before final staging. Root and parent document checks are reported separately and do not extend GPU qualification.
