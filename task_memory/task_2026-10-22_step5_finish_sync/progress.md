# D30 isolated implementation progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Complete focused source/CPU checks; record adjacent fixture blocker and parent handoff. |

- completed: read authoritative approval and D30 docs before any changes.
- completed: clean worktree/branch at requested fa5bae2; parent edits preserved.
- completed: registered Step family, engine-wave owner and R2 install/hash/error
  path inspection; bounded design and human receipt recorded before code edits.
- completed: focused RED→GREEN scheduler/observer verification.
- completed: bounded source/test commit
  `cf1a53f63eef0492c369517f38505d7d7283c4df` (4files319insertions/8deletions).
- in-progress: separate records commit and final parent handoff.
- pending: parent disposition of five newly failing direct-lane PD fixtures,
  phase review and integration; no completed whole-task claim.
- pending: parent strict77 and live native counter/reset validation.
- environment observation: login shell has no `python`; inspected parent records
  identify existing `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6). No host
  dependency/configuration changes are needed.

Initial checkpoint: design advances the requested seam, no new authority or
native acceptance. Stop further owner exploration now; next is focused RED.

- completed: scheduler RED 9 failures/7 passes → GREEN16; native scalar observer
  RED5 failures/1 pass → GREEN6. Logs `/data/ycfeng/tmp/d30_finish_sync_red03.log`,
  `d30_finish_sync_green01.log`, `d30_finish_observer_red01.log`,
  `d30_finish_observer_green01.log`. Initial60s RED matrix was interrupted; full
  rerun finished in65.18s with no source change.
- completed: existing native-family legacy32 and DP1 runtime checks3 PASS,
  `/data/ycfeng/tmp/d30_finish_legacy_runtime01.log`.
- observed: initial DP1 Step fixture at TP8 cannot fit the unchanged H800 memory
  budget (81,432,197,504 parameter bytes >77,309,411,328 requested). Test changes
  only to valid TP16 (half the shard); no memory guard relaxation or runtime fix.
- observed: adjacent R2 suite on frontier-py310 lacks aiohttp:120PASS/7SKIP/1FAIL.
  Existing asyncopd-py312 Python3.12.13/pytest9.0.2/aiohttp3.13.3 imports pass;
  rerun there, no dependency install. Broad GPU/CPU overhead work stays deferred.

- observed: TP16 through the old `_replica_config` fixture first constructs
  DP4/EP64, which does not divide352 experts; corrected only the test to construct
  legal DP1 directly. Three DP1 role checks PASS; production guards unchanged.
- completed: final27 focused checks PASS65.58s, including execution of the exact
  SHA-pinned native finish-check method extracted by AST (no vLLM import/GPU).
  Final adjacent R2 128PASS1.98s; legacy32/DP1 runtime3PASS16.10s. Counts overlap
  earlier iterations; final distinct total158 across these three suites.
- completed: four source/test files,327 changed lines (261 new-test lines plus
  66 existing-file insertions/deletions), all changed modules below2000 lines;
  `git diff --check` PASS. Native config/core SHA256 remain parent-pinned values.
- blocked: extra adjacent PD run68PASS/6FAIL; baseline scheduler supplied through
  a read-only import loader73PASS/1FAIL. Five new failures are direct-lane tests
  that never drive peer finish consensus; one READ-byte expectation fails on
  baseline too (actual102203904 vs expected103051776). See `issues.md`.
- decision: stop additional source/test edits before exceeding five files.
  Keeping the dedicated focused test module and adapting both affected PD test
  modules requires six source/test files, estimated <=450 changed lines. Parent
  owns the expansion/disposition because it is combining D29 and already edits
  `test_step5_measured_pd.py`. No test suppression or behavior bypass is added.

Next actions:
1. Parent either authorizes six-file scope or integrates these candidate commits
   with its existing D29 fixture corrections; drive every real DP lane in the
   five affected tests, preserving request/READ assertions.
2. Parent review/common merge and strict77; do not waive current adjacent failure.
3. Parent fresh R2 stage/pin/package and approved native run: verify every DP lane
   step1/16 return, active peer continuation and reset at next input-queue entry.
