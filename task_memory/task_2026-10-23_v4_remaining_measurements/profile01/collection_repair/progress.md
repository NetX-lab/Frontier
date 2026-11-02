# W128-PROFILE03-COLLECTION progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Committed verified source20660f4c; finalized matrix/source binding and prepared records-only archive commit. |
| 2026-10-24 | Bound the77-case unchanged domains; froze two source files after an auxiliary snapshot interruption and reran verification. |
| 2026-10-24 | Reproduced RED on unchanged worker, implemented the bounded repair and obtained36GREEN tests. |
| 2026-10-24 | Read authority, primary controls and exact source; started RED controls. |

## Status

- completed: worktree/branch/base and initially clean tree verified. Both source files match the allocated base; worker SHA256 matches the original root controls.
- completed: read applicable AGENTS, glossary, relevant ADR, primary requirements/plan/profile03 design and lifecycle evidence. Root already reproduced the two lifecycle defects; diagnosis starts by reusing these deterministic controls rather than speculative source changes.
- completed: separate unchanged baseline worker19PASS and manifest14PASS.
- completed: source-preserving RED: worker SHA unchanged;19PASS/16FAIL. Overrides stayed3, marker20 was not observed by10, stale listing made wait20 despite four existing paths; invalid values were accepted. Missing timing fields caused the expected diagnostic-control failures.
- completed: production repair through existing node profiler and wait/stop paths; no native escalation or auxiliary hard gate added. GREEN:36PASS, no failures/skips, including active headless and propagated native stop exception.
- completed: final frozen affected suites36/14/33/39/18/1/68/75PASS, separately; actual exits0, zero errors/failures/skips, source bytes preserved. Exact commands and JUnit/log hashes are in `final_checks.json`.
- completed: explicit unchanged-domain77-case matrix gate and final gross budget30/80 production,129/180 tests,159/260total.
- completed: verified source commit `20660f4c0776ddd75dc211f1f0819266b48ba23c`, exactly the two allocated files. `source_manifest.json` and `matrix_binding.json` bind final source hashes/commit to the final CPU checks and existing matrix.
- completed: all owned archive contents and evidence hashes are ready. The final records-only commit and clean-tree observation are reported in the owner response; no in-scope source/runtime work remains. Root independent review/integration remain outside this slice.

## Initial evidence

- `git rev-parse HEAD`: `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`; branch `v4-profile03-collection`; no initial changes.
- Worker SHA256: `9ec701ec12f7a5da0f442f994a353f98c5ae1d2880b9feca38106ea536fd5cab`.
- Existing tests SHA256: `ed20f690b6d5515ac21bd1a238b14fbcf94a00f8f1d3c91ea21ef52d433161dc`.
- CPU executable: `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13.
- Inspected unchanged `vllm_replay.stop_server`: already-returned fast path; SIGINT/SIGTERM/SIGKILL each with the supplied timeout; final unbounded wait. No wrapper or stop escalation change is needed.

## Repair evidence and reasons

- Baseline and RED/GREEN exact commands and logs: `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE03-COLLECTION/{baseline_commands.json,red_command.json,green_worker_command.json}`; raw logs in the same directory.
- Role values7/5 in tests establish configuration selection without baking later package512/64 into production. The per-role/lane profiler dictionary is copied; absent roles preserve the original max_iterations.
- Known peers already exist in `node_plan`; explicit paths remove dependency on drain directory enumeration for both wait and membership. The existing `await_condition` still enforces failed peers, active native server and finite peer deadline.
- Headless post-stop-marker wait now uses the existing peer budget, not the native signal timeout. Natural/active headless marker20 controls both wait20; late/missing markers remain a soft observation, and native23 remains worker1.
- `shutdown_timings` contains local monotonic start/end dictionaries for `drain_quorum_wait`, `head_marker_wait`, `native_stop`. Existing `shutdown_order` fields and post-stop marker format are unchanged. Timing is diagnostic only; no CPU-overhead or cross-host-order inference.
- Initial GREEN gross budget: production24add/6delete=30/80; tests115add/7delete=122/180; total152/260. Final test assertions later brought tests to129gross, total159.
- Key-step check-progress: `checkpoint.md`; no drift. Continue only affected verification, binding and closeout; no extra source/runtime work.

## Final assertion strengthening and verification interruption

- Added only test assertions that the planned profiler retains all unrelated fields and that a role override does not affect CUDA/URI profiler paths. Production source is unchanged. Final source SHA256s: worker `59cdd58d9fce0d3aa432caeace4411d8fef074f59857260e81c75d3441bcfa1b`, tests `84fa7a4be7a8a957eb62671e652d6e00ff48d153c92671bea7a53fe6528cea4c`.
- The first eight-suite batch has passing XML counts36/14/33/39/18/1/68/75, separately, but its final source-snapshot assertion failed because those assertions were edited while it ran. Original aggregate exit1 and unretained process exit-code limitation are recorded in `issues.md` and `verification_interruption.json`. No business test failure was observed.
- Corrected route: no further source edits; frozen worker rerun36PASS with exact before/after hashes; seven affected suites rerun separately to retain actual exit codes. This closes the evidence gap rather than changing business behavior.
- Matrix gate: original comparison SHA256 `7a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6`; all77 identical, complete comparison and empty missing/failure lists. All five domain trees match capture `4a1b53e6`; final source-commit binding follows the source commit. No new capture.

## Delivery state

- `final_checks.json`: actual frozen-source exits0 and no failures/errors/skips for all eight suites; SHA256 `5b9e9abde333bd28aec250844d90430270fda8a024dca2d0ad2088fb414dd74f`.
- `matrix_binding.json`: final source `20660f4c`, existing unchanged-domain77-case binding; SHA256 `90e666851baabb0e3b5627ef97883da869720fd9e4103f48820b0bcf50da3c59`.
- `source_changes.diff`: exact allocated-base-to-source-commit diff. `artifact_checks.json` records validation against a separate temporary Git index and source/record checks; it never changes the active index or worktree.
- The records-only staged whitespace check initially reported two unchanged pytest console lines in `red_worker.log` and exited2 before commit. Preserve exact raw bytes; validate raw-log identity separately and exclude only this log and the independently validated raw diff from ordinary staged whitespace checks. Original diagnostic and the exact lines are retained in `issues.md`/`artifact_checks.json`.
- Records are ignored by the repository's unchanged `.gitignore` rule `task_memory/*`; only this owned `collection_repair/` directory is force-staged for the archive commit. No unrelated path is staged.
- `summary.md` is the English archive; `record_hashes.json` hashes all delivered records except itself. Final archive commit ID and clean-tree result are reported in the owner response, avoiding a self-referential commit hash inside its own content.
- Final check-progress in `checkpoint.md`: source, verification and matrix PASS; only the mechanical records commit/clean-stop action remains. No new scope or approval is inferred.

## Root next actions, outside this slice

1. Independently review the exact two-file diff at `20660f4c`, using the local requirements and final evidence.
2. Integrate the source and records commits onto the root branch; DP ownership remains separate.
3. Prepare the later explicit profile03 package and actual image/runtime verification; package512/64 and record600000 values remain unchanged/not applied here. This owner stops after its clean archive commit.
