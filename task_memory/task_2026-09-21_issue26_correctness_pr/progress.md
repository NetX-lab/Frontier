# Issue 26 Correctness PR — Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Publication recorded: `6828581` and `311e076` pushed; PR 35 body updated for the workflow results. |
| 2026-09-24 | Workflow `wf_7606e14e-f10` results reconciled: six surviving findings mapped to existing fixes; the 32 unverified leads triaged; W6-R6 fixed in `6828581`; W6 and S9-08 records corrected; W7-R4 proposal; W9-04 reach beyond PP=1 recorded; case errata and S43/S42 plans updated. |
| 2026-09-24 | User decisions after G5 recorded (C4 option a; S43 and S42 become separate tasks; W9-05 worktrees removed). Review of the branch's fixes: ten commits `c647e95`..`6aee289`, proposals listed, dummy-mode question answered, candidate row S44 recorded, records corrected, S43/S42 task directories created. |
| 2026-09-23 | G4 complete (`exp-0923-230103-591735`, extraction PASS). G5 complete: T1 48/48 formal routes MATCH at PP2 (C3 PASS). C4 `SCENARIO_NOT_REACHED` in all four bursts. The third GPU job is not used. `workflow_gap_status` PASS with corrections S43/S42 pending review. Plan §18.21. |
| 2026-09-23 | G3 complete (`exp-0923-221233-009652`, extraction PASS, T1 replay 38/38 MATCH); G4 inputs amended per plan §18.20; launcher wrapper and poller fixed. |
| 2026-09-23 | G2 complete: harness under `tests/comparison/dp_placement_pp/`, case inputs, semantic table `PASS`, manifest ids filled, Frontier pre-check discriminates; plan §18.19. |
| 2026-09-23 | User direction "授权上述1-2，推进W9-05": W9-04 worktrees removed; W9-05 diagnosed and fixed (`75c1140`), B1–B8 pass; G1 write-through amendment `63ac6c6b9`; G2 started. |
| 2026-09-23 | User decisions: W9-04 fixed in this PR, W9-05 deferred, P5 worktrees removed. W9-04 fix `2ffb062` with regression case; A1–A7 pass. |
| 2026-09-23 | Step 9 P6 done: records and AGENTS.md scope committed (`c1a570d`, review wording `104b6ff`), pushed, PR 35 body updated with the W9 section and read back. Step 9 CPU packages complete; W9-04/W9-05 decisions pending. |
| 2026-09-23 | Step 9 P4 committed (`bacdbb4`) and P5 completed: C2 holds (24 of 24 PP=1 policy scenarios, 71 of 71 fidelity cases, 16 of 16 examples identical; 0 suite regressions). P5 found W9-04 (placeholder/join deadlock at `attn_dp=4`) and W9-05 (requests lost under KV pressure, also on `main`); W9-04 fix decision pending with the user. |
| 2026-09-23 | D9-2 decided (group-anchored key); P2 implemented and P3 unit tests added (132 targeted tests pass; the 19 new or changed cases fail on the pre-P2 tree). |
| 2026-09-23 | Step 9 P1(b) completed on seven shapes; D9-2 proposal recorded (design.md, plan §18.15); W9-03 observation; awaits the user's D9-2 decision. |
| 2026-09-23 | W9-01 merge-forward: `origin/main` merged (`dd9b8d9`); composition check K1–K4 pass on `03d5f24` (drain-reader fix); Step 9 P1(b) resumed. |
| 2026-09-23 | PR 36 pre-merge untrack (P6, `4d08c5d`) done; W9-01 copies refreshed. Merge-forward waits for the PR 36 merge. |
| 2026-09-23 | PR 36 round-2 review remediation recorded; the W9-01 composition check extended to PR 36 groups G9 and G10. |
| 2026-09-23 | W9-01 fixed on `fix/stage-admission-ordering` (draft PR 36); Step 9 PP>1 packages stay paused until it merges forward and passes G3b. |
| 2026-09-22 | Step 9 execution started: P1(a) oracle completed, P1(b) blocked by W9-01, design checkpoint partially settled, G1 ground-truth instrumentation completed and case binding written. |
| 2026-09-22 | FP8 native rerun PASS (`exp-0922-202645-561899`, 8 passed, exit 0). Second Step 9 plan review (plan §18.12, R9-01..R9-08) recorded; Step 9 still not started. |
| 2026-09-21 | Step 0 started: records landed, environment created, baseline pending. |
| 2026-09-22 | Maintainer review dispositions recorded; Checkpoint C closed: parent merged, W2 tests strengthened, W2 re-measured with one harness revision. |
| 2026-09-22 | Checkpoint D first half: W3, the shared monolithic forward lifecycle, implemented, tested against four deliberate-defect controls, and committed as `65ed8a7`. |
| 2026-09-22 | W3 fidelity matrix measured: 71 of 71 identical against the expectation recorded before the run. Step 3 closed. |
| 2026-09-22 | Checkpoint D second half: W4, the opt-in vLLM-style DP placement policy, implemented and committed as `10dd474`; measured against five deliberate-defect controls and a 71-of-71 identical fidelity matrix. Step 4 closed. |
| 2026-09-22 | Checkpoint E second half: W6 arithmetic repaired and CPU-validated; measurement ownership decided; native GPU validation and artifact identity left open. |
| 2026-09-22 | Checkpoint E first half: W5 closed as NOT PORTED by user decision after the premise check showed the collision unreachable on main; the drafted implementation was reverted before commit and archived as a patch. |
| 2026-09-22 | W6 artifact identity decided as document-only and written into the profiling guide; native parity test added and submitted to an H800 worker as `exp-0922-140423-075005`. |
| 2026-09-22 | W7 investigated while the GPU job queued: candidate gitlink unpublished, three payload defects confirmed by execution, companion-repository decision pending. |
| 2026-09-22 | W6 native parity PASS on H800: `exp-0922-145047-660565`, 8 of 8 at `rtol=0, atol=0`. Step 6 closed. |
| 2026-09-22 | W7 authorized and delivered: companion fix published as `eb7bc4f` with draft PR 1, Frontier gitlink moved, Frontier-side test added, and the governance scans narrowed to Frontier-owned sources. Step 7 closed. |
| 2026-09-22 | Step 8 §14.1 run: unit and integration suites at the baseline failure set, 16 architecture examples, four PP=2 cases, and a cold-then-warm predictor-cache pair. `tests/debug/` pointer defect found, deferred to `future.md`. |
| 2026-09-22 | Step 8 closed: §14.2 review recorded, PR 35 body updated with the Step 8 results and record links, `summary.md` written. Task technically complete; PR stays draft for user review. |
| 2026-09-22 | External review of PR34/PR35 applied (packages A–E; F excluded by the user): C34-01 merged in from the PR34 branch; C35-01 fixed with unit and real-loop hybrid-layer coverage and a negative control; C35-03/04 test wiring and optional-torch skip; C35-02/05 wording and record consistency; P9-01..06 folded into `plan.md` §18 and `design.md` W9. Step 9 remains planned, not started. See `test_report_2026-09-22_review_corrections.md` and the disposition table in `review.md`. |

## Status

| Field | Value |
| --- | --- |
| Correctness branch | `fix/issue26-correctness-pr` (worktree `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`) |
| Base at creation | `refactor/oversized-module-split` @ `41dabfb9d5ef3b51cdf3009d486450515d9a8d2d` (itself on `origin/main` `1f694f7`) |
| Prerequisite | MET. All four modules this PR edits are under the 2,000-line gate. The split's final record is 71 of 71 fidelity cases identical with no predictor cache differences, taken with the corrected gate; see the refactor task's Checkpoint B report. |
| Current step | Step 9 CPU packages P1–P6 complete (P2/P3 `2ffe78d`, P4 `bacdbb4`, records `c1a570d` and `104b6ff`). W9-04 fixed (`2ffb062`), A1–A7 pass. W9-05 fixed (`75c1140`), B1–B8 pass. G3–G5 complete, 2 of 3 authorized jobs used. C3 PASS. C4 `SCENARIO_NOT_REACHED` in all four bursts (plan §18.21), accepted by the user (option a). Fix review done 2026-09-24: `c647e95`..`6aee289` (plan §18.22, `test_report_2026-09-24_fix_review.md`). Workflow `wf_7606e14e-f10` reconciled 2026-09-24: W6-R6 test `6828581`, record corrections W6-R7 and W9-R4, W7-R4 proposal (test report §10). |
| Publication | PUSHED_VERIFIED 2026-09-24: `6828581` (test) and `311e076` (records) on top of `5e7221d`; remote head `311e076` equal to local after fetch. PR 35 body PATCHed 2026-09-23T19:47:38Z (UTC): the W6 native-parity and negative-control wording, the W9-04 reach beyond PP=1, the `6828581` row and workflow note in "Review of the landed fixes", W7-R4 in the proposals, the commit table and Status. Read back identical to the sent body; still draft. Backup: `/data/ycfeng/tmp/issue26-correctness-pr/pr_body_20260924b/pr35_body_before.md`. Companion PR 1 body unchanged. Earlier: PUSHED_VERIFIED 2026-09-24: remote head `b110eca` equal to local (review commits `c647e95`..`6aee289` plus the record commit). PR 35 body PATCHed 2026-09-23T18:41:47Z (UTC): Progress rows (W6, W7, W9-05, a Review row), the W2/W3 reachability corrections, the native FP8 `block_shape=[128, 128]` correction, W7's second commit and rewritten Frontier tests, the G5 decision, a section "Review of the landed fixes (2026-09-24)", the review material and commit tables, and Status. Read back identical apart from a trailing newline; still draft. Companion PR 1 body PATCHed for `ff11ee6` (second-commit section, 12 tests, scope of the flow-generation change); read back the same way; still draft. Backups: `/data/ycfeng/tmp/issue26-correctness-pr/pr_body_20260924/{pr35,companion_pr1}_body_before.md`. |
| Next action | User decisions on the review proposals (plan §18.22, test report §8): F-R5/F-R6, the strict `sync_entry` predicate, the sync-room alias refactor, S44 as a second S43 row, a native W6 FP8 rerun, the pinned calibration tools, W7-R4 (companion zero all-reduce; option a recommended), and removal of `.worktrees/review-final-{base,head}`. S43 and S42 continue in their own task directories. |

## Step status

| Step | Work package | Status | Test | Publication | User review |
| --- | --- | --- | --- | --- | --- |
| 0 | Worktree, references, baseline | PASS | PASS (baseline recorded) | PUSHED_VERIFIED | NOT_REVIEWED |
| 1 | Candidate/vLLM audit | PASS | n/a (source audit) | LOCAL_ONLY | NOT_REVIEWED |
| 2 | RR DP rotation | PASS | unit PASS (23 tests); matrix PASS against a stated expectation, re-measured 2026-09-22 with one harness revision | PUSHED_VERIFIED | REVIEWED (R35-01 closed) |
| 3 | Shared monolithic forward | PASS | unit PASS (23 new, 3717 total, failure set identical to the parent); integration PASS (real event loop, 4 mixed-phase cohorts); four deliberate-defect controls each fail for their own reason; matrix PASS, 71 of 71 identical against the stated expectation | PUSHED_VERIFIED | NOT_REVIEWED |
| 4 | Opt-in vLLM DP placement | PASS | unit PASS (61 new, 3778 total, failure set identical to the parent); integration PASS (3 cases in the real event loop, including a placement that diverges from round-robin); five deliberate-defect controls each fail for their own reason; matrix PASS, 71 of 71 identical against the stated expectation | PUSHED_VERIFIED | NOT_REVIEWED |
| 5 | Routing implementation identity | CLOSED, NOT PORTED (user decision 2026-09-22) | n/a: no source change; restored files re-run, failure set identical to the parent (torch-missing only) | PUSHED_VERIFIED (records + PR 35 section) | REVIEWED (user chose to keep the single global field) |
| 6 | Legacy fused-MoE profiling | PASS | CPU PASS (7 new tests; HEAD comparison shows the same single environment-dependent failure; default-environment suite unchanged at 84/3778). Native PASS: 8 of 8 in `tests/integration/test_moe_fused_expert_numerical_parity.py` on H800 — seven comparisons at `rtol=0, atol=0` plus the FP8 structural check — job `exp-0922-145047-660565`, re-run with the corrected `block_shape` wiring as `exp-0922-202645-561899` (8 passed), both under `codesign` | PUSHED_VERIFIED (source, tests, docs, records, PR 35 section) | — |
| 7 | Optional zero-payload backend | PASS. Companion fix published as `fwyc0573/frontier-htsim` `eb7bc4f` with draft PR 1; Frontier gitlink moved from `b8518af`; no Frontier source change | Companion 9 passed, negative control 6 of 9 fail on pristine sources. Frontier 4 passed, negative control 3 of 4 fail at the old gitlink. Clean checkout resolves `eb7bc4f` from the published remote, builds, and passes. Suite back to the 84-failure baseline with 3782 passing after narrowing three governance scans to Frontier-owned sources | PUSHED_VERIFIED | — |
| 8 | Combined regression, PR hand-off | PASS | unit 84 failed / 3782 passed with a `FAILED` set identical to the `origin/main` baseline; integration 15 passed / 22 skipped / 5 errors, the errors environmental and identical on the base; 16 of 16 architecture examples pass; 4 of 4 `PP=2` cases pass; cold and warm predictor-cache runs byte-identical | PUSHED_VERIFIED (records + PR 35 body carrying the Step 8 results, the record links and the implementation commits) | REVIEWED (external review 2026-09-22; corrections below) |
| 8+ | External review corrections A–E | PASS | unit 84 failed / 3789 passed / 50 skipped / 10 errors with the `FAILED` set identical to the baseline (+7 passes are the new tests, +1 skip and −1 error are the optional-torch module); mixed-forward unit 26 passed; real-loop hybrid-layer case 2 passed with the negative control failing on the pre-fix source; arithmetic 9 passed under torch | PUSHED_VERIFIED | NOT_REVIEWED |
| 9 | PP>1 support for `vllm_load_balancing` | DONE — P1–P6 completed; G3–G5 complete (C3 PASS, C4 SCENARIO_NOT_REACHED, accepted by the user 2026-09-24, option a); fix review `c647e95`..`6aee289` with final validation PASS; S43/S42 moved to their own tasks | unit and integration PASS; C2 PASS (24/24 policy scenarios, 71/71 fidelity, 16/16 examples identical; 0 suite regressions); C3 PASS (T1 48/48 formal at PP2) | PUSHED_VERIFIED `3526156`; PR 35 body updated | D9-2 decided by the user; W9-04 fixed by the user's decision (`2ffb062`); W9-05 fixed at the user's direction |

## Chronological updates

- 2026-09-22: W6 native parity closed. `exp-0922-145047-660565` on `gpu-h800-0110` under `codesign`, image `vllm/vllm-openai:v0.10.2` through the company proxy: **8 passed in 13.70 s**, every case at `rtol=0, atol=0` against vLLM's own `fused_experts`. Three earlier attempts failed for environment reasons, each now recorded: the `steptron_ci` pool had no capacity; the image ships neither `pytest` nor `nvidia-smi` and the `httpproxy` recipe returns 407 for pip; its injected driver sits at `/usr/local/nvidia/lib64` off the loader path; and `frontier/profiling/common/utils.py` imports `pandas`, which the image also lacks. Worker packages install from `http://mirrors.i.basemind.com/pypi/simple/`. `logs_rjob` returns empty for these jobs; `get_rjob_infos` then `logs_replica` returns the container lines.
- 2026-09-22: W7 delivered under the user's authorization. Companion repository `fwyc0573/frontier-htsim` gained branch `fix/zero-payload-input-handling` at `eb7bc4f` and draft PR 1: `tensor_bytes` now merges through a helper that treats only `None` and `""` as unset, the required-field check became a table carrying per field whether zero is legal, and a negative payload is rejected in the runner and in `Scenario.validate()`. Its `.gitignore` was narrowed from `tests/` to `tests/*` plus the published test, since a bare directory rule stops git descending and makes any negation unreachable. Frontier moved its gitlink and gained `tests/unit/test_collective_sim_zero_payload.py`; no Frontier source file changed. Both negative controls hold: 6 of 9 companion tests and 3 of 4 Frontier tests fail against the pre-fix sources, each reporting `missing required fields: ['tensor_bytes']`.
- 2026-09-22: The gitlink bump exposed a latent defect in the repository governance tests. Three of them walk `frontier/` recursively; with the optional submodule initialized that walk reaches 38 vendored files, one of which does not parse under Python 3, so `test_raw_model_profile_resolution_callsites_are_allowlisted` failed and the other two silently measured vendored code. `tests/frontier_sources.iter_frontier_sources()` now yields the 413 Frontier-owned files and all three scans use it. This would have hit any developer who followed the AGENTS.md instruction to initialize the submodule, with or without this branch.
- 2026-09-22: Checkpoint C. Merged the corrected parent (`bb582a4`, `7dd5982`, `6ef0a3c`) into this branch by merge rather than rebase, so the published review anchors stay valid and no commit is discarded. New base/head relationship: PR #35 head `fix/issue26-correctness-pr`, base `refactor/oversized-module-split` at `6ef0a3c`.
- 2026-09-22: R35-01 addressed. The placement tests now state where each request lands: three topologies with the full rotation written out by hand past its wraparound, driven through the public `schedule()` rather than `_schedule_batch_mode`, each run for both roles that reach batch-mode placement. A guard keeps that role list honest, since `TRANS` also falls through the dispatch but is declared and never constructed anywhere in `frontier/`. The source-text check is kept as governance only and says so. Negative control against the pre-fix method: 12 of 23 fail, controls pass, and the old code turns out to produce the expected sequence exactly for a single burst -- so the fix restored an existing rotation for incremental arrival rather than introducing a policy.
- 2026-09-22: W2 re-measured with harness and source at one revision: baseline `6ef0a3c` against candidate `ceac2b4`, both clean detached checkouts, one harness at `ceac2b4`, no filter, clean cache, 71 executed and 426 cache files each. 71 of 71 compared, 68 identical, and the mismatch set is exactly the three predicted cases, with no provenance findings and no cache differences. Lane occupancy reproduces the first measurement number for number. Full record in `validation.md`.
- 2026-09-22: The `cases.py` remedy wording was corrected. It said prefill placement "has to be validated by unit tests", which understates R35-02: the wrapper limit is not a runtime limit, and full coverage needs a fixture that builds a runtime configuration directly with durations injected at the predictor boundary. That fixture is W3 acceptance work.

- 2026-09-21: Baseline on the shared base recorded in the refactor task's Step 0 report; vLLM reference cloned (no tags in the fork; upstream `v0.10.2` comparison deferred to Step 1).
- 2026-09-21: Draft specification analyzed; eleven facts verified against main, the candidate, the submodule remote, and the host; planning interview settled twelve decisions (see `requirements.md`). Records landed under `task_memory/`, `.gitignore` narrowed, `plan.md` carries the Amendments table.
- 2026-09-21: Step 1 audit complete. Three pinned-source audits landed as `audit_scheduler.md`, `audit_predictor_profiling.md`, `reference_vllm_0_10_2.md`; dispositions and two decision checkpoints recorded in `review.md`. Execution order corrected to W2 first, then W3, then W4, because the candidate's report-order key depends on the shared forward identity. The vLLM fork was confirmed to be a direct descendant of upstream v0.10.2 with the DP placement files byte-identical to the tag.
- 2026-09-21: Merged the completed oversized-module split into this branch. Merge rather than rebase, so the published review anchors stay valid. The modules this PR edits are now `config.py` 788 with `cluster_config.py` 1888, the vLLM V1 replica scheduler 1386, the prediction model manager 722 and the MoE predictor 1557, each with named child modules that give the planned fixes a clear owner. Step 2 is unblocked.
- 2026-09-21: Step 2 implemented. `_schedule_batch_mode` now derives both the replica index and the DP lane from one ordinal that persists across scheduling calls, which is the rotation `_schedule_decode_lane_round_robin` already applies to the unified decode role. The per-replica grouping of the returned mapping is unchanged and is asserted separately.
- 2026-09-21: Four regression tests added to `tests/unit/test_cluster_scheduler_dp_lanes.py`, covering call partitioning, non-contiguous replica ids, DP1 through DP4, an empty scheduling call, and the preserved return order. Sensitivity was verified by stashing the fix and rerunning: three of the four fail on the pre-fix code for the right reason, alternating lanes collapsing to lane 0, and the return-order test passes both ways.
- 2026-09-21: `tests/unit/test_replica_identity_contract.py` pinned the literal expression `dp_id = local_idx % self._replica_dp_size`. The property it guards is that a non-FFN cluster scheduler derives the lane from the Replica-local DP size rather than a global or expert-parallel cardinality. The check now asserts that property over every lane assignment instead of one literal, so it survives a change to the ordinal but still fails if the lane is taken modulo anything else.
- 2026-09-21: Unit comparison against the refactor tip `db15e64` over 73 files: identical failure identities, 1808 to 1812 passing, the four new tests being the difference.
- 2026-09-21: Step 2 measured and PASS against the stated expectation. 71 of 71 cases compared, 68 identical, and the mismatch set is exactly the three cases predicted to move. Nothing moved that was not expected to, and nothing expected to move stayed. Lane occupancy confirms the direction: collapsed onto lane zero before, evenly spread after, with the control case unchanged. Full record in `validation.md`.
- 2026-09-21: Merged the refactor tip so this branch carries the four DP placement cases in its own fidelity case table. Without it the branch's own harness still had the 67-case table, and measuring the branch with its own tooling would have exercised a table that cannot see the fix.
- 2026-09-22: W3 implemented and committed as one unit (`65ed8a7`). A monolithic Replica now keeps one waiting room and one open-step namespace for both local phases, completes a cohort once, restores full-stage owners once, and then continues each source on its own batch through the phase helper it already had. The post_moe collective event class is chosen from cohort contents rather than from whichever lane closed the room, so a pure-prefill and a pure-decode cohort keep exactly the `EventType` priority they have today and only the mixed cohort -- which previously could not complete at all -- is new. The two near-duplicate sync entries collapsed into one `enter_layer_sync(..., mode)`, mirroring the existing `schedule_layer_wave(mode=...)`. I3 is enforced at group formation; I8 is deliberately excluded per Checkpoint D's "current-main metrics ownership" (recorded in `design.md`).
- 2026-09-22: One guard was added that the delegation would otherwise have dropped. Each per-phase helper refuses a legacy aggregate synchronization by checking its batch for the wave's lane timings, but only when it pops the room itself; the shared path hands it `direct_batch`, which skips that branch. The check now lives in `forward_collective`, once per source, against the marker its own phase writes.
- 2026-09-22: W3 acceptance. 23 behavior tests in `tests/unit/test_monolithic_mixed_forward_sync.py` cover every phase pairing including true mixed batches, both arrival orders, unequal source tokens, idle participation, duplicate ownership, successive layers, a decoding request inside a prefill batch, dense-layer transitions inside a MoE model, a missing wave marker, and a disabled metrics store. `tests/integration/test_monolithic_mixed_forward_runtime.py` builds `attn_tp=1, attn_dp=2, moe_tp=1, moe_ep=2` directly and runs the real `Simulator`: 24 cohorts, 4 of them mixed-phase, 4/4 requests completed, identical with reporting on and off.
- 2026-09-22: W3 controls, four trees and four distinct failures. The pre-fix source deadlocks in the real event loop ("Sequential simulation ended with non-empty scheduler state"); borrowed source timing trips "one attention-DP lane cannot occupy two open sync cohorts"; a repeated layer advance trips "Decode post_moe layer counter cannot advance"; a repeated ownership restoration trips "operation_id is already queued or active". On the pre-fix tree the same-phase pairs still complete, so the suite is not failing wholesale for an unrelated reason -- the mixed pairs fail at the empty collective list, which is the deadlock itself.
- 2026-09-22: W3 regression comparison against the branch parent `3d47417` in a dedicated detached worktree. `tests/unit`: 84 failures on both sides with identical identities, 3717 vs 3694 passing. `tests/integration`: the same five pre-existing errors (the PD-AF Reference checkout is absent on this host), 12 vs 11 passing. No regressions and no accidental fixes.
- 2026-09-22: W3 controls rebuilt against the final test file and re-run, so the recorded counts match what is delivered. Each tree now carries the final `frontier/` and the final `tests/` and differs from the delivered source by exactly one edit, except the baseline tree whose `frontier/` is the pre-fix parent in full. Counts: 22, 12, 2 and 16 of 23 unit tests fail respectively, and each runtime failure is distinct. The pre-fix tree's same-phase pairs pass every assertion about the forward itself and fail only at the final shared-room inspection; the mixed pairs fail earlier at an empty collective list, which is the deadlock isolated.
- 2026-09-22: W3 fidelity matrix PASS. Baseline `3d47417` against candidate `65ed8a7`, both clean detached checkouts with `source_dirty=False` and no dirty paths, one harness at `65ed8a7`, no filter, clean cache, 71 executed and 426 cache files each. 71 of 71 compared and **71 identical**, zero mismatches, zero provenance findings, zero predictor cache differences. That is exactly the expectation `design.md` recorded before the run; the conditional I7 branch was not taken. Note explicitly: the matrix cannot reach a multi-lane monolithic MoE forward at all, so a null result is the pass condition for "nothing else moved", not evidence that the defect is fixed. Full record in `validation.md` and `test_report_2026-09-22_w3_shared_monolithic_forward.md`.
- 2026-09-22: Step 3 published. `bdff4aa` pushed to `origin/fix/issue26-correctness-pr`; PR #35 body gained a W3 section that states the defect, the one-lifecycle fix, the event-priority constraint, the four controls with their distinct failures, and the matrix result together with what a null result does and does not mean. PR #35 stays draft. The W3 measurement worktrees `.worktrees/w3-baseline` and `.worktrees/w3-candidate` were removed after their manifests, results and `comparison.json` were written; the matrix output root is kept as evidence.

- 2026-09-22: W4 implemented and committed as one unit (`10dd474`). `VllmDPLoadBalancer` models the two halves of vLLM V1's internal DP mechanism separately: the frontend's `waiting * 4 + running` selection with its lowest-index tie break and its local waiting reservation, and the coordinator's publication schedule with the 100 ms changed / 5000 ms idle intervals, the 50 ms first-snapshot collection wait, and the latch that publishes the previous step's counts. Every constant is cited against vLLM v0.10.2. Out-of-order step reports warn and still apply, as the reference does. The object creates no events, so a drained simulation still drains. `VllmLoadBalancingClusterScheduler` is selected only by `--cluster_scheduler_config_type vllm_load_balancing`; `round_robin` stays the default.
- 2026-09-22: D1 settled empirically rather than by argument. A probe at the load-report emission boundary showed MoE dp2 report keys strictly increasing under both at-once and staggered arrivals, paired when both lanes are live, while dense dp2 under staggered online arrivals interleaves (lane 1 step 0 arriving after lane 0 step 2). The constructor therefore accepts MoE at any lane count and dense only at `attn_dp=1`, and says so in the error message. No separate Replica-scoped step identity was introduced.
- 2026-09-22: Two seams carry the policy without a mutable time bridge. `BaseClusterScheduler.schedule_at(time)` defaults to `schedule()` and `ClusterScheduleEvent` now calls it, so placement that depends on elapsed time receives the time as an argument; the policy's own `schedule()` raises rather than reusing a stale snapshot. `on_replica_batch_end` is inert by default and is called after `replica_scheduler.on_batch_end`, so a policy reading lane populations there sees the post-step state. `get_request_load()` delegates to the existing decision-log waiting accessor, which `tests/unit/test_module_split_boundaries.py` pins, so a load balancer and the decision log cannot disagree about what is waiting.
- 2026-09-22: W4 acceptance. 61 unit tests in `tests/unit/test_vllm_dp_load_balancer.py` cover selection, publication timing, ordering and validation, the real `vllm_v1` load accessor, the capability guard, seam inertness for all five existing policies, and config/CLI discovery. `tests/integration/test_vllm_dp_placement_runtime.py` runs three cases in the real event loop, each twice -- once under the policy and once under `round_robin` -- and records that routing times equal the `ClusterScheduleEvent` times at five distinct instants, that the report keys are ordered with every equal-key pair carrying two distinct lanes, that every report matches the post-step load while strictly fewer match the pre-step load, and that no event type is introduced.
- 2026-09-22: The discriminating runtime case. With identical arrivals, identical dummy-mode durations and identical lane capacity, the policy places `[0,1,1,0,1,1]` where round-robin places `[0,1,0,1,0,1]`, putting strictly fewer requests on the lane still draining one 40-token request. That is the property the policy exists for, measured rather than asserted.
- 2026-09-22: One real defect surfaced during the focused regression run and was fixed rather than explained away. Running the W4 unit file after a MoE configuration in the same process tripped the process-global `IS_MOE` latch ("already initialized to True, cannot change to False"), because the file builds both dense and MoE shapes. It now resets the simulation globals around each test with `global_vars.reset_global_vars()`, matching `tests/unit/test_config_owned_contracts.py`. Checks 1, 3, 4 and 5 were re-run afterwards and the controls were rebuilt against the final test files.
- 2026-09-22: W4 controls, five trees and five distinct failure subsets. Restoring `schedule()` in the event fails all 3 integration cases; moving the hook above `on_batch_end` fails 2 with `reports_after_the_lane_released_the_batch` at `0 == 10`; unweighting the waiting term fails exactly the weight and boundary tests; dropping the local reservation fails 4 unit and 2 integration; deleting the dense-lane guard fails exactly the `dense_multi_lane` construction case. The baseline control tree passes 64 of 64, so the harness itself is sound inside a control tree.
- 2026-09-22: W4 regression comparison against the branch parent `cdfcdf5`. `tests/unit`: 84 failures on both sides with identical identities, 3778 vs 3717 passing. `tests/integration`: the same five pre-existing errors (the PD-AF Reference checkout is absent on this host), 21 skipped on both, 15 vs 12 passing. A focused 46-file set covering cluster scheduling, the decision log and the two edited events: 51 failed / 1432 passed, every failure already in the known 84-failure baseline.
- 2026-09-22: W4 fidelity matrix PASS. Baseline `cdfcdf5` against candidate `10dd474`, both clean detached checkouts with `source_dirty=False` and no dirty paths, one harness at `10dd474`, no filter, clean cache, 71 executed and 426 cache files each. 71 of 71 compared and **71 identical**, zero mismatches, zero provenance findings, zero predictor cache differences -- exactly the expectation `design.md` recorded before the run. Note explicitly: no matrix case selects the new policy, so a null result is the pass condition for "nothing else moved", not evidence about the policy. Full record in `validation.md` and `test_report_2026-09-22_w4_vllm_dp_placement.md`.
- 2026-09-22: Step 4 published. `0fd12c4` pushed to `origin/fix/issue26-correctness-pr`; PR #35 body gained a W4 section that states the mechanism with its citations, the measured report-key table behind the dense restriction, the two seams, the placement divergence from round-robin, the five controls with their distinct failure subsets, and the matrix result together with the fact that no matrix case selects the policy. PR #35 stays draft. The W4 measurement worktrees `.worktrees/w4-baseline` and `.worktrees/w4-candidate` were removed after their manifests, results and `comparison.json` were written; the matrix output root is kept as evidence.

- 2026-09-22: W5 premise check before implementation. The review's Mechanism A rests on the routing distribution being settable per role. It is not: `ClusterConfig` declares no per-role `moe_routing_distribution_type` override, `get_field_value` always misses to the global `ReplicaConfig` field, and the generated CLI has one flag for it. Verified on this branch and on `main@1f694f7` with `git grep` and `python -m frontier.main --help`. Every cluster in a run resolves one routing path, so neither W5 mechanism can fire today; it could fire only after adding a per-role override. Put to the user with three options (global field only, global + per-role, port as specified); the user first chose global + per-role, and the config surface, single-owner resolver, routing-aware training signature and family gate, and a three-part registry key were drafted (uncommitted).
- 2026-09-22: W5 magnitude, measured on request. On the three checked-in `moe.csv` files carrying both routing paths, the matched `moe_gating_routing_topk` cost differs by 3.1% (median; up to 24% at the smallest token counts), 7.3-7.8% and 4.2-5.4% of the summed per-layer operator medians (a800 qwen3-a3b-30b-moe, h800 Phi-tiny-MoE, h800 step-moe-noquant-small). This is the cost of the *existing* mapping picking the wrong kernel for a scenario, and the reason the mapping stays; the drafted override itself changed nothing while unset.
- 2026-09-22: W5 closed as NOT PORTED. After the explanation of the collision case, of what `uniform_topk` is (the profiler's round-robin routing path, `moe_impl.py:75-103`), and of the three options, the user decided to keep the current contract: one global `moe_routing_distribution_type`, one derived path per run. The twelve drafted source/test files were restored with `git checkout` after saving the diff to `w5_reverted_moe_routing_runtime_path.patch` (745 lines). `review.md` W5 rows corrected and closed; `requirements.md` records the decision verbatim; `plan.md` section 11 is kept as specified with a closure note. No `frontier/` change, so no fidelity matrix run for this step.

- 2026-09-22: W5 closure published. `de2bee8` pushed to `origin/fix/issue26-correctness-pr`; PR #35 gained a "W5, which is not in this PR, and why" section carrying the reachability correction with its `cluster_role_config.py:58-63` citation, what the module decides (`moe_impl.py:75-103`, `:195-225`), the three-dataset magnitude table, and the explicit statement that the reverted override's own fidelity effect was zero by construction. The Status line now reads "W6 onward is still to come". PR #35 stays draft. Verified the published body matches what was sent (the one-byte difference is the trailing newline `gh --jq` adds).

- 2026-09-22: W6 reachability and magnitude, checked before implementing, following the rule the W5 revert established. The repaired path is live: `moe_vllm_kernel` selects `_run_fused_moe_iteration` whenever vLLM exposes the low-level API, and all five imported names resolve in the pinned reference v0.10.2, which is what `environment_profiling.yml` pins. Magnitude estimated analytically from the checked-in datasets: 16.5% of the corrected `moe_grouped_gemm` time at 4096 tokens on `a800/qwen3-a3b-30b-moe` (6.8% median, 26.3% max), 1.1-1.4% on the two h800 datasets that stop at 64 tokens. Well above the 0.5% bar, and the repair adds no configuration surface, so it was implemented.
- 2026-09-22: W6 implemented. `silu_and_mul` into a preallocated activation buffer, then `moe_sum` into a preallocated output, matching vLLM's `fused_experts_impl` operand for operand. Two buffers added at the allocation site, outside the timed step. No gated/non-gated branch: `profile_fused_moe_kernel` only ever materializes `w1` with `2 * E` rows, so a conditional would be unreachable. Two recorded adaptations were deliberately not followed — the `_custom_ops` import sits inside the low-level `try` (it cannot fail where that API exists, and this way a build lacking it takes the functional path), and main's `SiluAndMul` wrapper was not reused because its `forward` allocates on every call, which would land inside the timed region. Both deviations and their reasons are in `review.md`.
- 2026-09-22: W6 CPU validation. 7 tests in `tests/unit/test_moe_fused_expert_arithmetic.py` under the Torch environment, covering the arithmetic against a written-out reference, the discriminating comparison with the old slice, call order and operand provenance, routing-weight placement, the local reduction, FP8 quantizer input, workspace reuse across steps, and allocation-site dimensions under TP=2. Regression against a detached `HEAD` worktree: same single environment-dependent failure both sides. Default-environment suite unchanged at 84 failed / 3778 passed. No fidelity matrix: the simulator cannot import the changed module.
- 2026-09-22: W6 left two items open rather than deciding them unilaterally. Artifact identity — `resolve_grouped_gemm_backend` labels both the legacy and the functional path `vllm_fused`, so no checked-in row can be classified as a complete or an incomplete measurement, and `profiling_patch_tag` turns out to exist only in one CSV and nowhere in the source. Native GPU parity — the repaired path is selected only under `vllm>=0.10,<0.11`, and both local Torch environments carry newer vLLM.

- 2026-09-22: W6 artifact identity decided by the user: do not change the profiling metadata, record the limitation only. `docs/profiling/README.md` now states what `moe_grouped_gemm` measures, the size of the pre-repair gap, and that `moe_grouped_gemm_backend` and `profiling_patch_tag` cannot date a row, so the remedy is to re-profile. Confirmed while writing it that `profiling_patch_tag` holds three historical free-text values in one CSV and is written nowhere in the source. No column added, no admission gate.
- 2026-09-22: W6 native parity test added. `tests/integration/test_moe_fused_expert_numerical_parity.py`, 8 cases, adapted from the donor and extended to the plan's required matrix: Qwen3-A3B-30B shapes read from the checked-in model config at 4096 and 4097 tokens on EP ranks 0 and 1; a 257-token, 16-expert case at top-k 2 and 4 with popularity-weighted routing that leaves two local experts empty; repeated invocation with different inputs; and the FP8 path as a structural check. It drives `_run_fused_moe_iteration` with the same buffers, config and alignment `profile_fused_moe_kernel` uses and compares the output tensor against `fused_experts` at `rtol=0, atol=0`. Skips unless CUDA is present and `VLLM_API_VERSION == "0.10.x"`; verified to collect and skip cleanly locally. `expert_hidden_dim_per_partition` was dropped from the iteration signature in the same commit, since the gated activation reads the whole projection and the parameter selected nothing.
- 2026-09-22: W6 native parity submitted. StepMind Python `RJobBackend` from the local host, job `exp-0922-140423-075005`, creator `i-fengyicheng`, `steptron_ci` / `H800`, 1 GPU, image `artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2` (the official Docker Hub build through the company docker.io proxy), NFS source `100.96.128.195:/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`. The instrumented benchmark repo was not mounted and the `frontier-calibration` skill was not invoked: this compares tensors from vLLM's own `fused_experts` inside one process, which needs neither serving instrumentation nor an E2E calibration workflow.

- 2026-09-22: W7 investigated while the parity job queued. `fwyc0573/frontier-htsim` is public and readable, its only branch `main` is exactly the gitlink Frontier already pins, and the candidate's `e564935d` returns HTTP 422 with no local object store on this host containing it — an unpublished commit, not an access problem. The fix is entirely companion-side. Confirmed all three payload defects by running the published runner rather than reading it: an explicit `tensor_bytes = 0` and a deleted field produce the identical `missing required fields: ['tensor_bytes']` exit 2; `-1` passes validation and reaches the simulator invocation; and `set_if_none_or_zero` replaces an explicit CLI `0` with a positive spec value. Reachable from Frontier because `_validate_data_size` accepts zero and `moe_operator_times.py:512` computes the EP all-to-all payload as `embedding_dim * 2 * routed_tokens`, which is zero for an empty local lane. Frontier cannot fix it alone: a zero-byte collective still costs the intra-server latency, so short-circuiting would change the backend's semantics. Blocked on the user's decision about the second repository; nothing was created or pushed there.

- 2026-09-22: W6 and W7 published. Branch pushed through `cad3afd`; draft PR 35 body now carries a W6 section (what was missing, the reachability and magnitude checks made before implementing, the repair and its two recorded departures, measurement ownership, the artifact-identity decision, CPU validation and its stated limits, and the native parity matrix) and a W7 section (unpublished gitlink, the three executed confirmations, Frontier-side reachability, and the companion-repository blocker). PR remains draft. Default-environment suite re-run at `cad3afd`: 84 failed / 3778 passed / 49 skipped / 11 errors, identical to the W4 baseline.

## Step 3 scoping, as recorded before implementation

Kept as written so the implementation can be read against the scope it started from. Step 3 is now complete; see the W3 entries above and `design.md`.

The defect is present on main at three layers, and the fix has to change all three together or the intermediate state deadlocks differently:

1. `frontier/events/replica_stage_schedule_event.py:157-184` picks the prefill or the decode sync path from the batch's own `num_prefill_tokens`, so two DP lanes of one forward step can take different paths.
2. `frontier/scheduler/utils/sync_state.py:29-40` allocates two independent waiting rooms for `MONOLITHIC`.
3. `frontier/scheduler/utils/forward_sync_state.py:38-41` partitions the open-step binding table by kind.

One refinement over the audit's framing, from reading the code: `ForwardSyncState._next_step_id_by_replica` is **already** shared across kinds, keyed by replica alone. Only `_open_steps_by_kind` is partitioned. So step-id allocation is already Replica-scoped and monotonic; what is partitioned is the binding table and the waiting room. That narrows the change.

Surface: about 3,500 lines across `replica_stage_schedule_event.py`, `sync_entry.py`, `prefill_collective.py`, `decode_collective.py`, `ep_wave_schedule.py`, `ep_wave_inputs.py` and `base_cluster_scheduler.py`, with 11 call sites of the sync-kind and sync-path selection.

Blocked hunk carried from the audit: the candidate's decode final-metrics change calls `_create_corrected_execution_time_for_metrics`, which main deleted, so it needs rewriting against main's current execution-time ownership rather than porting. Resolved by exclusion: that hunk is I8, kept out of scope under Checkpoint D's "current-main metrics ownership" and recorded in the `design.md` scope table.

### 2026-09-22 — Step 8 §14.1, combined regression

Ran the selected suites together on `d881357` rather than package by package.
Full record: `test_report_2026-09-22_w8_combined_regression.md`; summary rows in
`validation.md`.

- Re-fetched `origin/main` before the run. Still `1f694f7` and an ancestor of
  `HEAD`, so no integration merge was needed and none was made.
- Unit: **84 failed, 3782 passed, 49 skipped, 11 errors**. The `FAILED` set is
  identical to the recorded baseline in both directions.
- Integration: **15 passed, 22 skipped, 5 errors**. The added skip is the W6 GPU
  parity module; the 5 errors are the absent pinned PD-AF Reference checkout and
  are identical on the base.
- All 16 release-supported architecture examples pass, covering co-location,
  sequential PDD, and sequential PD-AF in both offline and online modes.
- Four `PP=2` runs pass: co-location dense offline and online, co-location MoE
  with `Attn_TP=4, MoE_TP=2, MoE_EP=2`, and sequential PDD dense with both roles
  at `PP=2`. These put the changed cluster-scheduling, stage-dispatch, and
  metrics code on the multi-stage path, which the PP1-only DP placement policy
  cannot reach by itself.
- Ran the trained-predictor path cold and then warm against an empty scratch
  cache directory via `--metrics_config_cache_dir`, leaving the repository
  `cache/` untouched. Cold took 26.9 s and wrote 63 artifacts; warm took 2.1 s
  and wrote none; their `request_metrics.csv` outputs are byte-identical. This
  is the first-load path measured deliberately rather than inherited.

**Finding, deferred.** `AGENTS.md` §Tests names `comm_backend_tests/`, `debug/`,
and two `bash tests/debug/e2e-level/monolith_mode/scripts/*.sh` commands.
`tests/debug/` exists neither here nor on `origin/main`. The same missing tree
causes 10 of the 84 baseline unit failures, in
`test_colocation_release_review_contracts.py`, and a docstring at
`vllm_v1_engine_replica_scheduler.py:16` still points into it. That is one
pre-existing defect class from the release scrub, unrelated to Issue 26, and
repairing it means deciding what the published test surface should assert
against. Recorded in `future.md` and left untouched; the PP2 coverage was taken
through the example scripts instead.

**Hygiene.** `git status --porcelain` is empty afterwards. Example runs were
pointed at scratch metrics directories, and one leftover `outputs/examples/`
tree from an earlier iteration was removed after confirming it held 0 tracked
files; the 110 tracked files under `outputs/` are all still present.

### 2026-09-22 — Step 8 §14.2 and §14.3, hand-off

- §14.2: read `git diff 6ef0a3c..d881357` over `frontier/ tests/ docs/ examples/`,
  1925 lines, line by line. Answers to all nine review questions, the cleanup
  done in the pass, and an explicit statement that this was a **self-review and
  not independent** are in `review.md`.
- §14.3: updated PR 35's body in place rather than opening a duplicate, using
  `gh api --method PATCH` because `gh pr edit` fails on this repository with a
  Projects-classic GraphQL deprecation error. The body now carries the W6 and W7
  outcomes in its progress table, a combined-regression section with the numbers
  and what they do not prove, the deferred `tests/debug/` defect, absolute links
  to `plan.md` / `requirements.md` / `progress.md` / `review.md` /
  `validation.md` / `future.md`, a table of the implementation commits, and a
  status paragraph separating technical acceptance from GitHub's draft flag.
  Relative links were replaced with blob URLs because a PR body does not resolve
  repository-relative paths; the links were checked and return 200.
- Verified after the update: `draft: true`, `state: open`, base still
  `refactor/oversized-module-split`, head `8730509`, and the body read back byte
  for byte as sent. Nothing was merged, force-pushed, marked ready, or closed.
- `summary.md` written as the completion archive.

## External review corrections (2026-09-22)

Review document: `.local-draft/Frontier_PR34_PR35_Current_Code_and_PP_Extension_Review_2026-09-22.md` (local). Finding-by-finding disposition: `review.md`, "External review 2026-09-22 — findings disposition". Evidence: `test_report_2026-09-22_review_corrections.md`.

| Package | Finding | Where | State |
| --- | --- | --- | --- |
| A | C34-01 predictor-cache eligibility by executed case list | PR34 branch `2310417`, merged here as `0d025f8` | completed |
| B | C35-01 decode credit at a dense layer for a mixed source | `collective_timing.advance_decode_layer`, `dense_metrics.complete_dense_layer`; unit + real-loop tests | completed |
| C | C35-03 FP8 `block_shape` wiring + CPU boundary test; C35-04 optional-torch skip | `tests/integration/test_moe_fused_expert_numerical_parity.py`, `tests/unit/test_moe_fused_expert_arithmetic.py` | completed; native rerun PASS `exp-0922-202645-561899` |
| D | C35-02 scope table; C35-03 seven-plus-one wording; C35-05 records consistency and PR bodies | W6 report §5/§8, `docs/profiling/README.md`, `summary.md`, `review.md` D2, this file, PR34/PR35 bodies | completed |
| E | P9-01..P9-06 plan corrections | `plan.md` §18 (renumbered from §17) and §18.11, `design.md` W9 | completed (records only) |
| F | W9 implementation | — | **not started** (user: 暂不开启) |

## Step 9 — PP>1 support for `vllm_load_balancing` (pending)

| Date | State | Note |
| --- | --- | --- |
| 2026-09-22 | pending | Plan drafted in `plan.md` §18 with amendment A12 and `requirements.md` rows; awaiting user approval and the D-a..D-g answers. No source, GPU, or publication action taken. |
| 2026-09-22 | in-progress (planning closed) | User answered D-a..D-g; `plan.md` §18 finalized (decisions, G1–G5 ground-truth packages, calibration case binding, `codebase-design` vocabulary), `design.md` W9 written, `requirements.md` updated. Records committed and pushed (SHA in the commit log). No source edit, no GPU submission; next action: P1 probe + G1/G2 once the user confirms the start. |
| 2026-09-22 | in-progress (planning corrected) | External review P9-01..P9-06 applied: `plan.md` §17 → §18 (duplicate number), state table and preconditions replace the room-only hook rule and the steady-state claim, K1/K3/stride rejected as acceptance basis with invariants I1–I6, instrumentation chain and T2 qualification, CPU reference-loop oracle and valid controls, PP3 fixture with a valid layer count, revised graph and C1–C5 (`plan.md` §18.11, `design.md` W9). Records only; execution still awaits the user's start signal. |

- 2026-09-22: FP8 native rerun authorized by the user. Same launcher as `exp-0922-145047-660565`, worktree clean at `c231322`: `exp-0922-202645-561899`, `codesign` / H800 (`gpu-h800-0095`), `Succeeded`, `8 passed in 14.27s`, `W6:PARITY_EXIT=0`. Log via `logs_replica` only (`logs_rjob` empty); the first fetch two minutes after completion returned a truncated window, the second fetch five minutes later returned the full tail. Records: W6 report §8, corrections report C4, review.md C35-03, validation.md Step 6.
- 2026-09-22: Second Step 9 plan review at the user's direction (quality gates). Eight findings R9-01..R9-08 recorded in `plan.md` §18.12 with amendments to D9-1 (hook payload = completion hook signature), D9-2 (key both observation kinds by the observing iteration; first candidate `ForwardSyncState._next_step_id_by_replica`, I5 gap to be measured), P1(a) (oracle = engine loop feeding the real balancer), §18.10, §18.11 representation column, §18.2 (32-step all-reduce); `design.md` section "What the code already provides"; `review.md` disposition table. Records only; no Step 9 source change; execution awaits the start signal.

## Step 9 execution — P1 (2026-09-22)

User start signal: "开始执行step9", with the quality gates repeated (readability, maintainability, high-value changes only, no hard-coding, no temporary patches, no over-defense, no redundancy, plain names).

| Package | State | Evidence |
| --- | --- | --- |
| P1(a) reference-loop oracle | completed | `tests/comparison/dp_placement_pp/reference_loop.py`; `tests/unit/test_dp_placement_reference_loop.py` (9 passed, 1.21 s, `frontier-py310`). §18.11 state table confirmed as written; PP=1 shown to degenerate to "every iteration schedules and applies"; depth 3 shown to allow two consecutive admission-only publications, which rules out any stride constant. |
| P1(b) Frontier boundary probe | completed 2026-09-23 (see "Step 9 P1(b) and D9-2" below) | Three shapes probed (`attn_dp=2 PP=1`, `attn_dp=1 PP=2`, `attn_dp=1 PP=3`), tables in `plan.md` §18.13. The fourth shape, MoE `attn_dp=2, moe_ep=2, PP=2`, drains the event queue with requests unfinished — pre-existing defect W9-01 in `issues.md`. |
| Design checkpoint (D9-1, D9-2) | settled 2026-09-23 | D9-1's payload was settled from the P1 oracle. D9-2 is the group-anchored key, exact on all seven P1(b) shapes (`design.md` "Design checkpoint D9-2"); the user chose it on 2026-09-23. The first candidate, `ForwardSyncState._next_step_id_by_replica`, failed I5. |
| P2 implementation, P3 unit tests | completed 2026-09-23 | See "Step 9 P2 and P3" below. |
| P4 integration | completed 2026-09-23 (`bacdbb4`) | See "Step 9 P4 and P5" below. |
| P5 fidelity and regression | completed 2026-09-23 | See "Step 9 P4 and P5" below. |
| P6 documents and publication | completed 2026-09-23 (`c1a570d`, `104b6ff`) | AGENTS.md scope, design, plan §18.16, validation, review S9-01..S9-08, summary, W9 report; pushed; PR 35 body updated. |
| G3–G5 | blocked | GPU authorization is `BLOCKED` in the case manifest. |

W9-01 is not caused by this PR: `stage_execution_context.py`, `replica_stage_schduler.py` and `stage_contexts.py` are byte-identical to `main`. It is unobserved because every Simulator-level test with `attn_dp > 1` uses `num_pipeline_stages = 1` and no shipped example sets `attn_dp > 1`. Scope decision requested from the user; recommendation is to fix it as a separate correctness item rather than inside this feature branch.

Scope decision (2026-09-22): "采纳你的推荐，继续" — option 2, a separate correctness item. Fixed on `fix/stage-admission-ordering`, draft PR 36, rule commit `dac4e69`, validated against vLLM DP=2/PP=2 on 4×H800. Branch records are copied to `w9_01_stage_admission_ordering/`. Resume order: PR 36 merges, `main` is merged forward here, G3b (with W3) and the online groups G9 and G10 (with W2) rerun as the composition check, then P1(b) and D9-2 (`issues.md` W9-01, Resolution). PR 36 round-2 review fixes: `1661bf1`, `a8e8d8a`, `e35242f`, records `7a7c22e`. PR 36 untracked its task directory before merge at `4d08c5d` (R-11); the copies here are its published records.

W9-02: `attn_dp=2, moe_ep=2, PP=3` is rejected at construction (6 devices against node size 4). Plan C1's PP3 row amended to `attn_dp=1`.

### W9-01 merge-forward and composition check (2026-09-23)

| Step | Command / action | Evidence | Result |
| --- | --- | --- | --- |
| Request | "我已经完成 PR 36 merge，把 origin/main merge 进 fix/issue26-correctness-pr（用 merge，不 rebase），重跑 G3b、G9、G10 作为 composition check。通过后恢复 Step 9 的 P1(b) 和 D9-2；暂不处理 pr34和35的 gitingore" | `requirements.md` | recorded |
| Merge | `git merge --no-ff origin/main` (PR 36 squash `4ab1964`) | `dd9b8d9` | no conflict; the only source file is the rule file |
| Criteria | K1–K4 written before measuring | `plan.md` §18.14 | — |
| First sets | `c-merged` at `dd9b8d9`; `c-pr35` with the `1f694f7` rule file | scratch `composition/` | `c-merged` 51/51 success; `c-pr35` two cells `other_failure` from `KeyError: 'batches'` in the drain reader |
| Diagnosis | exported tree, room dump | `debug_rooms.py` in scratch | dispatched rooms keep an empty entry (`sync_entry.py`, same on `main`); the reader indexed it |
| Harness fix | `if not room.get("batches")` | `03d5f24` | both cells classify as `admission_deadlock` |
| Sets rerun | both sets at `03d5f24`; rule file restored and equal to `origin/main` | scratch | `c-merged` 51/51; `c-pr35` 16 `admission_deadlock`, 35 success |
| K2 compare | harness `compare` + `composition_check.py` | `w9_01_stage_admission_ordering/composition_evidence/` | 0 STOP; 7/8 EXPLAIN start-times-only; `G10-dense-dp2-pp3-n8` first divergence = same batch admitted earlier; K2 amended (plan §18.14) |
| K3 | `composition_check.py` | same | 22/22 all lanes; `after-r2` lane 0 only |
| K4 | `composition_run_suites.sh` on both trees; `composition_compare_junit.py` | same | 0 regressions; export-only failures are git-metadata failures; 487 targeted tests pass |
| Report | — | `test_report_2026-09-23_w9_01_composition_check.md` | PASS |

### Step 9 P1(b) and D9-2 (2026-09-23)

| Step | Command / action | Evidence | Result |
| --- | --- | --- | --- |
| Reference check | Read the busy loop, `step_with_batch_queue`, `execute_dummy_batch` and `get_dp_padding` in `.real-engine/vLLM-BS` | `design.md` "Design checkpoint D9-2", reference table | Each iteration launches one forward, real or the blocking dummy. Forwards pair per stage through the DP all-reduce, so the step key is the shared forward index. Amends R9-08. |
| Probe | `probe_boundaries.py <shape> <out>` for 7 shapes, one process each; `PYTHONPATH=<worktree>`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `FRONTIER_TMP_ROOT=/data/ycfeng/tmp`, `frontier-py310` | `/data/ycfeng/tmp/issue26-correctness-pr/step9_p1b/<shape>/` | All 7 complete 6/6 (108–210 records). The extended fields are stage-0 sealed, lane busy, lane queue and room group. |
| Scoring | `analyze_keys.py /data/ycfeng/tmp/issue26-correctness-pr/step9_p1b <out>` | `step9_p1b/evidence/key_scores.json` | The group-anchored key has 0 splits, merges or inversions and 0 ms replay mismatch in all 7 shapes. A and the lane counter fail as tabulated in plan §18.15. |
| Variant | Every-report-advance variant, candidate `every_report_advances` in `analyze_keys.py` | `key_scores.json` | Drifts on dp2 PP3 staggered: 6/5/5 on real-forward reports. Rejected. |
| Residual count | Completion-only report followed by an equal-key report on the same lane | `key_scores.json` rows | 17 pairs, 3 with changed counts. |
| Records | `design.md`; plan §18.15 and the §18.13 resolution; `issues.md` (W9-01 step 3, W9-02 narrowed, W9-03); test report §2 addendum | — | Done |
| Decision | D9-2 rule and the C1 PP3 amendment | `requirements.md` | User chose the group-anchored rule; C1 amended |

### Step 9 P2 and P3 (2026-09-23)

| Step | Change / command | Reason and expectation | Result |
| --- | --- | --- | --- |
| P2 source | `StageExecutionContext.joinable_forward_group_id`; inert `BaseClusterScheduler.on_replica_batch_scheduled`; its call in the MONOLITHIC/PREFILL admission loop of `BaseReplicaScheduler.on_schedule`, after `_num_running_batches += 1`; `VllmLoadBalancingClusterScheduler`: PP1 guard clause removed, group-anchored key with a held key per lane, `ForwardSyncState` use removed | D9-1..D9-3 as decided. Expect only the intended guard case to fail in the existing suites. | Existing suites: 1 failed (the `pipeline_parallel` rejection case), 112 passed. |
| P3 tests | `tests/unit/test_vllm_dp_load_balancer.py`: guard case inverted into PP2/PP3 construct cases (8 shapes) plus dense multi-lane PP2 and uneven-partition rejections; `_ScriptedReplica` drives scripted lane readings and real stage-0 forward groups; cases for PP1 (no schedule-time report), PP2 cold fill in both lane orders (no partial latch), PP3 two admission-only iterations, a full pipeline (MoE and dense), a completion plus the admission it makes room for (one key, no intermediate latch), drain to zero and new work, and the real admission loop at PP2; both seams in the inert and unknown-lane tests. `tests/unit/test_shared_forward_group_admission.py`: `joinable_forward_group_id` across bind, seal and release. | Expected reports written from the reference iteration, before running. Each new case must fail on the pre-P2 tree. | `pytest -q -p no:cacheprovider tests/unit/test_vllm_dp_load_balancer.py tests/integration/test_vllm_dp_placement_runtime.py tests/unit/test_dp_placement_reference_loop.py tests/unit/test_stage_execution_context.py tests/unit/test_shared_forward_group_admission.py`: 132 passed, 7.07 s. The same tests on `git archive HEAD` (pre-P2) plus the new test files: 19 failed, 72 passed; the 19 are exactly the new or changed cases. |
| Arrival-order check | Replay of the group-anchored keys from `key_scores.json` in report order, counting keys smaller than the last applied one | Recorded so the balancer's warning is not mistaken for a key defect later. | 1 report in 7 shapes (dp2 PP3 burst, t=0: lane 0 key 0 after lane 1 key 1); the reference has the same race. Noted in `design.md`. |

### Step 9 P4 and P5 (2026-09-23)

| Step | Change / command | Reason and expectation | Result |
| --- | --- | --- | --- |
| P4 tests | `tests/integration/test_vllm_dp_placement_runtime.py` rewritten around seam records: 5 PP shapes plus the plan §18.6 discriminating case against a completion-reporting control; `tests/integration/test_monolithic_mixed_forward_runtime.py` gains a PP2 `vllm_load_balancing` run | Each report must map to one reference iteration kind under its forward's key; the probe must be placed differently only because of what was published | 9 + 3 passed; report kinds per case and the probe snapshots in the W9 report §5. Committed `bacdbb4`. |
| P5 C2, policy | `step9_p5/c2_pp1_policy_matrix.py`: 24 PP=1 scenarios on `git archive` exports of `d1a2a06` (pre-P2 source) and `bacdbb4` | Artifacts, report stream (without key), key order, and selections identical; no added admission-only report | 24 of 24 identical. Two cases stall on both sides (W9-04) and two lose 4 requests on both sides (W9-05), with identical diagnostics and artifacts. |
| P5 C2, other schedulers | `step9_p5/run_fidelity.sh` on clean detached worktrees `.worktrees/p5-fidelity-{before,after}`; `step9_p5/run_examples.sh` + `compare_examples.py` | 71 of 71 and 16 of 16 identical | 71 of 71 identical, 0 provenance findings; 16 of 16 examples pass on both trees and are identical. |
| P5 suites | `composition_run_suites.sh` on the clean `bacdbb4` worktree, compared by test id against the K4 JUnit of `03d5f24` | 0 regressions, 0 new failures | unit 84 failed / 3829 passed / 51 skipped / 10 errors; integration 5 errors / 26 passed / 22 skipped; 0 regressions, 0 new failures, 0 skip changes. The added skip is `test_collective_sim_zero_payload` (submodule not initialized in the detached worktree); 4 passed in this worktree. |
| W9-04 investigation | `step9_p5/deadlock_trace.py`, `deadlock_sweep.py`; prototype `step9_p5/w9_04_prototype.patch` in a scratch export only | Establish the cause before proposing a change | Root cause and options in `issues.md` W9-04. Prototype: 72 of 72 sweep cells drain (was 66); 22 of 22 previously drained C2 cases identical. Not applied to the branch. |

### W9-04 fix (2026-09-23)

| Step | Change / command | Reason and expectation | Result |
| --- | --- | --- | --- |
| Decisions | AskUserQuestion: W9-04 "本 PR 修复 (Recommended)", W9-05 "暂缓，单独立项 (Recommended)", worktrees "删除 (Recommended)" | Recorded in `requirements.md` | P5 worktrees removed (`git worktree remove`; `--force` for two generated `config.json` files) |
| Precondition | Read `stage_execution_context.py` `try_acquire`/`release`/sealing and `ReplicaStageScheduler.is_busy` | The rule needs "busy while the room is open ⇒ joined this forward" | Holds (`review.md` F9-03) |
| Reproducer | `w9_04_fix/extract_trace.py`, `trace_probe.py` | A short deterministic trace for the regression case | 3 requests at 0 / 2 / 8 ms stall under both policies at `339e6bd` |
| Criteria | plan §18.17 A1–A7 | Fixed before measuring | — |
| Fix and test | `sync_entry.py` `_withdraw_idle_batches_of_joined_lanes`; case `moe_dp4_late_join` in `test_vllm_dp_placement_runtime.py` | Remove the stale placeholder; assert the race is reached and the run conserves work | `2ffb062`; 10 passed; neighbors 171 passed |
| A1–A7 | `w9_04_fix/run_a2_a3.sh`, `run_a4_a5_a6.sh`, the control tree, `run_examples.sh` | As in plan §18.17 | All PASS (`validation.md` "W9-04 fix") |

Evidence copies: `step9_p5/evidence/`. Raw runs: `/data/ycfeng/tmp/issue26-correctness-pr/step9_p5`.

### W9-05 fix (2026-09-23)

| Step | Change / command | Reason and expectation | Result |
| --- | --- | --- | --- |
| Direction | "授权上述1-2，推进W9-05" | Recorded in `requirements.md`; supersedes the deferral | W9-04 worktrees removed; W9-05 scheduled; G3–G5 authorized |
| Diagnosis | `w9_05/lost_request_probe.py`, `w9_05/membership_trace.py` on the C2 dense `tight_kv` case | Find where a lost request leaves every queue | Request 7 preempted at t=1.093 with 45 of 55 tokens, reset to 0, and dropped from the waiting queue at t=1.157 by the `num_new_tokens <= 0` branch (`issues.md` W9-05) |
| Reference check | vLLM v1 `_preempt_request` and the waiting loop | Preemption keeps output tokens; the waiting loop asserts `num_new_tokens > 0` | Rule: reset only a victim still in prefill |
| Criteria | plan §18.18 B1–B8 | Fixed before measuring | — |
| Fix and tests | `vllm_v1_kv_allocation.py` (reset guarded by `not victim.is_prefill_complete`; the cluster-type set deleted); new `tests/integration/test_vllm_v1_decode_preemption_runtime.py`; `tests/unit/test_pdaf_decode_attn_preemption.py` updated | A past-prefill victim resumes; a prefill victim still restarts | `75c1140`; 1 passed (1.44 s); unit module 8 passed; control at `2ffb062` fails `assert 0 == 34` |
| B2–B8 | `w9_05/run_validation.sh` (exports), `w9_05/run_matrix.sh` (worktrees), `composition_run_suites.sh` on `.worktrees/w9-05-after` | As in plan §18.18 | All PASS (`validation.md` "W9-05 fix"). The first B7 run on the export gave 88 git-provenance failures; rerun on the git worktree equals the W9-04 counts |
| Metrics caveat | `request_metrics.csv` preemption columns | Which column counts a MONOLITHIC preemption | Only `request_total_preemption_count`; recorded in `issues.md` W9-05 Limits |

Evidence copies: `w9_05/evidence/`. Raw runs: `/data/ycfeng/tmp/issue26-correctness-pr/w9_05`.
Validation worktrees `.worktrees/w9-05-{before,after}` remain; removing them needs approval.

### G1 ground-truth instrumentation (2026-09-22, completed)

Runs in parallel with P1 in the work graph and does not depend on W9-01.

`/data/ycfeng/Frontier/.real-engine/vLLM-BS` now has the local branch
`feature/frontier-comparison-instrumentation` created at the remote tip
`ea95f571e`, with the instrumentation committed as `494b9f327`. Tree clean,
nothing pushed, per decision D-b.

Five observations, four record kinds, one env gate
(`VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR`), buffered one file per process. The
engine iteration and its publication are one record because they happen in the
same turn of the busy loop under the same `(engine, wave, step)` key. Coordinator
publications carry a snapshot id that is sent on to the front ends, which is
what makes an applied snapshot and the placement made from it traceable back to
the engine reports behind them.

`vllm/v1/core/sched/scheduler.py` turned out not to need a change: the plan
expected a `dp_rank` column, but `SchedulerOutput` already carries the
scheduled request ids and the record is written by the engine that owns the
rank. The changed-file list is four files, 212 insertions, 10 deletions.

Case binding written to `calibration/dp_pp_case_001/` (`manifest.yaml`,
`case_init.md`, `g1_instrumentation.diff` with SHA-256
`84fc24db0e2411268a93f8be7ca5f8e4e5072ea09d86063cac2cfb98381feb2c`). Two
manifest decisions are `BLOCKED`: GPU authorization for the S0/S1 runs, and the
W9-01 scope decision.

Evidence: `test_report_2026-09-22_w9_pp_dp_placement.md` §3. The instrumented
engine paths are not executed yet; that is package G3 and needs a GPU host.

G1 amendment (2026-09-23, before any GPU run). vLLM 0.10.2 starts its engine
cores and the DP coordinator as forked processes by default
(`VLLM_WORKER_MULTIPROC_METHOD=fork`), and stops them with SIGTERM followed by
a kill. A forked child leaves through `os._exit`, so the `atexit` flush never
ran there, and a SIGTERM with the default action skips it as well. The buffered
records of every engine core and of the coordinator would have been lost. The
probes in `calibration/g1_writer_probes/` show both: `atexit_probe.py` and
`out.txt` (under fork no `atexit` ran, whether the child returned or was
terminated; under spawn only on return or a handled SIGTERM). New local commit `63ac6c6b9` ("Write Frontier placement records through as
they are made") writes each record to a line-buffered per-process file as it is
made, and restarts the sequence number in a forked child. A SIGTERM-killed forked
child then kept all 3 of its records (`trace_fork_probe.py`,
`trace_fork_probe_output.txt`). The diff artifact was regenerated as
`ea95f571e..63ac6c6b9`, SHA-256
`b6400d29710c6f0fac4d0e26c9078177a741d3984f9a07a72f838838eafcf8c7`.

### G2 harness and case (2026-09-23, completed)

Harness in `tests/comparison/dp_placement_pp/`:

- `make_trace.py` writes the trace CSV and `request_ids.json` from a workload file.
- `run_frontier_case.py` maps the engine file to a Frontier configuration and
  runs the fixed policy, the completion-reporting control and round-robin, one
  process each.
- `vllm_replay.py` serves the case and replays the trace over HTTP with
  `X-Request-Id`.
- `run_vllm_worker.sh` builds the overlay, refuses an existing output
  directory, and publishes to the case and to the archive.
- `extract_vllm_placement.py` joins the G1 records by correlation ids.

`tests/integration/test_vllm_dp_placement_runtime.py` gained a `build_config`
seam and records the request id of each placement. Plan §18.19 records how this
differs from the §18.5 G2 row and fixes the G4 sizing rule.

| Check | Command / evidence | Result |
| --- | --- | --- |
| P4 test after the seam | `pytest tests/integration/test_vllm_dp_placement_runtime.py -q` | 10 passed |
| Frontier pre-check, dummy 2 ms | `run_frontier_case.py` on `inputs/engine_g4.json`, `inputs/frontier_g4.json`, `inputs/trace` (receipt `calibration/dp_pp_case_001/runs/frontier_precheck_2ms/command_receipt.json`) | Burst `[0,1,0,1,0]` in all load-balancing runs. The probe `dpp1-b6` goes to lane 0 from `[[0,3],[1,1]]` under the fixed policy and to lane 1 from `[[3,0],[2,0]]` under the control. 33/33 requests complete with tokens conserved. First completion 0.248 s after the burst. |
| Earlier pre-checks | dummy 10 ms and 5 ms | No discrimination: a stage costs about 62 × the base time, so the warmups were still decoding at the burst. Fixed by 2 ms, 4 warmup decode tokens and an 8 s idle gap. |
| Extractor | PR 36 records `sa-pp-20260923b/runs/moe` (parse); synthetic fixture, positive and missing-receipt negative | Parses 130 engine iterations with unique keys. Fixture PASS, negative FAIL with "engine 1 published 1 reports, coordinator received 0". |
| Semantic table | `calibration/dp_pp_case_001/analysis/semantic_alignment_table.csv` | 41 rows: 36 MATCH, 2 declared MISMATCH (IPC latency, dummy timing), 3 UNSET resolved by the runs (KV blocks, burst route order, MoE routing records); status `PASS` |
| Checkout tuple | `git -C .real-engine/vLLM-BS` branch, ref, HEAD, porcelain; `git ls-remote`; diff SHA-256 | `feature/frontier-comparison-instrumentation` @ `63ac6c6b9`, clean, remote tip `ea95f571e`, diff `b6400d29…` equal to the artifact |

Found while writing the extractor: in the idle path of the busy loop no step
record is written, and `(engine, wave, step)` stays unique in the PR 36 records.
The extractor still pairs receipts by order within each engine, so a repeated
key could not mis-join.

### G3 native PP1 smoke (2026-09-23, completed)

| Check | Command / evidence | Result |
| --- | --- | --- |
| Run-check | `calibration/dp_pp_case_001/analysis/run_check_status.groundtruth_clean.json`, run `dpp-g3-20260923a` | `PASS` before launch; command receipt added after the run (below) |
| Submission | `run_dp_pp_job.sh` (scratchpad `dp_pp/`) with `RUN_TAG=dpp-g3-20260923a ENGINE_INPUT=inputs/engine_g3.json TRACE_INPUT=inputs/trace_g3 NUM_GPUS=2 NUM_CPUS=16 MEM_GB=96`; StepMind `RJobBackend`, codesign, H800 | Job `exp-0923-221233-009652`, 14:12:33Z to 14:16:29Z, `FINAL_STATUS succeeded` |
| Worker | `runs/groundtruth_clean/dpp-g3-20260923a/` (`worker_env.json`, `vllm_import.txt`, `overlay_report.json`, `COMPLETE`) | 2 H800 visible to torch; `vllm` and `frontier_trace` import from the overlay; `WORKER_STATUS=0`; 38/38 HTTP 200; 140985 blocks per engine |
| Extraction | `extract_vllm_placement.py` → `runs/groundtruth_clean/dpp-g3-20260923a/extraction/` | `PASS`: 565 iterations, 69 reports and 69 receipts paired, 50 publications, 49 applications, 38 placements, 0 out-of-order |
| T1 replay | `compare_placement.py` → `analysis/g3_t1_replay/` (receipt there) | 38/38 routes `MATCH` in engine and counts, 30/30 formal |
| Sizing and route order | `extraction/chain.json` | Plan §18.20: `f(20448)` about 98 ms, so `B = 20448`; the burst routed in the client's dispatch order b5, b4, b3, b1, b2, all 42.6–42.9 ms after dispatch |

Two launcher problems, both fixed in the scratchpad scripts before G4:

- The wrapper exited early. `w6_env.sh` turns on errexit and pipefail, and the
  first grep for the job name found nothing yet. The launcher kept running and
  the job was unaffected, but no receipt was written. The G3 receipt was written
  by hand from `submit.log` and marks this in `receipt_origin`. The wrapper now
  resets both options after sourcing.
- The replica-log poller never exited, because one log query hung. Each query
  is now bounded by a 60 s alarm.

### G4 inputs (2026-09-23, before submission)

Plan §18.20 records the amended design. The G4 files are `inputs/workload_g4.json`,
`inputs/trace_g4/` (51 rows, namespace `dpp2`, 48 formal ids) and
`inputs/engine_g4.json` (`max_num_batched_tokens` 20448). `make_trace.py`
accepts named `bursts` with `spacing_s`. The G2 and G3 traces regenerate
byte-identical from their workloads; their request-id rows gain only
`burst: ""`. `compare_placement.py` is new. It runs the T1 replay, qualifies
T2 per burst, and writes the gap table. `run_frontier_case.py` summarizes each
burst. The extractor now keeps the coordinator receipts in `chain.json`.

| Check | Command / evidence | Result |
| --- | --- | --- |
| Frontier pre-check, dummy 2 ms | `run_frontier_case.py` on `engine_g4.json`, `frontier_g4.json`, `trace_g4` (receipt `runs/frontier_precheck_g4_2ms/command_receipt.json`) | 51/51 complete with tokens conserved in all three runs. In every burst a–d, the fixed policy sends the probe to lane 0 from `[[0,1],[0,1]]`, and the control sends it to lane 1 from `[[3,0],[2,0]]`. |
| Semantic table | `analysis/semantic_alignment_table.csv` | 43 rows: 36 MATCH, 4 MISMATCH (S30, S31, S42, S43), 3 UNSET (S17, S33, S39); status `PASS` |
| Manifest | `manifest.yaml` | Parses; the workload points at `trace_g4`, and the G3 run is recorded. A pre-existing unquoted `producer` value that broke YAML parsing is now quoted. |

### G4 native PP2 ground truth (2026-09-23, completed)

| Check | Command / evidence | Result |
| --- | --- | --- |
| Run-check | `analysis/run_check_status.groundtruth_clean.json`, run `dpp-g4-20260923a` | `PASS` before launch (committed in `47d9190`). The command receipt was added after the run from the wrapper's `launcher_receipt.json`. |
| Submission | `run_dp_pp_job.sh` (scratchpad `dp_pp/`) with `RUN_TAG=dpp-g4-20260923a ENGINE_INPUT=engine_g4.json TRACE_INPUT=trace_g4 NUM_GPUS=4 NUM_CPUS=16 MEM_GB=128`, through StepMind `RJobBackend` on codesign, H800 | Job `exp-0923-230103-591735`, created 15:01:03Z, succeeded 15:07:04Z. Creator `i-fengyicheng`; NFS source is the current host. Launcher exit 0 (15:00:59Z to 15:14:32Z). |
| Worker | `runs/groundtruth_clean/dpp-g4-20260923a/` | 4 H800 visible to torch 2.8.0; the overlay `vllm` 0.10.2 is imported; `WORKER_STATUS=0`; 51/51 HTTP 200; KV tokens per PP worker `[4566896, 4556400]`, so 284775 blocks per engine |
| Extraction | `extract_vllm_placement.py` → `extraction/` (receipt there) | `PASS`: 1620 iterations, 92 reports paired with receipts, 83 publications, 81 applications, 51 placements, 0 out-of-order |
| Rule-4 sizing | `extraction/chain.json` | The 20448-token first chunk takes 108.73, 116.13, 110.76 and 108.39 ms (mean 111.00 ms) |
| Launcher logs | `launcher/{submit,worker}.log` (kept locally; `*.log` is gitignored) | `WORKER_COMMAND` line removed; a credential-pattern scan found 0 matches |

The case inputs were set from G4 before G5: `frontier_g4.json` gets
`num_blocks` 284775 and `dummy_execution_time_ms` 0.8943, which is
2.0 × 111.00 / 248.25. A check run gave a Frontier first completion of
111.14 ms. Semantic rows S17 (MATCH), S31 and S33 were updated. The table
is now 37 MATCH, 5 MISMATCH and 1 UNSET, and its status is `PASS`.

### G5 simulator runs and workflow-gap analysis (2026-09-23, completed)

| Check | Command / evidence | Result |
| --- | --- | --- |
| Pre-change baseline | `git archive d1a2a06 frontier data/config`, then `runs/frontier_pre_change_d1a2a06/pre_change_rejection.py` with that tree as the working directory | `ValueError`: the constructor rejects `num_pipeline_stages=2`. C4 reports this rejection in place of a placement. |
| Simulator run | `run_frontier_case.py` at `47d9190` on `engine_g4.json`, `frontier_g4.json`, `trace_g4` (`runs/frontier_g4/`, manifest written before launch, receipt there) | 51/51 complete with tokens conserved for `vllm_load_balancing`, `completion_reporting_control` and `round_robin`. `summary.json` is byte-identical to the check run. |
| Comparison | `compare_placement.py` → `analysis/g5_comparison/` (receipt there) | T1: 51/51 routes MATCH (48/48 formal). T2: every burst `SCENARIO_NOT_REACHED`. a–c fail trace order and output-before-probe; d fails one-snapshot, provenance and output-before-probe. |
| Workflow gap | `analysis/workflow_gap_table.csv`, `workflow_gap_summary.md`, `workflow_gap_status.json` | 4 MATCH and 7 MISMATCH, each mismatch with a first cause. `COMPLETE`/`PASS`, `correction_state=pending` (S43, S42) |

Decision under the §18.20 rule: the third GPU job is not used. A retune
of spacing or offsets cannot deliver a trace-ordered burst whose first two
routes include the long body and land within 0.5 ms. The frontend spends
about 7 ms processing the 40896-token body. Plan §18.21 records the reasoning
and the two candidate findings.

### Decisions after G5 and the fix review (2026-09-24)

User message, verbatim in `requirements.md`: C4 option (a); S43 and S42 as separate tasks; remove `.worktrees/w9-05-{before,after}`; is dummy-only logic repair sound; review the existing fixes and correct what the review finds.

| Item | Status | Evidence |
| --- | --- | --- |
| C4 | completed | Recorded in `requirements.md`, the case manifest decision `G5-review` and `analysis/workflow_gap_status.json` (`correction_state=not_applicable`, the contract's no-change value for this case, with S43/S42 under `transferred_corrections`); manifest `status: PASS`. |
| W9-05 worktrees | completed | `git worktree remove` (`--force` for `after`; its uncommitted files were byte-identical to `75c1140` or already committed), then `git worktree prune`. |
| Fix review | completed | Ten commits `c647e95`..`6aee289`; each finding confirmed before it was fixed, and each tested fix checked against a tree without it (`test_report_2026-09-24_fix_review.md` sections 3–4). |
| Dummy-mode question | completed | Plan §18.22 and test report section 5: adequate for timing-independent control-flow repairs, not for per-lane timing, load imbalance, timing-selected branches or calibration closure. |
| Candidate row S44 | recorded, proposal | Test report section 6: vLLM publishes counts before draining requests that arrived during the step; 1/92 native receipts with `waiting > 0` against 24/106 Frontier reports. |
| S43, S42 tasks | completed (created) | `/data/ycfeng/Frontier/task_memory/task_2026-09-24_s43_pp_empty_schedule_admission/` and `.../task_2026-09-24_s42_dp_wave_idle_forward/` (main checkout, local records): requirements with the verbatim decision, plan with scope, evidence, steps, draft criteria and blockers (timing, pinned tools, native dummy-pass records), progress. S44 is a proposed second S43 row. |
| Record corrections | completed | Reachability claims (design, validation, review), W6 `block_shape` (`[128, 128]` natively) and rerun status (summary, validation, W6 report, review-corrections report), W7 claims (review, validation, summary, W7 report), the audit PORT row, admission anchors (plan, calibration tables, future), the W9-05 addendum in `issues.md`, and my own W3-R3 wording (no group-formation check exists; the state is unreachable). |
| Final validation | completed | `final_20260924/run_final.sh`: `ba0a804` against `6aee289` on clean detached worktrees, all five expectations in `final_20260924/expectations.md` met (test report §7): suites 0 regressions; fidelity 74/74 identical; examples 16/16 identical; stage-admission 51/51 PASS; probe 72/72 drained. |
| Push and PR bodies | completed | Pushed `b110eca` (verified by fetch); PR 35 and companion PR 1 bodies PATCHed after backups and read back; both still draft. |

Reason for each change, expectation, method and result are in the test report. Commands for the per-fix verification:

- `pytest tests/integration/test_vllm_v1_decode_preemption_runtime.py`: HEAD 4 passed. The control trees fail the case each fix names (`review_20260924/w9_05_regress/negctl/matrix.txt`).
- `pytest tests/unit/test_monolithic_mixed_forward_sync.py` on a borrowed-timing tree: 2 failed, 23 passed (`review_20260924/w3_borrowed/result.txt`).
- `pytest tests/unit/test_moe_fused_expert_arithmetic.py tests/unit/test_moe_fused_event_contract.py` (openmopd):
  - HEAD 31 passed;
  - `f236c17~1` with the new tests: 10 failed, 21 passed; 3 fail on behavior, 7 on the removed `block_dims` argument (`final_20260924/w6_negctl.txt`).
- collective-sim at `ff11ee6`: companion 12 passed, Frontier 3 passed. At `eb7bc4f` the new cases fail (`review_20260924/w7_fix/`).
- `pytest tests/integration/test_vllm_dp_placement_runtime.py`:
  - HEAD 12 passed;
  - mutation M2: both stagger cases fail (`review_20260924/w9_stagger/`).

### Workflow `wf_7606e14e-f10` results (2026-09-24)

| Item | Status | Evidence |
| --- | --- | --- |
| Workflow status | completed | 152 agents started, 29 returned, 123 failed on the platform's weekly rate limit (test report §10). |
| Six surviving findings | completed, no new change | S0–S4 are W7-R1..R3 (`ff11ee6`, `6d621c8`, `7309f5d`); S5 is recorded as the W9-04 reach addendum (`issues.md`, `validation.md` A2b; `wf_7606e14e_f10/w9_04_reach.txt`). |
| 32 unverified leads | completed | Zero verdicts each, so not refuted. Main-session triage in test report §10: 29 map to existing fixes, records or proposals; three add work: #22 (W6-R6), #26 (W6-R7) and #31 (W9-R4, beyond the existing W9-R3). |
| W6-R6 | completed | `6828581`: `test_the_tile_config_and_the_alignment_follow_fused_experts`. HEAD 32 passed; mutants m5, m6, m7 each 1 failed, 31 passed (`wf_7606e14e_f10/w6_negative_controls.txt`). |
| W6-R7, W9-R4 records | completed | `validation.md` Step 6, `review.md` W6/§14.2/S9-08, `summary.md`, the W6 report §6 and §8. |
| W7-R4 | recorded, proposal | `review.md` W7, test report §3 and §8; probe `wf_7606e14e_f10/zero_byte_allreduce_probe.txt`. |
| Dummy-mode study | completed (merged) | Case errata in `workflow_gap_summary.md` and `semantic_alignment_summary.md`/table (S10, S39); test report §5 row; S43/S42 plans. |
| S43/S42 scope | completed by the main session | The workflow's scope agents returned nothing. S43 s2 now also compares the oracle with the policy's reports (W9-R4). |

Commands (worktree root, `PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp`):

- `/data/ycfeng/envs/openmopd-py312/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_moe_fused_expert_arithmetic.py tests/unit/test_moe_fused_event_contract.py`: 32 passed (Python 3.12.13, Torch 2.8.0+cu128, vLLM 0.11.0).
- The same command on each mutant tree under `/data/ycfeng/tmp/issue26-correctness-pr/wf_followup_20260924/w6_setup_negctl/`: 1 failed, 31 passed.
