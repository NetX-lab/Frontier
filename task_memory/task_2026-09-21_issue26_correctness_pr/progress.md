# Issue 26 Correctness PR — Progress

## Modification History

| Date | Change |
| --- | --- |
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
| Current step | Step 8 closed; external review corrections A–E applied 2026-09-22 (`test_report_2026-09-22_review_corrections.md`). Step 9 is planned (`plan.md` §18, corrected per the review) and **not started**. |
| Publication | PUSHED_VERIFIED: `f7c31e4` (C35-01 source + tests), `ca1b9b6` (FP8 `block_shape`, optional-torch skip, W6 report, profiling guide), `57ffa5b` (records, Step 9 plan corrections); remote head `57ffa5b` confirmed; PR34 correction `2310417` merged in as `0d025f8`. Draft PR 35 body PATCHed 2026-09-22T12:12Z through `gh api` and read back; still draft, MERGEABLE, base `refactor/oversized-module-split`. |
| Next action | **User decision.** Start signal for Step 9, at the first node of the `plan.md` §18.5 graph (plan amended by the §18.12 second review). The FP8 native rerun is done and PASS. PR 35 remains draft and nothing was merged. Carried forward in `future.md`: re-point the collective-sim gitlink at `main` once companion PR 1 merges, and repair the `tests/debug/` pointers that 10 baseline unit failures share. Retargeting PR 35's base to `main` waits on PR 34. |

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
| 9 | PP>1 support for `vllm_load_balancing` | PLANNED — plan corrected per the external review (`plan.md` §18.11); **not started** | n/a | PUSHED_VERIFIED (records only) | awaiting the user's start signal |

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
| P1(b) Frontier boundary probe | blocked | Three shapes probed (`attn_dp=2 PP=1`, `attn_dp=1 PP=2`, `attn_dp=1 PP=3`), tables in `plan.md` §18.13. The fourth shape, MoE `attn_dp=2, moe_ep=2, PP=2`, drains the event queue with requests unfinished — pre-existing defect W9-01 in `issues.md`. |
| Design checkpoint (D9-1, D9-2) | open | D9-1's payload is settled (the completion hook signature already carries lane, load and a key source). D9-2 is not: the candidate key `ForwardSyncState._next_step_id_by_replica` satisfies I1, I2, I3, I4 and I6 on the runnable shapes but fails I5, and no alternative can be checked against I1 without a running `attn_dp>1, PP>1` shape. |
| P2–P6, G3–G5 | paused | All depend on the design checkpoint or on that shape. |

W9-01 is not caused by this PR: `stage_execution_context.py`, `replica_stage_schduler.py` and `stage_contexts.py` are byte-identical to `main`. It is unobserved because every Simulator-level test with `attn_dp > 1` uses `num_pipeline_stages = 1` and no shipped example sets `attn_dp > 1`. Scope decision requested from the user; recommendation is to fix it as a separate correctness item rather than inside this feature branch.

W9-02: `attn_dp=2, moe_ep=2, PP=3` is rejected at construction (6 devices against node size 4). Plan C1's PP3 row amended to `attn_dp=1`.

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
