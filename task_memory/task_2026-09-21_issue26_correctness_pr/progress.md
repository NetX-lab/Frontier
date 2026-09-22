# Issue 26 Correctness PR — Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Step 0 started: records landed, environment created, baseline pending. |
| 2026-09-22 | Maintainer review dispositions recorded; Checkpoint C closed: parent merged, W2 tests strengthened, W2 re-measured with one harness revision. |
| 2026-09-22 | Checkpoint D first half: W3, the shared monolithic forward lifecycle, implemented, tested against four deliberate-defect controls, and committed as `65ed8a7`. |
| 2026-09-22 | W3 fidelity matrix measured: 71 of 71 identical against the expectation recorded before the run. Step 3 closed. |

## Status

| Field | Value |
| --- | --- |
| Correctness branch | `fix/issue26-correctness-pr` (worktree `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`) |
| Base at creation | `refactor/oversized-module-split` @ `41dabfb9d5ef3b51cdf3009d486450515d9a8d2d` (itself on `origin/main` `1f694f7`) |
| Prerequisite | MET. All four modules this PR edits are under the 2,000-line gate. The split's final record is 71 of 71 fidelity cases identical with no predictor cache differences, taken with the corrected gate; see the refactor task's Checkpoint B report. |
| Current step | Step 3 complete: implemented, measured, records written |
| Publication | PUSHED_VERIFIED (records) |
| Next action | Checkpoint D second half: W4, the opt-in vLLM-style DP placement, validating W3's shared forward identity at the report boundary per decision D1. |

## Step status

| Step | Work package | Status | Test | Publication | User review |
| --- | --- | --- | --- | --- | --- |
| 0 | Worktree, references, baseline | PASS | PASS (baseline recorded) | PUSHED_VERIFIED | NOT_REVIEWED |
| 1 | Candidate/vLLM audit | PASS | n/a (source audit) | LOCAL_ONLY | NOT_REVIEWED |
| 2 | RR DP rotation | PASS | unit PASS (23 tests); matrix PASS against a stated expectation, re-measured 2026-09-22 with one harness revision | PUSHED_VERIFIED | REVIEWED (R35-01 closed) |
| 3 | Shared monolithic forward | PASS | unit PASS (23 new, 3717 total, failure set identical to the parent); integration PASS (real event loop, 4 mixed-phase cohorts); four deliberate-defect controls each fail for their own reason; matrix PASS, 71 of 71 identical against the stated expectation | PUSHED_VERIFIED | NOT_REVIEWED |
| 4 | Opt-in vLLM DP placement | NOT_STARTED | — | — | — |
| 5 | Routing implementation identity | NOT_STARTED | — | — | — |
| 6 | Legacy fused-MoE profiling | NOT_STARTED | — | — | — |
| 7 | Optional zero-payload backend | NOT_STARTED (facts in `plan.md` A7) | — | — | — |
| 8 | Combined regression, PR hand-off | NOT_STARTED | — | — | — |

## Chronological updates

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

## Step 3 scoping, as recorded before implementation

Kept as written so the implementation can be read against the scope it started from. Step 3 is now complete; see the W3 entries above and `design.md`.

The defect is present on main at three layers, and the fix has to change all three together or the intermediate state deadlocks differently:

1. `frontier/events/replica_stage_schedule_event.py:157-184` picks the prefill or the decode sync path from the batch's own `num_prefill_tokens`, so two DP lanes of one forward step can take different paths.
2. `frontier/scheduler/utils/sync_state.py:29-40` allocates two independent waiting rooms for `MONOLITHIC`.
3. `frontier/scheduler/utils/forward_sync_state.py:38-41` partitions the open-step binding table by kind.

One refinement over the audit's framing, from reading the code: `ForwardSyncState._next_step_id_by_replica` is **already** shared across kinds, keyed by replica alone. Only `_open_steps_by_kind` is partitioned. So step-id allocation is already Replica-scoped and monotonic; what is partitioned is the binding table and the waiting room. That narrows the change.

Surface: about 3,500 lines across `replica_stage_schedule_event.py`, `sync_entry.py`, `prefill_collective.py`, `decode_collective.py`, `ep_wave_schedule.py`, `ep_wave_inputs.py` and `base_cluster_scheduler.py`, with 11 call sites of the sync-kind and sync-path selection.

Blocked hunk carried from the audit: the candidate's decode final-metrics change calls `_create_corrected_execution_time_for_metrics`, which main deleted, so it needs rewriting against main's current execution-time ownership rather than porting. Resolved by exclusion: that hunk is I8, kept out of scope under Checkpoint D's "current-main metrics ownership" and recorded in the `design.md` scope table.
