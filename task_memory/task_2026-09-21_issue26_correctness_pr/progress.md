# Issue 26 Correctness PR — Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Step 0 started: records landed, environment created, baseline pending. |

## Status

| Field | Value |
| --- | --- |
| Correctness branch | `fix/issue26-correctness-pr` (worktree `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`) |
| Base at creation | `refactor/oversized-module-split` @ `41dabfb9d5ef3b51cdf3009d486450515d9a8d2d` (itself on `origin/main` `1f694f7`) |
| Prerequisite | MET. All four modules this PR edits are under the 2,000-line gate; the split measured 67 of 67 fidelity cases identical with no predictor cache name differences. |
| Current step | Step 2 complete; Step 3 scoped, not started |
| Publication | PUSHED_VERIFIED (records) |
| Next action | Step 3: the shared monolithic forward lifecycle. Scoped below; not started. |

## Step status

| Step | Work package | Status | Test | Publication | User review |
| --- | --- | --- | --- | --- | --- |
| 0 | Worktree, references, baseline | PASS | PASS (baseline recorded) | PUSHED_VERIFIED | NOT_REVIEWED |
| 1 | Candidate/vLLM audit | PASS | n/a (source audit) | LOCAL_ONLY | NOT_REVIEWED |
| 2 | RR DP rotation | PASS | unit PASS; matrix PASS against a stated expectation | PUSHED_VERIFIED | NOT_REVIEWED |
| 3 | Shared monolithic forward | NOT_STARTED | — | — | — |
| 4 | Opt-in vLLM DP placement | NOT_STARTED | — | — | — |
| 5 | Routing implementation identity | NOT_STARTED | — | — | — |
| 6 | Legacy fused-MoE profiling | NOT_STARTED | — | — | — |
| 7 | Optional zero-payload backend | NOT_STARTED (facts in `plan.md` A7) | — | — | — |
| 8 | Combined regression, PR hand-off | NOT_STARTED | — | — | — |

## Chronological updates

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

## Step 3 scoping, not started

The defect is present on main at three layers, and the fix has to change all three together or the intermediate state deadlocks differently:

1. `frontier/events/replica_stage_schedule_event.py:157-184` picks the prefill or the decode sync path from the batch's own `num_prefill_tokens`, so two DP lanes of one forward step can take different paths.
2. `frontier/scheduler/utils/sync_state.py:29-40` allocates two independent waiting rooms for `MONOLITHIC`.
3. `frontier/scheduler/utils/forward_sync_state.py:38-41` partitions the open-step binding table by kind.

One refinement over the audit's framing, from reading the code: `ForwardSyncState._next_step_id_by_replica` is **already** shared across kinds, keyed by replica alone. Only `_open_steps_by_kind` is partitioned. So step-id allocation is already Replica-scoped and monotonic; what is partitioned is the binding table and the waiting room. That narrows the change.

Surface: about 3,500 lines across `replica_stage_schedule_event.py`, `sync_entry.py`, `prefill_collective.py`, `decode_collective.py`, `ep_wave_schedule.py`, `ep_wave_inputs.py` and `base_cluster_scheduler.py`, with 11 call sites of the sync-kind and sync-path selection.

Blocked hunk carried from the audit: the candidate's decode final-metrics change calls `_create_corrected_execution_time_for_metrics`, which main deleted, so it needs rewriting against main's current execution-time ownership rather than porting.
