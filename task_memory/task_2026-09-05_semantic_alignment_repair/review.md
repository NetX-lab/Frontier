## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Revalidated closed stale-owner and admission-ticket findings on current worktree. |
| 2026-09-05 | Reconciled final fresh trace evidence with the reviewed repair scope. |
| 2026-09-05 | Recorded implementation review and final evidence. |
| 2026-09-05 | Added independent PR review findings and published review comments for PR #28. |
| 2026-09-05 | Reviewed Replica-local collective backend correction and fresh smoke evidence. |
| 2026-09-05 | Completed final owner-lane review and logged deferred public-entry findings. |
| 2026-09-05 | Closed stale stage-end sibling wakeup finding with regression evidence. |

# Review

Target Component/Phase: co-location -> pd-disagg -> pd-af-disagg semantic repair

Reviewer Agent Identity: primary agent `/root`; independent review lanes `/root/pr_review` and `/root/arch_review`; validation lane `/root/repair_validation`

Inspected Artifacts: scheduler/config diffs, parallel mapping and collective layout code, scheduler identity paths, targeted unit output, direct smoke logs, metrics CSV/JSON, issue #27, PR #28

Identified Issues/Anomalies: PR #28 still cannot expose `attn_dp` through CLI; the canonical resolver and collective-sim attention layout ignore `attn_dp`; shared MoE sync-room IDs can collide across DP child schedulers; LOR/Random/Sticky schedulers lose DP lane ownership; decode sync IDs use EP cardinality for DP lanes; targeted lane-contract tests remain red. PD-AF direct smoke also reproduced a deterministic `_replica_load_tracker` key-shape failure in the validation lane. Two README documentation failures are unrelated to this PR.

Remediation/Verification Code Actions Taken: the initial implementation was reviewed and the findings were published to [PR #28 review](https://github.com/NetX-lab/Frontier/pull/28#pullrequestreview-5119710255), with eight numbered findings and a request-changes recommendation. The scheduler and metrics findings were fixed; the remaining backend finding is recorded and resolved below. The full unit suite still has unrelated README documentation failures outside this task.

Target Component/Phase: collective-sim and ASTRA-Sim analytical runtime materialization

Reviewer Agent Identity: primary agent `/root`; independent validation lane `/root/repair_validation`

Inspected Artifacts: `frontier/entities/cluster.py`, `frontier/cc_backend/backends/collective_sim_cc_backend.py`, `frontier/cc_backend/backends/astra_sim_analytical_cc_backend.py`, `tests/unit/test_cc_backend_replica_local_layout.py`, and fresh materialization output for `num_replicas=3`, `attn_dp=2`, `TP=4`, `EP=8`

Identified Issues/Anomalies: pre-repair runtime dimensions were world `24`, attention `TP4 x DP6`, and MoE `TP1 x DP3 x EP8`, which merged independent outer Replicas into collective domains.

Remediation/Verification Code Actions Taken: materialization now uses one Replica `world_size` and runtime replica count `1`; backend helpers keep attention `DP=2` and local MoE `DP=1, EP=8`. Focused regression returned `12 passed`; merged semantic targeted units returned `232 passed, 115 warnings`; co-location, PDD, and PD-AF direct smokes each exited `0`.

Target Component/Phase: asynchronous shared-domain stage completion

Reviewer Agent Identity: primary `/root`, independent architecture review
`/root/final_arch_review`, independent code review `/root/final_code_review`

Inspected Artifacts: `BatchStageEndEvent`, `ReplicaStageScheduleEvent`,
`BaseClusterScheduler._build_stage_execution_contexts`, admission release and
transition helpers, `BaseReplicaScheduler._create_batch`, runtime live-batch
materialization, owner-lane unit tests, and fresh two-request co-location
smoke logs.

Identified Issues/Anomalies: the original review correctly identified a
shared stage context and a lost owner identity. The current patch retains the
shared context as an intentional Replica-stage serialization approximation,
but explicitly wakes queued sibling lanes after release. The final completion
path now selects the batch owner rather than the historical `None` alias.
Independent re-review found no remaining owner-release or duplicate-wakeup
defect in the current diff. It also confirmed that public MoE wrappers retain
legacy parallel guards and that PD-AF `DECODE_ATTN` intentionally rejects
`attn_dp != 1`.

Remediation/Verification Code Actions Taken: owner identity is stored on
created and materialized batches, propagated through PREFILL/DECODE sync
events, used for stage scheduler and metrics selection, and released by the
matching `BatchStageEndEvent`. A stale active completion now invokes the same
sibling wakeup helper after releasing the batch owner; a mismatch-identity
regression confirms the batch owner remains authoritative. Owner-focused tests
pass `37`; the merged semantic command passes `710 passed, 19 skipped, 115
warnings`. The legacy shell guards and PD-AF schema limitation remain deferred
PR comments rather than changes outside the approved runtime scope.

Target Component/Phase: stale `BatchStageEndEvent` release path

Reviewer Agent Identity: primary `/root`; independent code review `/root/final_code_review`

Inspected Artifacts: `frontier/events/batch_stage_end_event.py`,
`StageExecutionContext` ticket release, sibling wakeup helper, and
`tests/unit/test_stage_execution_context.py`

Identified Issues/Anomalies: the review found that stale active completion could
leave a queued sibling ticket in the shared context after the owner was marked
idle, with no DES event remaining to retry the sibling.

Remediation/Verification Code Actions Taken: stale completion now uses the
batch's `_stage_owner_replica_local_id` for release and calls
`get_waiting_replica_stage_schedule_events()` after `on_stage_end()`. The
regression constructs mismatched event and batch identities, then verifies
owner release, idle context, and a returned sibling schedule event. The finding
is closed; no duplicate wakeup was observed because busy and empty lanes are
filtered.

Target Component/Phase: final fresh architecture and legal topology validation

Reviewer Agent Identity: primary `/root`, with prior independent reviews
`/root/pr_review`, `/root/final_arch_review`, and `/root/final_code_review`

Inspected Artifacts: fresh request/system metrics under
`/data/ycfeng/tmp/frontier_semantic_repair_final_*_20260905/`, the direct trace
under `/data/ycfeng/tmp/frontier_semantic_repair_final_tp4_dp2_ep8_20260905_trace/`,
and the associated runtime log.

Identified Issues/Anomalies: no new runtime defect. The public shell wrappers
still reject the valid `ATTN_TP=4, ATTN_DP=2, MOE_TP=1, MOE_EP=8` topology through
their legacy guard; this remains deferred because wrapper changes are outside
the approved runtime scope. Groundtruth rerun remains intentionally disabled.

Remediation/Verification Code Actions Taken: fresh co-location, PDD, and
PD-AF runs exited `0` and completed one request each. The direct legal topology
run completed two requests at E2E `971.9999999999973 ms` each, produced four
`ATTN_DP_LANE` stage-ledger rows, and logged `64` conservation, `128` barrier,
and `64` wave-end entries with all EP IDs `[0..7]` arriving. The evidence is
recorded as `PASS` in the repair receipt with `groundtruth_rerun=false`.

Target Component/Phase: final post-fix stale-owner, cohort, and mixed-layer review

Reviewer Agent Identity: primary `/root`, independently rechecked against the
prior findings from `/root/final_arch_review` and `/root/final_code_review`

Inspected Artifacts: `StageExecutionContext.transition_active_scope`,
`_resolve_forward_sync_cohort`, PREFILL/DECODE late-lane regressions,
missing-ticket promotion/restore regressions, the 22-file semantic suite, and
the fresh topology run under `/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905*`

Identified Issues/Anomalies: the earlier mixed-layer transition and late-lane
cohort concerns are no longer reproducible in the current worktree. Capacity
2 allows two independent full-stage owners to transition; open cohorts admit a
same-time real replacement, while closed cohorts allocate a new ID. No new
runtime defect was found. The two README marker failures and public wrapper
legacy guard remain deferred, as recorded above.

Remediation/Verification Code Actions Taken: the direct two-owner transition
probe passed; PREFILL/DECODE late-lane tests passed; production promotion and
restoration now fail fast on a missing admission ticket; the full semantic
suite passed `720`, skipped `19`, with `115` warnings. The fresh `TP4 x DP2 x
EP8` run exited `0`, completed two requests at `971.9999999999973 ms` each,
and recorded complete EP8 barrier arrival sets. Status: `PASS` for the
approved semantic repair scope; `REQUEST CHANGES` remains the PR disposition
until the deferred wrapper/schema follow-up is handled separately.
