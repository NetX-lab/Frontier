## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Closed stale-owner wakeup and missing-admission-ticket findings with fresh rerun evidence. |
| 2026-09-05 | Closed final fresh smoke and TP4 x DP2 x EP8 trace evidence. |
| 2026-09-05 | Recorded and resolved the PD-AF runtime blockers found by direct smoke testing. |
| 2026-09-05 | Resolved outer-Replica collective contamination in runtime backends. |
| 2026-09-05 | Recorded stage-owner completion repair and deferred shell/schema findings. |
| 2026-09-05 | Resolved stale stage-end sibling wakeup omission. |

# Issues

Resolved: `_replica_load_tracker` used integer keys while the repaired update path unpacked `(replica_id, dp_id)` keys. PD-AF now tracks full-stage Replica keys as integers and maps them to `(replica_id, None)` scheduler entries.

Resolved: utilization metrics allocated no local meters for non-FFN DP lanes, while repaired events supplied DP lane IDs. Metrics now size non-PD-AF non-FFN local scopes by `attn_dp`; `DECODE_ATTN` continues to use full-stage meters.

Resolved: PD-AF return batches do not carry `decode_attn_original_dp_id`. The return path now treats the absent field as the required full-stage `None` identity and rejects unexpected non-`None` values.

Open: documentation contract tests still report two pre-existing README marker failures; they are unrelated to this runtime repair and remain outside the approved scope.

Resolved: collective-sim and ASTRA-Sim analytical runtime materialization
treated outer Cluster replicas as attention-DP and MoE-DP participants. Each
Cluster backend now materializes one Replica pod (`world_size=8` for the
validated `TP4 x DP2 x EP8` case); outer `num_replicas` remains scheduler
capacity and no workload-level cross-Replica collective is modeled.

Resolved: shared MoE stage completion could release the full-stage alias after
one DP child lane completed. Batch creation and runtime live-batch
materialization now retain `_stage_owner_replica_local_id`; PREFILL/DECODE
completion resolves the owner lane before releasing its admission ticket and
schedules queued sibling lanes at the release boundary.

Resolved: stale `BatchStageEndEvent` completion released an active ticket and
marked the shared stage idle but returned without scheduling a queued sibling.
The stale path now resolves the batch owner identity, releases that owner, and
uses the same sibling-wakeup helper as normal completion. The regression also
covers an event identity that disagrees with the batch owner, proving that the
batch owner remains authoritative.

Resolved: cohort promotion and full-stage restoration previously filtered a
live batch without `_stage_admission_ticket` and continued with a partial
cohort. Production scheduler paths now fail fast with `ValueError`; standalone
layer probes retain their explicit no-context compatibility path without
claiming stage ownership. Parameterized regressions cover both promotion and
restoration.

Open, deferred by scope: public co-location, PDD, and PD-AF MoE shell wrappers
still validate the legacy `ATTN_TP == MOE_TP * MOE_EP` relation. A valid
`ATTN_TP=4, ATTN_DP=2, MOE_TP=1, MOE_EP=8` topology therefore exits at the
wrapper guard even though direct CLI/config construction succeeds. Updating
all online/offline wrappers is a cross-cutting example-surface change beyond
the approved runtime repair and is recorded in the PR review comments.

Open, intentional contract: PD-AF `DECODE_ATTN` has no cluster-specific
`attn_dp` field and inherits the top-level value before `ReplicaConfig` checks
the role. The role deliberately requires `attn_dp=1` and uses
`replica_local_id=None`; enabling DP for PD-AF PREFILL would require a new
cluster config field and a separate design decision.

Resolved, final evidence: fresh co-location, PDD, and PD-AF artifacts all
contain one completed request and the expected metrics. A fresh direct
`TP4 x DP2 x EP8` run completed both requests with four `ATTN_DP_LANE` stage
ledger rows, no residual scheduler state, and complete eight-member EP
barriers. The trace confirms that outer Replica capacity remains outside the
collective domain: the materialized world is `8`, attention is `TP4 x DP2`,
and MoE is `TP1 x DP1 x EP8`.

The post-fix rerun is archived under
`/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905/` with
the event log under
`/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905_trace/`.
It exited `0`, completed 2 requests at `971.9999999999973 ms` each, and
recorded `64` EP conservation, `128` EP barrier, and `64` EP wave-end entries.
