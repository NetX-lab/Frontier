## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Recorded final post-fix semantic suite and fresh TP4 x DP2 x EP8 trace. |
| 2026-09-05 | Recorded final fresh architecture smokes and the legal TP4 x DP2 x EP8 trace. |
| 2026-09-05 | Logged implementation, validation, and publication results. |
| 2026-09-05 | Recorded user-confirmed continuation for the reviewed semantic-alignment fixes. |
| 2026-09-05 | Isolated collective backend rank spaces to one Replica pod and reran architecture smokes. |
| 2026-09-05 | Closed stage-owner completion review and recorded public-example guard limitations. |
| 2026-09-05 | Fixed stale stage-end owner release and recorded fresh owner-lane validation. |

# Progress

- In progress: `SEMANTIC_ALIGNMENT_REPAIR_START: fail fast on missing admission tickets and wake stale owner queues`.

- Code-change marker: `SEMANTIC_ALIGNMENT_REPAIR_START: separate closed sync events from late Replica-local cohorts`.

### Sub-step 9: Reviewed stale-owner and admission-ticket invariants

- Authorization: user-confirmed continuation after the independent PR review;
  scope remains the existing semantic-alignment scheduler repair.
- Code-change marker: `SEMANTIC_ALIGNMENT_REPAIR_START: fail fast on missing admission tickets and wake stale owner queues`.
- Motivation: stale active stage completion could strand queued work on the
  released owner lane, while cohort promotion/restore silently discarded a
  missing admission ticket and continued with a partial cohort.
- Expectation: stale completion schedules both the owner lane and eligible
  sibling lanes; promotion and restore reject every live batch without an
  admission ticket with an explicit `ValueError`.

- Completed: created `.worktrees/semantic-alignment-repair-20260905` from `main`.
- Completed: restored co-location Replica-local DP lanes and PDD admission.
- Completed: repaired PD-AF scheduler and metrics lane contracts after two deterministic smoke failures.
- Completed: co-location tests `45 passed`; PDD tests `27 passed`; PD-AF tests `86 passed`.
- Completed: co-location, PDD, and PD-AF one-request dummy smokes exited `0` and wrote metrics.
- Completed: opened issue `https://github.com/NetX-lab/Frontier/issues/27`.
- Completed: opened PR `https://github.com/NetX-lab/Frontier/pull/28`.
- Pending outside this task: groundtruth rerun and formal timing parity.

### Sub-step 10: Final post-fix validation

- Motivation: close the reviewed stale-owner and missing-admission-ticket
  findings with evidence from the current worktree, rather than relying on
  pre-fix counts or an earlier topology trace.
- Expectation: the complete semantic suite passes; two attention-DP owners can
  transition through a dense layer; late PREFILL/DECODE lanes receive a fresh
  cohort after placeholder closure; and a fresh `TP4 x DP2 x EP8` run completes
  both requests with complete EP8 barriers.
- Method: ran the 22-file semantic command, the late-lane regressions, a direct
  two-owner transition probe, and a direct topology run with output and event
  logs under `/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905*`.
- Result: `720 passed, 19 skipped, 115 warnings`; direct transition probe passed;
  the topology run exited `0`, completed 2 requests at
  `971.9999999999973 ms` each, produced 4 `ATTN_DP_LANE` ledger rows, and
  recorded `64` EP conservation, `128` EP barrier, and `64` EP wave-end lines.
  All 128 barriers had expected and arrived IDs `[0,1,2,3,4,5,6,7]`.

## 2026-09-05 Reviewed-Fix Continuation

- Authorization: user confirmed continuation after PR #28 review.
- Code-change marker: `SEMANTIC_ALIGNMENT_REPAIR_START: reviewed attn_dp CLI, resolver, layout, scheduler, and sync identity fixes`.
- Scope: expose and propagate attention DP, preserve one Replica as one GPU pod, keep logical TP abstraction, route request ownership by `(replica_id, dp_id)`, and isolate DP-lane batch/synchronization identity.
- Out of scope: groundtruth rerun, timing comparison, legacy H800 CSV reuse, per-rank schedulers, and unrelated documentation failures.

### Sub-step 1: CLI and parallel-domain materialization

- Motivation: the reviewed PR admitted `attn_dp > 1` in Python objects but hid the field from the supported CLI, hard-coded the canonical resolver to `attn_dp=1`, and materialized attention collective DP from outer Replica capacity.
- Expectation: formal CLI/config construction and the canonical resolver produce `TP4 x DP2 x EP8`, while collective layout keeps attention DP local to one complete Replica pod.
- Method: exposed `ReplicaConfig.attn_dp`, added resolver propagation and local EP derivation, changed attention layout DP to `mapping.attn_dp`, and added parser/resolver/layout tests.
- Result: `python -m pytest tests/unit/test_parallel_semantics.py tests/unit/test_canonical_parallel_names.py -q -p no:cacheprovider` -> `8 passed`; `frontier.main --help` lists `--replica_config_attn_dp`.

### Sub-step 2: scheduler lane routing

- Motivation: reviewed non-RoundRobin schedulers still emitted `(replica_id, None)` for non-FFN requests, and load tracking mixed integer Replica keys with tuple lane keys.
- Expectation: every shared-domain request selects one explicit `(replica_id, dp_id)` owner; PD-AF `DECODE_ATTN` keeps its intentional full-stage `(replica_id, None)` identity.
- Method: routed LOR, Random, and Sticky Round Robin through Replica-local DP lanes, normalized RoundRobin load tracking, and updated the affected identity-contract tests.
- Result: `python -m pytest tests/unit/test_cluster_scheduler_dp_lanes.py tests/unit/test_replica_identity_contract.py tests/unit/test_moe_routing_conservation.py -q -p no:cacheprovider` -> `30 passed`.

### Sub-step 3: DP-scoped batch and decode-sync identity

- Motivation: independent DP child counters could reuse the same batch global ID in shared MoE waiting rooms, and decode-sync ID validation incorrectly used MoE EP cardinality.
- Expectation: batch IDs remain unique within each physical Replica's attention-DP lanes, default single-lane IDs remain stable, and decode-sync IDs accept the full configured attention-DP domain.
- Method: added Replica-local batch ID packing through the cluster scheduler, reused the lane cardinality for decode-sync IDs, and added direct identity/boundary tests.
- Result: `python -m pytest tests/unit/test_cluster_scheduler_dp_lanes.py tests/unit/test_shared_ep_layer_protocol_guard.py tests/unit/test_pd_moe_lifecycle_reproducer.py tests/unit/test_pd_decode_moe_layer_accounting.py -q -p no:cacheprovider` -> `40 passed`.

### Sub-step 4: metrics identity scope

- Motivation: stage-batch ledger rows labeled every non-`None` local identity as `EP_WAVE_LANE`, which misclassified attention-DP scheduler lanes.
- Expectation: `DECODE_FFN` local IDs remain `EP_WAVE_LANE`, shared-domain non-FFN local IDs become `ATTN_DP_LANE`, and full-stage `None` remains `FULL_STAGE_WORLD`.
- Method: centralized scope classification in `MetricsStore` and added role-specific assertions.
- Result: `python -m pytest tests/unit/test_cluster_scheduler_dp_lanes.py tests/unit/test_transfer_metrics_contract.py tests/unit/test_pdaf_m2n_metrics.py -q -p no:cacheprovider` -> `114 passed`.

### Sub-step 5: Replica-local collective backend materialization

- Motivation: runtime inspection showed `runtime_attn_dp * runtime_num_replicas`
  produced `TP4 x DP6` for three independent pods, and MoE runtime dimensions
  used outer replicas as a collective DP dimension.
- Expectation: every Cluster CC backend models one complete Replica pod, with
  attention `TP4 x DP2` and local MoE `TP1 x EP8`; outer replicas remain
  scheduler capacity only.
- Method: materialized physical topology from one `replica_config.world_size`,
  set backend runtime replica count to `1`, removed outer-replica factors from
  collective-sim and ASTRA-Sim analytical runtime dimensions, and added direct
  dual-backend regression tests.
- Result: `12 passed`; fresh direct dummy smokes all exited `0`:
  co-location E2E `1036.0 ms`, PDD E2E `60.52097152 ms`, PD-AF E2E
  `631.27610496 ms`. Post-fix materialization is world `8`, attention
  `{TP:4, CP:1, DP:2, EP:1}`, and MoE `{TP:1, CP:1, DP:1, EP:8}` for
  `num_replicas=3, attn_dp=2`.

### Sub-step 6: Correct MoE layout outer-Replica isolation

- Authorization: continuation after independent PR review; scope remains
  Replica-local collective backend materialization.
- Code-change marker: `SEMANTIC_ALIGNMENT_REPAIR_START: remove outer Replica capacity from MoE collective layout`.
- Motivation: `build_collective_sim_layout(domain="moe")` still used
  `cluster_num_replicas` as MoE DP, contradicting the materializer's
  `runtime_num_replicas=1` Replica-local backend contract.
- Expectation: MoE layout uses local `DP=1`, and its TP/CP/DP/EP product
  matches one complete Replica pod for every supported outer replica count.

### Sub-step 7: Stage-owner completion review and validation

- Motivation: the first reviewed wakeup patch could release the full-stage
  admission alias after one DP child completed instead of releasing the lane
  that created that batch. The other lane could remain busy and leave residual
  scheduler state.
- Expectation: every asynchronous PREFILL/DECODE layer transition, stage-end
  event, metrics row, and release operation follows the batch's saved
  `_stage_owner_replica_local_id`; `None` remains reserved for full-stage
  paths. A release wakes queued sibling lanes exactly once.
- Method: preserved the owner on batch creation and runtime live-batch
  materialization, threaded it through layer-sync transitions, selected the
  owner scheduler at final completion, and added a sibling-wakeup focused
  unit test.
- Result: focused owner tests passed (`37 passed`); the merged semantic
  command passed (`710 passed, 19 skipped, 115 warnings`); fresh two-request
  co-location `TP4 x DP2 x EP8` completed at `1.943999999999991 s` and
  `1.4579999999999942 s`, with E2E mean `1700.9999999999927 ms`, and no
  residual scheduler state.

### Sub-step 8: Stale stage-end sibling wakeup

- Motivation: an independently reviewed stale `BatchStageEndEvent` path
  released an active owner ticket and marked the shared stage idle, then
  returned without scheduling a queued sibling attention-DP lane. The sibling
  ticket could remain queued forever with no DES event left to retry it.
- Expectation: stale completion releases the batch owner lane and wakes each
  eligible queued sibling exactly as the normal completion path does; queued
  tickets remain owned by their lanes until their new schedule event runs.
- Method: resolve the owner identity from the batch in the stale branch, call
  the existing sibling-wakeup helper after `on_stage_end()`, and add a
  mismatch-identity regression that verifies owner release plus a returned
  sibling `ReplicaStageScheduleEvent`.
- Result: the stale-path regression passed; owner-focused tests now pass
  `37`; the complete semantic targeted command passes `710 passed, 19 skipped,
  115 warnings`. Source hygiene remains clean with `git diff --check` and
  `python -m compileall -q frontier tests/unit`.

### Review disposition

- Resolved: collective-sim and ASTRA-Sim analytical layouts no longer include
  outer Replica capacity in local collective dimensions.
- Resolved: stage completion releases the actual batch-owner lane and wakes
  queued sibling lanes.
- Resolved: stale active stage-end completion now wakes queued sibling lanes
  after releasing the batch owner, preventing residual queued tickets with no
  DES retry event.
- Deferred by approved scope: public MoE shell examples still contain legacy
  `ATTN_TP == MOE_TP * MOE_EP` guards; PD-AF `DECODE_ATTN` intentionally
  remains a full-stage role with `attn_dp=1`, so no new cluster-specific
  schema was introduced in this repair.

## Final Fresh Evidence

- Fresh artifact roots:
  - `/data/ycfeng/tmp/frontier_semantic_repair_final_coloc_20260905/`
  - `/data/ycfeng/tmp/frontier_semantic_repair_final_pdd_20260905/`
  - `/data/ycfeng/tmp/frontier_semantic_repair_final_pdaf_20260905/`
  - `/data/ycfeng/tmp/frontier_semantic_repair_final_tp4_dp2_ep8_20260905/`
  - `/data/ycfeng/tmp/frontier_semantic_repair_final_tp4_dp2_ep8_20260905_trace/`
- Architecture smoke evidence from fresh `request_metrics.csv` and
  `system_metrics.json` files:
  - co-location: one request, 10 tokens, E2E `1035.9999999999968 ms`,
    throughput `9.65250965250968 tokens/s`;
  - PDD: one request, 10 tokens, E2E `60.52097151999969 ms`, throughput
    `165.23198073076887 tokens/s`;
  - PD-AF: one request, 10 tokens, E2E `631.2761049599992 ms`, prefill
    `20.00000000000001 ms`, decode `611.2761049599992 ms`, decode-attn
    `480.0 ms`, decode-ffn `128.07099050666582 ms`.
- Direct legal topology trace used `ATTN_TP=4`, `ATTN_DP=2`, `MOE_TP=1`,
  `MOE_EP=8`, `world_size=8`, and `num_replicas=1`. Both requests completed
  with E2E `971.9999999999973 ms`, prefill `486.0000000000004 ms`, decode
  `485.99999999999693 ms`, and 6 tokens. The stage ledger contains four rows,
  `replica_local_id={0,1}`, `execution_scope={ATTN_DP_LANE}`, and request IDs
  `{0,1}`. The runtime log records `64` EP conservation entries, `128` EP
  barrier entries, `64` EP wave-end entries, and complete expected/arrived EP
  IDs `[0,1,2,3,4,5,6,7]`.
- Groundtruth remains intentionally untouched: `groundtruth_rerun=false`, and
  no legacy H800 CSV was used.

## Completion Gate Rerun

- Command: reran the complete 22-file semantic targeted suite from the current
  worktree; result `720 passed, 19 skipped, 115 warnings` with exit code `0`.
- Commands: reran `python -m compileall -q frontier tests`, parsed the repair
  receipt with `python -m json.tool`, and ran `git diff --check`; all passed.
- Fresh architecture roots:
  - `/data/ycfeng/tmp/frontier_semantic_repair_verify_coloc_20260905/`
  - `/data/ycfeng/tmp/frontier_semantic_repair_verify_pdd_20260905/`
  - `/data/ycfeng/tmp/frontier_semantic_repair_verify_pdaf_20260905/`
- Results: co-location completed 1 request at E2E `1035.9999999999968 ms` with
  throughput `9.65250965250968 tokens/s`; PDD completed 1 request at E2E
  `60.52097151999969 ms` with throughput `165.23198073076887 tokens/s`; PD-AF
  completed 1 request at E2E `631.2761049599992 ms`, prefill
  `20.00000000000001 ms`, decode `611.2761049599992 ms`, and throughput
  `15.84092906642435 tokens/s`.
- Provenance: no vLLM groundtruth rerun and no legacy H800 CSV reuse.
