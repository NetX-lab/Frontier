## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Logged implementation, validation, and publication results. |
| 2026-09-05 | Recorded user-confirmed continuation for the reviewed semantic-alignment fixes. |
| 2026-09-05 | Isolated collective backend rank spaces to one Replica pod and reran architecture smokes. |

# Progress

- Completed: created `.worktrees/semantic-alignment-repair-20260905` from `main`.
- Completed: restored co-location Replica-local DP lanes and PDD admission.
- Completed: repaired PD-AF scheduler and metrics lane contracts after two deterministic smoke failures.
- Completed: co-location tests `45 passed`; PDD tests `27 passed`; PD-AF tests `86 passed`.
- Completed: co-location, PDD, and PD-AF one-request dummy smokes exited `0` and wrote metrics.
- Completed: opened issue `https://github.com/NetX-lab/Frontier/issues/27`.
- Completed: opened PR `https://github.com/NetX-lab/Frontier/pull/28`.
- Pending outside this task: groundtruth rerun and formal timing parity.

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
