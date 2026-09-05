## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Added final post-fix suite count and fresh TP4 x DP2 x EP8 rerun path. |
| 2026-09-05 | Added final fresh architecture and TP4 x DP2 x EP8 validation evidence. |
| 2026-09-05 | Archived the semantic-alignment repair deliverables and validation status. |
| 2026-09-05 | Added Replica-local collective backend repair and fresh validation metrics. |
| 2026-09-05 | Added final owner-lane review and deferred public-entry findings. |
| 2026-09-05 | Added stale completion repair and final validation counts. |

# Task Overview

Restored vLLM-compatible Replica-local attention-DP request ownership while preserving Frontier's logical TP and shared EP abstractions.

# Deliverables Inventory

- `frontier/config/config.py`
- `frontier/config/parallel_semantics.py`
- `frontier/cc_backend/backends/collective_sim_cc_backend.py`
- `frontier/cc_backend/backends/astra_sim_analytical_cc_backend.py`
- `frontier/cc_backend/cc_backend_config.py`
- `frontier/entities/cluster.py`
- `frontier/execution_time_predictor/base_execution_time_predictor.py`
- `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`
- `frontier/metrics/metrics_store.py`
- `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`
- `frontier/scheduler/cluster_scheduler/lor_cluster_scheduler.py`
- `frontier/scheduler/cluster_scheduler/random_cluster_scheduler.py`
- `frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py`
- `frontier/scheduler/cluster_scheduler/sticky_round_robin_cluster_scheduler.py`
- `frontier/scheduler/replica_scheduler/base_replica_scheduler.py`
- `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py`
- `frontier/events/batch_stage_end_event.py`
- `tests/unit/test_cc_backend_replica_local_layout.py`
- `tests/unit/test_canonical_parallel_names.py`
- `tests/unit/test_cluster_scheduler_dp_lanes.py`
- `tests/unit/test_moe_predictor_layer_id_semantics.py`
- `tests/unit/test_moe_routing_conservation.py`
- `tests/unit/test_parallel_semantics.py`
- `tests/unit/test_pdaf_config_contract.py`
- `tests/unit/test_replica_identity_contract.py`
- `tests/unit/test_stage_execution_context.py`
- `tests/unit/test_pdaf_prefill_model_time.py`
- `repairs/semantic_alignment_repair_receipt.json`
- `task_memory/task_2026-09-05_semantic_alignment_repair/test_report_2026-09-05_architecture_smoke.md`
- Issue: https://github.com/NetX-lab/Frontier/issues/27
- PR: https://github.com/NetX-lab/Frontier/pull/28

# Validation Status

| Surface | Result |
| --- | --- |
| Co-location targeted units | PASS: 45 passed |
| PDD targeted units | PASS: 27 passed |
| PD-AF targeted units | PASS: 86 passed |
| Replica-local backend regression | PASS: 12 passed |
| Merged semantic targeted units | PASS: 720 passed, 19 skipped, 115 warnings |
| Co-location direct dummy smoke | PASS: exit 0, E2E 1036.0 ms, 10 tokens, 9.65250965250968 tokens/s |
| PDD direct dummy smoke | PASS: exit 0, E2E 60.52097152 ms, 10 tokens, 165.23198073076887 tokens/s |
| PD-AF direct dummy smoke | PASS: exit 0, E2E 631.27610496 ms, 10 tokens, 15.84092906642435 tokens/s |
| Backend materialization (`num_replicas=3`, `attn_dp=2`) | PASS: world 8; attention `TP4 x DP2`; MoE `TP1 x EP8` |
| Dual-DP co-location owner-lane smoke | PASS: 2 requests completed; completion `1.943999999999991 s` / `1.4579999999999942 s`; E2E mean `1700.9999999999927 ms`; no residual scheduler state |
| Owner/wakeup focused units | PASS: 37 passed |
| Merged semantic targeted units | PASS: 720 passed, 19 skipped, 115 warnings |
| Source hygiene | PASS: `git diff --check`; `compileall` |

The final fresh trace used one complete Replica pod with logical TP abstraction:
`ATTN_TP=4 x ATTN_DP=2` and Replica-local `MOE_TP=1 x MOE_EP=8`. Two requests
completed at E2E `971.9999999999973 ms` each; the four stage-ledger rows were
scoped to `ATTN_DP_LANE` and covered both request owners `{0,1}`. The runtime
log recorded `64` conservation entries, `128` eight-member EP barriers, and
`64` EP wave ends. Outer `num_replicas` remained scheduler capacity and did
not expand the collective world.

The post-fix rerun under
`/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905/` and
`/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905_trace/`
exited `0` with the same two-request result and complete EP8 barrier sets.

# Open Items/Future Extensions

Groundtruth rerun, formal Frontier-vLLM parity, and legacy H800 data regeneration remain outside this session.

The stale stage-end path now releases the saved batch owner and returns a
sibling schedule event after `on_stage_end()`, so queued attention-DP tickets
cannot remain stranded after a stale active completion.

Public MoE shell wrappers retain a legacy `ATTN_TP == MOE_TP * MOE_EP`
preflight guard and PD-AF `DECODE_ATTN` intentionally remains `attn_dp=1`;
these entry/schema changes remain deferred for a follow-up design decision.

Fresh artifact paths are retained under `/data/ycfeng/tmp/` in the test report;
they are not added to git. Groundtruth rerun and legacy H800 CSV reuse remain
disabled by design.
