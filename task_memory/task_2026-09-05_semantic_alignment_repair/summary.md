## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Archived the semantic-alignment repair deliverables and validation status. |
| 2026-09-05 | Added Replica-local collective backend repair and fresh validation metrics. |

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
- `tests/unit/test_cc_backend_replica_local_layout.py`
- `tests/unit/test_canonical_parallel_names.py`
- `tests/unit/test_cluster_scheduler_dp_lanes.py`
- `tests/unit/test_moe_predictor_layer_id_semantics.py`
- `tests/unit/test_moe_routing_conservation.py`
- `tests/unit/test_parallel_semantics.py`
- `tests/unit/test_pdaf_config_contract.py`
- `tests/unit/test_replica_identity_contract.py`
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
| Merged semantic targeted units | PASS: 232 passed, 115 warnings |
| Co-location direct dummy smoke | PASS: exit 0, E2E 1036.0 ms, 10 tokens, 9.65250965250968 tokens/s |
| PDD direct dummy smoke | PASS: exit 0, E2E 60.52097152 ms, 10 tokens, 165.23198073076887 tokens/s |
| PD-AF direct dummy smoke | PASS: exit 0, E2E 631.27610496 ms, 10 tokens, 15.84092906642435 tokens/s |
| Backend materialization (`num_replicas=3`, `attn_dp=2`) | PASS: world 8; attention `TP4 x DP2`; MoE `TP1 x EP8` |

# Open Items/Future Extensions

Groundtruth rerun, formal Frontier-vLLM parity, and legacy H800 data regeneration remain outside this session.
