## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Recorded the final 220-test focused rerun and clean compile/diff/status evidence. |
| 2026-09-15 | Added direct actual-replica-ID routing coverage and follow-up production constructor rerun evidence. |
| 2026-09-15 | Recorded the focused CPU regression that closes monolithic MoE routing identity drift and the manager device-event path contract. |

# Monolithic MoE Routing Identity and Manager Path Contract Report

## Execution

Worktree:

`/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`

Environment: Python 3.12.3, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, PyTorch 2.5.1+cu124; no Conda environment; `vllm`, `sglang`, and `aiter` are unavailable.

Focused command:

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn
PYTHONPATH=$PWD \
WANDB_DISABLED=true \
VIDUR_DISABLE_WANDB=1 \
FRONTIER_LOG_LEVEL=ERROR \
python -m pytest \
  tests/unit/test_device_timer_contract.py \
  tests/unit/test_gdn_training_predictor_increment8.py \
  tests/unit/test_gdn_hybrid_e2e_increment14ab.py \
  tests/unit/test_execution_time_predictor_max_tokens_budget.py \
  tests/unit/test_attention_tp_effective_mapping.py \
  tests/unit/test_moe_ep_non_dummy_matrix.py \
  tests/unit/test_measurement_family_selector.py \
  -q -p no:cacheprovider \
  --basetemp /data/ycfeng/tmp/pr31-wiring-focused-20260915-final2
```

Targeted commands:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR \
python -m pytest tests/unit/test_gdn_hybrid_e2e_increment14ab.py::test_hybrid_gdn_production_constructor_cpu_e2e \
  -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr31-wiring-production-20260915-routingfix

PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR \
python -m pytest tests/unit/test_moe_predictor_layer_id_semantics.py::test_monolithic_predictor_exposes_global_per_replica_layer_routing_details \
  -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr31-routing-contract-20260915

PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR \
python -m pytest tests/unit/test_measurement_family_selector.py::test_shared_manager_returns_complete_training_file_paths \
  -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr31-path-contract-20260915
```

## Criteria

1. A monolithic hybrid predictor must use the actual process-global `Replica.id` values that the scheduler places on `Batch.replica_id`.
2. Direct predictor tests that omit actual IDs must retain the established local-index routing map contract.
3. The manager's complete path API must expose compute, attention, and MoE device-event derivatives, including configured overrides and empty-path behavior.
4. Existing timer, GDN, hybrid, predictor, MoE, attention, and measurement-family behavior must remain green.

## Evidence

- **PASS — production constructor:** `1 passed in 4.43s`. The real synthetic profile fixture trained/loaded GDN, standard attention, and standard MoE models, constructed the Registry and Simulator, and completed one request.
- **PASS — routing identity:** the production test asserted that predictor routing keys equal `Simulator._clusters[ClusterType.MONOLITHIC].replicas` keys; the direct local-index routing contract passed `1` test in `2.60s`, and the direct actual-ID contract passed `1` test in `2.58s`.
- **PASS — manager paths:** the complete training-file-path dictionary contract passed `1` test in `2.73s`, including `compute_device_event_input_file`, `attention_device_event_input_file`, and `moe_device_event_input_file`.
- **PASS — focused suite:** `219 passed in 13.54s`.
- **PASS — final focused suite with direct actual-ID coverage:** `220 passed in 15.05s`.

The repaired failure was observed before this run as `routing_details missing target_replica_id 1`; the map had only local key `0`. The correction carries actual cluster IDs through `Simulator` and the MoE Registry and validates list type, cardinality, non-negative integer values, and uniqueness.

`SKIP: AMD/MI355X hardware unavailable`. These checks establish CPU construction, identity, and control-flow behavior only. They do not establish ROCm `DEVICE_EVENT`, AMD kernel timing, RCCL/AITER/MXFP4/SGLang runtime correctness, benchmark parity, or groundtruth parity.

## Full unit regression after the repair

Command:

```bash
PYTHONPATH=$PWD \
WANDB_DISABLED=true \
VIDUR_DISABLE_WANDB=1 \
FRONTIER_LOG_LEVEL=ERROR \
python -m pytest tests/unit -q -p no:cacheprovider \
  --basetemp /data/ycfeng/tmp/pr31-full-unit-20260915-routingfix
```

Observed result: **3276 passed, 19 failed, 25 skipped, 576 warnings** in **74.32s**. The complete log is `/data/ycfeng/tmp/pr31-full-unit-20260915-routingfix.log`. The 19 failing test names are identical to the baseline inventory: one missing `frontier.config_optimizer`, nine missing debug E2E assets, two stale release-example documentation contracts, four existing MLA/MHA/MQA analysis-builder contracts, and three stale top-level PDD documentation contracts. No candidate-only failure was introduced by this repair.
