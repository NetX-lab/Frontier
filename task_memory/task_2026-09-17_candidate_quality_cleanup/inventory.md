## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Enumerated every production diff path at the frozen revisions; ranked dependency and risk. |

# Frozen diff inventory

Comparison: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...c288a19f59bec09529ee18d782fa57218da2c781`.

Ranks: R1 = runtime contracts; R2 = metrics/reporting; R3 = profiling; R4 = training/config. Added lines are an inspection signal, not a quality score. Final disposition is reconciled with refactoring_record and the independent review inventories.

| File | Added | Removed | Rank |
| --- | ---: | ---: | --- |
| `frontier/attention/__init__.py` | 10 | 0 | R1 |
| `frontier/attention/families.py` | 85 | 0 | R1 |
| `frontier/attention/gdn/__init__.py` | 38 | 0 | R1 |
| `frontier/attention/gdn/config.py` | 265 | 0 | R1 |
| `frontier/attention/gdn/features.py` | 252 | 0 | R1 |
| `frontier/attention/gdn/guards.py` | 48 | 0 | R1 |
| `frontier/attention/gdn/memory.py` | 77 | 0 | R1 |
| `frontier/attention/gdn/state.py` | 68 | 0 | R1 |
| `frontier/attention/model_binding.py` | 113 | 0 | R1 |
| `frontier/attention/ops.py` | 1 | 0 | R1 |
| `frontier/attention/profiling_mapping.py` | 23 | 0 | R1 |
| `frontier/attention/trace_mapping.py` | 2 | 8 | R2 |
| `frontier/config/config.py` | 62 | 0 | R4 |
| `frontier/config/device_sku_config.py` | 14 | 0 | R4 |
| `frontier/config/model_config.py` | 156 | 8 | R4 |
| `frontier/config/node_sku_config.py` | 12 | 0 | R4 |
| `frontier/config/quantization_manager.py` | 50 | 6 | R4 |
| `frontier/config/utils.py` | 6 | 0 | R4 |
| `frontier/entities/__init__.py` | 2 | 0 | R1 |
| `frontier/entities/execution_time.py` | 259 | 149 | R1 |
| `frontier/entities/stage_execution_time.py` | 436 | 0 | R1 |
| `frontier/events/decode_sync_event.py` | 1 | 0 | R1 |
| `frontier/events/prefill_sync_event.py` | 1 | 0 | R1 |
| `frontier/execution_time_predictor/attention_tp_policy.py` | 2 | 2 | R1 |
| `frontier/execution_time_predictor/base_execution_time_predictor.py` | 38 | 4 | R1 |
| `frontier/execution_time_predictor/cache_io.py` | 32 | 4 | R4 |
| `frontier/execution_time_predictor/gdn_predictor.py` | 318 | 0 | R4 |
| `frontier/execution_time_predictor/measurement_input_paths.py` | 163 | 0 | R4 |
| `frontier/execution_time_predictor/random_forrest_execution_time_predictor.py` | 10 | 3 | R1 |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | 176 | 100 | R4 |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | 107 | 180 | R1 |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 157 | 209 | R1 |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 202 | 141 | R1 |
| `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py` | 2 | 0 | R1 |
| `frontier/metrics/constants.py` | 6 | 0 | R2 |
| `frontier/metrics/ep_wave_metrics.py` | 86 | 0 | R2 |
| `frontier/metrics/metrics_store.py` | 403 | 33 | R2 |
| `frontier/metrics/op_trace_utils.py` | 16 | 0 | R2 |
| `frontier/model_architectures.py` | 142 | 14 | R1 |
| `frontier/moe_ep_workload.py` | 78 | 10 | R1 |
| `frontier/operators/binding.py` | 5 | 5 | R1 |
| `frontier/operators/typed_contracts.py` | 11 | 11 | R1 |
| `frontier/profiling/attention/backends/__init__.py` | 14 | 0 | R3 |
| `frontier/profiling/attention/backends/vllm_rocm_attention_wrapper.py` | 329 | 0 | R3 |
| `frontier/profiling/collectives/benchmark_runner.py` | 13 | 6 | R3 |
| `frontier/profiling/collectives/collectives_impl.py` | 5 | 0 | R3 |
| `frontier/profiling/collectives/collectives_input.py` | 32 | 4 | R3 |
| `frontier/profiling/collectives/collectives_wrapper.py` | 7 | 2 | R3 |
| `frontier/profiling/collectives/main.py` | 198 | 55 | R3 |
| `frontier/profiling/common/accelerator.py` | 232 | 0 | R3 |
| `frontier/profiling/common/constants.py` | 4 | 0 | R3 |
| `frontier/profiling/common/cuda_timer.py` | 11 | 106 | R3 |
| `frontier/profiling/common/device_timer.py` | 130 | 0 | R3 |
| `frontier/profiling/common/layers/layernorm.py` | 29 | 4 | R3 |
| `frontier/profiling/common/layers/rotary_embedding.py` | 54 | 1 | R3 |
| `frontier/profiling/common/model_config.py` | 108 | 2 | R3 |
| `frontier/profiling/common/timer_stats_store.py` | 22 | 7 | R3 |
| `frontier/profiling/common/vllm_compat.py` | 34 | 0 | R3 |
| `frontier/profiling/experimental/__init__.py` | 1 | 0 | R3 |
| `frontier/profiling/experimental/sglang/__init__.py` | 8 | 0 | R3 |
| `frontier/profiling/experimental/sglang/attention.py` | 277 | 0 | R3 |
| `frontier/profiling/experimental/sglang/dense.py` | 144 | 0 | R3 |
| `frontier/profiling/experimental/sglang/gdn.py` | 178 | 0 | R3 |
| `frontier/profiling/experimental/sglang/gdn_trace.py` | 318 | 0 | R3 |
| `frontier/profiling/experimental/sglang/graph_replay.py` | 513 | 0 | R3 |
| `frontier/profiling/experimental/sglang/moe.py` | 342 | 0 | R3 |
| `frontier/profiling/experimental/sglang/routed_moe_replay.py` | 269 | 0 | R3 |
| `frontier/profiling/gdn/README.md` | 22 | 0 | R3 |
| `frontier/profiling/gdn/__init__.py` | 20 | 0 | R3 |
| `frontier/profiling/gdn/inputs.py` | 278 | 0 | R3 |
| `frontier/profiling/gdn/main.py` | 147 | 0 | R3 |
| `frontier/profiling/gdn/vllm_wrapper.py` | 741 | 0 | R3 |
| `frontier/profiling/linear_op/linear_op_impl.py` | 10 | 9 | R3 |
| `frontier/profiling/linear_op/linear_op_wrapper.py` | 1 | 1 | R3 |
| `frontier/profiling/linear_op/main.py` | 7 | 51 | R3 |
| `frontier/profiling/linear_op/profiling_plan.py` | 10 | 10 | R3 |
| `frontier/profiling/moe/main.py` | 8 | 55 | R3 |
| `frontier/profiling/moe/moe_impl.py` | 36 | 14 | R3 |
| `frontier/profiling/moe/moe_vllm_kernel.py` | 416 | 42 | R3 |
| `frontier/profiling/moe/moe_wrapper.py` | 47 | 10 | R3 |
| `frontier/profiling/utils/__init__.py` | 33 | 4 | R3 |
| `frontier/profiling/utils/confirmation.py` | 2 | 2 | R3 |
| `frontier/profiling/utils/singleton.py` | 4 | 0 | R3 |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | 13 | 8 | R1 |
| `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | 28 | 0 | R1 |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 81 | 15 | R1 |
| `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py` | 3 | 1 | R1 |
| `frontier/scheduler/utils/decode_collective.py` | 3 | 4 | R1 |
| `frontier/scheduler/utils/dense_metrics.py` | 26 | 28 | R2 |
| `frontier/scheduler/utils/ep_wave.py` | 4 | 0 | R1 |
| `frontier/scheduler/utils/ep_wave_schedule.py` | 7 | 0 | R1 |
| `frontier/scheduler/utils/execution_time_metrics.py` | 30 | 56 | R2 |
| `frontier/scheduler/utils/expert_parallel.py` | 22 | 2 | R1 |
| `frontier/scheduler/utils/memory_planner.py` | 89 | 9 | R1 |
| `frontier/scheduler/utils/prefill_collective.py` | 10 | 3 | R1 |
| `frontier/scheduler/utils/sync_entry.py` | 6 | 0 | R1 |
| `frontier/simulator.py` | 31 | 2 | R1 |
| `frontier/training/__init__.py` | 2 | 1 | R4 |
| `frontier/training/attention_trainer.py` | 1 | 1 | R4 |
| `frontier/training/base_trainer.py` | 1 | 0 | R4 |
| `frontier/training/cli.py` | 61 | 0 | R4 |
| `frontier/training/gdn_trainer.py` | 301 | 0 | R4 |
| `frontier/types/device_sku_type.py` | 1 | 0 | R1 |
| `frontier/types/measurement_type.py` | 1 | 0 | R1 |
| `frontier/types/node_sku_type.py` | 1 | 0 | R1 |
| `frontier/utils/param_counter.py` | 121 | 10 | R1 |
