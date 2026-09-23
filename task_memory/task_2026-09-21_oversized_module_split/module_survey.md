# Oversized Module Split — Structural Surveys

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Collected four read-only structural surveys at `1f694f7`. |

These surveys were produced by read-only, grep-backed inspection of `origin/main` at `1f694f7c549aa3aeeb7c5bbae04e119c09167a77`. Line numbers refer to that revision. Every "dead" or "unreferenced" claim must be re-verified with a fresh grep immediately before the corresponding cleanup edit in Step 2; the surveys are planning input, not deletion authority.

---

## S1. `frontier/config/config.py` (5720 lines)

### Inventory

| Lines | Contents |
| --- | --- |
| 1–67 | Imports |
| 69–117 | Five `*_RELEASE_ERROR` strings, `DISAGGREGATED_CLUSTER_FIELD_PREFIXES`, `DISAGGREGATED_CLUSTER_FIELD_NAMES` |
| 120–138 | `_get_cc_backend_configs()` lazy-import shim (6-tuple) |
| 141–310 | Request interval/length generator configs (10 dataclasses) |
| 312–389 | Request generator configs (Base, Synthetic, Trace) |
| 390–464 | Small replica scheduler configs (Base, Vllm, Lightllm, Orca, FasterTransformer, Sarathi) |
| 465–869 | `VllmV1SchedulerConfig` (~400 lines; `__post_init__` 695) |
| 870–1091 | Sj2q family + Sglang (4 subclasses of VllmV1) |
| 1092–1251 | `MetricsConfig` |
| 1252–1861 | `SpeculativeDecodingConfig`, incl. six trace/JSON loader staticmethods 1371–1668 |
| 1862–2098 | `ReplicaConfig` (`__post_init__` 1963) |
| 2099–2138 | Cluster scheduler configs (6 small dataclasses) |
| 2139–2470 | Execution-time predictor configs (Base 2140–2426, LinearRegression 2428, RandomForrest 2452) |
| 2471–5066 | `ClusterConfig`: fields 2501–3850 (co-location 2501, disaggregated 2515, PD unified decode 2729, AF pipeline 2808, AFD CUDA graph 2845, per-cluster replica scheduler 2871, per-cluster CC backend 3071); methods 3851–5065 |
| 5067–5720 | `SimulationConfig(ABC)`: fields 5068–5299, `__post_init__` 5300, validators 5361–5645, accessors 5646–5702, `create_from_cli_args` 5703, `to_dict` 5710, `write_config_to_file` 5717 |

`ClusterConfig` methods: `__post_init__` 3851; validators 3925/3966/3973/3994; monolithic setup 4000, disaggregated setup 4030; 4197–4234 dummy-mode and field-set introspection; `_create_replica_config_from_fields` 4236; `_validate_replica_config` 4309; cluster info/stats/printing 4386/4433/4506; `get_cluster_configs_for_disaggregation` 4585; predictor-per-cluster 4714; six `_create_*_cc_backend_config` 4748–5029; `_create_replica_config_copy` 5030.

### Coupling

- `ClusterConfig` holds defaults from nearly every other family (`RoundRobinClusterSchedulerConfig`, `SarathiSchedulerConfig`, `BaseExecutionTimePredictorConfig`, `ReplicaConfig`, lazy CC backend).
- `get_cluster_configs_for_disaggregation` (4585–4713) builds per-`ClusterType` copies and calls predictor/CC-backend factories.
- `SimulationConfig.__post_init__` (5300–5360) sets `global_vars` (5309–5336), prints cluster statistics (5355), normalizes the metrics dir (5358), writes the config file (5359): side effects inside dataclass init.
- Polymorphism: `BasePolyConfig` + `get_type()` on 24 classes keyed by `frontier.types` enums. Field-prefix dispatch through `DISAGGREGATED_CLUSTER_FIELD_PREFIXES`.
- CLI coupling: `create_flat_dataclass(cls)` 5705; `to_dict` depends on `__flat_config__` 5711.

### Importers

35 `from frontier.config.config import ...` sites in 29 files. `frontier/config/__init__.py` ends with `from .config import *`, so `from frontier.config import X` is also a public path. Most-imported names: `ReplicaConfig` (~44), `MetricsConfig` (~26), `ClusterConfig` (~20), `SimulationConfig` (~19), `RandomForrestExecutionTimePredictorConfig` (~15), `VllmV1SchedulerConfig` (~14), `BaseReplicaSchedulerConfig` (~9), `DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR` (~7). Any split must preserve both import paths.

### Cleanup candidates

| Item | Lines | Evidence |
| --- | --- | --- |
| `BaseExecutionTimePredictorConfig.validate_linear_op_input` | 2397 | def only, zero call sites |
| `print_cluster_statistics` | 4506 | only caller 5355 |
| `write_config_to_file` | 5717 | only caller 5359 |
| Identical `hasattr(base_config, ...)` triplets | 4839–4846, 4951–4958, 4996–5003 | repeated 3x across CC-backend creators |
| Six near-parallel CC-backend creators | 4748–5029 | same shape, table-driven candidate |
| `hasattr` on declared dataclass fields | 4515, 5679, 5711 | defensive on guaranteed attributes |
| Two release-guard ladders from the same constants | 3994–3999, 5465–5481 | `DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR` never raised in-file |
| Spec-decode trace loaders | 1371–1668 | file IO inside a config dataclass; each used once |
| Method-local re-imports | 4892–4893, 5309 | `replace` already imported at line 4 |
| 55 `getattr`/`hasattr` uses | — | concentrated in the disaggregation/CC-backend region |

---

## S2. `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` (5138 lines)

Module level: env-driven JSONL decision logger set up at import (58–79), `_log_frontier_vllm_v1_schedule_decision` 81–86, `PrefixCacheAdmission` 87–98, `_serialize_prefix_cache_binding` 99–109. `VLLMv1EngineReplicaScheduler(BaseReplicaScheduler)` 110–5138.

### Inventory

| Group | Lines |
| --- | --- |
| Construction / flag+state init (`__init__`, ~28 `getattr(self._config, ...)`) | 129–320 |
| Batch formation / active-set bookkeeping | 321–332, 395–418, 4948–5015 |
| Decode-attn cohort / wave (PD-AF) | 419–619, 4602–4612, 4613–4947 |
| Spec-decode / MTP monolithic-PP wait heuristics (largest theme, ~1300 lines) | 333–394, 620–1351, 2365–2696 |
| Prefix caching | 1352–1368, 1446–1638 |
| Request lookup / resource release | 1369–1445 |
| CUDA-graph capture sizing | 1639–1734 |
| Spec-decode batch metadata | 1735–1889 |
| Policy / iteration profile / fast lanes | 1890–2135 |
| Chunked-prefill budget / decision-log hooks | 2136–2209 |
| `on_batch_end` | 2210–2364 |
| Token ledger / KV accounting | 2697–2832 |
| KV block allocation | 2833–3024 |
| Preemption | 3025–3273, 3489–3538 |
| Phase 1 running | 3276–3488 |
| Phase 2 waiting / admission | 3539–3855 |
| Scheduling entry points (`_get_next_batch`, `_schedule_two_phase`, prefill/decode-only) | 3856–4601 |
| Public overrides (`num_pending_requests`, `peek_waiting_requests`, `is_empty`, `add_request`) | 5017–5138 |

No DP-lane code exists in this file. GDN state-slot handling is threaded through init/free/allocate (30 hits).

### Existing helpers not reused

| Responsibility | Existing helper | Relationship |
| --- | --- | --- |
| Prefix caching | `scheduler/utils/prefix_cache.py` | not imported; local admission/ledger 1446–1638 |
| Idle diagnostics | `scheduler/utils/scheduler_diagnostics.py` | not imported; `is_empty` 5043–5079 near-copies base 689–710 |
| AFD metadata | `scheduler/utils/afd_metadata.py::aggregate_afd_metadata` | not imported; local `_attach_afd_metadata_if_needed` 4948–5015 |
| Batch building | `scheduler/utils/batch_builders.py` | not imported |
| MTP metrics | `scheduler/utils/mtp_metrics.py` | not imported; inline in `on_batch_end` |
| Spec decode | `frontier/spec_decode` | genuinely delegated |

### Contract

Overrides of `BaseReplicaScheduler` (9): `_create_batch`, `_get_request_next_num_tokens`, `complete_kv_transfer_for_requests`, `on_batch_end`, `_get_next_batch`, `num_pending_requests`, `peek_waiting_requests`, `is_empty`, `add_request`.

Subclasses (`SGLangStyleReplicaScheduler`, three `SJ2Q*` schedulers) override private members: `_schedule_two_phase`, `_schedule_running_requests`, `_get_sorted_waiting_queue`, `_build_decode_waiting_queue`, `_get_request_next_num_tokens`, `_emit_schedule_decision_event`, `_resolve_iteration_round_class`, `_get_iteration_scheduler_profile`, `_maybe_promote_final_round_priority`, `_is_final_{prefill,decode}_fast_lane_request`, `on_batch_end`, `add_request`. These names must stay on the class (or be re-bound) after any split.

External callers: `complete_kv_transfer_for_requests` (`events/kv_cache_transfer_end_event.py:62`), `consume_monolithic_pp_*_followup_poll` (`events/replica_schedule_event.py:101–122`, behind `hasattr`), `get_decode_attn_active_stage_slots` (`scheduler/utils/pdaf_attention.py:107`, via `getattr`). Seven unit-test files bind private methods via `object.__new__`.

### Cleanup candidates

| Finding | Lines | Evidence |
| --- | --- | --- |
| Dead `_attach_afd_metadata_if_needed` (68 lines) | 4948–5015 | zero references besides def; duplicates `afd_metadata.aggregate_afd_metadata` |
| Duplicated `is_empty` body vs base | 5043–5079 | copy of base 689–710 plus two terms |
| `hasattr` on state set in `__init__` | 5056–5060 | 3 `hasattr` total |
| 173 `getattr(` calls (42 `self`, 28 `self._config`, 51 `request`) | `__init__` 129–320, MTP block 620–1351 | defaults unreachable for own attributes |
| Import-time side effects (log dir + handler) | 58–79 | |
| Method-local re-imports | 4865–4866, 4966, 4977 | `global_vars` already imported at 28 |

---

## S3. `frontier/execution_time_predictor/shared_prediction_model_manager.py` (4614 lines)

### Inventory

| Responsibility | Symbols (lines) |
| --- | --- |
| MoE family name helpers | 107–135, 382 |
| Model-architecture profile checks | 136–198 |
| Typed operator contract validation / identity | 199–381; methods 912–1052, 1107–1123 |
| Exact-lookup (query time) | 389–404, 4462–4487 |
| Class ctor / cluster requirement analysis | `ExecutionTimePredictionModelManager` 405, `__init__` 421–469 (18 registry attrs 424–449), 470–543 |
| Measurement family / device-event timer | 544–649, 4502 |
| sklearn estimator / scoring | 650–698 |
| Training orchestration | `_train_all_required_models` 699–842, GDN 843–892 |
| TP/EP key resolution | 893, 1124–1225 |
| Training signature | `_get_ffn_contract_signature` 1053–1106 |
| Per-family trainers | FFN 1340–1806, dense MLP 1807–1907, attention 1908–2346, MLA 2347–2532, residual 2533–2606, PP 2607–2644, TP 2645–2690, CPU overhead 2691–2764, `_train_single_model` 2765–2987 |
| MoE dataset contract validation | 1226–1339 |
| CSV loading + column validation | 2988–3797 |
| Derived features | 3798–3937 |
| Cache-key / identity | `_get_hash_relevant_config` 3938–4007, `_get_model_hash` 4008–4057 |
| Precision / measurement from df | 4058–4104 |
| Estimator registry & sharing | 4105–4389 |
| Query-time lookup | `get_model` 4390–4445 |
| Persistent cache + locking | 4446–4461 |
| Public API | `get_models` 4488, `get_models_for_cluster` 4510–4562, `get_required_capabilities` 4563, `get_training_file_paths` 4567, `get_training_context` 4583–4614 |

### Consumers

`sklearn_execution_time_predictor.py`: `get_gdn_predictor` (437), `get_models` (572), `get_models_for_cluster` (574). `simulator.py`: constructor (157), `get_training_file_paths` (177). No other production callers; the MoE predictor holds only a type annotation. Zero external references anywhere: `get_required_capabilities`, `get_training_context`, `get_model`. About 15 unit tests reach private methods via `object.__new__`.

### Overlapping modules

`cache_io.py` (atomic dumps, dataset fingerprint), `measurement_input_paths.py` (manager 616–649 and 4567 are thin wrappers), `profiling_metadata.py`, `attention_dataset_contract.py`, `attention_tp_policy.py`, `execution_time_predictor_registry.py`, `frontier/operators/typed_contracts.py`, `frontier/model_architectures.py`, `frontier/moe_routing_runtime.py`, `frontier/moe_gating_runtime.py`, `frontier/training/*_trainer.py` (`base_trainer.py:285` notes its MAPE must match this file: duplicated logic).

### Routing-runtime identity today (input to correctness Step 5)

| Site | Fact |
| --- | --- |
| 1294–1296, 1440–1442 | `resolve_moe_gating_routing_runtime_path(getattr(replica_config, "moe_routing_distribution_type", "balanced"))` |
| 1488–1502 | `runtime_path_key` is part of the per-call `moe_df_cache` tuple only when `base_model_name == "moe_gating_routing_topk"`; not persisted |
| 1377–1382 | `ffn_signature` has no routing-runtime component |
| 1053–1104 | `_get_ffn_contract_signature` has no routing-runtime component |
| 4048–4051 | `_get_model_hash` has no routing-runtime term; it enters only indirectly through `df_hash_str` of the filtered frame |
| 4105–4155 | cached identity validation compares layer cache identity only |

### Cleanup candidates

| Finding | Evidence |
| --- | --- |
| Dead `_get_moe_df_with_derived_features` (3919–3937) | def only |
| Unreferenced public `get_required_capabilities` (4563), `get_training_context` (4583) | def only |
| Test-only `_validate_cached_layer_cache_identity` (4105) | 2 in-file, 2 in one test |
| Duplicated MAPE (683) vs `sklearn_execution_time_predictor.py`, `training/base_trainer.py`, `vidur_cc_backend.py` | 4 definitions |
| Duplicated derived-feature helpers (3905/3912) | 3 files each |
| Four near-identical registry accessors 4156/4168/4180/4190 | `return getattr(self, attr)` bodies |
| `_serialize_selected_layer_cache_identity` recomputed at 242, 336, 1098, 1496, 2836, 4035, 4136, 4375, 4400 | same contract, four phases |
| 57 `getattr` + 7 `hasattr`; repeated `getattr(replica_config, "model_config", None)` at 923, 1013, 1125, 1827, 1873, 4318, 4349; default-`"balanced"` duplicated at 1295 and 1441 | defensive ladders |

---

## S4. `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` (3539 lines)

Single class `SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor)` 254–3539 plus 8 module helpers 87–251.

### Inventory

| Responsibility | Members (lines) |
| --- | --- |
| Module helpers / operator-family naming | 87–251 |
| Dataset loading / filtering / contract | `_validate_moe_dataset_contract` 1186–1298, `_train_moe_models` 1299–1508 (`moe_df_cache` 1373–1430), 1509–1559 |
| Routing distribution / details generation | 261–278, 766–886, `_simulate_routing_per_layer` 2557–2630 |
| Expert-load / lane workload | 284–430, 887–1022, `_build_moe_load_imbalance_features` 1791–1844 |
| EP sync / shared-domain cost | 1023–1154, 1910–1999, 2953–3030 |
| Routing-cost model selection | 279–283, 1155–1185, 1560–1587 |
| Token-count resolution | 1618–1716, 2103–2160 |
| Gating / shuffling / grouped-GEMM compute | 1588–1617, 1717–1790, 1845–1909, 2000–2102 |
| Attention query caching | 2161–2209 |
| Layer/stage orchestration | 431–765, 2210–2556, 2631–2952, 3233–3539 |
| MTP replay | 3031–3232 |
| Diagnostics | 274, 758–763, 2809–2838, 2882–2925, 3332–3439 |

### Inheritance and delegation

Overrides (10): `__init__`, `_train_models`, `_predict_for_compute_models`, `_register_additional_profiling_metadata_from_files`, `predict_moe_layer_time`, `predict_allgather_time`, `predict_alltoall_time`, `predict_stage_execution_time`, `_predict_mtp_terminal_row_time_ms`, `_predict_mtp_decoder_layer_time_ms`. 43 new methods. Zero calls into `shared_prediction_model_manager` (annotation only). Subclassed by `sklearn_disaggregation_execution_time_predictor.py` (overrides `predict_stage_execution_time`; calls `_resolve_layer_lane_workload`, `_get_cluster_replica_config`).

### Routing identity today (input to correctness Step 5)

`_moe_routing_distribution_type` (715–724) is both the expert-load shape fed to `generate_moe_routing_ratios` (794–801) and the sole input selecting the profiling runtime path (731–734, 279–283). `_moe_gating_routing_runtime_path` (731–734) is assigned and never read. `_build_moe_load_imbalance_features` hardcodes `load_distribution="runtime"` (1817) and pops it (1820). Load distribution and routing implementation identity are conflated in one scalar.

### Cleanup candidates

| Finding | Evidence |
| --- | --- |
| `_is_grouped_gemm_on_demand_mode` (2000–2014) | zero references |
| `_simulate_routing_per_layer` (2557–2630) | test-only; self-described legacy |
| `predict_monolithic_decode_shared_domain_lane_moe_times_ms` (1060–1154) | test-only public API |
| `_moe_gating_routing_runtime_path` field | written, never read |
| 29 `getattr` + 5 `hasattr` on own attributes set in `__init__` (e.g. 1245, 1384, 1693, 2565; 328, 1580, 3157; 1688–1689) | unreachable defaults |
| Duplicated MoE-layer classification | 431–475 vs 3127–3136 and 3191–3202 |
| Duplicated share-expert triples | 1103–1107 and 2330–2333 |
| Per-layer re-queries of gating/shuffling/grouped-GEMM/EP-comm without memoization | 3243–3255; only attention is cached 2161–2194 |
| INFO-level diagnostic blocks on every prediction | 2882–2925, 3405–3439 |
