# Oversized Module Split — Plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Initial plan: scope, proposed split boundaries, sequencing, fidelity matrix design, acceptance criteria. Boundaries are proposals derived from `module_survey.md`; each is confirmed against the code before its step starts. |
| 2026-09-21 | Recorded the boundaries actually implemented for `config.py` in section 3.1, which differ from the proposal. |

## 1. Scope and result

Branch `refactor/oversized-module-split` (base `1f694f7c549aa3aeeb7c5bbae04e119c09167a77`) brings the four modules below under the `AGENTS.md` 2,000-line gate through a cleanup-first pass and a functional split, with **no behavior or numeric change**. The Issue 26 correctness branch is based on this branch, so every boundary chosen here must leave the correctness fixes with a clear owner (see §3).

| Module | Lines | Target after split |
| --- | --- | --- |
| `frontier/config/config.py` | 5720 | ≤ 2,000 per child module |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 5138 | ≤ 2,000 per child module |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | 4614 | ≤ 2,000 per child module |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 3539 | ≤ 2,000 per child module |

Out of scope: the other five modules above 2,000 lines (`sklearn_execution_time_predictor.py` 8263, `metrics_store.py` 5586, `sklearn_disaggregation_execution_time_predictor.py` 2985, `profiling/attention/main.py` 2157, `entities/request.py` 2125), any behavior fix, any change to public CLI flags, config field names, metrics schema, or profiling CSV contracts.

## 2. Rules for every edit

1. **Cleanup before split.** Remove only code whose lack of references is re-verified by grep at edit time. Replace `getattr(self, "_x", default)` on attributes assigned in `__init__` with direct access only when the assignment is unconditional; otherwise leave it and record the reason.
2. **Move, do not rewrite.** A split moves functions and classes verbatim (imports adjusted). Renames are limited to module paths; class and public method names stay. Private methods overridden by subclasses or bound by tests (listed in `module_survey.md`) stay on the class or are re-exposed under the same name.
3. **Preserve import surfaces.** `frontier/config/__init__.py` star re-export and `from frontier.config.config import X` keep working through re-exports in the original module. Same for the three other modules.
4. **Fidelity gate after each step.** The matrix in §5 must be value-identical to the main baseline after every step, not only at the end. A step that changes any number is reverted or explained as an approved fidelity fix.
5. **Public draft PR after Step 1.** Each later step is committed and pushed as a coherent unit with its matrix result recorded in `progress.md` and `validation.md`.

## 3. Proposed boundaries (to confirm at each step)

### 3.1 `frontier/config/` (implemented)

The proposal below was revised during implementation. Two things forced the change. First, `ClusterConfig` alone is 2,588 lines, so leaving it in `config.py` would have kept that file above the gate no matter which leaf families moved out. Second, `flat_dataclass` resolves string annotations in the *defining module's* namespace, so every module must carry the imports its own annotations need; that is a correctness constraint on the split, not a style choice.

What was implemented:

| Module | Lines | Content |
| --- | --- | --- |
| `config.py` | 788 | `SimulationConfig`, the lazy CC-backend import note, and the re-export block that keeps `from frontier.config[.config] import X` working for all 36 names other modules import |
| `cluster_config.py` | 1888 | `ClusterConfig`: the flat per-role field surface, `__post_init__`, the validators, monolithic and disaggregated setup, `_validate_replica_config` |
| `cluster_role_config.py` | 586 | `_get_cc_backend_configs` and `ClusterRoleConfigBuilder`: the 13 methods that build the per-role replica, predictor and CC-backend configurations |
| `cluster_topology_summary.py` | 217 | `ClusterTopologySummary`: `_collect_cluster_info`, `get_server_count_metadata`, `print_cluster_statistics` |
| `release_guards.py` | 56 | The six release-guard messages and the two disaggregated field-name tables |
| `request_generator_config.py` | 262 | Arrival interval, request length and request generator families |
| `replica_scheduler_config.py` | 711 | Every replica scheduler configuration, including the vLLM V1 family and its Sj2q and SGLang subclasses |
| `metrics_config.py` | 173 | `MetricsConfig` |
| `speculative_decoding_config.py` | 622 | `SpeculativeDecodingConfig` and its trace loaders |
| `replica_config.py` | 250 | `ReplicaConfig` |
| `cluster_scheduler_config.py` | 48 | The cluster scheduler configuration family |
| `execution_time_predictor_config.py` | 314 | The predictor configuration family and its calibration scales |

The two groups extracted from `ClusterConfig` are **mixins that `ClusterConfig` inherits**, not free functions taking the config. That keeps the split a pure move: the method bodies, the method names and every call site are unchanged, including the four methods that external code and tests call directly (`get_cluster_configs_for_disaggregation`, `get_server_count_metadata`, `_validate_replica_config`, `_create_replica_config_copy`). Rewriting them as free functions would have touched every line of 731 moved lines and made the diff unreviewable, with the fidelity matrix as the only remaining check.

Naming note for review: `ClusterRoleConfigBuilder` and `ClusterTopologySummary` describe what each group produces. If a reviewer prefers different names, renaming them is mechanical and affects only three files.

### 3.1a Original proposal (superseded)

| Child module | Content (survey lines) | Approx. lines |
| --- | --- | --- |
| `config.py` (kept) | `SimulationConfig`, `ClusterConfig` core fields and validators, re-exports of everything below | ~1,900 after moves |
| `request_generator_config.py` | interval/length/request generator dataclasses (141–389) | ~250 |
| `replica_scheduler_config.py` | small scheduler configs + `VllmV1SchedulerConfig` + Sj2q/Sglang (390–1091) | ~700 |
| `speculative_decoding_config.py` | `SpeculativeDecodingConfig` (1252–1861) | ~610 |
| `replica_config.py` | `ReplicaConfig` (1862–2098) + cluster scheduler configs (2099–2138) | ~280 |
| `execution_time_predictor_config.py` | predictor configs (2139–2470) | ~330 |
| `metrics_config.py` | `MetricsConfig` (1092–1251) | ~160 |
| `cluster_config_factories.py` | `get_cluster_configs_for_disaggregation`, `_create_replica_config_from_fields`, `_create_replica_config_copy`, the six `_create_*_cc_backend_config` (4236–4308, 4585–4713, 4748–5066) as functions taking the `ClusterConfig` | ~900 |

Cleanup first: the three duplicated `hasattr(base_config, ...)` triplets and the six parallel CC-backend creators become one table-driven helper; `validate_linear_op_input` (dead), method-local re-imports, and `hasattr` on declared dataclass fields are removed. Existing `frontier/config/cluster_scheduler_config.py` in the donor branch is **not** copied; the boundary above is chosen from main's structure.

Correctness-branch owner after split: the opt-in DP placement config (Step 4) lands in `replica_config.py` next to the cluster scheduler configs; routing runtime override (Step 5) lands in `replica_config.py`.

### 3.2 `frontier/scheduler/replica_scheduler/`

| Child module | Content (survey lines) | Approx. lines |
| --- | --- | --- |
| `vllm_v1_engine_replica_scheduler.py` (kept) | class shell, `__init__`, scheduling entry points, phase 1/2, admission, `on_batch_end`, public overrides; delegates to the helpers below | ~1,900 |
| `vllm_v1_kv_allocation.py` | token ledger / KV accounting, block allocation, preemption (2697–3273, 3489–3538) | ~700 |
| `vllm_v1_mtp_wait.py` | target-embedded MTP monolithic-PP wait heuristics and terminal release (333–394, 620–1351, 2365–2696) | ~1,300 |
| `vllm_v1_prefix_cache.py` | prefix-cache admission/ledger and identity events (87–109, 1352–1368, 1446–1638) | ~250 |
| `vllm_v1_decode_attn_cohort.py` | PD-AF decode-attn cohort/wave logic (419–619, 4602–4947) | ~550 |
| `vllm_v1_iteration_policy.py` | policy / iteration profile / fast lanes / CUDA-graph capture sizing / spec-decode batch metadata (1639–2135) | ~500 |

Mechanism: helpers become plain functions or small stateless classes that take the scheduler instance; the class keeps thin methods with the original names so subclass overrides and `object.__new__` tests keep binding. Cleanup first: dead `_attach_afd_metadata_if_needed`, duplicated `is_empty` body, method-local re-imports, `getattr(self, ...)` on attributes assigned unconditionally in `__init__`. The import-time log handler (58–79) is kept but moved behind a function called from `__init__` only if the current behavior (env-var gated) is preserved exactly; otherwise it stays.

Correctness-branch owner after split: request-load accounting for Step 4 (`waiting` / `running` counts) is added to the kept class as one accessor; no change to the helper modules.

### 3.3 `frontier/execution_time_predictor/`

| Child module | Content (survey lines) | Approx. lines |
| --- | --- | --- |
| `shared_prediction_model_manager.py` (kept) | `ExecutionTimePredictionModelManager` ctor, cluster requirement analysis, training orchestration, public API, query-time `get_models*` | ~900 |
| `prediction_model_registry.py` | estimator registry & sharing, precision buckets, cache identity validation, `get_model` (4105–4445) | ~350 |
| `prediction_model_cache.py` | `_get_hash_relevant_config`, `_get_model_hash`, persistent load/store with locking (3938–4057, 4446–4487), reusing `cache_io.py` | ~200 |
| `profiling_dataframe_loaders.py` | CSV loaders, column validation, derived features (2988–3937) | ~950 |
| `family_trainers.py` | per-family trainers and `_train_single_model` (1340–2987) | ~1,650 |
| `layer_contract_resolution.py` | typed operator contract helpers, TP/EP key resolution, FFN signature, MoE dataset contract (199–381, 893–1339) | ~600 |

Cleanup first: dead `_get_moe_df_with_derived_features`, unreferenced `get_required_capabilities` / `get_training_context`, the four registry accessor bodies, duplicated MAPE (keep one definition and import it where `base_trainer.py` says it must match), duplicated derived-feature helpers, repeated `getattr(replica_config, "model_config", None)` ladders.

Correctness-branch owner after split: routing-runtime identity (Step 5) enters `layer_contract_resolution.py` (signature) and `prediction_model_cache.py` (hash) and `prediction_model_registry.py` (lookup) at one clearly named point each.

### 3.4 `sklearn_moe_execution_time_predictor.py`

| Child module | Content (survey lines) | Approx. lines |
| --- | --- | --- |
| `sklearn_moe_execution_time_predictor.py` (kept) | class shell, `__init__`, layer/stage orchestration, MTP replay, overrides (431–765, 2210–2556, 2631–2952, 3031–3539) | ~1,700 |
| `moe_routing_workload.py` | routing distribution/details generation, expert-load and lane workload (261–430, 766–1022, 1791–1844, 2557–2630) | ~600 |
| `moe_operator_times.py` | gating / shuffling / grouped-GEMM / EP-communication / token-count helpers and module-level operator-family helpers (87–251, 1155–1185, 1560–1790, 1845–2160) | ~800 |
| `moe_dataset_contract.py` | `_validate_moe_dataset_contract`, `_train_moe_models` dataset filtering (1186–1559) | ~380 |

Cleanup first: `_is_grouped_gemm_on_demand_mode` (dead), the never-read `_moe_gating_routing_runtime_path` field (**kept until Step 5 of the correctness PR decides its owner**; recorded, not removed here), `getattr(self, ...)` on own attributes, duplicated MoE-layer classification and share-expert triples. The INFO-level diagnostic blocks are left untouched in this PR because log output is not part of the fidelity gate and changing them is not a size problem.

Correctness-branch owner after split: separating load distribution from routing implementation identity (Step 5) is a change inside `moe_routing_workload.py` (load shape) and `moe_dataset_contract.py` (runtime row selection).

## 4. Sequencing and dependencies

```text
Step 0 (worktree, records, env, baseline)
  -> Step 1 (fidelity harness + main baseline capture)
  -> Step 2 (cleanup pass, all four modules; matrix)
  -> {Step 3 config split, Step 4 scheduler split}   # independent, may run in parallel worktrees
  -> {Step 5 manager split, Step 6 MoE predictor split}   # 6 depends on 5 only for shared helper placement
  -> Step 7 (full matrix + unit suites + PR hand-off)
```

Steps 3 and 4 touch disjoint packages and can be parallelized; Steps 5 and 6 share `frontier/execution_time_predictor/` and are sequenced. Each step ends with matrix PASS, commit, push.

## 5. Fidelity matrix design (Step 1 deliverable)

- **Harness location:** `tests/e2e/refactor_fidelity/` with a small driver that (a) generates a deterministic case manifest, (b) runs each case through the checked-in example wrappers or `python -m frontier.main` in two worktrees (baseline main at `1f694f7` and the refactor branch) with the same environment, (c) compares outputs. It reuses `tests/scratch_root.py`, the example wrappers, and case-generation ideas from `tests/e2e/moe_ep_non_dummy_matrix.py`, but does not depend on that 7,900-line harness or its pinned baseline commit.
- **Comparison rule:** `request_metrics.csv` compared value-by-value (exact equality; both runs are deterministic CPU simulations); `system_metrics.json` compared after removing timestamps, run ids, wall-clock durations, host/Python metadata, and output paths. The removal list is explicit in the harness and recorded in `validation.md`.
- **Coverage (≥ 50 cases):**

| Dimension | Values |
| --- | --- |
| Architecture | co-location, sequential PDD, sequential PD-AF |
| Model | dense (`llama2_7b_dense_example`), MoE (`Qwen3-30B-A3B-tiny`, `Phi-tiny-MoE-instruct`), PD-AF EP=2 topology |
| Predictor | dummy mode; checked-in `data/profiling/compute/h800/*` CSVs for dense and MoE |
| Mode | offline, online (Poisson arrivals) |
| Requests | 4 / 16 / 64 requests; prompt lengths 128 / 1024 / 4096; decode 16 / 128 |
| QPS (online) | 0.5, 2, 8 |
| Features | Chunked Prefill on/off, `decode_cuda_graph_mode` none/full_decode_only, prefix caching (fixture trace), speculative decoding (MoE recipe), Thinking Mode |
| Parallelism | TP1/TP2, PP1/PP2, `attn_dp` 1/2, EP 1/2 |

The manifest fixes the exact combinations (not a full cross product) so the total is 50–70 cases with predictable wall time. Cases that main itself fails are recorded as baseline failures and excluded from the pass count with their exception.

- **Baseline capture:** the main-side outputs are produced once from a read-only checkout of `1f694f7` (a separate worktree `.worktrees/fidelity-baseline-main`, never edited) and stored under `FRONTIER_TMP_ROOT/refactor-fidelity/baseline/`; their manifest hash is recorded in `validation.md`.

## 6. Acceptance criteria

- Each of the four modules and every new child module is ≤ 2,000 lines; the count is recorded in `validation.md` per step.
- Fidelity matrix: all non-baseline-failing cases identical after every step and at the end.
- Existing unit suites that touch the four modules pass or fail exactly as on main (baseline-failure list recorded at Step 0).
- No public import path breaks: a test imports every name listed in `module_survey.md` §S1 "Importers" from both `frontier.config` and `frontier.config.config`, and the corresponding names for the other three modules.
- Draft PR opened against `main` with the matrix summary, module line counts, and links to these records; no merge without user approval.

## 7. Validation commands (recorded per step in `validation.md`)

```bash
export PYTHONPATH=/data/ycfeng/Frontier/.worktrees/oversized-module-split
export FRONTIER_TMP_ROOT=/data/ycfeng/tmp/issue26-correctness-pr
PY=/data/ycfeng/envs/frontier-py310/bin/python
$PY -m pytest <selection recorded per step> -q -p no:cacheprovider
$PY tests/e2e/refactor_fidelity/run_matrix.py --baseline-worktree .worktrees/fidelity-baseline-main --candidate-worktree . --manifest tests/e2e/refactor_fidelity/manifest.json
```
