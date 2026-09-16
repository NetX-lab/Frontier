## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Compared clean-main and both candidate full CPU unit runs by exact nodes, observed causes, source, collection, and skips; documented four local predictor fixture repairs and final closure. |

# W10 full CPU unit comparison

Reviewer: `/root/w09_artifact_review`. Full-suite runs were executed by `/root`; this reviewer inspected their logs, independently collected exact nodes in both worktrees, compared failure source and dependencies, replayed the six failed CLI commands, and executed the two assigned predictor test files.

**Final result: 18 failed, 3584 passed, 25 skipped, 576 warnings in 96.78 s; 3627 collected.** All 18 failures match clean-main node IDs and causes. The first candidate's 27 additional failures are resolved in the final run; there are no candidate-only failures or new skipped nodes. This is a baseline-relative CPU unit gate, not an entirely green suite, native GPU evidence, or completion of the other W10 E2E/performance lanes.

## Execution and provenance

Clean-main worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915`, revision `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`, no source edits. Candidate worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`, HEAD `5a1cc2af280e5ae2d51caae55fe016c80bc3e2d2` plus local remediation. Final source snapshot: `/data/ycfeng/tmp/pr33-w10-final-source-2.diff`; the commit alone does not identify that candidate.

Verified shared runtime: `/usr/bin/python`, Python 3.12.3, no conda activation; NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9, pytest 9.1.1. The baseline's original MKL thread environment was not independently recoverable. Its traceback uses `/data/ycfeng/tmp/pr33-main-unit-baseline` for pytest temporary files, whereas candidate uses the default pytest temporary directory under `/data/ycfeng/tmp`; the baseline therefore had an explicit or equivalent temporary-root setting that is not reconstructed here. These are stated differences, not a claim of byte-identical invocations. Neither difference explains the observed missing assets, unchanged README assertions, or FlashInfer dependency failures.

Exact candidate full-suite command, run from candidate cwd; run 1 used `pr33-w10-final-unit.log`, run 2 used `pr33-w10-final-unit-2.log`:

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/unit -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w10-final-unit-2.log 2>&1
```

The baseline full-suite log is `/data/ycfeng/tmp/pr33-main-unit-baseline.log`. To repeat under the explicitly specified candidate environment, use the command above from the baseline cwd with a fresh baseline log path; that is a reproduction recipe, not an assertion that the historical baseline command is fully known.

Independent collection command (executed from each corresponding cwd):

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python -m pytest tests/unit --collect-only -q -p no:cacheprovider
```

Collection logs: `/data/ycfeng/tmp/pr33-w10-main-collection.log`, `/data/ycfeng/tmp/pr33-w10-candidate-collection.log`, `/data/ycfeng/tmp/pr33-w10-candidate-collection-2.log`. Complete node sets and raw failure excerpts are stored in `/data/ycfeng/tmp/pr33-w10-unit-comparison.json`.

## Criteria and observed results

PASS for this baseline-relative gate requires every retained failure to match exact node, cause, relevant source, and environment; no unexplained candidate failure; no unaccounted collection deletion or new skip. Count equality alone is insufficient.

| Snapshot | Collected | Passed | Failed | Skipped | Warnings | Duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean main | 3188 | 3144 | 19 | 25 | 576 | 76.86 s |
| Candidate first run | 3621 | 3551 | 45 | 25 | 576 | 99.82 s |
| Candidate final run | 3627 | 3584 | 18 | 25 | 576 | 96.78 s |

The first run has 18 common failures, 27 candidate-only failures, and one baseline-only failure. The final run retains exactly those 18 common failures. Durations are suite wall times for different test collections; they are not simulator performance ratios.

## Exact retained failures and source match

All 18 failing test-function ASTs are equal between baseline and inspected candidate. `README.md` and the three analysis CLI entry scripts are byte-identical. The optimizer package and the three referenced debug shell scripts are absent from both trees. Thus the comparison establishes the relevant source relationship, rather than assuming all baseline failures are environment-only.

| Exact node | Observed cause in both trees |
| --- | --- |
| `tests/unit/test_colocation_release_review_contracts.py::test_config_optimizer_help_prints_without_argparse_percent_crash` | Missing `frontier.config_optimizer` package; identical ModuleNotFoundError. |
| `tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_does_not_default_to_private_conda_path` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_resolves_latest_canonical_metrics_run_dir` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_uses_safe_pythonpath_expansion` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_conda_activation_temporarily_disables_nounset` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_colocation_release_review_contracts.py::test_readme_debug_scripts_do_not_use_post_increment_under_set_e` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_header_uses_current_tp_terms` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_satisfies_shared_parallel_domain` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_uses_valid_shared_parallel_domain` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_colocation_release_review_contracts.py::test_release_debug_scripts_use_canonical_metrics_resolver` | Missing `tests/debug/e2e-level/monolith_mode/scripts/` asset; identical missing path/command failure. |
| `tests/unit/test_examples_documentation_contracts.py::test_public_docs_list_the_complete_pdaf_v03_surface_and_boundaries` | Identical assertion against unchanged `README.md` content. |
| `tests/unit/test_examples_documentation_contracts.py::test_readme_documents_current_examples_surface` | Identical assertion against unchanged `README.md` content. |
| `tests/unit/test_mha_stage2_profile_modeling.py::test_stage2_cli_writes_reproducible_artifacts` | CLI dependency validation: `flashinfer-python` unavailable; direct replay agrees in both worktrees. |
| `tests/unit/test_mla_stage3_online_trace_builder.py::test_mla_stage3_cli_writes_trace_and_error_matrix` | CLI dependency validation: `flashinfer-python` unavailable; direct replay agrees in both worktrees. |
| `tests/unit/test_mqa_stage2_profile_modeling.py::test_mqa_stage2_cli_writes_reproducible_artifacts` | CLI dependency validation: `flashinfer-python` unavailable; direct replay agrees in both worktrees. |
| `tests/unit/test_pdd_public_surface_docs.py::test_public_docs_require_sequential_pdd_release_mode` | Identical assertion against unchanged `README.md` content. |
| `tests/unit/test_pdd_public_surface_docs.py::test_public_docs_state_internal_parallel_pdd_boundary_and_current_evidence` | Identical assertion against unchanged `README.md` content. |
| `tests/unit/test_pdd_public_surface_docs.py::test_top_level_docs_advertise_supported_pdd_without_upcoming_claims` | Identical assertion against unchanged `README.md` content. |

The three CLI tests capture subprocess stderr and initially expose only `CalledProcessError`. This reviewer extracted their exact argument arrays from both logs and replayed all six commands with `PYTHONPATH=.` in the corresponding cwd. MHA and MQA each returned 1 with `flashinfer-python is not installed in the active Python env`; MLA returned 1 with the same cause plus `expected 0.3.1.post1`. Exact commands, cwd, exit status and stderr are preserved in `/data/ycfeng/tmp/pr33-w10-cli-rca.json` and the comparison JSON. No dependency was installed and no source was altered for this reproduction.

The baseline-only failure is `tests/unit/test_examples_profiling_contracts.py::test_profiling_readme_documents_migration_scope_and_legacy_path`; it passes in both candidate runs after the in-scope profiling documentation update. Unrelated README/debug assets were not repaired to change the totals.

## First-run additional failures and closure

Each row below failed in the first candidate and passes in the final full-suite run. These rows retain the original failures and their actionable causes; the final green result is not substituted for the original record. Other lanes own their production/fixture corrections, which are preserved in the frozen source diff.

| Exact first-run node | Root cause / correction boundary |
| --- | --- |
| `tests/unit/test_attention_predictor_correctness.py::test_attention_trace_total_includes_decode[0-4-0.0]` | Fake model lacks normal `get_layer_attention_spec`; migrated to normal typed constructor. |
| `tests/unit/test_attention_predictor_correctness.py::test_attention_trace_total_includes_decode[5-4-2.0]` | Fake model lacks normal `get_layer_attention_spec`; migrated to normal typed constructor. |
| `tests/unit/test_attention_predictor_correctness.py::test_dsa_frozen_fails_fast_in_dummy_mode` | Fake model lacks normal `get_layer_attention_spec`; migrated to normal typed constructor. |
| `tests/unit/test_colocation_release_review_contracts.py::test_non_dummy_shared_model_manager_registers_profiling_metadata` | Manager fixture lacks required `get_gdn_predictor` owner API. |
| `tests/unit/test_comm_operator_families.py::test_moe_stage_num_layers_view_preserves_comm_operator_times` | Per-layer PP expectation conflicts with stage-owned PP; independent stage total oracle retained. |
| `tests/unit/test_metrics_stage_execution_time.py::test_stage_op_level_traces_include_gdn_metadata_without_duplication` | Fixture expects expanded layer ID while default trace scope is aggregated. |
| `tests/unit/test_mixed_layer_decode_ffn_scheduling.py::test_trained_decode_ffn_predictor_classifies_each_layer[3-False]` | Old communication spy rejects the explicit `include_stage_owned` parameter. |
| `tests/unit/test_mixed_layer_decode_ffn_scheduling.py::test_trained_decode_ffn_predictor_classifies_each_layer[4-True]` | Old communication spy rejects the explicit `include_stage_owned` parameter. |
| `tests/unit/test_mixed_layer_decode_ffn_scheduling.py::test_trained_decode_ffn_predictor_classifies_each_layer[59-True]` | Old communication spy rejects the explicit `include_stage_owned` parameter. |
| `tests/unit/test_mixed_layer_decode_ffn_scheduling.py::test_trained_decode_ffn_predictor_classifies_each_layer[60-False]` | Old communication spy rejects the explicit `include_stage_owned` parameter. |
| `tests/unit/test_mixed_layer_decode_ffn_scheduling.py::test_trained_dense_decode_ffn_constructs_execution_time_with_one_is_moe_source` | Old communication spy rejects the explicit `include_stage_owned` parameter. |
| `tests/unit/test_mixed_layer_decode_ffn_scheduling.py::test_trained_dense_decode_ffn_excludes_post_attention_layernorm` | Old communication spy rejects the explicit `include_stage_owned` parameter. |
| `tests/unit/test_mixed_layer_decode_ffn_scheduling.py::test_trained_moe_decode_ffn_excludes_post_attention_layernorm` | Old communication spy rejects the explicit `include_stage_owned` parameter. |
| `tests/unit/test_mla_core_native_op_tracing.py::test_op_level_tracing_generates_metadata_for_structured_mla_ops` | Full-suite quantization singleton leaks BF16 into FP16 fixture; focused file had passed. |
| `tests/unit/test_pd_decode_moe_layer_accounting.py::test_monolithic_shared_domain_terminal_collectives_account_all_layers[1]` | Fake scheduler lacks `_create_prefill_corrected_execution_time_for_metrics`. |
| `tests/unit/test_pd_decode_moe_layer_accounting.py::test_monolithic_shared_domain_terminal_collectives_account_all_layers[2]` | Fake scheduler lacks `_create_prefill_corrected_execution_time_for_metrics`. |
| `tests/unit/test_pd_decode_moe_layer_accounting.py::test_pp2_terminal_collectives_account_all_94_layers` | Fake scheduler lacks `_create_prefill_corrected_execution_time_for_metrics`. |
| `tests/unit/test_pd_decode_moe_layer_accounting.py::test_replayed_terminal_collective_fails_fast` | Fake scheduler lacks `_create_prefill_corrected_execution_time_for_metrics`. |
| `tests/unit/test_pd_decode_moe_layer_accounting.py::test_terminal_collective_increments_duplicate_active_request_once` | Fake scheduler lacks `_create_prefill_corrected_execution_time_for_metrics`. |
| `tests/unit/test_pd_decode_moe_layer_accounting.py::test_terminal_collective_skips_completed_request_in_mixed_batch` | Fake scheduler lacks `_create_prefill_corrected_execution_time_for_metrics`. |
| `tests/unit/test_pdaf_prefill_model_time.py::test_prefill_final_sync_uses_component_ledger_without_timestamp_residue` | Fixture asserts legacy one-layer wrapper while typed stage owns 32/2 layers. |
| `tests/unit/test_pdaf_prefill_model_time.py::test_prefill_sync_records_heterogeneous_layer_components_once` | Fixture asserts legacy one-layer wrapper while typed stage owns 32/2 layers. |
| `tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py::test_sklearn_runtime_dense_attention_support_gates_follow_role_names` | Fake model/predictor bypasses layer-spec contract; normal typed construction fixes failure with original numeric assertions. |
| `tests/unit/test_spec_decode_predictor_verify_prefill_path.py::test_predict_attention_layer_time_aggregates_verify_and_decode_with_sum_rule` | Fake model/predictor bypasses layer-spec contract; normal typed construction fixes failure with original numeric assertions. |
| `tests/unit/test_spec_decode_predictor_verify_prefill_path.py::test_predict_attention_layer_time_excludes_method_aware_proposer_overhead` | Fake model/predictor bypasses layer-spec contract; normal typed construction fixes failure with original numeric assertions. |
| `tests/unit/test_spec_decode_predictor_verify_prefill_path.py::test_predict_attention_layer_time_uses_hybrid_family_for_spec_piecewise_diagnostic` | Fake model/predictor bypasses layer-spec contract; normal typed construction fixes failure with original numeric assertions. |
| `tests/unit/test_transfer_metrics_contract.py::test_kv_ledger_binds_only_decode_entry_stage` | Ledger metrics fixture lacks `store_operation_metrics` configuration. |

This reviewer's assigned edits are confined to `tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py` and `tests/unit/test_spec_decode_predictor_verify_prefill_path.py`. `frontier/attention/model_binding.py` resolves the required `get_layer_attention_spec(global_layer_id)` API; the old `SimpleNamespace` model lacked it, while normal `BaseModelConfig` construction provides validated layer schedules. The revised fixtures reuse `cache_model` and `predictor_fixture_config`, call the real predictor constructor, and retain the original attention values 7/11/13 ms, speculative sum 23 ms, and hybrid diagnostic sum 39 ms. No production fallback was added. Both complete files pass: **25 passed in 2.97 s**.

Exact local verification:

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py tests/unit/test_spec_decode_predictor_verify_prefill_path.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w10-predictor-fixture-repair.log 2>&1
```

## Collection accounting

The first comparison has **3184 common nodes, 437 added nodes, 4 removed node names**: net +433. The final comparison has **3184 common nodes, 443 added nodes, 4 removed node names**: net +439. Every node is retained in the JSON. The four old names map to explicit contract migrations:

| Baseline node name | Candidate replacement | Reason |
| --- | --- | --- |
| `test_monolithic_multi_layer_default_aggregate_preserves_legacy_path` | `test_monolithic_multi_layer_default_aggregate_requires_typed_lane` | An implicit routed EP aggregate must supply a typed physical lane; the replacement asserts rejection before dummy timing. |
| `test_observed_mapping_includes_architecture_linear_attention_ops` | `test_observed_mapping_includes_architecture_attention_linear_ops_ops` | Architecture field renamed to `attention_linear_ops`; the same sharded/replicated classification assertions remain. |
| `test_dummy_layer_scaling_preserves_named_tp_components[prefill]` | `test_dummy_one_layer_scope_preserves_named_tp_components[prefill]` | Stage now explicitly owns layers; preserved named-TP numeric oracle compares the requested layer with the stage's first layer. |
| `test_dummy_layer_scaling_preserves_named_tp_components[decode]` | `test_dummy_one_layer_scope_preserves_named_tp_components[decode]` | Same typed stage ownership migration for decode. |

The final run adds six nodes after the first run:
- `tests/unit/test_attention_query_cache.py::test_direct_gdn_prediction_requires_loaded_artifact_before_dense_lookup[False]`
- `tests/unit/test_attention_query_cache.py::test_direct_gdn_prediction_requires_loaded_artifact_before_dense_lookup[True]`
- `tests/unit/test_dense_execution_time_layer_scaling.py::test_disaggregated_dense_stage_reuses_one_snapshot_with_exact_oracle[decode]`
- `tests/unit/test_dense_execution_time_layer_scaling.py::test_disaggregated_dense_stage_reuses_one_snapshot_with_exact_oracle[decode_attn]`
- `tests/unit/test_dense_execution_time_layer_scaling.py::test_disaggregated_dense_stage_reuses_one_snapshot_with_exact_oracle[decode_ffn]`
- `tests/unit/test_dense_execution_time_layer_scaling.py::test_disaggregated_dense_stage_reuses_one_snapshot_with_exact_oracle[prefill]`

Final added nodes by file, including the four replacement names above:

| File | Added nodes |
| --- | ---: |
| `tests/unit/test_attention_family_binding.py` | 4 |
| `tests/unit/test_attention_query_cache.py` | 12 |
| `tests/unit/test_collectives_increment11.py` | 6 |
| `tests/unit/test_collectives_rocm_runner.py` | 3 |
| `tests/unit/test_dense_execution_time_layer_scaling.py` | 5 |
| `tests/unit/test_device_timer_contract.py` | 6 |
| `tests/unit/test_execution_time_metrics_ownership.py` | 8 |
| `tests/unit/test_gdn_artifact_boundary.py` | 37 |
| `tests/unit/test_gdn_campaign_preflight.py` | 27 |
| `tests/unit/test_gdn_hybrid_e2e_increment14ab.py` | 6 |
| `tests/unit/test_gdn_increment6_memory.py` | 5 |
| `tests/unit/test_gdn_profile_samples.py` | 45 |
| `tests/unit/test_gdn_profiler_cpu_increment7.py` | 9 |
| `tests/unit/test_gdn_runtime_guards.py` | 11 |
| `tests/unit/test_gdn_scheduler_slots.py` | 13 |
| `tests/unit/test_gdn_semantic_core.py` | 12 |
| `tests/unit/test_gdn_state_lifecycle.py` | 2 |
| `tests/unit/test_gdn_training_predictor_increment8.py` | 8 |
| `tests/unit/test_hybrid_runtime_family.py` | 6 |
| `tests/unit/test_measurement_family_selector.py` | 5 |
| `tests/unit/test_measurement_path_precedence.py` | 23 |
| `tests/unit/test_metrics_stage_execution_time.py` | 4 |
| `tests/unit/test_model_architecture_registry.py` | 2 |
| `tests/unit/test_moe_ep_aggregate_admission.py` | 1 |
| `tests/unit/test_moe_fused_event_contract.py` | 18 |
| `tests/unit/test_moe_gating_constructor_boundary.py` | 4 |
| `tests/unit/test_moe_mxfp4_increment10.py` | 5 |
| `tests/unit/test_moe_native_admission.py` | 13 |
| `tests/unit/test_moe_predictor_layer_id_semantics.py` | 1 |
| `tests/unit/test_moe_shared_routing_helper.py` | 9 |
| `tests/unit/test_mtp_terminal_overshoot_ep_replay.py` | 2 |
| `tests/unit/test_operator_parity_op_family_coverage_oracle.py` | 1 |
| `tests/unit/test_profiling_accelerator.py` | 12 |
| `tests/unit/test_quantization_manager_registry.py` | 3 |
| `tests/unit/test_sglang_experimental_increment13.py` | 7 |
| `tests/unit/test_sglang_graph_replay_orchestration.py` | 13 |
| `tests/unit/test_sglang_replay_admission.py` | 12 |
| `tests/unit/test_sglang_routed_sorting.py` | 5 |
| `tests/unit/test_sklearn_disaggregation_execution_time_predictor.py` | 2 |
| `tests/unit/test_stage_execution_time.py` | 17 |
| `tests/unit/test_stage_finalized_contract.py` | 15 |
| `tests/unit/test_stage_reporting_contract.py` | 14 |
| `tests/unit/test_timer_owner_lifecycle.py` | 24 |
| `tests/unit/test_typed_ep_lane_contract.py` | 1 |
| `tests/unit/test_typed_ep_predictor_contract.py` | 2 |
| `tests/unit/test_vllm_rocm_attention_wrapper_increment9.py` | 3 |

## Skip accounting and limits

The exact 25 skipped nodes are identical in baseline, first candidate, and final candidate. This was verified by mapping pytest progress result characters to the collected node order, asserting the character count equals the collection count, and independently asserting all reconstructed failed nodes equal the short-summary failure list. No skip was inferred solely from the total count.

- 3 external profiling dataset-tool tests: `FRONTIER_PROFILING_DATASET_TOOLS` is unset; the helper explicitly skips before invocation.
- 3 optional live-probe artifact tests: MHA CR-005, MLA FlashInfer and MQA FlashInfer artifacts are absent at their expected paths.
- 19 PD-AF tests: explicit existing skips for retired local-DP multi-lane/full-stage-return contracts, including 15 parameterized corrupt-queue cases and four legacy lane/FIFO cases.

The complete skip node list is in the JSON; relevant unchanged skip/skipif source is in `test_frontier_profiling_skill_tools.py`, the three Stage2/Stage3 modeling test files, and `test_pdaf_cluster_scheduler_invariants.py`. None of these 25 skips is claimed as the native AMD hardware lane. Native checks remain separately **SKIP: AMD/MI355X hardware unavailable**. Synthetic CPU tests and fake event contexts do not establish ROCm execution or native numerical/timing parity.

## Status

The full CPU unit baseline-relative criterion is closed: exact retained failures are explained, all 27 additional first-run failures are resolved, collection changes are accounted for, and there are no new skips. The unrelated 18 baseline failures remain visible. Other W10 acceptance lanes and W11 performance/native limitations require their own evidence; this report does not replace them.
