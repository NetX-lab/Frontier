## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Cross-referenced the persisted real CPU hybrid Simulator E2E report and corrected the final commit-range reference to pushed HEAD `578785bb`. |
| 2026-09-15 | Added final whitespace hygiene commit `578785bb`, reran the full CPU unit suite at the pushed PR head, and confirmed unchanged baseline failures. |
| 2026-09-14 | Recorded final Review 2 focused checks, full CPU unit comparison, documentation closure, source-plan immutability, and publication readiness. |

# Final Review 2 Test Report

## Execution

Candidate:

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Branch: `feature-amd-sglang-gdn`
- Candidate HEAD: `578785bb` (final pushed head; `ffb9feea` remains the preceding functional fix)
- `origin/main`: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`
- Read-only source PR reference: `origin/pr-31` at `215a80a204ecdd8fc50275b5c48f14a86cedaab5`
- Source plan: `/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md`
- Source-plan SHA-256: `17365fe6a96cecf162d63f450fac3fd83bd91caaa2abf87f04c81131a08afd66`

Environment:

- Python: 3.12.3 (`/usr/bin/python`)
- NumPy: 2.4.6
- pandas: 3.0.3
- scikit-learn: 1.9.0
- Plotly: 6.8.0
- PyTorch: 2.5.1+cu124
- `vllm`: unavailable (`ModuleNotFoundError`)
- `sglang`: unavailable (`ModuleNotFoundError`)
- `aiter`: unavailable (`ModuleNotFoundError`)

The final full-unit output is preserved at `/data/ycfeng/tmp/pr31-final-unit-578785bb.log`.

## Commands and criteria

### Review 2 focused lane

Command:

```bash
python -m pytest \
  tests/unit/test_hybrid_runtime_family.py \
  tests/unit/test_gdn_profiler_cpu_increment7.py \
  tests/unit/test_sglang_experimental_increment13.py \
  tests/unit/test_sglang_graph_replay_orchestration.py \
  tests/unit/test_gdn_hybrid_e2e_increment14ab.py \
  tests/unit/test_gdn_training_predictor_increment8.py \
  tests/unit/test_gdn_increment6_memory.py \
  tests/unit/test_gdn_runtime_guards.py \
  tests/unit/test_metrics_stage_execution_time.py \
  -q -p no:cacheprovider
```

Expected: hybrid family/metrics/replay/GDN contracts pass without candidate regression.

Observed: **PASS — 56 passed in 5.62s**.

The current exact-HEAD re-review also ran `python -m pytest tests/unit/test_sglang_experimental_increment13.py -q -p no:cacheprovider` with **7 passed**, plus `python -m compileall -q frontier/profiling/experimental/sglang` and `git diff --check`. The earlier concern about raw builder tuple lengths is closed because `graph_replay._make_replay_call()` normalizes every builder before `profile_graph()` accesses `call[3]`, `reset()`, or `check()`.

### Existing execution and metrics compatibility lane

Command:

```bash
python -m pytest \
  tests/unit/test_execution_time_op_times.py \
  tests/unit/test_mla_core_native_op_tracing.py \
  tests/unit/test_ffn_memory_operator_families.py \
  tests/unit/test_moe_share_expert_operator_families.py \
  tests/unit/test_metrics_full_stage_scope.py \
  -q -p no:cacheprovider
```

Expected: dense execution, MLA, FFN/MoE operator families, and metrics remain compatible.

Observed: **PASS — 91 passed in 4.62s**.

### Profiling examples documentation contract

Command:

```bash
python -m pytest tests/unit/test_examples_profiling_contracts.py -q -p no:cacheprovider
```

Expected: profiling wrappers and their migration/output contracts pass.

Observed: **PASS — 15 passed in 0.93s**.

### Full CPU unit regression

Command:

```bash
python -m pytest tests/unit -q -p no:cacheprovider
```

Expected: no candidate-only failures; any existing baseline failures must be listed verbatim and compared by test name.

Observed: **FAIL as a repository-wide command, with no candidate-only regression — 3274 passed, 19 failed, 25 skipped, 576 warnings in 70.74s**.

Complete failure list:

```text
tests/unit/test_colocation_release_review_contracts.py::test_config_optimizer_help_prints_without_argparse_percent_crash
tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_does_not_default_to_private_conda_path
tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_conda_activation_temporarily_disables_nounset
tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_uses_safe_pythonpath_expansion
tests/unit/test_colocation_release_review_contracts.py::test_readme_debug_scripts_do_not_use_post_increment_under_set_e
tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_resolves_latest_canonical_metrics_run_dir
tests/unit/test_colocation_release_review_contracts.py::test_release_debug_scripts_use_canonical_metrics_resolver
tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_uses_valid_shared_parallel_domain
tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_header_uses_current_tp_terms
tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_satisfies_shared_parallel_domain
tests/unit/test_examples_documentation_contracts.py::test_readme_documents_current_examples_surface
tests/unit/test_examples_documentation_contracts.py::test_public_docs_list_the_complete_pdaf_v03_surface_and_boundaries
tests/unit/test_mha_stage2_profile_modeling.py::test_stage2_cli_writes_reproducible_artifacts
tests/unit/test_mla_core_native_op_tracing.py::test_op_level_tracing_generates_metadata_for_structured_mla_ops
tests/unit/test_mla_stage3_online_trace_builder.py::test_mla_stage3_cli_writes_trace_and_error_matrix
tests/unit/test_mqa_stage2_profile_modeling.py::test_mqa_stage2_cli_writes_reproducible_artifacts
tests/unit/test_pdd_public_surface_docs.py::test_top_level_docs_advertise_supported_pdd_without_upcoming_claims
tests/unit/test_pdd_public_surface_docs.py::test_public_docs_require_sequential_pdd_release_mode
tests/unit/test_pdd_public_surface_docs.py::test_public_docs_state_internal_parallel_pdd_boundary_and_current_evidence
```

The first ten failures are existing config/debug E2E asset conditions. The next two are stale release-example documentation contracts. The next four are existing MHA/MLA/MQA analysis-builder contracts. The final three are stale top-level `README.md` PDD contracts. The baseline report recorded the same 19 failure categories at `3144 passed, 19 failed, 25 skipped`; the candidate's additional tests account for the higher pass count, and the failure names remain unchanged. The requested documentation scope deliberately leaves top-level `README.md` untouched.

### Compilation and whitespace

Commands:

```bash
python -m compileall -q frontier tests
git diff --check
```

Expected: all touched Python files compile and no whitespace errors are present.

Observed: **PASS** for both commands.

The final commit-range hygiene check was also run after `578785bb`:

```bash
git diff --check origin/main..HEAD
```

Observed: **PASS**. The final commit removes two trailing spaces from the StageExecutionTime metrics fixture; `tests/unit/test_metrics_stage_execution_time.py` passed **4/4** before the commit.

### Non-dummy simulator CSV smokes

Commands already completed for final artifacts:

```bash
bash examples/profiling/smoke_simulator_dense_csv.sh \
  --metrics-output-dir /data/ycfeng/tmp/pr31-final-dense-20260914 \
  --run-id final_dense_csv

bash examples/profiling/smoke_simulator_moe_csv.sh \
  --metrics-output-dir /data/ycfeng/tmp/pr31-final-moe-20260914 \
  --run-id final_moe_csv
```

Expected: checked-in non-dummy CSVs train/load predictors, complete requests, and write canonical metrics.

Observed: **PASS**. Dense artifacts:

```text
/data/ycfeng/tmp/pr31-final-dense-20260914/qwen2_dense_test/offline_batch/final_dense_csv/request_metrics.csv
/data/ycfeng/tmp/pr31-final-dense-20260914/qwen2_dense_test/offline_batch/final_dense_csv/system_metrics.json
```

MoE artifacts:

```text
/data/ycfeng/tmp/pr31-final-moe-20260914/qwen3_30b_a3b_tiny/offline_batch/final_moe_csv/request_metrics.csv
/data/ycfeng/tmp/pr31-final-moe-20260914/qwen3_30b_a3b_tiny/offline_batch/final_moe_csv/system_metrics.json
```

These are CPU simulator workflow checks and do not establish AMD timing parity.

### CLI and wrapper checks

Commands:

```bash
bash examples/profiling/profile_linear_op.sh --dry-run
bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
bash examples/profiling/profile_moe.sh --dry-run
python -m frontier.profiling.gdn.main --help
python -m frontier.profiling.collectives.main --help
```

Expected: argument/path planning and optional-dependency-safe CLI imports succeed without launching GPU work.

Observed: **PASS** for all commands.

## Final acceptance matrix

| Criterion | Result | Evidence and practical limit |
| --- | --- | --- |
| Selective, independently reviewable integration | PASS | Increment commits from `0989f0ed` through pushed HEAD `578785bb`; original PR #31 remains unchanged. |
| Hybrid per-layer dispatch and ordered stage metrics | PASS | 56-test Review 2 lane, 91-test compatibility lane, synthetic eight-layer E2E and stage ledger checks. |
| Standard GDN training/prediction identity | PASS | Six estimator artifacts, manifest identity, changed-CSV fingerprint rejection, and fresh manager load tests. |
| Standard profiling output contracts | PASS | `DEVICE_EVENT`/`gdn.csv` schema and 15 profiling documentation-contract tests. |
| Dense/MoE CPU simulator workflow | PASS | Non-dummy dense and MoE CSV smokes completed and wrote request/system metrics. |
| Persistent hybrid CPU Simulator workflow | PASS with documented test seam | `test_report_2026-09-15_hybrid_simulator_e2e.md` records the real Simulator/Replica/MemoryPlanner run, one completed 18-token request, stage ledger, and per-layer GDN/dense trace identity. The simulator receives a prepared predictor; production-data model-manager construction remains open. |
| Full unit regression | BASELINE-COMPARED | 3274 passed, 19 failed, 25 skipped; failure names match the 19 recorded baseline failures. |
| Compilation and whitespace | PASS | `compileall` and `git diff --check`. |
| AMD ROCm/vLLM/AITER/MXFP4/RCCL/SGLang runtime | SKIP: AMD/MI355X hardware unavailable | No AMD host is available; CPU/import/schema checks do not establish GPU correctness. |
| AMD benchmark or groundtruth parity | SKIP: AMD/MI355X hardware unavailable | No benchmark or groundtruth comparison was executed or claimed. |

## Attribution and publication boundary

Selective integration commits, including final hygiene commit `578785bb`, preserve `Co-authored-by: powderluv <74956+powderluv@users.noreply.github.com>` wherever applicable. The older `e8ac59ae` commit has no trailer; no history rewrite was performed. GitHub contributor display may lag commit trailers and cannot be guaranteed by this local validation.

The branch was pushed as `origin/feature-amd-sglang-gdn`, and new PR [#33](https://github.com/NetX-lab/Frontier/pull/33) is open against `main`. Merge, rebase onto main, branch deletion, and any modification of the original PR remain prohibited until the user provides separate explicit authorization.
