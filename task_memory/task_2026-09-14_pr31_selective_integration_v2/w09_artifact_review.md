## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Independently reviewed W09 artifact boundaries and reproduced 25 failures with a real multi-row CPU fit/save/load campaign. |

# W09 Independent Artifact Boundary Review

## Scope and status

- Target component: W09 / N07 / N08 / T16–T17, GDN dataset identity, artifact validation, persistence, prediction, and cache reuse.
- Reviewer: `/root/w09_artifact_review`.
- Implementation: production changes belong to the integration owner; this lane changed no production files.
- Verification: **25 failed, 6 passed** in 5.90 seconds, intentionally exposing the outstanding defects before repair.
- Readability/ownership: the trainer, predictor, family schema, dataframe validation, BaseTrainer cache, persistence helper, and manager caller were inspected.
- Owned changes: `tests/unit/test_gdn_artifact_boundary.py` and this review document only.
- No commit or publication was performed by this lane.
- W00 benchmark completion was confirmed by the integration owner before test execution.

## Inspected artifacts and hunk dispositions

| File / boundary | Disposition | Finding and proposed correction |
| --- | --- | --- |
| `frontier/training/gdn_trainer.py`, new-file hunk, selectors and `_load_dataset` | BUGFIX | `model_architecture_profile`, `quant_signature`, and `device` are selected from row zero without checking uniqueness when selectors are absent. Runtime/shape checks use `dropna()` and admit partially missing selected identities. Extend the existing GDN schema boundary to require exactly one nonempty value across all selected rows for every required identity field. |
| Same trainer, `GDN_TASKS` | REFACTOR | Move the shared phase-qualified task contract into the dependency-light GDN schema responsibility, using existing family operator phases. Loading must consume the same canonical set without importing the trainer. |
| Same trainer, `_train_task_estimator` | REFACTOR | Its docstring explicitly describes a one-row fixture exception. Decide an explicit cardinality policy consistent with BaseTrainer or use sufficient fixture rows. This new test campaign uses four distinct rows per phase and exercises actual GridSearchCV. |
| Same trainer, artifact writing in `train` | BUGFIX | Direct `pickle.dump` to the final path truncates a previously valid artifact on interrupted serialization. Reuse `frontier/execution_time_predictor/cache_io.py:atomic_pickle_dump`. Publish a complete validated manifest after artifacts are available. |
| `frontier/execution_time_predictor/gdn_predictor.py`, new-file hunk, `from_directory` | BUGFIX | No schema-version check, no canonical task-set completeness/uniqueness check, no manifest-to-artifact task/feature-order/target check. An identity-equal artifact swap is accepted. Normalize and validate this boundary before returning a predictor. |
| Same predictor, `_predict_one` | BUGFIX | Exact lookup returns before the existing finite/nonnegative validation. Route exact and estimator values through the same numeric check. Preserve extrapolation warnings and estimator behavior. |
| Trainer and predictor, `_dataset_fingerprint` | REFACTOR | Implementations are identical and should share one existing cache/identity responsibility. Preserve exact CSV-byte identity semantics. |
| `frontier/attention/families.py`, GDN family | RETAIN / REUSE | Already declares operator phases and required profiling columns. Use this as the existing semantic source instead of creating unrelated operator/task registries. |
| `frontier/attention/profiling_mapping.py`, `validate_attention_profiling_dataframe` | RETAIN / EXTEND BOUNDARY | Currently checks required columns and measurement-type identity, not complete selected row identity. Extend in the GDN-specific selected-scope boundary without changing unrelated families. |
| `frontier/training/base_trainer.py`, `_train_single_model` and cache methods | RETAIN / INTEGRATION REVIEW | Normal multi-row fit and cache reuse work. GDN reuse is proven with RandomForestRegressor.fit forbidden during the second training call. BaseTrainer's cache writer also uses direct pickle writes; integration owner should assess narrowly while reusing atomic persistence. |
| `frontier/execution_time_predictor/cache_io.py`, `atomic_pickle_dump` | RETAIN / REUSE | Existing helper writes and fsyncs a temporary sibling, then replaces the destination; failed serialization removes the temporary file. No new transaction framework is needed for per-artifact atomic writes. |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py`, `_load_gdn_predictor_for_cluster` | RETAIN / CALLER REVIEW | Calls `GDNPredictor.from_directory` with model/device/TP/measurement/CSV identity and does not fit at runtime. A stronger loader closes this caller's artifact corruption gap. Required-field probing remains an ownership-cleanup item for the integration owner. |

## Test construction and independent expectations

The campaign reuses `tests/fixtures/pr31_hybrid/gdn.csv`, retaining its explicit `synthetic_cpu_v1` / `synthetic_cpu` identity. It expands each phase to four valid feature rows while preserving the fixture's constant operator times. This makes both exact lookup and non-exact estimator output independently predictable without a test-only estimator or constructor bypass.

- Prefill oracle in milliseconds: input projections `0.11`, core `0.22`, output projection `0.33`.
- Decode oracle in milliseconds: input projections `0.04`, core `0.05`, output projection `0.06`.
- The non-exact queries are prefill length `24` and decode batch `6`, absent from the four training rows.
- Manifest and pickle corruptions are made only inside isolated copies of actual trained artifacts.
- Persistence fault injection writes `b"interrupted serialization"` and raises during pickle serialization. The required outcome is that the previous complete artifact remains byte-identical and loadable.
- The second training call prohibits `RandomForestRegressor.fit`; success verifies actual cache reuse instead of merely observing a log message.

## Execution and environment

Repository: current `feature-amd-sglang-gdn` worktree. Root HEAD observed during review was `10326ce48835cc8345d42b7ec2a98506ed03bc74`; the integration owner was executing other packages concurrently. No GDN trainer/predictor production change occurred in this review lane.

Interpreter: `/usr/bin/python`, Python `3.12.3`, GCC `13.3.0`. No conda environment was activated for the command. NumPy `2.4.6`, pandas `3.0.3`, and scikit-learn `1.9.0` came from `/usr/local/lib/python3.12/dist-packages`; pytest `9.1.1` came from `/home/i-fengyicheng/.local/lib/python3.12/site-packages`.

Reproducible command, run from the repository root:

```bash
env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
  WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp \
  /usr/bin/python -m pytest tests/unit/test_gdn_artifact_boundary.py \
  -q -p no:cacheprovider \
  --basetemp /data/ycfeng/tmp/pr33-w09-artifact-review-20260916-final \
  > /data/ycfeng/tmp/pr33-w09-artifact-review-20260916-final.log 2>&1
```

Initial run before adding cache-reuse and estimator-validation controls: **25 failed, 2 passed in 8.06 seconds**, log `/data/ycfeng/tmp/pr33-w09-artifact-review-20260916.log`.

Final test-construction run: **25 failed, 6 passed in 5.90 seconds**, exit `1`, log `/data/ycfeng/tmp/pr33-w09-artifact-review-20260916-final.log`.

## Observed failing test nodes

All nodes below have the prefix `tests/unit/test_gdn_artifact_boundary.py::`.

| Node | Parameter IDs | Actual failure |
| --- | --- | --- |
| `test_manifest_corruption_rejected_before_prediction` | `schema`, `missing`, `duplicate`, `swapped`, `task`, `phase`, `features`, `target` | `Failed: DID NOT RAISE <class 'ValueError'>` |
| `test_artifact_metadata_must_match_declared_task` | `task`, `feature_names`, `target_col` | `Failed: DID NOT RAISE <class 'ValueError'>` |
| `test_omitted_selector_requires_unique_selected_identity` | `model_architecture_profile-another_profile`, `quant_signature-another_quantization`, `device-another_device` | `Failed: DID NOT RAISE <class 'ValueError'>` |
| `test_selected_identity_rejects_missing_row_values` | `model_architecture_profile`, `quant_signature`, `device`, `runtime_stack_signature`, `gdn_runtime_backend`, `model_dtype`, `hidden_size` | `Failed: DID NOT RAISE <class 'ValueError'>` |
| `test_exact_predictions_require_finite_nonnegative_values` | `-1.0`, `nan`, `inf` | `Failed: DID NOT RAISE <class 'ValueError'>` |
| `test_interrupted_artifact_publication_preserves_existing_artifact` | none | Final artifact became `b'interrupted serialization'`; equality with original valid pickle failed. |

The passing controls are `test_multirow_fit_fresh_load_exact_and_estimator`, `test_explicit_identity_selectors_select_one_complete_scope`, `test_multirow_training_reuses_existing_estimators`, and all three parameterizations of `test_estimator_predictions_require_finite_nonnegative_values`.

## Remaining work and limitations

1. Integration owner applies W09 production corrections after its W06/schema dependencies.
2. Run this file again together with the existing GDN training/predictor and production-constructor CPU E2E coverage.
3. Record source-bound repaired results and complete the W09 manifest-publication/cardinality policy checks.

This is a CPU artifact-boundary regression report. It establishes no native CUDA/ROCm GDN support, measured GPU parity, final-source acceptance, or multi-file publication atomicity. The injected interruption tests the first artifact rewrite; it does not prove consistency after successful replacement of some artifacts followed by a later interruption. Runtime-stack selector filtering and partial-manifest publication were reviewed as pending questions, not reproduced here.
