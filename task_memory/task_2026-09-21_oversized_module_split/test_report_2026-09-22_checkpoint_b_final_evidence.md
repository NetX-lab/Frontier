# Test Report 2026-09-22 — Checkpoint B: the final acceptance record for PR #34

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Initial report: clean recapture of both matrix sides, unit-suite regression comparison, retained split checks. |
| 2026-09-22 | C34-01: the cache-comparison row now names the two report keys it rests on, and section 2 records the re-derivation of the verdict with the corrected eligibility rule (executed case list, not filter alone). |

## What this closes

Maintainer review comments **R34-03** (publish an evidence record for the final
source and current matrix) and **R34-04** (retain the checks that specifically
protect the module split).

This report supersedes the earlier chronological runs as the acceptance record.
Those runs are not withdrawn — their per-case verdicts were genuine — but they
cannot serve as the final record, for reasons stated in section 1.

## Why the earlier evidence could not simply be cited

The review suggested proving that existing receipts already covered the final
tree rather than rerunning. That path was checked and is not available:

| Fact | Consequence |
| --- | --- |
| `db15e64..5ef96b5` is one commit touching only `tests/e2e/refactor_fidelity/cases.py` | The production tree was indeed unchanged at the tip. |
| `candidate_db15e64` held 67 records and **no DP-placement case** | The four cases that took the table from 67 to 71 had never run against the refactor tip. |
| The old `baseline` label's last execution was a filtered `dp_` run of four cases merged onto 67, with `cache_clean_before_run: false` | It was an assembled label: its cache listing did not come from one clean full run, and its `case_count` said 72 over 71 result lines because a retained record named a case id that had left the table. |
| Checkpoint A added per-case provenance | Every label captured before it now fails the gate for want of provenance, by design. |

Both sides were therefore recaptured as single clean full runs.

## Environment

| Field | Value |
| --- | --- |
| Host | `kun-workspace-vgen2`, CPU only |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python`, CPython 3.10.6 |
| Packages | numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, plotly 7.1.0, pytest 9.1.1 |
| Output root | `/data/ycfeng/tmp/issue26-correctness-pr/refactor-fidelity` (scratch, not committed) |

## 1. Fidelity matrix — the final record

Both sides were run by the harness committed at `bb582a4`, from a clean
detached checkout of that commit, so the measurement and the code being
measured agree and neither side was measured from a working tree.

```bash
# from .worktrees/fidelity-candidate-bb582a4, PYTHONPATH=$PWD
python tests/e2e/refactor_fidelity/run_matrix.py run \
  --repo-root .worktrees/fidelity-baseline-main --label baseline_v2 \
  --output-root "$OUT" --jobs 6 --clean-cache --continue-on-failure
python tests/e2e/refactor_fidelity/run_matrix.py run \
  --repo-root . --label candidate_bb582a4 \
  --output-root "$OUT" --jobs 6 --clean-cache --continue-on-failure
python tests/e2e/refactor_fidelity/run_matrix.py compare \
  --output-root "$OUT" --baseline-label baseline_v2 --candidate-label candidate_bb582a4
```

### Provenance of the two sides

| Field | `baseline_v2` | `candidate_bb582a4` |
| --- | --- | --- |
| Source revision | `1f694f7c549aa3aeeb7c5bbae04e119c09167a77` | `bb582a41702eb31edf4a6477fcf76e4f32d0f3ef` |
| Source working tree | clean | clean |
| Harness revision | `bb582a4` | `bb582a4` |
| Case filter | none | none |
| Cases executed in this run | 71 | 71 |
| `case_count` in the manifest | 71 | 71 |
| Cache cleaned before the run | yes | yes |
| Cache files produced | 426 | 426 |

### Result

| Measure | Expected | Actual | Result |
| --- | --- | --- | --- |
| Case table size | 71 | 71 | — |
| Cases compared | 71 | **71** | PASS |
| Identical | 71 | **71** | PASS |
| Mismatched | 0 | **0** | PASS |
| Baseline failures | 0 | **0** | PASS |
| Candidate-only failures | 0 | **0** | PASS |
| Cases missing from one side | 0 | **0** | PASS |
| Cases with missing evidence | 0 | **0** | PASS |
| Cases with differing definitions | 0 | **0** | PASS |
| Cases not compared | 0 | **0** | PASS |
| Cases not compared without explanation | 0 | **0** | PASS |
| Provenance findings | none | **none** | PASS |
| Predictor cache names compared | yes | **yes** (`predictor_cache_populated_cleanly: true`, `predictor_cache_compared: true`; both manifests list all 71 cases in `cases_executed_in_last_run`) | PASS |
| Predictor cache differences | 0 | **0 baseline-only, 0 candidate-only** | PASS |

**Comparison exit code 0.** This is the first run of this matrix in which
`cases_compared` equals the case-table size *and* the gate that checks it is
the corrected one, so the number means what it says.

The cache result is worth stating separately: all 426 cache file names matched,
across a table that includes the six trained-predictor cases, which means the
split changed no training identity and no model cache key. Output equality alone
could not show that, because retraining from the same CSV reproduces the same
numbers.

**Re-derived 2026-09-22 (C34-01).** The eligibility rule behind the cache row
was found too weak: it accepted `cache_clean_before_run` plus an empty
`case_filter`, which a `--start`/`--limit` continuation also satisfies. The rule
now requires each manifest's `cases_executed_in_last_run` to equal the full
case table. Both labels here record 71 executed cases, and rerunning
`run_matrix.py compare` on the retained output root with the corrected harness
reproduces this table exactly: 71 of 71 identical, `predictor_cache_compared:
true`, 0 baseline-only and 0 candidate-only cache files, exit code 0. The
acceptance verdict therefore stands under the corrected rule; see
`test_report_2026-09-22_cache_eligibility_correction.md`.

## 2. Unit suite — regression comparison

Whole `tests/unit` directory, both sides, same command, same exclusions.

**Excluded, not run:** ten modules fail at collection on both sides for missing
optional dependencies (`torch`, `matplotlib`), which the minimal release
environment deliberately omits:

```text
test_mla_native_profiling_wrapper        test_moe_routing_input_contract
test_moe_fused_event_contract            test_native_profiling_model_type_policy
test_moe_gating_constructor_boundary     test_profiling_timing_stats_contract
test_moe_load_distribution_contract      test_sim_walltime_scaling_plot
test_moe_native_admission                test_vllm_rocm_attention_wrapper_increment9
```

| Side | Failed | Passed | Skipped | Collection errors |
| --- | --- | --- | --- | --- |
| Baseline `1f694f7` | 84 | 3644 | 49 | 10 (excluded above) |
| This branch | 84 | **3679** | 49 | 10 (excluded above) |

**The 84 failing test identities are byte-identical between the two sides**
(`diff` of the sorted `FAILED` lists is empty). The 35 additional passing tests
are exactly the 35 added in Checkpoints A and B.

Inherited failures by module, all pre-existing on `main`:

| Module | Count | Cause |
| --- | --- | --- |
| `test_pdaf_parity_reference_observer_bootstrap.py` | 51 | Requires the pinned PD-AF reference checkout, absent on this host |
| `test_colocation_release_review_contracts.py` | 10 | Opens `tests/debug/` scripts not tracked on `main`; one spawns a bare `python` absent from PATH |
| `test_profiling_governance_minimal_red.py` | 5 | Profiling dependencies |
| `test_moe_mxfp4_increment10.py` | 5 | Profiling dependencies |
| `test_pdd_public_surface_docs.py` | 3 | Documentation contract drift on `main` |
| `test_pdaf_examples.py` | 3 | PD-AF reference checkout |
| `test_examples_documentation_contracts.py`, `test_collectives_increment11.py` | 2 each | Documentation / optional backend |
| three attention-modeling modules | 1 each | Profiling dependencies |

This is a regression comparison. It is **not** a claim that these 84 tests pass.

## 3. Retained checks for the split (R34-04)

`tests/unit/test_module_split_boundaries.py`, 13 tests, all passing. These ran
as task-local commands during the split; committing them is the difference
between a check that happened once and one that keeps happening.

| Review priority | Test |
| --- | --- |
| Generated CLI/config behavior beyond flag names | `test_no_new_config_dataclass_loses_its_annotations`, `test_the_flat_cli_can_still_be_generated` |
| Public re-exports and annotation resolution | `test_public_config_names_resolve_from_the_entry_point_that_is_imported`, `test_the_split_modules_stay_reachable_through_the_package` |
| Subclass override and `super()` resolution | `test_vllm_v1_scheduler_reaches_every_extracted_mixin`, `test_split_classes_keep_their_mixin_order`, `test_cluster_config_fields_come_only_from_the_owning_class`, `test_sglang_still_reaches_the_extracted_decision_log_helper` |
| Persistent estimator loading in a fresh manager | `test_a_cached_estimator_loads_into_a_fresh_registry` |
| Importability and the names that previously failed | `test_every_split_module_imports_in_a_fresh_interpreter_order`, `test_runtime_only_names_are_not_hidden_behind_type_checking` |

### One finding, pinned rather than fixed

`typing.get_type_hints(ClusterConfig)` raises `NameError: BaseCCBackendConfig`.
The same lookup fails on `1f694f7`, where the class still lived in the single
`config.py`, so this **predates the split**: `cc_backend_config` is annotated
with a name the module does not import at runtime, presumably to avoid a
circular import. The generated CLI is unaffected, which
`test_the_flat_cli_can_still_be_generated` asserts directly. It is recorded in
`KNOWN_UNRESOLVED_CONFIG_ANNOTATIONS` so that a *new* unresolvable annotation
fails the test, and so that fixing this one also fails it until the pin is
dropped.

## 4. Cross-revision cache loading

The committed unit test writes and reads a cache within one process. The review
asked for more: a **baseline-produced** cache loaded by the split code. The
recapture produced exactly that artifact, so it was used.

| Check | Result |
| --- | --- |
| Pickled estimators in `.worktrees/fidelity-baseline-main/cache` (written by `1f694f7`) | 142 |
| Unpickled by the split code | **142 / 142** |
| Also usable — `predict()` ran on those exposing `n_features_in_` | 86 |
| Loaded through `PredictionModelRegistry._load_model_from_cache` | ok |
| Failures | **0** |

A pickle records the module path of the class it holds, and moving code is
exactly what invalidates that. Matching cache *names* would not have caught it.

## Limits

- The matrix establishes that 71 supported configurations produce identical
  output. It says nothing about configurations outside the table and makes no
  accuracy claim.
- Dummy-mode cases exercise structure and lifecycle, not realistic latency. Six
  cases use the checked-in profiling CSVs, which is what covers dataset loading,
  training identity and the persistent cache.
- The predictor cache comparison compares file names, not pickle bytes, which
  are not required to be reproducible. Section 4 covers loadability separately.
- Section 4 loads the estimators the 71-case table happens to train. It is not
  an exhaustive inventory of every artifact Frontier can cache.
- Ten unit modules were not executed at all. Their coverage is unknown on both
  sides, not passing.
