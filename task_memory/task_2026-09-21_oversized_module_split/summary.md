# Oversized Module Split — Summary

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Placeholder created at Step 0. |
| 2026-09-21 | Completion archive written after the fourth and last module split. |

## Overview

Four Python modules that the Issue 26 correctness fixes must edit were far above the 2,000-line maintainability gate in `AGENTS.md`. Editing them in the same pull request as the behavior fixes would have made that diff unreadable, because a reviewer could not tell a moved line from a changed one. This branch brings all four under the gate through a cleanup-first pass and a functional split, with no behavior change and no numeric change, and the correctness branch is stacked on top of it.

| Module | Before | After (largest child) | Modules |
| --- | --- | --- | --- |
| `frontier/config/config.py` | 5720 | 1888 | 12 |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 5138 | 1386 | 8 |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | 4614 | 1700 | 6 |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 3539 | 1557 | 6 |

Five modules remain above 2,000 lines. All five were out of scope from the start and are named in `plan.md` section 1: `sklearn_execution_time_predictor.py` (8263), `metrics_store.py` (5586), `sklearn_disaggregation_execution_time_predictor.py` (2985), `profiling/attention/main.py` (2157), `entities/request.py` (2125).

## Deliverables

| Path | Contents |
| --- | --- |
| `frontier/config/` | 12 modules: the simulation config and its cluster topology, the cluster role builders and topology summary, and one module per configuration family |
| `frontier/scheduler/replica_scheduler/vllm_v1_*.py` | 8 modules: the scheduler plus MTP wait policy, role schedules, KV allocation, iteration policy, prefix cache, decode-attention cohort and the decision log |
| `frontier/execution_time_predictor/` | 12 modules across the two predictors: family trainers, dataframe loaders, model registry, layer contract resolution, identity helpers, and the MoE operator times, routing workload, dataset training, MTP replay and helpers |
| `tests/e2e/refactor_fidelity/` | The 67-case fidelity matrix that gates the branch |
| `task_memory/task_2026-09-21_oversized_module_split/` | `plan.md`, `module_survey.md`, `issues.md`, `progress.md`, the Step 0 baseline report and the Step 1 matrix report |

## How the no-change claim is supported

**The fidelity matrix.** 67 cases, each run through a checked-in example wrapper on both the branch and a read-only checkout of the base commit. Every artifact a run writes is compared and the file sets must match: `request_metrics.csv`, `system_metrics.json`, `frontier_stage_batch_ledger.jsonl`, `op_precision_metadata.csv` and `config.json`. The gate is exact equality with no tolerance, because a behavior-preserving refactor has no reason to change a simulated number. Only three run-specific absolute paths are normalized, and that list was checked against the real artifacts rather than assumed.

Coverage: co-location, sequential PDD and sequential PD-AF; dense and MoE; offline and online; the dummy predictor and the checked-in profiling CSVs; request counts from 4 to 64; prompts from 128 to 3584 tokens; three arrival rates; chunked prefill, all three decode CUDA graph modes, prefix caching, speculative decoding and thinking mode; TP, PP, attention DP and EP variations.

**The predictor cache names.** Retraining from the same CSV reproduces the same numbers, so a changed training identity or cache key would leave no trace in the outputs. The comparison therefore also checks the names of the predictor cache files each side produces, and classifies any difference as rekeyed, baseline-only or candidate-only. This is the specific evidence that moving the model hash, the cache keys, the registry and the FFN contract signature into different modules changed no training identity.

**The unit suites.** A selection that grows with each step, ending at 183 files, run on both sides with failure identities compared rather than counts. Identities matter: a count comparison would have hidden several of the defects below, because the same files also carry pre-existing failures.

**The CLI surface.** The generated flag set, 753 flags, compared after every step, plus a check that all 36 names other modules import from `frontier.config` still resolve from both import paths.

**A static check.** Every split module is parsed and any name loaded at runtime without a runtime binding is reported, counting `TYPE_CHECKING` imports as unbound. It found one defect the matrix had not yet reached.

## Observed validation results

| Step | Module | Unit selection | Fidelity matrix |
| --- | --- | --- | --- |
| 3 | `config.py` | 62 files, 10 failed / 671 passed, identical identities | 67 identical, 0 mismatched, 0 cache differences |
| 4 | vLLM V1 replica scheduler | 73 files, 13 failed / 1570 passed, identical identities | 67 identical, 0 mismatched, 0 cache differences |
| 5 | prediction model manager | 2300-odd tests, identical | 67 identical, 0 mismatched, 0 rekeyed models |
| 6 | MoE execution-time predictor | 183 files, 28 failed / 3016 passed, identical identities | recorded in `progress.md` when the final measurement lands |

The config and manager splits were additionally measured from detached checkouts at their own commits by a second session, so the tree measured was the commit and nothing else.

Pre-existing failures fall into two buckets and are not caused by this branch: ten in `test_colocation_release_review_contracts.py`, which open `tests/debug/` scripts that are not tracked on `main` at all, and several that raise `ModuleNotFoundError` on `torch`, which the minimal CPU environment deliberately excludes.

## What a reviewer should look at

The eleven entries in `issues.md` are the substance of the review. Eight are defects this branch introduced and fixed, and each records what found it. Three of those were names the moved or retained code loads at runtime without a runtime binding, which is now checked mechanically rather than by inspection.

The test changes divide into three kinds, and only the second is a judgment call:

1. A patch target moved with the code it patches. A test that patches a module-level name stops intercepting once the method reading it resolves that name in another module. Seven test files, no assertion changed.
2. One governance allowlist entry. `test_raw_model_profile_resolution_callsites_are_allowlisted` pins which functions may resolve a raw model architecture profile and how many times each may do so. Exactly one line changed, a file path; the function name, the kind, the expected count of 1 and the total of 11 entries are unchanged, so the property the gate protects is intact.
3. Three tests gained a line. `MOE_FAMILY` was one binding and is now imported by two modules that the path under test both read, so a fake operator family has to be installed in both. The assertions are unchanged, and the second patch is needed precisely because the production code genuinely reads the name in both places.

## Open and deferred work

- The final fidelity measurement of `cb54fb4` is the last outstanding gate; `progress.md` records its result.
- The naming of the extracted mixin classes is a review point, not a settled decision. Renaming any of them is mechanical and affects three files at most.
- Two modules keep a name that is reported by the static check and is correct: `BaseCCBackendConfig` in `frontier/config/cluster_config.py` is a string annotation the flat CLI generator resolves through its own lazy-import special case, exactly as the pre-split `config.py` did.
- `AGENTS.md` still points readers at `tests/unit/test_open_source_release_arch_guard.py` and two `tests/debug/` scripts that do not exist on `main`. Recorded as deferred; fixing it is not in this branch's scope.
