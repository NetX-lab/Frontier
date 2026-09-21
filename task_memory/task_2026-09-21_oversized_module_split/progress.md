# Oversized Module Split — Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Step 0: worktree and branch created, records landed, module surveys collected, baseline pending. |

## Status

| Field | Value |
| --- | --- |
| Branch | `refactor/oversized-module-split` |
| Base | `1f694f7c549aa3aeeb7c5bbae04e119c09167a77` (`origin/main`, fetched 2026-09-21) |
| Worktree | `/data/ycfeng/Frontier/.worktrees/oversized-module-split` |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python` |
| Current step | Step 0 (baseline recorded) |
| Publication | PUSHED_VERIFIED through the Step 0 records; Step 1 and 2 pending |
| Next action | Commit `.gitignore` + records, push, create the correctness worktree from this branch, push it, stop for user review (Q6=a). |

## Steps

| Step | Work | Status |
| --- | --- | --- |
| 0 | Worktree, records, environment, baseline | PASS |
| 1 | Fidelity matrix harness and main baseline capture | PASS (67 cases, baseline 67/67) |
| 2 | Cleanup pass per module, matrix re-run | IN_PROGRESS (config.py done and matrix-identical) |
| 3 | Split `config.py` | PASS (12 modules, largest 1888 lines) |
| 4 | Split `vllm_v1_engine_replica_scheduler.py` | NOT_STARTED |
| 5 | Split `shared_prediction_model_manager.py` | NOT_STARTED |
| 6 | Split `sklearn_moe_execution_time_predictor.py` | NOT_STARTED |
| 7 | Full matrix, unit suites, draft PR hand-off | NOT_STARTED |

## Chronological updates

- 2026-09-21: Branch created from `1f694f7`. `.gitignore` narrowed to `task_memory/*` with exceptions for the two task directories. Structural surveys of the four modules collected into `module_survey.md` (read-only inspection, grep-backed).
- 2026-09-21: Environment `/data/ycfeng/envs/frontier-py310` created (CPython 3.10.6). Baseline: 84 passed / 10 failed (all in `test_colocation_release_review_contracts.py`, caused by debug scripts absent from main and a bare `python` executable missing on PATH); co-location and PDD dense dummy smokes PASS. See `test_report_2026-09-21_step0_baseline.md`.
- 2026-09-21: Step 1 complete. Harness `tests/e2e/refactor_fidelity/` built: 67 cases across co-location, sequential PDD and sequential PD-AF, dense and MoE, offline and online, dummy and checked-in-CSV predictors. Self-check (same code twice) 13/13 identical. Baseline capture 67/67 after correcting two invalid case topologies. See `test_report_2026-09-21_step1_fidelity_matrix.md`.
- 2026-09-21: Step 2 started with `config.py`. Removed the unreferenced `validate_linear_op_input`; replaced the two CC-backend dispatch ladders with one ordered table; deduplicated four field-resolution closures and three identical field triplets; dropped a redundant method-local import. 5720 to 5687 lines. Generated CLI flag set identical (753 flags). Matrix 67/67 identical. A 138-name predictor cache difference was traced to the two profiling smoke wrappers defaulting their data base to an absolute path under the repository root, which enters the model hash; both cases now use a repository-relative base and both sides were recaptured.
- 2026-09-21: Step 3 started. Extracted eight leaf modules from `config.py`: `release_guards.py`, `request_generator_config.py`, `replica_scheduler_config.py`, `metrics_config.py`, `speculative_decoding_config.py`, `replica_config.py`, `cluster_scheduler_config.py`, `execution_time_predictor_config.py`. `flat_dataclass` resolves string annotations in the defining module's namespace, so each module carries the imports its own annotations need; the compatibility gate is the unchanged CLI flag set plus the 36 names other modules import from `frontier.config`.
- 2026-09-21: `config.py` split complete and gated. Twelve modules, largest 1888 lines. Gates: generated CLI flag set identical (753 flags); all 36 externally imported names resolve from both `frontier.config` and `frontier.config.config`; `ClusterConfig` keeps 181 dataclass fields and gains the two mixins in its MRO; the 62-file configuration unit selection gives 10 failed / 671 passed on both sides with byte-identical failure identities, all pre-existing drift; fidelity matrix 67 identical, 0 mismatched, 0 predictor cache name differences.
- 2026-09-21: One real defect was introduced and caught by the unit selection before the matrix reached it: `get_cluster_configs_for_disaggregation` constructs `ClusterConfig` at runtime and the extraction had imported the name only under `TYPE_CHECKING`. Recorded as I1 in `issues.md`, fixed with a lazy import.
