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
| 4 | Split `vllm_v1_engine_replica_scheduler.py` | PASS (8 modules, largest 1386 lines) |
| 5 | Split `shared_prediction_model_manager.py` | PASS (6 modules, largest 1700 lines) |
| 6 | Split `sklearn_moe_execution_time_predictor.py` | PASS (6 modules, largest 1557 lines) |
| 7 | Full matrix, unit suites, draft PR hand-off | NOT_STARTED |

## Chronological updates

- 2026-09-21: Branch created from `1f694f7`. `.gitignore` narrowed to `task_memory/*` with exceptions for the two task directories. Structural surveys of the four modules collected into `module_survey.md` (read-only inspection, grep-backed).
- 2026-09-21: Environment `/data/ycfeng/envs/frontier-py310` created (CPython 3.10.6). Baseline: 84 passed / 10 failed (all in `test_colocation_release_review_contracts.py`, caused by debug scripts absent from main and a bare `python` executable missing on PATH); co-location and PDD dense dummy smokes PASS. See `test_report_2026-09-21_step0_baseline.md`.
- 2026-09-21: Step 1 complete. Harness `tests/e2e/refactor_fidelity/` built: 67 cases across co-location, sequential PDD and sequential PD-AF, dense and MoE, offline and online, dummy and checked-in-CSV predictors. Self-check (same code twice) 13/13 identical. Baseline capture 67/67 after correcting two invalid case topologies. See `test_report_2026-09-21_step1_fidelity_matrix.md`.
- 2026-09-21: Step 2 started with `config.py`. Removed the unreferenced `validate_linear_op_input`; replaced the two CC-backend dispatch ladders with one ordered table; deduplicated four field-resolution closures and three identical field triplets; dropped a redundant method-local import. 5720 to 5687 lines. Generated CLI flag set identical (753 flags). Matrix 67/67 identical. A 138-name predictor cache difference was traced to the two profiling smoke wrappers defaulting their data base to an absolute path under the repository root, which enters the model hash; both cases now use a repository-relative base and both sides were recaptured.
- 2026-09-21: Step 3 started. Extracted eight leaf modules from `config.py`: `release_guards.py`, `request_generator_config.py`, `replica_scheduler_config.py`, `metrics_config.py`, `speculative_decoding_config.py`, `replica_config.py`, `cluster_scheduler_config.py`, `execution_time_predictor_config.py`. `flat_dataclass` resolves string annotations in the defining module's namespace, so each module carries the imports its own annotations need; the compatibility gate is the unchanged CLI flag set plus the 36 names other modules import from `frontier.config`.
- 2026-09-21: `config.py` split complete and gated. Twelve modules, largest 1888 lines. Gates: generated CLI flag set identical (753 flags); all 36 externally imported names resolve from both `frontier.config` and `frontier.config.config`; `ClusterConfig` keeps 181 dataclass fields and gains the two mixins in its MRO; the 62-file configuration unit selection gives 10 failed / 671 passed on both sides with byte-identical failure identities, all pre-existing drift; fidelity matrix 67 identical, 0 mismatched, 0 predictor cache name differences.
- 2026-09-21: One real defect was introduced and caught by the unit selection before the matrix reached it: `get_cluster_configs_for_disaggregation` constructs `ClusterConfig` at runtime and the extraction had imported the name only under `TYPE_CHECKING`. Recorded as I1 in `issues.md`, fixed with a lazy import.
- 2026-09-21: vLLM V1 replica scheduler split complete and gated. `vllm_v1_engine_replica_scheduler.py` went from 5138 lines to eight modules, largest 1386. Cleanup removed `_attach_afd_metadata_if_needed`, 65 lines with no caller that duplicated `scheduler/utils/afd_metadata.py`. Gates: CLI flag set identical; method resolution verified for every private method the four subclasses override, and for the two the base class also defines; unit selection of 73 files across `tests/unit` and `tests/integration` gives 13 failed / 1570 passed / 19 skipped / 5 errors on both sides with identical failure identities; fidelity matrix 67 identical, 0 mismatched, 0 predictor cache name differences.
- 2026-09-21: Four regressions were introduced by that split and fixed; see `issues.md` I5 to I8. Three were names the moved or retained code loads at runtime without a runtime binding, which is now checked mechanically: a script parses each split module and reports names loaded but neither imported, defined locally, nor builtin, counting `TYPE_CHECKING` imports as unbound. Its only remaining hit is a string annotation the flat CLI generator resolves through its own lazy-import special case.
- 2026-09-21: Independent confirmation of commit `99922d2` from a second session, measured in an isolated detached worktree at that commit rather than through the shared tree: 67/67 identical, 0 mismatched, 0 cache name differences.
- 2026-09-21: Prediction model manager split complete. 4614 lines to six modules, largest 1700. Cleanup removed three definition-only members. Unit selection widened to the predictor tests, 2300-odd tests over both sides: identical results, including identical failure identities. Eight tests needed their patch target moved with the code they exercise; see `issues.md` I9.
- 2026-09-21: MoE predictor split complete. 3539 lines to six modules, largest 1557. Cleanup removed `_is_grouped_gemm_on_demand_mode`, unreferenced. Unit selection widened again to 183 files: identical on both sides at 28 failed / 3016 passed / 55 skipped / 12 errors. Three tests needed the fake operator family installed in a second module; see `issues.md` I10. The first attempt at this selection aborted at collection and was discarded; see I11.
- 2026-09-21: Independent measurement of `3b00f16`, the manager split, from a detached checkout at that commit: 67/67 identical, 0 mismatched, and a clean predictor cache report with 0 rekeyed models. That is the direct evidence that moving the model hash, the cache keys, the registry and the FFN contract signature into different modules changed no training identity.
