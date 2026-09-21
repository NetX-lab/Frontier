# Test Report 2026-09-21 — Step 0 Baseline (`1f694f7`)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Initial report. |

## Environment

| Field | Value |
| --- | --- |
| Host | `kun-workspace-vgen2` (CPU only) |
| Worktree | `/data/ycfeng/Frontier/.worktrees/oversized-module-split` at `1f694f7c549aa3aeeb7c5bbae04e119c09167a77` (before any source change) |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python`, CPython 3.10.6 (uv venv; `uv pip install -e ".[test]"`) |
| Packages | numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, plotly 7.1.0, pytest 9.1.1 |
| Import path | `frontier` resolved from the worktree (`frontier/main.py` path verified); `frontier/` is a namespace package (no `__init__.py`) |
| Env vars | `PYTHONPATH=<worktree>`, `FRONTIER_TMP_ROOT=/data/ycfeng/tmp/issue26-correctness-pr`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1` |
| Raw logs | `/data/ycfeng/tmp/issue26-correctness-pr/step0/{pytest_baseline.log,coloc_dense.log,pdd_dense.log}` (scratch, not committed) |

## 1. Unit selection

Command:

```bash
python -m pytest tests/unit/test_cluster_scheduler_dp_lanes.py \
  tests/unit/test_colocation_release_review_contracts.py \
  tests/unit/test_config_owned_contracts.py \
  tests/unit/test_stage_execution_time.py \
  tests/unit/test_stage_finalized_contract.py \
  tests/unit/test_moe_routing_runtime.py -q -p no:cacheprovider
```

Expected: collection succeeds; failures, if any, are attributable to the unmodified base or the environment.

Actual: `10 failed, 84 passed in 3.42s`, exit code 1. **Result: PASS as a baseline record** (all failures are pre-existing on main or environmental; none involve the modules in scope).

| Failing test (`tests/unit/test_colocation_release_review_contracts.py`) | Cause (observed) | Class |
| --- | --- | --- |
| `test_config_optimizer_help_prints_without_argparse_percent_crash` | `FileNotFoundError: 'python'`: the test spawns the bare `python` executable, which is not on this host's PATH | environment |
| `test_debug_e2e_base_does_not_default_to_private_conda_path` | `tests/debug/e2e-level/monolith_mode/scripts/test_base.sh` does not exist on main (`git ls-files tests/debug` is empty) | pre-existing on main |
| `test_debug_e2e_conda_activation_temporarily_disables_nounset` | same missing `test_base.sh` | pre-existing on main |
| `test_debug_e2e_base_uses_safe_pythonpath_expansion` | same missing `test_base.sh` | pre-existing on main |
| `test_readme_debug_scripts_do_not_use_post_increment_under_set_e` | missing `test_dense_tp2_pp2_dummy.sh` | pre-existing on main |
| `test_debug_e2e_base_resolves_latest_canonical_metrics_run_dir` | missing `test_base.sh` (bash exit 127) | pre-existing on main |
| `test_release_debug_scripts_use_canonical_metrics_resolver` | missing `test_dense_tp2_pp2_dummy.sh` | pre-existing on main |
| `test_readme_moe_debug_script_uses_valid_shared_parallel_domain` | missing `test_moe_tp2_ep2_pp2_dummy.sh` | pre-existing on main |
| `test_readme_moe_debug_script_header_uses_current_tp_terms` | missing `test_moe_tp2_ep2_pp2_dummy.sh` | pre-existing on main |
| `test_readme_moe_debug_script_satisfies_shared_parallel_domain` | missing `test_moe_tp2_ep2_pp2_dummy.sh` | pre-existing on main |

Observation: `AGENTS.md` (§Tests) still tells readers to run the two missing debug scripts and `tests/unit/test_open_source_release_arch_guard.py`, which is also absent. Recorded as documentation drift; not fixed by either PR.

Per-file result: `test_cluster_scheduler_dp_lanes.py`, `test_config_owned_contracts.py`, `test_stage_execution_time.py`, `test_stage_finalized_contract.py`, `test_moe_routing_runtime.py` all passed; the 10 failures are confined to `test_colocation_release_review_contracts.py`.

## 2. Frontier-only CPU smokes (dummy predictor, analytical backend)

| Script | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Co-location dense offline | `PYTHON_BIN=$PY METRICS_OUTPUT_DIR=$FRONTIER_TMP_ROOT/step0/coloc_dense bash examples/architecture/co-location/offline/dense_model_basic.sh` | exit 0, `request_metrics.csv` and `system_metrics.json` written | exit 0; `request_metrics.csv` 16 rows; `system_metrics.json` sections `simulation_metadata, quantization_config, ttft_statistics, tpot_statistics, request_e2e_time_statistics, throughput_metrics, spec_decode_statistics, preemption_statistics, ...` | PASS |
| PDD dense offline (sequential) | `PYTHON_BIN=$PY METRICS_OUTPUT_DIR=$FRONTIER_TMP_ROOT/step0/pdd_dense bash examples/architecture/pdd/offline/dense_model_basic.sh` | exit 0, same artifacts | exit 0; `request_metrics.csv` 8 rows; sections include `kv_cache_transfer_statistics` | PASS |

Outputs were redirected to the scratch root; the worktree shows no generated files (`git status` lists only `.gitignore` and `task_memory/`).

## 3. Reference checkout

`fwyc0573/vLLM-BS` cloned to `/data/ycfeng/Frontier/.real-engine/vLLM-BS` (`.git/info/exclude`), detached at `ea95f571e20937c7c908c6d59ddd1cd6bf9268f1` ("Instrument vLLM attention ops for Frontier calibration"), contained in `origin/feature/frontier-comparison-instrumentation`. The fork exposes **no tags**, so the relationship to upstream `v0.10.2` could not be established from tags; Step 1 of the correctness task must add the upstream remote read-only and compare against the resolved `v0.10.2` commit, as the specification requires.

## Limits

- Dummy-mode smokes verify lifecycle and artifact production only, not numerical parity.
- The unit selection is a baseline record; the 10 failures were not reproduced with a different Python to separate PATH effects further, because their assertion messages already name the missing files or executable.
