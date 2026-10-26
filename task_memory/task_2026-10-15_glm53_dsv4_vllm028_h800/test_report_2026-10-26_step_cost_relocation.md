# Test report: step-cost tables moved out of the default profiling path (D-NO-STEP-COST-SHORTCUT)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: method correction ordered by YC, its checks and results. |

## 1. Finding and change

`frontier/execution_time_predictor/dsv4_step_cost_input.py` `load_step_cost_table` binds `dsv4_step_cost.json` when it is found beside the resolved CUDA_EVENT attention input. The H800 and H200 DP2 tables were in `data/profiling/compute/{h800,h200}/deepseek-ai/DeepSeek-V4-Flash/`, Frontier's default profiling input path. As a result, every non-dummy V4 run was priced by whole-step lookup and skipped operator model training, without any explicit choice. That contradicts D-NO-STEP-COST-SHORTCUT (requirements item 26), which limits the table to diagnostic use.

Change:

- **Tables copied, then removed from `data/profiling/`.** Both tables are now byte-identical copies in task directories, and their `data/profiling/` copies are removed. The P2-64 DP4 table was already in a task directory.

  | Table | SHA-256 | New location |
  | --- | --- | --- |
  | H800 P0 (DP2/EP16) | `ea817ce52d5ec8a67eeeec2079eb44c206039ce04608e75ebe9d26f603942f8c` | `h4a_p0_diagnostic/step_cost/dsv4_step_cost.json` |
  | H200 DP2/EP16 | `cdf7d1162bab16280c842fc45c93e88d6478e7f4c83db86f66db5bc067ce850d` | `h200_mainline/diagnostic/step_cost_attn_dp2/dsv4_step_cost.json` |
  | H200 DP4/EP32 (unchanged) | `b452351a7945aac9af36ec1841bd81add8fdafccdff1acbbb0991f4c501fac9d` | `h200_mainline/diagnostic/step_cost_attn_dp4/dsv4_step_cost.json` |

- **Drivers select the table explicitly.** `h4a_p0_diagnostic/h4a4_driver.py` selects each case's table through `random_forrest_execution_time_predictor_config_atten_input_file`, as the P2-64 case and the k=1 runs already did. The H4a-2 probe `h4a_p0_diagnostic/probes/probe_v4_pd_step_cost.py` does the same in measured mode.
- **Unit test.** `tests/unit/test_dsv4_step_cost_input.py` now uses a synthetic table in a temporary directory, and it asserts that the default profiling path binds no table.
- **Text only.** The reader's docstring and the V4 `runtime_limitations` text in `frontier/model_architectures.py` state the explicit selection.
- **Repository record.** `docs/adr/0002-operator-costs-not-measured-step-tables.md`.

## 2. Checks

Environment: Python 3.10.6 (`/data/ycfeng/envs/frontier-py310/bin/python`), `PYTHONPATH` set to the repository root, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. Scratch directory: `/data/ycfeng/tmp/glm53_dsv4/no_shortcut/`. "Before" is a `git archive` snapshot of HEAD `6fcb2cf5`, taken before the change.

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `sha256sum` of the copies | equal to the `data/profiling/` originals | `ea817ce5…` and `cdf7d116…`, both equal | PASS |
| 2 | H4a-2 probe `measured`, before (snapshot, default path) | runs; the table is bound from `data/profiling/` | exit 0 | PASS |
| 3 | Original probe `measured` after the change (default path, no explicit table) | stops with a clear error and does not price from a table | exit 1: `NotImplementedError: Attention family deepseek_v4_attention is not enabled for execution: DeepSeek-V4 execution requires compressed cache, compressor, indexer, mHC and mixed-precision operator coverage.` | PASS |
| 4 | Updated probe `measured` after the change (explicit table) against run 2 | every metrics file byte-identical | `request_metrics.csv`, `system_metrics.json`, `metrics_ground_truth.jsonl`, the transfer, stage-batch and EP-wave ledgers and `op_precision_metadata.csv` are identical. `config.json` differs only in the output, trace and attention-input paths | PASS |
| 5 | `h4a4_driver.build_config` and `step_cost_table` for native06, p2_32 and p2_64 | each case resolves to its relocated table with the recorded digest | `ea817ce5`, `cdf7d116` and `b452351a`, each OK; the attention input field is set to the table's directory | PASS |
| 6 | `pytest tests/unit/test_dsv4_step_cost_input.py` | 2 passed | 2 passed (before the change: 2 passed with the old test) | PASS |
| 7 | `pytest` on the 20 unit files whose names contain `dsv4`, `v4_` or `model_architecture` | no new failure | 734 passed, 83 skipped, 1 failed: `test_v4_trace_audit_interval_metrics.py::test_pins_and_caps`. It also fails on the HEAD snapshot (environment pin of `PYTHONPATH`); pre-existing and unrelated | PASS |

## 3. Limits

- **Not rerun.** No full P0, P2-32 or P2-64 diagnostic replay was rerun. Check 4 shows identical outputs for the same loader reading the same bytes from the new path, and check 5 shows that the driver resolves the recorded tables.
- **Old receipts.** Receipts of earlier runs keep the old `data/profiling/` path. Their SHA-256 identifies the relocated copy. `h4a5_attribution.py` would need that copy if one of those runs were attributed again.
- **Not run.** The 77-case fidelity regression and the dummy smoke matrix were not run. The only `frontier/` changes are a docstring and a warning string for the V4 profile, and the GLM, co-location, PDD and PD-AF paths do not read either.
