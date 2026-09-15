## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded the baseline environment, compilation, unit, fidelity, and non-dummy CSV smoke evidence. |

# Baseline Test Report

## Execution

- Worktree: `feature-amd-sglang-gdn` at `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Python: 3.12.3.
- Packages: NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, Plotly 6.8.0, PyTorch 2.5.1+cu124; vLLM/SGLang/AITER unavailable.
- Commands:
  - `python -m compileall -q frontier tests`
  - `timeout 300s python -m pytest tests/unit -q -p no:cacheprovider`
  - `python tests/integration/run_scheduler_refactor_fidelity.py --baseline "$PWD" --candidate "$PWD" --output <fresh-dir> --case co-location_offline_dense_model_basic_short --workers 1`
  - `WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 METRICS_OUTPUT_DIR=<isolated> RUN_ID=baseline_dense bash examples/profiling/smoke_simulator_dense_csv.sh`
  - `WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 METRICS_OUTPUT_DIR=<isolated> RUN_ID=baseline_moe bash examples/profiling/smoke_simulator_moe_csv.sh`

## Criteria and evidence

| Check | Expected result | Observed result |
| ----- | --------------- | --------------- |
| Compilation | Exit 0 | PASS |
| Unit baseline | Record all outcomes without relabeling existing failures | 3144 passed, 19 failed, 25 skipped; 576 warnings; exit 1 |
| Dummy self-fidelity | One selected case compares identical worktrees | PASS, `1 passed, 0 failed` |
| Dense CSV smoke | Non-dummy training/loading/simulation completes and writes metrics | PASS; one request completed and metrics artifacts written |
| MoE CSV smoke | Non-dummy training/loading/simulation completes and writes metrics | PASS; one request completed and metrics artifacts written |

The unit failures are pre-existing baseline conditions: missing `frontier.config_optimizer`, missing `tests/debug/e2e-level/monolith_mode` scripts, three analysis builder failures, and stale README/documentation contracts. They are retained as baseline failures and are not used as candidate success criteria.
