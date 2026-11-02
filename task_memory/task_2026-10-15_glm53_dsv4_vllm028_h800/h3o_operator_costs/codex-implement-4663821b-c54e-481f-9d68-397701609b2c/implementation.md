**1. Assignment and outcome**

Implemented B6: DeepSeek-V4 graph-replay attention aggregation with kernel-only scope costs, per-scope signed gaps, and measured layer-window statistics. Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; verified HEAD remains `d2a226fe`. Final acceptance remains with Claude.

The writable scope was limited to the named files under `frontier/profiling/attention/`, `frontier/profiling/kernel_gap/`, `tests/comparison/calibration/`, and `tests/unit/`. All implementation edits stayed within that scope. Generated files, test outputs, and scripts are under `/data/ycfeng/tmp/codex-implement/4663821b-c54e-481f-9d68-397701609b2c/b6`.

Read the applicable repository and parent `AGENTS.md` instructions; no applicable `CLAUDE.md` or nested instructions were found. Inspected the existing row builders, replay-gap helpers, A4 checker, relevant tests, bridge configuration, and lead baseline scripts. Preserved the adopted decisions: observed scope sets define rows; gaps stay signed; planner role participates in grouping; dummy rows use dispatch and metadata; eager formats and `compose_attention_phase` remain unchanged.

No material scope or behavior deviation was needed. Ordinary implementation choices include a `LayerCallTiming` result object, a shared attention-gap index, and validation that recorded per-scope kernel counts agree with owned kernel intervals.

**2. Changes**

All nine implementation-changed files are listed below. The two new files are identified explicitly.

| File | Change and reason |
| --- | --- |
| [dsv4_replay_rows.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsv4_replay_rows.py:106) — new | Added `DSV4_REPLAY_ROW_SCHEMA_VERSION = "dsv4_attention_replay.v1"` and `build_dsv4_replay_rows`. Validates engine policy/mode, unique engine indices, inventory/TP, realized cache groups, routing identity, measurements, layer calls, family scopes, kernel counts, and host-ahead timing. Groups by the assigned key and emits cost, signed-gap, and window statistics. Preserves observed index-score and short-context flags. |
| [dsa_rows.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsa_rows.py:110) | Extracted public `request_load_columns` so replay rows can describe mixed first-tail batches. `batch_load_columns` retains pure-batch context behavior and its mixed-batch error. Generalized `_stat_columns` with a prefix, retaining the existing default time-column names and order. |
| [dsv4_attention_plan.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsv4_attention_plan.py:263) | Added `require_engine_layer_inventory`, shared by replay aggregation and A4. Checks ordered layer IDs, variants, recorded ratios against planned `max(1, ratio)`, and target TP. Retains the native zero-versus-one ratio convention in the docstring. |
| [dsv4_attention_main.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsv4_attention_main.py:164) | Added the graph-replay aggregate branch using `read_replay_records`, the existing missing-engine check, replay schema/capture-mode identity, and the named replay op constant. Writes `attention_graph_replay_kernel_only.csv` and the existing summary JSON. Updated the module docstring. |
| [dsv4_replay_gaps.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/kernel_gap/dsv4_replay_gaps.py:139) | Added `LayerCallTiming`, `attention_gaps_by_layer`, and `layer_call_timing`. Moved the no-owned-kernel and non-positive-target reasons here. The shared function preserves A4’s exclusion order and exact signed-gap/window calculation. |
| [dsv4_composition_check.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/comparison/calibration/dsv4_composition_check.py:61) | Replaced duplicated inventory, gap indexing, admission, window, and scope-gap calculations with the shared helpers. Successful report output remains byte-identical on package 03. Inventory errors now include the source file. |
| [test_dsv4_replay_rows.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_replay_rows.py:86) — new | Added 34 CPU tests covering grouping coordinates, row order, known statistics with a negative gap, single-call composition, dummy layout, mixed first-tail load, observed indexer flags, validation errors, missing-engine handling, and aggregate filename/schema/graph identity. |
| [test_dsv4_replay_gaps.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_replay_gaps.py:373) | Added shared-timing tests for signed scope sums and the window, plus all four exclusion reasons and their precedence, including owned non-kernel activities. |
| [test_dsa_attention_profiling.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsa_attention_profiling.py:522) | Added coverage for mixed request loads, decode-token validation, unchanged pure-batch loads, and the retained context rejection. |

No tracked changes existed at the initial status check. The pre-existing untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/` were left untouched. During execution, changes appeared in `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/stage1a_replay_route.md` and `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/progress.md`. I did not edit either file; exclude those concurrent task-record changes when reviewing this implementation diff.

**3. Verification**

Executed with Python **3.10.6** at `/data/ycfeng/envs/frontier-py310/bin/python`, from the working root. All commands had timeouts. Python checks used:

```bash
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export TMPDIR=/data/ycfeng/tmp/codex-implement/4663821b-c54e-481f-9d68-397701609b2c/b6
ulimit -v 31457280
```

**PASS — required regression suites and new replay tests.** Exact final command:

```bash
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_replay_rows.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_replay_gaps.py tests/unit/test_dsv4_graph_bridge_check.py tests/unit/test_dsv4_kernel_identity_check.py tests/unit/test_dsv4_composition.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsv4_kv_cache_groups.py tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsa_extend_profiling.py tests/unit/test_dsa_full_dummy_profiling.py tests/unit/test_dsa_query_one_profiling.py tests/unit/test_dsa_predictor.py -q -p no:cacheprovider --basetemp "$TMPDIR/regressions" --junitxml "$TMPDIR/regressions.xml" > "$TMPDIR/regressions.log" 2>&1
```

Observed exit 0; output tail: `661 passed, 1 skipped in 20.04s`. Counts inspected from JUnit XML:

| Suite | Passed | Skipped |
| --- | ---: | ---: |
| test_dsv4_replay_rows.py | 34 | 0 |
| test_dsv4_composition_check.py | 13 | 0 |
| test_dsv4_replay_gaps.py | 68 | 0 |
| test_dsv4_graph_bridge_check.py | 18 | 0 |
| test_dsv4_kernel_identity_check.py | 42 | 0 |
| test_dsv4_composition.py | 47 | 0 |
| test_dsv4_graph_replay_profiling.py | 98 | 0 |
| test_dsv4_attention_profiling.py | 78 | 0 |
| test_dsv4_kv_cache_groups.py | 29 | 0 |
| test_dsa_attention_profiling.py | 26 | 0 |
| test_dsa_extend_profiling.py | 6 | 0 |
| test_dsa_full_dummy_profiling.py | 12 | 1 |
| test_dsa_query_one_profiling.py | 119 | 0 |
| test_dsa_predictor.py | 71 | 0 |

The skipped test was `test_dsa_full_dummy_profiling.py::test_locked_native_source_contract`: `Set FRONTIER_VLLM_BS_ROOT to check the pinned native git objects`. Its optional native-source check was not executed.

Initial focused checks exposed a missing local `record` assignment in the A4 refactor, an incorrect string-membership check against family operator objects, and an invalid eager-policy test fixture. These were corrected. Initial logs recorded `5 failed, 174 passed` and `23 failed, 10 passed`; the subsequent focused run recorded `46 passed`, and the final complete run above passed. Logs remain in `focused.log`, `replay-unit.log`, and `focused-fixed.log`.

**PASS — A4 unchanged.** Executed:

```bash
timeout 240s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_baseline.py "$TMPDIR/a4_report.json" > "$TMPDIR/a4.log" 2>&1
timeout 10s cmp /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_report_d2a226fe.json "$TMPDIR/a4_report.json"
```

Both exited 0. A4 output tail: `PASS 125`. `cmp` produced no output, confirming byte-identical reports.

**PASS — real-data aggregate.** Created scratch `raw/dsv4_raw_e0.jsonl` and `raw/dsv4_raw_e1.jsonl` as symlinks to the assigned read-only raw files. Executed:

```bash
timeout 260s /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/run_aggregate.py" > "$TMPDIR/aggregate.log" 2>&1
```

The scratch driver invokes `python -m frontier.profiling.attention.dsv4_attention_main aggregate` using the config’s exact `plan_args`, `graph_mode_flags("FULL_AND_PIECEWISE", cudagraph_capture_sizes)`, and these additional arguments:

```text
--raw_dir <scratch>/raw
--output_root <scratch>/profiling
--device h200
--source_revision 230239c8
--model_config_revision package_03
```

The complete resolved child command is saved in `aggregate-command.txt`, and the driver in `run_aggregate.py`. Exit 0; output tail:

```text
[dsv4] wrote 505 KERNEL_ONLY rows to /data/ycfeng/tmp/codex-implement/4663821b-c54e-481f-9d68-397701609b2c/b6/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv
```

Inspected CSV results exactly match acceptance:

| Coordinate | Observed rows |
| --- | ---: |
| decode | 400 |
| dummy | 50 |
| first_tail | 50 |
| prefill | 5 |
| FULL | 345 |
| PIECEWISE | 150 |
| NONE | 10 |

Sum of `layer_window_stats.count`: **1,925**. All rows carry `schema_version=dsv4_attention_replay.v1`, `graph_mode=FULL_AND_PIECEWISE`, `source_revision=230239c8`, and `model_config_revision=package_03`. Summary JSON is in scratch `raw/dsv4_aggregate_summary.json`.

**PASS — row composition.** Executed the uncommitted scratch script:

```bash
timeout 30s /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/check_row_composition.py" > "$TMPDIR/row_composition.log" 2>&1
```

For every row it composed scope cost medians plus signed-gap medians, using recorded kernel counts, zero class gaps, and the assigned runtime options. Exit 0. Absolute relative errors:

| Statistic | Observed |
| --- | ---: |
| Median | 2.0820011083611275e-16, effectively zero |
| p90 | 0.007676581455487807 = 0.767658% |
| Maximum | 0.031937491401803485 = 3.193749% |
| Single-call rows | 178 |
| Maximum single-call error | 2.220029793664001e-16 |
| Rows above 5% | 0 |

Both A4 limits pass. There are no differing row keys or above-5% rows to report. `row_composition.json` contains the summary and empty above-threshold list; `row_composition_all.json` contains all row keys, metadata, errors, and scope/window spreads. No statistics or grouping keys were tuned to meet the limits.

**PASS — eager/DSA byte preservation.** Before editing, generated CSV baselines from three repeated synthetic steps for each measurement type. After editing, rebuilt them with the same raw records and identity and executed `cmp` for each pair. All four comparisons exited 0:

```text
PASS byte-identical dsa CUDA_EVENT 2 rows
PASS byte-identical dsa KERNEL_ONLY 2 rows
PASS byte-identical eager_v4 CUDA_EVENT 5 rows
PASS byte-identical eager_v4 KERNEL_ONLY 5 rows
```

Raw fixtures and before/after CSVs are in scratch; output is in `preservation.log`. This supplements the full DSA and eager V4 regression suites.

**PASS — cold imports.** In separate fresh Python subprocesses, executed the following code for each required module:

```python
import importlib, sys
importlib.import_module(sys.argv[1])
bad = [name for name in sys.modules if name.split('.')[0] in ('torch', 'vllm', 'cuda')]
assert not bad, bad
print('PASS', sys.argv[1], 'no torch/vllm/cuda modules loaded')
```

Each subprocess had a 10-second timeout. Both exited 0:

```text
PASS frontier.profiling.attention.dsv4_replay_rows no torch/vllm/cuda modules loaded
PASS frontier.profiling.attention.dsv4_attention_main no torch/vllm/cuda modules loaded
```

Output is in `cold-imports.log`.

**FAIL, unchanged known failure — runtime import boundary.** Executed before and after implementation:

```bash
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_profiling_runtime_boundary.py::test_runtime_predictors_use_only_allowlisted_profiling_helpers -q -p no:cacheprovider --basetemp "$TMPDIR/boundary-before" > "$TMPDIR/boundary-before.log" 2>&1
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_profiling_runtime_boundary.py::test_runtime_predictors_use_only_allowlisted_profiling_helpers -q -p no:cacheprovider --basetemp "$TMPDIR/boundary-after" > "$TMPDIR/boundary-after.log" 2>&1
```

Both exited 1 with one failure: `deepseek_v4.py` imports `frontier.profiling.attention.dsa_rows`, which is missing from the test’s allowlist. Compared the logs: the failure text is identical except elapsed time. The predictor and boundary test were not modified.

`test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` was **NOT_RUN**, as the assignment required rerunning the second known failure only.

**PASS — static checks.** `timeout 10s git diff --check` exited 0. Parsed all nine changed Python files with `ast.parse(..., feature_version=(3, 10))` and checked their lengths: all pass Python 3.10 syntax and are below 2,000 lines; the largest is 586 lines.

**4. Remaining work and decisions for Claude**

No assigned implementation blocker remains. All requested CPU and package-03 checks were executed, with the optional native-source skip and expected unchanged boundary failure explicitly recorded.

Observed limitation: the pinned native-source test needs `FRONTIER_VLLM_BS_ROOT` to run. Claude may supply an appropriate pinned checkout and rerun that test if native-source verification is needed for acceptance. The runtime-boundary failure remains a separate existing issue; its unchanged evidence is in the before/after logs.

The predictor, gap reader, trained composition path, and scope-set prediction remain later assignments. No GPU execution, network access, dependency changes, commits, task-record edits, deployment, or publishing were performed.

**5. Review handoff**

Review the nine assigned files, including both untracked new Python files. Inspect the exact grouping key, observed CSA flags, dummy column omissions, shared exclusion order/window formula, and preservation of default statistics columns. The A4 successful report is byte-identical despite the refactor.

For acceptance, rerun the complete pytest command above, the A4 baseline plus `cmp`, `run_aggregate.py`, and `check_row_composition.py`. Use a fresh pytest scratch subdirectory on reruns. The generated [CSV](/data/ycfeng/tmp/codex-implement/4663821b-c54e-481f-9d68-397701609b2c/b6/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv), [row composition report](/data/ycfeng/tmp/codex-implement/4663821b-c54e-481f-9d68-397701609b2c/b6/row_composition.json), [A4 report](/data/ycfeng/tmp/codex-implement/4663821b-c54e-481f-9d68-397701609b2c/b6/a4_report.json), and [JUnit results](/data/ycfeng/tmp/codex-implement/4663821b-c54e-481f-9d68-397701609b2c/b6/regressions.xml) are ready for inspection.
