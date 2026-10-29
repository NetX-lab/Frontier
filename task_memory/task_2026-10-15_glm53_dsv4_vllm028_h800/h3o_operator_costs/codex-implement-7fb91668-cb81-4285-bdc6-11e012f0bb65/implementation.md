**1. Assignment and outcome**

Implemented stage 1a assignment B4 in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, against HEAD `d80af9de2552b0639ffa21106de5a90152ab7470`. The objective was to derive DeepSeek-V4 tile-schedule planner roles from vLLM 0.28 KV-cache groups, select an admissible reduced model, record and validate realized engine groups, and expand reduced kernel identities by full-model layer role. Implementation and CPU checks are finished; final acceptance remains with Claude.

The writable scope was the specifically named source files under `frontier/attention/`, `frontier/execution_time_predictor/`, `frontier/profiling/attention/`, the two calibration scripts, unit tests, and `h3o_operator_costs/p1h_bridge/package_plan.json`. All assignment edits stayed within that scope. Scratch scripts, archived HEAD files, caches, test temporary files, and outputs are under:

`/data/ycfeng/tmp/codex-implement/7fb91668-cb81-4285-bdc6-11e012f0bb65/b4/`

Read the applicable AGENTS.md guidance, the B3 assignment and lead report, and the pinned vLLM grouping source. The pinned source agrees with the supplied grouping rule. The implementation excludes MTP layers, requires layouts containing both CSA and HCA, uses Python 3.10 syntax, imports GPU packages only inside existing runtime functions, and adds no predictor cache.

The default reduced layout is 11 layers. Layers 0, 1, 2, and 3 plan; layers 4–10 reuse. Full-model role counts are uncompressed planner 2, CSA planner 1, CSA reuse 20, HCA planner 1, and HCA reuse 19.

Pre-existing work encountered: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/progress.md` already had an edit at session start, in addition to untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/`. These were not edited. No commits, task-record updates, package generation, GPU operations, network operations, dependency changes, deletions, or moves were performed.

**2. Changes**

Paths below are relative to the working root. All files are existing files except the new grouping test.

| File | Relevant lines/symbols | Change and reason | Final lines |
| --- | --- | --- | ---: |
| `frontier/attention/deepseek_v4.py` | Constants at 19; `_approximate_gcd` at 41; public grouping API at 61; planners at 107 | Added named cache constants, the per-variant cache table, metadata-source mapping, faithful minimum-padding GCD port, decoder KV-cache groups, and SWA-group planner roles. Documents pinned sources, page-size assumptions, native confirmation, and MTP exclusion. | 463 |
| `frontier/execution_time_predictor/deepseek_v4.py` | `read_isolated_scope`, 780 | Computes full decoder-layer planner roles once per call and selects `planners[layer_id]`. Other column uses and decode identity filters remain as instructed. | 965 |
| `frontier/profiling/attention/dsv4_attention_plan.py` | `summary`, 245; realized-group check, 263; admission checks, 279; `resolve_dsv4_layers`, 333 | Replaced the two-layers-per-variant rule with all four admission conditions. Default selects the fewest admissible leading layers. Summary includes predicted groups. Explicit six layers reports SWA group count 3 versus 2 and the groups. Added the shared realized-group validator. | 563 |
| `frontier/profiling/attention/dsv4_attention_runner.py` | `attention_inventory`, 303 | Maps realized `kv_cache_config.kv_cache_groups[].layer_names` through discovered module prefixes. Records sorted members in vLLM group order and rejects unknown names. Both existing eager and graph hook consumers inherit this inventory change. | 529 |
| `frontier/profiling/attention/dsv4_attention_rows.py` | `build_dsv4_rows`, 213 | Validates every indexed engine’s realized groups after the existing scope-timing guard. Updated role description. | 311 |
| `tests/comparison/calibration/dsv4_composition_check.py` | `planner_roles`, 61 | Retains ordered layer and TP checks; adds realized-group validation per run before returning rebuilt-plan roles. | 218 |
| `tests/comparison/calibration/dsv4_kernel_identity_check.py` | `layer_role_counts`, 110; expansion, 121; report, 224 | Replaced variant counts with `(variant, plans_tile_schedule)` counts. Requires every full role, rejects extra reduced roles, checks identical multisets across every reduced layer of a role, and expands one multiset by the native count. Report uses `layer_role_counts` entries with `variant`, `plans_tile_schedule`, and `count`. Updated exports and docstring. | 258 |
| `tests/unit/test_dsv4_kv_cache_groups.py` **new** | Worked layouts, 31; native groups, 53 | Tests every worked layout, all group members, planner layers, native written-out groups and role counts, GCD ties/bounds/error cases, and invalid variants/layouts. | 98 |
| `tests/unit/test_dsv4_attention_profiling.py` | Default layout, 245; admission, 272; group rejection, 1115; unknown cache, 1135; additional admission conditions, 1147 | Updated 11-layer overrides, five row roles, aggregation sample counts, engine records, and fake realized groups. Added reordered-group acceptance, missing/different/duplicate/member-order rejection, unknown-name rejection, and separate SWA variant-set and role-coverage tests. | 1157 |
| `tests/unit/test_dsv4_graph_measured.py` | Historical table check, 446; full-model role selection, 806 | Adds competing-row tests proving layer 1 selects planner True and layer 4 selects False. Historical warmed q1 rows remain immutable; their obsolete layer-1 reuse label is now explicitly rejected, while valid rows retain exact-scope checks. | 820 |
| `tests/unit/test_dsv4_graph_replay_profiling.py` | Fake cache config, 288; inventory assertion, 321 | Supplies realized KV-cache groups to graph-runner fakes and verifies the recorded groups. | 979 |
| `tests/unit/test_dsv4_composition_check.py` | Planned-record fixtures; realized-group tests, 205 | Uses admissible planned inventories and groups in mocks; covers passing, missing, and different groups while retaining ordered-layer, TP, and engine-key checks. | 217 |
| `tests/unit/test_dsv4_kernel_identity_check.py` | Role expansion, 88 | Covers several reduced layers per role, observable expansion factors 2/1/20/1/19, unequal multisets, missing kernels/roles, extra roles, and the report key. | 173 |
| `tests/unit/test_dsv4_pilot_cases.py` | Package plan rebuild, 50 | Replaces explicit six layers with the default and checks 11 layers, 25 cases, phase counts, and estimated steps. | 206 |
| `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/package_plan.json` | `plan_args`, 18 | Removed only `"--num_layers", "6"`. All other fields, including `source_revision`, are unchanged. | 131 |

Public APIs added: `deepseek_v4_kv_cache_groups(variant_ids)`, `deepseek_v4_tile_schedule_planners(variant_ids)`, and `require_realized_kv_cache_groups(plan, engine_record)`. The identity script’s replacement API is `layer_role_counts(config)`.

`dsv4_graph_replay_runner.py` did not need a source edit because it already calls `attention_inventory`.

**3. Verification**

All Python checks used `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6) on CPU. Test commands ran from the working root in bash with:

```bash
ulimit -v 31457280
export PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/data/ycfeng/tmp/codex-implement/7fb91668-cb81-4285-bdc6-11e012f0bb65/b4
export FRONTIER_TMP_ROOT=$TMPDIR
export OPENBLAS_NUM_THREADS=1
```

Executed the acceptance groups through `timeout 600 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/run_groups.py"`. Each group executed `timeout 240 /data/ycfeng/envs/frontier-py310/bin/python -m pytest FILES -q -p no:cacheprovider --basetemp "$TMPDIR/pytest-group-N-1"`, with the exact file arguments below. After two admission tests and the final identity-report field change, only groups 3 and 7 were rerun using `run_final_changed_groups.py`, `timeout 120`, and `pytest-group-N-final` temporary directories. Exact command arrays and actual tails are retained in `group-results.json` and `group-results-final.json`.

| Group | Exact pytest file arguments | Expected | Observed final result | Status |
| --- | --- | --- | --- | --- |
| New | `tests/unit/test_dsv4_kv_cache_groups.py` | All pass | 29 passed | PASS |
| 1 | `tests/unit/test_dsv4_replay_gaps.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_graph_bridge_check.py tests/unit/test_graph_capture.py tests/unit/test_graph_replay_trace.py` | All pass | 264 passed | PASS |
| 2 | `tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsv4_composition.py` | All pass | 72 passed | PASS |
| 3 | `tests/unit/test_dsa_query_one_profiling.py tests/unit/test_dsv4_attention_profiling.py` | All pass | 197 passed | PASS |
| 4 | `tests/unit/test_dsa_extend_profiling.py` | All pass | 6 passed | PASS |
| 5 | `tests/unit/test_kernel_gap_pricing.py tests/unit/test_kernel_gap_profiler.py` | Only documented failure | 29 passed, 1 failed | FAIL: documented baseline failure only |
| 6 | `tests/unit/test_dsa_full_dummy_profiling.py tests/unit/test_dsv4_graph_measured.py tests/unit/test_profiling_runtime_boundary.py` | Only documented failure | 135 passed, 76 skipped, 1 failed | FAIL: documented baseline failure only |
| 7 | `tests/unit/test_dsv4_pilot_cases.py tests/unit/test_dsv4_kernel_identity_check.py` | All pass | 71 passed | PASS |
| 8 | `tests/unit/test_dsv4_attention_workload.py tests/unit/test_dsv4_cache_layout.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_structural_pd.py tests/unit/test_dsv4_dp_sync.py tests/unit/test_dsv4_model_config.py tests/unit/test_dsv4_step_cost_input.py tests/unit/test_dsv4_operator_manifest.py tests/unit/test_dsv4_cache_zero_driver.py tests/unit/test_dsv4_cache_zero_execution.py` | All pass | 331 passed | PASS |

Final totals, without counting repeated checks: **1,134 passed, 76 skipped, 2 failed**. The two failures are exactly the lead’s documented baseline failures:

- `test_every_collective_launches_one_kernel_by_default`: `AttributeError: 'OperatorSpec' object has no attribute 'kernel_count'` at `test_kernel_gap_pricing.py:333`.
- `test_runtime_predictors_use_only_allowlisted_profiling_helpers`: `AssertionError: deepseek_v4.py`; extra profiling import `frontier.profiling.attention.dsa_rows`, at `test_profiling_runtime_boundary.py:59`.

Actual final output tails include:

```text
29 passed in 0.36s
264 passed in 3.32s
72 passed in 1.88s
197 passed in 2.18s
6 passed in 0.37s
1 failed, 29 passed in 1.83s
1 failed, 135 passed, 76 skipped in 65.40s (0:01:05)
71 passed in 1.80s
331 passed in 18.03s
```

Initial targeted verification reported 25 failures: one stale aggregation-count expectation and 24 historical layer-1 reuse expectations. The count now accounts for layers sampled per role. The historical tests now verify rejection of the incorrect layer-1 role without changing the measured files or relabeling their costs. Subsequent acceptance checks have no new failures. Initial evidence remains in `targeted.log`.

**Native inventory and cold imports — PASS.** Executed:

```bash
timeout 60 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/native_and_imports.py"
```

Mapped every native `layer_names` entry through the specified suffixes and compared groups, ignoring only outer group order, against the 43 decoder variants from `package_02/model/config.json`. Input was `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-canonical-20261025-01/artifacts/nodes/gpu-h200-0082.lgcm.sh.istep.fun/serving/rank0.json`.

Actual output:

```text
PASS native: 43 decoder layers, 5 groups, members=[62, 22, 21, 42, 20], equal ignoring group order
```

The script then imported all eight requested modules and checked `sys.modules` after every import. All passed with no `torch`, `vllm`, or `cuda` modules loaded. Final output is in `native_and_imports-final.log`.

**HEAD plan comparisons — PASS with a reported acceptance-wording discrepancy.** Executed the requested archive:

```bash
git archive HEAD frontier data/config | tar -x -C /data/ycfeng/tmp/codex-implement/7fb91668-cb81-4285-bdc6-11e012f0bb65/b4/head
```

Then executed `timeout 180 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/plan_checks.py"`. The script records every exact command, cwd, namespace override, stdout, and JSON output. `plan_entry.py` invokes each module’s parser and `_plan_command`; it suppresses warning timestamps for deterministic byte comparisons, following the lead’s B3 approach.

DSA configurations used GLM-5.3, TP8, block64: the existing 4160-context grid with budgets 1024/256, and native-defaults with context/capacity7168, budget8192, prefill6144, and q1 context6143–7166. **Both stdout and output JSON were byte-identical to HEAD.**

V4 configurations used `p1h_bridge/package_02/model/config.json`, TP8, block256, fp8, with the corresponding grid and native-defaults coordinates. Complete differing JSON field paths in both comparisons:

```text
hf_overrides.compress_ratios
hf_overrides.num_hidden_layers
shape.layers
```

Within `shape.layers`, layer 1 changes `plans_tile_schedule` from False to True and layers 6–10 are appended as reuse layers. Overrides grow from 6 to 11 decoder layers and ratios.

Complete differing stdout-summary paths in both comparisons:

```text
hf_overrides.compress_ratios
hf_overrides.num_hidden_layers
kv_cache_groups
reduced_layers
```

The CLI output acknowledgement is unchanged. The additional summary override differences are necessary because the existing inherited summary already serializes `hf_overrides`; see section 4. The literal two-field stdout expectation was observed failing before the comparison assertion was updated to the truthful four-field result. Evidence is retained in `plan_checks-summary-requirement-failure.log`.

V4’s existing CLI parser excludes `native_defaults`. The direct-CLI attempt exited 2. The final comparison uses `FRONTIER_PLAN_NATIVE=1` to set the parsed namespace to that policy, exactly as the lead’s B3 harness did. The parser was not changed. Evidence is in `plan_checks-native-cli-failure.log`; final successful comparisons are in `plan_checks.log` and the `dsa-*` / `v4-*` JSON and stdout files.

**Package 03 plan — PASS.** The same script rebuilt `package_plan.json` arguments with the package_02 model, `pilot_cases.json`, FULL_AND_PIECEWISE, and all 35 capture sizes. Actual summary:

```text
11 reduced layers; planner layers [0, 1, 2, 3]
25 cases: {prefill: 1, decode: 10, first_tail: 4, dummy: 10}
engine estimated steps [280, 55], total 335
5 groups, member counts [14, 6, 5, 10, 4]
```

Artifacts: `package03-work.json`, `package03-work.stdout`, and `package03-output.json`. This was planning only; no package_03 directory was generated.

**File and scope audit — PASS.** Executed `wc -l` on every changed/new file; counts are in section 2. Executed `timeout 60 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/file_audit.py"` and `git diff --check`. The audit confirmed every changed Python file parses with Python 3.10 grammar, all changed/new files are below 2,000 lines, package-plan JSON differs only by removing the two arguments, and tracked changes consist of the explicit B4 files plus the pre-existing progress edit. Output: `file_audit.log`.

**4. Remaining work and decisions for Claude**

No implementation prerequisite is blocked. The following review decisions and limitations remain:

- **Observed acceptance-wording discrepancy:** `DsaProfilePlan.summary()` at `frontier/profiling/attention/dsa_plan.py:330` includes `hf_overrides`, and the V4 summary inherits it. Therefore the required 6→11 override changes also appear in stdout. Recommendation: accept and record the four summary paths listed above. Keeping the old overrides in stdout would report an incorrect plan; removing the field would change the established summary schema.
- **Observed existing planning interface limitation:** V4’s parser permits eager and graph-replay runners, not `native_defaults`. Native-defaults comparison succeeded through the established namespace override. Recommendation: retain that harness convention for this assignment; a public parser change would need separate scope.
- **Observed historical input incompatibility:** the immutable warmed q1 table’s layer-1 row is labeled reuse, while the corrected full model requires planner True. Tests now reject that observation. Recommendation: do not relabel the measured row; obtain corrected measurement evidence before treating that row as a native layer-1 cost.
- **Observed unchanged decode filter:** `read_isolated_scope` at lines 800–801 still requires a singleton `global_layer_ids` and `num_layers_sampled=1`, as instructed. Inference: newly aggregated eager rows spanning several layers of one role require separate qualification before use through that exact-observation decoder path. This does not block the graph-replay package plan, whose row aggregation remains guarded off.
- **NOT_RUN:** realized-group collection against a real vLLM engine, planner kernel counts on GPU, and package 03 execution. CPU fakes exercise both hook consumers and unknown-name rejection; the native saved inventory independently confirms the grouping rule, but does not execute the new hook.
- Claude owns review, final acceptance, commits, source_revision updates, task records, package generation, and GPU execution. The two baseline failures remain outside this assignment.

An existing comment in `dsv4_attention_main.py` still describes the old two-layers-per-variant default. That file is outside the explicit B4 edit list; Claude can update the comment during review if desired.

**5. Review handoff**

Inspect the grouping port against pinned `kv_cache_utils.py`, especially tuple counts, lower bound, tie preference, and stride splitting. Inspect the four admission conditions and the shared realized-group validator’s outer-order-insensitive comparison with exact ordered member lists. Review inventory prefix mapping and the identity expansion checks for every reduced layer of a role.

Review the historical-table test adjustment as an explicit rejection of obsolete evidence, and verify that `package_plan.json` has only the two argument removals. The new test file is untracked and must be included with the implementation changes; preserve the pre-existing progress edit and untracked directories separately.

Rerun `run_groups.py`, `run_final_changed_groups.py`, `native_and_imports.py`, `plan_checks.py`, and `file_audit.py` with the documented CPU environment. Exact executed test commands and output tails are available in the scratch command-result JSON files. For subsequent GPU validation, first confirm recorded realized groups match the 11-layer prediction, then verify planner kernels occur only at layers 0–3 and compare identities by the five full-model roles.
