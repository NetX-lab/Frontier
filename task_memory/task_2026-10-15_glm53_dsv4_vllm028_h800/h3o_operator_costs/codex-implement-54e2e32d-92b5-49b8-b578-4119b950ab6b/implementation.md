**1. Assignment and outcome**

B16 requests the H200 V4 operator consumer: table promotion, one trainer used by both training paths, trained operator pricing, composed attention timing, segment-specific kernel gaps, KERNEL_ONLY selection, zero host time, and typed runner coverage failures.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Outcome: **partial**. Table promotion, reproducibility checks, a V4 gap precision fix, and independent CPU validation are finished. The common trainer and runtime consumer are not implemented. Final acceptance remains with Claude.

The assignment's named writable scope covered the specified predictor files and new neighboring modules, unit tests, `expert_parallel.py`, narrowly permitted metrics/schema changes, the new H200 data directory, and `h3o_operator_costs/h200_tables/`. Scratch was `/data/ycfeng/tmp/codex-implement/54e2e32d-92b5-49b8-b578-4119b950ab6b/b16/`. No Git mutations, package installation, network/GPU operations, delegation, or task-record updates were performed.

The applicable AGENTS.md, requirements, plan, issues, design, C1 coverage, and B11/B12/B13 reports were inspected. The explicit decisions to exclude EP32 shared-expert repeats and CUDA_EVENT tables from promotion were preserved. No alternative TP algebra, row-axis rule, routing-class implementation, contract bypass, or missing-row fallback was introduced.

**Observed prerequisites that stopped the affected implementation**

A CPU probe reproduced these failures without monkeypatching guards:

- `ModelArchitectureProfile.resolve_layer_contract(..., operator_name="dsv4_mhc_first_pre")` and the equivalent embedding query raise `NotImplementedError: Operator family deepseek_v4_mhc is not enabled for execution: DeepSeek-V4 mHC and endpoint scopes are source inventory only`. The family is disabled at `frontier/operators/families.py:314`.
- Independently, `profile.get_layer_contract_for_family("deepseek_v4_mhc")` raises `ValueError: Profile 'deepseek_v4' declares 0 contracts for family 'deepseek_v4_mhc'`. The resolver requires exactly one contract at `frontier/model_architectures.py:768`. Enabling the family flag alone would therefore be insufficient.
- A real MONOLITHIC, non-dummy H200 TP8/DP2/EP16 predictor, pointing at the promoted tables and with `skip_cpu_overhead_modeling=True`, fails at `base_execution_time_predictor.py:46`, before training. The error is `Attention family deepseek_v4_attention is not enabled for execution: DeepSeek-V4 execution requires compressed cache, compressor, indexer, mHC and mixed-precision operator coverage.` The base constructor passes `structural=False` for this configuration; `frontier/attention/families.py:652` disables measured execution. The probe separately confirmed that no step-cost table was bound.

The operator/profile files, attention-family file, and base-predictor file are outside the assignment's explicitly named writable files. Following the instruction to stop the affected part when the canonical contracts contradict the required loader, runtime integration was left pending. Evidence, including the constructor traceback, is in [prerequisites.json](/data/ycfeng/tmp/codex-implement/54e2e32d-92b5-49b8-b578-4119b950ab6b/b16/prerequisites.json).

**2. Changes**

Every changed or new repository file is listed below. There were no deletions.

| Path relative to working root | Change and purpose |
|---|---|
| `frontier/execution_time_predictor/kernel_gap.py` | Modified `read_dsv4_kernel_gap_ms` and `_read_gap_rows_ms` at lines 47–55. V4 uses pandas `float_precision="round_trip"`, preserving every recorded gap digit. The generic reader retains its original parser default. No V4 loading dispatch or runtime gap pricing was added. |
| `tests/unit/test_dsv4_h200_tables.py` | New, 13 parametrized cases. Checks promotion root resolution, lossless deterministic CSV union, non-executing revision extraction, four output hashes/counts, KERNEL_ONLY-only inputs, exact gap values, and invalid gap rows. |
| `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_tables.py` | New. `promote_tables` at line 57 invokes the existing attention and gap reducers on scratch copies of raw records, unions the accepted standalone/shared and EP16/EP32 MoE CSVs, and records hashes/counts/revisions. `union_csv` preserves serialized measurement values and leaves absent columns blank. `recorded_model_revision` reads the staged launcher's constant through AST without executing it. |
| `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/manifest.json` | New deterministic provenance manifest: 12 input paths and SHA-256 values, four output paths/hashes/counts, and source/model/native revisions. |
| `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv` | New producer-generated attention replay table, 4,945 rows. |
| `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/kernel_gap.csv` | New producer-generated four-segment gap table, four rows. |
| `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/linear_op_kernel_only.csv` | New lossless union: 36 standalone rows plus five EP16 shared-expert rows, 41 rows total. EP32 repeats are excluded. |
| `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/moe_kernel_only.csv` | New lossless union: 13 EP16 rows plus 20 EP32 rows, 33 rows total. |

The existing producer emits `attention_graph_replay_kernel_only.csv`. That filename was preserved; the existing V4 input-path resolver recognizes it. This differs from some filename wording in the brief, without changing the producer or input-path mechanism.

The manifest is [manifest.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/manifest.json). Its SHA-256 is `2bd2cad820279935450d6e4e810dc63e98fd840a5766c728b432bf56774b32b7`.

| Output | Rows | SHA-256 |
|---|---:|---|
| `attention_graph_replay_kernel_only.csv` | 4945 | `34c4f3a86822994b126d195a54fce934e208fd3ba9dff2cc4fddb560c129321f` |
| `kernel_gap.csv` | 4 | `d890b67a2a2b975f8a72659a7febd1a82a4b8133c01895d13ad352eaa1f4cb23` |
| `linear_op_kernel_only.csv` | 41 | `ec8a552aa672dc6b4faae45bb0633343f69bdffea8e1baa5fdd6269c856817ee` |
| `moe_kernel_only.csv` | 33 | `b522f687bd38d3404d980d43704b3a2041caad8f96ce033b76c7cbeef9fc4d2f` |

Recorded revisions:

- Attention source: `a689ed05f462eb269b9dffe640da151e6614a5ef`.
- Standalone and shared-expert source: `7753e5a951710d93c1426fc699614ff3d2cff5cc`.
- MoE source: `32d3d08b466a16c70a55f6f44359bd20106c91ca`.
- Attention model-config revision: `082a2aa294ad5f4056895c42b297b4af24703384`.
- Native revision: `2cf0a6915ce544dc493a0990f2ea38d81601128a`.

Pre-existing `.cortexkit/`, `.pi/`, and `remaining_measurements/` were preserved. A pre-existing task `progress.md` change was observed earlier in the session and was not edited; it was absent from the final status. The new promotion script and manifest are ignored by `.gitignore:171` (`task_memory/*`), so ordinary `git status` does not list them. Claude must account for those files when preparing a commit.

`read_isolated_scope` and `_read_isolated_component` remain. Their deletion and test migration depend on completing the canonical loader.

**3. Verification**

All CPU commands used Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), from the working root, with `ulimit -v 31457280` and a timeout. The common environment was:

```bash
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
WANDB_DISABLED=true
VIDUR_DISABLE_WANDB=1
OPENBLAS_NUM_THREADS=1
```

The regression and smoke commands also used `OMP_NUM_THREADS=1`. Below, `PY` denotes the interpreter above, `S` denotes the absolute scratch directory above, and `P` denotes `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables`. Commands were executed with the common environment through `env`; output redirections point to the named scratch logs.

| Check | Command arguments | Expected | Observed |
|---|---|---|---|
| First completed promotion | `timeout 600s env … "$PY" "$P/promote_tables.py" > "$S/promote_first_completed.log" 2>&1` | Four promoted tables and manifest | **PASS**, exit 0; counts/hashes above. |
| Second promotion | `timeout 600s env … "$PY" "$P/promote_tables.py" > "$S/promote_second.log" 2>&1` | Identical outputs | **PASS**, exit 0. Validation compared manifest bytes with `manifest_first.json` and rehashed all inputs/outputs. |
| Asset validation | `timeout 120s env … "$PY" "$S/validate_assets.py" > "$S/asset_validation.log" 2>&1` | Determinism, identity, contract and rank evidence | **PASS**, exit 0. Details below. |
| Focused unit checks | `timeout 60s env … "$PY" -m pytest tests/unit/test_dsv4_h200_tables.py tests/unit/test_kernel_gap_pricing.py -q -p no:cacheprovider --basetemp="$S/focused_after" -k 'not test_every_collective_launches_one_kernel_by_default' > "$S/focused_after.log" 2>&1` | All selected tests pass | **PASS: 33 passed, 1 deselected**, 3.21 s. |
| Import regression | `timeout 300s env … "$PY" -m pytest tests/unit/test_dsv4_h200_tables.py tests/unit/test_dsv4_runner_operators.py tests/unit/test_kernel_gap_pricing.py tests/unit/test_dsv4_replay_rows.py tests/unit/test_dsv4_replay_gaps.py tests/unit/test_dsa_predictor.py -q -p no:cacheprovider --basetemp="$S/import_regression" -k 'not test_every_collective_launches_one_kernel_by_default' > "$S/import_regression.log" 2>&1` | All selected tests pass | **PASS: 259 passed, 1 deselected**, 16.82 s. Six modules: five existing and one new; 246 existing cases and 13 new cases passed. |
| Placement baseline | `timeout 900s env … "$PY" -m pytest tests/integration/test_vllm_dp_placement_runtime.py -q -p no:cacheprovider --basetemp="$S/before_integration" > "$S/before_integration.log" 2>&1` | Record pre-existing failures | **FAIL: 1 failed, 18 passed**, 71.09 s. |
| Placement after change | Same placement command with `--basetemp="$S/after_integration"` and log `after_integration.log` | No added failure | **FAIL: the same 1 failed, 18 passed**, 71.19 s. |
| Dummy smoke matrix | `timeout 600s env … PYTHON_BIN="$PY" METRICS_OUTPUT_DIR="$S/smoke" bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir "$S/smoke_cache" > "$S/smoke.log" 2>&1` | Complete all cases | **PASS**, exit 0; all 12 dense/MoE, offline/online, co-location/PDD/PD-AF cases completed. |
| Prerequisite reproduction | `timeout 60s env … "$PY" "$S/probe_prerequisites.py" > "$S/prerequisites.log" 2>&1` | Reproduce guards without bypass | **PASS as a diagnostic**, exit 0. Actual predictor construction fails with the documented `NotImplementedError`; A1 is not satisfied. |
| Unused imports | `timeout 20s env … "$PY" /data/ycfeng/tmp/unused_imports.py frontier/execution_time_predictor/kernel_gap.py tests/unit/test_dsv4_h200_tables.py "$P/promote_tables.py" > "$S/unused_imports.log"` | No unused imports | **PASS**; each file reported `[]`. |
| Whitespace | `timeout 20s git diff --check` | No errors | **PASS**, exit 0, no output. |
| Module sizes | `timeout 20s wc -l frontier/execution_time_predictor/kernel_gap.py tests/unit/test_dsv4_h200_tables.py "$P/promote_tables.py" frontier/execution_time_predictor/sklearn_execution_time_predictor.py frontier/metrics/metrics_store.py` | Size limits maintained | **PASS**, counts below. |

Test discovery used `rg -l 'execution_time_predictor.kernel_gap|from .*kernel_gap import' tests/unit -g '*.py'`, following the higher-priority developer instruction to prefer `rg`. All returned modules were included in the six-module regression command.

The placement failure before and after is:

`tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`

At line 481, lane 2 starts as `(0, False)` while the test expects `(0, True)`. This is the same late-join failure in both runs; it was not modified.

Cold imports were executed with `timeout 30s env … "$PY" -` using a Python stdin program that launched four fresh child interpreters (`subprocess.run(..., check=True, timeout=10)`). Each imported one of `dsv4_predictor`, `kernel_gap`, the new promotion script via `runpy.run_path(..., run_name="cold_import")`, or the new test module via `runpy`. Each asserted that no loaded module's top-level name was `torch`, `vllm`, or `cuda`. **All four passed**; output is in `cold_imports.log`.

| Python module | Before | After |
|---|---:|---:|
| `kernel_gap.py` | 207 | 209 |
| `test_dsv4_h200_tables.py` | New | 106 |
| `promote_tables.py` | New | 139 |
| `sklearn_execution_time_predictor.py` | 8626 | 8626, unchanged |
| `metrics_store.py` | 5768 | 5768, unchanged |

The metrics file's observed baseline was 5,768 lines, rather than the brief's 5,744. Neither large file was edited.

**Acceptance 1–13 status**

| Acceptance | Status and evidence |
|---|---|
| 1. Tables | **PARTIAL.** Two completed promotions produced byte-identical manifest/output hashes. All 12 input hashes were revalidated. All 4,945 attention identities match `/data/ycfeng/tmp/b13r_accept/after/rows.json` across 95 identity fields. The existing attention loader accepts all 4,945 rows. All four gap values match exactly. A new complete operator loader is not implemented. Canonical validation passed 15 shared-expert contracts and 132 MoE contracts, while 65 mHC/endpoint contracts were blocked by the disabled family. |
| 2. A1, no dummy reads | **NOT_RUN / blocked.** Actual non-dummy predictor construction fails before training. The no-dummy-read sentinel and both full stage forwards were not implemented or executed. No stage component values are available. The diagnostic probe confirmed no step-cost binding. |
| 3. Exact lookup | **NOT_RUN.** No new operator/MoE prediction models were trained. Inspection confirms the existing trainer's duplicate-feature lookup reduction is arithmetic mean (`sklearn_execution_time_predictor.py:191`), and the shared path uses the same mechanism. No second reduction was added. |
| 4. Repeat and validation report | **PASS as reporting.** EP16/EP32 shared medians are listed below. All 227 KERNEL_ONLY/CUDA_EVENT coordinate comparisons are recorded in `asset_validation.json`; coordinate alignment was asserted before comparison. CUDA_EVENT tables were not promoted. |
| 5. Rank class | **PARTIAL.** The existing `materialize_balanced_rank_classes` function was used on all 33 G3h tasks. Each task's rank selects its recorded triple, and all 848 EP-rank bindings have a measured coordinate, covering EP16 and EP32. Runtime lane lookup is not implemented. `moe_shuffling` remains in the existing V4 path, so its required removal is unfinished. |
| 6. Composition and gaps | **PARTIAL.** Exact four-gap parsing and malformed-table checks pass. Composed runtime pricing, operator segment gaps, missing-file admission, and the EP wrapper change are not implemented or tested. Collective convention inspection is reported below. |
| 7. Both training paths | **NOT_RUN.** The common V4 trainer and both integrations remain unfinished. No trained model-name list or parity result exists. Inspection found no callable-interface prerequisite: the existing train callable shapes support the planned approach. |
| 8. A9 cache | **NOT_RUN.** Promotion determinism is established; trained-model cold/warm cache parity is not. |
| 9. Host time | **NOT_RUN / not implemented.** Zero stage CPU fields and the new typed `skip_cpu_overhead_modeling=False` guard remain unfinished. No existing configuration was changed by a new guard. |
| 10. Runner operators | **NOT_RUN / not implemented.** The requested `kv_cache_block_zero` coverage error is not added. The current constructor fails earlier on attention admission. Runner rows and additional phases remain B17 work. |
| 11. Other paths / A12 partial | **PASS for executed checks except the unchanged baseline failure.** 259 selected unit cases passed, known collective test deselected, placement remained 18 pass/1 fail, and all 12 dummy smokes passed. The 77-case fidelity matrix was not run; it remains Claude's post-commit check. |
| 12. New unit tests | **PARTIAL.** 13 new cases passed, covering promotion and gap precision/validation. Tests for the unfinished runtime decisions are not implemented. |
| 13. Hygiene | **PASS for changed files.** Module sizes are below 2,000, unused-import output is empty, `git diff --check` is clean, and all four cold-import probes load no torch/vLLM/CUDA modules. |

Exact recorded gap values, in microseconds:

| Segment | Gap (µs) |
|---|---:|
| `cuda_graph` | 0.49773414267747085 |
| `cuda_graph_piecewise` | 0.6249537992538311 |
| `eager` | 0.5446049595929223 |
| `eager_preparation` | 1.2208671118771937 |

The existing duplicate gate groups contain 12, 4, 4, 3, and 10 rows for L = 1, 2, 3, 4, and 768 respectively. The existing arithmetic-mean lookup would reduce their median times as follows; these are data-analysis results, not newly trained predictions:

| L | Gate linear mean (ms) | Gate top-k mean (ms) |
|---:|---:|---:|
| 1 | 0.0018720000000000002 | 0.0038241666666666667 |
| 2 | 0.002048 | 0.00374425 |
| 3 | 0.00221575 | 0.00384025 |
| 4 | 0.0025176666666666668 | 0.0036799999999999997 |
| 768 | 0.0084769 | 0.0043490000000000004 |

EP16/EP32 shared-expert repeats, median microseconds:

| Operator | L | EP16 | EP32 |
|---|---:|---:|---:|
| `share_expert_act` | 1 | 1.536000 | 1.535500 |
| `share_expert_act` | 2 | 1.568500 | 1.568000 |
| `share_expert_act` | 3 | 1.664000 | 1.648000 |
| `share_expert_act` | 4 | 1.632000 | 1.632000 |
| `share_expert_act` | 768 | 3.904000 | 3.936000 |
| `share_expert_down_proj` | 1 | 6.304000 | 6.304000 |
| `share_expert_down_proj` | 2 | 6.464000 | 6.464000 |
| `share_expert_down_proj` | 3 | 6.592000 | 6.608000 |
| `share_expert_down_proj` | 4 | 6.608000 | 6.656000 |
| `share_expert_down_proj` | 768 | 18.208500 | 18.272000 |
| `share_expert_up_proj` | 1 | 8.848500 | 8.768500 |
| `share_expert_up_proj` | 2 | 8.832000 | 8.880000 |
| `share_expert_up_proj` | 3 | 9.024000 | 9.088000 |
| `share_expert_up_proj` | 4 | 9.184000 | 9.135500 |
| `share_expert_up_proj` | 768 | 29.472000 | 29.488000 |

KERNEL_ONLY summed-kernel median versus CUDA_EVENT elapsed median, in microseconds. Ranges summarize the coordinate-level comparisons; these durations are reported, not gated. Shared-expert ranges include both EP repeat sets.

| Operator | Comparisons | KERNEL_ONLY range | CUDA_EVENT range | EVENT − KERNEL range |
|---|---:|---:|---:|---:|
| `dsv4_embedding` | 7 | 4.463000–80.320000 | 109.263998–123.408001 | 43.088001–109.601001 |
| `dsv4_final_norm` | 7 | 1.696000–29.008000 | 45.520000–54.032000 | 25.024000–45.360001 |
| `dsv4_lm_head` | 24 | 32.704000–34.432000 | 37.343999–38.928000 | 3.360001–6.063999 |
| `dsv4_mhc_attn_post_ffn_pre` | 5 | 7.136000–41.472000 | 206.176005–213.424005 | 171.791996–206.288005 |
| `dsv4_mhc_ffn_post_attn_pre` | 5 | 7.232000–40.687500 | 209.711999–224.143997 | 183.456497–210.080001 |
| `dsv4_mhc_final_post` | 5 | 2.240500–15.344000 | 58.192000–63.407999 | 47.839997–61.167499 |
| `dsv4_mhc_first_pre` | 5 | 7.456000–17.824000 | 149.935998–156.799994 | 137.584002–149.216494 |
| `dsv4_mhc_head` | 7 | 5.344000–174.432500 | 61.007999–177.552000 | 3.119500–59.488002 |
| `moe_gating_linear` | 33 | 1.728000–8.545000 | 30.320000–93.136001 | 21.776000–91.184001 |
| `moe_gating_routing_topk` | 33 | 3.648000–4.449000 | 41.792000–78.096002 | 37.407000–73.905002 |
| `moe_grouped_gemm` | 33 | 11.775000–1515.447500 | 187.263995–1553.344011 | 37.896511–183.600003 |
| `moe_shuffling` | 33 | 6.432000–72.034000 | 42.320000–94.496001 | 4.543998–87.649001 |
| `share_expert_act` | 10 | 1.535500–3.936000 | 29.792000–31.360000 | 27.295999–29.728000 |
| `share_expert_down_proj` | 10 | 6.304000–18.272000 | 69.183998–100.736000 | 62.767999–82.527500 |
| `share_expert_up_proj` | 10 | 8.768500–29.488000 | 69.103997–101.839997 | 60.127998–72.367997 |

Comparison inputs were the specified G2h `*_RECORD_FUNCTION` and `*_CUDA_EVENT` directories and G3h `G3H_MOE_RECORD_FUNCTION/ep{16,32}` and `G3H_MOE_CUDA_EVENT/ep{16,32}` directories, under `/data/ycfeng/tmp/glm53_dsv4/h200/`. All 227 individual pairs, coordinates, contract errors, duplicate groups, hashes, and identity results are retained in [asset_validation.json](/data/ycfeng/tmp/codex-implement/54e2e32d-92b5-49b8-b578-4119b950ab6b/b16/asset_validation.json).

Collective convention, **inspected rather than demonstrated through a trained V4 forward**: all seven physical V4 collective specs report `kernel_count=1`: embedding all-reduce, SP attention all-gather, SP attention reduce-scatter, SP final-mHC all-gather, logits gather, EP dispatch all-gather, and EP output reduce-scatter. Existing `ExecutionTime.add_kernel_gap` uses a collective's spec count when its time is positive and zero otherwise. Its EP ownership convention puts dispatch's gap in `pre_dispatch` and combine's gap in `routed_compute`. The requested V4 runtime integration has not been implemented, so this report does not claim those gaps are currently delivered to a non-dummy V4 forward.

Relevant unsuccessful checks were diagnosed rather than hidden. The first promotion attempt found an incorrect root-parent index; it was terminated before publication and corrected before the two completed runs. An early focused run overlapped table publication and observed 3,075 attention rows, yielding 1 failed/32 passed/1 deselected; the sequential rerun after publication passed. The scratch identity checker initially treated serialized layer-ID strings as lists and omitted variant-specific fields; it was corrected to compare the union of 95 identity fields while preserving blank-string versus missing-value representations. A later comparison assertion assumed a `row_axis` column in shared-expert inputs; it was corrected to align on tokens, selected TP, and typed contracts. The final asset validation passes. The existing attention loader emitted a pandas mixed-type warning for columns 49/50; it did not reject any row.

**4. Remaining work and decisions for Claude**

Recommended next action: supply or authorize the canonical mHC profile/registry contracts and the narrowly scoped measured-V4 admission change. At minimum, resolving the observed prerequisites requires work in `frontier/operators/families.py`, `frontier/model_architectures.py`, and either the base-predictor admission path or the attention-family admission mechanism (`base_execution_time_predictor.py` / `frontier/attention/families.py`). These should be completed together with the runtime consumer so enabling admission does not expose an incomplete pricing path.

Then finish the common V4 loader/trainer, both training registrations, exact operator/MoE lookup, runtime row binding, shuffling removal, composed attention pricing, segment-specific operator/collective gaps, EP wrapper routing, KERNEL_ONLY selection, host-time guard/zeros, and the B16 runner coverage failure. Migrate the isolated-reader tests and delete `read_isolated_scope` and `_read_isolated_component` only after their replacement is complete.

Observed facts are the guard failures, missing profile contract, passing independent table checks, existing arithmetic-mean duplicate reduction, and unchanged placement failure. The conclusion that complete canonical training and A1 cannot proceed within the named file scope follows from those observations; no substitute contract design was chosen.

The promotion script defaults to this session's scratch location and accepted absolute input paths. Another executor should pass `--scratch-dir` for its own approved scratch directory. Tables should be consumed only after the promotion process completes; publication is not transactional. The raw download directories were not modified.

The 77-case fidelity matrix, B17 runner rows/phases and admission work, commits, task records, and final acceptance remain with Claude.

**5. Review handoff**

Review [promote_tables.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_tables.py:57) for accepted-input provenance, scratch-only reduction, lossless unions, and deterministic manifest construction. Review [kernel_gap.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/kernel_gap.py:47) for the V4-only parser precision change and [test_dsv4_h200_tables.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_h200_tables.py:35) for its 13 cases.

Before committing this partial change, rerun the promotion sequentially and the focused/import regression commands, verify the ignored script and manifest are included, and preserve the known placement failure status. After the contract/admission prerequisites and remaining runtime implementation are complete, execute all currently NOT_RUN acceptance gates; the successful data checks do not substitute for A1, exact runtime lookup, composition, training/cache parity, host-time, or runner tests.

Scratch contains `promotion/`, both completed promotion logs, `manifest_first.json`, `validate_assets.py`, `asset_validation.json`, `comparison_summary.md`, the prerequisite probe/results, focused/import-regression logs, before/after integration logs and test directories, smoke outputs/cache/log, unused-import output, and cold-import output. No claim of Claude review or acceptance is made.
