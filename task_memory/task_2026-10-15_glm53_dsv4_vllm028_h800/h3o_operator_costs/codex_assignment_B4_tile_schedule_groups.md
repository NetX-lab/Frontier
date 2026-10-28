# Task: DeepSeek-V4 tile-schedule planner layers from vLLM's KV-cache groups, an 11-layer reduced model, realized-group records, and identity expansion per layer role (stage 1a assignment B4, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B4 of stage 1a. |

## Context

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `d80af9de`). Git is clean except the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`; do not touch them.
- Background (read first): `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/test_report_2026-10-28_b3_hold_pilot_identity.md`, section "Finding: tile-schedule planner kernels". Assignment B3 is `h3o_operator_costs/codex_assignment_B3_hold_pilot_identity.md` (same directory as this file); its code is committed in `d80af9de`.
- The defect in one paragraph: FlashMLA plans the decode tile schedule (kernel `smxx::decode::get_mla_metadata_kernel`, about 26 µs) once per layer type per SWA attention-metadata object, and each SWA KV-cache group has its own metadata object. Natively (43 layers, 2 SWA groups: even and odd layers) only layers 0, 1, 2 and 3 plan; layers 4-42 reuse. The 6-layer reduced engine forms 3 SWA groups ({0, 3}, {1, 4}, {2, 5}), so every layer plans and there is no reusing layer of any variant. Frontier's current rule ("the first layer of each variant plans") is also wrong for the full model: natively the uncompressed layer 1 plans too.
- vLLM 0.28.0 source (commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`, checkout `/data/ycfeng/tmp/opencode/vllm-v0.28.0`; read-only), verified by the lead:
  - `vllm/v1/core/kv_cache_utils.py:1545` `group_and_unify_kv_cache_specs`: the full-MLA specs form one group; the `SlidingWindowMLASpec` specs are grouped by `(block_size, sliding_window)`, which gives three uniform groups: SWA caches (one per decoder layer), CSA compressor states (compressor state and indexer-compressor state of each CSA layer) and HCA compressor states (one per HCA layer).
  - `kv_cache_utils.py:1588` `_approximate_gcd(values, lower_bound)`: brute force `d` in `[max(1, lower_bound), max(values)]`, padding `sum((d - x % d) % d)`, minimum padding wins and ties prefer the larger `d`; if `lower_bound > max(values)` it returns `lower_bound`.
  - `kv_cache_utils.py:1623` `_get_kv_cache_groups_uniform_groups`: `values = [g.get_num_layer_tuples() for g in grouped_specs]` (full-MLA group first), `d = _approximate_gcd(values, lower_bound=values[0])`. Each SWA-type uniform group is split into `G = cdiv(num_layers_per_size, d)` groups, group `i` taking `layer_tuples[i::G]`, where a layer tuple holds one spec of each page size of that uniform group, in layer order. The full-MLA group is never split.
  - `vllm/v1/kv_cache_interface.py:863` `get_num_layer_tuples`: the count of the most common page size in the group.
  - Page-size facts, confirmed by the native group structure below: the C4A KV, C4 indexer KV and C128A KV page sizes differ, so the full-MLA tuple count is `max(n_csa, n_hca)`; all SWA caches share one page size (tuple count `N`, the decoder-layer count); the CSA compressor state and the CSA indexer-compressor state differ (tuple count `n_csa`); the HCA compressor states give `n_hca`.
  - `vllm/models/deepseek_v4/attention.py:623-625` reads `attn_metadata.get(self.swa_cache_layer.prefix)`; `nvidia/flashmla.py:225-254` and `sparse_swa.py:711-743` (`build_tile_scheduler`) keep one `FlashMLASchedMeta` per layer type (SWA-only, C4A, C128A) in each SWA metadata object; the first layer of a type in that object launches the planner and later layers of that type reuse it. No planner runs in a step without decode tokens.
  - Cache prefixes (`attention.py:329-357, 839-869`, `compressor.py:295-299`): compressed-attention KV `module.prefix`; SWA `module.swa_cache_layer.prefix` (`f"{prefix}.swa_cache"`); compressor state `module.compressor.state_cache.prefix`; indexer KV `module.indexer.k_cache.prefix`; indexer-compressor state `module.indexer.compressor.state_cache.prefix`. Uncompressed layers have only the SWA cache.
  - The V2 model runner keeps the realized groups in `worker.model_runner.kv_cache_config.kv_cache_groups[i].layer_names` (`vllm/v1/worker/gpu/model_runner.py:495-511`); the profiling hooks are installed through `collective_rpc` after engine start-up, so the attribute exists when `attention_inventory` runs.
- Native evidence (read-only): serving inventory `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-canonical-20261025-01/artifacts/nodes/gpu-h200-0082.lgcm.sh.istep.fun/serving/rank0.json`, field `inventory.cache.kv_cache_groups` (each entry has `layer_names`). Its 5 groups: full MLA 62 names (41 `attn` of the CSA and HCA layers, 21 `attn.indexer.k_cache`); SWA layers 0, 2, …, 42; SWA layers 1, 3, …, 41; CSA states 42 names (`attn.compressor.state_cache` and `attn.indexer.compressor.state_cache` of the 21 CSA layers); HCA states 20 names. MTP is off in the native run (no speculative config). Lead script that prints them: `/data/ycfeng/tmp/glm53_dsv4/h200/b3_verify/native_kv_groups.py`.
- Model: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/package_02/model/config.json`: 43 decoder layers; `compress_ratios` = `[0, 0, 4, 128, 4, 128, …, 4, 128, 4, 0]` (44 entries; the last is the MTP layer). Layers 0 and 1 uncompressed; even layers 2-42 CSA (21); odd layers 3-41 HCA (20).
- Worked values of the rule (the lead checked them by hand; your tests must reproduce them):

  | Leading layers | values `[full, N, n_csa, n_hca]` | `d` | SWA groups | CSA-state groups | HCA-state groups | Planner layers |
  | --- | --- | --- | --- | --- | --- | --- |
  | 43 (native) | [21, 43, 21, 20] | 22 | 2: even, odd | 1 | 1 | 0, 1, 2, 3 |
  | 6 (B2/B3 pilot) | [2, 6, 2, 2] | 2 | 3: {0,3}, {1,4}, {2,5} | 1 | 1 | all six |
  | 7 | [3, 7, 3, 2] | 3 | 3 | 1 | 1 | — |
  | 8 | [3, 8, 3, 3] | 3 | 3 | 1 | 1 | — |
  | 9 | [4, 9, 4, 3] | 4 | 3 | 1 | 1 | — |
  | 10 | [4, 10, 4, 4] | 4 | 3 | 1 | 1 | — |
  | 11 | [5, 11, 5, 4] | 6 (pad 5 at d = 5 and 6; tie → 6) | 2: {0,2,4,6,8,10}, {1,3,5,7,9} | 1 | 1 | 0, 1, 2, 3 |
  | 12 | [5, 12, 5, 5] | 6 (tie with 5) | 2 | 1 | 1 | 0, 1, 2, 3 |
  | 13 | [6, 13, 6, 5] | 7 | 2 | 1 | 1 | 0, 1, 2, 3 |

  Full-model layer roles: uncompressed 2 planners and 0 reusing layers; CSA 1 planner (2) and 20 reusing; HCA 1 planner (3) and 19 reusing. The 11-layer model has the same roles present: uncompressed planners {0, 1}; CSA planner {2}, reusing {4, 6, 8, 10}; HCA planner {3}, reusing {5, 7, 9}.
- Test interpreter: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6; numpy, pandas, sklearn; no torch, vllm or cuda-bindings). Every module you change must import none of torch, vllm or cuda at module load. Python 3.10 syntax.

## Decisions and write scope

- Writable (granted paths): `frontier/attention/`, `frontier/execution_time_predictor/`, `frontier/profiling/attention/`, `tests/comparison/calibration/`, `tests/unit/`, `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/`. Within them change only:
  - `frontier/attention/deepseek_v4.py` (section 1);
  - `frontier/execution_time_predictor/deepseek_v4.py` (section 2, the `plans_tile_schedule` lookup only);
  - `frontier/profiling/attention/dsv4_attention_plan.py`, `dsv4_attention_runner.py`, `dsv4_attention_rows.py`, and `dsv4_graph_replay_runner.py` only if the inventory change needs it (sections 3-5);
  - `tests/comparison/calibration/dsv4_composition_check.py`, `dsv4_kernel_identity_check.py` (sections 5-6);
  - `p1h_bridge/package_plan.json` (section 7; nothing else under `p1h_bridge/`, and never `package_01/` or `package_02/`);
  - unit tests under `tests/unit/` (existing files updated; new test file `tests/unit/test_dsv4_kv_cache_groups.py`).
- Scratch for scripts and outputs: `/data/ycfeng/tmp/codex-implement/<request-id>/b4/` (or another directory under `/data/ycfeng/tmp`). No Git commands that change state; the lead commits. No GPU, network, package installs, deletions or moves.
- Keep each module at or below 2,000 lines. Plain domain names; no magic numbers in code paths (the variant ids come from `DEEPSEEK_V4_ATTENTION_FAMILY.supported_variants`; cache names and their per-variant table are named constants).

### 1. The grouping rule (`frontier/attention/deepseek_v4.py`)

Add the single authority for vLLM's DeepSeek-V4 KV-cache groups and the tile-schedule planner layers. Suggested public API (adjust names only if a clearly better plain name exists, and report it):

- A table of cache names per variant, for example `uncompressed: ("swa",)`, `csa: ("attention", "indexer", "swa", "compressor_state", "index_compressor_state")`, `hca: ("attention", "swa", "compressor_state")`.
- `deepseek_v4_kv_cache_groups(variant_ids: Sequence[str]) -> tuple[tuple[tuple[int, str], ...], ...]`: the vLLM 0.28 KV-cache groups of decoder layers `0 … len(variant_ids) - 1` as `(global_layer_id, cache)` members: the full-MLA group (all `attention` and `indexer` caches), then the SWA groups, the CSA-state groups and the HCA-state groups, each built exactly as in the Context (tuple counts, `_approximate_gcd` with the tie rule and lower bound, `cdiv` split, stride `i::G`). Members within a group sorted by `(layer, cache)`. Raise `ValueError` with a clear message for an unknown variant id or a layout without CSA or without HCA layers (the page-size facts above are verified only for layouts with both).
- `deepseek_v4_tile_schedule_planners(variant_ids) -> tuple[bool, ...]`: per layer, `True` when the layer is the first layer of its variant within its SWA group.
- Module or function docstring: cite the vLLM sources above at `2cf0a69`, state the page-size facts and that the native H200 inventory confirms the 43-layer groups, and that MTP layers (speculative decoding) are outside the rule; Frontier's V4 attention already rejects speculative decoding.
- Keep the private `_approximate_gcd` port faithful (same bounds, same tie rule, same error cases for empty or non-positive values).

### 2. Predictor role (`frontier/execution_time_predictor/deepseek_v4.py`)

In `read_isolated_scope`, replace the `"plans_tile_schedule": not any(...)` expression (line 783) with the section 1 rule over the model's decoder layers (`model.get_layer_attention_specs()` variant ids), indexed by `layer_id`. Compute it once per call; do not add a cache. The other uses of the column (lines 665, 743, 927) stay as they are.

### 3. Reduced layout (`frontier/profiling/attention/dsv4_attention_plan.py`)

- `Dsv4Layer.plans_tile_schedule` comes from the section 1 rule applied to the reduced layers (the leading `num_layers` layers). Update its comment.
- A reduced layout of `n` leading layers is admissible when all of these hold, compared with the full model (`num_hidden_layers` decoder layers):
  1. it holds every variant;
  2. its SWA, CSA-state and HCA-state groups have the same counts as the full model's;
  3. its SWA groups, in order, hold the same variant sets as the full model's;
  4. every `(variant, plans_tile_schedule)` role of the full model occurs in it.
- `num_layers=None` takes the fewest admissible leading layers (11 for DeepSeek-V4-Flash). An explicit `num_layers` that is not admissible raises `ValueError` naming the failed condition and the realized groups (6 must fail with the SWA group count 3 versus 2). `num_layers` equal to the full count is admissible.
- Remove `_LAYERS_PER_VARIANT` and the old "two of each variant" search and message.
- `Dsv4ProfilePlan.summary()` adds `kv_cache_groups` (the section 1 groups of the reduced layers, as lists). The plan JSON schema does not change otherwise; `hf_overrides` keeps shrinking `num_hidden_layers` and `compress_ratios[:num_layers]` as now.
- Add one function used by both consumers in sections 4 and 5, for example `require_realized_kv_cache_groups(plan, engine_record)`: compares the engine record's `kv_cache_groups` with the plan's predicted groups, order-insensitive over groups, exact within a group; raise `ValueError` with both lists on any difference or when the field is missing.

### 4. Realized groups (`dsv4_attention_runner.attention_inventory`)

- Add `kv_cache_groups` to the returned inventory: one entry per `worker.model_runner.kv_cache_config.kv_cache_groups` element, in vLLM order, each a sorted list of `[global_layer_id, cache]` from that group's `layer_names`. Map names through the discovered modules' cache prefixes (Context); a name that is not a discovered V4 cache raises `ValueError` naming it. Both the eager runner (`install_dsv4_hooks`) and the graph-replay runner (`install_graph_replay_hooks`) already call `attention_inventory`, so both record it.
- Update the runner unit-test fakes (`worker.model_runner.kv_cache_config`) accordingly.

### 5. Consumers of the planner role

- `dsv4_attention_rows.build_dsv4_rows`: after its existing `require_scope_timing` check, require the realized groups of every engine record through the section 3 function. Update the test fixtures (`_engine_record` and similar) to carry the predicted groups.
- `tests/comparison/calibration/dsv4_composition_check.planner_roles`: keep the layer inventory check; add the realized-group check per run; roles come from the rebuilt plan as now.

### 6. Identity expansion per layer role (`dsv4_kernel_identity_check.py`)

- Replace `variant_layer_counts` with full-model role counts: `Counter((variant, plans_tile_schedule))` of the full-model layers from the section 1 rule (via `resolve_dsv4_layers(..., num_hidden_layers)` or the rule directly).
- `expand_attention_kernels`: for each full-model role, the reduced layers with that role must exist, and all of them must have identical kernel multisets in the step (raise `ValueError` naming the role and layers otherwise); add `count × multiset` of one of them. A reduced role absent from the full model raises.
- Remove the module's `_LAYERS_PER_VARIANT`; the report key becomes, for example, `layer_role_counts` as a list of `{variant, plans_tile_schedule, count}`. Update `__all__` and the docstring line about one planner and one reuse layer.

### 7. Package plan (`p1h_bridge/package_plan.json`)

Remove `"--num_layers", "6"` from `plan_args` so the default rule selects the 11 layers. Leave every other field unchanged (the lead sets `source_revision` after committing B4).

### 8. Tests

- New `tests/unit/test_dsv4_kv_cache_groups.py`:
  - every row of the worked-values table (`d`, the group counts and members, the planner layers) through the public API;
  - the 43-layer groups equal the native inventory's group structure written out in the test (members as listed in the Context, with the cache names of section 1), citing the inventory path in a comment;
  - `_approximate_gcd` tie and bound cases (including `lower_bound > max(values)`), empty and non-positive values;
  - unknown variant and missing CSA or HCA raise.
- `test_dsv4_attention_profiling.py`: the default plan has 11 layers with labels `0 T, 1 T, 2 T, 3 T, 4-10 F` and `hf_overrides` for 11 layers; explicit 6 raises (SWA group count), explicit 11, 12, 13 and 43 are accepted; the rows test expects the five keys `(csa, True, [2])`, `(csa, False, [4, 6, 8, 10])`, `(hca, True, [3])`, `(hca, False, [5, 7, 9])`, `(uncompressed, True, [0, 1])`; the summary carries `kv_cache_groups`.
- `test_dsv4_graph_measured.py` (or the existing predictor test that covers `read_isolated_scope`): full-model layer 1 selects `plans_tile_schedule=True`, layer 4 selects `False`.
- `test_dsv4_pilot_cases.py`: replace `num_layers=6` with the default; `test_package_plan_cli_rebuilds_pilot_and_summary` still gets 25 cases, `{prefill 1, decode 10, first_tail 4, dummy 10}` and engines `[280, 55]`, and checks the 11 reduced layers.
- `test_dsv4_composition_check.py`: planner roles with the realized-group check (pass, missing field, different groups).
- `test_dsv4_kernel_identity_check.py`: the expansion with several layers per role (factors 2, 1, 20, 1, 19 for the 11-layer reduced model), unequal multisets within a role raising, a missing role raising, the report key.
- Runner tests: `attention_inventory` records the groups from a fake `kv_cache_config` and raises on an unknown name.

## Actions and acceptance

1. Implement sections 1-8.
2. Run from the working root in `bash` with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1`, the test interpreter, `ulimit -v 31457280` and timeouts. Baselines at HEAD `d80af9de` (rerun by the lead):
   - group 1: `tests/unit/test_dsv4_replay_gaps.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_graph_bridge_check.py tests/unit/test_graph_capture.py tests/unit/test_graph_replay_trace.py` → 262 passed;
   - group 2: `tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsv4_composition.py` → 72 passed;
   - group 3: `tests/unit/test_dsa_query_one_profiling.py tests/unit/test_dsv4_attention_profiling.py` → 184 passed;
   - group 4: `tests/unit/test_dsa_extend_profiling.py` → 6 passed;
   - group 5: `tests/unit/test_kernel_gap_pricing.py tests/unit/test_kernel_gap_profiler.py` → 29 passed, 1 failed (pre-existing: `test_every_collective_launches_one_kernel_by_default`);
   - group 6: `tests/unit/test_dsa_full_dummy_profiling.py tests/unit/test_dsv4_graph_measured.py tests/unit/test_profiling_runtime_boundary.py` → 133 passed, 76 skipped, 1 failed (pre-existing: `test_runtime_predictors_use_only_allowlisted_profiling_helpers`);
   - group 7: `tests/unit/test_dsv4_pilot_cases.py tests/unit/test_dsv4_kernel_identity_check.py` → 65 passed;
   - group 8: `tests/unit/test_dsv4_attention_workload.py tests/unit/test_dsv4_cache_layout.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_structural_pd.py tests/unit/test_dsv4_dp_sync.py tests/unit/test_dsv4_model_config.py tests/unit/test_dsv4_step_cost_input.py tests/unit/test_dsv4_operator_manifest.py tests/unit/test_dsv4_cache_zero_driver.py tests/unit/test_dsv4_cache_zero_execution.py` → 331 passed;
   - after the change: all pass except the two pre-existing failures; new tests pass. Report counts per group.
3. Native check (scratch script): map the native inventory's `layer_names` to `(layer, cache)` by the name suffixes (`attn`, `attn.indexer.k_cache`, `attn.swa_cache`, `attn.compressor.state_cache`, `attn.indexer.compressor.state_cache`) and compare with `deepseek_v4_kv_cache_groups` of the 43 native variants (order-insensitive over groups) → equal.
4. Plan comparison: extract HEAD with `git archive HEAD frontier data/config | tar -x -C <scratch>/head`; run the HEAD copy and the working tree:
   - `dsa_main plan` for one DSA configuration from the existing DSA tests (eager-chunked and native-defaults) → byte-identical stdout and `--output` file;
   - V4 `dsv4_attention_main plan` eager-chunked on the grid and native-defaults with `--model_config_path p1h_bridge/package_02/model/config.json` → parsed JSON differs only in `shape.layers` and `hf_overrides` (`num_hidden_layers`, `compress_ratios`); report every differing key path; stdout summary differs only in `reduced_layers` and the new `kv_cache_groups`;
   - the package 03 plan (`package_plan.json` `plan_args` with `--model_config_path p1h_bridge/package_02/model/config.json`, `--pilot_cases p1h_bridge/pilot_cases.json` and the FULL_AND_PIECEWISE flags with the 35 capture sizes): 11 layers, 25 cases, `{prefill 1, decode 10, first_tail 4, dummy 10}`, estimated steps `[280, 55]`.
5. Import check: importing `frontier.attention.deepseek_v4`, `frontier.execution_time_predictor.deepseek_v4`, `dsv4_attention_plan`, `dsv4_attention_runner`, `dsv4_attention_rows`, `dsv4_graph_replay_runner` and the two calibration scripts loads no `torch`, `vllm` or `cuda` module.
6. `wc -l` of every changed and new file.

## Handoff

- Report language: English.
- Material decisions not settled here (a fact the vLLM source or the native inventory contradicts, a test that cannot be updated without changing behavior outside this assignment, a needed edit outside the write scope) go back to Claude in the report with evidence; continue the independent parts.
- Include: changed paths with a summary per file, the public API added, executed checks with actual output tails (group counts, the native check, the plan comparisons with the differing key paths, the package 03 plan summary), scratch paths, deviations with reasons, blockers, and limits (the realized-group record is untested against a real vLLM engine).
