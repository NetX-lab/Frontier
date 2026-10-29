# Test report: stage 1a assignment B4 (tile-schedule planner layers from vLLM KV-cache groups)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead verification of Codex assignment B4. |

## Scope

- Assignment: [h3o_operator_costs/codex_assignment_B4_tile_schedule_groups.md](h3o_operator_costs/codex_assignment_B4_tile_schedule_groups.md); Codex request `7fb91668-cb81-4285-bdc6-11e012f0bb65`, report `h3o_operator_costs/codex-implement-7fb91668-cb81-4285-bdc6-11e012f0bb65/implementation.md`, outcome `implemented`.
- Cause: the B3 finding ([test_report_2026-10-28_b3_hold_pilot_identity.md](test_report_2026-10-28_b3_hold_pilot_identity.md), "Finding"): the 6-layer reduced engine plans the FlashMLA decode tile schedule in every layer, native plans in layers 0-3 only.
- Baseline: HEAD `d80af9de`. All checks below were rerun by the lead on CPU; no GPU was used.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6, no torch or vllm); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `ulimit -v 31457280`.
- Lead scratch: `/data/ycfeng/tmp/glm53_dsv4/h200/b4_verify/`.

## What changed

- `frontier/attention/deepseek_v4.py`: `deepseek_v4_kv_cache_groups(variant_ids)` reproduces vLLM's decoder KV-cache groups: one full-MLA group (compressed KV and indexer caches), then the SWA caches, the CSA states and the HCA states, each split into `cdiv(count, d)` strided groups with `d = _approximate_gcd([max(n_csa, n_hca), N, n_csa, n_hca], lower_bound=max(n_csa, n_hca))`. `_approximate_gcd` is a line-for-line port of vLLM's function. `deepseek_v4_tile_schedule_planners` marks the first layer of each variant in each SWA group.
- The predictor (`read_isolated_scope`) and the profiling plan (`resolve_dsv4_layers`) both use this rule. The default reduced model is the fewest leading layers whose group counts, ordered SWA variant sets and (variant, role) coverage equal the full model's: 11 layers, planners 0-3. An explicit `--num_layers 6` is rejected.
- The eager and graph-replay runners record `kv_cache_groups` from `worker.model_runner.kv_cache_config` and reject unknown cache names. `build_dsv4_rows` and the A4 planner-role check require the recorded groups to equal the plan's.
- `dsv4_kernel_identity_check.py` expands per (variant, planner role) with full-model counts {uncompressed planner 2, CSA planner 1, CSA reuse 20, HCA planner 1, HCA reuse 19} and requires every reduced layer of a role to carry the same kernel multiset.
- `p1h_bridge/package_plan.json`: `--num_layers 6` removed.
- Lead correction: the comment on the default reduced layout in `frontier/profiling/attention/dsv4_attention_main.py` (outside Codex's write list) now describes the new rule.

## Code review

- `_approximate_gcd` compared with `vllm/v1/core/kv_cache_utils.py` (local tree `/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm`, lines 1958-1988): validation, lower bound, the early return and the tie rule are identical.
- Every changed path is inside the assigned write list, except the lead's comment fix. All hunks were read.
- Module sizes after the change: `frontier/attention/deepseek_v4.py` 463, `execution_time_predictor/deepseek_v4.py` 965, `dsv4_attention_plan.py` 563, `dsv4_attention_runner.py` 529 lines.

## Checks

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `GROUPS_SEL="1" bash b4_verify/run_pytest.sh` (replay gaps, composition check, graph replay profiling, bridge check, graph capture, replay trace) | all pass | 264 passed | PASS |
| 2 | group 2 (DSA attention profiling, V4 composition) | all pass | 72 passed | PASS |
| 3 | group 3 (DSA query-one, V4 attention profiling) | all pass | 197 passed | PASS |
| 4 | group 4 (DSA extend) | all pass | 6 passed | PASS |
| 5 | group 5 (kernel gap pricing and profiler) | only the known failure | 29 passed, 1 failed (`test_every_collective_launches_one_kernel_by_default`, failing at `d80af9de` too) | PASS |
| 6 | group 6 (full dummy, graph measured, runtime boundary) | only the known failure | 135 passed, 76 skipped, 1 failed (`test_runtime_predictors_use_only_allowlisted_profiling_helpers`, same assertion as at the B3 baseline) | PASS |
| 7 | group 7 (pilot cases, kernel identity check) | all pass | 71 passed | PASS |
| 8 | group 8 (other V4 unit tests) | all pass (baseline 331) | 331 passed | PASS |
| 9 | `pytest tests/unit/test_dsv4_kv_cache_groups.py` (worked layouts 6-13 and 43, native groups, gcd ties and bounds, invalid layouts) | all pass | 29 passed | PASS |
| 10 | `b4_verify/native_groups_check.py`: predicted 43-layer groups against the native P2 serving inventory (`dsv4-h200-p2-canonical-20261025-01`, prefill rank 0, `inventory.cache.kv_cache_groups`) | equal, order-insensitive | groups [62, 22, 21, 42, 20] on both sides, equal; planners [0, 1, 2, 3] | PASS |
| 11 | `b4_verify/plan/plan_checks.py`: DSA grid and native plans, `git archive d80af9de` against the working tree | byte-identical | both byte-identical | PASS |
| 12 | same script: V4 grid, q1, prefill and native plans | plan differs only in the reduced layout | plan diffs only `hf_overrides.compress_ratios`, `hf_overrides.num_hidden_layers`, `shape.layers`; summary diffs these overrides plus the new `kv_cache_groups` and `reduced_layers`; 11 layers, planners 0-3 | PASS |
| 13 | same script: package 03 plan from `package_plan.json` | 11 layers, 25 cases, steps [280, 55] | 11 layers, 25 cases {prefill 1, decode 10, first_tail 4, dummy 10}, steps [280, 55], 5 groups with members [14, 6, 5, 10, 4] | PASS |
| 14 | cold `import` of the seven changed or dependent modules (`frontier.attention.deepseek_v4`, the V4 predictor, plan, runner, rows, main, `frontier.main`) | exit 0 | all exit 0 | PASS |

The summary includes `hf_overrides` because the V4 summary inherits `DsaProfilePlan.summary()`; the override change is the intended 6-to-11 layer change. The native-policy V4 plan is built through the harness override `FRONTIER_PLAN_NATIVE=1`, as in B3, because the V4 parser does not offer `native_defaults`.

## Behavior changes and limits

- Historical isolated V4 tables measured with the old rule (the warmed q1 table) label layer 1 as a reuse layer; natively layer 1 plans. The predictor now refuses that row for layer 1 instead of pricing it with the wrong role (test `test_real_ongoing_decode_exact_scopes_reject_obsolete_layer_role`). New measurements are needed; the row is not relabeled.
- Rows and bridge data taken from 6-layer engines (B2, B3) carry wrong reuse labels and no `kv_cache_groups`; they can no longer be rebuilt under the new plan.
- The decode filter of `read_isolated_scope` still requires a single `global_layer_ids` entry and `num_layers_sampled == 1`. Eager rows aggregated over several layers of one role need a separate qualification before they can be used there. Graph-replay row aggregation stays refused, so package 03 is not affected.
- Not run on a GPU: the realized-group hook against a real vLLM engine, planner kernel positions, and package 03. The native serving inventory confirms the rule for 43 layers; the 11-layer prediction is first confirmed by package 03.
- 11 layers instead of 6 raise the GPU time of package 03; the run stays within the P1h cap (about 1.67 GPU-hours left).
