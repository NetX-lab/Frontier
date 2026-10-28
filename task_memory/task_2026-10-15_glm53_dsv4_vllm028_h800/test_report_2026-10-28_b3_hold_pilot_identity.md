# Test report: stage 1a assignment B3 (step hold, pilot cases, DP-peer emulation, A4 own-gap gate, kernel identity check)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead verification of Codex assignment B3 and the tile-schedule planner finding. |

## Scope

- Assignment: [h3o_operator_costs/codex_assignment_B3_hold_pilot_identity.md](h3o_operator_costs/codex_assignment_B3_hold_pilot_identity.md); Codex request `1042c9ef-35cf-4c99-a6a7-52795291a84a`, report `h3o_operator_costs/codex-implement-1042c9ef-35cf-4c99-a6a7-52795291a84a/implementation.md`, outcome `implemented`.
- Baseline: HEAD `e59ac014`. All checks below were rerun by the lead on CPU; no GPU was used.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6, no torch or vllm); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `ulimit -v 31457280`.
- Lead scratch: `/data/ycfeng/tmp/glm53_dsv4/h200/b3_verify/`.

## Lead corrections during review

1. `build_dsv4_rows` now calls `require_scope_timing(plan.limits.engine_policy)` first, so a graph-replay plan cannot aggregate rows from eager timing (Codex's write list did not include `dsv4_attention_rows.py`). Test: `test_package_plan_cli_rebuilds_pilot_and_summary` expects "engine_policy='graph_replay' cannot collect or aggregate".
2. The assignment's context line wrongly said layers 0, 1 and 42 are uncompressed; the model has 2 uncompressed, 21 CSA and 20 HCA layers (layer 42 is CSA). The code uses `resolve_dsv4_layers` and was already right; the assignment text is corrected.

## Checks

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `bash b3_verify/run_pytest.sh` group 1 (replay gaps, composition check, graph replay profiling, bridge check, graph capture, replay trace) | all pass | 262 passed | PASS |
| 2 | group 2 (DSA attention profiling, V4 composition) | all pass | 72 passed | PASS |
| 3 | group 3 (DSA query-one, V4 attention profiling) | all pass | 184 passed | PASS |
| 4 | group 4 (DSA extend) | all pass | 6 passed | PASS |
| 5 | group 5 (kernel gap pricing and profiler) | only the known failure | 29 passed, 1 failed (`test_every_collective_launches_one_kernel_by_default`, failing at `e59ac014` too) | PASS |
| 6 | group 6 (full dummy, graph measured, runtime boundary) | only the known failure | 133 passed, 76 skipped, 1 failed (`test_runtime_predictors_use_only_allowlisted_profiling_helpers`, failing at `e59ac014` too) | PASS |
| 7 | group 7 (pilot cases, kernel identity check) | all pass | 65 passed | PASS |
| 8 | `b3_verify/plan/plan_checks.py`: six legacy plans (DSA grid and native, V4 grid, native, prefill, q1) from `git archive HEAD` and the working tree | byte-identical | all six identical | PASS |
| 9 | package_02 plans of the three modes, HEAD vs working tree | differ only by the new `step_hold_us` field | only `step_hold_us` differs | PASS |
| 10 | `tests/analysis/v4_runtime_census.py` on P2-32, HEAD vs working tree (`b3_verify/census/`) | byte-identical | JSON SHA256 `02d5dc92…1382`, summary `ba6befde…556e`, both equal | PASS |
| 11 | `b3_verify/rebuild_bridge_raw.py` re-reduction of the B2 bridge tree (`b3_verify/bridge_reduced/`) | call rows equal the worker's | 12 cases, all call rows equal (`rebuild.log`) | PASS |
| 12 | `dsv4_graph_bridge_check.py --config bridge_reduced/bridge_config.json --report` | B2 result | `PASS: 3 runs, FULL=24, PIECEWISE=72` | PASS |
| 13 | `python -m frontier.profiling.kernel_gap.dsv4_replay_gaps --raw_files … --output_dir … --device h200 --model … --diagnostics_path gap_diagnostics.json` | exit 1, no CSV, B2 values | exit 1, "eager_preparation: no host-ahead sample"; `cuda_graph` 0.40292, `cuda_graph_piecewise` 0.96182, `eager` 1.93453 µs; diagnostics equal to the B2 evidence except the new `hold_by_step_mode_and_case` | PASS |
| 14 | `dsv4_composition_check.py --config bridge_reduced/bridge_config.json` | gate {no_host_ahead_call 38, pass 36}; class-value diagnostic {38, fail 28, pass 8} | gate FAIL {no_host_ahead_call 38, pass 36}; diagnostic {38, 28, 8}; largest p90 own-gap error 1.85e-16 | PASS (expected on B2 data: no hold) |
| 15 | `dsv4_kernel_identity_check.py` against P2-32 and P2-64 (`identity_p32.json`, `identity_p64.json`) | runs; B2 pilot is not native-identical | 144 of 144 steps FAIL in both, no error reasons | PASS (tool), see finding |
| 16 | `dsv4_attention_main plan` with `package_plan.json` arguments (`package03_plan.json`) | 25 cases, 335 steps | {prefill 1, decode 10, first_tail 4, dummy 10}; 335 steps (e0 280, e1 55) | PASS |

## Finding: tile-schedule planner kernels (blocks package 03)

Observation (`b3_verify/identity_summary.py`, `metadata_kernel_layers.py`, `pilot_layer_kernels.py`, `native_metadata_kernel.py`, `native_metadata_positions.py`, `planner_durations.py`):

- In every PIECEWISE and FULL decode step of the B2 pilot, the expanded kernels differ from the closest native candidate only by `smxx::decode::get_mla_metadata_kernel(GetDecodeSchedMetaParams)` (39 extra) and by grid-only differences from shape.
- Pilot: the planner kernel runs once in every one of the 6 layers (grid (1,1,1), block (32,1,1), scope `dsv4_sparse_attention`), about 26.1 µs each. The expansion therefore gives 43 planners per step.
- Native P2-32 decode lane: exactly 4 per forward (decode FULL 58 forwards, dummy FULL 59), before attention kernels 0, 1, 2 and 3. Layers 0-3 plan; layers 4-42 reuse. Median duration 25.73 µs.
- Overcount: about 39 × 26 µs ≈ 1 ms per decode step.

Cause (vLLM 0.28.0 source, `2cf0a69`):

- `nvidia/flashmla.py:225-254` and `sparse_swa.py` `build_tile_scheduler`: one `FlashMLASchedMeta` per layer type (SWA-only, C4A, C128A) in each SWA attention-metadata object. The first call of each type launches the planner; later calls of that type reuse it.
- Metadata comes from the layer's SWA cache group (`attention.py:623`, `attn_metadata.get(self.swa_cache_layer.prefix)`), so planning is per (SWA KV-cache group, layer type).
- `kv_cache_utils._get_kv_cache_groups_uniform_groups` splits the SWA layers into `cdiv(N, d)` groups taken with stride (`layer_tuples[i::G]`). Here `d = _approximate_gcd([full-MLA tuples, N, C4-state tuples, C128-state tuples], lower_bound=full-MLA tuples)`.
- Full model: values [21, 43, 21, 20] give d = 22 and 2 SWA groups (even and odd layers). Planners: layers 0, 1, 2 and 3.
- Native serving inventory (`dsv4-h200-p2-canonical-20261025-01`, prefill rank 0, `cache.kv_cache_groups`) confirms this. There are 5 groups: full MLA (62 names), SWA even (0, 2, …, 42), SWA odd (1, 3, …, 41), C4 states (42 names) and C128 states (20 names). Script: `b3_verify/native_kv_groups.py`.
- 6-layer reduced engine: values [2, 6, 2, 2] give d = 2 and 3 SWA groups ({0, 3}, {1, 4}, {2, 5}). Every layer plans, which matches the pilot trace.
- Frontier's rule "the first layer of each variant plans" also disagrees with native: the uncompressed layer 1 plans natively.

Consequence: the 6-layer reduced engine has no reuse layer for any variant, so its rows cannot price the 39 reusing layers. Package 03 is held until the fix (assignment B4) is verified.

## Limits

- All checks are CPU replays of the B2 GPU data. The step hold, the pilot phases and DP-peer emulation have not run on a GPU yet.
- Without a hold the B2 data cannot pass the A4 gate (38 groups have no host-ahead call). The gate's pass criterion is first tested by package 03.
