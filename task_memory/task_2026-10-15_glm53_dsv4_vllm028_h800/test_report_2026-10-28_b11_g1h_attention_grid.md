# Test report: B11 and B11b, the G1h attention grid (design entry 14, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead rerun of the B11b acceptance before the B11 + B11b commit. |

## Scope

- B11 (Codex `9abfeda9`, [h3o_operator_costs/codex_assignment_B11_g1h_attention_windows.md](h3o_operator_costs/codex_assignment_B11_g1h_attention_windows.md)): case builder, FCFS case prediction, coverage check and G1h planning files. Its full-grid check failed for two grid questions.
- Lead decision D-G1H-GRID-AMENDMENT ([requirements.md](requirements.md) item 35, [h3o_operator_costs/c1_coverage.md](h3o_operator_costs/c1_coverage.md) §4.1 amendment): the first-tail mix rule and the 127-token context allowance.
- B11b (Codex `ebbfa1e0`, [h3o_operator_costs/codex_assignment_B11b_g1h_grid_amendment.md](h3o_operator_costs/codex_assignment_B11b_g1h_grid_amendment.md); report `h3o_operator_costs/codex-implement-ebbfa1e0-dd55-490b-b1d1-4d453d7fc0a4/implementation.md`): the amendment, the five residual placements and review findings R1-R5.

## Environment

- Working root `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, HEAD `eaee3550` with the B11, B11b and (separate, uncommitted) B12/B12b changes in the tree.
- `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6, no torch or vLLM); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`; `ulimit -v 31457280`. Scratch `/data/ycfeng/tmp/b11b_review/`.

## Lead rerun

| # | Command (from the working root) | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `pytest tests/unit/test_dsv4_pilot_cases.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsa_extend_profiling.py tests/unit/test_dsa_query_one_profiling.py tests/unit/test_dsv4_features.py tests/unit/test_dsv4_replay_rows.py tests/unit/test_dsv4_kernel_identity_check.py tests/unit/test_dsa_full_dummy_profiling.py tests/unit/test_v4_attention_coverage.py -q -p no:cacheprovider` | all pass; the optional pinned-native Git-object check may skip | 617 passed, 1 skipped (`test_dsa_full_dummy_profiling.py:166`, needs `FRONTIER_VLLM_BS_ROOT`) | PASS |
| 2 | `python h3o_operator_costs/g1h/build_cases.py`, then `cmp` of the four generated JSON files with the Codex copies | exit 0; byte-identical (deterministic) | exit 0; `pilot_cases.json`, `coverage_grid.json`, `gpu_time_estimate.json`, `package_plan.json` identical | PASS |
| 3 | `tests/analysis/v4_attention_coverage.py cases --package-plan g1h/package_plan.json`, then `check --predicted --require-complete-grid --c1-json evidence/coverage/{p2_32,p2_64,p0}.json` | exit 0; zero missing windows, outside steps and mismatches; every native serving class inside | exit 0; 989 measured steps; 1,041 windows predicted; 0 mismatches, 0 missing, 0 outside; native classes P2-32 43/43, P2-64 54/54, P0 48/48; all 7 multi-tail coordinates inside | PASS |
| 4 | Codex compatibility script `/data/ycfeng/tmp/codex-implement/ebbfa1e0-dd55-490b-b1d1-4d453d7fc0a4/b11b/check_compatibility.py` (baseline `dsa_plan.py` read with `git show 5083d036`) | package 03 plan byte-identical; 64 GLM plans equal | `package03_bytes_equal: true` (11,881 bytes, SHA-256 `8bb82d07…6011`), `glm_equal: 64` | PASS |
| 5 | `/data/ycfeng/tmp/unused_imports.py` on the changed attention modules and the coverage driver; `git diff --check` | `[]`; exit 0 | `[]` for each; exit 0 | PASS |

Codex-reported checks not rerun by the lead: the package 03 raw-record coverage check (21 covered windows, 72 steps outside the grid, all 175 predicted steps matched) and the cold-import check (no `torch`, `vllm` or `cuda` module loaded).

## Grid result

| Category | Windows | Cases | Measured steps |
| --- | ---: | ---: | ---: |
| FULL serving | 351 | 336 | 336 |
| PIECEWISE without tails | 351 | 336 | 336 |
| First tail | 318 | 296 | 296 |
| Multi-tail | 7 | 7 | 7 |
| Dummies | 13 | 13 | 13 |
| Prefill | 1 | 1 | 1 |
| Total | 1,041 | 989 | 989 |

The mix rule removes 33 duplicate first-tail windows from B11's 1,074. The five residual placements were corrected in the construction, not in the grid:

- W24 / own 18 / high: the continuing requests are placed from their first pure decode state before the tail chunk.
- W8 and W16 / own 6 / low: the fixed point cycled; the nearest phase-preserving adjustment moves the non-writer from 6,144 to 6,146, inside the allowance.

The full case prediction still has to reach the window before a case is admitted.

## GPU time estimate (CPU model from the P1h package 03 timings)

0.98 GPU-hours for the admitted grid (engine 0: 0.52, engine 1: 0.32, engine 2: 0.14); envelope with one case per window 1.08. Below the C1 G1h share (1.5) and the G1h-G4h cap (7).

## Limits

- CPU evidence only: no GPU kernel identity for engine 2 (budget 24,576). Its larger allocation can change the block-zeroing work, which is a pre-DP operator, not an attention scope.
- One measured step per case gives no repeated-window variance.
- The optional pinned-native Git-object test is skipped.
