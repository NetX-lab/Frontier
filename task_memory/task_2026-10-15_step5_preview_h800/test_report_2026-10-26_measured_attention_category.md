# Test report: measured Step attention priced within its measured category (D31 step D(a))

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: unit-test triage against the baseline and the 77-scenario fidelity comparison. |

## Change under test

- `frontier/execution_time_predictor/step5.py`: `STEP5_CATEGORY_COLUMNS` lists the columns that
  split measured rows into comparable groups (layer, attention variant, phase, graph mode,
  attention-metadata flag, native workload code). The coverage domains use it.
- `frontier/execution_time_predictor/step5_cost.py`:
  - A layer that was not profiled is priced from the first measured layer with the same physical
    attention block (`step5_attention_signature`). A layer with no such measured layer raises
    "no measured layer with the same attention block".
  - A query is predicted only when its category was measured and every feature lies inside that
    category's measured range; otherwise it raises. Before, the predictor needed an exact
    measured shape for every layer and shape.
- Tests: `tests/unit/test_step5_measured_replay.py` (new test
  `test_unmeasured_layers_and_shapes_are_priced_within_the_measured_category`, updated error
  strings), `tests/unit/test_step5_measured_training.py` (`_row` gains `block_table_columns`),
  `tests/unit/test_step5_measured_ffn_replay.py` (see below).

## Environment

- Unit tests: `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python` (Python 3.10;
  no `torch` and no `aiohttp`), `PYTHONPATH=<worktree>:<worktree>/tests/unit`.
- Fidelity: `/data/ycfeng/envs/frontier-py310/bin/python`, `PYTHONPATH=<worktree>`.
- Baseline worktree: `/data/ycfeng/Frontier/.worktrees/step5-d31-native-e275c83` (commit
  `e275c83a`).

## Results

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| All Step unit-test files on the candidate (`/data/ycfeng/tmp/d31_pytest/run_all4.sh`; it skips the `canonical_native`, `linear_op_activation` and `native_position_cyclic` files, which need `aiohttp` or GPU packages) | Every failure also fails on the baseline | 133 failed, 2583 passed, 205 skipped (`/data/ycfeng/tmp/d31_pytest/all4.log`) | see triage |
| The 133 failed IDs on the baseline (`/data/ycfeng/tmp/d31_pytest/run_base_failed.sh`) | Same IDs fail | 132 failed, 1 passed (`/data/ycfeng/tmp/d31_pytest/base_failed.log`) | 1 candidate-only failure |
| Candidate-only failure, after the test update | Pass | `test_step5_measured_ffn_replay.py`, `test_step5_measured_replay.py`, `test_step5_measured_training.py`: 54 passed | PASS |
| Fidelity matrix `tests/e2e/refactor_fidelity/run_matrix.py` against baseline `e275c83` | 77/77 identical | 77/77 identical (`/data/ycfeng/tmp/d31_fidelity_da/comparison.json`, log `/data/ycfeng/tmp/d31_fidelity_da.log`) | PASS |

Triage of the candidate-only failure: `test_attention_cache_preserves_proof_and_does_not_invent_layer_coverage`
asserted that layer 1 raises "coverage is unavailable" when only layer 0 was measured. Layers 0
and 1 are both sliding-window layers with the same attention block, so the new behaviour prices
layer 1 from layer 0, as intended. The test is renamed
`test_attention_cache_preserves_proof_and_does_not_invent_block_coverage`. It now checks that
layer 1 gets layer 0's operator costs and that layer 3, a sparse (DSA) layer with a different
block, raises "no measured layer with the same attention block".

The 132 failures shared with the baseline are in `test_step5_native_reference` (85),
`http_capture` (15), `canonical_g3` (14), `linear_op_ownership` (4), `request_free_profiling` (4),
`model_config` (3), `native_pd` (3), `structural_execution` (2), `boundary_probe` (1) and
`native_pd_r2_stage` (1). They fail the same way without this change, so they are outside step D.

## Limits

- The fidelity matrix exercises the existing dense/MoE/co-location/PDD scenarios. It does not
  run a Step-5 measured simulation, because the H200 Step-5 profile tables do not exist yet
  (step D(g)). The new range check is covered only by the unit tests until then.
- The predictor-cache naming check of `run_matrix.py` (`--clean-cache`) was not run, because it
  deletes `<repo>/cache`.
