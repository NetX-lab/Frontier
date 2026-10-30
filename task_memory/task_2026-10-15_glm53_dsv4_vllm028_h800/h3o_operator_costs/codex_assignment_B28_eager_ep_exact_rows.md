# Task: price the eager EP calls from the isolated measurements, and price each uneven EP dispatch from its own measured row (stage 4 assignment B28, issue I-008, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Created from issue I-008 and the revised decision of the G5h report. |

## Goal

Frontier prices each collective of a DeepSeek-V4 (V4) or GLM forward with the `measured` CC backend (`frontier/cc_backend/backends/measured_cc_backend.py`), which reads a table of collective times measured on the serving hardware. Two parts of that pricing are wrong for the eager prefill EP calls. Both are documented in issue I-008 and in the G5h report (paths under "Scope and context").

1. **Wrong timing mode in the H200 table.** The table `data/profiling/network/h200_dgx/summary.csv` holds back-to-back timings: the benchmark times 10 calls in a row. In a native eager prefill forward, consecutive EP calls are milliseconds apart, and each call starts when the last rank arrives. The isolated runs (one aligned call per repeat) reproduce the native critical path within 1-18 µs; the back-to-back timings miss it by −160 to +260 µs. Native critical path per layer, against the isolated and back-to-back medians:

   | Call (lane vector) | Native | Isolated | Back-to-back |
   | --- | --- | --- | --- |
   | EP32 combine (`eager_t6144_1_1_1`) | 2,171.2 µs | 2,172.0 µs | 2,011.1 µs |
   | EP32 dispatch (`eager_t6144_1_1_1`) | about 1,048 µs | 1,043.0 µs | 1,031.9 µs |
   | EP16 combine (`eager_t6144_1`) | 717.0 µs | 734.6 µs | 763.9 µs |
   | EP16 dispatch (`eager_t6144_1`) | 541.8 µs | 560.2 µs | 801.8 µs |

   For decode (CUDA graph) EP calls, TP calls and the 1-token eager EP rows, back-to-back is the closer mode. Those rows stay as they are.

2. **Uneven dispatch reads the uniform curve.** When the DP lanes hold different token counts, the sequence-parallel (SP) rows of the EP ranks differ, and vLLM runs a different NCCL kernel (`AllGatherV` instead of `AllGather`). The combine already handles this: an uneven combine is priced only from its own measured row (exact lane vector and bytes), and an unmeasured vector raises `MeasuredCCOutOfRangeError`. The dispatch does not. It reads the uniform curve at the largest lane's bytes, so a `(6144, 1)` dispatch on EP16 gets the price of the uniform `eager_t6144_6144` row (801.7 µs) instead of its own measured `eager_t6144_1` row (560.2 µs). The W7b report (`test_report_2026-10-20_w7b_sp_collectives.md`, line 209) already recorded that this rule can exceed 10% error above 512 tokens.

The lead's arithmetic for one 6,144-token prefill forward (43 layers), from `h3o_operator_costs/evidence/collective_kernels/expected_prices.txt`:

| Case | Dispatch, current → planned | Combine, current → planned | Change per forward |
| --- | --- | --- | --- |
| P2-32 (EP16, lanes `(6144, 1)`) | 801.7 → 560.2 µs | 763.9 → 734.6 µs | −11.64 ms |
| P2-64 (EP32, lanes `(6144, 1, 1, 1)`) | 1,076.0 → 1,043.0 µs | 2,011.1 → 2,172.0 µs | +5.50 ms |

## Decisions (lead, final)

- **D1, table.** In the H200 table, replace every EP-group row of an eager scenario that has a 6,144-token lane with the matching row of the isolated run. EP16 rows come from G5h-16 isolated `_01`, EP32 rows from G5h-32 isolated `_01`. Every other row stays byte-identical.
- **D2, consumer.** An uneven EP dispatch is priced from its own measured `AllGatherV` row, exactly as the uneven combine is priced from its own `Reduce` row. This applies to both models: the V4 operator `dsv4_ep_dispatch_allgather` and the GLM operator `expert_parallel_alltoall_dispatch`. Selection stays spec-driven (collective alias, communication domain and implementation), never by operator name.
- **D3, no fallback.** An uneven dispatch whose lane vector or bytes have no measured row raises `MeasuredCCOutOfRangeError`, as the combine does today. Do not fall back to the uniform curve, scale a nearby row or interpolate between vectors.
- **D4, equal SP rows.** Rows whose raw token counts differ but whose SP rows are equal (for example `t2_r2` and `t8_r8` on H800, which run `AllGather`) stay holdouts, as they are for the combine today.
- **D5, single rule owner.** The rule "this request is priced from an exact measured row" has one owner, in `measured_cc_backend.py`. The A10 observer (`h3o_operator_costs/a10/a10_observer.py`) duplicates the combine-only version of this rule today; make it use the backend's rule instead.

## Work

### 1. Rebuild the H200 table

Write `b28/build_table.py`. It reads the base table, from the base worktree named under "Scope and context", and writes the table in the working root.

- **Input checks.** Assert these SHA-256 values before any change:
  - base table: `f4d52911770cc39939dddc0fa98ce30b530b1cd60ea2bd94070d293e200ca559`;
  - G5h-16 isolated: `e26e697a948c8f1dcb9a5190cd828450219697b254c4d1205d163e47a5719886`;
  - G5h-32 isolated: `8ed9d86330cf7c462867efc23e83a7e2d23bc7d8eb7a1a8e50b9150c1bc907a0`.
- **Selection.** Select the rows with `group == "ep"`, `cuda_graph == "False"`, and a lane of 6,144 tokens in `tokens_per_dp_rank`. The lead counted 108 rows: 54 `moe_dispatch_all_gatherv` and 54 `moe_combine_reduce_scatterv`. That is 18 EP16 rows (3 scenarios, 2 calls, 3 passes) and 90 EP32 rows (15 scenarios, 2 calls, 3 passes). Assert these counts.
- **Matching.** The key is `(group_size, pass, scenario, call)`. Each selected row must have exactly one isolated row with the same key. Assert that these columns are identical between the two rows: `op`, `group`, `group_size`, `cuda_graph`, `calls_per_forward`, `tokens_per_dp_rank`, `max_input_bytes`, `max_output_bytes`, `kernels`. The lead found no difference.
- **Write.** Replace each selected line with the isolated file's line, byte for byte. Keep the CRLF line endings, the header, the row order and the row count (384). Every other line stays byte-identical.
- **Record.** Write `b28/table_check.json` with:
  - the input and output SHA-256 values;
  - the counts by group size and call;
  - for each replaced row: its 0-based data-row index, its key, and its old and new `median_us`.
- **Provenance.** Update `data/profiling/network/h200_dgx/summary_provenance.json`:
  - the new `sha256` of the table;
  - a `rule` text that adds the replacement rule;
  - one new `sources` entry per isolated run. Each entry has the same fields as the existing entries, plus `stage_file`, `stage_commit` and `replaced_rows` (the 0-based data-row indices). Use `replaced_rows` instead of `output_rows`.
  - The two existing entries keep their `output_rows`.
  - Values for the new entries:

  | Field | G5h-16 isolated `_01` | G5h-32 isolated `_01` |
  | --- | --- | --- |
  | `stage` | `g5h16_tp8_dp2_ep16_isolated_01` | `g5h32_tp8_dp4_ep32_isolated_01` |
  | `stage_file` | `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/stage_g5h16_isolated_01.json` | `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/stage_g5h32_isolated_01.json` |
  | `stage_commit` | `895f61af` | `70790174` |
  | `rjob` | `exp-1029-012627-132543` | `exp-1030-234535-245357` |
  | `local` | `/data/ycfeng/tmp/glm53_dsv4/h200/g5h16_iso_01/download/summary.csv` | `/data/ycfeng/tmp/glm53_dsv4/h200/g5h32_iso_01/download/summary.csv` |
  | `cloud` | `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/g5h16_tp8_dp2_ep16_isolated_01/summary.csv` | `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/g5h32_tp8_dp4_ep32_isolated_01/summary.csv` |
  | `topology` | `TP8 DP2 EP16` | `TP8 DP4 EP32` |

### 2. Price each uneven EP dispatch from its measured row

In `measured_cc_backend.py`:

- Generalize the exact-row path from EP `reduce_scatterv` to EP `all_gatherv`. Today it consists of `_retained_combine_rows`, `unequal_samples`, `_unequal_combine_times`, the routing in `predict_operator_collective`, and `_predict_unequal`.
- Hold the per-call evidence in one table entry per call, not in a second copy of the functions:
  - **Combine (`reduce_scatterv`, `ep`).** Kernels start with `ncclDevKernel_Reduce_Sum_bf16_`. Input bytes are the gathered rows times the row bytes. Output bytes are the largest local rows times the row bytes. The row bytes are a multiple of the BF16 element size. These are today's checks; keep them unchanged.
  - **Dispatch (`all_gatherv`, `ep`).** Kernels start with `ncclDevKernel_AllGatherV_`. Input bytes are the largest local rows times the row bytes. Output bytes are the gathered rows times the row bytes. The row bytes are an integer.
  - Derive both geometries from the producer, `ep_workload` in `frontier/operators/sequence_parallel_moe.py`. Check them against both checked-in tables.
  - Example: H200 `eager_t6144_1_1_1` dispatch has `max_input_bytes` 6,346,752 = 768 × 8,264 and `max_output_bytes` 50,972,352 = 6,168 × 8,264.
- Rows that fail these checks raise `ValueError` at load time, as the combine does now.
- Rename the attributes, functions and error text so that the names describe both calls. Today's error text, "no exact per-root Reduce measurement", is combine-specific. Update every user of the old names:
  - `tests/unit/test_measured_cc_backend.py`;
  - `tests/unit/test_v4_measured_collective_consumer.py`;
  - `tests/unit/test_v4_replay_census.py`, line 105;
  - `h3o_operator_costs/a10/a10_observer.py`, lines 75-81.
- Expose the rule "this `(curve key, SP rows)` request is priced from an exact row" once. Use it from `predict_operator_collective` and from `a10_observer.collective_request` (D5). The observer's output fields and meaning stay the same, except that an uneven dispatch is now reported as exact.
- Update the module docstring.

In `frontier/operators/families.py` and `frontier/operators/sequence_parallel_moe.py`:

- Give both dispatch specs the SP-rows builder that the combine uses:
  - the V4 spec at line 809: today `sp_rows_builder=(_combine_sp_rows if alias == "reduce_scatter" else None)`;
  - the GLM `expert_parallel_alltoall_dispatch` spec.
- Rename `_combine_sp_rows` and `combine_sp_rows` to a name that fits both calls.
- Check that the GLM dispatch alias path (`_ep_phase_alias`) and the non-measured backends keep their current results. They ignore `sp_rows` today.

### 3. Audit caches

Find every cache or memo that can hold a dispatch price or a forward time containing one, and whose key does not include the SP rows or lane vector. Candidates:

- the V4 `_prediction_memo` in `dsv4_predictor.py`, near line 324;
- the B24 EP-lane reuse memo;
- the A10 caches.

For each one, state whether it can now return a wrong price, with the evidence. If a fix is needed outside the writable paths, stop and report.

### 4. Tests

- **Replace** `tests/unit/test_glm_sp_moe_comm.py::test_unequal_dispatch_keeps_largest_rank_uniform_curve`. The new test checks that a GLM uneven dispatch on the H800 table uses its exact row. It also checks that this exact price differs from the uniform-curve price, so the test can tell the two rules apart.
- **Update `tests/unit/test_measured_cc_backend.py`.**
  - Update the counts: the `== 9` at line 447 and the `not backend._unequal_combine_times` at line 229.
  - Update the `_unequal_combine_row` helper and `test_uneven_dispatch_uses_caller_supplied_largest_rank_bytes`.
  - Add rejection tests for a dispatch row with a wrong kernel, wrong input bytes and wrong output bytes.
- **Update `tests/unit/test_v4_measured_collective_consumer.py`.**
  - `test_grouped_dispatch_and_combine_match_ordered_eager_grid` asserts the exact path for the dispatch as well.
  - `test_exact_combine_is_spec_driven_with_renamed_operator` covers the dispatch.
  - `test_h200_row_classes_are_separate` gets the new counts. Today: 16 exact entries and 48 holdouts, all dispatch. Derive the new counts and explain them in the report.
  - Add a test that an unmeasured uneven dispatch vector or size raises `MeasuredCCOutOfRangeError`.
- **Add the decision values.** Through the V4 operator path (`measured_collective_time` with a real context, as the consumer tests do), the checked-in H200 table prices:
  - EP16 `(6144, 1)`: dispatch 560.2 µs and combine 734.6 µs;
  - EP32 `(6144, 1, 1, 1)`: dispatch 1,043.0 µs and combine 2,172.0 µs;
  - each equal to the median of the three isolated rows, and to these values rounded to 0.1 µs.
- **Coverage check.** For both checked-in tables (H200 and H800), the ordered lane vectors (with group size) that have an exact dispatch row equal those that have an exact combine row. Report any difference. Name each vector that would now raise for the dispatch while its combine is measured. If such a vector exists on H800, a GLM run that worked before would now fail; report it, and do not add a fallback.
- No skip or `xfail`, and no weakened assertion. List each changed test line with its reason.

### 5. Price check and curve diff

Write `b28/price_check.py` and save its output as `b28/price_check.txt`. Run it once against the base worktree and once against the working root.

- It prices the four calls of the table in Work 4 through the operator path.
- It prints the per-forward change over 43 layers.
- **Expected base values:** 801.7, 763.9, 1,076.0 and 2,011.1 µs (the "current" values in `expected_prices.txt`).
- **Expected candidate values:** the isolated medians above.
- **Expected changes:** −11.64 ms for P2-32 and +5.50 ms for P2-64, to 0.01 ms.

Write `b28/curve_diff.py` and save its output as `b28/curve_diff.txt`. Load the base and candidate backends for both tables and list every difference: curve points (key, bytes, time), exact entries, and holdout counts.

- **Expected on H200:**
  - uniform-curve points change only where the replaced uniform rows sit: `eager_t6144_6144` on EP16 and `eager_t6144_6144_6144_6144` on EP32, for dispatch and for combine;
  - the existing combine exact entries change value only for the replaced scenarios;
  - the dispatch exact entries are new;
  - nothing else changes.
- **Expected on H800:** no curve or combine change; the dispatch exact entries are new.

### 6. Documentation

- In `docs/general/measured-communication.md`, replace the "Unequal SP row counts…" bullet, which says that dispatch keeps the largest-rank bytes, with the current rule for both calls.
- Keep the rest of the document unchanged.

## Acceptance

1. `b28/build_table.py` runs and asserts as described. The output table has 384 rows. `b28/table_check.json` lists 108 replaced rows. Every non-selected line is byte-identical to the base.
2. `b28/price_check.txt` shows the expected base values, candidate values and per-forward changes from Work 5.
3. `b28/curve_diff.txt` shows only the expected differences from Work 5.
4. The new and changed tests pass. Name and count them in the report.
5. **Regression, candidate against base.**
   - Selection: every `tests/unit` module that contains any of these strings: `measured`, `MeasuredCC`, `families`, `sequence_parallel_moe`, `sp_rows`, `get_comm_operator`, `dsv4`, `v4_`, `glm`, `a10`, `h200_dgx`, `h800_dgx`. Add the list `/data/ycfeng/Frontier/.worktrees/b24-v4-ep-lane-reuse/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b24/regression_targets.json`. Record the selection in `b28/regression_targets.json`.
   - Run the selection on the base worktree and on the working root, with the same command.
   - Compare the outcome of each test node. Every difference must be one of the intended test changes; list them. Any failure must also appear on the base side, with the same message.
   - Use a scratch working directory, as B24 did: a `data` symlink to the repository's read-only data, absolute test paths and the repository on `PYTHONPATH`. This keeps the tests' default `outputs` and cache writes in scratch.
6. `git diff --stat` shows only paths under "Writable paths". Each Python module stays at or below 2,000 lines.

Test command template, from the scratch working directory (`<root>` is the working root or the base worktree):

```bash
ulimit -v 31457280
env PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0 \
  TMPDIR=<scratch>/tmp FRONTIER_TMP_ROOT=<scratch>/tmp \
  PYTHONPATH=<root>:<root>/tests/unit:<root>/tests/comparison/calibration:<root>/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages \
  timeout 3600 /data/ycfeng/envs/frontier-py310/bin/python -m pytest <absolute test paths> -q -p no:cacheprovider \
  --basetemp=<scratch>/tmp/<label> --junitxml=<scratch>/<label>.xml
```

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/b28-eager-ep-exact-rows`. It is a Git worktree on branch `b28-eager-ep-exact-rows`, created from the main commit that adds this file. Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Base worktree (read-only):** `/data/ycfeng/Frontier/.worktrees/b28-base`, detached at the same commit.
- **Task records (read-only)** are in the main tree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, task directory `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/`. The same files are in your working root.
  - `issues.md`: I-008, with its "Measured" and "Repair (B28)" blocks.
  - `test_report_2026-10-28_g5h_collective_kernel_comparison.md`: the I-008 section and "Revised decision (2026-10-30)".
  - `h3o_operator_costs/evidence/collective_kernels/expected_prices.py` and `expected_prices.txt`: the lead's arithmetic. It mocks the planned rule.
  - `test_report_2026-10-20_w7b_sp_collectives.md`: the origin of the current dispatch rule.
- **Writable paths, in the working root:**
  - `frontier/cc_backend/backends/measured_cc_backend.py`;
  - `frontier/operators/families.py` and `frontier/operators/sequence_parallel_moe.py`;
  - `tests/unit/`: only the tests that the intended change affects, each listed in the report;
  - `data/profiling/network/h200_dgx/summary.csv` and `summary_provenance.json`;
  - `docs/general/measured-communication.md`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_observer.py`;
  - your records under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b28/`.
  - If you need another path, ask in the report; do not change it yourself.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b28/`.

## Restrictions

- The lead commits. Do not run Git commands that change state. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
- No GPU, network, job submission or package installs.
- No deletions or moves outside your scratch.
- Two A10 replays are running under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/`. Do not stop or signal them, and do not write there. Do not run recursive `grep` or `find` over `/data/ycfeng/tmp/glm53_dsv4/`. Read the two isolated source files only at the paths given above.
- No temporary patches, scaling factors, fallbacks or guards for states the code cannot reach. No literal that should come from the table or the operator spec.
- Python 3.10. Memory `ulimit -v 31457280`, and a `timeout` on every command.

## Expected return

- Report language: English. Write the report to `h3o_operator_costs/b28/report.md` and return its content.
- The diff summary, with the reason for each changed function, test line and data row group.
- The outputs of `build_table.py`, `price_check.py` and `curve_diff.py`, and the H200 and H800 coverage check.
- The cache audit (Work 3).
- The regression comparison: commands, counts and node-level differences.
- Deviations and their reasons, and limits.
