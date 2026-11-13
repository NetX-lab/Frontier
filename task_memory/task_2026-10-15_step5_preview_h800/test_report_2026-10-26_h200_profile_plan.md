# Test report: H200 Step-5 profile plan and inputs (D31 step D(d)/(e))

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | First version: H200 profile plan, role inputs, DP-lowered graph mode, DSA last-region pricing. |

## Scope

- D(d): one profile plan for both H200 topologies (`h200_32`: TP8/DP2/EP16, `h200_64`: TP8/DP4/EP32)
  and the attention cells of the two role packages.
- D(e): the role package builder (`frontier/profiling/step5/job.py`) takes device, attention TP and
  expert-parallel sizes from the plan instead of H800 literals.
- Two model changes the plan needs:
  - The profiler accepts an explicit graph mode lower than the native dispatch of the cell. The DP
    forward-start agreement runs every lane at the lowest lane mode, so a decode lane runs
    PIECEWISE when another lane holds a prompt tail.
  - DSA contexts. Repeated DSA timing needs region-aligned contexts (multiples of
    `region_block_size` = 8), so the highest measured DSA context is 7160 while decode reaches 7166.
    A DSA context up to one region above the highest measured context is now priced at that
    context. The lower bound and all other columns stay exact.

## Changed files

- `frontier/profiling/step5/h200_profile_plan.json` (new), `h200_prefill_inputs.json` (6 cells),
  `h200_decode_inputs.json` (160 cells).
- `frontier/profiling/step5/p0_profile_plan.json`: `serving_topology_per_role` becomes
  `serving_topologies_per_role` keyed by case, plus `device`. Plans written with the old key, such as
  the D29 bound H800 plans, no longer load; none of them is used by the current plan.
- `frontier/profiling/step5/job.py`, `frontier/profiling/step5/inputs.py`,
  `frontier/execution_time_predictor/step5_cost.py`,
  `tests/integration/step5_profiling/verify_fixed_probe.py`.
- New scripts: `tests/integration/step5_profiling/h200_profile_inputs.py` writes the two input files;
  `tests/integration/step5_profiling/h200_profile_coverage.py` checks them.
- Tests: `tests/unit/test_step5_attention_profiling.py`, `tests/unit/test_step5_measured_replay.py`,
  `tests/unit/test_step5_request_free_profiling.py`.

## Decode cell grid (per layer: SWA layer 0, DSA layer 3)

Contexts: low 6144, middle 6656, high 7166 (layer 0) or 7160 (layer 3); prompt tail context 6128.

| Group | Cells | Content |
| --- | ---: | --- |
| FULL decode | 27 | n in {1,2,4,8,16,32,64,128} at low and high; n in {8,32,128} at middle; n = 3, 50, 100; 5 cells padded to a larger agreed width |
| PIECEWISE decode | 9 | decode-only lanes run PIECEWISE beside a lane with a tail |
| Mixed PIECEWISE | 16 | n decodes plus 1–3 tails of 16 tokens |
| Tail-only PIECEWISE | 6 | 1–3 tails |
| FULL dummy | 11 | widths 1–128 |
| PIECEWISE dummy | 11 | widths 16–256 |

PREFILL: eager 6128 and 6144 prompts plus one eager dummy slot per layer.
MoE token counts cover W, 2W and 4W of each width W, because it is not settled whether Frontier
prices MoE with per-lane or DP-summed tokens.

## Commands and results

Environment: `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python` (Python 3.10) for unit
tests and CPU checks; `/data/ycfeng/envs/frontier-py310/bin/python` for the fidelity matrix.

1. CPU coverage check (`/data/ycfeng/tmp/d31_h200_grid/run.sh`, now
   `tests/integration/step5_profiling/h200_profile_coverage.py <repo> <max tails>`).
   Every reachable lane state of the fixed workload is mapped to attention features and compared with
   the range of the planned cells of the same category.
   - At most 3 prompt tails per decode step: 48,364 checks, **PASS**.
   - First attempt failed on two ranges (mixed tokens 159 > 158, padded rows 231 > 224); fixed by adding
     mixed cells (125 decodes + 3 tails, high, width 176) and (1 + 1, middle, width 256).
   - 4 tails per step: **FAIL** (tail tokens 64 > 48, mixed tokens 174 > 173, tail batch 4 > 3). This is
     a known limit of the plan. Frontier raises an explicit range error if a run reaches it.
2. Dry-run package description (`/data/ycfeng/tmp/d31_h200_grid/describe.sh`):
   - PREFILL: H200, `step_main`, 1 GPU, non-preemptible, timeout 14400 s, base image; 6 event rows and
     6 GPU-active rows.
   - DECODE: same resources; 160 event rows, 0 GPU-active rows.
   - Both: commands run linear and MoE for each method, then attention; `--expert_parallel_sizes 16 32`
     and `--device h200` are present; no `h800`; no native binding. PASS.
3. Unit tests (`/data/ycfeng/tmp/d31_pytest/run_dde.sh`): 9 Step-5 suites.
   - First run: 386 passed, 6 failed, 1 skipped. Five failures are on the known baseline list
     (`/data/ycfeng/tmp/d31_pytest/all4_failed.txt`, `torch` is not installed). The sixth was the new DSA
     test: the range check passed, but the fitted model's own declared feature maximum (8184) rejected
     8190. The first change only relaxed the range check.
   - Fix: the context columns of a DSA query in the last score region are priced at the highest
     measured context, and the cost is marked as not an exact lookup. The test now asserts that the
     price at 8190 equals the price at 8184.
   - Final run: **387 passed, 5 failed, 1 skipped**; all 5 failures are on the known baseline list.
     New test `test_h200_packages_cover_both_topologies_from_one_plan` passes.
4. Fidelity matrix (`/data/ycfeng/tmp/d31_fidelity_dde_run.sh`, `tests/e2e/refactor_fidelity/run_matrix.py`,
   baseline worktree `step5-d31-native-e275c83`, both sides run fresh, output
   `/data/ycfeng/tmp/d31_fidelity_dde/`): **77/77 identical**, no failures. Predictor cache names are
   not compared, because a clean-cache run needs approval; this matches the earlier D(a)–D(c) runs.

## Limits

- The cells are not yet measured on a GPU; D(g) runs them.
- More than 3 prompt tails in one decode step is outside the plan (see check 1).
- The DSA last-region rule assumes the cost within one score region is flat; the profile measures only
  aligned contexts, so this is not measured.
