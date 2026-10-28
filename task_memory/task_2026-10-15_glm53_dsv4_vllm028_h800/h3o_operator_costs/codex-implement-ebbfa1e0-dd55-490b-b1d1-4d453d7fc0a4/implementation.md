**1. Assignment and outcome**

B11b applies decision D-G1H-GRID-AMENDMENT, resolves the five residual first-tail placement failures, fixes review findings R1–R5, regenerates the G1h planning files, and verifies the CPU acceptance conditions. Outcome: **implemented**. Final review and acceptance remain with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, observed HEAD `eaee355046ea809b716f2e9abf8b00d18fa72cea`.

The authorized write scope was the assigned attention modules under `frontier/profiling/attention/`, `tests/analysis/v4_attention_coverage.py`, `tests/unit/test_v4_attention_coverage.py` and applicable B11 test files, and `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h/`. All changes stay within that scope. Below, `H3O` means `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs`, and `G1H` means `H3O/g1h`.

Read the applicable root and parent AGENTS.md instructions, B11 and B11b assignments, B11's implementation report, requirements item 35, the C1 §4.1 amendment, issue I-001, the existing planning and prediction code, and relevant design/replay/source evidence. No applicable CLAUDE.md was found.

Settled decisions preserved:

- With t first tails, each mix becomes `(max(n4, t), max(n128, t))`; duplicate windows are removed while preserving their first grid occurrence.
- The context allowance is defined once from the C1 ratios: `max(ratios) - 1 = 127`. Continuing contexts remain inside 6,144–7,166; cached 6,143 is reserved for first tails.
- Each generated case has zero warm-up steps and one measured step.
- Native engine limits, the requested amended grid, and the existing prediction/admission mechanism remain authoritative.
- `G1H/package_plan.json` retains B11's source revision `5083d0369a5f9b86c333b30fce78154a3364225f`.

No GPU operation, network access, installation, package build, submission, Git mutation, commit, deletion, move, task-record update outside G1H, or delegation was performed.

**2. Changes**

All 12 changed files existed as B11 work when this assignment began. B11b adds no new repository file. A before snapshot and a diff isolating B11b are retained in scratch.

| Changed file | Change and reason |
| --- | --- |
| `frontier/profiling/attention/dsa_vllm_runner.py` | Moves `requests_match_case` to the module imports at line 46. Removes its function-local CPU import in `_WorkerState.begin_step`. GPU imports remain lazy. |
| `frontier/profiling/attention/v4_attention_windows.py` | Defines the decision and derived allowance in C1; amends mixes and removes duplicates in `required_windows:73`; uses existing phase/mode constants; removes allowance parameters from `matching_profiles:102` and `assign_windows:141`. |
| `frontier/profiling/attention/v4_attention_cases.py` | Moves prediction and pilot-loader imports to the module top; uses existing constants; removes the impossible-tail guard and hard-coded boundary message. Shared candidate helpers begin at line 21. `_context_adjustments:89` provides phase-preserving adjustments when inversion cycles. `_place:105` predicts continuing construction before tail construction, then verifies the full case. `build_attention_cases:155` uses the fixed C1 allowance and records the decision. |
| `tests/analysis/v4_attention_coverage.py` | Removes the allowance parameter from `check_steps:80`, removes `--context-tolerance` and package-plan fallback handling from `main:146`, and reads the allowance from C1. Streaming record comparison is preserved. |
| `tests/unit/test_v4_attention_coverage.py` | Updates B11 expectations for the complete amended grid and fixed allowance. Adds mix/order/deduplication tests at line 379, allowance boundary tests at line 404, and all five residual-placement tests at line 443. Safely mocks package creation when testing the now-complete package wrapper. |
| `G1H/build_cases.py` | Removes the tolerance CLI option and old package-plan setting; removes fixed execution blockers. Regenerates complete files, category counts, shared-case counts, and timing. Per-engine timing records now use `engines` rather than B11's `executable_subset`; the full-grid envelope remains available. |
| `G1H/prepare_package.py` | Removes dependence on the deleted `execution_blockers` field. Gates package creation on `coverage_grid.json` completeness at line 22. This was necessary to keep the wrapper compatible with R4. Package creation was only mocked in tests. |
| `G1H/pilot_cases.json` | Regenerated with 989 admitted cases across three engines. Every case has one measured step and zero warm-up steps. |
| `G1H/coverage_grid.json` | Regenerated with all 1,041 amended windows predicted and `complete: true`; fixed allowance and D-G1H-GRID-AMENDMENT recorded; provisional status removed; category and sharing counts included. |
| `G1H/gpu_time_estimate.json` | Recomputed per engine and in total from the P1h timing inputs. Includes the admitted full-grid run and a conservative envelope charging a separate case per window. |
| `G1H/package_plan.json` | Regenerated run lists; removed `coverage_context_tolerance` and fixed `execution_blockers`; preserved source_revision and package destination settings. |
| `G1H/README.md` | Rewritten in plain English for the complete grid, with counts, decision reasons, placement corrections, multi-tail construction and native evidence, GPU estimates, reproduction commands and remaining GPU limitations. |

`dsa_plan.py` and `dsa_case_prediction.py` were inspected and verified byte-identical to their assignment-start contents. Other B11 acceptance test files were unchanged. Concurrent B12 work in `frontier/moe_ep_workload.py`, `frontier/profiling/moe/`, MoE/routing tests and G3h was preserved. The existing `.cortexkit/`, `.pi/` and `remaining_measurements/` directories were left alone. The lead's modified requirements and C1 task records were read without editing.

There is no deviation from the amended grid or engine limits. Removing the package wrapper's stale blocker-field dependency is a related correction required by R4. The generated timing-key change is identified above for review.

**Residual placement findings and results**

The complete native request vectors, prompt vectors, dispatches and writers are in `scratch/handoff_details.json`; the failing construction traces are in `scratch/residual_before.json`. Request indices below are zero-based. **Every request in every listed measured step has query tokens = 1.** All five steps dispatch PIECEWISE at their requested width.

| Window | Cause found | Correction | Predicted measured cached tokens and writers |
| --- | --- | --- | --- |
| W8, own6, low, mix (5,1) | Fixed-point inversion alternated between six and seven construction steps around the 6,143-token chunk cap. | Predict continuing construction separately; detect the repeated prompt vector and try the nearest phase-preserving context adjustment. Move the non-writer from 6,144 to 6,146. | Cached vector `[6146,6147,6147,6147,6147,6143]`. Ratio-4 writers: requests `[1,2,3,4,5]`; ratio-128 writer: `[5]`. Case `g1-e1-0787-first_tail`; 7 estimated engine steps. |
| W16, own6, low, mix (5,1) | Same construction cycle as W8; synchronized width changes dispatch, not FCFS request construction. | Same adjustment and full prediction verification. | Cached vector `[6146,6147,6147,6147,6147,6143]`. Ratio-4 writers `[1,2,3,4,5]`; ratio-128 writer `[5]`. Case `g1-e1-0853-first_tail`; 7 steps. |
| W24, own18, high, mix (1,1) | The inverse update changed construction duration by one step. Early requests reached 7,167 before the tail's measured step and no longer scheduled, producing zero matched steps. | Invert the continuing requests' first pure decode state, accounting for the intervening tail chunk before verifying the full first-tail case. Original measured contexts are preserved. | Requests 0–16: cached 7,166; request 17: cached 6,143. Both ratio writer lists are `[17]`. Case `g1-e1-0960-first_tail`; 18 steps. |
| W24, own18, high, mix (7,1) | Same overshoot; early requests at the top context stopped during tail construction. | Same construction-first prediction correction; original contexts preserved. | Requests 0–5: cached 7,163; requests 6–16: cached 7,166; request 17: cached 6,143. Ratio-4 writers `[0,1,2,3,4,5,17]`; ratio-128 writer `[17]`. Case `g1-e1-0961-first_tail`; 18 steps. |
| W24, own18, high, mix (4,4) | Same overshoot among the high non-writing requests. | Same correction; original ratio-128 boundary contexts preserved. | Requests 0–2: cached 7,039; requests 3–16: cached 7,166; request 17: cached 6,143. Both writer lists are `[0,1,2,17]`. Case `g1-e1-0962-first_tail`; 18 steps. |

Native evidence: pinned vLLM source `/data/ycfeng/tmp/opencode/vllm-v0.28.0`, verified revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`. In `vllm/v1/core/sched/scheduler.py`, lines 525 and 751 process running requests before waiting requests; lines 563–577 cap work by the chunk threshold, budget and `max_model_len - num_computed_tokens - num_sampled_tokens_per_step`; lines 952–966 cap waiting chunks; lines 1380–1397 advance computed tokens. These rules explain the cycles and the high-context stop. The native prediction was not weakened to admit a failing case.

The multi-tail construction remains engine 2 with budget 24,576 and prefill chunk cap 6,143. All tail chunks fit together beside continuing decodes, and the next measured step contains every tail at cached 6,143/query 1. Query-one attention branches and actual-row planning are supported by `sparse_swa.py:531,712–788`, `flashmla.py:148–169`, compressor slicing and scratch slicing. The larger allocation can change zeroing work at `sparse_swa.py:551–552`; construction workspace, allocation and autotuning can also differ. CPU evidence establishes the request shapes and branches, not GPU kernel identity.

**3. Verification**

All checks used Python **3.10.6** at `/data/ycfeng/envs/frontier-py310/bin/python`, `ulimit -v 31457280`, a timeout on every command, disabled bytecode, and scratch test outputs. The following aliases expand the executed paths:

```bash
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
B11B_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python
B11B_SCRATCH=/data/ycfeng/tmp/codex-implement/ebbfa1e0-dd55-490b-b1d1-4d453d7fc0a4/b11b
B11B_H3O=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs
B11B_G1H="$B11B_H3O/g1h"
export TMPDIR="$B11B_SCRATCH"
```

| Acceptance | Result | Expected and observed |
| --- | --- | --- |
| 1. Package 03 unchanged | **PASS** | Rebuilt JSON and serialized bytes equal the frozen package 03 plan. 11,881 bytes; SHA256 `8bb82d07eedaea7382bb39d25e3356417fee30ce7121bbd480beff711f406011`. |
| 2. GLM plans unchanged | **PASS** | All 64 plans compared by B11 remain byte-identical to the same pre-B11 planner baseline. |
| 3. Package 03 prediction | **PASS** | All 175 measured steps of 25 cases match complete ordered request dictionaries and dispatch dictionaries. Zero mismatches or missing steps. |
| 4. Complete grid/native classes | **PASS** | Generation reports complete. `cases` emits 989 steps. `check --predicted --require-complete-grid` exits 0: all 1,041 windows predicted, zero missing windows, zero outside predicted steps, zero mismatches, zero missing cases/steps. Native classes inside: P2-32 43/43, P2-64 54/54, P0 48/48. All seven observed multi-tail coordinates are inside. |
| 5. Package 03 raw coverage | **PASS** | Streaming check exits 0: 21 covered windows, 1,020 missing windows and 72 steps outside the amended grid. Prediction still matches all 175 steps. |
| 6. Tests | **PASS; one existing optional check NOT_RUN** | Final group: 617 passed, 1 skipped, exit 0, 33.66 seconds. Coverage file contributes 126 passing tests, including 56 added parametrized checks. |
| 7. GPU time | **PASS as CPU estimate** | Admitted complete-grid estimate 0.980161 GPU-hours; separate-case envelope 1.078888. Both below 1.5, 3 and 7 hours. GPU timing was NOT_RUN. |
| 8. Hygiene | **PASS** | All nine combined B11/B11b Python modules at or below 2,000 lines; unused-import checker returns `[]` for each; cold imports exit 0 with no torch/vllm/cuda modules; `git diff --check` exits 0. Final module hashes match hygiene evidence. |

Executed compatibility command:

```bash
timeout 60s "$B11B_PYTHON" "$B11B_SCRATCH/check_compatibility.py"
```

This reuses B11's exact compatibility script and baseline source, copied into B11b scratch. It verifies the baseline bytes against read-only `git show` at `5083d0369a5f9b86c333b30fce78154a3364225f`, rebuilds package 03 from its arguments and pilot file, and compares the 64 GLM configurations. The matrix covers sequence limits `{1,2,4,8}`, contexts `{4160,8192}`, budgets `{(256,1024),(2048,)}`, capacities `{20000,200000}`, and measured decode counts `{3,8}`. Evidence: `plan_compatibility.json` and `plan_compatibility.log`.

Executed generation and final coverage commands:

```bash
timeout 120s "$B11B_PYTHON" "$B11B_G1H/build_cases.py"
timeout 120s "$B11B_PYTHON" tests/analysis/v4_attention_coverage.py cases \
  --package-plan "$B11B_G1H/package_plan.json" \
  --output "$B11B_SCRATCH/g1h_predictions.jsonl"
timeout 120s "$B11B_PYTHON" tests/analysis/v4_attention_coverage.py check \
  --package-plan "$B11B_G1H/package_plan.json" \
  --raw "$B11B_SCRATCH/g1h_predictions.jsonl" --predicted --require-complete-grid \
  --c1-json "$B11B_H3O/evidence/coverage/p2_32.json" \
            "$B11B_H3O/evidence/coverage/p2_64.json" \
            "$B11B_H3O/evidence/coverage/p0.json" \
  --output "$B11B_SCRATCH/g1h_coverage.json"
```

All exited 0. Evidence: `build_cases_final.log`, `cases_final.log`, `coverage_final.log`, `g1h_predictions.jsonl` and `g1h_coverage.json`. The final generated pilot and plan were checked again after updating count metadata.

| Category | Requested/predicted windows | Cases | Distinct shared cases | Additional window aliases | Measured steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL | 351 | 336 | 15 | 15 | 336 |
| PIECEWISE without tails | 351 | 336 | 15 | 15 | 336 |
| First tail | 318 | 296 | 17 | 22 | 296 |
| Multi-tail | 7 | 7 | 0 | 0 | 7 |
| Dummies | 13 | 13 | 0 | 0 | 13 |
| Prefill | 1 | 1 | 0 | 0 | 1 |
| Total | **1,041** | **989** | **47** | **52** | **989** |

Decision 1 removes **33 duplicate first-tail windows** from B11's 1,074. Case sharing is separate from window deduplication. FULL W1 dummy still serves both planner roles once, yielding 13 physical dummy windows for 14 role uses.

Executed raw-record check:

```bash
timeout 120s "$B11B_PYTHON" tests/analysis/v4_attention_coverage.py check \
  --package-plan "$B11B_H3O/p1h_bridge/package_plan.json" \
  --raw /data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_03/download/P1H_BRIDGE/bridge/FULL_AND_PIECEWISE/raw \
  --c1-json "$B11B_H3O/evidence/coverage/p2_32.json" \
            "$B11B_H3O/evidence/coverage/p2_64.json" \
            "$B11B_H3O/evidence/coverage/p0.json" \
  --output "$B11B_SCRATCH/p1h_coverage.json"
```

The 21 covered windows are:

| Category/mode | Covered coordinates |
| --- | --- |
| FULL serving | W1/own1, low and spread, mixes (0,0) and (1,0): four windows; W2/own2/low/(0,0): one. |
| PIECEWISE serving | W4/own2/low/(0,0): one. |
| First tail | W1/own1/(1,1) at low, high and spread: three; W4/own4/low/(1,1): one. |
| FULL dummy | W1, W2, W4, W8 and W16: five. |
| PIECEWISE dummy | W1, W2, W4 and W8: four. |
| NONE dummy | W1: one. |
| NONE prefill | W6144, uncached, mix (1,1): one. |

Outside steps by case total 72: `e0-decode-b2-p6144` 4; `e0-decode-b4-p6144`, `b8-p6144`, `b16-p6144`, `b24-p6144` and `b8-p6144-full16` 8 each; `e0-decode-b2-p6144-piecewise4` 4; `e0-decode-b8-p6144-piecewise9` 8; `e0-decode-b4-p7150` 8; `e1-first_tail-b3-p6144` 2; `e1-first_tail-b7-p6144` 3; `e1-first_tail-b10-p6144` 3. Complete coordinates and reasons are retained in `p1h_coverage.json`. Outside-grid coverage is distinct from prediction mismatch.

Executed final test group:

```bash
timeout 180s "$B11B_PYTHON" -m pytest \
  tests/unit/test_dsv4_pilot_cases.py \
  tests/unit/test_dsv4_graph_replay_profiling.py \
  tests/unit/test_dsv4_attention_profiling.py \
  tests/unit/test_dsa_attention_profiling.py \
  tests/unit/test_dsa_extend_profiling.py \
  tests/unit/test_dsa_query_one_profiling.py \
  tests/unit/test_dsv4_features.py \
  tests/unit/test_dsv4_replay_rows.py \
  tests/unit/test_dsv4_kernel_identity_check.py \
  tests/unit/test_dsa_full_dummy_profiling.py \
  tests/unit/test_v4_attention_coverage.py \
  -q -rs -p no:cacheprovider --basetemp="$B11B_SCRATCH/pytest-final-1"
```

Observed: **617 passed, 1 skipped**, exit 0. The skipped existing check is `test_dsa_full_dummy_profiling.py:166`, which requires `FRONTIER_VLLM_BS_ROOT` for older pinned native Git objects. It was unset, as in B11. This optional check was NOT_RUN; it is not claimed successful.

An earlier coverage-only run reported 121 passed and five failures because the newly added residual test fixtures supplied engine index 1 without engine index 0. The loader correctly rejected the non-contiguous indices. The fixtures were corrected to include both engines; all five residual tests pass in the final group. Evidence: `pytest-new-1.log` and `pytest-final-1.log`.

Executed hygiene:

```bash
timeout 60s "$B11B_PYTHON" "$B11B_SCRATCH/check_hygiene.py"
```

The helper runs `/data/ycfeng/tmp/unused_imports.py` over the nine modules, read-only Git checks, and fresh-process imports of the changed/new B11 modules, test module, both G1h builders and existing bridge/composition/identity drivers. All checks exited 0. Evidence: `hygiene.json`, `hygiene.log`, `unused_imports.log`, `unused_imports.command.json` and `cold_import.json`.

| Module | Lines |
| --- | ---: |
| `dsa_plan.py` | 1,151 |
| `dsa_vllm_runner.py` | 937 |
| `dsa_case_prediction.py` | 184 |
| `v4_attention_cases.py` | 217 |
| `v4_attention_windows.py` | 171 |
| `v4_attention_coverage.py` | 181 |
| `test_v4_attention_coverage.py` | 459 |
| `G1H/build_cases.py` | 147 |
| `G1H/prepare_package.py` | 31 |

GPU estimate:

| Engine | Budget | Cases/measured steps | Estimated engine steps | GPU seconds | GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 8,192 | 686 | 6,098 | 1,873.393642 | 0.520387 |
| 1 | 8,192 | 296 | 3,016 | 1,163.583642 | 0.323218 |
| 2 | 24,576 | 7 | 20 | 491.603642 | 0.136557 |
| Total | | **989** | **9,134** | **3,528.580926** | **0.980161** |

The model charges 486.103642 seconds of cold process/start/recapture time to every engine: P1h engine0 total 576.103642 seconds minus approximately 90 seconds of cases. Removing 163 × 0.2-second holds from that case interval and dividing by 280 estimated steps gives 0.205 seconds per unheld step and 0.405 seconds per held measured step. The separate-case full-grid envelope is **1.078888 GPU-hours**, comprising engine0 0.589704, engine1 0.352400 and engine2 0.136784. Both estimates are below the 1.5-hour G1h share, three-hour reduction threshold and seven-hour G1h–G4h cap. No reduction is required; the full grid remains in the files.

**4. Remaining work and decisions for Claude**

No CPU acceptance blocker remains. No additional writable prerequisite was needed.

Claude retains review, acceptance, commits, replacement of source_revision after committing B11 and B11b, package building and submission. The stale B11 revision was intentionally preserved. The generated timing JSON's per-engine key change to `engines` should be included in review.

Observations: all generated cases pass the native CPU prediction and coverage assignment; the specified retained package 03 records match; the complete grid estimate is below budget. Limitations: no GPU measurement or kernel-identity qualification was performed for the larger-budget multi-tail engine; allocation-sized zeroing, construction workspace and autotuning may differ. Per-case profiler overhead is estimated, and one measured step provides no repeated-window variance. The optional older-native Git-object test remains unexecuted.

Scratch root: `/data/ycfeng/tmp/codex-implement/ebbfa1e0-dd55-490b-b1d1-4d453d7fc0a4/b11b`. Important handoff files are `b11b.diff` (isolates this assignment from B11), `before/`, `residual_before.json`, `handoff_details.json`, final generation/coverage logs and JSON, compatibility evidence, final pytest log, and hygiene evidence. No stale B11 scratch report is used for the final counts.

**5. Review handoff**

Inspect the shared grid amendment and insertion-order deduplication, the sole allowance setting, construction-first inversion in `_place`, and the phase-preserving adjustment used by the two low residual windows. Confirm that admission still requires the full predicted measured step to be assigned to the requested window. Review all five explicit request vectors in `handoff_details.json` and their regression tests.

Inspect the package wrapper's complete-grid gate, removal of stale settings, preservation of source_revision, generated sharing/timing fields and rewritten README. Use `b11b.diff` to distinguish these changes from pre-existing B11 and concurrent B12 work.

After substantive edits, rerun compatibility, the listed test group with a fresh scratch basetemp, generation plus complete-grid coverage, package 03 raw coverage and hygiene. Final acceptance and any later GPU qualification remain Claude-owned; no acceptance or commit by Claude is claimed.
