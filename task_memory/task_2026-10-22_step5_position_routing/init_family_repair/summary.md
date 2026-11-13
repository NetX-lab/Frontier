# D29-INIT-FAMILY-RESOLUTION completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive the verified four-line loader correction and actual-loader regression. |

## Overview

The typed MoE loader now resolves the existing model-derived attention family once
per call and passes its ID using the existing validator keyword. Every row retains
its real JSON/schema/ownership validation before filtering. No cache, API change,
fallback, dataset change, peer merge or submodule change was made.

Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-loader-family`.
Branch: `feat/step5-d29-loader-family`.
Baseline: `24f970cbd6cae059467f68e0e1cf771b07aeecfe`.
Code-only commit: `ee7e66ee439054978897c91fa32598fc13804883`.

## Deliverables

- `frontier/execution_time_predictor/profiling_dataframe_loaders.py`: four added
  source lines, once-per-call family derivation and existing keyword.
- `tests/unit/test_moe_grouped_gemm_padded_features.py`:34 added lines, actual-loader
  RED/GREEN regression reusing `_position_source_rows`. Two loads derive once each,
  entire selected frames stay equal, and corrupting an unselected row/operator
  still raises before filtering.
- `task_memory/task_2026-10-22_step5_position_routing/init_family_repair/code.diff`:
  exact code-only baseline-to-commit diff.
- `task_memory/task_2026-10-22_step5_position_routing/init_family_repair/test_report_2026-10-23_init_family_repair.md`:
  exact commands, expectations, observations, dependencies and limits.
- `task_memory/task_2026-10-22_step5_position_routing/init_family_repair/accounting.json`:
  AST, hashes,38/45 repair gross, retained13/17 and231/237 path gross, root aggregate
  conservative addition and unchanged gitlink.
- `task_memory/task_2026-10-22_step5_position_routing/init_family_repair/evidence/`:
  selected raw RED/GREEN, affected-suite, dependency and baseline logs.
- This directory's `requirements.md`, `plan.md`, `progress.md`, `checkpoint.md`:
  scope, route, self-check and parent next actions. Records are committed separately.

## Observed validation

RED: actual three-row loader performs9 Step config derivations, failing the one-call
criterion after complete selected frame equality passed (1FAIL,1.85s).
GREEN:1PASS,2.31s; exactly one derivation on each call, equal complete selected frames,
and unselected enum corruption rejected. Full MoE/routing files29PASS,33.18s.
Full typed-governance52PASS,14.44s via existing same-ABI Torch supplement; one
Torch/NumPy ABI warning remains. Original primary-environment five missing-Torch
failures exactly match frozen e41. No dependency installation or stub used.
AST/whitespace/caps PASS; root gitlink3dac53c unchanged. Existing negatives unchanged.

## Remaining limitations and next actions

Primary owns independent review, integration, final-source accounting and freeze.
This slice does not admit full CLI, initialization, native/GPU, G3 or E2E results.
Parent3000-row timing remains prior measured evidence, not a newly measured
whole-loader speedup. The initial unexplained startup interval remains separate.
Torch supplement is CPU-test-only and does not qualify NumPy/Torch conversion or
GPU kernels. No unfinished in-scope source change remains.
