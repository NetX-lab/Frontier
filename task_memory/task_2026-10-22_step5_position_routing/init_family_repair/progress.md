# D29-INIT-FAMILY-RESOLUTION progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Close governance52PASS via existing Torch supplement; code-only commit ee7e66ee. |
| 2026-10-23 | GREEN, MoE/routing29PASS, measured caps; record supplemental suite dependency limits. |
| 2026-10-23 | Actual loader regression RED: nine real config derivations; apply the four-line source correction. |
| 2026-10-23 | Record isolated baseline and approved actual-loader regression seam. |

- completed: Read root instructions, the required parent requirements/plan/scope,
  parent measurement and investigation report, glossary/ADR and TDD/diagnosis skills.
  Diagnosis already has an independently measured single-variable parent control;
  this slice locks it down at the explicitly approved real loader seam rather than
  repeating full scans or reopening the mechanism decision.
- completed: Created new branch `feat/step5-d29-loader-family` with
  `git worktree add -b feat/step5-d29-loader-family /data/ycfeng/Frontier/.worktrees/step5-d29-loader-family 24f970cb`.
  Target did not exist; initial worktree clean. `git diff --exit-code 24f970cb e41abc07
  -- frontier tests` passed. Root gitlink is `3dac53c544e546574cb6e8078cd14053d0916113`.
- completed: Added one 34-gross-line actual-loader regression using the existing
  `_position_source_rows` fixture plus a third row outside selected `moe_layer_ids`.
  The regression wraps and executes actual `validate_step5_config`, compares the
  complete selected dataframe (including typed JSON, duration and kernel_count),
  checks successive calls separately and corrupts an unselected operator's enum in
  the unselected third row. No validator or fixture writer is replaced.
- completed: RED on unmodified source: `1 failed in 1.85s`, actual `9 != 1` config
  derivations. Selected dataframe equality passed before the expected assertion.
  Raw evidence: `/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/red.log`.
- completed: Minimum source correction adds four lines: derive family once and
  pass existing `architecture_family_id` keyword. The same all-row typed map and
  every downstream filter/validation stay intact.
- completed: Actual loader GREEN (`1 passed in 2.31s`): two successive calls each
  derive once, complete selected frames match, and invalid typed enum on unselected
  row/operator raises before filtering with only one derivation.
- completed: Full grouped-GEMM and routing runtime files (`29 passed in 33.18s`).
  Combined MoE/routing/typed-governance run:76PASS/5FAIL in33.07s; all five failures
  are missing `torch` in existing Python3.10 producer imports, not loader failures.
- completed: AST and whitespace PASS. Delta relative24f970cb: source4/test34/total38;
  retained relative1aee0af: source13<=17/test231<=237. `accounting.json` preserves
  measured values and conservative root aggregate addition (4016<=CORE4404 with
  model-group105 reserved;9309<=global10000 with model-group105/parser50 reserved).
  Actual aggregate reconciliation including peer work remains root-owned.
- completed: Frozen e41 typed-governance baseline confirms the exact same five
  missing-Torch failures (47PASS/5FAIL in2.29s). Existing Python3.12 environment lacks
  fasteners/ddsketch/sklearn/plotly, so it is not a usable predictor supplement.
  Existing same-ABI distillm-py310 has Torch2.1.2; append its site-packages after the
  Frontier environment for this one test process, preserving primary NumPy/pandas/
  sklearn. Full governance52PASS in14.44s, one NumPy/Torch ABI warning. No install,
  environment modification, stubs or GPU execution. This is CPU producer/loader
  coverage, not qualification of Torch/NumPy conversion or GPU kernels.
- completed: Progress self-check in `checkpoint.md`; no D1–D7 drift or S1–S6 defect.
  No new investigation or unrelated environment repair is needed.
- completed: Code-only commit `ee7e66ee439054978897c91fa32598fc13804883` contains
  exactly the two owned paths, 38 insertions/0 deletions. Rechecked whitespace,
  numstat and root gitlink after commit; worktree has no tracked code modifications.
- completed: Archived seven selected raw logs, exact `code.diff`, commands,
  `accounting.json`, `evidence_manifest.json` and English completion summary in this
  directory. Manifest checksums and exact five-ID dependency failure-set equality
  verified. Records form the separate record-only delivery commit after codeee7e66ee;
  exact record hash is discoverable from this branch history and the handoff result.
  No unfinished in-scope code or validation work remains.
- environment: Existing `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6.
  Bare `python` is absent (exit127); no installation or environment changes made.
- resource enforcement: `systemd-run --user --scope -p MemoryMax=30G` cannot move
  PIDs into a cgroup (I/O error, permission denied); system scope requires interactive
  authentication. These failures occurred before pytest execution. Tests instead
  use the parent's existing RLIMIT_AS30GiB pattern and `timeout 60s`; no resource
  bound was raised. Scope errors were not test RED evidence.

## Final boundary / primary next actions

1. Primary reviews code commit `ee7e66ee439054978897c91fa32598fc13804883` and
   `code.diff`; integration/review/freeze are intentionally not performed here.
2. Primary merges/cherry-picks the separate code and record deliveries according to
   its final branch plan, rechecks owner/global accounting against peer repairs,
   and reruns integrated final-source gates.
3. Complete CLI/native/GPU/G3/E2E and unexplained initial startup time remain parent
   work, not accepted or measured by this loader slice. Parent3000-row timing is
   preserved only as prior evidence, not a new whole-loader performance claim.
