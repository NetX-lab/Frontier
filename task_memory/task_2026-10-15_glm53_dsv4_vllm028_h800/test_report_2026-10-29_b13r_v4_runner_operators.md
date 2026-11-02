# Test report: B13r, the V4 model-runner operators (design entry 15 part 2, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: lead review, corrections and rerun before the B13r commit. |

## Scope

- B13r (Codex `88f4275b`, [h3o_operator_costs/codex_assignment_B13r_v4_runner_operators.md](h3o_operator_costs/codex_assignment_B13r_v4_runner_operators.md)) covers the four entry-15 operators that run on V2 model-runner objects outside the model forward: `kv_cache_block_zero`, `dsv4_block_table_write`, `dsv4_input_preparation` and `dsv4_sampling`, plus the diagnostic `dsv4_compute_logits`.
  - Path A: owned step scopes in the graph-replay runner, with owner `(-1, scope)`, a forward marker `frontier_v4_model_forward|<step>`, and a census of every recorded-step kernel, memcpy and memset.
  - Path B: direct calls to the native zeroer and block tables on a 43-layer eager engine with 8 routed experts, giving 394 rows over two methods.
  - The G2h runner package plan is in `h3o_operator_costs/g2h_runner/`.
- Codex report: `h3o_operator_costs/codex-implement-88f4275b-dfa7-4b03-aaff-075c4f4d852a/implementation.md`. Codex scratch: `/data/ycfeng/tmp/codex-implement/88f4275b-dfa7-4b03-aaff-075c4f4d852a/b13r/`. Lead scratch: `/data/ycfeng/tmp/b13r_accept/`.

## Lead corrections

| Change | Reason |
| --- | --- |
| Restored the graph-replay `_scope_call` to annotate every scope, containers included. The assertion `len(native_tree.annotations) == len(expected)` in `test_dsv4_graph_replay_profiling.py` is back. | **The error was in the lead's assignment.** Its Goal section said containers use `annotate=False` "because kernel-only annotations must not nest" and cited `dsa_vllm_runner.py:271-303`. That rule belongs to the direct DSA runner. Graph replay admits only KERNEL_ONLY (`dsa_vllm_runner.py:660-663`), and at HEAD its `_scope_call` passes the default `annotate=True`. `attribute_replay_trace` gives each kernel to the innermost nested range, and A2 rejects a kernel owned directly by a container. Following the assignment, Codex dropped the container annotations and added CUDA-event machinery to make up for them. |
| Removed the CUDA_EVENT branch inside the step-scope calls. | Unreachable: graph replay admits only KERNEL_ONLY. |
| Replaced the split sampling segments (sampling / logits / sampling, built as an unannotated container) with nesting: `dsv4_compute_logits` is a child scope of `dsv4_sampling` in `SCOPE_TREE`. Added `test_trace_reader_gives_nested_logits_kernels_to_the_logits_scope`, which runs the real `attribute_replay_trace`, and tests for a logits call outside sampling and for ownership restoration after a failing logits call. | Innermost-range attribution already separates the LM head from the row gather and sampler. The split was needed only because of the wrong container rule. |
| Replaced the boolean `diagnostic_only` with `row_role` (`cost`, `validation`, `diagnostic`), set by `REPLAY_ROW_ROLES` for path A and `COST` for path B. | The boolean marked path A write rows as cost sources. The assignment makes them validation rows, checked against path B. |
| Moved the imports of `MODEL_FORWARD_PREFIX`, `write_replay_report`, `forward_correlations` and `_Range`/`_RangeIndex` to module level. | No import cycle requires a function-local import. Checked: `dsv4_runner_rows` imports `dsa_rows`, `dsv4_step_scopes` and `graph_replay_trace`, and none of these imports `dsa_vllm_runner`. |
| Reverted two edits outside the assignment: a chdir fixture in `tests/unit/test_mla_native_profiling_wrapper.py` and the removal of the unused `GraphNode` import from `tests/unit/test_dsv4_graph_replay_profiling.py`. | Unrelated to B13r. Both are recorded as deferred findings below. |
| Updated `g2h_runner/README.md`: logits are nested under sampling (no CUDA-event split), and rows carry `row_role`. | The README described the replaced design. |

## Checks after the corrections

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `ulimit -v 31457280`. The torch CPU runs add `/usr/local/lib/python3.10/dist-packages` and set `CUDA_VISIBLE_DEVICES=""`. The broad run also appends `/usr/lib/python3/dist-packages` and sets `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. Commands are in `/data/ycfeng/tmp/b13r_accept/r2/rerun.sh`; logs are in the same directory.

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | Acceptance 1: the G1h-02 records (`/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw/`) reduced by `/data/ycfeng/tmp/b13r_accept/baseline.py` with the current code, against the same script on an independent export of HEAD `4ed287fc` (`before/`) | rows and gaps byte-identical | gaps SHA-256 `fb9d6a29…5ee5` and rows SHA-256 `ba1ad088…a575` (4,945 rows) in both. The four gap values are unchanged: `cuda_graph` 0.49773414267747085, `cuda_graph_piecewise` 0.6249537992538311, `eager` 0.5446049595929223, `eager_preparation` 1.2208671118771937 µs | PASS |
| 2 | Focused pytest: `test_dsv4_runner_operators.py`, `test_dsv4_graph_replay_profiling.py`, `test_dsv4_replay_gaps.py`, `test_graph_replay_trace.py`, `test_dsv4_attention_profiling.py`, `test_profiling_runtime_boundary.py` (log `r2/focused.log`) | all pass | 345 passed (47 in the new runner-operator file) | PASS |
| 3 | Broad pytest: the 93 files of `selected_tests.json`, with the 3 known cases deselected, run from the repository root (log `r2/broad.log`) | all pass; skips only for missing vLLM or absent `/tmp` probe artifacts | 2555 passed, 5 skipped, 3 deselected | PASS |
| 4 | Hygiene: Codex `hygiene.py` (unused imports, 12 cold imports with torch, vLLM, CUDA, fasteners and sklearn blocked, line counts) | no new unused import; 12 cold imports pass; every module at or below 2,000 lines | only the existing `GraphNode` unused import (present at HEAD); 12 PASS; replay runner 622, DSA runner 942, runner rows 146, block operators 309 lines | PASS |
| 5 | `g2h_runner/prepare_package.py --check-plan` from the repository root | phases G2H_RUNNER_REPLAY and G2H_BLOCK_OPERATORS; cases 686/296/7; 394 direct rows; expected layout 139 segments / 10 chunks; worker import probe PASS; no package built | all as expected; `package_built=false`; estimate 1.00-1.19 GPU-hours | PASS |
| 6 | `git diff --check` | clean | clean | PASS |

Deselected, unchanged from earlier reports: `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` (`OperatorSpec` has no `kernel_count`; it fails at HEAD too) and two `test_profiling_governance_minimal_red.py` MoE cases (local Triton needs the missing `setuptools`).

A first broad run started from a scratch directory failed in 426 tests and errored in 467, all with `data/config/models/... not found`: the tests read model files by relative path. That run was invalid. The rerun from the repository root is check 3.

Codex's Acceptance 2-10 (scope and census stand-ins, the direct-grid dry run with 394 rows, registry preservation for 11 families and 50 plans, the package plan and hygiene) ran before the corrections. The corrections change scope nesting and row roles only. Checks 2 to 5 cover those paths again.

## Deferred findings

- `tests/unit/test_mla_native_profiling_wrapper.py` writes `unused/profiler_traces` relative to the current directory. The empty directories it created in the repository root and in `/data/ycfeng/tmp` were removed with `rmdir`.
- `tests/unit/test_dsv4_graph_replay_profiling.py` keeps an unused `GraphNode` import (present at HEAD).

## Not verified on CPU (left to the G2h runner job on 1 H200)

- The native zeroer segment layout: 139 segments and 10 chunks, KV-cache groups `[62, 22, 21, 42, 20]`, block sizes `[256, 64, 64, 4, 8]` and entries `[24, 96, 96, 1536, 768]`. The worker checks these before timing.
- That each direct call launches one kernel (`_zero_kv_blocks_kernel` grid `[n, 139, 10]`; `_apply_write_kernel` grid `[writes, 1, 1]`), with separate H2D intervals, and the hold slack.
- The four gap values and attention rows on the same 989 case IDs against G1h-02, owned kernels against `g2h_runner/native_kernel_notes.md`, and all unowned summaries.

## GPU allowance

One H200 on step_main, non-preemptible, timeout 7,200 s. Builder estimate 1.00-1.19 GPU-hours; G1h-G4h used about 1.03 of 7 GPU-hours before this job.
