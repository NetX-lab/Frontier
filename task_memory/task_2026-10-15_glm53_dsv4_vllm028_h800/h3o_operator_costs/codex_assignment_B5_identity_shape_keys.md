# Task: key the DeepSeek-V4 kernel identity check by step shape, with a NO_REFERENCE status (stage 1a assignment B5, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B5 of stage 1a. |

## Goal

`tests/comparison/calibration/dsv4_kernel_identity_check.py` compares each recorded pilot step's expanded attention kernels with native H200 forward kernels. It now selects native candidates by step class and CUDA Graph mode only, and takes the closest one. When the native traces have no forward of the pilot step's shape, it compares against another shape and reports the shape-dependent kernels as missing. On the package 03 data this produces 129 (P2-32) and 121 (P2-64) false FAIL steps, while every step that has a native forward of its shape passes.

After the change:

- A native forward is a reference for a pilot step only when both have the same **shape key**: (step class, mode, padded rows, eager rows), defined below.
- A step without a reference gets status `NO_REFERENCE`. A step with references is `PASS` when the closest reference (fewest missing kernels, as now) contains the expected multiset, otherwise `FAIL`.
- The report lists the covered and the uncovered shapes.
- On the package 03 data the expected results are:
  - P2-32: 46 `PASS`, 129 `NO_REFERENCE`, 0 `FAIL`, verdict `PASS`;
  - P2-64: 54 `PASS`, 121 `NO_REFERENCE`, 0 `FAIL`, verdict `PASS`.

## Approach and decisions

- **Evidence:** the lead's diagnostic script `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/shape_matched.py` (read-only) and the test report `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/test_report_2026-10-28_p1h_package03.md`, section "Kernel identity".
  - The diagnostic keys by (class, mode, padded rows) only. It gives P2-32 46/129/0 and P2-64 54/110/11.
  - The 11 P2-64 FAIL steps are PIECEWISE at 16 padded rows. In PIECEWISE the eager-break attention kernels run on the scheduled rows: pilot 8 and 11 rows, native 9 and 13. So eager rows must be part of the key.
- **Padded rows:**
  - These are the token rows the forward dispatches. In both the pilot and native traces, the per-layer kernel `_fused_q_kv_rmsnorm_kernel` has the padded rows as its grid x. Keep this kernel name as one named module constant, with a comment saying why it measures the row count.
  - Pilot: padded rows = `cudagraph_dispatch.num_tokens` of the step record. Each step's expanded attention kernels must contain the row-count kernel with exactly one grid, whose x equals that value; otherwise raise `ValueError`. This check is the evidence that the native reading below is correct. It holds for all 175 package 03 steps (FULL 104, PIECEWISE 60, NONE 11), as the lead checked.
  - Native: padded rows = grid x of the row-count kernel in the forward's dispatch kernels, which must have exactly one grid; otherwise raise `ValueError` naming the trace and forward.
- **Eager rows:**
  - These are the rows the eager attention kernels execute. Attention runs eagerly in modes `PIECEWISE` and `NONE`; in `FULL` it is inside the graph and sized by the padded rows. Keep the set of eager-attention modes as one named constant.
  - For FULL the eager rows are `None`.
  - For PIECEWISE and NONE:
    - pilot: the sum of `num_scheduled_tokens` over the step's `requests`;
    - native: `context_tokens + generation_tokens` from the step range that the census already parses (`v4_runtime_census._forward_segments`, `record["step"]`);
    - a dummy (pilot phase `dummy`, empty `requests`; native class `dummy`, no logits) executes all of its dispatched rows, so its eager rows are its padded rows.
- **Census:**
  - `tests/analysis/v4_runtime_census.py` `ForwardKernels` gains the scheduled token count of the forward's step range (`None` without one), so the identity check does not parse traces a second time.
  - Keep the census's other outputs unchanged.
- **Statuses and verdict:**
  - `NativeCandidate` deduplicates by shape key plus the kernel multiset.
  - `match_kernels` selects candidates by the whole shape key.
  - Case status: `FAIL` if any step fails; else `NO_REFERENCE` if any step has no reference; else `PASS`.
  - Verdict: `FAIL` if any step fails; else `PASS` if at least one step passes; else `NO_REFERENCE`.
  - Exit code 0 only for `PASS`.
  - The report gains a shape coverage list: for each pilot shape key, its pilot case ids, step status counts and the native forward count at that key. The printed summary gives the status counts and the referenced and unreferenced shapes.
- **Remove the margin cases:**
  - Remove `--margin_cases`, the `margin` fields and their tests. They marked pilot cases outside native coverage; `NO_REFERENCE` now derives that from the data.
  - No other caller uses them; only this tool and its unit test do.
- **Docstring:** update the module docstring. Qualification covers only the shapes the native traces contain.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `56e06be2`).
  - Untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`: leave them alone.
  - The lead is editing task records under `task_memory/`: do not write there.
- **Writable paths:** `tests/comparison/calibration/`, `tests/analysis/`, `tests/unit/`. Within them change only:
  - `dsv4_kernel_identity_check.py`;
  - `v4_runtime_census.py` (the `ForwardKernels` field and its filling);
  - `tests/unit/test_dsv4_kernel_identity_check.py`;
  - no census unit test exists; cover the new `ForwardKernels` field in the identity unit test.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b5/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits.
  - No GPU, network, package installs, deletions or moves.
  - Python 3.10 syntax.
  - No torch, vllm or cuda imports at module load.
  - Keep modules at or below 2,000 lines.
- **Read-only data:**
  - Bridge config of the package 03 analysis copy: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge_config.json` (its `output_root` is the copy `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge`).
  - Native traces:
    - P2-32: `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces`
    - P2-64: `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces`
  - Each of the two identity runs takes several minutes, and native traces are read one at a time.
- **Environment:**
  - `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vllm);
  - `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`;
  - `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`;
  - `ulimit -v 31457280`;
  - a `timeout` on every command.

## Acceptance

1. **Unit tests:** `pytest tests/unit/test_dsv4_kernel_identity_check.py` (updated) passes. It covers:
   - `NO_REFERENCE` for a different padded row count;
   - `NO_REFERENCE` for a different eager row count in PIECEWISE;
   - FULL matching that ignores the scheduled rows;
   - the pilot raising when the row-count grid differs from the recorded dispatch;
   - a native forward without a unique row-count grid raising;
   - dummy eager rows equal to padded rows;
   - deduplication by shape;
   - the three verdicts and their exit codes;
   - the coverage list.
2. **Regression tests:** the existing suites that import the changed modules still pass:
   - the composition and bridge checks;
   - every other unit test that imports `v4_runtime_census` (none besides the identity test at HEAD);
   - `tests/unit/test_dsv4_graph_replay_profiling.py`;
   - `tests/unit/test_dsv4_attention_profiling.py`.

   Report the counts.
3. **Real data:** `dsv4_kernel_identity_check.py --config /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge_config.json --native_trace_dirs <P2-32> --output <scratch>/identity_p32.json`, and the same for P2-64, give the counts and verdicts in "Goal". Report the coverage lists.
4. **Imports:** a cold `import dsv4_kernel_identity_check` and `import v4_runtime_census` exit 0.

## Expected return

- Report language: English.
- Changed paths, each with a summary.
- Executed checks with their output tails.
- Scratch paths, deviations and their reasons, and limits.
- If the real-data counts differ from "Goal", report the differing steps with their shape keys and missing kernels. Do not tune the key to reach the numbers; return the evidence.
