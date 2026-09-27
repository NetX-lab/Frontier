# Issue 26 Correctness PR — Validation Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created. Environment recorded; baseline results are recorded in the refactor task's Step 0 report because both branches share the same base commit. |

## Environment

| Field | Value |
| --- | --- |
| Host | `kun-workspace-vgen2` (CPU master) |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python` (uv-managed CPython 3.10.6) |
| Install | `uv pip install -e ".[test]"` from the active worktree |
| Scratch root | `FRONTIER_TMP_ROOT=/data/ycfeng/tmp/issue26-correctness-pr` |
| vLLM reference | `.real-engine/vLLM-BS` at `ea95f57` |

## Per-checkpoint evidence

| Step | Purpose | Source under test | Command | Outcome | Baseline comparison | Limits |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Baseline for touched areas | `1f694f7` | See `task_memory/task_2026-09-21_oversized_module_split/test_report_2026-09-21_step0_baseline.md` | 84 passed / 10 failed (pre-existing or environmental, listed in the report); two dummy smokes PASS | Base itself | Shared with the refactor branch (same base). |
