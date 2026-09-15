## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded standalone NCCL/RCCL collective profiler CPU and schema verification. |

# Increment 11 Verification Report — Standalone RCCL Profiler

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: `3.12.3`; CPU environment; `torch 2.5.1+cu124`; Ray is unavailable and no AMD/MI355X device is present.
- Commands:

  ```bash
  python -m pytest \
    tests/unit/test_collectives_increment11.py \
    tests/unit/test_collectives_rocm_runner.py \
    tests/unit/test_collective_timing.py \
    tests/unit/test_profiling_accelerator.py -q -p no:cacheprovider
  python -m frontier.profiling.collectives.main --help
  python -m compileall -q frontier/profiling/collectives frontier/profiling/utils \
    tests/unit/test_collectives_increment11.py tests/unit/test_collectives_rocm_runner.py
  git diff --check
  ```

## Criteria and evidence

| Criterion | Evidence | Result |
| --- | --- | --- |
| Standalone entrypoint | `frontier.profiling.collectives.main` lazily imports Ray, supports local multiprocessing, and `--help` succeeds without Ray or GPU initialization. | PASS |
| ROCm backend contract | Worker initializes `torch.distributed` with backend `nccl`; on ROCm this maps to RCCL. No simulator communication code or automatic `network_device=mi355x_ubb` activation was changed. | PASS (source review) |
| Dtype byte accounting | `CollectivesInput` accepts only FP16/BF16/FP32; `GraphedCollective.element_size` and `CollectiveWrapper.profile()` derive bytes from the actual tensor dtype. CPU tests cover BF16 (2 bytes) and FP32 (4 bytes). | PASS |
| World-size/layout validation | Partial-node layouts and one-worker cases are rejected; generated grids carry the requested precision. | PASS; collective contract tests `9 passed` |
| Output schema | `_write_results()` writes flattened `time_stats.*`, rank/world-size/collective metadata, and `profiling_precision` under `<output>/collective/<timestamp>/<collective>.csv`. | PASS |
| Existing collective behavior | Existing collective timing and accelerator tests remain green. | PASS; combined suite `29 passed` |
| Compilation and whitespace | Targeted compileall and `git diff --check` exited successfully. | PASS |

## Hardware boundary

`SKIP: AMD/MI355X hardware unavailable` — no RCCL process-group initialization, all-reduce timing, Kineto NCCL/RCCL event capture, or multi-rank GPU output was executed. CPU/schema results do not establish RCCL correctness or benchmark parity. A real GPU reproduction command remains in the CLI help/source path and must be run in a ROCm environment before claiming hardware validation.
