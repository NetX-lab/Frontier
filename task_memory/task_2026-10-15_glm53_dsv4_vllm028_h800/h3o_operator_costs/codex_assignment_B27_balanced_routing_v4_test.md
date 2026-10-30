# Task: update the balanced-routing cache test to the V4 kernel-only contract (stage 4 assignment B27, issue I-002, CPU only, test only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Created from issue I-002. |

## Goal

`tests/unit/test_balanced_routing.py::test_selected_producer_rows_train_and_reuse_distinct_cache` fails in both of its parametrizations on main. The failure dates from B16b (`61c2f9d3`) and has nothing to do with simulator behavior: the test was not updated when B16b changed the measurement contract for DeepSeek-V4.

The lead reproduced it on main `591d7c54` (output `/data/ycfeng/tmp/i002_before.txt`):

```text
FAILED tests/unit/test_balanced_routing.py::test_selected_producer_rows_train_and_reuse_distinct_cache[order0]
tests/unit/test_balanced_routing.py:316: KeyError: 'moe_gating_routing_topk'
    assert manager.get_models_for_cluster(role)["eager"][op] is cold[role]
```

Cause, from the code:

- `dsv4_measurement_type` (`frontier/execution_time_predictor/dsv4_measurement.py`) returns `MeasurementType.KERNEL_ONLY` for every V4 model.
- `_get_measurement_types_for_cluster` (`frontier/execution_time_predictor/shared_prediction_model_manager.py:248`) then returns only `[KERNEL_ONLY]` for every V4 role. `get_models_for_cluster` (line 598) consequently gives an empty `eager` view for V4.
- The test trains the V4 gating models from `CUDA_EVENT` rows with `_active_measurement_type = CUDA_EVENT`, and reads them back from the `eager` view.

The test module runs only when `torch` imports (`pytest.importorskip("torch")`). Suites that run without torch skip it, which is why the failure went unreported.

## Work

1. Change the test so that it builds and reads V4 models under the V4 contract:
   - profiling rows with the measurement type that V4 trains from;
   - the matching active measurement type on the manager;
   - cluster-view assertions on the `kernel_only` family.
2. Add one assertion of the contract itself: for each V4 role, the `eager` view of `get_models_for_cluster` is empty.
3. Keep every other check of the test, with the same strength:
   - one distinct cache entry per routing path;
   - the predicted values 2.0 and 7.0;
   - a warm lookup that does not fit;
   - the two wrong-row cases that raise `ValueError` matching `routing.*path`.
4. If the test cannot pass without a change under `frontier/`, do not make that change. Stop and report the reason, with the evidence.

## Acceptance

1. With CPU torch on the path (command below), `tests/unit/test_balanced_routing.py` gives 128 passed and 0 failed. Before the change it gives 126 passed and 2 failed.
2. `git diff --stat` shows only `tests/unit/test_balanced_routing.py` and your report.
3. No assertion of the test is removed or weakened, and no skip or `xfail` is added. List each changed line with its reason.

Command, run in the working root:

```bash
ulimit -v 31457280
env PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0 \
  PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages \
  timeout 900 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_balanced_routing.py -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/codex-implement/<request-id>/b27/pytest
```

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/b27-balanced-routing-v4-test`. It is a Git worktree on branch `b27-balanced-routing-v4-test`, created from the main commit that adds this file. Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records (read-only):** in the main tree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, task directory `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/`: `issues.md` (I-002).
- **Writable paths, in the working root:**
  - `tests/unit/test_balanced_routing.py`;
  - your report under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b27/`.
  - If you need another path, ask in the report; do not change it yourself.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b27/`.
- **Restrictions:**
  - The lead commits. Do not run Git commands that change state. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
  - No GPU, network, job submission or package installs.
  - No deletions or moves outside your scratch.
  - Four A10 runs are simulating under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/`. Do not stop or signal them. Do not run recursive `grep` or `find` over `/data/ycfeng/tmp/glm53_dsv4/`.
  - Python 3.10. Memory `ulimit -v 31457280`, and a `timeout` on every command.

## Expected return

- Report language: English. Write the report to `h3o_operator_costs/b27/report.md` and return its content.
- The diff of the test, with the reason for each changed line.
- The before and after pytest summary lines, with the commands.
- Deviations and their reasons, and limits.
