# D23 startup census verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record parent RED/GREEN results and native limitations. |

## Environment and commands

Worktree `/data/ycfeng/Frontier/.worktrees/step5-preview`; Python `/data/ycfeng/envs/asyncopd-py312/bin/python` (3.12.13). No conda activation, dependency change or GPU execution. Common environment:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp
export STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm
```

RED02:

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_forward_census.py -k 'startup_worker_execute or locked_startup_method' -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_census_red02.xml > /data/ycfeng/tmp/d23_census_red02.log 2>&1
```

Expected: the old observer falsely emits startup forwards. Actual: **2 FAIL / 1 PASS**, 0.52s; both failures are `AssertionError: startup worker calls contaminated serving census`, with exactly one unwanted start/end pair. This uses the real observer wrappers and extracted native `StepTimingCollector.record_batch`; GPU operations are CPU test boundaries. RED01 also exposed an overly broad assertion including `hook_installed`; RED02 corrects the test signal before implementation.

Affected GREEN:

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd_r2_stage.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_census_green01.xml > /data/ycfeng/tmp/d23_census_green01.log 2>&1
```

Expected: no startup census entries; subsequent useful/idle forwards remain paired, native errors propagate, observation faults remain soft, and stage pins remain valid. Actual: **181 PASS**, 2.67s, no failures/errors/skips.

Final fixture refinement explicitly uses warmup-prefixed native input IDs both inside and outside compilation, proving names are not the classifier:

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_forward_census.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_census_green02.xml > /data/ycfeng/tmp/d23_census_green02.log 2>&1
git diff --check
```

Actual: **19 PASS**, 0.56s; diff check PASS. This is an overlapping focused rerun, not 19 additional unique checks.

## Cause and correction

Locked `v1/worker/gpu_worker.py:940–989` calls `kernel_warmup(self)` and `warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)` inside compilation. The old observer labeled every worker `execute_model` as scheduled serving work. Its direct-runner startup test never traversed this nested worker callback. A fresh observer reproduces the defect, excluding stale thread-local state as the cause.

The native compilation wrapper now establishes a scoped startup marker, always clears it on return/exception, and the existing business/runtime observers exclude only that lifecycle. No request-name filtering, guessed count subtraction, new GPU reads/fences, native return changes or exception suppression were added.

## Limits

This validates the future observer CPU path. Capture01's original runtime FAIL remains immutable; its source-defined 192-startup/201728-serving diagnostic projection is not promoted to a full native verdict. Native G3 still lacks independent state equivalence, successful cancel/reuse, pressure and simultaneous-transfer overlap. No package was submitted or native result reclassified.
