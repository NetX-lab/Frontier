# Parent G3/reference and MRv2 forward verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded primary implementation verification before native staging. |

## Environment and commands

- Worktree: `/data/ycfeng/Frontier/.worktrees/step5-canonical-g3`.
- Interpreter: `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp`.
- `STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`.
- `FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python`.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp \
  STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm \
  FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python \
  /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest \
  tests/unit/test_step5_native_reference.py \
  tests/unit/test_step5_native_forward_census.py \
  tests/unit/test_step5_canonical_g3.py \
  tests/unit/test_step5_native_pd_r2_stage.py \
  tests/unit/test_step5_native_pd_r2_observation.py \
  tests/unit/test_step5_native_pd_r2_verdict.py \
  tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py \
  tests/unit/test_calibration_pd_harness.py \
  tests/unit/test_canonical_client_metrics.py tests/unit/test_step5_client_metrics.py \
  /data/ycfeng/Frontier/.worktrees/step5-r2-correction/tests/unit/test_step5_native_pd_r2_trace_replay.py \
  --import-mode=importlib -q --tb=short -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/d21_g3_complete02 \
  --junitxml=/data/ycfeng/tmp/d21_g3_complete02.xml \
  > /data/ycfeng/tmp/d21_g3_complete02.log 2>&1
```

## Observed results

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Saved reference implementation baseline | Existing affected suite passes |420 PASS,10.28s (`d21_g3_resume01.log`) | PASS |
| Runtime seam plus original native checks | Useful/dummy host seam, errors and stages preserved |436 PASS,13.70s (`d21_g3_runtime01.log`) | PASS |
| Initial direct-reference failure test | Failed reference must appear in errors, next PD wave completes |1 FAIL/26 PASS: ignored validation-error return | FAIL, corrected |
| First importlib full collection | Both new modules import |2 collection errors from unqualified sibling test imports | FAIL, corrected |
| Final broad parent suite | All cases pass without skips |685 PASS,0 skipped,41.97s (`d21_g3_complete02.log`, XML beside it) | PASS |
| Final metadata-source pins and stage/seam checks | Added default-model/attention source bindings remain valid |69 PASS,1.51s (`d21_g3_pins_final01.log`, XML beside it) | PASS |
| `git diff --check` | No whitespace errors |Exit0 | PASS |

Final pin check used the same environment and pytest flags, selecting `tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_canonical_g3.py`, with `--basetemp=/data/ycfeng/tmp/d21_g3_pins_final01 --junitxml=/data/ycfeng/tmp/d21_g3_pins_final01.xml`. Source commit: `1171ab6`.

Real HTTP tests prove routing and failure isolation using local test servers. Real CPU Torch tests prove complete resident-view hashing and logical-byte comparison, including noncontiguous views, padding and aliases. The native-host test executes the locked `record_batch` method body, while GPU/model/DP execution is represented by explicit CPU seam fixtures. Counter fixtures reject missing ranks/ends, duplicate forwards, wrong widths/contexts/FULL mode, missing metadata, scheduler discrepancies, TP divergence and fenced/startup rows.

## Limits

No GPU operation or native G3 admission is established by these results. Actual FULL/PIECEWISE/dummy shapes, clean hardware timing, complete reference equivalence, pressure/reset/reuse/abort outcomes and the final four10% errors still require native evidence. Independent numerical differences retain the existing UNVERIFIED disposition; there is no new tolerance or batch-drift waiver. Diagnostic callbacks remain soft gates, including when a callback itself raises.
