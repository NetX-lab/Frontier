# G3/reference and MRv2 producer integration

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded primary merged-tree verification and current native evidence boundary. |

## Scope and result

- Integrated G3/reference implementation1171ab6 and records througha7d6708, plus corrected MRv2 profiling producer82f4452.
- Consolidated source: `145a2f577642c43702cd9e0b1e24d3af6ba66822`, branch `feat/step5-preview`.
- Both merges completed without conflicts. Unrelated `.pi/` and `.cortexkit/` remain untracked and untouched.
- **PASS:**685 native/reference checks in42.72s and465 producer checks in33.30s; no failures, collection errors or skipped checks in either XML.
- **PASS:** consolidated clean145a2f5 runs77/77 legacy scenarios and strictly matches every case against the retained clean5af7638 baseline;zero mismatches,failures,missing evidence or predictor cache-name differences. Logger fix b862e57's earlier36 checks and separate77-case comparison remain retained evidence.

## Exact primary commands

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-preview`.
Both test interpreters report Python3.12.13; simulator subprocesses use the existing `frontier-py310` environment.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp \
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm \
FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python \
/data/ycfeng/envs/asyncopd-py312/bin/python -m pytest \
tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_forward_census.py \
tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd_r2_stage.py \
tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_pd_r2_verdict.py \
tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py \
tests/unit/test_calibration_pd_harness.py tests/unit/test_canonical_client_metrics.py \
tests/unit/test_step5_client_metrics.py \
/data/ycfeng/Frontier/.worktrees/step5-r2-correction/tests/unit/test_step5_native_pd_r2_trace_replay.py \
--import-mode=importlib -q --tb=short -p no:cacheprovider \
--basetemp=/data/ycfeng/tmp/d21_g3_merged01 --junitxml=/data/ycfeng/tmp/d21_g3_merged01.xml \
> /data/ycfeng/tmp/d21_g3_merged01.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu \
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm \
PYTHONPATH=$PWD:/data/ycfeng/tmp/step5_profiling_20261020/extra_site \
TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
/data/ycfeng/envs/oneshot-opd-py312/bin/python -B -c \
"import sys,types,pathlib; p=types.ModuleType('tests'); p.__path__=[str(pathlib.Path.cwd()/'tests')]; sys.modules['tests']=p; import pytest; raise SystemExit(pytest.main(['tests/unit/test_step5_mrv2_dummy_profiling.py','tests/unit/test_step5_request_free_profiling.py','tests/unit/test_step5_attention_profiling.py','tests/unit/test_step5_device_activity.py','tests/unit/test_step5_fixed_probe.py','tests/unit/test_step5_attention_replay.py','tests/unit/test_step5_attention_judgment.py','tests/unit/test_step5_attention_timing.py','tests/unit/test_step5_boundary_probe.py','-q','--tb=short','-p','no:cacheprovider','--junitxml=/data/ycfeng/tmp/d21_mrv2_merged01.xml']))" \
> /data/ycfeng/tmp/d21_mrv2_merged01.log 2>&1
```

The explicit local `tests` namespace avoids the profiling environment's unrelated installed package. No native GPU execution is substituted in these CPU checks.

### Consolidated clean-tree fidelity

The primary created detached worktree `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-145a2f5` at the exact consolidated commit, preserving the primary's ongoing documentation edits. Existing clean baseline artifacts were reused without modification; all77 candidate cases used clean caches. The actual calls were:

```bash
# cwd=/data/ycfeng/Frontier/.worktrees/fidelity-candidate-145a2f5
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run \
  --repo-root "$PWD" --label candidate \
  --output-root /data/ycfeng/tmp/step5_native_integration_fidelity_20261022 \
  --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare \
  --output-root /data/ycfeng/tmp/step5_native_integration_fidelity_20261022 \
  --baseline-label /data/ycfeng/tmp/step5_event_logging_clean_fidelity_20261021/baseline \
  --candidate-label /data/ycfeng/tmp/step5_native_integration_fidelity_20261022/candidate
```

Both exit0. `candidate.log`, `compare.log`, `comparison.json` and all case artifacts are under that output root. Absolute labels preserve each original source/output identity during comparison. This closes the integration's required legacy-fidelity check; it does not measure native accuracy or claim a new wall-time/memory budget result.

## Acceptance evidence

Native/reference tests exercise the real local HTTP direct-D route, failure isolation, real CPU Torch complete state/alias hashing, logical cache contents, strict retirement controls, native host-method execution, post-DP width and complete forward accounting faults. Producer tests execute source-bound MRv2 dummy input/metadata paths and preserve D22 independent-cohort semantics. The new observer reads host arrays/metadata shapes, not GPU tensors, for forward accounting.

Fresh native capture `exp-1021-235955-532667` uses separately sealed source025d9a9 and49 files/50 installed native pins. Initial creator/resource/RDMA/image/local-mount verification passes; four subsequent replica snapshots report Running. Details and raw artifact references: `../task_2026-10-21_step5_priority_runs/canonical_g3/native_run.md`.

## Remaining limits

No new native G3/G4 outcome, qualified H800 profile, completed full-CLI receipt or four-metric accuracy result is claimed. Actual native reference/lifetime/pressure/abort and complete forward evidence still need inspection. The next consumer work must preserve the distinction between isolated no-metadata work and actual metadata-bearing MRv2 dummy execution, and reflect native FULL_AND_PIECEWISE. Old no-metadata profile packages remain withdrawn. The final10% gates and same-hardware32→64 release condition remain unchanged.
