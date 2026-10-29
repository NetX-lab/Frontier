# MRv2 producer correction: parent verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record parent takeover,source review,affected regression and old-stage audit. |

## Scope and result

**PASS at source/CPU level; native execution and profile admission remain unverified.**
The delegated correction stopped on a model usage limit. The parent inspected its
saved code and independently reran the affected checks instead of accepting the
interrupted status as a completed delivery.

The code change replaces the MRv1-based serving-idle claim with an explicit
post-DP-sync MRv2 descriptor. Native `InputBatch.make_dummy`, `BlockTables`,
`Step5ModelState.prepare_attn` and the layer's metadata builder create the actual
padding/slot/request-state metadata. FULL uses native capture-buffer restaging.
The separate isolated no-metadata capability is retained with a distinct physical
scope;it is not serving idle coverage. Consumer admission and D22 timing are unchanged.

## Source review

Locked source root:
`/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`.

- `v1/worker/gpu_worker.py:1467-1474` and `gpu/model_runner.py:821-911,1810-1953`:
  local one-token execution can receive a different synchronized descriptor;
  serving dummy execution builds attention metadata and is omitted from graph statistics.
- `v1/worker/gpu/input_batch.py:130-220` and `gpu/model_runner.py:1603-1617`:
  dummy split,zero positions,padding flags,zero block tables and invalid slots
  match the producer's native-helper calls. The eight-token test descriptor is
  a CPU fixture,not an observed serving width.
- `models/step5/nvidia/model_state.py:32-64`:persistent request-state indices
  and FULL builder restaging are preserved.
- `models/step5/nvidia/sparse_summary_cache.py:179-223`:negative slots update
  distinct scratch pages;their updates do not permit changing live cache pages.
  Existing fault controls reject live KV and summary mutations.

## Exact parent execution

Host:kun-workspace-vgen2. Worktree:
`/data/ycfeng/Frontier/.worktrees/step5-p0-profiles`.
Python:`/data/ycfeng/envs/oneshot-opd-py312/bin/python`,verified Python3.12.13.

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu \
  STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm \
  PYTHONPATH="$PWD:/data/ycfeng/tmp/step5_profiling_20261020/extra_site" \
  TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B -c \
  "import sys,types,pathlib; p=types.ModuleType('tests'); p.__path__=[str(pathlib.Path.cwd()/'tests')]; sys.modules['tests']=p; import pytest; raise SystemExit(pytest.main(['tests/unit/test_step5_mrv2_dummy_profiling.py','tests/unit/test_step5_request_free_profiling.py','tests/unit/test_step5_attention_profiling.py','tests/unit/test_step5_device_activity.py','tests/unit/test_step5_fixed_probe.py','tests/unit/test_step5_attention_replay.py','tests/unit/test_step5_attention_judgment.py','tests/unit/test_step5_attention_timing.py','tests/unit/test_step5_boundary_probe.py','-q','--tb=short','-p','no:cacheprovider','--junitxml=/data/ycfeng/tmp/step5_p0_profiles_20261021/mrv2_parent01.xml']))" \
  > /data/ycfeng/tmp/step5_p0_profiles_20261021/mrv2_parent01.log 2>&1
```

Expected:no failures/errors/skips in the affected suite;intentional source-bound
CPU fixtures must preserve metadata,padding and cache invariants.
Actual:**465 passed in20.07s**,zero failures/errors/skips. JUnit records19.165s
for its suite interval. These are two different reported durations,not GPU timing.
The existing `tests` namespace setup only resolves the installed-package collision
inside this test process;it does not replace native helper bodies.

## Integrity and protocol audit

Parent rehashed all17 native source files listed in
`/data/ycfeng/tmp/step5_p0_profiles_20261021/mrv2_source_stage_audit01.json`.
For each old stage,all3415 payload hashes,the exact file inventory and the
manifest hash match. Both remain not for submission. No old verdict was recomputed.

Compared with4173fba:`timing.py`,`replay.py`,`scopes.py` are byte-identical;
`summarize_replays`,`_scoped_replay`,`_gpu_activity`,`_validate_output_state`
are AST-identical. Durable receipt:`evidence/mrv2_parent01.json`.

## Remaining execution

The ordinary graph tables cannot provide the missing dummy descriptor census.
Collect it through the native observer,then connect the distinct physical identity
to the consumer and generate compatible H800 inputs. No guessed dummy grid,
fresh stage,GPU run or formal accuracy PASS was produced by this correction.
