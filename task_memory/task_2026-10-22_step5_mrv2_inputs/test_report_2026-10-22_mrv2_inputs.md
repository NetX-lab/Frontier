# D23-MRV2-INPUTS CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record inherited baselines, real-seam RED/GREEN and observed descriptor verification. |

## Scope and environment

Worktree `/data/ycfeng/Frontier/.worktrees/step5-mrv2-inputs`, branch `feat/step5-mrv2-inputs`, base `7fa73f9`. Task/session `D23-MRV2-INPUTS` / `ses_ea08efc2cffe6Z1Rc7srkVdebh`.

- Measured CPU interpreter `/data/ycfeng/envs/frontier-py310/bin/python`: Python3.10.6, pytest9.1.1, pandas2.3.3, sklearn1.7.2.
- Source-bound profiling CPU interpreter `/data/ycfeng/envs/oneshot-opd-py312/bin/python`: Python3.12.13, pytest9.1.1, Torch2.13.0+cu130. CUDA allocation and kernels are CPU test boundaries; no GPU was used.
- No conda activation, dependency changes, credentials, platform/cloud/SSH/push or further delegation. `TMPDIR=/data/ycfeng/tmp`, `PYTHONDONTWRITEBYTECODE=1`, disabled W&B. Native sources are existing local exported bytes.

## Pass conditions and actual results

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Inherited producer baseline |465 pass, no failure/error/skip|465 pass/24.84s|PASS|
| Initial affected measured baseline |No failure/error/skip|141 pass/55.42s|PASS|
| New real build_row/rows_to_dataframe mixed producer feedback |RED catches exact scalar-scope failure|3 failures/2.12s at incompatible physical_scope|Expected RED|
| First implementation plus measured fit/replay |Same input GREEN|43 pass/12.96s|PASS|
| Expanded identity/invalid-fields/union regressions |No failures|46 pass/13.67s|PASS|
| Final producer first attempt |Existing scope assertion must reflect authorized isolated diagnostics|465 pass,2 obsolete rejection assertions fail/17.00s|FAIL, resolved below|
| Final producer after narrow assertion correction |Original465 plus2 source-bound FULL/null cases pass|467 pass/16.70s|PASS|
| Final broad measured/fit/replay/bundle/PD suite |No failures/errors/skips|263 pass/191.88s|PASS|
| Final focused suite including all-three-kind cache |No failures/errors/skips|49 pass/17.56s|PASS|
| Actual13 descriptor groups |All splits/raw null/metadata/table extents preserved|13/13 pass;26 SWA/DSA CPU rows fit/cold/warm/count|PASS CPU only|

The first producer final failures were not inherited failures: `test_step5_request_free_profiling.py` intentionally expected every isolated scope to be rejected by the old loader. The new requirement permits those truthful diagnostic rows. The tests now require their isolated kind-bound identity and still reject relabeling them with native request-owned scope. No timing or admission rule was relaxed.

## Exact measured test commands

All commands run from the worktree root. Common environment:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
```

Inherited baseline:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest \
tests/unit/test_step5_measured_training.py tests/unit/test_step5_measured_replay.py \
tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_manager.py \
tests/unit/test_step5_measured_interfaces.py tests/unit/test_step5_measured_public_predictor.py \
tests/unit/test_step5_model_bundle.py -q --tb=short -p no:cacheprovider \
--junitxml=/data/ycfeng/tmp/d23_inputs_baseline_measured.xml \
> /data/ycfeng/tmp/d23_inputs_baseline_measured.log 2>&1
```

RED: same executable/env, `-m pytest tests/unit/test_step5_mrv2_inputs.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_inputs_red.xml`, log `/data/ycfeng/tmp/d23_inputs_red.log`.

GREEN01: `test_step5_mrv2_inputs.py test_step5_measured_training.py test_step5_measured_replay.py`; GREEN02: `test_step5_mrv2_inputs.py test_step5_attention_input_union.py`; same options with `d23_inputs_green01` / `d23_inputs_green02` XML/log stems.

Final focused: same two files as GREEN02, after the all-three-kind regression, with `d23_inputs_final_focused` XML/log stem. This includes39 focused kind cases and10 union cases; it overlaps the broad suite and is not an additional independent test population. All final XML files independently show0 failures/errors/skips.

Final measured command:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest \
tests/unit/test_step5_mrv2_inputs.py tests/unit/test_step5_attention_input_union.py \
tests/unit/test_step5_measured_training.py tests/unit/test_step5_measured_replay.py \
tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_manager.py \
tests/unit/test_step5_measured_interfaces.py tests/unit/test_step5_measured_public_predictor.py \
tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_ffn.py \
tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_stage_boundaries.py \
tests/unit/test_step5_measured_pd.py -q --tb=short -p no:cacheprovider \
--junitxml=/data/ycfeng/tmp/d23_inputs_final_measured.xml \
> /data/ycfeng/tmp/d23_inputs_final_measured.log 2>&1
```

## Exact producer command

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu \
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm \
PYTHONPATH=$PWD:/data/ycfeng/tmp/step5_profiling_20261020/extra_site TMPDIR=/data/ycfeng/tmp \
WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
/data/ycfeng/envs/oneshot-opd-py312/bin/python -B -c \
"import sys,types,pathlib; p=types.ModuleType('tests'); p.__path__=[str(pathlib.Path.cwd()/'tests')]; sys.modules['tests']=p; import pytest; raise SystemExit(pytest.main(['tests/unit/test_step5_mrv2_dummy_profiling.py','tests/unit/test_step5_request_free_profiling.py','tests/unit/test_step5_attention_profiling.py','tests/unit/test_step5_device_activity.py','tests/unit/test_step5_fixed_probe.py','tests/unit/test_step5_attention_replay.py','tests/unit/test_step5_attention_judgment.py','tests/unit/test_step5_attention_timing.py','tests/unit/test_step5_boundary_probe.py','-q','--tb=short','-p','no:cacheprovider','--junitxml=/data/ycfeng/tmp/d23_inputs_final_producer02.xml']))" \
> /data/ycfeng/tmp/d23_inputs_final_producer02.log 2>&1
```

The inherited baseline and first final attempt used the identical command with output stems `d23_inputs_baseline_producer` / `d23_inputs_final_producer`. The local `tests` namespace avoids an unrelated installed package, as in the parent's original465 command.

## Actual descriptor evidence

Read only primary local `task_memory/task_2026-10-21_step5_priority_runs/canonical_g3/native_judgment/{report.md,mrv2_descriptors.json,producer_descriptor_check.json}`. An inline Python3.10 command iterated `dummy_descriptors_tp0`, constructed shared `MRv2DummyDescriptor` from the raw descriptor and retained evidence reference, checked exact query arrays, metadata92/92, four tables `[actual_num_reqs,448]`, slots `[4,num_tokens]`, and raw max_query_len round-trip. For layers0/3 it used the checked-in `producer_row` → `_write` (`rows_to_dataframe`) → `load_step5_attention_rows` → `_train` shared manager, separated by actual KERNEL_ONLY/CUDA_EVENT methods. Cold and warm caches checked exact duration0.5/count17 and domain `(1.,)`; no warm estimator creation was permitted.

Executable inline command body is archived in `observed_descriptor_command.md`; invocation was `env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'` with that body and `PY` terminator.

Selected receipt: `observed_descriptors.json` (13 PASS,26 rows,5 FULL/raw-null groups); scratch inputs/caches `/data/ycfeng/tmp/d23_inputs_observed13/{KERNEL_ONLY,CUDA_EVENT}/`. The duration/count values are labelled CPU fixtures, not measured native timings. Persistent regressions exercise source-bound native InputBatch/state/builder behavior for observed FULL1 and FULL2/null and direct producer/consumer round-trip. No new profiling grid or scheduler descriptor was inferred.

## Limits

This proves the producer-to-existing-consumer CPU path, cache/bundle identity and exact cost/count lookup. It does not qualify native timing/G3/state/pressure, role graph-family mapping, scheduler rendezvous or predictive accuracy. The existing one-slot binder remains isolated diagnostic; parent owns its MRv2 replacement. Original native startup192 gate FAIL, reference state/pressure UNVERIFIED and two HTTP502 remain unchanged. Reached context maximum7166 is an observation, not7167. Parent re-runs full acceptance after integration.
