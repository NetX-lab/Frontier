# D23-MRV2-FORWARD CPU Test Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive executable CPU results and original-source RED evidence. |

## Environment and scope

- Host: kun-workspace-vgen2. CWD:
  `/data/ycfeng/Frontier/.worktrees/step5-mrv2-forward`.
- Commit: `3cad8dbe398304d798a323389f9e3d18500a42de`, base `5c2319cd`.
- Main CPU Python: `/data/ycfeng/envs/frontier-py310/bin/python`, Python
  3.10.6, GCC 11.3.0. Producer CPU Python:
  `/data/ycfeng/envs/oneshot-opd-py312/bin/python`, Python 3.12.13,
  Clang 22.1.3. No conda activation was used.
- All measurement values in these tests are explicitly unadmitted CPU fixtures.
  No GPU, authentication, platform, SSH, cloud or push operation was performed.

## Results

| Check | Expected | Observed | Status |
| --- | --- | --- | --- |
| Actual first-stage-event regression on original event source | Fail before agreement | 12/12 fail, missing descriptor before prediction | RED confirmed |
| Focused first-call/lifecycle/producer-cache-count-lookup | All supported inputs bind before first prediction | 30 passed, 7.408s | PASS |
| Affected stage/context/EP/width/engine checks, excluding inherited selector file | No regression | 176 passed, 12.117s | PASS |
| Measured fit/replay/bundle/FFN/PD | No source inconsistency remains | 268 passed, 213.415s | PASS |
| Producer baseline | Existing truthful descriptor/schema tests preserved | 467 passed, 22.030s | PASS |
| DP dummy runtime, EP materialization, placement | Existing runtime unchanged | 55 passed, 124.094s | PASS |
| Broadened stage suite including legacy selector file | No new failing case | 204 passed, 7 failed | Inherited failures only |
| Immutable inputs worktree legacy selector baseline | Reproduce exact failing cases/errors | 30 passed, same 7 failures, 2.328s | Baseline match |

All green XML suites have zero failure/error/skip. Broadened selector failures
are identical by `(classname, name, failure message)` on candidate and base:
3 spec request fixtures lack `num_processed_tokens`; 4 shared-manager fixtures
lack attention geometry (`int(None)`). These unrelated fixtures were not edited.

First-call assertions inspect P/D, useful+dummy and asymmetric useful+useful,
FULL/PIECEWISE/NONE/capture miss, local metadata that is not agreement, original
query/context/request arrays, null dispatch fields and real InputBatch dummy
splits. Costs and kernel counts come through cold fit, warm cache (refit forbidden),
published/loaded bundle and exact runtime lookup in the same first event call.
Missing kind coverage rejects in both directions. Empty lane wake uses the real
V1 engine and arrival event. Four-layer event completion exercises owner
promotion/restoration and terminal release for all-MoE and dense0/1/2→MoE3
layouts, with zero dummy serving requests/layer credit. Held dispatched context
survives later request preemption. EP NONE keeps local physical widths.

## Exact commands

Main CPU prefix for each pytest invocation below:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest
```

Append `-q --tb=short -p no:cacheprovider` and the stated XML/log suffix. The
literal complete invocations are:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_final_focused.xml > /data/ycfeng/tmp/d23_forward_final_focused.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py tests/unit/test_shared_forward_group_admission.py tests/unit/test_stage_execution_context.py tests/unit/test_monolithic_mixed_forward_sync.py tests/unit/test_forward_sync_state.py tests/unit/test_cuda_graph_activation_evidence.py tests/unit/test_stage_admission_pp_tools.py tests/unit/test_predictor_effective_tokens.py tests/unit/test_moe_routing_conservation.py tests/unit/test_vllm_v1_engine_idle_time.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_final_stage02.xml > /data/ycfeng/tmp/d23_forward_final_stage02.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_inputs.py tests/unit/test_step5_attention_input_union.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_measured_replay.py tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_manager.py tests/unit/test_step5_measured_interfaces.py tests/unit/test_step5_measured_public_predictor.py tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_stage_boundaries.py tests/unit/test_step5_measured_pd.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_final_measured.xml > /data/ycfeng/tmp/d23_forward_final_measured.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_step5_pd_placement.py tests/integration/test_vllm_dp_dummy_pass_runtime.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_dp_runtime.xml > /data/ycfeng/tmp/d23_forward_dp_runtime.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm PYTHONPATH=$PWD:/data/ycfeng/tmp/step5_profiling_20261020/extra_site TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/oneshot-opd-py312/bin/python -B -c "import sys,types,pathlib; p=types.ModuleType('tests'); p.__path__=[str(pathlib.Path.cwd()/'tests')]; sys.modules['tests']=p; import pytest; raise SystemExit(pytest.main(['tests/unit/test_step5_mrv2_dummy_profiling.py','tests/unit/test_step5_request_free_profiling.py','tests/unit/test_step5_attention_profiling.py','tests/unit/test_step5_device_activity.py','tests/unit/test_step5_fixed_probe.py','tests/unit/test_step5_attention_replay.py','tests/unit/test_step5_attention_judgment.py','tests/unit/test_step5_attention_timing.py','tests/unit/test_step5_boundary_probe.py','-q','--tb=short','-p','no:cacheprovider','--junitxml=/data/ycfeng/tmp/d23_forward_final_producer.xml']))" > /data/ycfeng/tmp/d23_forward_final_producer.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -c "import subprocess,sys,pytest; import frontier.events.replica_stage_schedule_event as event; source=subprocess.check_output(['git','show','5c2319cd:frontier/events/replica_stage_schedule_event.py'],text=True); exec(compile(source,'5c2319cd:replica_stage_schedule_event.py','exec'),event.__dict__); raise SystemExit(pytest.main(['tests/unit/test_step5_mrv2_forward.py','-k','stage_event_first_prediction','-q','--tb=short','-p','no:cacheprovider']))" > /data/ycfeng/tmp/d23_forward_original_event_red.log 2>&1
```

The broadened stage command was the second command above plus
`tests/unit/test_measurement_family_selector.py`, with suffix
`d23_forward_final_stage`; it ran before the two additional dense-layout test
parameters (211 cases then, 30 final focused cases now). Baseline command ran
in immutable input worktree:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_measurement_family_selector.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_inherited_selector_baseline.xml > /data/ycfeng/tmp/d23_forward_inherited_selector_baseline.log 2>&1
```

## Verified limits and deferred work

- Strict77 legacy artifact matrix and full fresh CLI run belong to parent after
  consolidation. They were not rerun or claimed passing here.
- Existing PP1/no-prefix/no-MTP and topology restrictions remain unchanged.
- Existing `docs/general/module-size-exceptions.md:49-59,107-135` was inspected.
  Giant predictor gains only a seven-line delegate enabling the already-existing
  measured families; graph-policy classification stays in the small module.
  No giant manager change, second registry/publisher, scheduler queue or new
  event type. Split remains the documented later measurement-state move.
- This CPU result does not establish native qualification or accuracy. Original
  G3 startup gate FAIL/192, state-pressure UNVERIFIED and HTTP502 observations
  remain parent-owned and unchanged.
