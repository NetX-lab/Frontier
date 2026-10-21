# D23-MRV2-FORWARD CPU Test Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive executable CPU results and original-source RED evidence. |
| 2026-10-22 | Add P2 capacity correction RED/GREEN,489 affected passes and exact commands. |

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

## P2 review correction validation

Initial correction HEAD: `d8e3e941`, source `3cad8dbe`. Same worktree and
Python3.10.6/GCC11.3.0, pytest9.1.1; no conda activation. Only local CPU
tests ran. Parent reported444 affected+measured passes on `3cad8dbe`;
the repeated same suites now include45 additional cases (489 total).
No producer suite, full CLI, strict77, GPU/platform/auth/cloud/SSH/push or
fresh CLI fixture access was performed.

Source mapping: locked `vllm/v1/worker/gpu/cudagraph_utils.py:158` uses
`scheduler_config.max_num_seqs`; uniform no-MTP FULL candidates at321-326
cannot exceed that request capacity. PIECEWISE candidates at346-365 have
`num_reqs=None` and can use wider token captures. Frontier V1
`vllm_v1_engine_replica_scheduler.py:181-183` maps `batch_size_cap` to the
existing `_max_num_running_reqs`. Each DP engine of a Replica uses the same
replica scheduler configuration. Agreement now takes that existing value;
it does not infer request capacity from token capture sizes.

| Check | Expected | Observed | Status |
| --- | --- | --- | --- |
| Tight real V1 idle-lane wakeup/arrival/stage first-call RED before behavior edits | cap3/captures[1,2,4,8] decode3+dummy1: PIECEWISE4 for full_and_piecewise; NONE3/1 for full_decode_only |4 failed,2 passed,29 deselected;12.54s | RED confirmed |
| Same6 real engine/event cases after correction | Correct modes/widths for both P/D roles; FULL1 remains legal |6 passed,29 deselected;11.98s | PASS |
| Complete focused first-call/lifecycle/lookup suite | Both arrival orders, useful+useful/dummy, FULL<=cap, miss, other policies, prefill7->PIECEWISE8, unchanged owner/query/context/null fields |75 passed;14.85s | PASS |
| Same affected stage suite as earlier176-case check | All previous cases plus45 new cases |221 passed;18.75s | PASS |
| Same measured fit/replay/bundle/FFN/PD suite | Required source contracts and explicit direct-call capacity remain correct |268 passed;202.83s | PASS |
| Whitespace integrity | No diff whitespace errors | `git diff --check` exit0 | PASS |

Exact RED assertions, preserved before the implementation edit:

```text
test_empty_lane_wakeup_runs_real_v1_engine_dummy_and_stage_arrival[full_and_piecewise-3-PIECEWISE-widths1-prefill]
test_empty_lane_wakeup_runs_real_v1_engine_dummy_and_stage_arrival[full_and_piecewise-3-PIECEWISE-widths1-decode]
E   AssertionError: assert ['full', 'full'] == ['piecewise', 'piecewise']
test_empty_lane_wakeup_runs_real_v1_engine_dummy_and_stage_arrival[full_decode_only-3-NONE-widths2-prefill]
test_empty_lane_wakeup_runs_real_v1_engine_dummy_and_stage_arrival[full_decode_only-3-NONE-widths2-decode]
E   AssertionError: assert ['full', 'full'] == ['none', 'none']
```

All GREEN XML has0 failures/errors/skips. RED XML has6 tests,4 failures,
0 errors/skips. Raw logs/XML are `/data/ycfeng/tmp/d23_forward_p2_` plus
`red`, `green`, `focused`, `stage` or `measured` and `.log`/`.xml`.
The observed worktree diff changes only2 core modules,2 focused unit test
files and this task's records. There is no change to iteration metadata,
timing/cost providers, the policy registry, owner lifecycle or CLI fixtures.

### P2 exact commands

All commands ran in `/data/ycfeng/Frontier/.worktrees/step5-mrv2-forward`.
The RED command returned1 with the exact assertions above; all other test
commands returned0. The real first-call seam is the existing V1 engine dummy
wakeup -> `BatchStageArrivalEvent` -> `ReplicaStageScheduleEvent` -> agreement
-> first workload prediction, not just a policy-function assertion.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py -k empty_lane_wakeup_runs_real_v1_engine_dummy_and_stage_arrival -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_p2_red.xml > /data/ycfeng/tmp/d23_forward_p2_red.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py -k empty_lane_wakeup_runs_real_v1_engine_dummy_and_stage_arrival -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_p2_green.xml > /data/ycfeng/tmp/d23_forward_p2_green.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_p2_focused.xml > /data/ycfeng/tmp/d23_forward_p2_focused.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py tests/unit/test_shared_forward_group_admission.py tests/unit/test_stage_execution_context.py tests/unit/test_monolithic_mixed_forward_sync.py tests/unit/test_forward_sync_state.py tests/unit/test_cuda_graph_activation_evidence.py tests/unit/test_stage_admission_pp_tools.py tests/unit/test_predictor_effective_tokens.py tests/unit/test_moe_routing_conservation.py tests/unit/test_vllm_v1_engine_idle_time.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_p2_stage.xml > /data/ycfeng/tmp/d23_forward_p2_stage.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_inputs.py tests/unit/test_step5_attention_input_union.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_measured_replay.py tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_manager.py tests/unit/test_step5_measured_interfaces.py tests/unit/test_step5_measured_public_predictor.py tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_stage_boundaries.py tests/unit/test_step5_measured_pd.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_p2_measured.xml > /data/ycfeng/tmp/d23_forward_p2_measured.log 2>&1

git diff --check
```

### P2 limits

Observed tests prove the corrected CPU dispatch and first-call binding and
preservation assertions under the supported no-MTP path. They do not execute
native CUDA graphs or establish native accuracy. Parent owns independent
re-review, merge, complete fresh CLI and strict77. This correction introduces
the required keyword `request_capacity` on `bind_forward_start`; direct callers
must supply the existing scheduler capacity, never the max capture size.

## Fixed-capacity primary verification

Primary resumed the saved one-line change and focused config-fixture updates
after the owner encountered provider HTTP502. This is a source invariant
clarification: native captures use the fixed `max_num_seqs` at construction;
the equivalent Frontier input is `_config.batch_size_cap`. The only mutable
admission override requires thinking requests, which the current Step config
rejects. Existing supported-state tests pass before and after; no artificial
thinking state or RED counterexample is asserted.

Environment: `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6,
pytest9.1.1; CWD `/data/ycfeng/Frontier/.worktrees/step5-mrv2-forward`.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py tests/unit/test_shared_forward_group_admission.py tests/unit/test_stage_execution_context.py tests/unit/test_monolithic_mixed_forward_sync.py tests/unit/test_forward_sync_state.py tests/unit/test_cuda_graph_activation_evidence.py tests/unit/test_stage_admission_pp_tools.py tests/unit/test_predictor_effective_tokens.py tests/unit/test_moe_routing_conservation.py tests/unit/test_vllm_v1_engine_idle_time.py tests/unit/test_step5_measured_public_predictor.py tests/unit/test_step5_structural_pd.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_fixed_parent_takeover01.xml > /data/ycfeng/tmp/d23_fixed_parent_takeover01.log 2>&1
```

Expected: supported arrival-order/cap3/wide-PIECEWISE behavior, first-call
identity, owner completion, direct predictor and Step thinking rejection all
pass. Observed: **242 passed in33.92s**, zero failures/errors/skips; PASS.
Parent full CLI and clean77 remain separate integration gates. No GPU or
measurement-table work was part of this verification.
