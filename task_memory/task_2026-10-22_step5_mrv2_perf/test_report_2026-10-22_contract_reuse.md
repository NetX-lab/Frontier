# D23-MRV2-CONTRACT-REUSE verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded RED->GREEN fixed-work regression, unchanged cache/trace feedback, targeted tests and limitations. |

## Scope and environment

Worktree `/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf`, branch `feat/step5-mrv2-perf`, diagnostic baseline `74980bb`, task `D23-MRV2-CONTRACT-REUSE`, session `ses_e9f519775ffe3Cv0bhEzJSKNhH`.

CPU-only Python3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`, `-B` and `PYTHONDONTWRITEBYTECODE=1`. All commands run from this worktree; raw artifacts/logs use `/data/ycfeng/tmp`. No GPU/platform/external operation, initialization optimization, original fixture/cache mutation, production output cache, fullCLI retry, increased2400s bound, estimator selection cache or diagnostic script edit. New unit fixtures use the existing small CPU fitting helpers in fresh pytest temp directories, not the original94392/184-key artifacts. Injected count/registry failures are in-memory test objects restored by pytest.

Observed supported lifetime: `BaseExecutionTimePredictor.__init__` binds model/replica config and validates Step role, scheduler, PP/TP/EP groups (`base_execution_time_predictor.py:35-91`). Production prediction does not reassign those bindings. The typed contract uses role-owned `_get_ffn_tp_key_for_linear_op` (`sklearn_execution_time_predictor.py:2889`). Current-family model selection stays in `_get_family_model` (`prediction_model_registry.py:365`) on every query. The cache stores only metadata and fixed precision/serialized contract identity by physical layer/operator; distinct model/TP/EP deployments use separate predictor instances.

## Regression RED -> GREEN

Exact initial command:

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_contract_reuse.py -q --basetemp=/data/ycfeng/tmp/d23_contract_red01
```

Before production edits, output duration/count/precision/layer/features passed, then the fixed-work assertion failed: `Counter({'resolve_module_precision':72,'resolve_layer_contract':30,'describe_step5_layer':12,'_get_family_model':6})`, expected two descriptions. Raw captured output: `/home/brainpp/.opencode-v2/xdg/data/opencode/shell/1e16c1414e84507ddac054d99fb4b6d4b0d5aa0f/sh_160c2ee06001bRMlF7Kq7GYZuX.out` (only decisive failure lines1-33 are needed).

Final regression command:

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_contract_reuse.py -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_green03
```

PASS:7 tests in9.18s. The repeated-query test explicitly separates first construction from warm queries: two first descriptions, then zero descriptions/precision/contract resolution and four current model selections for four actual queries. Other tests verify16/32-token output differences, invalid/extra/out-of-bounds features, missing/invalid count coverage after warm binding, registry replacement/removal without a second model cache, exception-safe nested same/different-layer restoration, independent predictor/attentionTP metadata, FP8/BF16 layers, actual fitted CUDA_EVENT/KERNEL_ONLY family isolation including different model/count hashes, and missing-family errors. Existing role/group constructor rejection tests also pass in the structural group below.

## Unchanged original-scale fast feedback

Exact command (original diagnostic script unchanged):

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --iterations 2 --no-profile --output /data/ycfeng/tmp/d23_contract_wall01
```

Parent baseline `/data/ycfeng/tmp/d23_parent_red01/receipt.json`:64 queries0.532956s,loading1.468s,RED/sourceTruthPASS at74980bb. Corrected work:64 EP lane queries/456 operator queries0.073476986s,loading1.475100046s,sourceTruthPASS,RED/exit1. Speedup7.253x,86.21% wall reduction. Every `costs` dictionary and artifact descriptor compares exactly equal to parent baseline; feature scale, source hashes, paired counts, physical layer/precision and all lane populations are retained.

Measured metadata total0.013793973s, operator total0.058792206s. First layer3/88 queries0.011140663/0.009286188s. Excluding only those two first binds when reporting warm distribution (not when deciding the unchanged gate):62 queries mean0.843437ms,max1.256945ms. The unmodified gate uses max over all64 queries against0.968239524ms from the original2400s budget, so it remains RED. This is an explicit remaining bound, not a reason to bypass construction or change the gate. Approximately1.19s/round warm FFN cost is inferred from88 routed layers x16 lanes x measured warm mean; it is not a measured scheduler round or full1024 result.

Enabled trace command:

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --iterations 2 --no-profile --trace --output /data/ycfeng/tmp/d23_contract_trace01
```

SourceTruthPASS; work0.073809794s; trace456 events/344788bytes/0.072226543s; persisted truthPASS; RED/exit1. Artifacts: `receipt.json`, `op_traces.jsonl`. Earlier baseline trace was228 events/172466bytes/0.04050s, a different query count; per-event cost decreased from approximately177.6us to158.4us, but the full MetricsStore event loop is not this trace seam.

Profile command:

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --iterations 2 --output /data/ycfeng/tmp/d23_contract_profile01
```

SourceTruthPASS,RED/exit1;0.140356977s measured with cProfile. Artifacts: `prediction.prof`, `profile.txt`, `receipt.json`. Dominant cumulative costs now include912 duration/count lookups0.051s and456 proof JSON decodes0.034s. `_get_family_model`:456 calls/~0.005s, intentionally still dynamic. No source-proof/numeric validation suppression.

| Call | Before profile (32 lane queries) | After profile (64 lane queries) |
| --- | ---: | ---: |
| `describe_step5_layer` |32|2|
| `resolve_module_precision` |192|12|
| `resolve_layer_contract` |228|18|
| `validate_step5_config` |97|7|
| `_get_family_model` |228|456|

Baseline `/data/ycfeng/tmp/d23_perf_ffn06/prediction.prof` has half the lane/operator count; do not present raw counts as equal-work comparison. The persistent warm-phase regression supplies the exact work-elimination gate.

## Relevant existing tests

All use the same environment prefix above and `timeout 60s`; logs are redirected to the listed exact path. Successful exits are backed by test assertions and inspected final summaries.

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_replay.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_measured_interfaces.py tests/unit/test_on_demand_prediction_contract.py -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_existing_measured01 > /data/ycfeng/tmp/d23_contract_existing_measured01.log 2>&1
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_measured_public_predictor.py tests/unit/test_measured_attention_root_metrics.py tests/unit/test_metrics_stage_execution_time.py -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_public_root01 > /data/ycfeng/tmp/d23_contract_public_root01.log 2>&1
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_forward02 > /data/ycfeng/tmp/d23_contract_forward02.log 2>&1
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_structural_execution.py -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_structural02 > /data/ycfeng/tmp/d23_contract_structural02.log 2>&1
```

| Group | Expected | Actual | Status |
| --- | --- | --- | --- |
| Measured FFN/attention/fitting/on-demand contract |all selected assertions pass|91 passed35.54s|PASS|
| Public stage/root/metrics |all selected assertions pass|44 passed40.46s|PASS|
| MRv2 forward/lane agreement |all selected assertions pass|75 passed14.11s|PASS|
| Structural/group guards |all selected assertions pass|38 passed,2 failed17.81s|FAIL,unrelated stale stub|

Structural failures: `test_first_serving_event_includes_input_scope_and_preserves_launch_start[prefill/decode]`, `tests/unit/test_step5_structural_execution.py:421`, `frontier/events/replica_stage_schedule_event.py:81`: `AttributeError: 'types.SimpleNamespace' object has no attribute 'take_forward_start_batch'`. These unchanged files differ by zero lines from74980bb; failure occurs before predictor pricing, on an old stub still exposing `pop_batch_if_not_busy`. No repair to parent files. This is inspected evidence of unrelated scope, not a baseline rerun claim.

Optional combined/split PD commands below exceeded the same60s bound; incomplete verdict, not PASS. Stop retries because these transfer/scheduler fixtures do not exercise the fixed-binding seam. No timeout increase.

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py tests/unit/test_step5_structural_execution.py tests/unit/test_step5_measured_pd.py -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_forward01 > /data/ycfeng/tmp/d23_contract_forward01.log 2>&1
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_measured_pd.py -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_pd02 > /data/ycfeng/tmp/d23_contract_pd02.log 2>&1
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_measured_pd.py -k 'not measured_nixl_uses_covered_read_cost_and_prompt_tail_in_existing_owners and not shared_legacy_glm_nixl_keeps_non_dummy_full_prompt_contract' -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_pd03 > /data/ycfeng/tmp/d23_contract_pd03.log 2>&1
```

## Scope acceptance and limits

Final source commit: `ee28bf7ea43d3c7c73b7176eeb10b6f07d51492a`.

Final regression, with explicit PREFILL/DECODE instances and supported full-EP layouts at the approved seam:

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf:/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_contract_reuse.py -q --show-capture=no --basetemp=/data/ycfeng/tmp/d23_contract_green_final01 > /data/ycfeng/tmp/d23_contract_green_final01.log 2>&1
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --iterations 2 --no-profile --trace --output /data/ycfeng/tmp/d23_contract_committed01
```

Regression7 PASS9.15s. Committed-source receipt explicitly identifies `ee28bf7`:64 lanes/456 operator queries0.074077174s,loading1.490313328s,7.194605x baseline speedup. Every cost dictionary and artifact descriptor exactly equals parent baseline. SourceTruthPASS, unchanged gateRED/exit1. Warm62-query mean0.844448ms. Trace456 events/344787bytes0.068545761s,persisted truthPASS; one-byte size variation reflects serialized measured trace timing, not dropped events or proof differences.

`git diff --check`:PASS. Diagnostic scripts, original fixture/cache artifacts, parent event/test sources unchanged. Production Step cost module220->263 lines; sklearn_moe1814->1791; over-limit core modules untouched, recorded in the existing size-boundary document. No debug instrumentation added to production.

Fixed construction regression is GREEN and all exact source/proof assertions are PASS. The unchanged max-query latency gate is still RED; original1024 offline/online completion, initialization998/1004s attribution, full scheduler/public pipeline performance, native admission and strict77 are not verified by this subtask. Parent owns review, integration and those acceptance runs. This report does not authorize additional optimization.
