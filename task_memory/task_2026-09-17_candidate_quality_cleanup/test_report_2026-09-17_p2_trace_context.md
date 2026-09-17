## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Completed M4/M5, focused regressions, full event equality and construction-count evidence; corrected DP classification from complete caller RCA and removed unreachable shape support. |

# P2 Trace Context and GDN Metadata Cleanup

## Scope and plan

- Pre-change HEAD: `b0b9683b8fe6d03a6465b6e8fd7ac5662175d415`. Both owned production files were clean. Existing edits in `tests/unit/test_metrics_stage_execution_time.py` belong to main and are preserved.
- Own only `frontier/metrics/op_trace_utils.py`, `frontier/metrics/ep_wave_metrics.py`, this report, and a new minimal regression file if needed. No commits, other production/test edits, P3/P4 work, or subagents.
- Dependency: `read M4/M5 and producer/sink contracts -> focused baseline -> bounded edits -> event equality/call-count checks -> focused regression -> main handoff`.
- Status: completed; ready for main integration and source-stable broader P2/E2E verification. No commit made.
- Codebase-design informed reuse of the existing family lookup and phase-local context lifetime. Planning-with-files is scoped to this authorized report; shared task records remain main-owned.

## Preserved invariants

- M4 uses the attention-family descriptor and existing `_get_family_operator_by_name`, retaining visible GDN input/output `[tokens, hidden_size]` payloads and all precision/size metadata.
- M5 retains positive-operator order, per-lane work, phase barrier starts, cursor advancement, routing token semantics, metrics/ledger contents and per-operator tensor metadata.
- `ExecutionTime.moe_phase_operator_times()` returns an immutable tuple. Context construction is allowed only for a traced phase containing positive work. Lane token summation occurs only with tracing enabled. No persistent cache is introduced.
- `TraceStore.log_event()` buffers events without modifying metadata. Preserve independent per-event parallel metadata dictionaries even when their values are computed once per phase.

## Environment and execution

CPU; Python 3.12.3; `/data/ycfeng/tmp/quality-review-env/bin/python`, isolated uv venv with same-interpreter system-site-packages inheritance; no conda or GPU execution. Commands run from `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

Focused baseline command:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_op_trace_utils.py tests/unit/test_attention_trace_mapping.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_trace.py tests/unit/test_typed_ep_trace_contract.py tests/unit/test_stage_reporting_contract.py::test_ep_lane_reporting_preserves_work_and_barrier_gaps -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p2-trace-context-baseline-20260917
```

Observed: **56 passed in 6.48s**. This establishes the focused pre-change control, not full P2 or simulator fidelity acceptance.

## Regression development and observed failures

- First new-test run: **12 failed in 6.72s**, all in fixture construction because `dp_output_allreduce` is not accepted in `ExecutionTime.op_times`. It has a supported scalar argument `dp_output_allreduce_time`; corrected the fixture instead of relaxing production validation.
- Second run with the scalar argument: **6 passed / 6 failed in 6.39s**. Four failures expose the missing GDN family lookup as intended; the zero-work tracing failure reflects the planned one-per-traced-lane token sum. The positive-work case failed because `compute_op_trace_meta('dp_output_allreduce', 'COMM', context)` reaches `QuantizationManager.get_precision` with the unsupported DP name. Initial classification as a production bug was incomplete; the completed caller RCA below supersedes it.
- Completed DP RCA (main-provided and locally verified): `BaseExecutionTimePredictor.predict_dp_moe_allreduce_times` at `base_execution_time_predictor.py:413` explicitly retires the attention-DP scope and returns `(0.0, 0.0)`. No production override exists; MoE/disaggregation callers use this seam or initialize the values to zero. Positive DP tracing in the synthetic fixture is therefore **unsupported current runtime, not a production bug**. Existing scalar timing construction and the retired-zero seam remain supported and unchanged. Removed only the two candidate-added unreachable DP names from the COMM shape tuple; precision validation already rejected them before reaching shape handling. No registry entry, alias, fallback or semantic change was introduced. The five-positive-operator fixture leaves DP durations at their production zero values.
- Concurrent main commits advanced HEAD to `3a080f5c32f2b16056fd0a1f7b4f10d75e07cba3`; comparison confirms both owned production files remain byte-identical to the recorded baseline before our edits.
- Corrected-fixture pre-change run: **6 passed / 6 failed in 6.07s**; failures were the intended context count (5 versus 3), traced-lane sum (0 versus 1 for zero work), and four missing GDN family-lookup calls. After M4/M5: **12 passed in 6.11s**. Expanded focused run: **83 passed in 7.05s**.
- Per main's scaffolding review, removed the four redundant GDN lookup-spy tests and per-operator/sum/parallel implementation spies. The final new file is 113 lines and eight parameterized cases. It retains one context-construction spy, reporting-off/zero-work coverage, event timestamp/duration values, routing-sensitive tensor metadata, layer identity and independent parallel dictionaries. Existing GDN tests plus direct equality below cover M4. Fixture lane compute includes all three compute phases, and zero-work timing is zero throughout.

## Implemented refactors

| Module | Problem / root cause | Refactor / reused owner | Preserved behavior / verification |
| --- | --- | --- | --- |
| `frontier/metrics/op_trace_utils.py:537` | Consumer duplicated four GDN operator names already owned by the family descriptor. | Existing `_get_family_operator_by_name(GATED_DELTA_NET_ATTENTION_FAMILY, op_name)` replaces the tuple. No general registry expansion. | Four operators' complete metadata matches baseline; visible input/output `[8, 128]`, each 2,048 bytes in the FP16 fixture. |
| `frontier/metrics/ep_wave_metrics.py:40` | Lane token sum, context and parallel metadata were rebuilt for every operator despite phase invariance. | Sum once per traced lane; use existing `OpTraceContext` and `build_parallel_context` once per positive traced phase; keep operator metadata per operator. | Full events, metrics calls and ledger equality in 64 combinations. Context/parallel construction 5 -> 3, lane sum 5 -> 1, operator metadata remains 5. |
| `frontier/metrics/op_trace_utils.py` COMM shape tuple | Added DP shape labels represented a retired runtime scope and were unreachable behind precision validation. | Remove only `dp_input_allreduce` / `dp_output_allreduce` labels; retain the existing predictor's explicit zero contract and public scalar timing fields. | Targeted retired-zero regression plus focused trace suite; unsupported positive DP inputs still fail at the same precision validation. |

Retained intentionally: tracing gate and positive-work guard avoid trace validation/context construction for disabled or empty phases; ledger optionality is configuration-driven. `parallel_context.copy()` preserves independent event dictionaries; its values are flat integers, so no deep copy is needed. No production attributes, persistent caches, optional sentinels or compatibility paths were added.

Changed paths, exclusively:

- `frontier/metrics/op_trace_utils.py`
- `frontier/metrics/ep_wave_metrics.py`
- `tests/unit/test_ep_wave_trace_context.py` (new)
- `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p2_trace_context.md`

Production diff: 19 insertions / 17 deletions across the two owned files. Other workers' scheduler/reporting/test/task-record changes were not edited or staged.

## Final focused verification

Criteria: existing attention/MLA/EP trace and materialization behavior remains valid; only positive traced phases construct contexts; emitted event metadata and isolation remain unchanged. Exact command:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_op_trace_utils.py tests/unit/test_attention_trace_mapping.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_trace.py tests/unit/test_typed_ep_trace_contract.py tests/unit/test_stage_reporting_contract.py::test_ep_lane_reporting_preserves_work_and_barrier_gaps tests/unit/test_ep_wave_trace_context.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_prefill_ep_wave_materialization.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p2-trace-context-final-20260917 --tb=short
```

**PASS: 79 passed in 7.21s** before the explicitly requested unreachable DP-label deletion. Final verification adds `tests/unit/test_moe_predictor_layer_id_semantics.py::test_attention_dp_moe_communication_is_zero_when_shared_ep_owns_collective` to this same command, with `--basetemp=/data/ycfeng/tmp/quality-p2-trace-context-retired-zero-20260917`. `git diff --check -- frontier/metrics/op_trace_utils.py frontier/metrics/ep_wave_metrics.py tests/unit/test_ep_wave_trace_context.py task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_p2_trace_context.md` returned success with no output.

**Final PASS: 80 passed in 7.12s**, including the retired-zero regression, after all production/test edits. No further owned production/test changes or running suites remain. Exact final command:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_op_trace_utils.py tests/unit/test_attention_trace_mapping.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_trace.py tests/unit/test_typed_ep_trace_contract.py tests/unit/test_stage_reporting_contract.py::test_ep_lane_reporting_preserves_work_and_barrier_gaps tests/unit/test_ep_wave_trace_context.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_moe_predictor_layer_id_semantics.py::test_attention_dp_moe_communication_is_zero_when_shared_ep_owns_collective -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p2-trace-context-retired-zero-20260917 --tb=short
```

## Direct pre/post comparison

Criteria: exact equality, with no numeric tolerance, of each full `TraceEvent.to_dict()`, metric call and lane ledger row. Six Boolean dimensions are tracing, operation metrics, ledger, zero work, layer expansion and global `write_metrics`: 64 combinations. The same batch/plan is reused within each before/after pair, so IDs are not normalized away. This isolates the two owned modules against `b0b9683b` using the same current dependencies.

Observed output:

```text
EP equality: PASS 64 cases; 80 events per revision; full event/metric/ledger values equal
GDN equality: PASS 4 operators; all metadata equal, visible shape [8,128], 2048 bytes per input/output
Call counts [context, parallel, operator metadata, lane token sum]: PASS [[5, 5, 5, 5], [3, 3, 5, 1]]
Separate DP precision issue: confirmed same ValueError before and after; no compatibility fallback added
```

The diagnostic's historical label "Separate DP precision issue" is superseded by the completed RCA: this rejection is for unsupported synthetic positive DP tracing, not a reachable runtime defect.

Exact executed command (inline diagnostic only; no additional persistent test/script):

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY'
import itertools
import subprocess
import sys
import types
from unittest.mock import Mock

from frontier.attention.families import GATED_DELTA_NET_ATTENTION_FAMILY
from frontier.metrics import ep_wave_metrics, op_trace_utils
from frontier.types import ClusterType
from tests.unit.test_ep_wave_trace_context import _reporting_case

revision = 'b0b9683b8fe6d03a6465b6e8fd7ac5662175d415'
def baseline_module(name, path):
    module = types.ModuleType(name)
    sys.modules[name] = module
    source = subprocess.check_output(['git', 'show', f'{revision}:{path}'], text=True)
    exec(compile(source, f'{revision}:{path}', 'exec'), module.__dict__)
    return module

before_meta = baseline_module('p2_before_meta', 'frontier/metrics/op_trace_utils.py')
before_wave = baseline_module('p2_before_wave', 'frontier/metrics/ep_wave_metrics.py')
for name in ('OpTraceContext', 'build_parallel_context', 'compute_op_trace_meta'):
    setattr(before_wave, name, getattr(before_meta, name))

cases = event_count = 0
for tracing, operations, ledger, zero_work, expand, write in itertools.product((False, True), repeat=6):
    store, plan = _reporting_case(tracing=tracing, operations=operations,
                                  ledger=ledger, zero_work=zero_work)
    store._config.write_metrics = write
    store._should_expand_layers.return_value = expand
    outputs = []
    for reporter in (before_wave.record_ep_wave, ep_wave_metrics.record_ep_wave):
        store.trace_store.log_event.reset_mock()
        store._push_metric.reset_mock()
        store._frontier_ep_wave_lane_ledger_rows.clear()
        reporter(store, plan, time=1.0, replica_id=0, stage_id=2,
                 cluster_type=ClusterType.MONOLITHIC)
        outputs.append((
            [call.args[0].to_dict() for call in store.trace_store.log_event.call_args_list],
            list(store._push_metric.call_args_list),
            list(store._frontier_ep_wave_lane_ledger_rows),
        ))
    assert outputs[0] == outputs[1], (tracing, operations, ledger, zero_work, expand, write)
    cases += 1
    event_count += len(outputs[1][0])
print(f'EP equality: PASS {cases} cases; {event_count} events per revision; full event/metric/ledger values equal')

store, plan = _reporting_case()
replica = store._cluster_configs[ClusterType.MONOLITHIC].replica_config
ctx = op_trace_utils.OpTraceContext(ClusterType.MONOLITHIC, replica.model_config,
                                   replica, 8, 8, 8, 8, False)
for operator in GATED_DELTA_NET_ATTENTION_FAMILY.e2e_trace_ops():
    old = before_meta.compute_op_trace_meta(operator.name, 'COMPUTE', ctx)
    new = op_trace_utils.compute_op_trace_meta(operator.name, 'COMPUTE', ctx)
    assert old == new
    assert new['tensor_shape'] == {'input': [8, 128], 'output': [8, 128]}
    assert new['tensor_size_bytes'] == {'input': 2048, 'output': 2048}
print('GDN equality: PASS 4 operators; all metadata equal, visible shape [8,128], 2048 bytes per input/output')

counts = []
for module in (before_wave, ep_wave_metrics):
    originals = {name: getattr(module, name) for name in (
        'OpTraceContext', 'build_parallel_context', 'compute_op_trace_meta')}
    for name, function in originals.items():
        setattr(module, name, Mock(wraps=function))
    module.sum = Mock(wraps=sum)
    module.record_ep_wave(store, plan, time=1.0, replica_id=0, stage_id=2,
                          cluster_type=ClusterType.MONOLITHIC)
    batch = plan.phase_times.lane_records[0].batch
    counts.append([getattr(module, name).call_count for name in originals] + [
        sum(call.args[0] is batch.num_tokens for call in module.sum.call_args_list)])
    for name, function in originals.items():
        setattr(module, name, function)
    del module.sum
assert counts == [[5, 5, 5, 5], [3, 3, 5, 1]], counts
print(f'Call counts [context, parallel, operator metadata, lane token sum]: PASS {counts}')
for module in (before_meta, op_trace_utils):
    try:
        module.compute_op_trace_meta('dp_output_allreduce', 'COMM', ctx)
    except ValueError as error:
        assert "Unsupported operation 'dp_output_allreduce'" in str(error)
    else:
        raise AssertionError('DP baseline failure unexpectedly disappeared')
print('Separate DP precision issue: confirmed same ValueError before and after; no compatibility fallback added')
PY
```

## Handoff and limits

- M4/M5 implementation and focused verification: complete. No pending owned edits or tests; no commit.
- Main's next steps: (1) integrate the four exact paths, (2) run source-stable broader P2 and eight E2E cases. Open owned issues: none; no DP semantic decision or registry expansion is required.
- No simulator wall-clock, full-unit, native-device or P3/P4 claim is made. Reduced constructor counts are observed work reduction, not an E2E speedup measurement. The direct fixture is one lane; existing focused EP/materialization tests supply multi-lane/barrier coverage.
