# D23-ROOT-METRICS Test Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive RED/GREEN, call-site metadata, ownership checks, and verification limits. |

## Environment

- Worktree: `/data/ycfeng/Frontier/.worktrees/step5-root-metrics`.
- Baseline: `61cbac9a3e8c29190b91a5d4e8f7fa2ad9fcd359`.
- Verified source commit: `3a189ba`.
- Python: `/data/ycfeng/envs/frontier-py310/bin/python`, 3.10.6, GCC 11.3.0; pytest 9.1.1. No separate conda activation.
- Environment for tests/repros: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp`; Python flags `-B`; pytest flags `-q -p no:cacheprovider`.
- CPU-only synthetic fixtures; no actual full CLI, native/GPU runs, or profile admission.

## Commands and observed results

### Original parent reproduction — RED then GREEN

The following command ran before modifying source. RED log: `/data/ycfeng/tmp/d23_root_metrics_red01.log`, exit 1, exact `KeyError: 'step5_layer_e2e'` at `metrics_store.py:4504` through the existing stage ledger.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.entities.time_components import AttentionOperatorTimes, OperatorCost
from frontier.metrics.metrics_store import MetricsStore
from frontier.types import MeasurementType
from tests.unit.test_execution_time_op_times import _base_execution_kwargs
kwargs = {name: False if isinstance(value, bool) else 0. for name, value in _base_execution_kwargs().items()}
kwargs.update(num_layers_per_pipeline_stage=1, global_layer_id=0, attention_family_id="step5_attention", attention_variant_id="swa", attention_operator_times=AttentionOperatorTimes({"step5_layer_e2e": .6244}, costs={"step5_layer_e2e": OperatorCost(.6244, 23., MeasurementType.KERNEL_ONLY, "cpu_fixture_duration", "cpu_fixture_count", "cpu_fixture", True)}))
stage = StageExecutionTime([ExecutionTime(**kwargs)])
object.__new__(MetricsStore)._build_frontier_stage_batch_component_ledger(stage)
PY
```

GREEN rerun used the same input and final call assigned to `ledger`, followed by assertions `ledger['step5_layer_e2e'] == .6244` and `sum(ledger.values()) == stage.total_time * 1000`. Log: `/data/ycfeng/tmp/d23_root_metrics_repro_green01.log`, exit 0:

```text
{'root_ms': 0.6244, 'ledger_sum_ms': 0.6244, 'stage_total_ms': 0.6244, 'kernel_count': 23.0}
```

### Focused regression — RED

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_measured_attention_root_metrics.py -q -p no:cacheprovider
```

- `/data/ycfeng/tmp/d23_root_metrics_tests_red02.log`: `8 failed, 7 passed in 2.95s`, exit 1. Failures specifically hit scalar-map KeyError and the trace inclusive-family whitelist rejection; this is the accepted pre-fix RED.
- Earlier `/data/ycfeng/tmp/d23_root_metrics_tests_red01.log`: `8 failed, 7 passed in 2.36s`, including four invalid shared TP-domain fixture setups. The fixture was corrected before source edits; those setup failures are not counted as bug evidence.
- First source fix: `/data/ycfeng/tmp/d23_root_metrics_tests_green01.log`, `15 passed in 6.11s`.
- Expanded validation briefly failed because its test fixture appended DSA already included by the structural registry: `/data/ycfeng/tmp/d23_root_metrics_tests_green02.log`, 4 failed / 18 passed. The fixture now replaces the existing family instead, without production changes. `/data/ycfeng/tmp/d23_root_metrics_tests_green03.log`: `22 passed in 6.10s`.

### Final focused and adjacent checks — PASS

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_measured_attention_root_metrics.py tests/unit/test_attention_trace_mapping.py tests/unit/test_metrics_stage_execution_time.py tests/unit/test_stage_execution_time.py tests/unit/test_execution_time_op_times.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_replay.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_public_predictor.py -q -p no:cacheprovider -k 'not structured_attention_legacy_coverage_uses_family_execution_time_attrs'
```

Final log `/data/ycfeng/tmp/d23_root_metrics_final_verified.log`: `161 passed, 1 deselected in 27.48s`, exit 0. Includes 23 new focused cases and 138 adjacent cases. Prior equivalent final run `/data/ycfeng/tmp/d23_root_metrics_final_green.log`: `161 passed, 1 deselected in 25.79s`.

### Baseline fixture defects — FAIL, not suppressed production errors

An initial adjacent run included `tests/unit/test_glm53_attention_cost_contract.py` and no deselection: `/data/ycfeng/tmp/d23_root_metrics_legacy_green01.log`, `1 failed, 148 passed, 17 errors in 22.68s`.

- GLM53 fixture calls deleted private `_canonical_operator_execution_time_attr_items`, giving 17 setup errors.
- `test_structured_attention_legacy_coverage_uses_family_execution_time_attrs` monkeypatches the family catalog without clearing its cached attribute registry; in a combined run it observes 0 instead of 1.7.

Both were checked read-only at parent HEAD, with bytecode/cache writes disabled:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_glm53_attention_cost_contract.py::test_inclusive_scopes_round_trip_without_double_count tests/unit/test_attention_trace_mapping.py::test_structured_attention_legacy_coverage_uses_family_execution_time_attrs -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_attention_trace_mapping.py tests/unit/test_metrics_stage_execution_time.py -q -p no:cacheprovider
```

Parent logs `/data/ycfeng/tmp/d23_root_metrics_baseline_failures.log` (`1 passed, 1 error in 0.85s`) and `/data/ycfeng/tmp/d23_root_metrics_baseline_cache.log` (`1 failed, 32 passed in 2.45s`) reproduce the respective defects. No production shim or unrelated old-test edit was made. Independent cache-isolated DSA inclusive projection tests exercise exact totals and missing/extra/legacy-child rejection in the new focused file.

## Acceptance observations

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Original root / ledger / stage | .6244 ms each | .6244 ms each; absolute/relative discrepancy 0 | PASS |
| Original measured count | 23 kernels | 23.0 retained | PASS |
| Two-layer root sum | .6244 + .2 = .8244 ms | .8244 ms, no owner-root 99 ms | PASS |
| Kernel-only gap | .001 ms × 23 = .023 ms | Separate ledger gap .023 ms | PASS |
| Stage-owned work | emb .11, cast .12, final residual .13, PP .14, schedule .15 | Exactly those fields, no nonowner 99 ms | PASS |
| Mixed stage total | 1.8074 ms | Ledger equals StageExecutionTime total 1.8074000000000001 ms; discrepancy 0 | PASS |
| Diagnostic children | .4/.5 ms remain diagnostic, not billed | No child ledger/span/cost; only root | PASS |
| Emitted root proofs | Per-layer counts 23 and 7, exact hashes/evidence/method/lookup | Preserved for CUDA_EVENT/KERNEL_ONLY and expansion on/off | PASS |
| JSONL trace persistence | Same proof after TraceStore flush | OperatorCost dictionaries round-trip unchanged | PASS |
| Root tensor boundaries | Source hidden input/output, actual global layer | [8, hidden] input/output, correct geometry; source evidence separate from measured cost evidence | PASS |
| Invalid scopes | Reject missing/invalid cost, zero positive-duration count, root+child, other family, model/legacy overlaps, unknown name | ValueError, no fallback | PASS |
| Legacy tables | No empty root metric for dense models | Root metric excluded by existing structural family filter | PASS |
| Diff hygiene | No whitespace errors; scoped files only | `git diff --check` exit 0; seven source/test/exception files | PASS |

The mixed-stage inspection log `/data/ycfeng/tmp/d23_root_metrics_mixed_total.log`
records the actual runtime and ledger totals plus every nonzero component;
the `.01 ms` attention norm is separate from the root, not a diagnostic child.

## Limits and unresolved checks

- This task does not prove actual 6144/1024 offline/online CLI completion, READ/readiness/source release, native accuracy, or admission. Parent owns those checks and legacy77.
- Existing scalar aggregate metric/trace adapters are unchanged; the inspected measured P/D producer publishes StageExecutionTime and traces authoritative per-layer records. An aggregate attention map intentionally has no per-layer OperatorCost, so it is not presented as a measured proof.
- Root trace duration remains the measured duration; KERNEL_ONLY cost metadata explicitly says `includes_kernel_gap=False`. Existing component ledgers separately include kernel gap. The legacy trace emitter has no separate kernel-gap spans; this task does not claim trace cursor parity with total occupancy for gap-bearing stages or alter that behavior.
- Two pre-existing old test fixture defects remain unresolved as detailed above. Their exclusion is explicit, not an all-suite PASS.
