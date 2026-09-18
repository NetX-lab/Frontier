## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Completed the authorized P2 M1b/M2 scheduler helper cleanup: 110 before / 114 after focused tests and five byte-identical timing snapshots. |

# P2 M1b/M2 scheduler reporting cleanup

Status: completed; ready for the main agent to commit. Owner: bounded scheduler sidecar. Production edits are limited to `dense_metrics.py`, the reporting wrappers in `base_cluster_scheduler.py`, and their `decode_collective.py` / `prefill_collective.py` callers. Related fixtures/tests are in scope except `tests/unit/test_metrics_stage_execution_time.py`, which the main agent owns. No metrics-store/EP-reporter/trace-utils edits, shared record updates, or commit were performed by this sidecar.

## Source and plan

- Initial HEAD: `b0b9683b8fe6d03a6465b6e8fd7ac5662175d415`.
- Initial dirty path: `tests/unit/test_metrics_stage_execution_time.py` only; preserved and excluded from this test selection.
- Fixed original candidate: `c288a19f59bec09529ee18d782fa57218da2c781`; fixed main: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Dependency: focused baseline -> migrate obsolete helper tests and simplify the live model/copy contract -> focused regression and output comparison -> handoff without commit.
- Use the existing model count/layer APIs and finalized Stage ownership. Tests migrate to live helpers; no new wrapper, cache, fallback or model classification is introduced.

## Preserved invariants

1. Actual dense layer IDs each receive their own predictor result; full-stage scope and explicit stage owner survive correction.
2. Scalar/nonmixed models still return isolated metrics copies without making dense predictions, including zero-MoE and all-MoE cases.
3. Single-layer extraction still rejects a multi-layer prediction; timing payloads, identity and split-TP explicit-zero versus omitted values remain intact.
4. Removing unused wrapper arguments does not alter actual scheduler elapsed time, stage start timestamps, events or reporting scope.

## Execution and evidence

- Before: **110 PASS in 7.13 s**, log `/data/ycfeng/tmp/quality-p2-scheduler-before-20260917.log`.
- Tool issue: `apply_patch` is absent from PATH in this shell. The first report patch did not execute and no source changed; the independently launched baseline passed. Located the installed patch engine at `/home/i-fengyicheng/.local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex`, invoked using `--codex-run-as-apply-patch`. No environment files were modified.
- During discovery, `notes.md` and `task_memory/env_handbook.md` were absent, and the older guessed `vendor/.../codex/codex` binary path was absent. The actual binary is under `vendor/.../bin/codex`.
- After: **114 PASS in 7.10 s**, log `/data/ycfeng/tmp/quality-p2-scheduler-after-20260917.log`. The four added cases cover scalar, dense, zero-MoE and all-MoE copy/no-prediction behavior. There were no focused test failures.
- After source: HEAD `3a080f5c32f2b16056fd0a1f7b4f10d75e07cba3` plus the eight scheduler/test paths listed below. Main concurrently committed S1 (`c86e32b7`) and dead readers/unused lookup (`3a080f5c`); this was a shared-worktree focused check, not an isolated whole-tree experiment.
- Five deterministic before/after JSON snapshots are **byte-identical** (`cmp` exit 0). Their values include type, layer IDs, layer count, model time, total time, schedule owner, operator map, original source time, copy identity and predictor layer calls.
- Scoped `git diff --check` passed. A production/unit-test search found no remaining references to `predict_dense_reference`, `first_dense_layer_id`, or `_create_corrected_execution_time_for_metrics`.

## Bounded changes and exact paths

| Path | Change | Preserved behavior / verification |
| --- | --- | --- |
| `frontier/scheduler/utils/dense_metrics.py` | Delete test-only `predict_dense_reference` and its obsolete `first_dense_layer_id` resolver. Use `BaseModelConfig.get_num_moe_layers()` / `num_layers` to detect mixed models and `is_moe_layer()` per actual layer. Declare the existing scalar-or-Stage timing contract. Do not create a mixed Stage copy that is immediately discarded. | Keep per-physical-layer dense prediction, original stage owner, single-layer validation, nonmixed copies and unchanged source snapshots. Five exact numeric comparisons and ownership/reporting tests pass. |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | Delete test-only `_create_corrected_execution_time_for_metrics` and its unused import. Remove two unused arguments from the live prefill reporting wrapper; access the required replica model directly. | Production scheduler construction supplies the model. The live wrapper still delegates to the same stage-aware helper. |
| `frontier/scheduler/utils/decode_collective.py` | Remove discarded elapsed/start arguments at the live reporting call. | Actual elapsed time, stage start time, batch-stage overrides, events and metrics scope are unchanged. |
| `frontier/scheduler/utils/prefill_collective.py` | Remove the same discarded arguments at the live reporting call. | Full-stage attention prediction and actual reporting timestamps remain unchanged. |
| `tests/unit/test_execution_time_metrics_ownership.py` | Migrate singleton/unexpected-multilayer tests to the live prefill helper; supply required count/layer methods. Add four nonmixed-copy cases. | Preserve entity/layer identity, operator map, exact predictor-call and rejection assertions; verify no prediction and copy isolation in all four added cases. |
| `tests/unit/test_execution_time_op_times.py` | Test the existing live `build_metrics_execution_time` helper directly instead of the unused scheduler wrapper. | Existing omitted-versus-explicit-zero split-TP assertions are unchanged. |
| `tests/unit/test_stage_reporting_contract.py` | Replace the dense-adapter fixture's list API with the required count API. | All existing per-layer numerical, full-stage scope and owner assertions remain intact. |
| `tests/unit/test_pd_decode_moe_layer_accounting.py` | Update the scheduler double's live-wrapper signature. | Existing decode per-layer accounting and completion assertions remain intact. |

No edits were made to `tests/unit/test_pdaf_prefill_model_time.py`; its existing assertions still verify full-stage prediction is supplied to reporting. Other workers' `ep_wave_metrics.py`, `op_trace_utils.py`, shared records and `test_ep_wave_trace_context.py` changes are excluded from this handoff.

Root cause: obsolete tests retained production interfaces for a first-dense reference and a discarded-argument copy wrapper, even though production already used actual physical layers and the shared timing-copy helpers. The resulting defensive resolver duplicated model policy. Reusing the canonical model methods and testing the live helpers removes that accidental interface without adding a replacement wrapper or policy registry. The codebase-design skill's interface-first testing guidance informed this migration; planning-with-files evidence was confined to this report under the explicit task authorization.

### Deliberately retained

- Scalar versus Stage branching represents two supported timing inputs, not incomplete mocks.
- Scalar/nonmixed metrics copies isolate later source edits; mixed Stage construction finalizes its supplied layers and retains the explicit original stage owner.
- Each actual dense layer receives its own prediction; no first-layer substitution or aggregate scaling is introduced.
- The single-layer extraction guard remains: a predictor returning multiple layers violates the real contract.
- Split-TP omitted versus explicit-zero behavior is meaningful and remains covered by the migrated tests.

## Reproducible verification

Working directory for all commands:

```text
/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn
```

Interpreter: `/data/ycfeng/tmp/quality-review-env/bin/python`, **Python 3.12.3**, dedicated virtual environment; no conda activation. Temporary output and pytest basetemp are under `/data/ycfeng/tmp`. The test command limits BLAS/OpenMP threads and disables pytest cache and bytecode writes.

The command below was run once with `frontier_check_phase=before` before editing and once with `frontier_check_phase=after` after editing (not as a two-pass run against the same source):

```bash
frontier_check_phase=after
set -o pipefail
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH \
  PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 \
  WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python -m pytest \
  tests/unit/test_execution_time_metrics_ownership.py \
  tests/unit/test_execution_time_op_times.py \
  tests/unit/test_stage_reporting_contract.py \
  tests/unit/test_pd_decode_moe_layer_accounting.py \
  tests/unit/test_pdaf_prefill_model_time.py \
  tests/unit/test_prefill_ep_wave_materialization.py \
  tests/unit/test_decode_ep_wave_materialization.py \
  tests/unit/test_metrics_full_stage_scope.py \
  tests/unit/test_stage_finalized_contract.py \
  -q -p no:cacheprovider --tb=short \
  --basetemp=/data/ycfeng/tmp/quality-p2-scheduler-${frontier_check_phase}-20260917 \
  2>&1 | tee /data/ycfeng/tmp/quality-p2-scheduler-${frontier_check_phase}-20260917.log
```

Observed final lines:

```text
110 passed in 7.13s
114 passed in 7.10s
```

Acceptance: all original focused assertions pass, migrated tests retain their meaning, and all four new copy/no-prediction cases pass. Runtime durations above describe pytest execution only and are not a performance measurement.

### Exact before/after timing characterization

| Input | Layer IDs | Model ms before = after | Owner schedule ms | Total seconds before = after | Predicted dense layer IDs |
| --- | --- | --- | --- | --- | --- |
| Scalar | `[7]` | 2 | 13 | 0.015000000000000001 | `[]` |
| Dense Stage | `[7,8,9]` | 10 | 13 | 0.023 | `[]` |
| Zero-MoE Stage | `[7,8,9]` | 10 | 13 | 0.023 | `[]` |
| All-MoE Stage | `[7,8,9]` | 10 | 13 | 0.023 | `[]` |
| Mixed Stage | `[7,8,9]` | 38 | 13 | 0.051000000000000004 | `[7,9]` |

For mixed Stage, operator totals remain `attn_prefill=10 ms` and `mlp_up_proj=28 ms`; predictor-owned `schedule_time=999 ms` is not charged in place of the original 13 ms stage owner. Original model time stays 10 ms. Every result is a distinct metrics object. All numeric differences are exactly zero; this is a deterministic contract characterization, not native-hardware fidelity.

Artifacts:

- `/data/ycfeng/tmp/quality-p2-scheduler-before-values-20260917.json`
- `/data/ycfeng/tmp/quality-p2-scheduler-after-values-20260917.json`

The following reproduces the after artifact; the before characterization used the same five inputs and snapshot fields before source edits:

```bash
set -o pipefail
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. \
  TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true \
  VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY' \
  | tee /data/ycfeng/tmp/quality-p2-scheduler-after-values-20260917.json
import json
from types import SimpleNamespace
from unittest.mock import Mock
from frontier.entities import StageExecutionTime
from frontier.scheduler.utils.dense_metrics import build_prefill_metrics_execution_time
from frontier.types import ClusterType
from tests.unit.test_stage_finalized_contract import _layer

snapshots = {}
for name, is_moe, moe_ids, scalar in [
    ('scalar', True, (8,), True),
    ('dense', False, (), False),
    ('zero_moe', True, (), False),
    ('all_moe', True, tuple(range(10)), False),
    ('mixed', True, (8,), False),
]:
    model = SimpleNamespace(
        is_moe=is_moe, num_layers=10,
        get_moe_layer_ids=lambda ids=moe_ids: ids,
        get_num_moe_layers=lambda ids=moe_ids: len(ids),
        is_moe_layer=lambda layer_id, ids=moe_ids: layer_id in ids,
    )
    layers = tuple(
        _layer(layer_id, is_moe=layer_id in moe_ids,
               op_times={'attn_prefill': attention_ms},
               schedule_time=13.0 if layer_id == 7 else 0.0)
        for layer_id, attention_ms in [(7, 2.0), (8, 3.0), (9, 5.0)]
    )
    source = layers[0] if scalar else StageExecutionTime(layers, stage_execution_time=layers[0])
    def predict(*args, **kwargs):
        layer_id = kwargs['layer_id']
        attention_ms, dense_ms = {7: (2.0, 11.0), 9: (5.0, 17.0)}[layer_id]
        return StageExecutionTime((_layer(
            layer_id, op_times={'attn_prefill': attention_ms, 'mlp_up_proj': dense_ms},
            schedule_time=999.0,
        ),))
    predictor = SimpleNamespace(predict_stage_execution_time=Mock(side_effect=predict))
    result = build_prefill_metrics_execution_time(
        original_execution_time=source, sample_batch=object(), predictor=predictor,
        stage_id=2, cluster_type=ClusterType.PREFILL, model_config=model,
    )
    owner = result.stage_execution_time if isinstance(result, StageExecutionTime) else result
    snapshots[name] = {
        'kind': type(result).__name__, 'num_layers': result.num_layers,
        'layer_ids': list(result.global_layer_ids) if isinstance(result, StageExecutionTime) else [result.global_layer_id],
        'model_ms': result.model_time_ms, 'total_s': result.total_time,
        'schedule_ms': owner.schedule_time, 'ops': dict(result.op_times),
        'source_model_ms': source.model_time_ms, 'copied': result is not source,
        'predicted_layers': [call.kwargs['layer_id'] for call in predictor.predict_stage_execution_time.call_args_list],
    }
print(json.dumps(snapshots, sort_keys=True, indent=2))
PY
cmp /data/ycfeng/tmp/quality-p2-scheduler-before-values-20260917.json \
    /data/ycfeng/tmp/quality-p2-scheduler-after-values-20260917.json
```

## Handoff and verification limits

The bounded implementation and focused verification are complete. No new semantic discrepancy or unresolved design decision was found. No commit was made, as requested. Main owns the next source freeze, eight-case E2E and broader regression; this report does not claim those checks ran on these edits. No additional cleanup or test campaign was started by this sidecar after this handoff.
