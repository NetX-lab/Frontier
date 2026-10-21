# Actual04 core blocker: measured Step root missing from metrics projection

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record paired actual subprocess failure and minimal source request. |

## Observed actual failure

Tested core: `c9e9388f117c625c6d546d1d083f8aaf3f824f51`, unchanged.
Both actual `python -B -m frontier.main` processes exit1 without timeout:
- Offline:1032.6621223730035 seconds.
- Online:1032.4319722638465 seconds.

The exact stack is `DenseLayerCompleteEvent.handle_event -> complete_dense_layer
-> handle_prefill_sync_collective:250 -> MetricsStore.on_replica_stage_schedule
-> _build_frontier_stage_batch_component_ledger:4504`:

```text
if attributes[name] in StageExecutionTime.stage_owned_fields:
KeyError: 'step5_layer_e2e'
```

Each scheduler log contains one useful P batch `[6144]`, no D decisions, and
nine P records. Last recorded simulation times are85.20541516799993 offline and
85.45541516799992 online. These are scheduler observations, not final E2E metrics.
The persistent event trace, request/system metrics, completed transfer ledger,
and completed stage ledger do not exist because the exception precedes terminal
write. READ/readiness/source release and1024 useful D forwards are UNVERIFIED.

The stack is the final P-stage reporting path; the model has92 layers with final
dense layer91. The unflushed event payload itself is unavailable, so the exact
layer91 attribution is source-based inference, not a saved event observation.

## Minimal executable CPU reproduction

Run from this worktree with the existing environment; no actual-main process,
new fit, cost adjustment or simulator replacement is required:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.entities.time_components import AttentionOperatorTimes, OperatorCost
from frontier.metrics.metrics_store import MetricsStore
from frontier.types import MeasurementType
from tests.unit.test_execution_time_op_times import _base_execution_kwargs

kwargs = {name: False if isinstance(value, bool) else 0.
          for name, value in _base_execution_kwargs().items()}
kwargs.update(num_layers_per_pipeline_stage=1, global_layer_id=0,
              attention_family_id="step5_attention", attention_variant_id="swa",
              attention_operator_times=AttentionOperatorTimes(
                  {"step5_layer_e2e": .6244}, costs={"step5_layer_e2e": OperatorCost(
                      .6244, 23., MeasurementType.KERNEL_ONLY,
                      "cpu_fixture_duration", "cpu_fixture_count", "cpu_fixture", True)}))
stage = StageExecutionTime([ExecutionTime(**kwargs)])
object.__new__(MetricsStore)._build_frontier_stage_batch_component_ledger(stage)
PY
```

Expected and observed: `KeyError: 'step5_layer_e2e'` at metrics_store.py:4504.
The root.6244ms/count23 matches the synthetic P6144 fixture; unrelated components
are zero to isolate the defect. Layer and stage model totals both.6244ms, while
`canonical_operator_execution_time_attrs().get("step5_layer_e2e")` isNone.
Complete actual reproduction log:`/data/ycfeng/tmp/d23_main_ledger_repro.log`.

## Smallest source request to parent

Fix the existing metrics projection for supported scalar-free measured attention
roots, not the input schema or first-call agreement:
1. `frontier/metrics/metrics_store.py:4502-4505`: do not assume every canonical
   op_time has a legacy scalar attribute. Preserve stage-only scalar ownership
   while retaining the inclusive measured root exactly once in the component sum.
2. `frontier/attention/trace_mapping.py:155-220` currently rejects the same root
   after the scalar lookup is passed. The Step family registers root
   `step5_layer_e2e` with predictor_target=True, e2e_trace_target=False and no
   scalar at `frontier/attention/families.py:639-642`. Its root-only cost cannot be
   converted to nine diagnostic child costs. Reuse the existing structural
   cost/projection seam and preserve truthful root/count/evidence ownership.
3. Check the analogous operation metrics iterator, stage-owner projection and
   actual ledger sum: ledger sum must equal stage total, root charged once,
   real typed metadata and per-replay count retained. Merely changing the lookup
   to `.get()` is insufficient: the direct trace call already reproduces
   `ValueError: Unexpected attention timing for inclusive family step5_attention:
   ['step5_layer_e2e']`.

No core edits are made in this worktree. Do not disable the ledger, split the
root into fake child costs or silently drop it to pass completion.

## Other observed warning and limits

Both logs emit generic `quantization_manager.py:810` warnings for mixed FP8/BF16
tables: `attn_pre_proj`, `attn_post_proj`, `moe_grouped_gemm`, approximation.500.
The typed Step replay test separately observes exact source precision/duration
and count7, but final actual runtime cost publication is blocked. Whether the
generic warning affects a live typed operator remains unresolved; it is not
accepted as scaling evidence or an accuracy result. Parent should inspect it
while fixing core reporting, rather than relabeling mixed source precision.

Full6144/1024 offline AND online criterion is FAILED/BLOCKED, not PASS. Inputs and
all generated costs remain explicitly unadmitted synthetic CPU fixtures.
