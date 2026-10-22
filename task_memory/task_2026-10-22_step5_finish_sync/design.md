# D30 finish synchronization design

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Extend approved D30-SCOPE-7-750 design with diagnostic consumer and real DES fixture lifecycle. |
| 2026-10-22 | Record bounded design before source/test edits. |

## Binding and ownership

The registered `STEP5_ATTENTION_FAMILY.family_id` is the classification key,
not a model name. `ModelConfig.get_layer_attention_specs()` already supplies
this identity; the predictor uses it for Step execution and forward-start
eligibility (`base_execution_time_predictor.py:39-89`). The existing V1 engine
`_runs_dp_dummy_passes` owns MoE attention-DP waves on the engine-queue roles.
Store the native Step interval/first-step pair in a small family-keyed policy
table beside the existing legacy constant. Bind once in that scheduler; do not
add a CLI flag, another scheduler, a topology validator or an alternative config
path. The next supported native family needs one policy entry, not new caller
branches. Leave the legacy policy `(32, false)` unchanged.

Actual supported Step DP>1 execution is paired P/D with V1/PP1, prefix/MTP off,
MoE TP1 and EP=attention TP×DP. Existing owners enforce that scope; the policy
also applies to its structural CPU replay. Step co-location DP1 does not run a
DP wave and is unchanged. Unsupported Step layouts gain no new admission.
The fidelity gain is the native finish boundary, including the idle step1
case, while request queues, held outputs and peer participation keep one owner.

## Native scalar observation

Extend the existing R2 observer source list/install path to pinned engine core.
Wrap `_has_global_unfinished_reqs` before/after for dp rank, wave, step counter,
configured interval, pending pause, local unfinished and native return. Label
the sync predicate as source-derived, not a measured collective timestamp.
Observe `_process_input_queue` entry to record the actual post-wave reset before
the native input wait. Never call the native scheduler again to obtain state.
Reuse R2 safe callbacks, source hashing, bounded JSONL and failure disabling.
The native body executes once outside observer callbacks; original errors
propagate. No GPU API, tensor copy, fence or additional collective is needed.

## File budget and verification

Initial candidate used four source/test files: existing V1 scheduler, existing R2 observation
module, one focused test module (scheduler boundaries plus observer fault
injection), and the existing R2 source-inventory expectation. Target below 500 changed lines; scheduler is 1801 lines, R2 module
1219, both below the 2000-line limit. No module split is needed. Parent-owned
staging derives pins from the existing source list, so no new staging owner is
needed. Focused tests land and fail against the old scheduler before the fix.
Then verify transparent observer calls/errors and run existing legacy dummy/DP1
regressions. Live native reset/step trajectories remain pending.

## Approved consumer and fixture completion

YC's D30-SCOPE-7-750 receipt at b8be287 supersedes the initial numerical limit:
exactly the seven paths in `requirements.md`, at most750 total insertions and
deletions relative to b8be287, including cf1a53f. D30 does not depend on D29.

The two counter event names share one `DP_COUNTER_EVENTS` classification, which
is part of existing `METADATA_EVENTS`. The loader admits them immediately after
observer/native error handling, before normalization or request/engine/TP checks.
Instrumentation details retain each untouched raw object and JSONL file/line;
the loader returns no request/transfer record for either counter. No requirement,
qualification or native PASS predicate uses these rows. Regression feeds actual
instrumented emitter JSONL into both `load` and `judge` with valid request rows
and metadata alone; original unknown/observer error rejection remains unchanged.

The two existing PD test modules share a test-only `_PDExecution` queue. It uses
the production `(time, id, event_type)` priority and dispatches every returned
event. Its only pauses are observed batch/output/READ boundaries; remaining
events stay queued for resume/drain. This drives stage arrivals, schedules,
all92 layers, real request-free dummy stage ends and consensus peer releases.
It records batch token/context snapshots without changing runtime state and
asserts drained lane wave/counter/dummy state. Existing retry faults and READ
cost fixtures remain test inputs, not simulated timing measurements.

Fixture `_StageCosts` uses existing `ExecutionTime`, `StageExecutionTime`,
structured communication timing, registered per-layer attention binding and
existing CPU construction values. Fixed1ms attention/FFN/dispatch/combine inputs
are explicitly test-only and independent of measured NIXL READ cost. No actual
predictor/model/routing/profile file is modified; every EP wave still materializes
through the existing workload owner. The parent READ102203904/1020-block fix is
unchanged. Native call/return/error transparency and pinned source remain intact.
