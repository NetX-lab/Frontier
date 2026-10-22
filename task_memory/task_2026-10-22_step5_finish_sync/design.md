# D30 finish synchronization design

## Modification History
| Date | Change |
| --- | --- |
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

Four source/test files: existing V1 scheduler, existing R2 observation
module, one focused test module (scheduler boundaries plus observer fault
injection), and the existing R2 source-inventory expectation. Target below 500 changed lines; scheduler is 1801 lines, R2 module
1219, both below the 2000-line limit. No module split is needed. Parent-owned
staging derives pins from the existing source list, so no new staging owner is
needed. Focused tests land and fail against the old scheduler before the fix.
Then verify transparent observer calls/errors and run existing legacy dummy/DP1
regressions. Live native reset/step trajectories remain pending.
