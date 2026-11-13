# JSON reuse control plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Complete the control loop; paired savings are small and no production allocation is made. |
| 2026-10-24 | Diagnose inconsistent arm timings with one balanced within-process CPU/GC observation. |
| 2026-10-24 | Bind the existing current-source profile and rank three falsifiable causes. |

Dependency: retained red performance evidence -> source-bound controls ->
{complete-output parity, counts and timing} -> disposition/allocation decision.
This is independent of final frozen-regression auditing, snapshot-copy control
and initialization-local reuse analysis; their files and mechanisms are untouched.

The retained named-query production wave exceeds the26.608ms all-other-costs-zero
average workload allowance. Its cProfile file is
`/data/ycfeng/tmp/d29_named_model_query01/candidate/pricing.prof`, SHA256
`7598e372b22b0147d4e1717f65530c746f9c6cb1eb67bdd199d2c1e15dfb7ca0`.
It records1136 loads/704 dumps; JSON decoder/encoder self time is approximately
25/21ms in a216ms profiled wave. These are instrumented timings, not a new full
CLI result or exclusive wall-time percentages.

1. Registry key decoding is repeated inside the per-field predicate. Decode
   once per candidate key, within the current call. Prediction: fewer than272
   registry-key loads with identical ordered selection/errors and no retained state.
2. Point selection repeatedly encodes the same current context/shared facts
   across four phases. Prepare these two strings once within the point query.
   Prediction: canonical calls fall by160/wave while all comparisons still run.
3. Remaining selector normalization/admission work may dominate after these
   local changes. Quantify the remaining codec calls; do not infer permission to
   cache mutable mappings or current-model results from their repetition.

Run each first-two control independently and their union against one same-source
baseline. Reuse the exact protected body, not a reduced equality oracle. Preserve
all original artifacts. Stop if a correctness predicate fails; a low-gain
correct control is reported without production allocation.

The four invocations have completed their protected checks. Registry loads fall
1136→976; point dumps fall704→544; their union reduces both. The independent
wall timings are inconsistent with simple additive savings. One shared
loaded process alternates all four methods in balanced order, observes enabled
GC and CPU/wall clocks, and retains all24 samples without exclusions. Class
methods are restored between arms; no cross-query result cache is introduced.
All24 samples are complete. Medians90.982/89.274/87.092/88.348ms and GC-sensitive
means do not qualify a meaningful full-wave reduction. Complete-output and
counter checks pass; the original protected error-prefix rule remains exact.
Disposition is NO_PRODUCTION_ALLOCATION, not implementation acceptance. There
is no further active step in this investigation. Snapshot-copy, initialization
reuse and required-shape investigations remain independent parent tasks.
