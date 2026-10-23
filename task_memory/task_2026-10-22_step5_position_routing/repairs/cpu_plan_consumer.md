# Explicit CPU composition consumer decision

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record bounded investigation and allocate the existing CPU fixture consumer/trace correction. |

TaskID: D29-CPU-PLAN-CONSUMER. Read-only investigation by
ses_e99d764e3ffei6w2gP6rN3XrxY; root owns this implementation decision.

## Observed gap and existing opt-in

CORE's actual position_collective_costs writer emits schema/evidence/device,
candidate_only/native_admitted and exact count/cost/buffer records; it does not
emit native qualification. The existing backend reads artifact.qualification first,
raising KeyError before CPU evidence can be consumed. A prototype using actual
writer output reached matching keys for dispatch/combine/shared_tp/routed_tp.
This investigation is not implementation or full CLI acceptance; the owner must
retain a reproducible RED/GREEN test using the real writer.

Existing explicit manifest.collective_plan_input_file -> position fixture runner
profiling_data_dir -> device/collective_local.json suffices. No new CLI flag,
backend, registry or CORE runner change is required. Root does not adopt the
investigator's aggregate/per-file budget estimates from other checkout paths;
the independently verified stream budgets remain authoritative.

## Root decision and required behavior

The approved CPU actual-main fixture may consume its recorded synthetic
CollectivePlanCost composition through the existing backend. This does not
authorize native measured full-plan lookup: native costs still come from the
finite fit/HTSIM path, ignoring a supplied counts entry's cost field.

Require explicit matching CPU provenance on artifact AND plan.shared, correct
schema/device, native_admitted=False, candidate_only=True, exact key/context/buffer
and cost/count identity. Reject CPU/native combinations in both directions and
any native-labelled receipt. Return the original explicit CPU evidence without
manufacturing qualification. Preserve the existing composition invariant checks.

Trace numerical_source must also identify CPU synthesis; it must not describe
synthetic CPU terms as measured local service. Native trace labels remain unchanged.
Required checks include four real-writer positive phases, missing/mismatched
identity and admission negatives, native-label negatives and a native cost-injection
test proving the native path does not use a full-duration lookup.

## Allocation and dependencies

PRICING is sole writer for these four already-approved paths:
- frontier/cc_backend/backends/collective_sim_cc_backend.py
- tests/unit/test_collective_sim_cc_backend.py
- frontier/metrics/op_trace_utils.py
- tests/unit/test_op_trace_utils.py

Reserve at most120 additional gross lines within the unchanged3956 stream ceiling.
At sourcef41d1ee9 the independently verified original-baseline total is3305, leaving
651 for this work plus ongoing native compute-context producer closure. Report exact
combined scope if the work cannot fit; no implicit borrowing. Per-path design
allowances may be redistributed within this root allocation, not added to it.

This CPU consumer is independent of the native observed-point producer and the
multiple-native-binding registry repair. Those remain required for their respective
paths. Full fixture/CLI, independent review and native/GPU acceptance remain open.
