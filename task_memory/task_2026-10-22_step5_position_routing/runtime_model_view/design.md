# Runtime model-view investigation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Complete the two-arm control and eleven contract groups; hold implementation for the unrelated-model error-policy decision. |
| 2026-10-24 | Primary resumes the cancelled design task and selects one bounded registry-view control. |

## Scope and evidence

Parent requirements and the existing repair authority remain authoritative.
The read-only remaining-runtime design worker was cancelled without delivering
a proposal. Primary resumes the investigation; no production allocation exists.
Source is the accepted key union `46fe83ac` / `30fb6a30`.

`moe_routing_workload.py:194` requests a whole family view on every lane, then
lines198–204 use only models named `moe_grouped_gemm`. The view in
`prediction_model_registry.py:515–554` derives contracts for every legacy name.
Its other callers publish bundles or expose complete model views and still need
that behavior. This is a query-interface mismatch, not a reason to cache views
or selected estimators.

Existing 62,567-byte `key-a/warm.prof` and `key-b/warm.prof` were read under
60 seconds / 30G, with no model/CSV/trace scan or new timing run. They record:

| Profile | Whole pricing, instrumented ms | View calls / cumulative ms | Contract resolutions / cumulative ms |
| --- | --- | --- | --- |
| key-a | 240.880 | 16 / 69.589 | 384 / 60.682 |
| key-b | 236.587 | 16 / 69.197 | 384 / 60.254 |

These nested cumulative times must not be added. They are old diagnostic
profiles, not current unprofiled timings or a promised speedup. The raw paths
are `/data/ycfeng/tmp/d29_immutable_plan_preparation/key-{a,b}/warm.prof`;
the current functions preserve the implicated call chain.

The proposed control adds an optional `model_name` query constraint to the
existing `_models_view_for_family` interface and requests `moe_grouped_gemm`
from the actual point selector. Unfiltered callers retain the existing full
view. All requested native candidates, groups and their current estimators
remain visible; no numerical result, whole proof or model view is retained.

## Checks and decision points

1. Reuse the accepted real snapshot/backend/cached-manager 208-event wave.
   Both arms use current key source; only two in-memory method bodies differ.
   Run each arm within60 seconds/30G, sequentially, with three warm samples.
2. Check the unfiltered view and filtered selected objects against the original
   view before pricing. Keep the accepted complete output/error/snapshot body
   unchanged. Require288 numerical lookups,144 fresh proofs,64 collective
   prices,16 actual points and the complete existing payload/ledger/JSONL bytes.
3. Measure removed unrelated contract derivations, not just elapsed time.
   Separately inspect supported estimator/row/identity/config/error behavior.
   In particular, skipping incidental checks of unrelated models may change
   error timing; do not treat normal-wave parity as proof this is acceptable.
4. Allocate nothing until the measured gain and supported negative cases settle
   the interface. A required behavioral change goes to YC before implementation.

Tentative production paths are `prediction_model_registry.py` and
`moe_routing_workload.py`, with existing binding/runtime tests and any directly
affected stubs. The measured scratch method diff is14 gross / +4 retained
(registry12, point caller2). This supersedes the initial40–70 source estimate.
Propose a100-gross ceiling, source20/tests80, entirely within CORE's231 lines
of retained headroom; PRICING has2 and would not fund this change. This is not
an allocation. Actual path caps and complete mixed-view/selected-error tests
are required after the behavior decision.

## Measured result and open decision

Warm pricing median is0.122211→0.089447 seconds (three samples per arm,
−26.81%). The same38-entry unfiltered view yields the same two selected GEMM
objects. Contract derivations fall384→0; numerical lookups288, fresh proofs144,
collective prices64 and actual points16 stay unchanged. Complete208-event
payload,15 complete JSONLs and ledger are byte-identical. This is a control,
not production implementation or full-workflow acceptance.

Eleven case groups preserve the selected model's collision, row corruption,
empty-row failures and live estimator/row/identity/family behavior. One group
demonstrates the real difference: registering a conflicting native artifact for
`moe_gating_linear` makes the existing full view and old GEMM point query raise;
the narrowed GEMM query succeeds. The unfiltered view still raises. The generic
registry's flattened-view errors are not necessarily the same as the errors
from a context-qualified numerical lookup; do not promise that all removed
unrelated-view errors will be raised by that later lookup.

YC decision required: permit the GEMM point selector to validate only its named
model, keeping initialization/public unfiltered views complete, or retain the
existing every-point whole-view error behavior. Current-model checks and live
selection remain required in either case. Known trainer writers register by
the logical cache name; publication rejects name divergence. The real manager
control asserts that invariant and retains the selector's cache-name guard.
No new writer invariant or error policy has yet been approved.

Evidence and remaining tests are in
`test_report_2026-10-24_runtime_model_view.md` and `evidence/verification.json`.

This control alone cannot establish the full2400s workflow. The accepted
FP8 emission sample still exceeds the26.608ms average whole-wave allowance;
output storage and the remaining execution costs are separate open work.
