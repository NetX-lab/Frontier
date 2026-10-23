# D29 Frontier position-routing plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Apply checkpoint corrections: effective111 scope, current acceptance frontier, and final-source verification order. |
| 2026-10-23 | Record D29-SCOPE-107-10000 approval; start three sole-writer implementation streams with explicit quotas. |
| 2026-10-23 | Complete all design reviews and parent107-path/9942-line reconciliation; request model/scope decision before implementation. |
| 2026-10-22 | Remove unsupported exact-NCCL reconstruction prerequisite; require finite explicit model assumptions and subsequent qualification. |
| 2026-10-22 | Require an explicit communication-model decision if measured-plan lookup would replace the retained simulator approach. |
| 2026-10-22 | Assign six review findings and three disjoint mandatory design streams; remove unrelated new-policy scope. |
| 2026-10-22 | Record review and complete-scope dependencies after the first design delivery. |

## Current acceptance route

The three implementation streams have delivered their source work. CORE/registry
and joint producer reviews are closed; PRICING's selected prepare metadata P2 also
closed on1be91bad after parent196PASS and independent No findings. Original scope.json stays intact.
Apply repairs/{registry_scope_amendment,core_review_scope_amendment,
legacy_output_scope_amendment,prepare_metadata_scope_amendment}.json in that order:
111 effective paths, CORE46/4404, native19/1582, PRICING46/3956; source9942 and
global10000 gross ceilings are unchanged. Apply repairs/final_scope_reconciliation.json
last for the adopted actual per-path allowances; no path/owner total changes.
requirements.md records YC's authority.

Dependency: prepare metadata repair -> affected checks and independent re-review
-> final source freeze -> {strict77, full-unit failure/error-set comparison,
full offline/online actual-main, final T2 source binding, actual per-path accounting}
-> integration into feat/step5-preview -> package readiness -> resource/run decision
-> fresh native/profile evidence -> G3 and independent four-metric E2E acceptance.

In parallel now: finish the already-running main04 diagnostic on frozen60ba4c75;
prepare T2 bindings and reconcile task records. main04 began before review closed
and cannot substitute for final CLI acceptance. Its source and artifacts stay
unchanged. Final T2 acceptance must verify the actual final script/helper bytes.

| Actionable subtask | Recorded state / completion condition |
| --- | --- |
| Approval, ownership and baseline | Complete; effective111 under existing budgets. |
| Native19 and T2 engine limits | Source/CPU complete; runtime remains separate. |
| Derived image and source receipts | Published and verified; runtime remains separate. |
| CORE/SOURCE46 | Source review and affected fixture repairs complete. |
| Registry multiple bindings | Source review and mixed-view repair complete. |
| PRICING46 | Source review complete;3879/3956 gross. |
| Complete integration and final freeze | Final code frozen1be91bad; main-branch code/docs integration awaits final gates. |
| Legacy strict77 | Final03 on1be91bad:77/77 identical,902 artifact hashes verified. |
| Full-unit failure/error-set gate | Closed on1be91bad with explicit101PASS environment supplement;267/31 baseline sets retained,no new code failures. |
| Fresh fixture and full CLI | fixture03 complete; final offline and online each require6144/1024,2400s and all retained validation checks. |
| Independent source review | All recorded source findings closed on1be91bad. |
| T2 final-source binding | Parent verifies final1be tree,26 files and63 checksums; physical staging/seal follows CLI completion. |
| Final per-path accounting | Complete:111 allocated/110 changed,9062 source/9076 global; adopted same-owner zero-sum reconciliation. |
| Package/GPU/G3/E2E | Blocked by workflow readiness and separate resource approval; original gates remain. |

D30 remains integrated4ec5b23. repairs/repair_plan.md owns detailed assignments
and verification. Native CPU-gap work remains last. Deferred independent-consumer
and module-split work is indexed in future.md, outside this acceptance route.

## Historical design notes (superseded by the approved model and route above)

Native design corrections and D30 run independently. Confirm which shared
interfaces are necessary for the active Step path; avoid unrelated mode work.
Resolve source-local routing width and physical collective mapping before
calling the scope complete. Include actual-main fixture acceptance. The exact
assignments and findings are maintained in `review.md`. New-policy PD-AF/MTP
and speculative legacy fallback adaptations are outside this Step P/D repair;
existing ratio paths remain unchanged. Reconcile shared source-frame interfaces
and overlapping code hunks before calculating a complete budget.

The pricing draft's measured-plan lookup is a proposed change in numerical
source. Compare it with extending the existing simulator's plan/flow interface;
retain the prior collective_sim decision unless the user explicitly changes it.
The selected communication design must have a complete budget before the joint
scope request. Do not equate the proposed backend class name with unchanged
modeling behavior. See `pricing_design.md` sections 1 and 3.2 for parent evidence.

The communication design must specify a finite model, not require lossless native
per-channel reconstruction before implementation. Preserve all workload/group/
phase facts; explicitly distinguish scheduling/protocol/local-work/overlap
assumptions from observed facts and obtain decisions for material alternatives.
Native per-rank counts, scope/transport and duration comparisons plus E2E gates
qualify the implemented model. Add finer channel/timed-local machinery only
when observed errors justify it. This corrects the design prerequisite; it
does not authorize a numerical assumption or a larger code scope by itself.

Acceptance: exact own-position counts and conservation, mask/physical scope,
typed duration/count/source validation, complete new-policy input coverage,
legacy fidelity, and the parent's full request/native checks. Math-only probes
do not satisfy implementation or runtime acceptance.
