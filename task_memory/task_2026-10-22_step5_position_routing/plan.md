# D29 Frontier position-routing plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Resume in current session on frozene41abc07;run fullCLI,requiredregressions and isolated T2 rebinding concurrently. |
| 2026-10-23 | Complete the active CLI repair and consolidate reviewed code/records for YC's session handoff; retain fresh workflow gates as pending. |
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
next,then cli_placement_scope_amendment.json for the two-file CORE repair. No
path/owner total changes; the latter transfers40 unused CORE allowance and limits
the new repair to65 gross.
requirements.md records YC's authority.

Completed handoff boundary: CLI placement repair -> parent real-constructor checks
and independent re-review -> feature-branch code/record consolidation. YC now
defers handoff and resumes here. Code merge4ed67d78 and T2 records merge
9c77b559 are delivery checkpoints,not full workflow admission.

Current dependency: freezee41abc07 (complete) -> {required regression/full
actual-main checks,final T2 source binding,updated actual per-path accounting}
-> package readiness -> resource/run decision
-> fresh native/profile evidence -> G3 and independent four-metric E2E acceptance.

The earlier main04 attempt on60ba is diagnostic only. Main05 on1be supplies the
current placement failure; its raw command/log/execution receipts stay preserved.
T2 has a verified1be binding and its records are now on the feature branch.
Final target staging/seal and rebinding to the later accepted candidate remain.

| Actionable subtask | Recorded state / completion condition |
| --- | --- |
| Approval, ownership and baseline | Complete; effective111 under existing budgets. |
| Native19 and T2 engine limits | Source/CPU complete; runtime remains separate. |
| Derived image and source receipts | Published and verified; runtime remains separate. |
| CORE/SOURCE46 | Source review and CLI repair closed;parent57PASS/6unchangedSKIP,No findings;3873/4404 gross. |
| Registry multiple bindings | Source review and mixed-view repair complete. |
| PRICING46 | Source review complete;3879/3956 gross. |
| Complete integration and final freeze | Complete:e41abc07 frozen clean in verify-step5-d29-integrated-05,gitlink3dac53c. |
| Legacy strict77 | Final03 on1be91bad:77/77 identical,902 artifact hashes verified. |
| Full-unit failure/error-set gate | Closed on1be91bad with explicit101PASS environment supplement;267/31 baseline sets retained,no new code failures. |
| Fresh fixture and full CLI | main05 failed;bounded repair accepted;main06 now running on frozene41,modeall,6144/1024,2400s per mode. |
| Independent source review | Earlier findings closed on1be91bad;CLI40-gross repair closed No findings at75ad26f9. |
| T2 final-source binding | 1be packet preserved;owner prepares e41/integrated05 rebind separately;physical staging/seal follows active CPU gates. |
| Final per-path accounting | Delivery remeasured:111 allocated/110 changed,9102 source/9116 global,zero per-path violations. |
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
