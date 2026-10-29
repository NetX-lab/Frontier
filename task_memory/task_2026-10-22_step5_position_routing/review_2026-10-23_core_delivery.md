# CORE delivery review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record independent44-path review and assign three P2 repairs to the original owner. |

TaskID: D29-CORE-DELIVERY-REVIEW. Reviewer:
ses_e9abd8364ffee5sQKdpiQh3yH4. Frozen COREd96b3279,
verify-step5-core-delivery-01/core_parent/core_full.diff. Reviewer inspected all44
owned paths, deleted hunks, related callers/tests, requirements/design/repair plan
and parent143 PASS/4 SKIP evidence. No tests, platform operations or writes by reviewer.

## Findings

1. P2: moe_impl.py306–308 verifies the class/public selection and bound
   _select_experts, but published BaseRouter._select_experts delegates to the
   simulator's _compute_routing. A foreign bound replacement of that actual
   implementation still passes. Add its actual callable to existing verification;
   use the real inheritance/delegation pattern in the negative test. The earlier
   _select_experts replacement finding is fixed, but provenance remains incomplete.
2. P2: moe_routing_runtime.py91–94,129–131 omits routing_input from required
   position columns and conditionally skips source validation when absent. Automatic
   preflight/training can accept the row before later point recovery needs it.
   Require the supplied descriptor and run existing source validation unconditionally;
   add real automatic admission/trainer negatives.
3. P2: run_step5_measured_main.py234–248 counts zero expert assignments as empty,
   rather than zero-live source ownership. Missing compute_proofs defaults to an
   empty mapping, so deleting dummy source gate/router/prepare proofs while retaining
   one empty-expert proof passes. Require applicable operator/proof correspondence
   and positive duration/count for the actual zero-live source owner; test deletion.

## Disposition and limits

All three are assigned to the original CORE owner in an isolated repair worktree,
under repairs/core_review_scope_amendment.json. Registry's four paths remain owned
by its independent repair stream. No source/GPU/native/E2E acceptance is claimed.
Previously disclosed registry/CPU backend/native producer seams are separate
integration dependencies, not newly discovered CORE regressions. Root's size-note
update remains pending. Actual CLI seam03 was not polled or accepted by this review.
