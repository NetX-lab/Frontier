# V4 native dispatch evidence plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Established bounded implementation, verification, freeze and root-only submission sequence. |
| 2026-10-22 | Added root-owned reviewed balanced-routing binding before freeze; deferred CPU overhead. |

Dependency graph: source/ownership design -> implementation + CPU controls -> independent review + root affected checks + reviewed balanced-routing binding -> all-writers-stop freeze -> fresh package/image/pin checks -> root quota/submission -> full native acceptance.

1. Inspect the actual existing observer, frozen checker and package, choose the smallest complete host association and bounded evidence representation. Record file ownership and size estimate before editing.
2. Implement captured classifier/DP evidence and source-backed verification without changing native behavior. Test missing/incorrect fields and successful-forward/replay reconciliation, including observer failure isolation.
3. Root verifies the changed slice and obtains independent review. Historical R2/native05 fixtures stay unqualified where inputs are absent. Stop writing before successor freeze.
   Root also incorporates reviewed balanced runtime binding before freeze; this agent does not implement routing or equate uniform_random with balanced.
4. Package a new generation through the existing builder. Validate actual module origins, dependencies, all deterministic predicates and locked image startup without CUDA execution.
5. Root submits one approved32H800 run only after fresh quota/integrity checks. Native32 observer failures are soft gates. Collect actual business/rank/lifecycle/terminal evidence and analyze without relaxing assertions.

Independent work: V4 prefill capacity correction/supplement and the specifically authorized retained-evidence GLM hardcoding investigation. Neither blocks this CPU implementation.

CPU-overhead analysis, measurement and suites are deferred until the final correction
phase, after workflow and GPU alignment including op gap. Existing dispatch CPU
correctness checks are not CPU-overhead measurements and remain in scope.
