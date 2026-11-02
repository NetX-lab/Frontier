# Native06 prelaunch source review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Root archived the independent reviewer's final native builder and host-only checker correction review. |

**Task ID:** W128-V4-NATIVE06-PRELAUNCH-SOURCE-REVIEW

**Reviewer:** read-only reviewer, session `ses_e9b6e7bcfffevLPIR9BYvjfRmy`

**Verdict:** APPROVE — source only; not launch admission.

## Scope and findings

Inspected current requirements/plan/progress, definitive receipts, preserved builder baseline, final builder in `v4-balanced-native-run`, R3, startup/bootstrap, submitter, host manifest writer and frozen-predicate tests. No actionable source findings.

Builder references mean `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py` at e9e79d4.

- Resources and preservation (`44–54,74–75,220–222`): only generation06, existing package refused, payload3600/startup1800/replay1800/readiness32GPUh. Inherited eight-GPU workers and four replicas match approved32H800. Native/C26 staging remains unchanged.
- Actual binding (`99–124`):74 native sources, committed harness/strategy, complete hash-bound Frontier input and actual immutable commit supplied to R3. Schema-origin and effective-DP P2s remain CLOSED.
- Host-only removal (`123–125,182`): no worker import dependency removed. R3 path/hash stays in the case manifest; R3 `218–221` rejects changed frozen R2 bytes. Bootstrap still verifies every remaining tool. The mount receipt records16 original tools passing and both host-only paths raising FileNotFoundError.
- Frozen authority: replacement boundaries remain unchanged; `tests/unit/test_native_v2_serving_inventory.py:493–501` retains six non-graph predicate AST-equality checks. Freeze still requires stopped writers, review and all computable checks passing.

## Acceptance and limits

Receipts support merged133PASS, Frontier source unchanged from39b2bf8, actual A432+B316+C202=950 within952, dispatch491. Reviewer did not independently run tests/Git/Docker/platform or inspect generated packages. Root saved this report because reviewer tools were read-only.

At review handoff, root still owned final clean strict77, stopped-writer/R3+R2 freeze, generated package/seal checks, locked-image complete tool/source/binary and actual-origin checks, submission and runtime/business evidence. Host checker hashes must match the freeze receipt before invocation. Native05 remains historical UNVERIFIED.
