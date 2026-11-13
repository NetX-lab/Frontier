# Named-model query independent reviews

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record the two independent reviews of the committed named-query component. |

Component: D29-NAMED-MODEL-QUERY. Fixed point
`617d832bed8f52d6b240e4e33023ed3e5d06dacd`; reviewed HEAD
`3d75c786781802ad2753cbdd81d989c9b694660c`; code `1737ce9c`.
The captured command is `git diff 617d832b...3d75c786`; diff, commit list and
changed-file list are retained under `/data/ycfeng/tmp/d29_named_query_parent01/`.
The two reviews ran independently as capacity became available under the
three-agent limit. Neither reviewer saw the other report.

## Standards

Reviewer/task: `ses_e939c2ecfffekQW62zyM7DVG2a` /
`D29-NAMED-QUERY-STANDARDS`. **No findings.**

Inspected the four complete code diffs, surrounding code/callers, repository and
global instructions, `module-size-exceptions.md`, requirements, plan, amendment,
design/progress, production test report and `verification.json`.

- `prediction_model_registry.py:515–558`: no documented-standard violation;
  naming and module ownership follow `AGENTS.md:791–800`. The three short filters
  apply to different registry representations; extraction would not clearly
  improve maintainability. The private keyword has a present caller, so neither
  Duplicated Code nor Speculative Generality is actionable.
- `moe_routing_workload.py:194–205`: the registry owns selection and the caller
  declares the required name. The existing cache-name predicate remains. No
  Feature Envy, second query implementation or classification system was added.
- `test_prediction_measurement_bindings.py:300–348`: existing fixtures and
  parameterization cover mixed entries, typed errors and role exclusion;
  duplication is not actionable.
- `test_moe_routing_runtime.py:151–178`: branches express different injected
  failures, not a production classification scheme. The approved unrelated-model
  error difference is not a defect.

Both source modules, 1161 and 855 lines, remain below the 2000-line soft limit in
`AGENTS.md:785`. Execution evidence was read, not rerun by this reviewer.
The composed audit, frozen scenario validation required by `AGENTS.md:801` and
full CLI remain parent responsibilities. This axis makes no Spec judgment.

## Spec

Reviewer/task: `ses_e939959e0ffeODtASlPcBfp4z0` /
`D29-NAMED-QUERY-SPEC`. **No findings.**

Inspected the supplied diff, four production/test files and callers; parent and
component requirements, plan, progress, design, allocation, test report,
verification/final guards/manifest, pre-change contract cases and wave wiring.

- No missing component requirement was found. Parent acceptance, composed audit,
  integration, frozen regressions and full workflow are explicitly pending in
  `runtime_model_view/requirements.md:41–43` and `plan.md:26–31`.
- No scope creep: parent `requirements.md:338–362` approves the unrelated
  flattened-view error timing change. Four paths match the allocation, with
  recorded gross usage 14/3/49/30 = 96. No donor code, public API/config/module,
  native key or new cache changed.
- `prediction_model_registry.py:523–557` filters registered names before costly
  work while preserving selected native/typed checks and default unfiltered
  behavior. `moe_routing_workload.py:194–205` requests grouped GEMM and retains
  cache-name and point selection. Added tests cover selected collisions, the
  approved unrelated collision difference, typed errors, role exclusion and
  order/object identity.

Read receipts show focused 117 PASS/1 SKIP to 125 PASS/1 SKIP, all 118 retained
statuses unchanged; 208 events and 19 byte comparisons; 288/144/64/16/192 live
calls unchanged and contract derivations 384 to 0. The existing Torch-dependent
governance failure remains. These are supplied execution results, not reviewer
reruns. Full 6144/1024/2400s CLI, native/GPU/G3/E2E remain unqualified.

**Summary:** Standards: 0 findings. Spec: 0 findings. No remediation requested.
