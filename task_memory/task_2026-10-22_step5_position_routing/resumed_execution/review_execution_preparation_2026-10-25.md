# D29 current execution preparation review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | Close both existing independent follow-up reviews on the final Git/script bytes; both report No findings. |
| 2026-10-24 UTC / 2026-10-25 HKT | Retain initial independent reports and task-only repairs; follow-up review remains pending. |

Component: current joint-timing and zero-GPU worker preparation. Initial fixed point e08a133d419fa27008d9ccd5261f05ef065322f1; reviewed head c52eca18ff4f1a80819395f7cd36eab80e428133. Follow-up compares c52eca18 to the current task-only working changes plus the new SDK checker. Production source2ca5e571/1461573d is unchanged. Both reviewers are read-only and do not run setup, pricing, worker, imports or StepMind operations.

## Standards

Reviewer:01a16f98-2468-7c30-8040-3f29d40a8fdb. Initial report:

1. P2: trace header identity is incompletely checked. joint_timing.py160 checks field sets, version and timestamp but not simulation_info evidence, label and source_commit. PASS_COMPLETE_OUTPUT_PARITY therefore does not prove source and cold/warm identity. It violates the complete output/proof/key/order requirement and exact identity check. Compare expected metadata values.
2. P3: unnecessary identity fallbacks remain. joint_setup.py73/80 uses get(cache_model_name,label), although all76 entries in the two frozen bundles have cache_model_name. It violates the Engineering rule against branches/defaults for unreachable states. Require the field and retain missing-field errors.
3. P3: the prohibited filler name data represents serialized model bytes in joint_setup.py74/81. Rename it model_bytes.
4. Baseline judgement only, possible Speculative Generality: unused --profile and --iterations options have no consumers. Remove them so the setup does not imply profiling or configurable iteration support.

Remediation: exact expected simulation_info is now a required validator argument and is shared with the initialization value; three independent corruption controls change evidence, label and source_commit. Required cache names, model_bytes and removed options/imports are implemented. Finalstatic05 passes10 expected rejection controls; all prior7 remain. No production source or pricing changes. Independent follow-up result: **No findings** in documented standards or baseline smells. Reviewer01a16f98-2468-7c30-8040-3f29d40a8fdb confirms the three original documented findings and optional unused-flags observation are resolved. joint_timing.py164 compares full simulation_info; joint_setup.py69/76 requires cache_model_name and70/77 uses model_bytes; unused parameters/imports are removed. worker_entry.py56 verifies both Git assets before PATH/GIT_CONFIG_GLOBAL and the original binding check. The actual configuration trusts exactly the two owned repositories, with no fallback, production-budget increase or host/image global configuration mutation. Git04/static06 receipts match current bytes; old failures remain. The reviewer only reads ASTs/files/hashes and performs no runtime or platform operation.

## Spec

Reviewer:01a16f98-608b-79b2-99b1-23d3b1e223f1. Initial report: No findings on e08a133d…c52eca18. No missing/partial requirement, unrequested behavior, or determinable incorrect implementation/claim in the preparation slice.

The reviewer independently checked25 source hashes,6 inputs,262 fixture metadata,13 archives and6 Python ASTs. The retained complete reference has16lanes,208operators and144compute proofs; pricing, ledger and operator values/proofs match, and retained byte counts/SHA256 match reconstruction. The preparation report limits this to checker preparation. Two new ordinary pricing calls remain unapproved; actual worker/fullCLI/RSS/final qualification remain pending. The reviewer wrote no files and executed no setup/pricing/StepMind.

Independent follow-up result: **No findings** in missing/partial requirements, unrequested behavior or wrong implementation/claim. Reviewer01a16f98-608b-79b2-99b1-23d3b1e223f1 inspects the final c52eca18 working slice including original SDK field/default-mount evidence and the Git assets/configuration. Worker asset verification precedes source binding, and the hashes match the final plan. Exactly the current step5-preview/collective checkout are trusted. Git04 passes the fixed-image real binding calls; static06 preserves complete16lane/208operator/144proof and all10 rejection controls, with no-operation dry-run/bash-n. Original failures and diagnostic03's no-PASS limit are preserved. The reviewer reads only static files; pricing remains unapproved, shape6/6 unchanged and actual worker/full workflow remain unqualified.

## Original SDK object checker

The parent retains worker_spec01 field AttributeError, the exact original checker, and the mutation-tainted worker_spec02 mount comparison. The exact intermediate checker matches its executionSHA e6e7e788. Finalworker_spec03 passes without client/auth/query/submission and verifies the unmodified requested mount list, exact4CPU/0GPU/64000Mi/1replica/H800/image/NFS and only the documented mandatory SDK tenant mount. The original SDK adds that mount in ResourceConfig.__iter__; no task reads/writes it, and no SDK override or API wrapper is used.

Evidence: evidence/execution_preparation_2026-10-25/{worker_spec01,worker_spec02,worker_spec03,preparation_static04,preparation_static05} receipts/stdout/stderr and final checker/scripts. These are preparation checks only. Both axes are now closed on the final prepared bytes. Commit the task-only result before the single authorized readiness submission; keep records HEAD stable during the actual worker.

Final review totals: Standards0 documented findings/0 baseline smells; Spec0 findings. Actual worker, current pricing, fullCLI and full-workload RSS are outside this preparation review.
