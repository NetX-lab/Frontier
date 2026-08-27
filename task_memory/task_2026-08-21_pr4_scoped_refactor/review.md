## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-27 | Added CP-61 for SCOPE-042 combine lookup-order correction, typed fixture migration, and fresh dispatch/combine evidence. |
| 2026-08-27 | Added CP-60 for fresh focused/PR17 verification, corrected static owner audit, and PR20/PR21 ancestry evidence. |
| 2026-08-27 | Added CP-61 for the pre-commit narrow A' merge-readiness checkpoint and independent-review status. |
| 2026-08-27 | Added CP-59: independent formal-path reachability review, active-topology GREEN evidence, and aggregate-heterogeneous residual disposition. |
| 2026-08-27 | Added CP-58 for the SCOPE-041 active-role topology propagation RCA/docs freeze and方案 A handoff. |
| 2026-08-27 | Added CP-57 for fresh role-capability and topology mismatch GREEN evidence and closed SCOPE-040b/SCOPE-041. |
| 2026-08-27 | Added CP-56 for the SCOPE-041 strict descriptor/predictor topology docs freeze. |
| 2026-08-27 | Added CP-55 for the SCOPE-040b PD-shaped role-capability RCA and docs freeze. |
| 2026-08-27 | Added CP-52 for the SCOPE-040 routing-map constructor RCA/docs freeze and deterministic RED probe. |
| 2026-08-27 | Added CP-51 for the SCOPE-039 GREEN repair, including the early dummy/non-dummy DECODE_FFN role guard. |
| 2026-08-27 | Added CP-50 for the SCOPE-039 public dummy attention-only RED evidence. |
| 2026-08-27 | Added CP-49 for the SCOPE-039 disaggregation dummy attention-only RCA/docs freeze. |
| 2026-08-27 | Added CP-47/48 for the completed lane/admission/mixed-layer implementation and shared-collective payload audit. |
| 2026-08-27 | Added CP-46 for the lane-local conservation RCA and docs-first correction gate. |
| 2026-08-27 | Added CP-45 for the post-A' ledger/admission verification audit and descriptor-context topology residual watch. |
| 2026-08-26 | Added CP-44 documenting the narrow A' admission decision, concrete-classification audit, and docs-freeze consistency check. |
| 2026-08-26 | Added CP-43 for the raw/effective width conservation repair and EP>1 admission decision gate. |
| 2026-08-26 | Added CP-42 for the dummy terminal-MTP lane RCA, typed seam repair, and numeric verification. |
| 2026-08-26 | Added CP-41 for SCOPE-032 RED/GREEN verification and the completed PDD attention-only identity repair. |
| 2026-08-26 | Added CP-40 for the EP>1 aggregate/structural-MTP audit and the open PDD attention-only identity seam. |
| 2026-08-26 | Added CP-39 for the predictor global-layer identity propagation audit and RED/GREEN verification. |
| 2026-08-26 | Recorded CP-38 terminal MTP EP>1 hook GREEN evidence and numeric phase/layer verification. |
| 2026-08-26 | Recorded CP-37 terminal MTP overshoot EP>1 RCA, RED evidence, and the hook-based remediation boundary. |
| 2026-08-26 | Reviewed and accepted the descriptor-backed EP trace helper migration at commit `183cfd61`. |
| 2026-08-26 | Recorded the implementation handoff review and the raw-map trace boundary that must close before merge readiness. |
| 2026-08-26 | Recorded the MONOLITHIC initial-decode boundary audit and focused regression decision. |
| 2026-08-25 | Recorded the post-implementation MTP scope audit, trace ownership evidence, and escalation gate. |
| 2026-08-25 | Recorded the docs-freeze review: formal RCA, A' ownership graph, and TDD dependency plan are complete; the dirty Gate 2 probe remains provisional. |
| 2026-08-25 | Reviewed Gate 2 typed-lane integration, zero-routed shuffling admission, and focused/broad verification evidence. |
| 2026-08-22 | Recorded the evidence-backed six-item deferred-WATCH audit and its non-blocking disposition. |
| 2026-08-22 | Recorded the post-publication producer-metadata and MTP interface audits and synchronized both PR descriptions. |
| 2026-08-22 | Archived the two final PR #21 review rounds, the Option-B warning audit, and the concurrent Claude timeout without inferring a verdict. |
| 2026-08-22 | Recorded the normal PR #21 push at `54e123ea`, updated body evidence, and confirmed both PRs remain open and unmerged. |
| 2026-08-22 | Recorded the final PR #21 architecture re-review as CLEAR and the code/spec lane as APPROVE after closing decode memory-accounting WATCH. |
| 2026-08-22 | Recorded the focused repair checkpoint and the fresh post-repair Claude timeout without verdict. |
| 2026-08-22 | Recorded Claude's resumed REQUEST CHANGES verdict and the independent reproduction of both P1 findings. |

## CP-49 - SCOPE-039 dummy attention-only docs freeze

- **Target Component/Phase:** Disaggregation dummy execution owner before the
  RED regression.
- **Reviewer Agent Identity:** `/root` (direct call-chain/source audit).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  the shared `SklearnMoEExecutionTimePredictor` dummy owner,
  `_predict_attention_only_stage_execution_time()`, and the existing
  disaggregation predictor tests.
- **Identified Issues/Anomalies:** The public `include_ffn=False` selector is
  validated and used by the non-dummy branch, but the dummy branch forwards no
  equivalent selector. The disaggregation helper therefore returned positive
  FFN fields (`mlp_norm=10.0 ms`, `mlp_up=10.0 ms`, post-attention `50.0 ms`)
  in a `180.0 ms` attention-only probe.
- **Remediation/Verification Code Actions Taken:** Recorded SCOPE-039's RCA,
  ownership boundary, compatibility default, and RED/GREEN dependency chain in
  the task docs. No production or test source was changed in this checkpoint.
- **Final Status:** **PASS for docs freeze; TDD RED is the next gate.**

## CP-50 - SCOPE-039 public dummy attention-only RED

- **Target Component/Phase:** Public disaggregation dummy path after the
  docs-first contract freeze.
- **Reviewer Agent Identity:** `/root` (fresh focused test run).
- **Inspected Artifacts:** The new parameterized regression in
  `tests/unit/test_sklearn_disaggregation_execution_time_predictor.py` and its
  public `predict_stage_execution_time()` call chain.
- **Identified Issues/Anomalies:** Both shared-domain roles failed the same
  semantic assertion: `get_single_layer_post_attention_time()` returned
  `50.0 ms` for `include_ffn=False` instead of `0.0 ms`. The run completed
  normally with `2 failed, 32 deselected in 2.66s`; no fixture/import error
  occurred.
- **Remediation/Verification Code Actions Taken:** Kept the regression as the
  RED gate and recorded the exact values in the issue, plan, harness, and
  progress records. No production code was changed before this evidence.
- **Final Status:** **PASS for the RED gate; owner handoff is next.**
| 2026-08-22 | Recorded the 20-minute Claude Opus PR #20 review attempt, timeout evidence, and still-open external-review gate. |
| 2026-08-22 | Reviewed the maintainer-selected exact-row default policy, RED/GREEN evidence, and committed PR #20 regression matrix. |
| 2026-08-22 | Added the split-PR remote checkpoint and finite exact-row producer audit for PR #20. |
| 2026-08-21 | Added the committed-HEAD closing checkpoint with fresh matrix, direct-probe, PDD, path, and main-checkout evidence. |
| 2026-08-21 | Added the Wave E direct lookup/PDD checkpoint with reproducible commands, numeric evidence, and residual-data disposition. |
| 2026-08-21 | Reviewed option A EP-domain implementation, per-model CLI resolution, and 50-test regression evidence. |
| 2026-08-21 | Reviewed the MoE routing/load fix and recorded the unresolved EP-domain policy gate. |
| 2026-08-21 | Reviewed the first Wave D sampling-domain endpoint slice and recorded numeric superset evidence. |
| 2026-08-21 | Reviewed Wave C TP consumer migration and recorded the post-commit 235-test regression evidence. |
| 2026-08-21 | Reviewed the Wave C profiling-plan catalog cleanup and recorded the 136-test regression evidence. |
| 2026-08-21 | Reviewed Wave B import-boundary extraction and recorded focused regression evidence. |
| 2026-08-21 | Recorded A1 lookup-contract verification, optional metadata limits, and clean-base failure attribution. |
| 2026-08-21 | Added the initial analysis checkpoint review record. |
| 2026-08-21 | Added the dependency-cost and adapter-necessity checkpoint. |
| 2026-08-21 | Recorded the maintainer-approved operator-query binding name and plan-freeze review. |
| 2026-08-21 | Recorded the function-level source audit and deep-module seam review. |

# Checkpoint Reviews

## CP-59 - SCOPE-041 formal-path reachability and active-topology GREEN

- **Target Component/Phase:** Active-role topology propagation through the
  disaggregation public stage and inherited MoE admission chain.
- **Reviewer Agent Identity:** `/root/behavior_review` (independent call-chain
  review), with `/root` performing the fresh executable probe and source
  re-vet.
- **Inspected Artifacts:** `frontier/simulator.py:150-180`,
  `frontier/config/config.py:4605-4721`,
  `SklearnDisaggregationExecutionTimePredictor.__init__` and routed stage
  branches, `SklearnMoEExecutionTimePredictor.predict_moe_layer_time()`,
  `_get_moe_tokens_input()`, `_get_expert_parallel_communication_time()`,
  and the focused admission/disaggregation tests.
- **Identified Issues/Anomalies:** No supported-runtime defect remains. The
  formal simulator creates one predictor per disaggregation role and binds
  role-local EP/top-k fields. The only residual is a manually reused
  `cluster_type=None` aggregate predictor asked to execute a heterogeneous
  role; downstream helpers have no such context contract and can retain
  representative self-field values.
- **Remediation/Verification Code Actions Taken:** Extended the existing MoE
  API with optional active `ep_size`/`router_topk`, forwarded them from all
  three routed disaggregation branches, and kept `_admit_routed_ep_aggregate()`
  as the sole validator. A real inherited-chain probe recorded two admissions,
  both `EP=2/top-k=2`, despite representative top-k `1`; it stopped at the
  intentional unsupported-operation boundary before model/communication
  lookup. The fresh focused matrix passed `309` tests in `79.09s`.
- **Final Status:** **PASS for the supported formal runtime path.** The
  heterogeneous aggregate case is a documented future interface boundary.
  Final static gates, coherent commit, and PR20/PR21 read-only merge audit
  remain pending; remote PR state is unchanged.

## CP-58 - SCOPE-041 active-role topology propagation RCA/docs freeze

- **Target Component/Phase:** Aggregate disaggregation non-dummy routed
  prediction after the strict descriptor/predictor validator.
- **Reviewer Agent Identity:** `/root` (direct call-chain audit; independent
  implementation review remains pending).
- **Inspected Artifacts:**
  `SklearnDisaggregationExecutionTimePredictor.predict_stage_execution_time()`,
  `SklearnMoEExecutionTimePredictor.predict_moe_layer_time()`,
  `_admit_routed_ep_aggregate()`, all three disaggregation routed-role call
  sites, and the existing aggregate admission tests.
- **Identified Issues/Anomalies:** The first admission uses active
  `DECODE_FFN` `EP=2/router_topk=2`, but the inherited second admission falls
  back to aggregate representative `router_topk=1` and rejects the valid
  descriptor. Existing dummy coverage returns before the inherited second
  admission and therefore does not expose this defect.
- **Remediation/Verification Code Actions Taken:** Recorded the RCA and
  approved方案 A in `requirements.md`, `design.md`, `issues.md`, `plan.md`,
  `harness.md`, and `progress.md`. No production or test source was changed
  in this docs checkpoint.
- **Final Status:** **PASS for docs freeze; TDD RED regression is next.**

## CP-51 - SCOPE-039 dummy attention-only GREEN

- **Target Component/Phase:** Disaggregation dummy owner and the shared
  public attention-only role boundary.
- **Reviewer Agent Identity:** `/root` (fresh TDD RED/GREEN verification;
  independent diff review remains a later gate).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  `tests/unit/test_sklearn_disaggregation_execution_time_predictor.py`, and
  the SCOPE-039 sections of `issues.md`, `plan.md`, `harness.md`, and
  `progress.md`.
- **Identified Issues/Anomalies:** The dummy owner dropped the public
  `include_ffn=False` selector and unconditionally emitted shared-domain FFN
  fields. A full affected matrix also exposed that the existing
  `DECODE_FFN` invalid-role check ran only after the dummy branch.
- **Remediation/Verification Code Actions Taken:** Added one default-compatible
  `include_ffn` parameter at the existing dummy owner seam, forwarded the
  public selector, gated shared-domain FFN/MoE fields, and moved the
  `DECODE_FFN` invalid-role check to the common public boundary. The isolated
  selector regression passed `2` cases; the complete affected matrix passed
  `234` tests in `3.12s`, including the dummy role rejection and existing
  MoE/EP/MTP/layer-identity cases.
- **Final Status:** **PASS for SCOPE-039.** Final static gates, independent
  review, coherent commits, and PR20/PR21 ownership audit remain pending.

## CP-52 - SCOPE-040 routing-map constructor RCA/docs freeze

- **Target Component/Phase:** Aggregate disaggregation predictor construction
  with the public default `cluster_type=None`.
- **Reviewer Agent Identity:** `/root` (direct source audit and controlled
  constructor probe; production repair review remains pending).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  predictor factory/default signatures, and the existing disaggregation
  predictor unit fixtures.
- **Identified Issues/Anomalies:** The constructor initializes three routing
  attributes, then deletes two siblings in each role iteration. A deterministic
  stubbed probe reproduced `AttributeError` on the next iteration at
  `_decode_routing_details`; `DECODE_ATTN` also deletes the same attributes.
- **Remediation/Verification Code Actions Taken:** Recorded the stable
  lifecycle contract in `issues.md`, `design.md`, `plan.md`, and `harness.md`.
  No production or test source was changed in this docs checkpoint.
- **Verification:** The controlled probe produced the exact failure above;
  the next gate is a minimal RED regression followed by removal of destructive
  deletion while retaining `None` for non-materialized roles.
- **Final Status:** **PASS for RCA/docs freeze; implementation pending.**

## CP-53 - SCOPE-040 constructor RED regression

- **Target Component/Phase:** TDD RED gate for aggregate routing-map
  construction with `cluster_type=None`.
- **Reviewer Agent Identity:** `/root` (fresh focused test execution; code
  review remains pending).
- **Inspected Artifacts:** The new constructor regression in
  `tests/unit/test_sklearn_disaggregation_execution_time_predictor.py` and
  the real constructor role loop.
- **Identified Issues/Anomalies:** The test failed at the repeated destructive
  deletion with `AttributeError` before any model, routing, CSV, or backend
  operation. The observed result was `1 failed in 3.5s`.
- **Remediation/Verification Code Actions Taken:** Recorded the exact RED
  command and stack evidence in `issues.md` and `progress.md`; production
  code remains unchanged until the next owner-fix sub-step.
- **Final Status:** **PASS for the RED gate; production repair pending.**

## CP-54 - SCOPE-040 constructor GREEN

- **Target Component/Phase:** Routing-map lifecycle correction in the
  disaggregation predictor constructor.
- **Reviewer Agent Identity:** `/root` (fresh focused verification; an
  independent code review remains a later gate).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  `tests/unit/test_sklearn_disaggregation_execution_time_predictor.py`, and
  all production `routing_details` readers.
- **Identified Issues/Anomalies:** None after the correction. All production
  readers use the stable attributes and treat `None` as missing role data at
  their existing validation boundary.
- **Remediation/Verification Code Actions Taken:** Removed only the six
  destructive `del` statements. The focused constructor matrix passed `1`
  case; the complete disaggregation predictor file passed `38` tests in
  `2.50s`. Aggregate, explicit routed, and `DECODE_ATTN` attribute values
  matched the lifecycle contract.
- **Final Status:** **PASS for SCOPE-040 implementation slice; broader
  focused matrix, static gates, independent review, and PR20/PR21 audit remain
  pending.**

## CP-01 - Analysis baseline and scope reset

- **Target Component/Phase:** New PR initialization and current-main evidence
  inventory.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** New worktree status and log; legacy PR commit
  `37130a80421a82a0f43383cf29f11764eb383bab`; old task requirements/findings/
  issues/design; current `sklearn_execution_time_predictor.py`,
  `shared_prediction_model_manager.py`, `sklearn_moe_execution_time_predictor.py`,
  trainers, operator registry, scheduler, and profiling samplers; focused
  baseline test output.
- **Identified Issues/Anomalies:** Residual direct dictionary lookups; unsafe
  legacy fallback/clamp; runtime imports of profiling code; prefix-based TP
  routing; domain coverage not proven for multi-feature in-flight inputs; old
  branch scope is much larger than the requested PR.
- **Remediation/Verification Code Actions Taken:** No production/test code
  change. Created the scoped task documents, recorded inclusion/exclusion
  boundaries, and ran the focused baseline (`37 passed`).

## CP-01 disposition

The analysis checkpoint is ready for the maintainer's one-at-a-time seam
decision. It is not an implementation approval and does not claim independent
code or architecture review.

## CP-02 - Dependency boundary and adapter reassessment

- **Target Component/Phase:** Profiling/runtime dependency boundary and exact
  operator resolver design.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** Direct-import inventory under `frontier/training`,
  `frontier/execution_time_predictor`, `frontier/scheduler`, and
  `frontier/spec_decode`; line counts and consumers for CPU/PP validation,
  MoE input, MTP `ModelConfig`, and non-KV memory; `operators/spec.py`,
  `operators/families.py`, `attention/profiling_mapping.py`, the old PR commit,
  and the prior task's R-025/R-026 adapter proposal plus registry recheck.
- **Identified Issues/Anomalies:** Training has no direct profiling
  implementation import; pure CSV validators are shared by producers and
  consumers; MTP/memory modules are large exceptions; the existing unified
  registry already contains most proposed adapter metadata. The old adapter
  proposal also depends on a D24/N5 catalog absent from current main; an
  independent adapter would duplicate declarations without a demonstrated
  current-main consumer.
- **Remediation/Verification Code Actions Taken:** No production/test code
  change. Updated `findings.md`, `design.md`, `issues.md`, `plan.md`, and
  `progress.md` to use a staged boundary and `bind_operator_query` over the
  existing registry. The adapter is deferred unless a concrete owner mismatch
  is found.

## Review disposition

The dependency boundary and operator-query seam are now confirmed. The plan is
ready for a final maintainer review of file ownership and acceptance gates.
Implementation remains prohibited until that plan gate is explicitly accepted.

## CP-04 - Function-level ownership and seam-depth review

- **Target Component/Phase:** Final analysis handoff before implementation.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** Current predictor lookup methods, MoE finite and
  on-demand branches, shared-manager and trainer TP methods, attention TP
  policy, profiling linear-op plan, sampling helpers, and the updated
  `findings.md`/`plan.md`/`design.md`.
- **Identified Issues/Anomalies:** The central high-dimensional path checks the
  runtime cache before exact lookup and clamps model output; finite one- and
  two-feature paths bypass it; MoE `_round_to_valid_key` silently caps tokens;
  four TP consumers still use name prefixes; and profiling keeps a literal
  attention list beside the registry.
- **Remediation/Verification Code Actions Taken:** No production/test code
  change. Recorded the exact function ownership, kept CPU/PP and MTP policies
  explicit, narrowed the file map, and applied the deep-module test to require
  a small binding interface with exact membership and collision checks.

## CP-04 disposition

The source audit is complete and the implementation map is bounded. The
maintainer's continuation instruction confirmed the map and acceptance
criteria; the A1 implementation review is recorded in CP-05.

## CP-03 - Operator query binding naming and plan freeze

- **Target Component/Phase:** Registry admission seam and analysis-to-
  implementation handoff.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `requirements.md`, `design.md`, `plan.md`,
  `issues.md`, `harness.md`, `progress.md`; current
  `frontier/operators/binding.py`, `spec.py`, `families.py`; predictor prefix
  branches and existing family resolvers.
- **Identified Issues/Anomalies:** The old `exact resolver` label did not
  describe the domain responsibility. Unqualified `operator_binding` could be
  confused with the existing model-to-family `FamilyBinding`. A separate
  profiling-facing adapter remains unsupported by current-main evidence.
- **Remediation/Verification Code Actions Taken:** Maintainer selected
  `operator_query_binding` / `bind_operator_query`. Updated the task documents
  with the runtime/registry-only responsibility, reuse of
  `frontier/operators/binding.py`, the conditional new-file rule, the staged
  dependency allowlist, and the bounded implementation file map. No production
  or test source was changed.

## CP-05 - A1 lookup contract implementation review

- **Target Component/Phase:** Wave A1 registry admission and high-dimensional
  runtime lookup contract.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `frontier/operators/binding.py`,
  `frontier/operators/__init__.py`,
  `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`,
  `tests/unit/test_operator_registry.py`,
  `tests/unit/test_on_demand_prediction_contract.py`, and all A1 commands in
  `test_report_2026-08-21_a1_lookup_contract.md`.
- **Identified Issues/Anomalies:** The exact lookup/cache precedence and
  invalid-output behavior are corrected. Current-main producers emit feature
  names, model hash, and exact lookup rows, but do not emit bounds, domain,
  relational-constraint, or identity records; those checks therefore remain
  optional and cannot yet prove producer-side domain coverage. The combined
  attention logger assertions fail on clean base because `frontier` logger
  propagation is disabled; direct stdout values remain numerically correct.
- **Remediation/Verification Code Actions Taken:** Added fail-fast schema,
  finite-input, optional-bound/constraint/identity, exact-row, runtime-cache,
  and estimator-output validation; wired exact lookup descriptors; added the
  focused regression test; ran `76` + `49` focused passes, `111` passes with
  the two clean-base logger failures, py-compile, and whitespace checks. No
  logging change was made in A1.

## CP-05 disposition

A1 is ready for its sub-step commit. A2 must add a test-first ordinary finite
resolver and must not promote currently unpublished metadata into a mandatory
producer contract without new evidence and scope review.

## CP-06 - Wave B staged import boundary review

- **Target Component/Phase:** Runtime predictor/training import boundary and
  pure MoE feature ownership.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `frontier/moe_load_imbalance.py`,
  `frontier/profiling/moe/moe_input.py`,
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  `tests/unit/test_profiling_runtime_boundary.py`, the CPU/PP schema and
  validation helpers, and the named scheduler/MTP exception imports.
- **Identified Issues/Anomalies:** The runtime MoE predictor directly imported
  a mixed profiling input file. CPU/PP helpers are side-effect-free and shared
  by producer and consumer; non-KV-memory and MTP structural configuration
  remain larger, explicitly scoped exceptions.
- **Remediation/Verification Code Actions Taken:** Extracted only the pure
  `MoELoadImbalanceInput` feature derivation into the runtime-owned module and
  retained the profiling import as an exact compatibility re-export. Added an
  AST allowlist/no-training-import boundary test. The RED run showed `2 failed,
  1 passed`; GREEN boundary tests passed `3`, runtime/MoE regressions passed
  `66`, profiling compatibility passed `39`, and py-compile/whitespace checks
  passed.

## CP-06 disposition

The staged boundary is implemented within the approved file map. The runtime
path has no direct benchmark/GPU/profiling-CLI dependency; shared CSV helpers
and named exceptions remain documented rather than duplicated. Proceed to the
registry-consumer wave.

## CP-07 - Wave C profiling-plan catalog cleanup review

- **Target Component/Phase:** Wave C registry consumer migration, profiling
  plan slice.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `frontier/profiling/linear_op/profiling_plan.py`,
  `frontier/operators/families.py`, `frontier/model_architectures.py`,
  `tests/unit/test_ffn_memory_operator_families.py`, the targeted architecture
  tests, and commit `809531e8`.
- **Identified Issues/Anomalies:** The profiling plan exported five attention
  lists that duplicated architecture-profile declarations. The pre-existing
  dense output order places the first memory operator before architecture-
  specific replicated attention operators; changing that order would alter
  profiler enablement output. No producer-side runtime-domain metadata was
  added in this slice.
- **Remediation/Verification Code Actions Taken:** Removed the five duplicate
  constants; derived attention names from `linear_attention` and memory names
  from `MEMORY_FAMILY` declaration order; retained the existing MTP registry
  exception. The RED assertion failed before the edit, then the focused test
  passed, the combined Wave C matrix passed `136` tests in `6.68s`, and
  `git diff --check` passed. The slice was committed as `809531e8`.

## CP-07 disposition

The profiling-plan slice meets its local acceptance criteria. Remaining Wave C
TP consumer changes are still uncommitted and require their own regression and
review checkpoint before Wave C is marked complete.

## CP-08 - Wave C TP consumer migration review

- **Target Component/Phase:** Wave C unified-registry TP consumer migration.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `frontier/operators/binding.py`,
  `frontier/execution_time_predictor/attention_tp_policy.py`,
  `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`,
  `frontier/execution_time_predictor/shared_prediction_model_manager.py`,
  `frontier/training/linear_op_trainer.py`,
  `frontier/training/attention_trainer.py`,
  `tests/unit/test_operator_query_tp_consumers.py`, and commit `4c03b243`.
- **Identified Issues/Anomalies:** The old consumers classified names by
  prefix or local literal sets. The existing `add` timing alias maps two memory
  physical operators, so an unscoped direct binding must remain ambiguous while
  TP resolution may admit it only after same-family/same-mode proof. MTP names
  remain outside the ordinary registry by explicit structural design.
- **Remediation/Verification Code Actions Taken:** Added exact registry and
  architecture-profile TP resolution with conflict checks; routed all four
  consumers through it; derived the attention policy set from the architecture
  registry; retained MTP and effective attention-TP semantics. The post-commit
  focused matrix passed `235` tests in `8.50s`; a direct contradictory-profile
  case raised the expected error; `git diff --check` passed.

## CP-08 disposition

Wave C meets its acceptance criteria. Wave D must now audit runtime-domain
coverage and sampling bounds without moving sampling configuration into the
runtime module.

## CP-09 - Wave D sampling-domain endpoint slice

- **Target Component/Phase:** Wave D primitive profiling-domain endpoint
  coverage for Linear, standard Attention, and profiling-owned MoE token grids.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `frontier/profiling/utils/__init__.py`,
  `frontier/profiling/moe/moe_input.py`,
  `tests/unit/test_profiling_sampling_domain.py`, the profiling-plan and
  metadata regression files, and direct helper probes for `runtime_max=10`,
  `1000`, and `5000`.
- **Identified Issues/Anomalies:** The shared token grid stopped at `4992` for
  a `5000` maximum; oversized explicit token extensions were silently ignored;
  sequence and prefill-chunk grids excluded a non-canonical maximum; the
  attention batch grid omitted `129..130`; and the independent MoE config had
  the same token endpoint gap. The existing chunk grid also emitted duplicate
  boundary anchors.
- **Remediation/Verification Code Actions Taken:** Added test-first endpoint,
  deterministic-order, mode-separation, and explicit-bound tests. Appended
  requested positive endpoints, unioned explicit token extensions, and
  deduplicated affected grids without changing runtime ownership or CSV
  schemas. RED reproduced all targeted gaps; GREEN passed `10` focused tests
  and `35` existing profiling regressions. Direct values showed shared
  token/sequence/chunk maxima `5000`, Attention decode batch maximum `130`,
  and MoE config token maximum `5000` for corresponding runtime maxima.

## CP-09 disposition

The primitive endpoint slice meets its local acceptance criteria and is ready
for a sub-step commit. Joint mixed-attention composition, KV interaction, and
MoE routing/expert-load envelope coverage remain open under Wave D; this
checkpoint does not claim full D24-5/D24-8 coverage.

## CP-10 - Wave D mixed-Attention envelope slice

- **Target Component/Phase:** Wave D joint mixed-Attention sampling coverage
  for legacy mixed-prefill, online-grid mixed-prefill, and true-mixed
  prefill+decode generators.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `frontier/profiling/utils/__init__.py`,
  `frontier/profiling/attention/mixed_attention_input.py`,
  `frontier/profiling/attention/true_mixed_batch_input.py`,
  `frontier/profiling/attention/main.py`,
  `tests/unit/test_profiling_sampling_domain.py`, and the prior-task ISS-028,
  D24-7, and D24-8 records.
- **Identified Issues/Anomalies:** Legacy mixed-prefill dropped requested
  `kv_cache_size=992` at `max_seq_len=1000` because its smallest candidate
  sequence was `64`; overlapping candidate ranges emitted `294` rows but only
  `287` unique structural identities. Online explicit lists replaced rather
  than extended configured ranges. True-mixed defaults stopped at the supplied
  lists (`64/128` in the probe), below legal `1000/999` endpoints.
- **Remediation/Verification Code Actions Taken:** Added test-first boundary,
  uniqueness, union, and endpoint checks. The existing sampling helper now
  deduplicates legacy identities, adds a valid per-KV context-boundary shape,
  unions online explicit axes with their configured ranges, and unions
  true-mixed prefill/decode endpoints. Updated the two CLI help strings to
  describe extension semantics. Focused tests passed `14`; the affected
  profiling matrix passed `49`; the supplementary Attention/profiling matrix
  passed `53`; direct probes reported legacy maxima `max_context=1000`,
  `kv_max=992`, online `9` union points, and true-mixed `prefill=5000`,
  `decode_kv=4999` with all rows valid; the cross-wave registry/predictor
  matrix passed `249` in `15.08s`.

## CP-10 disposition

The mixed-Attention slice meets its local acceptance criteria and is ready for
a sub-step commit. It does not claim complete Wave D coverage: MoE routing,
expert-load, and joint TP/EP envelope sampling remain pending.

## CP-11 - Wave D MoE routing/load and EP envelope review

- **Target Component/Phase:** MoE route identity, load-distribution edge cases,
  and the remaining joint TP/EP profiling-domain audit.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `frontier/profiling/moe/moe_wrapper.py`,
  `frontier/profiling/moe/load_distribution.py`,
  `frontier/profiling/moe/moe_input.py`, `frontier/profiling/moe/main.py`,
  `frontier/config/config.py`,
  `tests/unit/test_moe_routing_input_contract.py`, the qwen2 model config and
  canonical CSV, and the two Wave D probe logs under `/data/ycfeng/tmp/`.
- **Identified Issues/Anomalies:** The explicit-count identity mismatch,
  unreachable skewed expert, and high-`top_k` hot-subset collapse were
  reproducible and are fixed. A separate policy mismatch remains: runtime
  accepts all divisors, the helper enumerates `[1,2,4,8]`, the CLI defaults to
  `[1]`, and qwen2 canonical data contains `{1,2}`.
- **Remediation/Verification Code Actions Taken:** Added test-first fail-fast
  count validation (including strict integral-type admission) and corrected
  both distribution formulas. Focused GREEN passed `5`; the affected
  regression matrix passed `85`; numeric invariants
  are recorded in `test_report_2026-08-21_wave_d_moe_routing.md`. No EP policy,
  runtime config, CLI default, or canonical CSV was changed. The legal-EP
  policy is escalated for maintainer selection before the next Wave D edit.

## CP-11 disposition

The routing/load sub-step meets its local acceptance criteria and is ready for
its own commit. Wave D remains blocked at the EP-domain policy choice, not at
the corrected route identity or distribution formulas.

## CP-12 - Option A MoE EP domain implementation

- **Target Component/Phase:** Wave D runtime-legal MoE EP sampling envelope.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `frontier/profiling/moe/moe_input.py`,
  `frontier/profiling/moe/main.py`,
  `tests/unit/test_moe_ep_sampling_domain.py`, the runtime divisibility check
  in `frontier/config/config.py`, and the profiling example dry-run.
- **Identified Issues/Anomalies:** The old helper and CLI did not cover legal
  divisors such as EP=3 for a 60-expert model. Explicit canonical CSVs still
  contain narrower historical domains, so code coverage and measured-data
  coverage must remain distinct.
- **Remediation/Verification Code Actions Taken:** Added one pure divisor
  resolver, derived default local widths from it, changed an omitted CLI EP
  selection to per-model resolution, validated explicit selections, and stored
  the resolved domain in the profiling config artifact. Focused GREEN passed
  `6`; the affected regression matrix passed `50`; direct probes and the
  example dry-run recorded the expected domains without changing canonical
  data.

## CP-12 disposition

Option A is implemented for the profiling plan and CLI selection contract.
The remaining work is offline regeneration of missing CSV rows (outside this
code-only scope) and Wave E direct runtime cases; no further EP policy choice
is open.

## CP-13 - Wave E direct lookup and heterogeneous PDD review

- **Target Component/Phase:** Wave E focused verification for ordinary finite
  lookup, cache precedence, runtime model fallback, and heterogeneous PDD role
  identity.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** `test_report_2026-08-21_wave_e_direct_cases.md`,
  `tests/unit/test_finite_prediction_lookup.py`,
  `tests/unit/test_moe_finite_prediction_lookup.py`,
  `tests/unit/test_on_demand_prediction_contract.py`, the PDD metrics under
  `/data/ycfeng/tmp/pr4_wave_e_pdd_final/`, and the current worktree status.
- **Identified Issues/Anomalies:** The first draft of the reproducibility
  probe used a two-feature query with a one-feature builder and failed during
  probe setup with `ValueError: unexpected features`; this was a report-script
  defect, not a product failure. The aggregate test command also has an
  environment-sensitive collection failure when `PYTHONPATH` is omitted.
  `SCOPE-011` producer-side optional metadata and newly exposed canonical EP
  rows remain open by scope.
- **Remediation/Verification Code Actions Taken:** Corrected the inline probe
  to declare `("batch_size", "num_tokens")` and reran it successfully. The
  direct output showed Linear `4.5`, Attention decode `4.25`, Attention
  prefill `5.5`, MoE grouped-GEMM `6.25`, exact-measured `2.25`, and NaN
  rejection before model calls. The focused matrix passed `111` tests and the
  broader cross-wave matrix passed `306` tests (the earlier narrower matrix
  passed `265`). The PDD case exited `0`, completed
  `2/2` requests, processed `24` tokens, and recorded `2` KV transfers with
  `8,388,608` bytes total; PREFILL and DECODE predictor objects were distinct
  and shared one model manager.

## CP-13 disposition

Wave E is complete for the scoped code and runtime lifecycle. The branch has
no remaining implementation wave in `plan.md`; producer-side optional-domain
metadata, canonical CSV regeneration, and the pre-existing logger capture
repair remain explicitly deferred follow-up work.

## CP-14 - Committed-HEAD closing verification

- **Target Component/Phase:** Final archive verification for the scoped lookup,
  registry, profiling-domain, and sequential-PDD implementation.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** The dedicated worktree at
  `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr4-scoped-lookup-boundary-20260821`,
  the 26-file regression command, the direct probe output, fresh PDD metrics at
  `/data/ycfeng/tmp/pr4_completion_pdd_20260821/`, `git diff origin/main...HEAD`,
  all task Markdown headers, the README/canonical-data path audit, and the
  main checkout status.
- **Identified Issues/Anomalies:** No new product failure appeared. The main
  checkout still has only its pre-existing `?? =10.1`, `tests/scratch/`, and
  `tests/unit/test_moe_ep_non_dummy_matrix.py` entries. The known
  `SCOPE-011`, canonical EP-row, logger-capture, and deferred ownership items
  remain open by scope.
- **Remediation/Verification Code Actions Taken:** Re-ran the complete matrix
  (`306 passed in 19.79s`), the direct probe (`direct_probe PASS`), and the
  sequential PDD case (exit `0`). Parsed fresh metrics showing `2/2`
  completed requests, `24` total tokens, `2` KV transfers,
  `8,388,608` bytes, and `1.3355443200000483 ms` total transfer time. Re-ran
  branch/diff, history, forbidden-path, and main-checkout audits; all passed.

## CP-14 disposition

The committed branch satisfies the scoped implementation and verification
gates. No production or canonical-data edit is required for this archive
refresh; the listed residual items remain explicitly deferred.

## CP-15 - Split PR #20 remote and producer-completeness review

- **Target Component/Phase:** Remote PR #20 lookup/registry base after the
  exact-lookup persistence follow-up commits.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:** Remote PR #20 metadata, commits `653f8526`,
  `0a898729`, and `53499911`; both training entry points; all persistence flag
  call sites; finite/on-demand prediction materialization; the 12-file focused
  matrix; and a direct two-row estimator probe.
- **Identified Issues/Anomalies:** Remote PR state is clean and has no review or
  CI record. The affected product tests pass, while the known caplog issue
  remains. Exact measured metadata survives caches only for call sites that
  opt in; ordinary Linear, standard Attention, and one-feature MoE producers
  still use the false default.
- **Remediation/Verification Code Actions Taken:** Updated the task ledger and
  preserved both remote PRs without merging. No production edit was made
  pending the producer-policy decision. Recorded representative serialized map
  sizes (`156` to `712` bytes for the checked examples) to make the policy
  tradeoff concrete. The broader PR-A matrix passed `286` tests with only the
  two known logger cases deselected. A synthetic PR #20 + PR #21 merge tree was
  constructed successfully as
  `a19525e00323edcdebecc88791c860f65d9764ea`.

## CP-15 disposition

PR #20 remains the active fix target. Resolve `SCOPE-014`, rerun the focused
and PR-A matrices, update the remote PR body with fresh evidence, and then wait
for maintainer review before advancing PR #21.

## CP-16 - Universal exact-row persistence review

- **Target Component/Phase:** PR #20 `SCOPE-014` producer completeness at
  commit `d45d836c`.
- **Reviewer Agent Identity:** `/root` (primary agent; independent approval is
  not claimed).
- **Inspected Artifacts:**
  `SklearnExecutionTimePredictor._train_model`,
  `ExecutionTimePredictionModelManager._train_single_model`,
  `tests/unit/test_on_demand_prediction_contract.py`, the RED/GREEN logs, the
  12-file focused matrix, the 23-file broader matrix, and the committed-HEAD
  direct pickle probe.
- **Identified Issues/Anomalies:** The pre-fix false defaults made persistence
  depend on scattered call-site admission. The RED tests reproduced the gap
  in both entry points. No current call site passes an explicit false value,
  and the audited training keys remain scalar numeric. The same two
  `SCOPE-013` caplog failures remain unrelated and numerically correct.
- **Remediation/Verification Code Actions Taken:** Changed only the two
  defaults to `True`; retained explicit opt-out capability and all existing
  lookup validation. Default cache-round-trip tests passed `2`; focused
  regression produced `135 passed` plus two known caplog failures; committed
  broader regression passed `287` with those two deselected. The direct probe
  returned exact measured `4.0` ahead of stale cache `99.0`, runtime-cache
  `8.25` ahead of the model, and model `7.5` twice with one invocation.

## CP-16 disposition

`SCOPE-014` is resolved and pushed on PR #20 at `d45d836c`. The remote PR
body contains the verification evidence, PR #20 and PR #21 both report
`OPEN/CLEAN/MERGEABLE`, and both remain unmerged for maintainer review.

## CP-17 - Ask Claude review attempt for PR #20

- **Target Component/Phase:** External review of the complete PR #20 range
  `origin/main...d45d836c`.
- **Reviewer Agent Identity:** StepCode Claude Opus 5 with `effort=max`,
  session `bea7b31b-657c-44ff-9373-560e940151c5`. `/root` verified execution
  evidence only; Claude emitted no final reviewer verdict.
- **Inspected Artifacts:** The complete PR diff and latest commits, task
  requirements, `frontier/operators/binding.py`, operator TP declarations,
  attention TP policy and trainer consumers, shared predictor model
  management, MoE load-imbalance extraction, existing test reports, and
  focused runtime probes. The local artifact is
  `.worktrees/pr4-lookup-registry-core-20260822/.omx/artifacts/ask-claude-pr20-review-20260822-122037.md`.
- **Identified Issues/Anomalies:** The configured `1200s` limit expired after
  `55` successful read-only tool calls and before Claude produced a verdict or
  finding list. GNU `timeout` returned `124`; the child received `SIGTERM` and
  exited `143`. A later `401 Unauthorized` came from StepCode's post-session
  feedback collector and was not the cause of the review timeout.
- **Remediation/Verification Code Actions Taken:** Made no source, remote, or
  PR-state change. Preserved the exact prompt, raw CLI output, duration, tool
  counts, and log paths in the required Ask artifact; independently checked
  the StepCode debug/session logs to separate timeout from the feedback
  collector warning.

## CP-17 disposition

The requested invocation ran with the exact backend/model/effort and
20-minute limit, but it did not produce a usable review conclusion. PR #20 and
PR #21 remain unmerged. The external-review gate stays open pending maintainer
selection of a longer rerun or a bounded continuation/synthesis invocation.
Local CLI help confirms that `--resume` is supported; the captured Claude
conversation ID is `40e27923-c316-4f00-806b-b5500569309c`.

## CP-18 - Resumed Claude verdict and independent P1 verification

- **Target Component/Phase:** PR #20 external-review completion and
  maintainer escalation before repair.
- **Reviewer Agent Identity:** StepCode Claude Opus 5 with `effort=max`,
  resumed conversation `40e27923-c316-4f00-806b-b5500569309c`; `/root`
  independently inspected and reproduced both blocking findings.
- **Inspected Artifacts:** Claude resume artifact
  `.worktrees/pr4-lookup-registry-core-20260822/.omx/artifacts/ask-claude-pr20-review-resume-20260822-130144.md`;
  PR #20 commit range `origin/main...d45d836c`; the two MoE training call
  sites; both unified cache entry points; the target-embedded MTP registry and
  its accessor; the three TP-key consumers; their existing unit tests; the
  original and replay probe logs under `/data/ycfeng/tmp`.
- **Identified Issues/Anomalies:** Claude returned `REQUEST CHANGES` with two
  P1 findings. First, the MoE call sites explicitly pass
  `persist_exact_lookup=len(feature_cols) > 1`, so the default-`True` repair
  does not cover one-feature MoE. Second, three TP-key consumers repeat
  `{"mtp_fusion_proj", "lm_head_linear"}` instead of reading the existing
  `get_target_embedded_mtp_linear_ops()` declaration. Four P2 observations
  remain non-blocking and the existing `SCOPE-013` logger/caplog failure
  remains baseline behavior.
- **Remediation/Verification Code Actions Taken:** Kept production and test
  sources unchanged. Replayed a one-feature MoE cache round trip and observed
  measured `4.0`, stale runtime-cache/model value `99.0`, absolute error
  `95.0`, and relative error `2375%` when persistence is disabled; enabling
  persistence restored `{(1.0,): 9.0, (2.0,): 4.0}` and returned `4.0`.
  Extended the MTP registry in-process with
  `mtp_registry_extension_probe`; predictor, shared manager, and trainer all
  rejected it, confirming `3/3` consumers bypass the registry accessor.
  Evidence is recorded in
  `test_report_2026-08-22_pr20_claude_review_verification.md`.

## CP-18 disposition

The external-review gate is complete and PR #20 is not merge-ready. Repair
authorization remains pending for the confirmed one-feature MoE persistence
defect and for the scope of the MTP consumer cleanup. The evidence supports a
focused MTP repair limited to the three TP-classification consumers plus their
existing regression file; the other same-name occurrences have distinct
enumeration, CSV-schema, profiling-kernel, quantization, or runtime-policy
roles and require a separate broader audit before any change. PR #20 and PR
#21 remain unmerged.

## CP-19 - Focused repair implementation review

- **Target Component/Phase:** PR #20 one-feature MoE exact-row persistence and
  target-embedded MTP TP registry consumption.
- **Reviewer Agent Identity:** `/root` (primary agent; independent external
  approval is not claimed).
- **Inspected Artifacts:** Commits `3e9c0374` and `18d1a23e`; both MoE
  producer call sites; the unified `_train_model` and `_train_single_model`
  persistence mechanisms; all three MTP TP-key consumers; the four modified
  regression files; affected and broad pytest logs; direct numeric probe; and
  final worktree/remote branch state.
- **Identified Issues/Anomalies:** The pre-fix four-case run reproduced exactly
  the three expected defects (`3 failed, 1 passed`). No additional product
  failure appeared after the focused changes. The two known `SCOPE-013`
  logger/caplog parameterizations remain excluded from the broad matrix.
- **Remediation/Verification Code Actions Taken:** Removed both MoE
  feature-count persistence overrides, retained the unified explicit opt-out,
  and routed predictor, shared manager, and Linear trainer MTP membership
  through `get_target_embedded_mtp_linear_ops()`. Added producer,
  cache-round-trip, and registry-extension regressions. Fresh verification
  passed four targeted tests in `5.11s`, the affected-file matrix with
  `62 passed`, and the broad matrix with
  `289 passed, 2 deselected in 10.93s`. The direct probe reloaded
  `{(1.0,): 9.0, (2.0,): 4.0}`, returned measured `4.0` ahead of stale
  `99.0` with absolute error `0.0` and `0` model calls, and resolved TP `2`
  for all `3/3` MTP consumers.

## CP-19 disposition

`SCOPE-014` and the focused `SCOPE-016` defect are resolved at pushed PR #20
HEAD `18d1a23e`. The wider MTP unified-family/interface design remains outside
this repair and is recorded as a separate audit in `future.md`.

## CP-20 - Fresh Ask Claude review after repair

- **Target Component/Phase:** External read-only re-review of complete PR #20
  range `origin/main...18d1a23e`.
- **Reviewer Agent Identity:** StepCode Claude Opus 5 with `effort=max`,
  session `703700ab-27ef-433f-9a9d-405a389c8525`; `/root` verified execution
  and repository-state evidence.
- **Inspected Artifacts:** Complete PR commit list and diffstat, both focused
  repair commits, two delegated focused audits, a late rest-of-diff audit
  attempt, selected PR-added unit tests, StepCode session/debug logs, and the
  Ask artifact
  `.worktrees/pr4-lookup-registry-core-20260822/.omx/artifacts/ask-claude-pr20-focused-repair-rereview-20260822-145617.md`.
- **Identified Issues/Anomalies:** The configured `1200s` budget expired before
  the parent Claude process synthesized a final finding list or verdict.
  StepCode recorded `7` tool calls and `1` failure; the failed call was a late
  reviewer subtask. The child exited `143`, and GNU `timeout` returned `124`.
- **Remediation/Verification Code Actions Taken:** Preserved the exact prompt,
  raw CLI output, session metrics, and log paths. Confirmed that the PR #20
  worktree remained clean and that local and remote heads both equal
  `18d1a23e1a04f3dcad2d2d1376451190d2b5300a`. No source, commit, push,
  merge, or PR-state change came from Claude.

## CP-20 disposition

The maintainer-requested fresh 20-minute invocation is complete, while Claude
provided no final `APPROVE`, `REQUEST CHANGES`, or `COMMENT` verdict. The
independent focused-repair evidence remains green. PR #20 and PR #21 stay open
and unmerged; another Claude continuation or longer review requires explicit
maintainer authorization.

## CP-21 - PR #21 architecture re-review

- **Target Component/Phase:** Final architecture review of the stacked PR #21
  repair range `18d1a23e..b8149c32`.
- **Reviewer Agent Identity:** Independent architecture lane
  `/root/pr21_arch_review`.
- **Inspected Artifacts:** PR #21's 11 profiling/test paths, synthetic tree
  `dc8081821c0e36fb007f3c67902a88541189d119`, standard Attention input and
  wrapper allocation paths, mixed planners, MoE confirmation helpers, focused
  regressions, and final static audits.
- **Identified Issues/Anomalies:** The first pass found one WATCH: decode
  memory filtering omitted the current token while wrapper allocation and
  `is_valid()` counted it. No other architecture blocker was found.
- **Remediation/Verification Code Actions Taken:** Added the boundary
  regression, changed `is_under_memory_limit()` to use decode
  `kv_cache_size + 1`, committed `b8149c32`, and reran the broad matrix
  (`322 passed, 2 deselected`), targeted matrix (`33 passed`), routing matrix
  (`13 passed`), and numeric capacity probes.
- **Final Architectural Status:** **CLEAR**. The re-review confirmed one
  logical-length contract across validation, memory filtering, and wrapper
  block allocation; no remaining BLOCK or WATCH was identified.

## CP-21 disposition

The architecture lane is complete and clear. The code/spec lane is also
complete with an `APPROVE` result and no severity-rated findings. The local
review synthesis is complete; outward-facing delivery remains a separate
push gate.

## CP-22 - PR #21 code/spec re-review

- **Target Component/Phase:** Final code/spec review of the stacked PR #21
  focused-repair range and its composition with PR #20.
- **Reviewer Agent Identity:** Independent code/spec lane
  `/root/pr21_code_review`.
- **Inspected Artifacts:** PR #21's focused profiling and test paths, the
  synthetic tree `dc8081821c0e36fb007f3c67902a88541189d119`, targeted sampling
  and EP tests, MoE routing/load tests, numeric probes, compile output, and
  static hygiene checks.
- **Identified Issues/Anomalies:** No CRITICAL, HIGH, MEDIUM, or LOW finding.
  The reviewer separately reported an accidental local merge commit
  `54e123ea`; it was unpushed, conflict-free, and did not modify file content.
- **Remediation/Verification Code Actions Taken:** Kept the merge commit as
  the non-rewriting ancestry that records PR #20 head `18d1a23e` as the second
  parent. Confirmed the local tree is clean and the final review records use
  `b8149c32` / `dc808182`.
- **Final Code/Spec Status:** **APPROVE**. The lane confirmed the decode,
  mixed-axis, MoE routing, EP-domain, aggregate-workload, and static gates.

## CP-22 disposition

Both independent review lanes are complete: architecture **CLEAR** and
code/spec **APPROVE**. PR #21 is ready for the normal push; neither PR is to
be merged in this step.

## CP-23 - PR #21 remote delivery

- **Target Component/Phase:** Outward-facing publication of the approved PR
  #21 stack.
- **Reviewer Agent Identity:** `/root` (read-only remote verification; no
  independent review claim).
- **Inspected Artifacts:** Local merge commit `54e123ea`, remote branch refs,
  PR #20 and PR #21 metadata, and the updated PR #21 body.
- **Identified Issues/Anomalies:** None. GitHub reported PR #21
  `OPEN/CLEAN` with base `18d1a23e`; PR #20 remained `OPEN/CLEAN` at
  `18d1a23e`.
- **Remediation/Verification Code Actions Taken:** Pushed with a normal
  non-force update from `03952bf1` to `54e123ea`, updated the validation
  summary, and confirmed both PRs remain unmerged.
- **Final Delivery Status:** **PUBLISHED / UNMERGED**.

## CP-24 - Two-round PR #21 review and concurrent Claude run

- **Target Component/Phase:** Published PR #21 option-B head
  `18d1a23e..8cc267ea`, including target-local explicit-KV filtering and
  discard feedback.
- **Reviewer Agent Identity:** Round-1 code/spec lane
  `/root/pr21_code_review`; Round-1 architecture lane
  `/root/pr21_arch_review`; Round-2 code/spec lane
  `/root/pr21_postrepair_codespec`; Round-2 architecture lane
  `/root/pr21_postrepair_architecture`; external lane StepCode Claude Opus 5.
- **Inspected Artifacts:**
  `/data/ycfeng/tmp/pr21_r1_codespec_8cc267ea.md`,
  `/data/ycfeng/tmp/pr21_fresh_arch_review_20260823.md`,
  `/data/ycfeng/tmp/pr21_r2_codespec_8cc267ea.md`,
  `/data/ycfeng/tmp/pr21_r2_architecture_8cc267ea.md`, and
  `/data/ycfeng/tmp/ask_claude_pr21_optionb_20260822.raw.log`.
  The Round-1 architecture artifact explicitly targets the preceding
  `a8d75b84` tip; it is retained as pre-final evidence and is not used as the
  final `8cc267ea` architecture verdict.
- **Identified Issues/Anomalies:** Round-1 code/spec returned **APPROVE** with
  no HIGH/CRITICAL finding and noted direct-CLI default derivation as a
  non-blocking follow-up. Round-2 code/spec returned **CLEAR/APPROVE** with no
  severity-rated defect and recorded token-list capacity semantics, strict
  Python integer validation, duplicate targets, and broad automatic exception
  catches as follow-ups. Round-2 architecture returned **CLEAR with WATCH**:
  Option-B warning/union/fail-fast behavior is observable, while the
  true-mixed aggregate-token versus rounded-block mismatch is pre-existing and
  outside this PR.
- **Remediation/Verification Code Actions Taken:** No reviewer changed
  production code or remote state. The follow-up items were recorded in
  `issues.md` and `future.md`; the standard-attention warning audit is captured
  in CP-25 below.
- **External review disposition:** The concurrent Claude command used
  `claude-opus-5`, `effort=max`, and `timeout 1800s`. The child exited `143`
  at the limit and emitted no final synthesis, so the run is **inconclusive**
  and is not counted as approval.

## CP-24 disposition

The final PR #21 head is independently **CLEAR/APPROVE** for the focused
Option-B scope, with the architecture WATCH items explicitly deferred. The
Claude timeout remains an evidence artifact, not a verdict. PR #20 and PR #21
remain open and unmerged.

## CP-25 - Option-B discard-feedback verification

- **Target Component/Phase:** Standard-attention explicit decode KV filtering
  and parent-process feedback at PR #21 head `8cc267ea`.
- **Reviewer Agent Identity:** `/root` (direct case-driven verification; no
  independent review claim).
- **Inspected Artifacts:** `frontier/profiling/attention/main.py`,
  `tests/unit/test_attention_memory_filter_contract.py`, the 59-test
  sampling/EP/routing matrix, and the inline warning/union probe.
- **Identified Issues/Anomalies:** None in the approved scope. The filter
  intentionally drops rows that exceed a target's physical capacity, but
  explicit drops are visible before worker dispatch and global coverage is
  checked after target-local filtering.
- **Remediation/Verification Code Actions Taken:** Ran the focused warning
  contract (`3 passed`), the combined PR #21 matrix (`59 passed`), compileall
  (`rc=0`), and `git diff --check`. With capacities `256` and `512` and
  requested KV `[255, 300, 511]`, the small target retained `[255]` and
  emitted one `RuntimeWarning` for two dropped combinations (`[300, 511]`);
  the large target retained all three values. The union was
  `[255, 300, 511]`. An all-target drop of `511` raised
  `ValueError` with the target-capacity map.
- **Final Status:** **PASS**. The warning includes KV values, model, TP,
  capacity, retained values, and dropped count; the parent summary prints
  requested/retained coverage and target capacities.

## CP-25 disposition

The user-visible discard feedback requirement is closed. No production change
is required beyond the already pushed `8cc267ea` repair.

## CP-26 - Producer-side optional metadata audit

- **Target Component/Phase:** PR #20 lookup metadata producer/serializer
  boundary and its relationship to PR #21.
- **Reviewer Agent Identity:** `/root/pr21_postrepair_codespec` (read-only
  source/AST audit).
- **Inspected Artifacts:** `sklearn_execution_time_predictor.py`,
  `shared_prediction_model_manager.py`, `frontier/training/base_trainer.py`,
  runtime `model_info` builders, pickle-cache paths, and the optional metadata
  consumer in the on-demand lookup path.
- **Identified Issues/Anomalies:** Production assignment counts for
  `_feature_bounds`, `_feature_domain`, `_feature_constraints`, `_identity`,
  and `_model_identity` are all zero. Runtime descriptors do not pass those
  fields through, and standalone `BaseTrainer` persists only the raw
  estimator.
- **Remediation/Verification Code Actions Taken:** No source change. Kept
  the fields optional, recorded the producer/descriptor/cache round-trip
  contract as a separate cross-cutting follow-up, and explicitly excluded it
  from PR #21's profiling-envelope scope.

## CP-26 disposition

`SCOPE-011` remains open and pre-existing. Making the fields mandatory would
require coordinated producer, serializer, runtime-descriptor, and reload
changes plus a new round-trip regression; it is not a focused PR #21 repair.

## CP-27 - MTP unified-family/interface audit

- **Target Component/Phase:** Target-embedded MTP operator registry,
  TP-consumer mapping, and extension completeness.
- **Reviewer Agent Identity:** `/root/pr21_postrepair_claude` (read-only
  source and runtime probe).
- **Inspected Artifacts:** `frontier/spec_decode/mtp_registry.py`,
  `frontier/operators/families.py`, `frontier/operators/binding.py`,
  `frontier/profiling/linear_op/linear_op_impl.py`, three TP consumers,
  model-name enumerators, and profiling-plan snapshots.
- **Identified Issues/Anomalies:** `mtp_fusion_proj` and `lm_head_linear` are
  separate physical `ColumnParallelLinear` operators and are not members of
  the generic `OperatorFamilySpec` registry. The focused repair correctly
  makes all three TP consumers read the dedicated MTP accessor, but a new
  MTP registry member does not yet reach every model-name and profiling-plan
  enumeration site. The method contract and module-level helper can also
  diverge if independently mutated.
- **Remediation/Verification Code Actions Taken:** No source change. Kept
  the focused TP repair in PR #20, recorded full MTP family/interface
  unification and extension propagation as a separate medium-priority audit,
  and did not treat the current two-op behavior as a PR blocker.

## CP-27 disposition

PR #20's focused MTP repair remains valid for its agreed scope. A complete
unified-family/interface migration crosses shared registry, profiling,
runtime-policy, and model-enumeration contracts and requires a new design
decision.

## CP-28 - Remote PR description synchronization

- **Target Component/Phase:** Published PR metadata after Option-B repair.
- **Reviewer Agent Identity:** `/root` (read-only remote verification).
- **Inspected Artifacts:** GitHub PR #20/#21 bodies, head/base refs, merge
  state, and auto-merge state after the authorized `gh pr edit` operations.
- **Identified Issues/Anomalies:** The prior bodies contained stale heads and
  omitted the final warning-feedback evidence.
- **Remediation/Verification Code Actions Taken:** Updated PR #20 to report
  `18d1a23e` and PR #21 to report `8cc267ea`, including current validation
  numbers and the inconclusive Claude timeout. Both PRs remain `OPEN/CLEAN`,
  with `mergedAt=null` and auto-merge unset.

## CP-28 disposition

Remote descriptions now match the published branches. No merge, force-push,
or source change occurred.

## CP-29 - Deferred WATCH audit

- **Target Component/Phase:** Remaining PR #21 architecture WATCH items at
  head `8cc267ea`.
- **Reviewer Agent Identity:** `/root/pr21_postrepair_architecture`
  (read-only source and direct probes).
- **Inspected Artifacts:** true-mixed planner and wrapper block allocation,
  target-key aggregation and worker loops, CLI parser/default validation,
  helper integer normalization, online-grid exception paths, and
  `num_tokens_list`/`extra_num_tokens` handling.
- **Identified Issues/Anomalies:** The audit reproduced six distinct behaviors:
  aggregate-token versus rounded-block filtering can disagree; duplicate
  targets can repeat work while tuple-key metadata undercounts it; direct CLI
  `max_seq_len`-only invocation can fail before derivation; programmatic
  helpers silently coerce non-integral values; automatic grids can skip
  injected `RuntimeError`/`ValueError`; and `extra_num_tokens` can exceed the
  exact-list capacity.
- **Remediation/Verification Code Actions Taken:** No source change. Recorded
  each behavior with its current contract and deferred the required
  block-aware, CLI, API, exception, or capacity-interface decisions.

## CP-29 disposition

All six WATCH items are real, reproducible, and pre-existing or orthogonal to
the Option-B explicit-KV repair. They do not invalidate warning visibility,
retained-union coverage, or all-target fail-fast behavior, so PR #21 remains
unchanged.

## CP-30 - Gate 2 typed EP lane integration review

- **Target Component/Phase:** Post-PR17 PR20 integration repair for the
  regular-Batch MoE workload contract and physical zero-routed lanes.
- **Reviewer Agent Identity:** `/root` (implementation lane; independent
  review agents remain separate and their findings require re-verification).
- **Inspected Artifacts:** `frontier/moe_ep_workload.py`,
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  the scheduler's `LayerEPWorkload`/`EPBatchGroup` construction paths, and the
  six focused test files.
- **Identified Issues/Anomalies:** The original entity-type guard rejected
  valid regular-Batch callers; global maps and local-width maps require
  different feature widths; canonical zero-load lanes were represented by
  full maps and therefore were not caught by an emptiness check. The broad
  unit run also exposed 18 unrelated release/docs/analysis baseline failures.
- **Remediation/Verification Code Actions Taken:** Added the typed lane value
  object and canonical splitter, threaded identity through both predictor
  classes, added fail-fast map/topology/source/layer checks, added regression
  tests, and skipped zero-sum shuffling model lookup after non-negative-count
  validation. The RED test reproduced the defect; GREEN passed. Focused
  verification passed `107`; PR17-sensitive verification passed `188` plus
  `1 xfailed`; compile and whitespace checks passed. The broad baseline was
  recorded without widening scope.

## CP-30 disposition

The zero-routed admission defect is resolved locally and the focused evidence
is clean. The integration remains **WATCH** until the independent code and
architecture reviews re-vet the typed identity contract, the local changes are
committed as a coherent sub-step, and PR21 is refreshed from the corrected
PR20 base. No remote state was changed.

## CP-31 - A' docs-freeze review

- **Target Component/Phase:** Post-PR17 PR20/PR21 conflict-resolution design
  before implementation.
- **Reviewer Agent Identity:** `/root` (direct source and document audit).
- **Inspected Artifacts:** `requirements.md`, `issues.md`, `design.md`,
  `plan.md`, `harness.md`, `progress.md`, `frontier/moe_ep_workload.py`, the
  scheduler EP plan/entity paths, predictor call sites, and the existing MTP
  phase path.
- **Identified Issues/Anomalies:** The physical EP protocol is symmetric, but
  the software contract is split between global `Batch` maps and local
  `EPBatchGroup` maps. The provisional probe also contains a width-validation
  escape hatch, duplicate identity ownership, a half-migrated predictor
  interface, and synthetic MTP entities.
- **Remediation/Verification Code Actions Taken:** Added the formal RCA,
  ownership graph, descriptor invariants, zero-lane/barrier rules, pure MTP
  contract, ordered TDD plan, and harness gates. No production or test source
  changed in this checkpoint.
- **Final Status:** **PASS for docs freeze; implementation pending.** The next
  allowed code action is the descriptor-contract RED tests.

## CP-32 - Scheduler/entity typed-lane propagation

- **Target Component/Phase:** `scheduler-plan/entity-propagation`.
- **Reviewer Agent Identity:** `/root` (direct implementation and independent
  focused verification).
- **Inspected Artifacts:** `frontier/entities/batch.py`,
  `frontier/entities/batch_stage.py`,
  `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`,
  `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py`,
  `frontier/events/replica_stage_schedule_event.py`,
  `frontier/metrics/metrics_store.py`, and the EP scheduler/entity tests.
- **Identified Issues/Anomalies:** The plan/entity API still accepted a raw map,
  stage-ledger projection copied that map into a mutable second owner, and
  several fixtures encoded expert IDs inconsistent with canonical contiguous
  ownership.
- **Remediation/Verification Code Actions Taken:** Propagated the immutable
  `EPLaneWorkload`, attached it to `BatchStage`, moved map creation to the
  metrics output boundary, added descriptor identity assertions, and migrated
  fixtures to fixed-width legal topology. Commit `c70d2052` contains only this
  sub-step's source and tests.
- **Verification:** `611 passed, 19 skipped, 1 xfailed`; targeted module
  `py_compile` and `git diff --check` passed.
- **Final Status:** **PASS.** Predictor-interface RED tests are the next gate.

## CP-33 - MTP and trace ownership audit

- **Target Component/Phase:** Post-implementation A' contract audit before
  the remaining MTP/observability edits.
- **Reviewer Agent Identity:** `/root` (direct source audit; no external review
  conclusion is claimed).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_execution_time_predictor.py:5107-5347,5364-5443,5689-5707`,
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:2165-2235,2558-2670`,
  `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py:493-612,3835-3850,4167-4182`,
  `frontier/events/replica_stage_schedule_event.py:395-405`, and the MTP/typed-lane
  focused tests.
- **Identified Issues/Anomalies:** The physical MoE MTP path is descriptor-only,
  while generic target-embedded MTP still uses short-lived block/terminal
  `Batch` objects and copied request progress. The trace helper still accepts a
  raw map even though all callers derive it from a typed descriptor.
- **Remediation/Verification Code Actions Taken:** Added SCOPE-026 with the
  two evidence-backed options, split the harness gate, and inserted a typed
  trace-helper sub-step. No production or test source was changed during this
  audit.
- **Escalation:** A maintainer decision is required before expanding the shared
  MTP predictor interface. Option A (narrow physical lane seam) is recommended;
  the typed trace migration remains independent and can follow that decision.
- **Final Status:** **BLOCKED AT DECISION GATE, NOT AT ENVIRONMENT.**

## CP-34 - MONOLITHIC initial-decode frontier audit

- **Target Component/Phase:** `monolithic-initial-boundary-audit` in the
  approved token-ledger addendum.
- **Reviewer Agent Identity:** `/root` (direct production-method probe).
- **Inspected Artifacts:** `frontier/entities/request.py:1190-1335`,
  `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:2665-2770`,
  the target-embedded MTP metadata builder, and the token-ledger regression
  fixtures.
- **Identified Issues/Anomalies:** The scheduler exposes two related values at
  the prefill boundary: request progress already includes one output token,
  while the scheduler-visible computed frontier intentionally remains at the
  prompt width. This can look like a second physical token contract if it is
  read without the lifecycle context.
- **Remediation/Verification Code Actions Taken:** Ran a real `Request` matrix
  (`P=8`, `D=32`, planned `0/1/2/4`) through the production helper methods.
  The boundary state (`processed=9`, decode progress `1`) yielded frontier `8`
  and widths `1/1/2/4`; post-boundary states yielded full widths `1/2/3/5`.
  No source change was made. A focused regression will preserve this rule.
- **Final Status:** **PASS.** The `max(planned_drafts, 1)` branch is a
  scheduler admission/frontier projection, not a compute/gating/materializer
  width. The canonical physical token ledger remains unchanged.

## CP-35 - Docs-first implementation handoff review

- **Target Component/Phase:** Remaining post-PR17 narrow A' implementation
  before predictor/communication/trace commits.
- **Reviewer Agent Identity:** `/root` (direct source and task-document audit).
- **Inspected Artifacts:** The current worktree diff, all
  `EPLaneWorkload`/`LayerEPWorkload` consumers, the scheduler event trace call,
  and the approved design/plan/harness/issues records.
- **Identified Issues/Anomalies:** The provisional dirty slice contains the
  intended typed descriptor propagation, but the event trace helper still
  receives a raw map at the logging call boundary. The large MoE predictor diff
  also contains formatting noise and requires whitespace-insensitive semantic
  review before commit.
- **Remediation/Verification Code Actions Taken:** Added the implementation
  handoff ownership record, named SCOPE-029, and constrained the next steps to
  descriptor/predictor/communication/MTP/trace gates. No additional production
  behavior was changed during this documentation checkpoint.
- **Final Status:** **PASS for docs; implementation and focused verification
  remain in progress.**

## CP-36 - Typed-trace observability migration

- **Target Component/Phase:** `typed-trace-observability`.
- **Reviewer Agent Identity:** `/root` (direct implementation and focused
  verification).
- **Inspected Artifacts:**
  `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`,
  `frontier/events/replica_stage_schedule_event.py`,
  `tests/unit/test_typed_ep_trace_contract.py`, and all production call sites
  found by the repository-wide trace search.
- **Identified Issues/Anomalies:** The helper's old signature accepted
  `per_expert_tokens: Dict[int, int]`, allowing a caller to bypass typed lane
  identity and topology validation. The direct event path still projected the
  map before entering the helper.
- **Remediation/Verification Code Actions Taken:** Changed the helper to
  require `EPLaneWorkload`, derive `ep_id`, `moe_expert_parallel_size`, and the
  serialization map internally, and migrated PREFILL, DECODE, and direct
  DECODE_FFN callers. Added the minimal contract regression and kept the
  legacy log field as an output projection.
- **Verification:** The RED test first failed with the expected unexpected
  keyword error. Fresh GREEN verification passed `168` tests across the trace,
  EP materialization, and non-dummy MoE matrix; module compilation and
  `git diff --check` passed. Commit `183cfd61` contains only this sub-step.
- **Final Status:** **PASS.** No production trace helper accepts a raw map as
  its workload input; the next gate is predictor-interface audit.

## CP-37 - Terminal MTP overshoot EP>1 RED and repair design

- **Target Component/Phase:** `terminal-mtp-red` before the physical-lane hook.
- **Reviewer Agent Identity:** `/root` (direct production call-chain audit and
  TDD RED verification).
- **Inspected Artifacts:**
  `VLLMv1EngineReplicaScheduler._build_spec_decode_batch_metadata()`,
  `SklearnExecutionTimePredictor._get_mtp_terminal_overshoot_time()`,
  `SklearnMoEExecutionTimePredictor._get_execution_time_internal()`,
  `predict_moe_lane_phase_times()`, `LayerEPWorkload.lane()`, the canonical EP
  payload builder, and `tests/unit/test_mtp_terminal_overshoot_ep_replay.py`.
- **Identified Issues/Anomalies:** Real terminal metadata produces a row, but
  the generic terminal `Batch` has no physical EP lane descriptor. EP>1 MoE
  replay therefore reaches the all-to-all payload builder and fails with
  `ValueError` before physical lane aggregation. The fixture's phase values
  sum to `25.0ms` (`attention=7.0`, phase maxima `2+2+4+4+6=18.0`), not
  `27.0ms`.
- **Remediation/Verification Code Actions Taken:** Corrected the RED expected
  value, reran the real test, and recorded the failure as the intended seam
  signal. The approved repair is a default terminal-row hook plus a MoE
  override that reuses the existing typed lane phase API, preserves
  `stage_id`/`layer_id`, scales only per-layer phases by `num_layers`, and
  leaves the generic scheduler-independent shape adapter unchanged.
- **Verification:** Metadata assertions passed; one focused test failed with
  exactly `expert-parallel all-to-all payload requires an EPLaneWorkload
  descriptor`. No cache, schema, or fixture setup error preceded the target
  failure. Production edits are intentionally pending this RED checkpoint.
- **Final Status:** **PASS for RED/design gate; implementation pending.**

## CP-38 - Terminal MTP EP>1 physical-lane hook GREEN

- **Target Component/Phase:** `terminal-row-hook` and
  `focused-terminal-green`.
- **Reviewer Agent Identity:** `/root` (direct implementation and fresh
  verification).
- **Inspected Artifacts:**
  `sklearn_execution_time_predictor.py`,
  `sklearn_moe_execution_time_predictor.py`,
  `tests/unit/test_mtp_terminal_overshoot_ep_replay.py`, the structural MTP
  replay tests, and the typed EP predictor contract tests.
- **Identified Issues/Anomalies:** The RED fixture lacked the real
  `moe_expert_parallel_size` field and stopped in configuration validation
  before reaching the intended seam. No production fallback was added; the
  fixture was corrected to match `ReplicaConfig`.
- **Remediation/Verification Code Actions Taken:** Added the default
  terminal-row hook, extracted one shared typed-lane phase aggregator, and
  added the EP>1 MoE override. The helper materializes only
  `LayerEPWorkload.lane(ep_id)`, calls the canonical five-phase API for every
  participant, and keeps the attention probe at one layer so CPU/pipeline
  scope is not multiplied.
- **Verification:** Terminal replay returns `25.0 ms`; lane IDs `[0, 1]`,
  local widths `[4, 4]`, routed sum `2`, phase maxima `(2, 2, 4, 4, 6) ms`.
  `num_layers=3` returns `61.0 ms` with attention probe `num_layers=1`.
  EP=1 and dense fallbacks each return `5.0 ms`. The terminal/structural/typed
  subset passes `32` tests and the complete focused matrix passes `178` tests;
  compile and whitespace checks pass.
- **Final Status:** **PASS.** The terminal physical-lane seam is closed;
  focused report and sub-step commits remain before PR20/PR21 merge-readiness.

## CP-39 - Predictor global-layer identity propagation audit

- **Target Component/Phase:** `predictor-layer-identity-audit` after the
  terminal MTP physical-lane hook.
- **Reviewer Agent Identity:** `/root` (direct call-chain audit and focused
  TDD verification; no independent review conclusion is claimed).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`,
  `tests/unit/test_moe_predictor_layer_id_semantics.py`, the terminal MTP
  regression, and the structural MTP replay tests.
- **Identified Issues/Anomalies:** The public predictor accepted the real global
  `layer_id`, but the internal execution-time method previously omitted it;
  attention probes therefore used layer zero and terminal replay substituted
  `pipeline_stage`. Stage identity and layer identity are distinct in the
  post-PR17 contract.
- **Remediation/Verification Code Actions Taken:** Added a compatibility-
  defaulted internal `layer_id`, forwarded the public value, and used it for
  attention and terminal MTP calls. The RED propagation assertion failed with
  `KeyError: 'layer_id'`; the repaired layer/terminal/structural subset passed
  `32` tests. No stage-derived layer fallback or second predictor wrapper was
  added.
- **Final Status:** **PASS.** The remaining action is the fresh combined
  verification and coherent sub-step commits before the local PR20/PR21
  merge-readiness audit.

## CP-40 - EP>1 boundary audit and PDD attention-only identity seam

- **Target Component/Phase:** `all-to-all-structural-MTP-audit` and the
  pending `pdd-attention-only-identity-repair`.
- **Reviewer Agent Identity:** `/root` (direct production call-chain audit;
  TDD RED is pending and no repair conclusion is claimed).
- **Inspected Artifacts:** `frontier/operators/families.py`,
  `frontier/operators/spec.py`, the aggregate `CommPayloadContext` payload
  path, structural-MTP registry/config loaders, PDD scheduler call sites, and
  `_predict_attention_only_stage_execution_time()`.
- **Identified Issues/Anomalies:** EP>1 routed aggregate dispatch/combine
  already fails at payload construction without an `EPLaneWorkload`, so a new
  caller-side guard would duplicate the existing owner. The local structural
  registry loads configs with `48/2/20` layers and explicit MoE layer zero;
  two names have missing local JSON and remain a coverage gap. Separately,
  the PDD attention-only helper receives no `layer_id` and hard-codes zero,
  despite scheduler callers passing non-zero global identities.
- **Remediation/Verification Code Actions Taken:** Recorded the audit in the
  design/harness and queued a single private-helper parameter propagation
  regression. No production edit or speculative configuration asset was added
  in this checkpoint.
- **Final Status:** **PASS for audit; SCOPE-032 remains open pending RED/GREEN.**

## CP-41 - PDD attention-only identity repair RED/GREEN

- **Target Component/Phase:** `pdd-attention-only-identity-repair` after the
  EP>1 boundary audit.
- **Reviewer Agent Identity:** `/root` (direct TDD implementation and fresh
  focused verification).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  `tests/unit/test_sklearn_disaggregation_execution_time_predictor.py`, the
  PDD scheduler `include_ffn=False` call sites, and the layer/terminal MTP
  regression files.
- **Identified Issues/Anomalies:** Before the edit, the real public path
  passed `layer_id=17` to the predictor but the private helper queried
  attention with `layer_id=0`; the focused assertion failed exactly as
  `assert 0 == 17`.
- **Remediation/Verification Code Actions Taken:** Added the private helper's
  compatibility-defaulted `layer_id`, forwarded the public value, and added
  one focused regression. No stage-derived identity or fallback wrapper was
  added.
- **Verification:** The affected matrix passed `64` tests in `2.77s`; the
  captured attention call received `layer_id=17` while `stage_id=3` remained
  unchanged.
- **Final Status:** **PASS.** SCOPE-032 is resolved; the next gate is the
  full focused report, dirty-diff review, and local PR20/PR21 merge-readiness.

## CP-42 - Dummy terminal-MTP physical-lane repair

- **Target Component/Phase:** Dummy branch of the typed terminal-MTP MoE lane
  phase seam, after the non-dummy terminal hook and PDD identity repair.
- **Reviewer Agent Identity:** `/root` (direct production call-chain audit and
  focused verification; independent review is still pending).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  `frontier/operators/families.py`, the terminal-MTP regression, the typed
  predictor/scheduler tests, and the direct dummy probe.
- **Identified Issues/Anomalies:** The typed lane helper entered
  `_get_execution_time_internal()` even when dummy mode was enabled. That path
  reached the intentionally unsupported dummy MoE all-to-all predictor and
  raised `NotImplementedError` before producing phases. The existing PDD
  comments also described MoE TP all-reduce as lane-local although the registry
  payload builder uses the source batch's shared pre-routing width.
- **Remediation/Verification Code Actions Taken:** Added the optional typed
  lane argument to `_get_dummy_execution_time()` and routed the dummy branch
  through it. Kept the non-dummy branch unchanged. Updated only the misleading
  PDD communication comments. Zero-routed lanes now retain dispatch/combine and
  shared all-reduce while returning `0.0` for routed shuffling/grouped-GEMM.
- **Verification:** The direct dummy probe returned `27.0 ms`; lane phase sums
  were `7.0` and `5.0 ms`, with lane `1` routed count `0` and no positive-load
  lookup. Generic/`step2_mini`/`step3_text` probes returned `27.0/33.0/33.0 ms`.
  The focused A' matrix passed `350` tests in `10.68s`; compile and whitespace
  checks remained clean.
- **Final Status:** **PASS for this sub-step.** The production/test/doc files
  still require coherent commits, a fresh final audit, and an independent
  code-review checkpoint before merge-readiness is reported.

## CP-43 - Physical source-width conservation and aggregate admission audit (historical)

- **Target Component/Phase:** `raw-width-conservation` plus the then-open
  `EP>1-aggregate-admission` gate. This checkpoint predates the decision
  recorded in CP-44.
- **Reviewer Agent Identity:** `/root` (direct TDD implementation and source
  audit; no independent review conclusion is claimed).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  `tests/unit/test_typed_ep_predictor_contract.py`,
  `frontier/moe_ep_workload.py`, the canonical communication payload builder,
  and the typed-lane scheduler/entity call chain.
- **Identified Issues/Anomalies:** The explicit-lane source-batch fallback
  multiplied compute-effective width (`8`) by `router_topk` and rejected a
  materialized physical workload with `10` assignments (`expected 16`). The
  all-to-all operator already rejects routed aggregate calls without a lane,
  but dummy public aggregate prediction can return structurally before that
  boundary while non-dummy paths may perform lookup work first.
- **Remediation/Verification Code Actions Taken:** Added one regression with
  physical width `5`, effective width `8`, and top-k `2`; observed the intended
  RED `allocated 10, expected 16`; changed only the fallback conservation
  expression to `batch.total_num_tokens * router_topk`. The typed predictor
  contract module then passed `28` tests in `2.55s`. No admission change was
  made while the dummy compatibility surface remains a design fork.
- **Final Status:** **PASS for width repair at that checkpoint.** The
  aggregate-admission decision was subsequently resolved by CP-44; this
  historical entry intentionally preserves the earlier observation.

## CP-44 - Narrow A' aggregate admission docs freeze

- **Target Component/Phase:** EP>1 routed aggregate public predictor boundary
  after the raw-width conservation repair.
- **Reviewer Agent Identity:** `/root` (direct source audit and maintainer
  decision recording; implementation review remains pending).
- **Inspected Artifacts:** `design.md`, `harness.md`, `plan.md`, `issues.md`,
  `requirements.md`, the two public predictor entry points, the scheduler lane
  materializers, and the communication payload builder.
- **Identified Issues/Anomalies:** The prior docs described communication
  payload construction as the only admission boundary and left the dummy vs
  non-dummy decision pending, although the selected architecture requires a
  semantic predictor boundary before mode-dependent lookup.
- **Remediation/Verification Code Actions Taken:** Recorded narrow A' as the
  approved contract, documented the concrete MONOLITHIC/PDD classification
  rules, separated the early semantic check from the final payload invariant,
  and added zero-lookup acceptance criteria for dummy and non-dummy failures.
  No production code or test source was changed in this docs sub-step.
- **Verification:** Cross-file scan confirms the decision is resolved and the
  old pending/communication-only wording is removed from the active sections.
  The next gate is TDD RED for the shared admission helper.
- **Final Status:** **PASS for docs freeze; implementation pending.**

## CP-45 - Post-A' ledger and admission verification audit

- **Target Component/Phase:** Final focused verification preparation after the
  narrow A' implementation and token-ledger repairs.
- **Reviewer Agent Identity:** `/root/audit_ep_aggregate/code_spec_lane`
  (read-only source/doc audit; parent integration review remains pending).
- **Inspected Artifacts:**
  `tests/unit/test_mtp_token_ledger_repair.py`,
  `tests/unit/test_moe_ep_aggregate_admission.py`,
  `frontier/moe_ep_workload.py`, the shared MoE admission helper and both
  public predictor seams, plus `plan.md`, `issues.md`, `harness.md`,
  `progress.md`, and the terminal-MTP test report.
- **Identified Issues/Anomalies:** The token-ledger test path was present and
  GREEN but was not reflected in the active harness checkboxes. The active
  issue/plan text still described the implemented admission and width repairs
  as pending. A direct context probe showed that the narrow helper admits
  internally valid descriptors with predictor EP `2`/descriptor EP `4` and
  predictor top-k `2`/descriptor top-k `1`; this is a separate topology-context
  contract and is not proven reachable from production role-specific
  materializers.
- **Remediation/Verification Code Actions Taken:** Updated only task-memory
  status/evidence records, marked the completed ledger/admission gates, and
  recorded the context mismatch as residual WATCH `SCOPE-035`. No production
  validator, caller-side guard, topology inference, or fallback was added.
  Ran the standalone ledger file (`8 passed in 2.51s`) and the exact ten-file
  focused matrix (`181 passed in 3.37s`). The active nine-document history,
  marker, placeholder, and conflict scan plus `git diff --check` were clean.
- **Final Status:** **PASS for the documentation and verification audit.**
  Coherent commits, independent code review, and local PR20/PR21
  merge-readiness remain open in the parent task.

## CP-46 - Lane-local conservation ownership docs freeze

- **Target Component/Phase:** `lane-conservation-RCA` before the helper and
  fixture correction.
- **Reviewer Agent Identity:** `/root` (direct source audit) with independent
  confirmation from `/root/behavior_review/architecture_lane`.
- **Inspected Artifacts:** `frontier/moe_ep_workload.py`,
  `frontier/entities/batch.py`, scheduler EP materialization, the terminal-MTP
  lane replay, `_admit_routed_ep_aggregate()`, and
  `tests/unit/test_moe_ep_aggregate_admission.py` plus the typed predictor
  contract tests.
- **Identified Issues/Anomalies:** The helper's explicit-lane fallback treated
  `batch.total_num_tokens * router_topk` as the expected count for one physical
  lane. The focused module therefore rejected a valid zero lane with
  `allocated 0, expected 4`; a partial lane is affected by the same domain
  error. A batch-attached lane entity has a different contract because its
  `total_num_tokens` is lane-local post-routing width.
- **Remediation/Verification Code Actions Taken:** Updated the task design,
  issue, plan, harness, and progress records to assign aggregate conservation
  to `LayerEPWorkload`/scheduler, local-width equality to lane entities, and
  descriptor/topology admission only to source-batch plus explicit-lane calls.
  No production or test source was changed in this docs checkpoint.
- **Verification:** The pre-fix admission run remains `25 passed, 1 failed`
  with the targeted zero-lane error. The next gate is a focused RED/GREEN
  fixture and helper correction, followed by the full affected matrix.
- **Final Status:** **PASS for the docs freeze; implementation pending.**

## CP-61 - SCOPE-042 EP payload admission ordering

- **Target Component/Phase:** Residual typed EP dispatch/combine consumer
  boundary after the option-1 decision.
- **Reviewer Agent Identity:** `/root` (direct call-chain audit, TDD RED/GREEN
  reproduction, and source re-vet; independent final review remains pending).
- **Inspected Artifacts:** `_validate_ep_barrier_arrival()`,
  `_get_step3_ep_alltoall_payload_bytes()`,
  `on_ep_alltoall_dispatch_ready()`, and
  `on_ep_alltoall_combine_ready()` in
  `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`; the model
  architecture resolver; `tests/unit/test_typed_ep_scheduler_consumers.py`,
  `tests/unit/test_pdaf_step3_combine_payload.py`, and the PD-AF invariant
  fixtures; and the active task docs.
- **Identified Issues/Anomalies:** Combine resolved architecture collective
  policy before typed payload admission. A malformed final lane could therefore
  fail at the profile accessor rather than the descriptor boundary. The initial
  residual matrix had `14` expected fixture failures because legacy successful
  and predictor-error lanes lacked descriptors.
- **Remediation/Verification Code Actions Taken:** Kept option 1's barrier
  identity owner unchanged; moved the existing payload-owner call ahead of
  combine collective resolution and reused its returned all-to-all payload;
  migrated only fixtures whose intended behavior requires reaching the
  predictor/payload path; parameterized typed missing-descriptor and width
  mismatch regressions across dispatch and combine.
- **Verification:** The focused payload/typed-consumer/PD-AF invariant command
  passed `392` tests with `19` skips. Both malformed-lane cases made zero
  `predict_alltoall_time` calls and preserved the prior valid lane in the
  waiting room. Valid payload timing and predictor-error/non-finite-time
  assertions continued to pass.
- **Final Status:** **PASS for SCOPE-042.** Full static gates, independent
  review, coherent commits, and PR20/PR21 read-only ownership audit remain
  open.

## CP-57 - SCOPE-040b/041 fresh GREEN and owner re-vet

- **Target Component/Phase:** Role-capability and strict descriptor/predictor
  topology gates after the approved narrow A' implementation.
- **Reviewer Agent Identity:** `/root` (fresh source audit and direct probes;
  independent behavior review remains a separate final gate).
- **Inspected Artifacts:** The disaggregation constructor and
  `_ROUTED_ROLE_CONFIG_ATTRIBUTES` declaration in
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  `_admit_routed_ep_aggregate()` in
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  `ClusterConfig._setup_disaggregated_configs()`, and the focused predictor,
  admission, typed-lane, scheduler, and communication tests.
- **Identified Issues/Anomalies:** Fresh probes found no remaining defect in
  the approved boundaries. The legal PD and PD-AF aggregate shapes select
  different routed role subsets, and manually divergent descriptors are
  rejected before mode-specific work as intended.
- **Remediation/Verification Code Actions Taken:** Recorded the exact role
  calls and map states, the explicit missing-role exception with
  `routing_calls=0`, and eight dummy/non-dummy MONOLITHIC/disaggregation
  topology mismatch outcomes with `downstream_calls=0`. Re-ran the complete
  disaggregation predictor file (`40 passed in 2.71s`), aggregate admission
  file (`29 passed in 2.75s`), and typed predictor/scheduler pair (`32 passed
  in 2.77s`).
- **Final Status:** **PASS for SCOPE-040b and SCOPE-041.** Static gates,
  broader scoped matrix, independent review, and coherent commit remain open.

## CP-56 - SCOPE-041 strict descriptor/predictor topology docs freeze

- **Target Component/Phase:** Strict EP/top-k equality at the shared routed
  aggregate admission boundary.
- **Reviewer Agent Identity:** `/root` (direct source and call-order audit;
  independent behavior review remains pending).
- **Inspected Artifacts:** `_admit_routed_ep_aggregate()` in
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  both public predictor seams, scheduler/materializer descriptor owners,
  `tests/unit/test_moe_ep_aggregate_admission.py`,
  `tests/unit/test_typed_ep_predictor_contract.py`, and the active task docs.
- **Identified Issues/Anomalies:** The descriptor and predictor can be built
  from different role-specific configuration views. Before this decision,
  internally valid EP `4`/top-k `1` descriptors could pass a helper configured
  for EP `2`/top-k `2`; dummy and non-dummy paths did not share a complete
  context-equality gate.
- **Remediation/Verification Code Actions Taken:** Froze one shared helper as
  the owner of positive-integer EP validation, descriptor presence, exact
  router top-k equality, exact EP equality, descriptor identity, and applicable
  lane-local conservation. The helper must run before mode-specific timing or
  lookup. No caller-side guard, topology inference, synthetic descriptor, or
  fallback was introduced in this docs checkpoint.
- **Verification:** Existing focused regressions and the current helper source
  show the intended checks; fresh mismatch probes, role-capability GREEN, and
  final static gates are the next evidence step.
- **Final Status:** **PASS for docs freeze; SCOPE-041 implementation evidence
  pending.**

## CP-47 - Lane conservation, aggregate admission, and mixed-layer GREEN

- **Target Component/Phase:** SCOPE-036/037/038 production correction after the
  docs-first ownership freeze.
- **Reviewer Agent Identity:** `/root` (direct source re-vet and fresh focused
  verification; independent final review remains a later checkpoint).
- **Inspected Artifacts:**
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`,
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`,
  `frontier/execution_time_predictor/base_execution_time_predictor.py`,
  `frontier/moe_ep_workload.py`, the admission/mixed-layer/layer-identity tests,
  and the typed EP/MTP/communication ten-file matrix.
- **Identified Issues/Anomalies:** The historical MONOLITHIC checkpoint still
  described an implicit multi-layer aggregate compatibility bypass; Direction
  A supersedes that provisional behavior. No new production defect was found
  after aligning PDD classification and limiting conservation checks to their
  actual owner.
- **Remediation/Verification Code Actions Taken:** Passed the public predictor's
  resolved classification into the existing dummy owners, applied the shared
  routed admission before mode-specific work, and retained lane-local equality
  only for a batch-attached descriptor. Preserved `LayerEPWorkload` aggregate
  conservation, scheduler identity/barriers, and the payload builder's final
  structural validation. The fresh matrices passed `199` tests in `4.05s` and
  `183` tests in `31.47s`.
- **Final Status:** **PASS for the implementation checkpoint.** Independent
  diff review, PR20/PR21 ownership audit, and coherent commits remain open.

## CP-48 - Shared collective batch-semantics audit

- **Target Component/Phase:** Source effective width versus physical lane width
  across registered MoE communication operators.
- **Reviewer Agent Identity:** `/root` (direct call-chain inspection and
  reproducible registered-operator probe).
- **Inspected Artifacts:** `frontier/operators/families.py`,
  `frontier/operators/spec.py`, `frontier/entities/batch.py`,
  `frontier/moe_ep_workload.py`, scheduler lane materializers in
  `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`, and
  `tests/unit/test_comm_operator_families.py`.
- **Identified Issues/Anomalies:** No reachable production owner mismatch was
  found. Shared MoE-TP payloads intentionally consume the batch's pre-routing
  compute-effective width; EP all-to-all intentionally consumes the explicit
  lane routed count. Production `EPBatchGroup` builders preserve the source
  compute width through `moe_pre_routing_effective_total_tokens`.
- **Remediation/Verification Code Actions Taken:** Ran a direct case with raw
  width `5`, effective width `8`, top-k `2`, aggregate assignments `10`, lanes
  `[6, 4]`, and participant IDs `(0, 1)`. Observed shared all-reduce `128`
  bytes and EP all-to-all `[96, 64]` bytes. A zero lane retained shared `128`
  bytes and produced EP payload `0`. Kept the existing context/operator
  interface unchanged and added no fallback or source resolver.
- **Final Status:** **PASS.** The shared collective contract is consistent with
  the typed-lane architecture; `SCOPE-035` remains a separate residual WATCH.

## CP-61 - Pre-commit narrow A' merge-readiness checkpoint

- **Target Component/Phase:** Narrow A' implementation before the separate
  production/test and documentation commits and local PR21 consolidation.
- **Reviewer Agent Identity:** `/root/behavior_review` (independent runtime
  behavior review); `/root` performed the source/diff audit. Independent code
  review was still pending at this checkpoint.
- **Inspected Artifacts:** The seven production modules changed by A', the
  typed-lane and admission regressions, the formal PDD EP=2 probe, the
  post-PR17 integration ancestry, and the PR21 merge tree.
- **Identified Issues/Anomalies:** No CRITICAL, HIGH, or BLOCK runtime issue.
  The only retained WATCH is an unsupported manually reused heterogeneous
  `cluster_type=None` aggregate predictor, already documented in `future.md`.
  PR21 is not an ancestor of the integration head; its local merge has one
  expected content conflict in `frontier/profiling/README.md`.
- **Remediation/Verification Code Actions Taken:** Recorded fresh focused
  results (`309 passed in 7.55s` and `169 passed in 7.60s`), independent typed
  resolver/runtime evidence, and the exact PR21 ancestor/conflict facts. No
  remote push, merge, force-push, or PR-state change was performed.
- **Final Status:** **PASS for the behavior gate; final code-review verdict,
  static gates, commits, local PR21 merge, and merged-tree verification remain
  open.**

## CP-55 - SCOPE-040b PD-shaped role-capability RCA/docs freeze

- **Target Component/Phase:** Aggregate disaggregation predictor constructor
  after the SCOPE-040 stable-map lifecycle correction.
- **Reviewer Agent Identity:** `/root` (direct source audit; independent
  behavior review is queued before the owner fix is accepted).
- **Inspected Artifacts:** `ClusterConfig._setup_disaggregated_configs()` in
  `frontier/config/config.py`, the constructor and routing owner in
  `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`,
  the aggregate constructor regression, and the active task design/plan/gates.
- **Identified Issues/Anomalies:** A legal PD-shaped config contains
  `prefill_replica_config` and `decode_replica_config` but no
  `decode_ffn_replica_config`; the constructor's fixed three-role aggregate
  set still calls the missing DECODE_FFN role and raises
  `AttributeError: 'NoneType' object has no attribute 'model_config'`.
  The failure occurs before model, CSV, backend, or predictor lookup.
- **Remediation/Verification Code Actions Taken:** Updated the RCA, design,
  dependency chain, harness gates, and progress records. The agreed repair is
  one declaration-driven role-to-config-attribute mapping filtered by actual
  non-`None` role configs; no production code was changed in this docs freeze.
  The next gate is a focused PD-shaped RED regression with downstream lookup
  count `0`.
- **Final Status:** **PASS for the docs freeze; implementation pending.**
