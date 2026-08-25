## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-25 | Recorded the docs-freeze review: formal RCA, A' ownership graph, and TDD dependency plan are complete; the dirty Gate 2 probe remains provisional. |
| 2026-08-25 | Reviewed Gate 2 typed-lane integration, zero-routed shuffling admission, and focused/broad verification evidence. |
| 2026-08-22 | Recorded the evidence-backed six-item deferred-WATCH audit and its non-blocking disposition. |
| 2026-08-22 | Recorded the post-publication producer-metadata and MTP interface audits and synchronized both PR descriptions. |
| 2026-08-22 | Archived the two final PR #21 review rounds, the Option-B warning audit, and the concurrent Claude timeout without inferring a verdict. |
| 2026-08-22 | Recorded the normal PR #21 push at `54e123ea`, updated body evidence, and confirmed both PRs remain open and unmerged. |
| 2026-08-22 | Recorded the final PR #21 architecture re-review as CLEAR and the code/spec lane as APPROVE after closing decode memory-accounting WATCH. |
| 2026-08-22 | Recorded the focused repair checkpoint and the fresh post-repair Claude timeout without verdict. |
| 2026-08-22 | Recorded Claude's resumed REQUEST CHANGES verdict and the independent reproduction of both P1 findings. |
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
