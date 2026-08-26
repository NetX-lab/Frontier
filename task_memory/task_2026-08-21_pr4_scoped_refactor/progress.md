## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-26 | Recorded the EP>1 aggregate/structural-MTP audit and opened the focused PDD attention-only layer-identity repair gate. |
| 2026-08-26 | Completed the predictor-layer-identity audit; recorded RED/GREEN propagation evidence and the remaining commit/merge-readiness gates. |
| 2026-08-26 | Implemented and verified the terminal MTP EP>1 physical-lane hook; recorded `178` focused passes and numeric evidence. |
| 2026-08-26 | Confirmed the terminal MTP overshoot EP>1 RED and recorded the hook-based physical-lane repair handoff. |
| 2026-08-26 | Completed the typed-trace observability migration with RED/GREEN evidence and committed it as `183cfd61`. |
| 2026-08-26 | Recorded maintainer approval of the shared token-ledger repair and completed the docs-first implementation handoff. |
| 2026-08-25 | Audited the implemented A' MTP seam versus generic shape replay and opened the maintainer decision gate; no shared-interface edit made. |
| 2026-08-25 | Froze the A' RCA/design/plan before implementation and marked the existing Gate 2 probe as provisional. |
| 2026-08-25 | Implemented and verified typed EP lane workload admission, closed the zero-routed shuffling model-query defect, and recorded the broad unit baseline. |
| 2026-08-22 | Synchronized both remote PR descriptions and archived the producer-metadata and MTP interface audit conclusions. |
| 2026-08-22 | Closed the Option-B warning-feedback gate, archived both PR #21 review rounds and the Claude timeout, and recorded final local evidence. |
| 2026-08-22 | Restored the task context, pushed the PR #21 option-B repair through the HTTPS proxy, and restarted fresh review lanes against head `8cc267ea`. |
| 2026-08-22 | Verified the final PR #20 + PR #21 synthetic stack after the decode memory-accounting repair, recorded numeric envelope evidence, and handed the final tip to independent re-review. |
| 2026-08-22 | Ran the fresh post-repair Claude review for the exact 20-minute limit and archived its timeout without verdict. |
| 2026-08-22 | Completed, verified, committed, and pushed the two maintainer-approved focused repairs on PR #20. |
| 2026-08-22 | Resumed the Claude PR #20 review, captured its REQUEST CHANGES verdict, and independently reproduced both P1 findings. |
| 2026-08-22 | Ran the requested 20-minute Claude Opus review attempt for PR #20 and recorded its timeout before verdict. |
| 2026-08-22 | Implemented and verified the maintainer-selected default exact-row persistence policy on PR #20. |
| 2026-08-22 | Restored the remote two-PR state, ran the PR #20 focused matrix, and isolated the remaining exact-row producer gap. |
| 2026-08-21 | Re-ran the committed-HEAD matrix, direct probe, and PDD case; recorded fresh numeric metrics and output paths. |
| 2026-08-21 | Stabilized the archive verification wording so commit-count changes do not stale the record. |
| 2026-08-21 | Recorded post-commit branch, diff, path, and main-checkout verification. |
| 2026-08-21 | Completed the closing staged checks and refreshed the cross-wave regression evidence. |
| 2026-08-21 | Completed Wave E direct predictor/PDD verification, corrected the reproducibility probe, and recorded residual metadata/data gates. |
| 2026-08-21 | Implemented and verified option A: all runtime-legal MoE EP divisors now drive the default helper and omitted CLI domain. |
| 2026-08-21 | Completed the MoE routing/load contract slice with RED/GREEN, numeric probes, and an open EP-domain decision record. |
| 2026-08-21 | Completed the Wave D mixed-Attention sampling slice with RED/GREEN, deterministic probes, and a 49-test profiling regression matrix. |
| 2026-08-21 | Completed the first Wave D sampling-domain endpoint slice; recorded RED/GREEN, regression, and numeric superset evidence. |
| 2026-08-21 | Completed Wave C TP consumer migration through the unified registry and recorded the post-commit regression matrix. |
| 2026-08-21 | Completed the Wave C profiling-plan catalog cleanup and committed its focused regression evidence. |
| 2026-08-21 | Completed staged import-boundary verification and extracted the pure MoE feature object into runtime ownership; registry consumer migration is next. |
| 2026-08-21 | Completed MoE finite lookup migration and removed grouped-GEMM upper-token clamping; recorded RED/GREEN evidence. |
| 2026-08-21 | Completed finite Attention and speculative consumer migration; recorded direct-index audit. |
| 2026-08-21 | Moved finite model schema validation before exact lookup after an exact-hit RED test. |
| 2026-08-21 | Completed A2 finite one-feature resolver and consumer migration; recorded RED/GREEN evidence. |
| 2026-08-21 | Recorded A1 lookup-contract verification, optional metadata limits, and the A2 handoff. |
| 2026-08-21 | Started implementation after maintainer approval; recorded A1 binding RED/GREEN evidence. |
| 2026-08-21 | Started the new scoped task, verified the worktree and baseline, and created the analysis documents. |
| 2026-08-21 | Added the maintainer's dependency and adapter concerns to the active analysis record. |
| 2026-08-21 | Recorded the accepted staged boundary and `operator_query_binding` naming decision; froze the planning file map. |
| 2026-08-21 | Completed the function-level lookup/TP/sampling audit and narrowed the implementation handoff to named methods. |

# Progress Log

## 2026-08-21 - Task continuation and isolation

- **Status:** Complete.
- **Motivation:** Continue from the old PR #4 review without importing its
  scope-crept implementation.
- **Expectation:** A dedicated worktree based on current `origin/main`, with no
  source changes in the main checkout.
- **Method:** Verified `git worktree list`, branch `refactor/pr4-scoped-lookup-boundary-20260821`,
  and HEAD `9e9fa94c`; inspected the old task archive and old PR commit.
- **Result:** Worktree is isolated and clean before task-document creation.

## 2026-08-21 - Baseline verification

- **Status:** PASS.
- **Motivation:** Establish that the current base's registry, predictor budget,
  and profiling metadata tests are green before planning changes.
- **Expectation:** No pre-existing failure in the focused baseline.
- **Method:** Ran the exact command recorded in
  `test_report_2026-08-21_baseline.md`.
- **Result:** `37 passed in 6.26s`; no production/test files changed.

## 2026-08-21 - Current-main evidence inventory

- **Status:** Complete.
- **Motivation:** Avoid carrying old-branch line references or resolved N3/N4/N5
  assumptions into the new plan.
- **Expectation:** Identify reachable direct lookup, profiling import, prefix
  classification, registry, and sampling sites on `origin/main`.
- **Method:** Read-only `rg`/`nl` inspection of predictor, manager, trainer,
  registry, scheduler, and profiling modules; inspected legacy PR commit with
  `git show`.
- **Result:** Evidence is recorded in `findings.md`; no source was edited.

## 2026-08-21 - Planning document creation

- **Status:** Complete for the analysis phase.
- **Motivation:** Make the new PR's scope, acceptance gates, exclusions, and
  decision dependency auditable before implementation.
- **Expectation:** Required task-memory taxonomy exists and raw requirements are
  separated from design choices.
- **Method:** Added `requirements.md`, `plan.md`, `findings.md`, `design.md`,
  `issues.md`, `notes.md`, `harness.md`, and supporting reports in this task
  directory.
- **Result:** Documentation is present; implementation remained intentionally
  paused at that earlier checkpoint before the dependency and naming decisions
  were split and confirmed.

## 2026-08-21 - Dependency and adapter re-evaluation

- **Status:** Complete for this analysis checkpoint.
- **Motivation:** The maintainer challenged the assumption that all
  `frontier.profiling` imports must be removed and questioned the value of a new
  profiling-facing adapter.
- **Expectation:** Separate easy runtime-only migrations from shared CSV
  helpers, benchmark implementations, and explicit exceptions; verify whether
  the existing unified registry already covers the proposed adapter contract.
- **Method:** Counted direct Python imports and consumer modules; inspected
  `OperatorSpec`, family registries, attention profiling mappings, MoE feature
  construction, MTP structural config, and non-KV memory estimators.
- **Result:** `frontier/training` has no direct profiling implementation import;
  CPU/PP schema validators are shared and non-trivial to relocate; the MoE
  feature object is the only small extraction candidate; MTP and non-KV memory
  are large/named exceptions. Existing registry fields and resolvers make an
  independent adapter unnecessary by default. Updated `findings.md`,
  `design.md`, `issues.md`, and `plan.md`; no production or test source was
  changed.

## 2026-08-21 - Historical adapter proposal cross-check

- **Status:** Complete.
- **Motivation:** Ensure the adapter recommendation was not being inherited
  from the old scope-crept branch without checking its ownership boundary.
- **Expectation:** Distinguish current-main requirements from R-025/R-026
  artifacts introduced only by the previous continuation.
- **Method:** Read the old descriptor design and registry recheck, then checked
  whether `frontier/profiling/operator_identity.py` and the related cache
  contract exist on current `origin/main`.
- **Result:** The old adapter proposal bridges a D24/N5 catalog absent from
  current main. Its unknown-name evidence remains a reason for exact admission,
  but not a reason to import the second catalog into this PR. Updated
  `findings.md`, `design.md`, and `issues.md`; no source was changed.

## 2026-08-21 - Fresh analysis-checkpoint verification

- **Status:** PASS for the current-main baseline only.
- **Motivation:** Apply the verification-before-completion gate before
  reporting the analysis checkpoint and ensure documentation edits did not
  alter production or test source.
- **Expectation:** The recorded focused baseline remains green and the worktree
  has no tracked changes under `frontier/` or `tests/`.
- **Method:** Re-ran the exact focused pytest command with the pinned Python
  interpreter and checked `git diff --name-only` for production/test paths.
- **Result:** `37 passed in 23.64s`, `0 failed`, `0 skipped`; tracked
  production/test diff count `0`. This is baseline evidence only, not proof
  that the planned lookup or boundary fixes are implemented.

## 2026-08-21 - Final checkpoint verification refresh

- **Status:** PASS for the current-main baseline only.
- **Motivation:** Refresh the evidence after splitting the grill-me decisions
  and finalizing the task documents.
- **Expectation:** Focused tests and static task gates remain green with no
  production/test source changes.
- **Method:** Re-ran the same four-test focused command and the task-document,
  whitespace, and tracked-diff checks.
- **Result:** `37 passed in 2.91s`, `0 failed`, `0 skipped`; history/static
  checks passed and production/test tracked diff count remained `0`.

## 2026-08-21 - Operator query binding decision and plan freeze

- **Status:** Complete for the naming and architecture seam; implementation
  remains paused.
- **Motivation:** The maintainer confirmed the staged dependency boundary and
  selected option A for the registry-facing operator component.
- **Expectation:** Replace the vague `exact resolver` label with a domain name,
  preserve the existing unified registry, and make the first implementation
  file ownership auditable without adding a profiling adapter.
- **Method:** Recorded `operator_query_binding` / `bind_operator_query` in
  `requirements.md`, `design.md`, `issues.md`, `plan.md`, `review.md`, and
  `harness.md`; added a bounded candidate-file map and final pre-edit gates.
- **Result:** Dependency and naming decisions are marked complete; the plan is
  ready for the final maintainer review. No production or test source changed.

## 2026-08-21 - Function-level source audit and deep-module review

- **Status:** Complete for the analysis phase; final maintainer gate remains.
- **Motivation:** The broad candidate file map still left a risk that a later
  implementation would silently widen the lookup or registry slice.
- **Expectation:** Name every direct lookup/classification/sampling function,
  distinguish owner-specific exceptions, and verify that `bind_operator_query`
  has a small interface with enough behavior behind it to justify a seam.
- **Method:** Inspected the predictor, MoE predictor, shared manager, trainers,
  attention TP policy, profiling plan, and sampler functions on current
  `origin/main`; applied the codebase-design deletion/depth test to the binding
  surface.
- **Result:** Added a function-level ownership table to `findings.md`, narrowed
  `plan.md` waves/file map, recorded `_round_to_valid_key` as a prohibited
  silent clamp, and documented the binding interface invariants. No production
  or test source changed.

## 2026-08-21 - Final planning-checkpoint verification

- **Status:** PASS for the analysis checkpoint; implementation remains gated.
- **Motivation:** Apply the verification-before-completion rule after the
  final documentation edits and distinguish baseline evidence from future
  implementation acceptance.
- **Expectation:** The focused current-main baseline remains green, task docs
  retain the required history section, whitespace is clean, and no production
  or test source has changed.
- **Method:** Ran the pinned Python/pytest command recorded in
  `test_report_2026-08-21_planning_checkpoint.md`, `git diff --check -- frontier tests`,
  the task-document history scan, and the tracked-source diff check.
- **Result:** `37 passed in 2.91s`, `0 failed`, `0 skipped`; history scan passed
  for `14` Markdown files; `git diff --check` passed; tracked source diff count
  was `0`. The remaining prefix/list matches are intentionally unimplemented
  targets documented in `findings.md`, not a claim that the future refactor is
  complete.

## 2026-08-21 - A1 registry binding admission

- **Status:** GREEN; lookup contract in progress.
- **Motivation:** Make ordinary runtime operator admission exact and registry-driven before routing any predictor query.
- **Expectation:** Physical names and unique profiling aliases resolve through the unified registry; unknown names, family mismatches, alias collisions, and disabled families fail before runtime dispatch.
- **Method:** Added `OperatorQueryBinding` and `bind_operator_query` to `frontier/operators/binding.py`, exported the API, and added six focused tests in `tests/unit/test_operator_registry.py`.
- **Result:** RED first produced `ImportError: cannot import name 'bind_operator_query'`; the minimal implementation then passed `6 passed, 9 deselected in 0.24s`. Registry regression passed with `26 passed in 0.28s`. No CSV, sklearn, cache, or profiling-runner dependency was introduced.

## 2026-08-21 - A1 validated on-demand lookup contract

- **Status:** Complete for A1; A2 finite-consumer migration pending.
- **Motivation:** The old path checked a sorted caller-key cache before model
  metadata, treated a missing exact row as unconditional permission to predict,
  wrote measured values into the runtime cache, and clamped invalid estimator
  output. These behaviors make a legal coverage gap indistinguishable from an
  identity or schema defect.
- **Expectation:** Validate the model-bound schema and physical inputs before
  either cache tier; prefer an exact measured value, then a validated
  process-local result, then one canonical model call; reject malformed output
  and keep measured metadata immutable.
- **Method:** Added the narrow validation/resolution path in
  `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`,
  passed exact lookup metadata to ordinary multi-feature, attention mixed, and
  MLA descriptors, and added
  `tests/unit/test_on_demand_prediction_contract.py`. Bounds, relational
  constraints, and identity checks accept only explicit metadata; current-main
  producers do not emit those optional fields yet.
- **Result:** The fresh focused contract run passed `76` tests; the MLA/manager
  and attention-cache run passed `49` tests; the combined matrix passed `111`
  tests and failed `2` logger-capture assertions that also fail on clean
  `origin/main`. The direct runtime values in those failures were
  `4.610000 ms` and `6.610000 ms`, while `caplog.records` was empty because the
  Frontier root logger disables propagation. Py-compile and `git diff --check`
  passed. Full commands and environment are in the A1 test report.

## Pending next action

Proceed to the staged profiling/runtime import-boundary audit and registry
consumer migration. Keep optional CPU/PP zero behavior separate and do not
make unpublished domain metadata mandatory until a real producer is
implemented.

## 2026-08-21 - Wave D MoE routing and load contract

- **Status:** Complete for routing identity and distribution edge cases; joint
  TP/EP sampling-domain coverage remains pending a maintainer decision.
- **Motivation:** `MoEWrapper._prepare_routing_inputs()` generated `topk_ids`
  independently of caller-supplied `expert_token_counts`, allowing the
  shuffling/GEMM workload and the CSV feature label to describe different
  routes. The `skewed` sampler made expert `0` unreachable, and
  `extremely_skewed` collapsed to all-expert sampling when `router_topk` was
  wider than its hot subset.
- **Expectation:** Derive one authoritative local count vector from the
  generated route, require explicit counts to be an exact validated match,
  keep every expert reachable in `skewed`, and preserve a hot subset large
  enough for the requested `top_k`.
- **Method:** Added the smallest RED tests in
  `tests/unit/test_moe_routing_input_contract.py`; validated iterable shape,
  boolean/integer/non-negative values (using `operator.index`) and exact equality in
  `frontier/profiling/moe/moe_wrapper.py`; adjusted the two load-distribution
  formulas in `frontier/profiling/moe/load_distribution.py`.
- **Result:** The first RED caught an integral float accepted as an integer;
  GREEN then passed `5` focused tests. The affected regression matrix passed
  `85` tests. With `num_experts=16`, `top_k=8`, `num_tokens=512`,
  `seed=42`, `extremely_skewed` produced `423/512` all-hot rows, count sum
  `4096 = 512*8`, active experts `16`, CV `0.8284853744919523`, and Gini
  `0.420654296875`; `skewed` produced count sum `4096`, active experts `16`,
  CV `0.25803804646379486`, Gini `0.142791748046875`, and expert-0 count
  `109`. Py-compile and whitespace checks passed.

## 2026-08-21 - MoE EP sampling-domain audit

- **Status:** Open design gate; no production change made.
- **Motivation:** The runtime accepts any positive EP divisor of
  `total_expert_num`, while the profiling helper enumerates only divisors
  `[1, 2, 4, 8]` and the CLI defaults to `[1]`.
- **Expectation:** Establish whether all runtime-legal EP divisors are part of
  the supported profiling domain before changing either side.
- **Method:** Probed `get_default_moe_profiling_config(num_experts=60,
  router_topk=4)`, the CLI declaration in `frontier/profiling/moe/main.py`,
  `ReplicaConfig.__post_init__`, and the canonical qwen2 CSV without editing
  data.
- **Result:** The helper returns local widths `[60, 30, 15]`, representing EP
  `{1, 2, 4}`; runtime-legal divisors are
  `{1,2,3,4,5,6,10,12,15,20,30,60}`; the CLI default is `[1]`; and
  `data/profiling/compute/a100/qwen2_moe_example/moe.csv` has `524` rows with
  EP `{1,2}` only. This proves an inconsistent potential envelope, but not
  which EP set the product intends to support.

## 2026-08-21 - A2 finite one-feature lookup and consumer migration

- **Status:** Complete for this sub-step; Attention multi-feature paths pending.
- **Motivation:** Finite one-feature consumers treated materialized prediction
  rows as the complete runtime domain. A legal key outside that table raised a
  coverage error even though the trained estimator was available, and direct
  indexing bypassed runtime-cache validation.
- **Expectation:** Resolve exact measured row first, then a validated
  process-local model result, then one canonical estimator call; reject
  malformed values without nearest-row substitution or clamping.
- **Method:** Added `_get_prediction_for_features(...)` to the existing
  predictor module and routed ordinary Linear/FFN, architecture-profile
  one-feature, Rope, finite KV-cache-save, and named-operator consumers through
  it. Added `tests/unit/test_finite_prediction_lookup.py` with RED coverage for
  miss fallback, repeated cache hits, nearest-row rejection, and invalid values.
- **Result:** RED produced `6 failed` at the old key-miss branch. After the
  exact-hit schema-order correction, focused verification passed `7`
  finite-lookup tests and `72` Linear/FFN/effective-token/on-demand
  regressions; additional manager/speculative/max-token coverage passed `46`
  tests. Model fallback returned `4.5` once, repeated query hit `{(2.0,): 4.5}`,
  and invalid cache/model values failed before cache writes. Full details are
  in `test_report_2026-08-21_a2_finite_lookup.md`.

## 2026-08-21 - A2 exact-hit schema ordering correction

- **Status:** Complete.
- **Motivation:** A finite exact row could be returned before the fallback
  estimator's declared feature schema was checked, allowing an identity/schema
  mismatch to bypass the common pre-cache validation rule.
- **Expectation:** A model feature-name or feature-count mismatch fails before
  exact lookup, runtime-cache lookup, or model invocation.
- **Method:** Added an exact-hit schema-mismatch regression to
  `tests/unit/test_finite_prediction_lookup.py` and moved the existing model
  schema/count checks ahead of finite-table lookup in
  `_get_prediction_for_features(...)`.
- **Result:** The new RED failed because the old helper returned the exact row;
  after the move, the focused finite suite passed `7` tests and the combined
  finite/on-demand/FFN/effective-token suite passed `72` tests. The mismatch
  case records `model.calls == 0`.

## 2026-08-21 - A2 Attention and speculative finite consumers

- **Status:** Complete for dense Attention; MoE finite paths pending.
- **Motivation:** Dense Attention decode/prefill and speculative finite branches
  still indexed prediction tables directly, so legal two-feature runtime keys
  outside the materialized rows raised `KeyError` and skipped the trained model
  and runtime cache.
- **Expectation:** Preserve all batching, phase, and calibration semantics while
  routing decode, prefill, speculative verify, and speculative normal-decode
  queries through exact-row -> runtime-cache -> model resolution.
- **Method:** Added RED tests with real `Batch` and `SpecDecodeBatchMetadata`
  objects, then replaced the remaining dense predictor table indexing with
  `_get_prediction_for_features(...)` and explicit feature order.
- **Result:** RED reproduced two Attention `KeyError` failures and two
  speculative finite coverage failures. GREEN passed `9` finite Attention tests,
  `11` spec-path tests, and `25` combined finite/Attention/max-token regressions;
  the dense predictor now has no direct prediction-table indexing. Model calls
  were exactly `1` per missing key and repeated queries hit the process-local
cache. Evidence is in `test_report_2026-08-21_a2_finite_lookup.md`.

## 2026-08-21 - Wave B staged import boundary

- **Status:** Complete for the predictor/training boundary; named non-KV-memory
  and MTP structural-config exceptions remain explicit.
- **Motivation:** The runtime MoE predictor imported a pure feature dataclass
  from a mixed profiling input module, while the agreed boundary permits
  shared CSV validators but keeps benchmark/GPU implementation producer-only.
- **Expectation:** Give the pure feature derivation one runtime-owned
  definition, preserve the profiling import path, and enforce an exact import
  allowlist without moving shared CSV rules or adding an adapter layer.
- **Method:** Added `frontier/moe_load_imbalance.py`, changed the runtime
  consumer to import it, and made `frontier/profiling/moe/moe_input.py` a
  compatibility re-export plus sampling configuration. Added the AST-based
  boundary test, observed its two-test RED failure, then ran the focused and
  compatibility regressions.
- **Result:** Boundary tests passed `3`; runtime/MoE regressions passed `66`;
  profiling compatibility tests passed `39`. Class identity remained true and
  identical allocation produced `total_routed_tokens=24`,
  `load_imbalance_cv=0.0`, and `load_gini_coefficient=0.0` on both import
  paths. Full evidence is in
  `test_report_2026-08-21_import_boundary.md`.

## 2026-08-21 - A2 MoE finite consumers

- **Status:** Complete for MoE finite lookup; profiling/runtime boundary audit pending.
- **Motivation:** MoE gating, legacy shuffling, and standard grouped-GEMM
  still treated the finite materialized table as the complete runtime domain;
  grouped-GEMM additionally clamped queries above `_max_tokens` to the last
  profiled row.
- **Expectation:** Route all finite MoE queries through the validated exact-row
  -> process-local runtime-cache -> model fallback path, preserve existing
  calibration/zero-token behavior, and pass the original grouped-GEMM token
  count to the model when the table has no exact row.
- **Method:** Added
  `tests/unit/test_moe_finite_prediction_lookup.py`, observed the five-test RED
  failure, migrated the four finite owner branches in
  `sklearn_moe_execution_time_predictor.py`, and removed `_round_to_valid_key`
  plus its direct table helper.
- **Result:** Focused MoE tests passed `5`; MoE/effective-token regressions
  passed `58`; shared finite/on-demand regressions passed `26`. Gating and
  shuffling each returned `4.5` with one model call and cache key `(2.0,)`.
  Grouped-GEMM queries `6` above `_max_tokens=4` returned model values `6.25`
  and `7.0` for direct/allocation inputs, with the model seeing `6.0` and no
  upper-row substitution. Full evidence is in
  `test_report_2026-08-21_moe_finite_lookup.md`.

## 2026-08-21 - Wave C profiling-plan catalog cleanup

- **Status:** Complete for this sub-step; remaining Wave C TP consumers are
  still in progress.
- **Motivation:** `frontier/profiling/linear_op/profiling_plan.py` exposed five
  historical attention lists beside the model architecture registry. This
  duplicated operator ownership and allowed future additions to drift between
  profiling and runtime consumers.
- **Expectation:** Remove the duplicate lists, derive ordinary attention names
  from `ModelArchitectureProfile.linear_attention`, derive memory names from
  `MEMORY_FAMILY`, preserve the existing replicated output order, and keep the
  MTP structural registry exception unchanged.
- **Method:** Added RED assertions for the five public constants, replaced the
  profiling-plan references with registry-derived sequences, ran the focused
  plan/architecture tests and the Wave C regression matrix, then committed the
  two-file slice as `809531e8`.
- **Result:** RED failed on the still-exported `ATTN_BASE_OPS`; GREEN passed
  `3` targeted tests and `136` tests in the combined matrix. Dense plan output
  retained `input_layernorm`, `post_attention_layernorm`, `add`, `emb` before
  generic attention and FFN names. No canonical data, README, or main checkout
  changed.

## 2026-08-21 - Wave C TP consumer migration

- **Status:** Complete for Wave C; Wave D profiling-domain coverage is pending.
- **Motivation:** Four runtime/training TP consumers still inferred operator
  ownership from local replicated sets or `attn_`/`mlp_` name prefixes. This
  allowed undeclared names to enter attention/FFN TP policy and required a
  consumer edit for each new operator.
- **Expectation:** Resolve generic operators through `bind_operator_query` and
  resolve architecture-specific linear attention names through the explicit
  architecture profile; preserve MTP structural handling and effective TP
  validation; reject unknown names and conflicting declarations before policy
  dispatch.
- **Method:** Added `resolve_operator_query_tp_mode` to the existing binding
  module, migrated the predictor, shared manager, and two trainers, derived
  `ATTENTION_LINEAR_OPS` from the architecture registry, and added focused
  consumer tests. The RED tests covered undeclared `attn_*`/`mlp_*` names and
  many-to-one `add` alias handling before the migration. Committed the slice as
  `4c03b243`.
- **Result:** The post-commit matrix passed `235` tests in `8.50s`; the direct
  conflict case rejected a profile declaring one operator as both replicated
  and sharded with an explicit `ValueError`. No prefix-based TP branches remain
  in the migrated consumers, and no profiling adapter or CSV dependency was
  introduced.

## 2026-08-21 - Wave D sampling-domain endpoint slice

- **Status:** Complete for primitive endpoint coverage; joint Attention/MoE
  envelope planning remains in progress.
- **Motivation:** The profiling grids stopped below legal runtime maxima when
  a maximum was not a canonical grid anchor. Explicit `extra_num_tokens` above
  the default grid was also silently discarded, making a requested profiling
  extension disappear before measurement.
- **Expectation:** Preserve deterministic bounded grids while including the
  requested upper endpoint for shared token, sequence, prefill-chunk, and
  attention batch axes. Preserve explicit decode upper-bound rejection. Keep
  MoE sampling configuration in profiling ownership and include its configured
  token endpoint as well.
- **Method:** Added
  `tests/unit/test_profiling_sampling_domain.py` first and verified RED (`4`
  endpoint failures, then `2` additional endpoint failures). Updated only
  `frontier/profiling/utils/__init__.py` and
  `frontier/profiling/moe/moe_input.py`: union explicit token extensions,
  append positive requested endpoints, and deduplicate/sort the affected
  grids. No runtime module, CSV schema, or benchmark wrapper changed.
- **Result:** The focused sampling suite passed `10` tests; the existing
  profiling-plan/metadata/attention-boundary matrix passed `35` tests. Direct
  probes reported the following maxima: for `runtime_max=5000`, shared tokens,
  sequence lengths, and prefill chunks each reached `5000`; attention decode
  batch `max_batch_size=130` reached `130`; the independent MoE config reached
  `5000` instead of the old `4992`. `python -m py_compile` and
  `git diff --check` passed.

## 2026-08-21 - Wave D mixed-Attention envelope slice

- **Status:** Complete for the three existing mixed-Attention generators;
  MoE routing, expert-load, and joint TP/EP envelope coverage remain pending.
- **Motivation:** The legacy mixed-prefill generator filtered high KV values
  because its smallest sequence anchor was `64`, emitted duplicate structural
  workloads from overlapping ranges, and omitted a non-canonical context
  endpoint. The online-grid generator replaced configured ranges whenever an
  explicit list was supplied, contrary to the accepted D24-7 union rule. The
  true-mixed generator accepted only fixed CLI lists and therefore missed the
  legal prefill and decode upper endpoints.
- **Expectation:** Every requested mixed mode retains a deterministic valid
  context-boundary workload; explicit online axes extend rather than replace
  configured ranges; true-mixed prefill reaches
  `max_seq_len - prefill_kv_cache_size` and decode KV reaches
  `max_seq_len - 1`; no duplicate structural workload reaches measurement.
- **Method:** Added four regression tests before changing production code and
  observed RED (`4 failed, 10 passed`). Updated only the existing sampling
  utility and its CLI help: deduplicated legacy workloads, added per-KV legal
  boundary shapes, unioned online explicit lists, and unioned true-mixed
  endpoints. Re-ran focused tests, deterministic direct probes, and the
  affected profiling matrix.
- **Result:** Focused sampling tests passed `14`; the expanded profiling
  regression matrix passed `49` in `6.22s`; an additional Attention/profiling
  compatibility matrix passed `53`; the cross-wave registry/predictor matrix
  passed `249` in `15.08s`. Direct probes showed legacy
  `even/random/both` counts `198/63/261`, unique counts equal to emitted
  counts, and `max_context=1000` with `kv_max=992`; online explicit/default
  axes produced all `9` points from the `3x3` union; true-mixed at
  `max_seq_len=5000` reached prefill `5000` and decode KV `4999`, with every
  row valid. `py_compile` and `git diff --check` passed.

## 2026-08-21 - Wave D MoE EP domain resolver

- **Status:** Complete for the EP policy and CLI/helper derivation; canonical
  CSV regeneration and Wave E direct simulator cases remain pending.
- **Motivation:** Runtime accepted every positive divisor of `total_expert_num`,
  but the helper and CLI exposed narrower, unrelated EP lists. The maintainer
  selected option A so a legal runtime EP cannot be excluded by default
  sampling policy.
- **Expectation:** Return every positive divisor deterministically, map each to
  its local expert width, resolve omitted CLI EP selections per model, reject
  explicit non-divisors before profiling, and preserve explicit smoke-script
  selections.
- **Method:** Added RED tests in
  `tests/unit/test_moe_ep_sampling_domain.py`; implemented the pure resolver
  and default helper update in `frontier/profiling/moe/moe_input.py`; changed
  `frontier/profiling/moe/main.py` to resolve domains per model, use resolved
  counts for the progress bar/confirmation, and record them in
  `moe_config.yaml`.
- **Result:** Focused GREEN passed `6` tests. The combined EP, sampling,
  import-boundary, MoE family, and finite-lookup matrix passed `50` tests.
  Direct probes returned EP `{1,2,3,4,5,6,10,12,15,20,30,60}` for 60 experts
  and local widths `[60,30,20,15,12,10,6,5,4,3,2,1]`; explicit `[20,2,20,1]`
  resolved to `[20,2,1]`; non-divisor `7` failed with the legal-domain error.
  The profiling example dry-run retained its explicit EP=1 smoke command.

## 2026-08-21 - Wave E direct verification and report repair

- **Status:** Complete for the scoped code slice; residual producer metadata
  and canonical CSV work remain open by design.
- **Motivation:** Close the direct-verification wave with numeric evidence for
  the ordinary finite lookup paths and prove that heterogeneous PDD roles do
  not collapse into one predictor object.
- **Expectation:** A legal miss calls the model once and then hits the runtime
  cache; Attention preserves feature order; MoE grouped-GEMM forwards the
  original in-flight token count; exact measured data outranks a stale cache;
  invalid features fail before model access; PDD creates distinct PREFILL and
  DECODE predictors with a shared model manager and completes KV transfer.
- **Method:** Ran a complete inline Python probe using the focused test
  builders, the 111-test focused matrix, the broader 306-test cross-wave
  matrix (with the earlier narrower 265-test matrix retained historically), a
  simulator-construction identity probe, and the sequential PDD example. All
  temporary bytecode and outputs were written under `/data/ycfeng/tmp`.
- **Result:** The direct probe printed `direct_probe PASS`; Linear returned
  `4.5` twice with `1` model call, Attention decode returned `4.25` twice with
  `1` call, Attention prefill returned `5.5` twice with `1` call, MoE returned
  `6.25` twice while the model saw token count `6.0`, exact measured data
  returned `2.25` with `0` calls over a stale cache value `9.0`, and NaN was
  rejected with `0` calls. The PDD run exited `0`, completed `2/2` requests,
  processed `24` tokens, and recorded `2` KV transfers totaling `8,388,608`
  bytes. The report is
  `test_report_2026-08-21_wave_e_direct_cases.md`.
- **Error record:** The first probe draft passed a two-feature query to a
  builder that declared only `num_tokens`, producing the expected
  `ValueError: unexpected features`. The script was corrected to declare
  `("batch_size", "num_tokens")`; the corrected command passed without a
  production change. An earlier aggregate pytest attempt omitted
  `PYTHONPATH=$PWD` and failed collection with ten `ModuleNotFoundError`s;
  the documented environment rerun passed `111` and `265` tests.

## 2026-08-21 - Closing verification refresh

- **Status:** PASS for the current HEAD; post-commit verification is recorded
  below.
- **Motivation:** Apply a fresh completion gate after re-staging the final
  task-memory updates, so the reported regression evidence matches this exact
  worktree state.
- **Expectation:** The complete cross-wave matrix remains green, staged
  whitespace and path audits remain clean, and every task Markdown file keeps
  its required modification-history header.
- **Method:** Ran the 26-file cross-wave pytest command with
  `PYTHONPATH=$PWD`, `-p no:cacheprovider`, and bytecode redirected under
  `/data/ycfeng/tmp`; ran `git diff --cached --check`, the 25-file history
  scan, and the README/canonical-data path audit.
- **Result:** `306 passed in 19.57s`; staged whitespace, history, and path
  audits all passed. The exact command and output are recorded in
  `test_report_2026-08-21_wave_e_direct_cases.md`.

## 2026-08-21 - Post-commit verification

- **Status:** PASS.
- **Motivation:** Verify the committed archive rather than relying only on the
  pre-commit index state.
- **Expectation:** The dedicated branch is clean, the complete branch diff has
  no whitespace errors or forbidden README/canonical-data paths, and the main
  checkout remains untouched by this task.
- **Method:** Checked `git status --short --branch`, `git diff origin/main...HEAD
  --check`, the branch file/stat summary, the forbidden-path audit, and the
  main checkout status at `/data/ycfeng/stepfun-performance-optimization/Frontier`.
- **Result:** The worktree is clean and ahead of `origin/main`; the full diff
  whitespace and forbidden-path audits passed. The main checkout
  still has its pre-existing `?? =10.1`, `tests/scratch/`, and
  `tests/unit/test_moe_ep_non_dummy_matrix.py` entries and no task changes were
  made there.

## 2026-08-21 - Fresh committed-HEAD rerun

- **Status:** PASS.
- **Motivation:** Re-establish completion evidence after resuming the task so
  the final report is based on commands executed in this continuation.
- **Expectation:** The complete regression matrix, direct lookup contract, and
  sequential PDD lifecycle remain green without changing the worktree or
  canonical profiling data.
- **Method:** Ran the documented 26-file pytest matrix with
  `PYTHONPATH=$PWD`, `-p no:cacheprovider`, and bytecode under
  `/data/ycfeng/tmp`; ran the inline Linear/Attention/MoE probe; ran the
  sequential PDD example with `NUM_REQUESTS=2`, `PREFILL_TOKENS=8`,
  `DECODE_TOKENS=4`, and `DECODE_CUDA_GRAPH_MODE=none`; parsed the generated
  JSON/CSV metrics with structured readers.
- **Result:** The matrix passed `306` tests in `19.79s`. The probe returned
  Linear `4.5` twice with one model call, Attention decode `4.25` twice with
  one call, Attention prefill `5.5` twice with one call, MoE grouped-GEMM
  `6.25` twice while the model saw `6.0`, exact measured `2.25` with zero
  model calls over stale cache `9.0`, and a NaN `ValueError` before model
  access. PDD exited `0`; `request_metrics.csv` contained `2` rows and `2`
  unique requests, each with `8` prefill plus `4` decode tokens; fresh system
  metrics reported mean TTFT `358.0 ms`, mean TPOT
  `357.99999999999994 ms`, mean request E2E `1790.66777216 ms`, `2` KV
  transfers, `8,388,608` bytes, and `1.3355443200000483 ms` total transfer
  time. Fresh artifacts are under
  `/data/ycfeng/tmp/pr4_completion_pdd_20260821/meta_llama_llama_2_7b_hf/offline_batch/completion_pdd_20260821/`.

## 2026-08-22 - Remote two-PR context restoration

- **Status:** Complete.
- **Motivation:** Resume from the actual remote stack rather than the archived
  monolithic branch.
- **Expectation:** Identify the two PR numbers, current heads, stack order,
  review/check state, and merge state while preserving the maintainer's merge
  gate.
- **Method:** Bootstrapped the company proxy with
  `eval "$(curl -fsS http://deploy.i.shaipower.com/httpproxy)"`, fetched
  `origin`, and queried GitHub with `gh pr list`, `gh pr view`, and the review
  comment/review APIs.
- **Result:** PR #20 points to `53499911` against `main`; PR #21 points to
  `03952bf1` against the PR #20 branch. Both report `OPEN`, `MERGEABLE`, and
  `CLEAN`; neither has a review decision, review record, inline comment, issue
  comment, or CI status rollup. No merge action was performed.

## 2026-08-22 - PR #20 focused verification refresh

- **Status:** Known-base failure isolated; product-focused tests passed.
- **Motivation:** Verify the three post-split exact-lookup persistence commits
  on the actual PR #20 tip before continuing the stack.
- **Expectation:** Lookup, registry, finite fallback, persistence, import
  boundary, and affected predictor tests pass; the previously recorded logger
  capture defect remains separately attributable.
- **Method:** Ran 12 affected unit-test files with the pinned Python 3.13.13
  environment, `PYTHONPATH=$PWD`, disabled pytest cache, and bytecode under
  `/data/ycfeng/tmp`.
- **Result:** `134 passed`, with the same two pre-existing
  `test_attention_trace_total_includes_decode` caplog failures. Runtime stdout
  recorded the correct totals `4.610000 ms` and `6.610000 ms`; the failure
  remains `caplog.records == []` under `frontier.logger` propagation behavior.
  Log: `/data/ycfeng/tmp/pr4_pr20_focused_20260822.log`.

## 2026-08-22 - Finite exact-row producer audit

- **Status:** Open design gate; production code unchanged in this continuation.
- **Motivation:** The lookup resolver now prefers estimator-attached exact
  measured rows, and the latest PR #20 commits claim finite-producer
  preservation. The producer coverage must match that contract.
- **Expectation:** Every ordinary model routed through finite lookup either
  publishes `_frontier_exact_lookup` automatically or declares the exact
  finite-producer set in one auditable place.
- **Method:** Audited all `_train_model` and `_train_single_model` call sites,
  inspected finite prediction materialization, and ran a real two-row estimator
  training probe with and without `persist_exact_lookup=True`.
- **Result:** The default training path produced
  `default_has_exact_lookup=False`; the explicit path produced
  `{(1.0,): 9.0, (2.0,): 4.0}`. Ordinary Linear, standard Attention, and
  one-feature MoE call sites still omit the flag, so measured-row precedence is
  not yet producer-complete for those paths. Representative checked-in data
  makes the storage tradeoff small: 8-key Linear maps serialize to `176`
  bytes, 24-key decode Attention to `712` bytes, 10-key prefill Attention to
  `306` bytes, and 7-key MoE maps to `156` bytes.

## 2026-08-22 - PR #20 broad regression and PR #21 composition check

- **Status:** PASS for the checks independent of `SCOPE-014`.
- **Motivation:** Continue useful verification while the producer-policy gate
  waits for maintainer selection.
- **Expectation:** The wider PR-A matrix passes after excluding only the
  already isolated logger/caplog test, and the latest PR #20/PR #21 tips form a
  conflict-free stacked tree.
- **Method:** Ran the 23-file PR-A matrix with
  `-k 'not test_attention_trace_total_includes_decode'`; ran
  `git merge-tree --write-tree` between current PR #20 and PR #21 tips; checked
  the PR #21 diff paths against the latest base.
- **Result:** `286 passed, 2 deselected in 10.72s`. Git produced synthetic tree
  `a19525e00323edcdebecc88791c860f65d9764ea` without conflict. PR #21 remains
  a nine-file profiling-only diff against current PR #20. No rebase, push, or
  merge was performed. Full evidence is in
  `test_report_2026-08-22_pr20_remote_split_audit.md`.

## 2026-08-22 - SCOPE-014 universal exact-row persistence

- **Status:** Implemented, committed, and locally verified; remote push
  pending.
- **Motivation:** Ordinary Linear, standard Attention, and one-feature MoE
  models inherited `persist_exact_lookup=False`, so a model-cache round trip
  could replace a measured point with the estimator's regressed value.
- **Expectation:** Both unified runtime training/cache entry points persist
  scalar-numeric measured rows by default; exact measured data remains ahead
  of runtime cache and model fallback after pickle reload.
- **Method:** After the maintainer selected option A, changed the existing
  round-trip regression to omit the explicit flag and added the corresponding
  shared-manager regression. The RED run failed both tests with
  `AttributeError` for missing `_frontier_exact_lookup`. Changed only the two
  default values to `True`, reran focused and broader matrices, exercised a
  direct cache round trip, and committed the three-file sub-step as
  `d45d836c`.
- **Result:** GREEN passed both default round-trip tests. The focused matrix
  recorded `135 passed` plus the same two known `SCOPE-013` caplog failures;
  stdout remained `4.610000 ms` and `6.610000 ms`. The committed broader
  matrix passed `287` tests with those two cases deselected. The direct probe
  preserved `{(1.0,): 9.0, (2.0,): 4.0}`, returned exact `4.0` over stale
  runtime cache `99.0`, returned cached `8.25`, and returned model `7.5`
  twice with one model call. `py_compile`, whitespace, and forbidden-path
  checks passed. The commit was pushed through the configured HTTPS proxy,
  and PR #20's body was refreshed with these values.

## 2026-08-22 - Post-push remote and stacked-PR check

- **Status:** PASS; both PRs remain unmerged.
- **Motivation:** Confirm the outward-facing state after updating the base PR
  and ensure PR #21 still composes cleanly without refreshing its branch.
- **Expectation:** PR #20 points to `d45d836c`; PR #21 retains its profiling-
  only head and base relationship; GitHub and local merge analysis report no
  conflict.
- **Method:** Bootstrapped the company HTTPS proxy, pushed the ordinary
  fast-forward, updated PR #20's body, queried both PRs through `gh`, and ran
  `git merge-tree --write-tree` between the new base and PR #21 head.
- **Result:** PR #20 is `OPEN/CLEAN/MERGEABLE` at `d45d836c`; PR #21 is
  `OPEN/CLEAN/MERGEABLE` at `03952bf1` with PR #20's branch as its base.
  Neither PR has a review, comment, or CI status. The synthetic stacked tree is
  `2a1ed9c5bb1b1ed412f5cef7f3f8e7e262bb9506`; PR #21 retains exactly nine
  profiling-only diff paths. No merge, rebase, force-push, or PR #21 update
  occurred.

## 2026-08-22 - Ask Claude PR #20 review

- **Status:** Invocation completed at the hard timeout; review verdict
  unavailable.
- **Motivation:** Obtain an independent Claude review of the complete PR #20
  lookup/registry diff before maintainer merge approval.
- **Expectation:** Claude returns a severity-ranked verdict with concrete
  file/line evidence within `1200s`.
- **Method:** From the PR #20 worktree, ran
  `timeout 1200s stepcode claude --model 'claude-opus-5' --effort max -p --`
  against `origin/main...d45d836c`. The prompt required read-only review of
  the complete diff, requirements R-004/R-005/R-007/R-013/R-015/R-016,
  lookup precedence, cache round trips, registry ownership, boundary rules,
  reachable test gaps, and explicit separation of `SCOPE-013`.
- **Result:** Claude used the full `1200s`, made `55` successful read-only tool
  calls with `0` tool-call failures, and emitted no final answer before GNU
  `timeout` returned `124` and terminated the child (`143`). StepCode's later
  feedback-collector request returned `401 Invalid web access key`; log timing
  proves this happened after timeout handling. The exact prompt and raw output
  are stored in
  `.worktrees/pr4-lookup-registry-core-20260822/.omx/artifacts/ask-claude-pr20-review-20260822-122037.md`.
  Local CLI help confirms a follow-up can resume Claude conversation
  `40e27923-c316-4f00-806b-b5500569309c`. No code, remote PR, merge, rebase,
  or force-push action occurred.

## 2026-08-22 - Resumed Claude PR #20 verdict

- **Status:** Complete; Claude verdict is `REQUEST CHANGES`.
- **Motivation:** Finish the external review after the first exact 20-minute
  invocation exhausted its budget before synthesis.
- **Expectation:** Resume the same Claude evidence context, stop broad
  investigation, and obtain a severity-ranked merge-readiness conclusion
  without changing files or Git state.
- **Method:** Ran StepCode Claude Opus 5 with `effort=max`, `--resume
  40e27923-c316-4f00-806b-b5500569309c`, an outer `timeout 1200s`, and a
  bounded synthesis prompt. The prompt required findings first, exact
  file/line evidence, reachable impact, root-cause repair, separation from
  baseline `SCOPE-013`, and an explicit final verdict.
- **Result:** The resume exited `0` and returned `REQUEST CHANGES`. Claude
  reported two P1 findings: one-feature MoE explicitly disables exact-row
  persistence, and three target-embedded MTP TP-key consumers repeat a local
  membership set despite the existing MTP registry accessor. Four P2
  observations were classified as non-blocking. The artifact is
  `.worktrees/pr4-lookup-registry-core-20260822/.omx/artifacts/ask-claude-pr20-review-resume-20260822-130144.md`;
  the raw output is
  `/data/ycfeng/tmp/ask_claude_pr20_review_resume_raw_20260822-130144.log`.

## 2026-08-22 - Independent verification of Claude P1 findings

- **Status:** Both P1 findings confirmed; production and test sources
  unchanged pending maintainer authorization.
- **Motivation:** Treat external-review output as a hypothesis until the
  current PR #20 code and reachable behavior independently reproduce it.
- **Expectation:** P1-1 must show a one-feature MoE cache round trip losing a
  measured row only when the explicit persistence condition is false. P1-2
  must show that adding a target-embedded MTP operation to its declared
  registry fails to update all three TP consumers.
- **Method:** Inspected
  `shared_prediction_model_manager.py:1050`,
  `sklearn_moe_execution_time_predictor.py:966`,
  the three TP-key methods, and
  `spec_decode/mtp_registry.py:44-47,169-170`. Ran an isolated predictor and
  shared-manager pickle round trip under `/data/ycfeng/tmp`, then extended
  `_TARGET_EMBEDDED_MTP_LINEAR_OPS` in-process with
  `mtp_registry_extension_probe` and called predictor, shared-manager, and
  trainer TP resolution. Replayed the same command to confirm report
  reproducibility.
- **Result:** With persistence disabled, the one-feature model reloaded
  without `_frontier_exact_lookup` and returned stale value `99.0` at a row
  measured as `4.0`: absolute error `95.0`, relative error `2375%`. With
  persistence enabled, both cache entry points reloaded
  `{(1.0,): 9.0, (2.0,): 4.0}` and the predictor returned `4.0`. The MTP
  extension was present in the registry tuple, while predictor, shared
  manager, and trainer each raised `ValueError`; rejected consumers were
  `3/3`. Original logs:
  `/data/ycfeng/tmp/pr20_claude_review_p1_verify_20260822.log` and
  `/data/ycfeng/tmp/pr20_claude_review_p1_registry_verify_20260822.log`.
  Reproducibility log:
  `/data/ycfeng/tmp/pr20_claude_review_verification_replay_20260822_131557.log`.
  Full criteria and commands are in
  `test_report_2026-08-22_pr20_claude_review_verification.md`.

## 2026-08-22 - MTP repair-scope evidence audit

- **Status:** Complete for escalation; no MTP source edit made.
- **Motivation:** Separate the confirmed TP-classification defect from every
  other occurrence of the same operation labels before presenting a scope
  choice.
- **Expectation:** Each option must map to real files and materially different
  behavior in this repository.
- **Method:** Searched all code/test occurrences of `mtp_fusion_proj`,
  `lm_head_linear`, and `get_target_embedded_mtp_linear_ops`; read the
  surrounding predictor, shared-manager, trainer, quantization, profiling
  implementation, CSV-column, and runtime-policy paths; inspected Wave C
  commit `9c09ac74`.
- **Result:** The reproduced failure is confined to three TP-key consumers,
  and one existing accessor exactly represents their membership decision.
  Other occurrences include model enumeration, required CSV columns,
  quantization defaults, physical profiling modules/timers, constants, and
  draft-model runtime policies. A focused repair touches the three consumers
  plus `tests/unit/test_operator_query_tp_consumers.py`; a comprehensive MTP
  consolidation reaches additional shared roles and requires a separate
  cross-cutting audit. `SCOPE-016` records both options and recommends the
  focused repair.

## 2026-08-22 - Focused repair implementation and verification

- **Status:** Complete at pushed PR #20 HEAD `18d1a23e`.
- **Motivation:** Close the two independently reproduced P1 root causes
  without widening the patch into a cross-cutting MTP redesign.
- **Expectation:** Current one-feature MoE producers preserve exact measured
  rows after cache reload, and every target-embedded MTP TP consumer follows
  the existing registry accessor. The four direct regression cases, affected
  files, broad PR #20 matrix, direct mechanism probe, and static checks must
  pass.
- **Method:** Added producer, cache-round-trip, and registry-extension
  regressions first. The four-case RED run produced `3 failed, 1 passed`.
  Removed the two MoE feature-count persistence overrides and committed as
  `3e9c0374`. Replaced the three local MTP name sets with
  `get_target_embedded_mtp_linear_ops()` and committed as `18d1a23e`. Ran
  sub-step GREEN, the affected-file matrix, the 23-file broad matrix, an
  in-process exact-row/registry-extension probe, compile checks, and
  `git diff --check`.
- **Result:** MoE GREEN passed `3` tests in `4.71s`; MTP GREEN passed `1` in
  `4.73s`; affected files passed `62` tests in `6.51s`. Fresh final
  verification passed the four direct cases in `5.11s` and the broad matrix
  with `289 passed, 2 deselected in 10.93s`. Reloaded exact rows were
  `{(1.0,): 9.0, (2.0,): 4.0}`; measured `4.0` beat stale `99.0`, absolute
  error was `0.0`, and estimator calls were `0`. All three MTP consumers
  accepted the added registry member and resolved TP `2`, changing the
  reproduced result from `0/3` to `3/3`. Compile and diff checks exited `0`;
  the worktree remained clean. Full evidence is in
  `test_report_2026-08-22_pr20_focused_repair.md`.

## 2026-08-22 - PR #20 focused-repair push

- **Status:** Complete; PR remains open and unmerged.
- **Motivation:** Publish the verified root-cause repairs for maintainer and
  external review while preserving maintainer merge authority.
- **Expectation:** The remote PR #20 branch matches local
  `18d1a23e1a04f3dcad2d2d1376451190d2b5300a`; PR #21 remains unmerged and
  unchanged.
- **Method:** Bootstrapped the company HTTPS proxy and fast-forward pushed
  `refactor/pr4-lookup-registry-core-20260822`.
- **Result:** Local and remote PR #20 branch heads both resolved to
  `18d1a23e1a04f3dcad2d2d1376451190d2b5300a`. No merge, rebase,
  force-push, auto-merge enablement, or PR #21 branch change was performed.

## 2026-08-22 - Fresh Claude review after focused repair

- **Status:** Invocation complete at hard timeout; final Claude verdict
  unavailable.
- **Motivation:** Re-review the complete PR #20 diff after closing the two
  previously confirmed P1 findings.
- **Expectation:** Claude verifies the two repairs, checks the remaining PR
  diff for reachable regression, and returns an explicit `APPROVE`,
  `REQUEST CHANGES`, or `COMMENT` within `1200s`.
- **Method:** From the clean pushed PR #20 worktree, ran
  `timeout 1200s stepcode claude --model 'claude-opus-5' --effort max -p --`
  against `origin/main...18d1a23e`. The read-only prompt prioritized the two
  prior findings, complete-diff regression review, exact file/line evidence,
  separation from `SCOPE-013`, and a required final verdict. Preserved raw
  CLI, prompt, StepCode session, and debug logs.
- **Result:** The outer command returned `124`; StepCode's child exited `143`.
  Session `703700ab-27ef-433f-9a9d-405a389c8525` consumed exactly `1200s`,
  recorded `7` tool calls and `1` tool failure, and emitted no final assistant
  answer. The late failed call was a rest-of-diff reviewer subtask; two
  focused reviewer subtasks completed internally, but their evidence was not
  synthesized into a final P0/P1/P2 list or verdict. No file or Git state
  changed. The required artifact is
  `.worktrees/pr4-lookup-registry-core-20260822/.omx/artifacts/ask-claude-pr20-focused-repair-rereview-20260822-145617.md`.

## 2026-08-22 - PR #21 synthetic-stack verification

- **Status:** Local verification complete; remote PR #21 remains unchanged and
  unpushed.
- **Motivation:** Confirm that the four local PR #21 focused repairs compose
  with the maintainer-approved PR #20 base before requesting an outward-facing
  update.
- **Expectation:** The synthetic merge must be conflict-free; existing lookup,
  registry, profiling, and metadata behavior must remain green; the repaired
  endpoint and aggregate-workload contracts must produce the expected numeric
  values.
- **Method:** Exported synthetic tree
  `3ad1ff8bf383e9b64eedd851a11789c2b5ee028d` (PR #20
  `18d1a23e` plus PR #21 `a7498d15`) under
  `/data/ycfeng/tmp/pr21-stacked-final-20260822-6JXqYk`. Ran the broad
  simulator-only matrix with Python `3.13.13`, the MoE routing/load matrix
  separately with `/usr/bin/python` `3.12.3`, then ran the PR #21 targeted
  matrix, numeric probes, `py_compile`, `git diff --check`, changed-path
  audit, and conflict-marker audit. All temporary logs and bytecode stayed
  under `/data/ycfeng/tmp`.
- **Result:** Broad synthetic-stack matrix passed `321` tests with `2`
  pre-existing logger cases deselected in `10.48s`. The separate routing/load
  matrix passed `13` tests in `6.03s`; the PR #21 targeted sampling/EP matrix
  passed `32` tests in `1.83s`. Seven production files compiled cleanly.
  The synthetic merge tree was conflict-free and the stacked diff contained
  `26` changed paths overall, with no README or canonical profiling-data
  path. Standard decode resolved `kv_cache_size=999` and wrapper
  `total_len=1000` for `max_seq_len=1000`; invalid legacy, online-grid, and
  true-mixed axes raised `ValueError` at planning time; the MoE confirmation
  reported `12,432`, `37,296`, and aggregate `49,728` configurations.

## 2026-08-22 - PR #21 routing numeric environment audit

- **Status:** Contract passes; exact skew count is environment-dependent and
  is recorded as evidence rather than a fixed assertion.
- **Motivation:** The historical handoff cited `61/1024` expert-0 routes for
  the skewed seed-42 probe, while the required `/usr/bin/python` validation
  environment produced a different exact count.
- **Expectation:** Repeated execution must be deterministic within each
  Torch build, and both environments must satisfy the actual contract:
  expert zero remains reachable and the extremely-skewed hot subset remains
  populated.
- **Method:** Ran the same source and seed under `/usr/bin/python` with
  Torch `2.5.1+cu124` and under `aic-step-design` with Torch `2.12.1+cpu`.
  Compared the resulting route counts without changing source.
- **Result:** Both runs produced `423/512` extremely-skewed hot rows.
  Skewed expert-0 routes were `69/1024` on Torch `2.5.1+cu124` and
  `61/1024` on Torch `2.12.1+cpu`. The persisted unit contract checks
  reachability, not a backend-specific exact count; no source change is
  justified.

## 2026-08-22 - PR #21 review handoff

- **Status:** Independent review lanes in progress; no source or remote state
  change.
- **Motivation:** Re-evaluate the repaired diff after the prior architecture
  blocker and code-review findings were addressed.
- **Expectation:** The architecture lane returns `CLEAR`, `WATCH`, or
  `BLOCK`; the code/spec lane returns severity-rated findings and an explicit
  recommendation. Missing evidence remains a review gate.
- **Method:** Reassigned the existing architecture and code-review agents to
  inspect `18d1a23e..a7498d15` and synthetic tree `3ad1ff8b` read-only.
- **Result:** Pending agent reports. Push and PR-body updates remain gated on
  explicit maintainer authorization.

## 2026-08-22 - PR #21 decode memory-accounting repair

- **Status:** Complete and committed as `b8149c32`; remote PR #21 remains
  unchanged.
- **Motivation:** The architecture review found that standard decode memory
  filtering counted `kv_cache_size` but omitted the current decode token,
  while `is_valid()` and wrapper block allocation already counted it.
- **Expectation:** A decode input with `kv_cache_size=32` must fail a
  `32`-token capacity check and pass a `33`-token capacity check; prefill
  accounting must remain unchanged.
- **Method:** Added
  `test_decode_memory_limit_reserves_the_current_token` first. The RED run
  failed `1` test because capacity `32` returned `True`. Changed
  `AttentionInput.is_under_memory_limit()` to use
  `kv_cache_size + 1` for decode and `kv_cache_size + prefill_chunk_size` for
  prefill. The GREEN case passed, the affected sampling/EP matrix passed
  `33`, and the change was committed after the focused test passed.
- **Result:** Direct boundary values are decode capacity `32 -> False`,
  capacity `33 -> True`; architecture re-review closed the WATCH and returned
  `CLEAR`.

## 2026-08-22 - Final PR #21 synthetic-stack verification

- **Status:** Local final verification complete; push remains authorization-
  gated.
- **Motivation:** Re-run the full evidence chain after the memory-accounting
  repair and before final review synthesis.
- **Expectation:** The updated synthetic stack remains conflict-free and all
  prior PR #20/PR #21 behavior stays green.
- **Method:** Exported synthetic tree
  `dc8081821c0e36fb007f3c67902a88541189d119` from PR #20 base
  `18d1a23e` and PR #21 tip `b8149c32` to
  `/data/ycfeng/tmp/pr21-stacked-final2-20260822-3LCkDr`. Re-ran the broad
  simulator matrix, routing/load matrix, targeted sampling/EP matrix, final
  numeric probes, `py_compile`, `git diff --check`, path audit, and
  conflict-marker audit.
- **Result:** Broad matrix passed `322` tests with `2` logger cases
  deselected in `13.24s`; routing/load passed `13` in `6.13s`; targeted
  sampling/EP passed `33` in `1.92s`; seven production files compiled; all
  static audits passed. Numeric probes reported decode endpoint `999/1000`,
  memory boundary `32 -> False` and `33 -> True`, mixed-axis fail-fast, MoE
  aggregate `49,728`, and extreme-skew hot rows `423/512`.

## 2026-08-22 - Architecture re-review disposition

- **Status:** CLEAR.
- **Motivation:** Close the architecture WATCH raised against the previous
  `a7498d15` tip.
- **Expectation:** The final architecture lane must verify that decode
  validation, memory filtering, and wrapper allocation use one logical-length
  contract.
- **Method:** The independent architecture agent inspected `b8149c32` and
  synthetic tree `dc808182`, re-ran focused boundary cases and compatibility
  tests, and checked changed-file blob identity.
- **Result:** The lane reported `CLEAR`; no remaining BLOCK or WATCH exists in
  the 11-file PR #21 diff or its composition with PR #20.

## 2026-08-22 - Final review synthesis and delivery authorization

- **Status:** Review complete; delivery in progress.
- **Motivation:** Close the remaining local review gate after the independent
  code/spec lane returned `APPROVE` and the architecture lane returned
  `CLEAR`.
- **Expectation:** The PR #21 merge-commit history remains intact, the final
  task records use the committed tip and synthetic tree, and the remote branch
  advances without rewriting history.
- **Method:** Retained local merge commit `54e123ea` (parents `b8149c32` and
  approved PR #20 head `18d1a23e`), corrected stale report references, and
  prepared a normal push of `refactor/pr4-profiling-envelope-ep-20260822`.
- **Result:** Local review records now show `APPROVE`/`CLEAR`; no source files
  changed. Remote push and post-push read-only verification are the remaining
  actions.

## 2026-08-22 - PR #21 push and remote closure

- **Status:** Complete; PR #21 is published and remains unmerged.
- **Motivation:** Advance the maintainer-approved stacked PR after both
  independent review lanes closed cleanly.
- **Expectation:** A normal push preserves the merge ancestry, updates the PR
  body with final evidence, and leaves PR #20 untouched and both PRs open.
- **Method:** Ran the targeted pre-push matrix (`33 passed in 1.82s`),
  `py_compile` (`7/7`), diff/path/conflict audits, then pushed
  `54e123ea` through the HTTPS proxy and updated PR #21's body. Queried the
  branch refs and PR metadata read-only afterward.
- **Result:** PR #21 head is `54e123ea`, base is `18d1a23e`, and GitHub reports
  `OPEN/CLEAN`; PR #20 remains at `18d1a23e`, `OPEN/CLEAN`. Neither PR was
  merged or force-pushed.

## 2026-08-22 - Two-round PR #21 review and concurrent Claude run

- **Status:** Complete; final local review and warning verification are archived.
- **Motivation:** Obtain two independent local review verdicts and a separate
  external second opinion after PR #21 publication.
- **Expectation:** Both local lanes inspect the same immutable
  `18d1a23e..8cc267ea` range without edits, while Ask Claude runs with a hard
  `1800s` timeout; all findings remain evidence-backed and both PRs stay open.
- **Method:** Bootstrapped the company HTTPS proxy, pushed the fast-forward
  `c4ac39a9..8cc267ea`, and verified the remote branch ref. Started fresh
  architecture and code/spec review agents in read-only mode. Started
  `stepcode claude --model claude-opus-5 --effort max` under `timeout 1800s`;
  raw output is being captured under `/data/ycfeng/tmp/`.
- **Result:** PR #21 is `OPEN`, head `8cc267ea`, base `18d1a23e`,
  `mergedAt=null`, and auto-merge is unset. PR #20 remains `OPEN/CLEAN` at
  `18d1a23e`. Round-1 code/spec returned `APPROVE`; Round-2 code/spec returned
  `CLEAR/APPROVE`; Round-2 architecture returned `CLEAR with WATCH`. The
  Round-1 architecture artifact targeted the preceding `a8d75b84` tip and is
  retained as pre-final evidence only. Claude's `1800s` invocation exited
  `143` without a verdict.

## 2026-08-22 - Option-B warning-feedback verification

- **Status:** Complete; no further production change required.
- **Motivation:** The maintainer selected Option B but required every
  target-local explicit KV discard to be visible, so users do not spend time
  editing parameters merely to avoid a premature `ValueError`.
- **Expectation:** Standard explicit KV values are filtered per `(model, TP)`;
  warnings identify every discarded output group; retained values union across
  targets; all-target absence remains fail-fast.
- **Method:** Ran
  `tests/unit/test_attention_memory_filter_contract.py` (`3 passed`),
  the combined sampling/EP/routing matrix (`59 passed`), `compileall`
  (`rc=0`), `git diff --check`, and an inline warning/union probe.
- **Result:** For capacities `256` and `512` with requested KV
  `[255, 300, 511]`, the small target retained `[255]` and emitted one
  `RuntimeWarning` for `2` dropped combinations (`[300, 511]`), while the
  large target retained all three. The union was `[255, 300, 511]`. An
  all-target drop of `511` raised
  `ValueError: no physically legal pairing ... [511]` with target capacities.
  Warning and coverage details are in
  `test_report_2026-08-22_pr21_option_b_warning_feedback.md`.

## 2026-08-22 - Post-publication audit and PR description synchronization

- **Status:** Complete; no production source change required.
- **Motivation:** Reconcile the published PR descriptions with the actual
  remote heads and filter independent audit findings before deciding whether
  any deferred item belongs in PR #21.
- **Expectation:** PR descriptions state the current immutable heads and
  Option-B evidence; cross-cutting producer or MTP redesign work remains
  outside the focused repair.
- **Method:** Updated PR #20 and PR #21 descriptions through the authorized
  HTTPS-proxied GitHub command. Audited current source with AST and targeted
  runtime probes for optional producer metadata and MTP registry extension
  behavior.
- **Result:** PR #20 is `OPEN/CLEAN` at `18d1a23e`; PR #21 is `OPEN/CLEAN` at
  `8cc267ea`; neither has auto-merge enabled. The producer audit found zero
  production assignments for `_feature_bounds`, `_feature_domain`,
  `_feature_constraints`, `_identity`, and `_model_identity`, plus a
  pre-existing standalone `BaseTrainer` metadata gap. The MTP audit confirmed
  the focused TP-consumer repair is correct, while complete family/interface
  unification and all-site extension propagation remain separate follow-up
  work. No production/test source or canonical data changed.

## 2026-08-22 - vLLM context-limit semantic cross-check

- **Status:** Complete; source-only confirmation.
- **Motivation:** Confirm that the Option-B runtime gate uses the variable that
  represents the model's advertised context limit rather than a batch or
  profiling bound.
- **Expectation:** The vLLM V1 source should show `max_model_len` enforcing
  prompt-plus-output context and `AttentionMetadata.max_seq_len` acting as a
  current-batch or capture bound.
- **Method:** Read the checked-out vLLM V1 model config, scheduler,
  input-processor, GPU-runner, and attention-metadata paths.
- **Result:** `ModelConfig.max_model_len` is derived from model metadata and
  used by request stopping, prompt/output validation, KV-block allocation, and
  GPU-runner end-index checks. `AttentionMetadata.max_seq_len` is populated
  from the current batch's optimistic longest sequence and only uses the model
  limit for capture. Therefore the 256K/1M-style context limit corresponds to
  `max_model_len`; Frontier's `max_seq_len` profiling envelope and
  `max_model_len` runtime boundary are semantically distinct. This supports
  allowing explicit `KV + query_tokens` above the profiling envelope when the
  runtime boundary and physical capacity are satisfied.

## 2026-08-22 - Final focused verification after description sync

- **Status:** PASS.
- **Motivation:** Re-run the focused PR #21 contract after the documentation
  and remote-description updates, confirming that no source behavior changed.
- **Expectation:** Warning visibility, target-local filtering, retained union,
  all-target fail-fast, sampling domains, EP divisors, and routing contracts
  remain green.
- **Method:** Ran the reproducible combined matrix from
  `test_report_2026-08-22_pr21_option_b_warning_feedback.md`, then ran
  `compileall`, `git diff --check`, and read-only GitHub PR-state queries.
- **Result:** `59 passed in 3.08s`; compile and whitespace checks passed. PR
  #20 remains `OPEN/CLEAN` at `18d1a23e`; PR #21 remains `OPEN/CLEAN` at
  `8cc267ea`; both report `mergedAt=null` and no auto-merge request. The
  source worktree stayed clean.

## 2026-08-25 - Gate 2 typed-lane repair and zero-routed verification

- **Status:** Focused and PR17-sensitive gates PASS; broad unit run has only
  unrelated post-PR17 baseline failures.
- **Motivation:** Gate 2 selected the typed lane-local workload contract after
  direct probes showed regular `Batch` callers could not reach load-aware
  predictors, while blindly accepting a global map would send the wrong expert
  width to local-width profiling models. A follow-up probe found that a
  physical zero-routed EP lane still queried `moe_shuffling` with
  `total_routed_tokens=0`.
- **Expectation:** Regular `Batch` global maps split into canonical local lanes;
  EP=1 maps use the single local lane; partial local maps without lane identity
  fail before model access; every physical zero-load lane skips on-demand
  shuffling lookup while retaining its lane topology and barrier identity.
- **Method:** Added the immutable `EPLaneWorkload` contract and canonical global
  splitter, threaded it through shared MoE and disaggregation predictors, added
  identity/topology/domain regressions, and changed shuffling admission to
  validate counts then skip lanes whose routed-token sum is zero. Ran the RED
  probe first, then focused, PR17-sensitive, compile, whitespace, and full unit
  matrices.
- **Result:** The RED test failed because a zero-load lane invoked the fake
  model; the GREEN test passed with exactly one model call and captured loads
  `[8]`. Focused predictor/disaggregation/workload tests passed `107`; the
  PR17-sensitive matrix passed `188` with `1 xfailed`; `py_compile` and
  `git diff --check` passed. Full `tests/unit` reported `2776 passed`,
  `25 skipped`, `1 xfailed`, and `18 failed`; all 18 failures are unrelated
  release/docs/analysis-tool baseline surfaces and touch none of the six
  typed-lane files.

## 2026-08-25 - Docs-first A' checkpoint

- **Status:** Complete; production implementation is the next sub-step.
- **Motivation:** The post-PR17 merge moved exact stage/KV provenance into
  scheduler-owned context while the provisional Gate 2 probe mixed physical
  workload data with runtime identity and retained raw-map escape paths.
- **Expectation:** The task documents must state one canonical physical lane
  descriptor, one owner for each field, symmetric EP=1/EP>1 behavior, zero-lane
  semantics, and the pure MTP boundary before code changes proceed.
- **Method:** Added R-023 through R-025, formal RCA entries SCOPE-024/025, the
  A' ownership graph and invariants, the dependency-ordered TDD plan, and the
  implementation harness gates. Existing probe evidence remains archived but
  is explicitly provisional.
- **Result:** Documentation scan and terminology checks are the RED/GREEN gate
  for this sub-step. No production or test source was changed by the docs
  checkpoint.

## 2026-08-25 - Scheduler/entity typed-lane propagation

- **Status:** Complete; committed as `c70d2052`.
- **Motivation:** The canonical `EPLaneWorkload` descriptor must survive the
  scheduler plan, EP entity, stage ledger, and direct event path without a
  second mutable expert-token owner. Existing EP fixtures also used the removed
  raw-map constructor.
- **Expectation:** `EPBatchGroupPlan`, `EPBatchGroup`, and `BatchStage` carry the
  same descriptor; compatibility maps are created only at explicit output
  boundaries; stage/KV provenance and complete EP barriers remain unchanged.
- **Method:** Threaded `lane_workload` through scheduler plan/materialization and
  predictor-evaluation lane construction, replaced stage-ledger map copying
  with typed attachment, updated the direct event path and metrics projection,
  and migrated all scheduler/entity EP fixtures to legal fixed-width topology.
- **Result:** The focused scheduler/entity matrix passed `611` tests with `19`
  skips and `1` expected failure. Module compilation and `git diff --check`
  passed. Provisional predictor files remain uncommitted and unchanged by this
  sub-step.

## 2026-08-25 - Post-implementation MTP and trace boundary audit

- **Status:** Historical decision-gate record; resolved by the 2026-08-26
  narrow A' approval.
- **Motivation:** Verify that the implementation satisfies the frozen A'
  wording instead of treating the remaining generic MTP synthetic shape as an
  unexamined compatibility detail.
- **Evidence:** The MoE override consumes typed lane descriptors and aggregates
  the five physical phase values with lane-wise `max()`. The generic replay
  still creates block and terminal `Batch` objects so the existing predictor
  API can receive per-request token vectors and speculative metadata. The
  objects do not enter scheduler admission or carry EP identity. The trace
  helper receives only descriptor-backed map projections and emits logs.
- **Disposition:** The maintainer selected the narrow physical-lane contract.
  The generic block/terminal replay remains a scheduler-independent shape
  adapter, and no shared MTP interface expansion is made.

## 2026-08-26 - Token-ledger RCA and implementation handoff

- **Status:** Documentation sub-step complete; implementation is next.
- **Motivation:** The maintainer confirmed the shared `Batch` compute-contract
  repair after the direct probe showed `planned=[2,1]`, `verify=[3,2]`, raw
  physical width `5`, current compute lookup `8`, transfer width `5`, and
  top-k-2 assignment count `10`.
- **Expectation:** The task record must state one canonical owner for each
  projection, identify `8` as a double-counted lookup rather than a physical
  workload, and preserve the narrow A' typed-lane boundary.
- **Method:** Added R-026, SCOPE-027, the canonical ledger in `design.md`, the
  dependency-ordered RED/GREEN addendum in `plan.md`, and hard token-ledger
  gates in `harness.md`. The approved production changes are limited to the
  shared target-embedded MTP compute branch and the first structural
  verification-block shape.
- **Result:** Documentation now records the exact equations and acceptance
  criteria. No production or test source changed in this documentation step;
  the next step is the TDD RED regression.

## 2026-08-26 - Token-ledger TDD RED gate

- **Status:** RED confirmed; production edits remain pending.
- **Motivation:** Prove that the new regression cases detect the two approved
  contract defects on the current implementation before changing production
  code.
- **Expectation:** A registered target-embedded MTP batch with
  `planned=[2,1]`, `verify=[3,2]` must expose compute and transfer width `5`;
  the first structural replay block must preserve `[3,2]`; materializer
  assignments must conserve `5 * top_k(2) = 10`.
- **Method:** Ran
  `PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider
  tests/unit/test_mtp_token_ledger_repair.py` against the uncorrected
  implementation.
- **Result:** `2 failed, 1 passed`. The compute assertion observed
  `total_num_tokens=5` but helper result `8`; the structural capture observed
  `num_tokens=[1,1]` versus expected `[3,2]`; the materializer observed
  `routing_token_count=5`, `total_routed_assignments=10`, and lane-sum `10`.
  Both failures are the intended RED signal, not fixture/import errors.

## 2026-08-26 - Shared Batch compute contract GREEN

- **Status:** Complete; structural replay remains the next independent
  production sub-step.
- **Motivation:** Remove the target-embedded MTP planned-draft double count at
  the shared `Batch` owner rather than compensating in individual predictors.
- **Expectation:** The same `[3,2]` physical batch returns compute width `5`
  and transfer width `5`, while explicit CUDA Graph/AFD metadata and unrelated
  effective-token paths retain their existing precedence and values.
- **Method:** Changed only the registered target-embedded MTP branch in
  `frontier/entities/batch.py` to return `_total_num_tokens`; ran the new
  ledger assertion, the full `tests/unit/test_predictor_effective_tokens.py`,
  the CUDA Graph/AFD stage-token tests, module compilation, and whitespace
  validation.
- **Result:** `40 passed` for the ledger assertion plus predictor-effective
  token suite, `9 passed` for CUDA Graph/AFD stage-token coverage,
  `py_compile` succeeded, and `git diff --check` was clean. No caller-specific
  bypass or new metadata field was introduced.

## 2026-08-26 - Structural MTP verification shape GREEN

- **Status:** Complete; the MONOLITHIC boundary audit is the next sub-step.
- **Motivation:** Keep post-forward rejection outcomes from shrinking the
  physical target-verification shape used by structural replay.
- **Expectation:** For `verify=[3,2]` and `rejected=[2,1]`, the first replay
  block constructs `Batch.num_tokens=[3,2]`; rejected values remain available
  only for outcome/progression accounting.
- **Method:** Changed the first-block shape calculation in
  `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` to
  consume complete verification widths. Ran the new token-ledger regression,
  the structural MoE replay tests, and the ordinary speculative verify-prefill
  matrix, followed by module compilation and whitespace validation.
- **Result:** The targeted structural assertion passed, the existing
  structural-MTP file passed `4` tests, the combined token-ledger and verify
  prefill run passed `17` tests, `py_compile` succeeded, and `git diff --check`
  was clean. No MTP descriptor or shared predictor interface was added.

## 2026-08-26 - MONOLITHIC initial-decode boundary audit

- **Status:** Complete; the existing scheduler branch is retained and a focused
  regression is the next implementation step.
- **Motivation:** Determine whether the `max(planned_drafts, 1)` branch is an
  accidental third token ledger or an intentional scheduler frontier needed by
  the MONOLITHIC prefill-completion lifecycle.
- **Expectation:** A real `Request` probe must distinguish the one-time
  prefill boundary from ordinary target-verification batch width and show that
  the shared compute/materializer ledger remains independent.
- **Method:** Called the production
  `VLLMv1EngineReplicaScheduler._get_scheduler_num_computed_tokens()` and
  `_get_request_next_num_tokens()` methods on a real `Request` with
  `num_prefill_tokens=8`, `num_decode_tokens=32`, and target-embedded MTP.
  Covered `planned_drafts` values `0, 1, 2, 4`, post-boundary progress, and
  explicit scheduler frontiers.
- **Observed values:** At `num_processed_tokens=9`
  (`num_processed_decode_tokens=1`), the scheduler frontier is `8` and next
  widths are `[1, 1, 2, 4]`. At `num_processed_tokens=10` and `12`, the
  frontier is `9` and `11`, and next widths are `[1, 2, 3, 5]`. With explicit
  frontiers `8, 9, 10, 12` and `planned=2`, next widths are `2, 3, 3, 3`.
- **Result:** The branch is an intentional one-time admission projection:
  MONOLITHIC `Request.on_batch_end()` grants one output token at the final
  prefill callback, while the scheduler frontier remains at the prefill width
  until the first decode scheduling step. No physical-width correction is
  needed in this seam. Evidence and acceptance criteria are recorded in
  `issues.md:SCOPE-028`, `design.md`, and `harness.md`.

## 2026-08-26 - Docs-first implementation handoff audit

- **Status:** Complete for documentation; implementation remains in progress.
- **Motivation:** The approved narrow A' design must be applied to the
  provisional dirty implementation without allowing raw expert maps or
  scheduler identity to leak back into predictor and trace contracts.
- **Expectation:** Every remaining code slice has one owner, one focused gate,
  and one commit boundary. The trace helper is explicitly identified as the
  remaining raw-map production input site.
- **Method:** Re-read the current diff with whitespace ignored, traced all
  `EPLaneWorkload`/`per_expert_tokens` call sites, and recorded the ownership
  boundaries in `design.md`, `issues.md`, `plan.md`, and `harness.md`.
- **Result:** The documentation checkpoint is complete. The next code action
  is to run the existing typed-lane RED matrix, then finish and commit the
  predictor/communication/trace slices in dependency order.

## 2026-08-26 - Typed-trace observability migration

- **Status:** Complete; committed as `183cfd61`.
- **Motivation:** Close the remaining production boundary where the EP trace
  helper accepted a raw expert-token map even though scheduler and entity code
  already owned an immutable `EPLaneWorkload` descriptor.
- **Expectation:** The helper accepts only the typed descriptor, derives
  `ep_id`, EP size, and the fixed-width map from it, and preserves the existing
  log record shape and scheduler trace identity. Raw maps remain output views.
- **Method:** Added
  `tests/unit/test_typed_ep_trace_contract.py`, observed the intended RED
  `TypeError` for the missing `lane_workload` keyword, migrated the helper and
  all three production callers, then ran the focused trace/materialization
  matrix.
- **Result:** The RED test passed after the migration. Fresh verification
  passed `168` tests; `py_compile` and `git diff --check` were clean. The
  helper's production workload contract no longer accepts `per_expert_tokens`.

## 2026-08-26 - Terminal MTP overshoot EP>1 RED

- **Status:** RED confirmed; production edits remain pending.
- **Motivation:** The real terminal metadata path must remain compatible with
  the typed EP lane contract after PR17 conflict repair. A terminal synthetic
  batch cannot carry a fabricated lane descriptor, so the test must expose the
  missing physical seam before any hook implementation.
- **Expectation:** The scheduler builder produces a terminal row with
  `verify=[3]` and `terminal_verify=[[1]]`; the EP=2 MoE predictor then reaches
  a narrow physical-lane hook that can materialize two participants and
  aggregate their five phases.
- **Method:** Added
  `tests/unit/test_mtp_terminal_overshoot_ep_replay.py` using the real
  `VLLMv1EngineReplicaScheduler._build_spec_decode_batch_metadata()` and a
  non-dummy `SklearnMoEExecutionTimePredictor`. Corrected the fixture's
  expected phase total from `27.0` to `25.0` (`attention=7`, lane maxima sum
  `18`). Ran:

  ```text
  PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider \
    tests/unit/test_mtp_terminal_overshoot_ep_replay.py
  ```

- **Observed result:** `1 failed`. Metadata assertions passed. The failure is
  the expected production error from
  `frontier/operators/families.py:375`:
  `ValueError: expert-parallel all-to-all payload requires an EPLaneWorkload
  descriptor`. The test reaches the target seam without profiling-cache,
  model-schema, or fixture-construction errors.
- **Next action:** Add the default terminal-row hook and MoE physical-lane
  override described in `design.md`, then rerun this exact test before the
  broader focused matrix.

## 2026-08-26 - Terminal MTP physical-lane hook implementation

- **Status:** Complete for the terminal hook sub-step; PR20/PR21
  merge-readiness remains pending.
- **Motivation:** Close the confirmed EP>1 MoE terminal replay failure without
  putting scheduler identity or a fabricated lane descriptor into the generic
  synthetic MTP `Batch`.
- **Expectation:** Dense and EP=1 terminal rows keep the generic stage path;
  EP>1 MoE rows use one shared attention/overhead probe and the existing typed
  five-phase lane seam, with only per-layer physical phases scaled.
- **Method:** Added the default `_predict_mtp_terminal_row_time_ms()` hook in
  `sklearn_execution_time_predictor.py`; extracted the reusable typed-lane
  aggregation helper and added the EP>1 MoE override in
  `sklearn_moe_execution_time_predictor.py`. Corrected the terminal regression
  fixture to carry the real `moe_expert_parallel_size=2` topology field.
- **Result:** The terminal replay returns `25.0 ms` from `7.0 ms` shared
  attention/total scope plus phase maxima `(2, 2, 4, 4, 6) ms`. Lane IDs are
  `[0, 1]`, local expert widths are `[4, 4]`, and routed counts sum to `2`.
  A `num_layers=3` direct probe returns `61.0 ms` and sends `num_layers=1`
  to the attention probe. EP=1 and dense fallbacks each return `5.0 ms` via
  one generic stage call. The terminal/structural/typed subset passes `32`
  tests; the full focused matrix passes `178` tests.

## 2026-08-26 - Predictor layer-identity propagation audit

- **Status:** Complete; production and regression changes are ready for the
  dedicated sub-step commit.
- **Motivation:** The public MoE predictor already received a global
  `layer_id`, but the internal execution-time seam dropped it. Attention probes
  defaulted to layer zero and terminal MTP replay used `pipeline_stage`, which
  can select the wrong layer for mixed-layer or non-zero-layer requests.
- **Expectation:** Preserve `pipeline_stage`/`stage_id` for stage-local work and
  forward the explicit global `layer_id` through routing lookup, internal
  execution-time prediction, attention prediction, and the terminal MTP hook.
  Legacy internal callers without a layer identity retain the documented
  compatibility default.
- **Method:** Added `layer_id: int = 0` to
  `_get_execution_time_internal()`, forwarded the public argument, replaced
  the fixed attention layer and terminal-stage substitution, and added
  propagation assertions in
  `tests/unit/test_moe_predictor_layer_id_semantics.py`. The RED assertion
  first failed with `KeyError: 'layer_id'` because the internal call omitted
  the keyword.
- **Result:** The layer-semantics, terminal-MTP, and structural-MTP subset
  passed `32` tests after the repair. The focused report records the fresh
  combined matrix, compile, whitespace, and subsequent PR20/PR21 audit gates.
  No identity inference, fallback scaling, or second predictor wrapper was
  introduced.

## 2026-08-26 - Boundary audit and PDD attention-only identity gap

- **Status:** Audit complete; SCOPE-032 is the next TDD sub-step.
- **Motivation:** Close the remaining identity seam before the local PR20/PR21
  merge-readiness audit. The public PDD scheduler calls already carry a global
  layer identity, so the helper must preserve it rather than silently querying
  layer zero.
- **Method:** Read the routed EP operator payload path and directly constructed
  aggregate `CommPayloadContext` inputs for both all-to-all directions. Both
  failed at the existing payload boundary with
  `ValueError: expert-parallel all-to-all payload requires an EPLaneWorkload
  descriptor`, confirming that the operator contract already owns this guard.
  Enumerated the structural-MTP registry/configs: local entries loaded with
  `num_layers=48`, `2`, and `20`, all with `layer0_is_moe=True`; two names
  failed explicitly because their local JSON paths are absent.
- **Expectation:** Add one focused regression that sends a non-zero global
  `layer_id` through `predict_stage_execution_time(..., include_ffn=False)` and
  observes the current helper's `layer_id=0` lookup. Then propagate that one
  parameter without changing stage, communication, or MTP ownership.
- **Result:** The audit produced no justification for an extra EP guard or
  speculative config asset. Production repair and RED evidence remain pending.
