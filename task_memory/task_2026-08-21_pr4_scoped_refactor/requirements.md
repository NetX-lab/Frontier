## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-27 | Confirmed方案 A for propagating active disaggregation topology through the inherited MoE API. |
| 2026-08-27 | Approved strict descriptor/predictor topology validation for routed EP admission. |
| 2026-08-27 | Selected the unified routed-aggregate fail-fast contract after the PDD non-dummy RCA. |
| 2026-08-27 | Approved inclusion of the mixed-layer dummy predictor repair (SCOPE-036) in the current merge-readiness scope. |
| 2026-08-26 | Recorded the maintainer's narrow A' decision for unified EP>1 routed aggregate admission in dummy and non-dummy modes. |
| 2026-08-26 | Confirmed the shared `Batch` compute-contract repair and the structural MTP verification-width correction after the token-ledger RCA. |
| 2026-08-25 | Recorded the post-PR17 PR20/PR21 merge-conflict RCA, the maintainer-approved A' typed-lane contract, and the docs-first implementation request. |
| 2026-08-22 | Recorded the maintainer's Option-B decision for target-local explicit KV filtering and mandatory discard warnings. |
| 2026-08-22 | Recorded the request for two independent PR #21 review rounds plus a concurrent 30-minute Ask Claude review. |
| 2026-08-22 | Recorded maintainer approval of PR #20 without merge and the instruction to advance stacked PR #21. |
| 2026-08-22 | Recorded the maintainer-approved focused repair, targeted verification, PR #20 push/re-review, and separate MTP interface audit boundary. |
| 2026-08-22 | Recorded the maintainer's instruction to resume the timed-out Claude conversation and complete the PR #20 review. |
| 2026-08-22 | Recorded the requested 20-minute Ask Claude review of remote PR #20. |
| 2026-08-22 | Recorded the maintainer's option-A decision for universal exact-row persistence at the unified runtime training/cache entry points. |
| 2026-08-22 | Recorded the two-PR remote continuation, review-before-merge gate, and HTTPS proxy requirement. |
| 2026-08-21 | Created the raw intent record for the scoped PR #4 refactor continuation. |
| 2026-08-21 | Recorded maintainer concerns about dependency-removal cost and the need for an independent profiling-facing adapter. |
| 2026-08-21 | Confirmed the staged dependency boundary and selected `operator_query_binding` / `bind_operator_query` for registry admission. |
| 2026-08-21 | Maintainer selected EP policy A: profile every runtime-legal positive divisor and derive helper/CLI domains from it. |

# Requirements

This file is the source of truth for user intent. It records requests and
constraints only; implementation choices belong in `plan.md` and `design.md`.

## R-001 - Start a new scoped PR from the old review

- **Attribute:** `[Original Request]`
- **Raw request:** Continue the work from
  `/data/ycfeng/stepfun-performance-optimization/Frontier/task_memory/task_2026-08-02_pr4_lookup_cache_review` and design a new, scoped, modular PR that addresses the root issues from the old PR #4 review.
- **Captured intent:** Preserve useful evidence from the old task, but do not carry its broad implementation wholesale.

## R-002 - Clean PR initialization and isolated worktree

- **Attribute:** `[Original Request]`
- **Raw request:** Initialize a new task-memory directory/file and set up a clean, dedicated Git worktree branch for the new PR.
- **Captured intent:** Keep the current checkout and unrelated user files untouched.

## R-003 - Analysis and planning only in the current phase

- **Attribute:** `[Original Request]`
- **Raw request:** The current phase is analysis and planning only. Do not write production code until the architecture, scope, and plan are verified.
- **Captured intent:** Documentation and read-only verification are allowed; production and test source changes are not.

## R-004 - Profiling/runtime boundary

- **Attribute:** `[Original Request]`
- **Raw request:** Treat profiling as an offline benchmark generator that writes benchmark CSV files, with the stated non-KV-cache-memory exception. Runtime must not invoke, import, or directly depend on profiling. On a predictor-cache miss, runtime locates raw CSV data and fits/trains a predictor on demand; simulation queries the predictor and its memory cache.
- **Captured intent:** Use a unidirectional, file-based boundary and avoid a separate Contract/Data Guarantee abstraction layer.

## R-005 - Explicit operator registry and no name heuristics

- **Attribute:** `[Original Request]`
- **Raw request:** Every operator and operator family must be explicitly registered and predefined. Do not classify by prefixes, substrings, splits, or scattered literal sets. Use Frontier's unified operator registry and APIs.
- **Captured intent:** Unknown or unsupported operators must fail fast, and adding an ordinary operator should require one unified-registry declaration.

## R-006 - Profiling sampling and domain coverage

- **Attribute:** `[Original Request]`
- **Raw request:** Follow the sampling rules established in the previous review task. The profiling measurement domain must be greater than or equal to the runtime parameter domain so uncertain in-flight combinations are covered.
- **Captured intent:** Preserve deterministic, operator-specific, physically bounded sampling rather than an unbounded Cartesian product.

## R-007 - Port confirmed fixes and resolve cache mismatches

- **Attribute:** `[Original Request]`
- **Raw request:** Identify and incorporate confirmed critical fixes from the previous review, resolve lookup-cache mismatch and inconsistency issues from PR #4, and justify every refactor with concrete root-cause evidence.
- **Captured intent:** Retain fail-fast validation and the verified legal-miss resolution principles, without importing unrelated publication/provenance work.

## R-008 - Grill-me decision protocol

- **Attribute:** `[Original Request]`
- **Raw request:** Proactively grill the maintainer about ambiguous bug fixes, borderline scope, and key design trade-offs before proceeding; ask one focused question at a time and provide recommended alternatives.
- **Captured intent:** Important boundary and operator-resolution decisions must be explicitly confirmed before implementation planning is treated as approved.

## R-009 - Current-phase deliverables

- **Attribute:** `[Original Request]`
- **Raw request:** Deliver (1) a gap analysis between legacy PR #4 and the target architecture, (2) an itemized inclusion list of bugs/features, and (3) a step-by-step implementation plan for review.
- **Captured intent:** Keep the deliverables auditable in the new task-memory directory and link them to the old review evidence.

## R-010 - Inherited repository constraints

- **Attribute:** `[Original Request]`
- **Raw request:** Follow the repository `AGENTS.md` rules: no fallback patches, fail fast on unexpected conditions, preserve the main checkout, do not modify `README.md`, do not alter canonical profiling data or remote PR state, and store task reports under `task_memory/<task_dir>/`.
- **Captured intent:** Any later implementation must remain small, modular, and directly verifiable.

## R-011 - Do not force a zero-import migration without evidence

- **Attribute:** `[Original Request]`
- **Raw request:** The maintainer questions whether removing predictor/training
  paths' direct dependency on profiling implementation is reasonable, because it
  may require many code and module changes. Appropriate dependencies may remain
  when that avoids broad edits, unless the affected code is easy to migrate.
- **Captured intent:** Evaluate dependency removal by concrete migration cost and
  preserve small, side-effect-free shared dependencies when justified.

## R-012 - Reassess the independent profiling-facing adapter

- **Attribute:** `[Original Request]`
- **Raw request:** The maintainer questions the necessity and value of an
  independent profiling-facing adapter when profiling only needs to provide CSV
  and memory support.
- **Captured intent:** Prove a real ownership or interface gap before adding an
  adapter; do not introduce a second metadata source merely to satisfy a layer
  diagram.

## R-013 - Select the operator query binding name and seam

- **Attribute:** `[Original Request]`
- **Raw request:** "A，继续" in response to the naming candidates for the
  registry-facing operator lookup component.
- **Captured intent:** Use `operator_query_binding` as the conceptual module
  name and `bind_operator_query` as the API name. Prefer reusing the existing
  `frontier/operators/binding.py` surface before creating a new file. Keep the
  component runtime/registry-facing, with no independent profiling-facing
  adapter or second operator catalog.

## R-014 - Select the MoE EP profiling-domain policy

- **Attribute:** `[Original Request]`
- **Raw request:** "A，继续" in response to the two MoE EP sampling-domain
  alternatives.
- **Captured intent:** Adopt option A: every positive divisor accepted by the
  runtime's `total_expert_num % moe_expert_parallel_size == 0` rule belongs to
  the profiling domain. Derive the default MoE profiling helper and the CLI's
  omitted-argument behavior from that domain. Keep explicitly supplied EP
  lists supported through strict divisibility validation, and preserve the
  existing low-cost example scripts that pass an explicit EP list.

## R-015 - Continue the two-PR remote stack without merging

- **Attribute:** `[Original Request]`
- **Raw request:** Restore the task context from
  `.worktrees/pr4-scoped-lookup-boundary-20260821/task_memory/task_2026-08-21_pr4_scoped_refactor`,
  continue the remaining work after splitting the task into two remote PRs,
  keep both PRs unmerged until maintainer review approval, and set the HTTPS
  proxy for remote Git operations.
- **Captured intent:** Treat remote PR #20 as the lookup/registry base and PR
  #21 as the profiling-envelope/EP stack. Process the base PR first, preserve
  the stacked relationship, use the company proxy bootstrap for each remote
  operation, and leave merge authority with the maintainer.

## R-016 - Select universal exact-row persistence

- **Attribute:** `[Original Request]`
- **Raw request:** "A，继续" after reviewing the two producer-policy options.
- **Captured intent:** Make both unified runtime training/cache entry points
  persist scalar-numeric measured rows by default. Preserve an explicit opt-out
  for a future producer with an unsupported key schema, prove the default
  behavior through a real model-cache round trip, and keep unmeasured legal
  queries on the existing runtime-cache/model fallback path.

## R-017 - Run an external Claude review of PR #20

- **Attribute:** `[Original Request]`
- **Raw request:** "调用ask claude 进行pr#20的审阅，设定timeout limit20mins"
- **Captured intent:** Invoke the configured Ask Claude backend against the
  complete PR #20 diff with a hard 20-minute limit, preserve the review
  evidence locally, and keep the remote PRs unmerged while the result is
  assessed.

## R-018 - Resume the Claude PR #20 review

- **Attribute:** `[Original Request]`
- **Raw request:** "继续 resume"
- **Captured intent:** Resume the preserved Claude conversation, request an
  immediate synthesis from the evidence already collected, capture the final
  verdict as a second Ask artifact, independently verify every blocking
  finding, and keep both PRs unmerged while the maintainer decides the repair
  scope.

## R-019 - Apply the focused repair and re-review PR #20

- **Attribute:** `[Original Request]`
- **Raw request:** "确认，采纳focused repair,完成两类修复和针对性验证,推送 PR #20 后重新调用 Claude review,将 MTP unified-family/interface 设计作为独立后续审计项。"
- **Captured intent:** Close the confirmed one-feature MoE exact-row and
  target-embedded MTP TP-consumer defects with the smallest root-cause
  changes, verify both behaviors directly, push the updated PR #20 branch,
  run a fresh Ask Claude review with the configured 20-minute timeout, keep
  both PRs unmerged, and defer MTP family/interface redesign to a separately
  scoped evidence audit.

## R-020 - Freeze approved PR #20 and advance PR #21

- **Attribute:** `[Original Request]`
- **Raw request:** "pr#20通过，暂不merge。现在推进pr#21"
- **Captured intent:** Treat PR #20 as maintainer-approved while leaving it
  open and unmerged. Move the active review and verification target to stacked
  PR #21, preserve its base on the PR #20 branch, and advance its
  profiling-envelope/MoE-EP changes toward a grounded merge-readiness
  conclusion without changing PR #20.

## R-021 - Run independent PR #21 reviews and concurrent Claude review

- **Attribute:** `[Original Request]`
- **Raw request:** "对pr#21进行2轮独立review，与此同时，后台调用ask claude（timeout limit为30mins）"
- **Captured intent:** Run two independent read-only review rounds against the
  pushed PR #21 stack, while concurrently invoking the local Ask Claude
  backend with a hard `1800s` timeout. Preserve all review evidence, keep both
  PRs open and unmerged, and treat the MTP unified-family/interface audit as a
  separate follow-up.

## R-022 - Adopt Option B with explicit discard visibility

- **Attribute:** `[Original Request]`
- **Raw request:** "采用方案b，但是适当强化丢弃反馈，必须输出warning等被丢弃的输出提醒。理由是，方案a可能会大幅度提升 用户对 参数修正 的耗时（以避免valueerror），这不合理。"
- **Captured intent:** For standard attention explicit decode KV values, filter
  each selected `(model, tensor_parallel_size)` target against its physical
  KV capacity and keep running when only that target drops a value. Every
  target-local explicit drop must emit a visible `RuntimeWarning` and a
  parent-process coverage summary that identifies the dropped KV values,
  dropped combination count, model, TP, physical capacity, and retained
  values. Union retained values across targets and raise `ValueError` only
  when no selected target can retain a requested explicit value. Keep the
  explicit runtime boundary fail-fast (`KV + 1 <= max_model_len`) and leave
  automatic standard decode sampling bounded by `max_seq_len - 1`. Scope this
  contract to standard attention; mixed and true-mixed target semantics remain
  separate follow-up decisions.

## R-023 - Reconcile the two PRs after the post-PR17 main movement

- **Attribute:** `[Original Request]`
- **Raw request:** Restore and deeply review PR #20 and PR #21 from
  `task_memory/task_2026-08-02_pr4_lookup_cache_review` and the scoped
  continuation, account for the merge of PR #17 into `main`, and prepare a
  complete conflict-resolution and merge plan without treating the old branch
  as current main.
- **Captured intent:** Use the post-PR17 integration worktree as the source of
  truth, preserve exact stage/KV provenance introduced by PR #17, and keep the
  two stacked PRs' ownership boundaries explicit.

## R-024 - Approve the typed-lane architecture direction

- **Attribute:** `[Original Request]`
- **Raw request:** After reviewing the EP=1 versus EP>1 behavior and the
  modularity/call-contract audit, approve Gate 2 option A and continue with the
  canonical typed-lane repair.
- **Captured intent:** Adopt the A' contract: one canonical lane constructor,
  immutable physical `EPLaneWorkload`, descriptor propagation through the EP
  plan/entity, predictor consumption of the descriptor, and scheduler ownership
  of runtime identity, events, and barriers. Keep EP=1 as a physical degenerate
  case of the same interface.

## R-025 - Write the RCA/design before implementing A'

- **Attribute:** `[Original Request]`
- **Raw request:** First land the complete RCA, problem analysis, and A'
  repair approach in the task documents. Then implement and verify the repair
  from that frozen documentation.
- **Captured intent:** Documentation is the first implementation sub-step.
  Production changes must follow TDD, remain modular, remove temporary probe
  escapes, and preserve the PR #17 runtime-identity boundary.

## R-026 - Confirm the shared MTP/MoE token-ledger repair

- **Attribute:** `[Original Request]`
- **Raw request:** Confirm the recommended repair: correct the shared
  `Batch.get_effective_total_tokens_for_compute()` contract so target-embedded
  MTP does not add `planned_draft_tokens` twice, and make structural MTP replay
  use the complete verification width rather than subtracting
  `rejected_draft_tokens`.
- **Captured intent:** Keep `planned_draft_tokens` as scheduling and acceptance
  metadata, use the target-forward physical width for compute and MoE gating,
  use `Batch.total_num_tokens` as the pre-routing materializer input, and keep
  routed assignment counts lane-local after top-k routing. Preserve explicit
  AFD/CUDA Graph padding semantics and the narrow A' physical-lane boundary.

## R-027 - Select narrow A' EP>1 aggregate admission

- **Attribute:** `[Original Request]`
- **Raw request:** "1，继续"
- **Captured intent:** Adopt narrow A'. Keep each concrete predictor's existing
  layer/cluster aggregate classification. Once a call is classified as routed
  MoE, require an explicit `EPLaneWorkload` for EP>1 in both dummy and
  non-dummy modes before timing, measurement, or lookup. Preserve
  attention-only, dense, EP=1, valid-lane, zero-routed-lane, and MTP explicit
  lane behavior. Implement the invariant through one shared protected helper,
  without synthetic descriptors, raw-map inference, caller-side duplicate
  guards, or temporary scaling.

## R-028 - Include SCOPE-036 mixed-layer dummy classification repair

- **Attribute:** `[Original Request]`
- **Raw request:** "A"
- **Captured intent:** Include the newly verified mixed-layer dummy predictor
  defect in the current PR20/PR21 merge-readiness scope. Make dummy execution
  timing consume the same concrete `layer_id` classification already used by
  the public predictor and non-dummy path. Dense layers must return dense
  execution semantics with zero MoE fields; actual MoE layers must preserve
  their existing MoE semantics. Keep `DECODE_ATTN` attention-only, preserve
  multi-layer aggregate behavior, avoid scheduler-side overrides and
  name-based classification, and verify both monolithic and disaggregation
  entry points before the final merge audit.

## R-029 - Select unified routed-aggregate fail-fast policy

- **Attribute:** `[Original Request]`
- **Raw request:** "A"
- **Captured intent:** Select direction A from the post-implementation PDD
  aggregate RCA. Treat an implicit multi-layer FFN aggregate as routed when
  the existing model-level classification is MoE, and require its physical
  `EPLaneWorkload` for EP>1 before dummy timing, measurement activation, or any
  model/backend/communication lookup. Keep dense and attention-only aggregates
  on their existing owners, use `is_moe_layer(layer_id)` for concrete
  single-layer classification, and fail explicitly when a concrete MoE config
  lacks that callable. This decision supersedes the earlier compatibility
  wording that left routed implicit aggregates on a lookup path without a
  physical lane.

## R-030 - Require strict descriptor/predictor topology consistency

- **Attribute:** `[Original Request]`
- **Raw request:** "A" after reviewing why an `EPLaneWorkload` descriptor and
  a predictor can carry different EP or router top-k values even when the
  runtime appears to use one model configuration.
- **Captured intent:** Include the descriptor/predictor topology contract in
  the current PR20/PR21 merge-readiness scope. A routed MoE admission must
  compare the physical descriptor's `moe_expert_parallel_size` and
  `router_topk` with the active predictor topology, and fail fast on any
  mismatch before dummy timing, measurement activation, model lookup,
  backend lookup, or communication lookup. Keep the check in the existing
  shared predictor admission owner; do not infer topology from names or raw
  maps, fabricate a descriptor, add a caller-side duplicate guard, or add a
  compatibility fallback. Preserve scheduler/materializer ownership of
  physical lane construction and keep EP=1 as the same typed interface's
  degenerate physical case.

## R-031 - Propagate active role topology through inherited MoE prediction

- **Attribute:** `[Original Request]`
- **Raw request:** "A"
- **Captured intent:** Adopt方案 A for the newly identified aggregate
  disaggregation topology-context gap. The active role's existing
  `moe_expert_parallel_size` and `router_topk` must travel through the full
  inherited `predict_moe_layer_time()` call chain, so every admission of one
  routed stage uses the same active topology. Keep strict fail-fast mismatch
  behavior, preserve the shared admission owner and role configuration as the
  source of truth, and avoid mutable-field overrides, synthetic descriptors,
  duplicate caller guards, fallbacks, or unrelated refactors.

## R-032 - Keep EP barrier identity and payload admission as separate owners

- **Attribute:** `[Original Request]`
- **Raw request:** "1，继续"
- **Captured intent:** Select option 1 for the residual EP consumer boundary:
  keep `_validate_ep_barrier_arrival()` responsible for scheduler identity,
  participant, and waiting-room invariants, while the existing
  `_get_step3_ep_alltoall_payload_bytes()` owner performs the final typed
  `EPLaneWorkload` admission. The final lane must be rejected before
  architecture collective resolution, communication predictor/backend lookup,
  trace publication, or waiting-room commit. Preserve the temporary waiting
  room state for an earlier valid lane, avoid duplicating descriptor checks at
  the barrier boundary, and keep both dispatch and combine on the same payload
  owner.
