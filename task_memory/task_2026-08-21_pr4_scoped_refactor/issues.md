## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-25 | Opened the post-implementation MTP scope audit and recorded the typed trace boundary evidence. |
| 2026-08-25 | Added the formal post-PR17 typed-lane RCA and reclassified the existing Gate 2 probe as provisional until the A' ownership and interface invariants are implemented. |
| 2026-08-25 | Resolved the Gate 2 regular-Batch EP workload contract and zero-routed shuffling admission defect; recorded unrelated broad-unit baseline failures. |
| 2026-08-22 | Archived the six deferred-WATCH probes and confirmed their scope and non-blocking status. |
| 2026-08-22 | Added the read-only producer-metadata and MTP interface audit dispositions and confirmed both are outside PR #21. |
| 2026-08-22 | Resolved the standard-attention Option-B discard-visibility issue and recorded the PR #21 review follow-ups and Claude timeout. |
| 2026-08-22 | Closed the PR #21 routing-count discrepancy as Torch-version-dependent evidence and recorded the synthetic-stack verification gate. |
| 2026-08-22 | Opened SCOPE-017 for the fresh post-repair Claude invocation that consumed its full 20-minute budget without a final verdict. |
| 2026-08-22 | Resolved SCOPE-014 and SCOPE-016 through the maintainer-approved focused repair; deferred MTP unified-family/interface design to a separate audit. |
| 2026-08-22 | Closed the Claude timeout issue after a successful resume, reopened SCOPE-014 with direct one-feature MoE evidence, and opened SCOPE-016 for the confirmed MTP TP-consumer half-migration. |
| 2026-08-22 | Opened SCOPE-015 after the bounded Claude review exhausted 20 minutes before emitting a verdict. |
| 2026-08-22 | Resolved SCOPE-014 with maintainer-selected universal default persistence and committed cache-round-trip evidence. |
| 2026-08-22 | Reopened producer completeness for finite exact rows after auditing the split PR #20 tip. |
| 2026-08-21 | Closed finite lookup, registry classification, and code-side sampling envelope issues; retained producer metadata and canonical-data gaps as explicit residuals; assigned a unique ID to the pre-existing logger issue. |
| 2026-08-21 | Closed the legal-EP policy gate with option A and recorded the resolver/CLI implementation. |
| 2026-08-21 | Recorded the MoE routing/load fixes and the unresolved legal-EP sampling-domain decision. |
| 2026-08-21 | Closed the confirmed mixed-Attention sampling gaps; retained the MoE routing/load portion as open. |
| 2026-08-21 | Resolved the compute-predictor MoE profiling import by extracting the pure feature object; retained CSV, non-KV-memory, and MTP exceptions. |
| 2026-08-21 | Recorded the A1 optional-metadata producer gap and clean-base logger attribution. |
| 2026-08-21 | Opened the scoped issue ledger and recorded root causes, scope decisions, and the initial seam gate. |
| 2026-08-21 | Reclassified profiling imports by migration cost and deferred the independent adapter pending proof of an owner gap. |
| 2026-08-21 | Recorded the confirmed staged boundary and resolved the operator-query naming/seam decision. |
| 2026-08-21 | Closed the function-level source audit and made the remaining file-map approval the only pre-edit gate. |

# Issues and Decisions

## Open issues

### SCOPE-018 - Exact skewed-route count varies across supported Torch builds

- **Status:** Resolved as an environment-specific measurement note; no code
  defect found.
- **Evidence:** The same PR #21 source, seed `42`, and
  `generate_expert_routing(512, 8, 2, "skewed")` produced `69/1024`
  expert-0 routes under `/usr/bin/python` with Torch `2.5.1+cu124` and
  `61/1024` under `aic-step-design` with Torch `2.12.1+cpu`. Both builds
  produced `423/512` hot rows for the `extremely_skewed` case.
- **Root cause:** `torch.multinomial` implementation/version differences
  change the exact seeded sample while preserving the declared distribution
  contract.
- **Impact:** A numeric probe that hard-codes one backend's exact route count
  would create a false regression signal. The production contract only
  requires valid dimensions, expert-0 reachability for the skewed distribution,
  and a populated hot subset for the extreme distribution.
- **Resolution:** Keep the existing contract tests and report the interpreter
  and Torch version with numeric evidence. No source change or canonical data
  change is justified.

### SCOPE-019 - Explicit standard-attention KV drops need visible feedback

- **Status:** Resolved by PR #21 commit `8cc267ea` under the maintainer-selected
  Option-B policy.
- **Evidence:** A standard explicit request with
  `max_seq_len=256`, `max_model_len=512`, and KV values `[255, 300, 511]`
  produced one `RuntimeWarning` for the capacity-`256` target. The warning
  named dropped values `[300, 511]`, dropped combination count `2`, model,
  TP size, physical capacity `256`, and retained value `[255]`. A second
  capacity-`512` target retained all three values without a discard warning;
  the retained union was `[255, 300, 511]`.
- **Root cause:** Target-local physical filtering previously had no explicit
  visibility contract, so a user could mistake omitted profiling rows for a
  complete measurement result and repeatedly edit parameters to avoid a
  global `ValueError`.
- **Impact:** A target with insufficient physical KV capacity can legitimately
  omit rows while another selected target still provides coverage. Raising on
  the first local miss would increase user iteration time and would reject a
  measurable multi-target request.
- **Decision:** Use Option B. Filter each `(model, TP)` target independently,
  emit a `RuntimeWarning` before worker dispatch for every target-local
  explicit drop, print the requested/retained union and target capacities, and
  raise only when a requested value is absent from every target.
- **Runtime boundary:** Explicit standard decode values still fail fast when
  `kv_cache_size + 1 > max_model_len`; automatic standard decode remains
  bounded by `max_seq_len - 1`. Mixed/true-mixed capacity semantics are not
  widened by this issue resolution.

### SCOPE-020 - Concurrent Claude review timed out before PR #21 verdict

- **Status:** Resolved as an inconclusive external-review artifact; no
  approval or rejection is inferred.
- **Evidence:** The requested Ask Claude command used `claude-opus-5`,
  `effort=max`, and an outer `1800s` timeout against PR #21 head
  `8cc267ea`. The raw log is
  `/data/ycfeng/tmp/ask_claude_pr21_optionb_20260822.raw.log`; GNU `timeout`
  returned `143` from the child, and Claude emitted no final review body.
- **Root cause:** The parent review process exhausted its time budget during
  delegated inspection before final synthesis.
- **Impact:** This invocation cannot be cited as `APPROVE`, `REQUEST CHANGES`,
  or `COMMENT`; the independent local Round-1/Round-2 review artifacts remain
  the only completed review evidence.
- **Resolution:** Archive the timeout and keep PR #20 and PR #21 open and
  unmerged. Any resume or longer external review requires a new explicit
  maintainer request.

### SCOPE-015 - Claude PR #20 review timed out before verdict

- **Status:** Resolved by the maintainer-requested resume invocation.
- **Evidence:** The exact required command ran for `1200s` with Claude Opus 5
  and `effort=max`. StepCode recorded `55` tool calls, `0` tool failures, and
  no final assistant response before GNU `timeout` returned `124` and the
  child exited `143`. The follow-up resumed conversation
  `40e27923-c316-4f00-806b-b5500569309c`, exited `0`, and returned
  `REQUEST CHANGES`.
- **Root cause:** The complete 23-file PR investigation consumed the entire
  configured review budget before Claude reached its synthesis/final-output
  stage. The visible `401 Unauthorized` occurred afterward in StepCode's
  optional feedback collector (`Invalid web access key`) and did not terminate
  the review.
- **Resolution:** The bounded resume prompt directed Claude to synthesize the
  evidence already collected rather than restart broad investigation. The
  final artifact is
  `.worktrees/pr4-lookup-registry-core-20260822/.omx/artifacts/ask-claude-pr20-review-resume-20260822-130144.md`.
  Independent local probes confirmed both P1 findings before escalation.

### SCOPE-014 - Finite exact-row persistence is only partially admitted

- **Status:** Resolved by commit `3e9c0374`; verified again at final pushed
  HEAD `18d1a23e`.
- **Evidence:** At PR #20 head `53499911`, `_train_model` and
  `_train_single_model` default `persist_exact_lookup=False`. Explicit
  persistence exists for mixed Attention, true-mixed decode, MLA, CPU
  overhead, load-aware MoE, and KV-cache-save models. Ordinary Linear,
  standard Attention prefill/decode, and one-feature MoE training calls still
  omit it. A real two-row estimator probe observed no
  `_frontier_exact_lookup` under the default and the exact map
  `{(1.0,): 9.0, (2.0,): 4.0}` under the explicit path. Commit `d45d836c`
  changed both entry-point defaults to `True`, while
  `shared_prediction_model_manager.py:1050` and
  `sklearn_moe_execution_time_predictor.py:966` still explicitly pass
  `len(feature_cols) > 1`. The direct one-feature MoE replay therefore
  observed no exact map and returned `99.0` for a measured row whose value is
  `4.0`; the absolute error is `95.0` and the relative error is `2375%`.
- **Root cause:** Commit `653f8526` encoded persistence eligibility as
  feature-count greater than one at the two MoE producer call sites. Commit
  `d45d836c` corrected the unified defaults but left those explicit arguments
  in place, so they override the selected universal scalar-numeric policy.
- **Impact:** A cached one-feature MoE estimator can return its regressed or
  stale runtime-cache value at a measured feature key and lose the required
  `exact measured -> runtime cache -> model` precedence across process/cache
  boundaries.
- **Decision:** The maintainer selected option A. Both unified runtime
  training/cache entry points persist scalar-numeric measured rows by default;
  the two current one-feature MoE producers belong to that policy.
- **Resolution:** Removed the two feature-count overrides so both current MoE
  producers use the unified scalar-numeric persistence default. The explicit
  opt-out parameter remains available at the unified training/cache entry
  points for a future producer with an unsupported key schema.
- **Verification after repair:** The producer and cache-round-trip regressions
  pass. Reloaded exact rows are
  `{(1.0,): 9.0, (2.0,): 4.0}`; the measured query returns `4.0` ahead of
  stale `99.0`, calls the estimator `0` times, and has absolute error `0.0`.
  See `test_report_2026-08-22_pr20_focused_repair.md`.

### SCOPE-016 - Target-embedded MTP TP classification bypasses its registry

- **Status:** Focused defect resolved by commit `18d1a23e`; broader
  unified-family/interface design is deferred to a separate audit.
- **Evidence:** The authoritative declaration is
  `frontier/spec_decode/mtp_registry.py:44-47`, exposed by
  `get_target_embedded_mtp_linear_ops()` at lines `169-170`.
  `SklearnExecutionTimePredictor._get_linear_op_tp_key`,
  `ExecutionTimePredictionModelManager._get_linear_op_tp_key`, and
  `LinearOpTrainer._get_training_tp_key` instead repeat the local set
  `{"mtp_fusion_proj", "lm_head_linear"}`. An in-process registry extension
  added `mtp_registry_extension_probe`; all `3/3` TP consumers rejected it
  with `ValueError: Unsupported linear op for TP mapping`.
- **Root cause:** Wave C commit `9c09ac74` migrated ordinary TP classification
  to `resolve_operator_query_tp_mode` but retained and moved the target-
  embedded MTP special case as a local literal branch. The existing structural
  MTP registry accessor already owns this membership, so the migration left
  one sibling category on a second source of truth.
- **Impact:** Adding a target-embedded MTP linear operation to its registry
  updates the profiling plan but leaves predictor, shared-manager, and trainer
  TP selection stale. The new operation then fails during training or runtime
  initialization until three distant files repeat the new magic string.
- **Option A — focused TP-consumer repair (recommended):** Import and use
  `get_target_embedded_mtp_linear_ops()` in the three TP-key consumers and add
  one extension regression to
  `tests/unit/test_operator_query_tp_consumers.py`. This directly closes the
  reproduced `3/3` failure while preserving the distinct enumeration,
  CSV-schema, profiling-kernel, quantization, and runtime-policy roles of other
  same-name occurrences.
- **Option B — broad MTP consumer unification:** Audit and migrate every
  occurrence that enumerates these two names, including model-name lists,
  required CSV columns, quantization defaults, and profiling implementation
  branches. This can reduce more duplication, while it expands the repair
  across shared interfaces and more than the current four-file sub-step. It
  requires a separate evidence pass and cross-cutting approval.
- **Decision:** The maintainer selected option A. The three TP-key consumers
  now obtain membership from `get_target_embedded_mtp_linear_ops()`. The
  registry-extension regression resolves TP value `2` in predictor, shared
  manager, and trainer, so accepted consumers changed from `0/3` to `3/3`.
  Option B remains outside PR #20 and is recorded in `future.md` as the MTP
  unified-family/interface audit.

### SCOPE-017 - Fresh post-repair Claude review timed out before verdict

- **Status:** Open external-review limitation; the requested invocation
  completed, while the requested final verdict remains unavailable.
- **Evidence:** The fresh command used StepCode Claude Opus 5,
  `effort=max`, and outer `timeout 1200s` against
  `origin/main...18d1a23e`. GNU `timeout` returned `124`; StepCode forwarded
  `SIGTERM` and the child exited `143`. Session
  `703700ab-27ef-433f-9a9d-405a389c8525` recorded one prompt, seven tool
  calls, one failed late reviewer call, and no final assistant response.
- **Root cause:** Claude delegated two long focused audits and then started a
  third rest-of-diff audit near the end of the parent budget. The parent
  session exhausted all 1200 seconds before synthesizing the completed
  evidence into a verdict.
- **Impact:** This fresh invocation cannot be cited as `APPROVE`,
  `REQUEST CHANGES`, or `COMMENT`. The code remains independently verified,
  and PR #20 remains unmerged pending maintainer review.
- **Alternatives:** With explicit maintainer authorization, resume the
  preserved Claude conversation using a synthesis-only prompt, or run a
  longer external review. The current task does not silently extend the
  requested 20-minute budget.
- **Artifact:**
  `.worktrees/pr4-lookup-registry-core-20260822/.omx/artifacts/ask-claude-pr20-focused-repair-rereview-20260822-145617.md`.

### SCOPE-011 - Optional lookup-domain metadata has no current-main producer

- **Status:** Open; producer-side follow-up after the scoped A1/B
  implementation.
- **Evidence:** A source inventory on current `origin/main` finds producer
  writes for `_frontier_feature_names`, `_frontier_model_hash`, and
  `_frontier_exact_lookup`, but no producer writes for `_feature_bounds`,
  `_feature_domain`, `_feature_constraints`, or `_identity`.
- **Root cause:** The A1 runtime validator can consume an explicit domain
  descriptor, while the existing training/descriptor builders still expose
  only feature schema, model hash, and measured-row lookup metadata.
- **Impact:** A1 can reject malformed explicit metadata, but it cannot claim
  that runtime physical bounds or artifact identity are already published and
  enforced end to end.
- **Resolution:** Keep these fields optional in A1. Add a concrete producer and
  persisted-cache/reload verification before making any field mandatory. Do not
  grow this into a new Contract/Data Guarantee layer.

### SCOPE-021 - Producer descriptor and standalone trainer metadata are incomplete

- **Status:** Open; deferred outside PR #21.
- **Evidence:** A read-only AST audit at PR #20 head `18d1a23e` and PR #21
  head `8cc267ea` found zero production assignments for
  `_feature_bounds`, `_feature_domain`, `_feature_constraints`, `_identity`, or
  `_model_identity`. Runtime `model_info` builders pass feature names, model,
  exact lookup, and model handles, but do not pass those optional fields.
  `frontier/training/base_trainer.py` still pickles the raw estimator without
  the `_frontier_*` descriptor metadata.
- **Root cause:** Optional consumer validation was implemented before a
  producer-to-descriptor-to-cache contract was published. The standalone
  trainer path also predates the unified metadata-bearing producers.
- **Impact:** The optional checks are available for explicitly supplied
  metadata, but production caches do not yet prove physical-domain or identity
  coverage after reload.
- **Decision:** Keep the fields optional and leave PR #21 unchanged. A future
  repair must define one descriptor schema, populate it in unified and
  standalone producers, pass it through runtime `model_info`, persist it, and
  verify consumer behavior after cache reload. This is a shared data-contract
  change and requires a separate scope decision.

### SCOPE-022 - MTP registry is dedicated, not fully unified

- **Status:** Open; independent follow-up audit, not a PR #20/21 blocker.
- **Evidence:** `mtp_fusion_proj` and `lm_head_linear` are separate physical
  `ColumnParallelLinear` operators with independent timing names. They are
  declared in `frontier/spec_decode/mtp_registry.py`, are absent from the
  generic `OperatorFamilySpec` registry, and `bind_operator_query()` rejects
  them as unknown. The three repaired TP consumers share the dedicated MTP
  accessor, but model-name enumerators and an import-time profiling-plan
  snapshot still use the two-name literal.
- **Root cause:** The focused repair unified TP membership reads without
  changing the distinct MTP structural, profiling, and runtime-policy
  catalogs.
- **Impact:** The current two target-embedded MTP operators behave correctly,
  while adding a new MTP operator still requires more than one registry
  change. The method contract tuple and module-level helper can also diverge
  under independent mutation.
- **Decision:** Keep the focused TP repair in PR #20 and defer a full
  `OperatorFamilySpec`/MTP interface migration. The follow-up must trace
  physical enumeration, CSV schema, profiling kernels, quantization,
  runtime policy, and all consumers before proposing a cross-cutting change.

### SCOPE-023 - Deferred PR #21 WATCH items are real but outside the repair

- **Status:** Open; individually verified and deferred.
- **Evidence and disposition:**
  - **True-mixed rounding:** aggregate-token filtering can retain a row that
    per-sequence ceil-to-block allocation cannot fit. With
    `max_seq_len=256`, `max_model_len=512`, block size `16`, and capacity
    `336`, the aggregate predicate retained six rows while the block-safe
    predicate retained five; the wrapper reproduced `22 > 21` blocks. This
    predates the Option-B filter and needs a separate block-aware contract.
  - **Duplicate targets:** duplicate `(model, TP)` inputs are overwritten in
    tuple-key metadata and undercount `total_work`, while raw worker loops can
    repeat the work and overwrite output paths. This is a CLI normalization
    decision, not an explicit-KV legality defect.
  - **CLI derivation:** parser defaults of `4096` make
    `--max_seq_len 8192` fail the existing `max_model_len >= max_seq_len`
    check; release wrappers already derive and pass both values. This is an
    ergonomics follow-up.
  - **Strict integers:** direct helper APIs use `int()` and silently turn
    values such as `511.9` into `511` and `True` into `1`; CLI parsing mostly
    rejects these first. This needs a separate API-contract decision.
  - **Automatic exception catches:** automatic online-grid points catch
    `RuntimeError`/`ValueError` and skip them, while explicit points re-raise;
    fault injection confirmed automatic failures can be silently omitted.
    Narrowing catches requires a structured skip-reason/coverage contract.
  - **Token-list capacity:** exact `num_tokens_list` values enforce
    `max_num_tokens`, while positive `extra_num_tokens` may exceed it and are
    passed to wrappers. Existing tests intentionally exercise the extended
    domain, so truncation would change a declared contract.
- **Decision:** Keep all six items out of PR #21. Each requires a separate
  design or interface decision; none invalidates target-local warning
  feedback, retained union, or all-target fail-fast behavior.

### SCOPE-013 - Attention logger capture failures predate A1

- **Status:** Confirmed pre-existing; deferred from A1.
- **Evidence:** The two attention total tests fail on both the A1 worktree and
  clean `origin/main` with `caplog.records == []`, while stdout reports
  `total_attention_time_ms=4.610000` and `6.610000` for the two cases.
- **Root cause:** `frontier.logger` configures the Frontier root logger with
  `propagate=False`, so pytest's `caplog` handler does not receive the records.
- **Impact:** The combined regression command reports two failures even though
  the direct numeric trace and decode component are correct.
- **Resolution:** Leave logging untouched in A1; track a separate test-harness
  repair only if the maintainer explicitly expands scope.

### SCOPE-001 - Residual finite-lookup mismatch

- **Status:** Resolved by Waves A1/A2 and the finite Attention/MoE migrations on 2026-08-21.
- **Root cause:** Runtime key construction and finite prediction materialization
  do not share one explicit per-operator domain and resolution contract.
- **Evidence:** The direct lookup regressions now cover Linear, Attention
  decode/prefill, MoE gating/shuffling/grouped-GEMM, and the high-dimensional
  on-demand path. A finite miss reaches the model once, repeats hit the
  process-local cache, and the grouped-GEMM query preserves the original
  in-flight token count.
- **Impact resolved:** A legal finite miss no longer aborts before the
  estimator is consulted, while illegal or malformed inputs fail before model
  access.
- **Resolution:** One validated lookup path now enforces
  `exact measured -> runtime cache -> model`, rejects invalid outputs, and
  keeps measured rows read-only. See
  `test_report_2026-08-21_wave_e_direct_cases.md` and the 111/265-test
  regression matrices.

### SCOPE-002 - Legacy PR #4 fallback is unsafe

- **Status:** Confirmed; do not port verbatim.
- **Root cause:** Dictionary absence was treated as sufficient authority for
  model prediction; invalid output was clamped and the finite table was mutated.
- **Impact:** Illegal shape, stale identity, schema drift, and legal coverage
  gaps become indistinguishable.
- **Resolution principle:** Validate first; reject invalid output; keep runtime
  memoization separate from measured data.

### SCOPE-003 - Runtime imports profiling implementation

- **Status:** Resolved for the compute-predictor path; named exceptions remain
  explicit and separately scoped.
- **Root cause:** Predictor, shared manager, MoE feature path, scheduler, and
  MTP runtime import modules under `frontier.profiling`.
- **Impact:** Benchmark/GPU implementation imports can make runtime behavior
  depend on the profiling environment. A literal zero-import migration would
  also move shared CSV invariants and create broad churn.
- **Resolution:** Runtime predictor modules now import the pure
  `frontier.moe_load_imbalance` feature module rather than the mixed profiling
  input file. The profiling module re-exports the same class for compatibility.
  Side-effect-free CPU/PP schema/validation helpers remain an explicit
  allowlist; benchmark runners, GPU wrappers, and profiling CLIs remain
  producer-only. Non-KV-memory estimation and MTP structural config remain
  named exceptions.
- **Verification:** `tests/unit/test_profiling_runtime_boundary.py` passed `3`
  focused tests; the runtime/MoE regression set passed `66` and the profiling
  compatibility set passed `39`.

### SCOPE-008 - All-or-nothing dependency removal is over-broad

- **Status:** Confirmed process/design finding.
- **Evidence:** `frontier/training` has no direct Python import of
  `frontier.profiling`; CPU/PP helpers are shared by producer and consumer;
  `common/model_config.py` is 570 lines with about 18 consumers; the non-KV
  estimator pair is 1,546 lines and is already an explicit exception.
- **Root cause:** The earlier design grouped pure CSV readers, runtime feature
  derivation, GPU measurement, and structural model configuration under one
  "profiling implementation" label.
- **Resolution:** Apply a staged boundary by side effect and migration cost,
  not by package name alone. No broad relocation is approved in the current
  phase.

### SCOPE-004 - Operator admission is shape-driven

- **Status:** Resolved for the scoped PR by Wave C and the focused
  `SCOPE-016` repair.
- **Root cause:** Reachable `startswith("attn_")` branches and duplicated
  literal sets select TP/owner behavior after a registry already exists.
- **Impact:** Unknown names can receive a plausible but wrong TP or dataset
  route; adding an operator requires several files.
- **Resolution to retain:** `bind_operator_query` and the ordinary TP consumer
  resolver use
  exact unified-registry membership and explicit architecture metadata;
  unknown, ambiguous, or conflicting declarations fail before routing.
- **Resolution evidence:** The three target-embedded MTP TP-key consumers now
  use the existing `mtp_registry` accessor. A registry extension reaches all
  `3/3` consumers without another local-name edit.

### SCOPE-005 - Independent profiling-facing adapter is not justified

- **Status:** Resolved by maintainer decision on 2026-08-21.
- **Evidence:** `OperatorSpec` already declares `profiling_target`,
  `profiling_key`, and `tp_mode`; family registries and attention/MoE resolvers
  already provide exact membership and schema APIs. Profiling output is a CSV,
  not an object consumed by a second registry. The old R-025 timing catalog
  (`frontier/profiling/operator_identity.py`) is absent from current
  `origin/main` and belongs to the scope-crept branch.
- **Root cause of the earlier proposal:** Timing aliases and pseudo-model names
  were mistaken for proof that a new owner catalog was required.
- **Resolution:** Reuse the existing unified registry with the
  `operator_query_binding` / `bind_operator_query` seam. Add an explicit
  mapping table only when current registry metadata cannot express a concrete
  physical-to-timing mapping. Do not add an independent profiling-facing
  adapter unless that owner gap is demonstrated during implementation.

### SCOPE-009 - Binding name and placement ambiguity

- **Status:** Resolved by maintainer decision on 2026-08-21.
- **Root cause:** `exact resolver` described an implementation action but not
  the domain responsibility, while unqualified `operator_binding` could be
  confused with the existing model-to-family `FamilyBinding`.
- **Resolution:** Use `operator_query_binding` as the conceptual module name
  and `bind_operator_query` as the API name. Extend the existing
  `frontier/operators/binding.py` first; split a separate file only if actual
  implementation size or import structure proves the file is overloaded.

### SCOPE-006 - Profiling domain may be narrower than runtime domain

- **Status:** Code envelope resolved; canonical measured-data coverage remains
  open and is explicitly deferred to offline profiling.
- **Root cause:** Config-derived token grids and multi-feature Attention/MoE
  runtime combinations are not proven to share one coverage envelope.
- **Impact:** A simulator can generate a valid key absent from CSV/model lookup.
- **Resolution principle:** Deterministic per-family sampling with
  `Profiling Domain >= Runtime Domain`, physically bounded and reported.
- **Attention resolution evidence:** The primitive token/sequence/chunk/batch
  endpoints and the mixed-Attention legacy, online-grid, and true-mixed
  endpoint/union paths now have focused RED/GREEN coverage. Direct probes show
  `max_context=1000` with `kv_max=992`, complete default-plus-explicit online
  axes, and true-mixed prefill/decode maxima `5000/4999`; all emitted rows are
  valid and deterministic.
- **Resolution evidence:** Primitive and mixed-Attention endpoints, MoE route
  identity/load edge cases, and the all-legal-divisor EP policy now have
  focused RED/GREEN tests and direct probes. The code-side sampling envelope
  therefore reaches the tested runtime endpoints.
- **Remaining gap:** Existing canonical CSVs still lack newly exposed EP
  rows. A full measured-data superset claim requires an offline profiling run
  and publication, which is outside this code-only PR.

### SCOPE-012 - Runtime-legal MoE EP divisors are not represented consistently

- **Status:** Resolved by maintainer decision and implementation on 2026-08-21.
- **Evidence:** `ReplicaConfig.__post_init__` accepts every positive
  `moe_expert_parallel_size` that divides `total_expert_num`. For the checked-in
  `qwen2_moe_example` model (`60` experts), the legal set is
  `{1,2,3,4,5,6,10,12,15,20,30,60}`. In contrast,
  `get_default_moe_profiling_config()` enumerates only divisor candidates
  `[1,2,4,8]`, yielding local widths `[60,30,15]`, and the actual profiling CLI
  defaults `--expert_parallel_sizes` to `[1]`. The canonical A100 qwen2 CSV
  contains `524` rows with EP `{1,2}`.
- **Root cause:** Runtime validation, the convenience profiling helper, the
  profiling CLI default, and published data use different EP policies. No
  single registry or declared supported-EP contract connects them.
- **Impact:** A runtime configuration such as qwen2 EP=3 is mathematically
  valid but has no default profiling plan or checked-in measured rows. A local
  helper-only edit would not affect the CLI, while a CLI/runtime policy edit
  changes the supported parallelism contract and measurement cost.
- **Decision:** The maintainer selected option A. The profiling helper now
  derives local expert widths from every positive divisor, and the CLI resolves
  an omitted `--expert_parallel_sizes` per model through the same resolver.
  Explicit lists remain accepted only when every value is a legal divisor;
  example scripts continue to pass explicit small lists for bounded smoke runs.
- **Residual data work:** Existing canonical CSVs remain unchanged by scope.
  They must be regenerated by the offline profiler before a runtime
  configuration using a newly covered divisor can claim measured-data coverage.

### SCOPE-010 - Ordinary operator classification remains duplicated

- **Status:** Resolved for ordinary operators and target-embedded MTP TP
  membership.
- **Root cause:** Four TP consumers still classify by `startswith("attn_")`,
  while `profiling/linear_op/profiling_plan.py` keeps a second literal
  attention catalog. MoE load-imbalance mode also has a local two-name set.
- **Impact:** A new ordinary operator can require edits outside the unified
  registry, and an unknown name can receive a plausible TP/dataset route.
- **Resolution to retain:** The ordinary TP consumers, attention policy, and
  profiling-plan enumeration consume the unified registry/architecture
  profile. The existing structural MTP registry remains the correct owner for
  target-embedded MTP names.
- **Resolution evidence:** The three MTP TP-consumer copies were replaced by
  `get_target_embedded_mtp_linear_ops()`. Wider MTP family/interface
  consolidation remains a separate future design audit rather than part of
  this focused defect repair.

### SCOPE-007 - Current and old branch evidence must not be conflated

- **Status:** Process issue; controlled.
- **Root cause:** The old worktree contains N3/N4/N5 and descriptor artifacts
  that do not exist on `origin/main`.
- **Resolution:** This task cites old artifacts only as evidence and bases any
  implementation plan on current-main files; no old branch cherry-pick.

### SCOPE-018 - Regular-Batch MoE workload domain and zero-routed lane admission

- **Status:** Resolved in the local Gate 2 integration worktree; commit and
  stacked-PR refresh remain pending review.
- **Observed behavior:** On-demand `moe_shuffling` previously required an
  `EPBatchGroup`, so regular `Batch` callers failed before model invocation.
  Removing that guard without a lane adapter would have used global expert
  width for EP-local profiling. After typed lane materialization was added,
  a complete global map with one positive lane and one all-zero lane still
  caused the zero lane to query the model with `total_routed_tokens=0`.
- **Root cause:** Workload-domain identity was encoded in the entity class,
  and shuffling admission checked only `len(per_expert_tokens)`, not the
  physical routed-token sum.
- **Impact:** Regular PREFILL/DECODE callers could fail despite valid routing;
  a zero lane could query outside the measured profiling domain and a model
  trained only on positive routed loads could fail or return an invalid value.
- **Resolution:** `EPLaneWorkload` carries `ep_id`, EP topology, global expert
  IDs, source batch identity, target replica, and layer identity. Complete
  global maps split through canonical contiguous ownership; EP=1 is represented
  as the unique local lane; partial maps without identity fail fast. Shuffling
  validates non-negative counts and returns zero for a lane whose routed-token
  sum is zero before model lookup. Grouped GEMM already follows the same
  zero-sum rule.
- **Evidence:** RED reproduced a zero-load model call; GREEN captured one
  positive-lane call (`8` routed tokens) and no zero-lane call. Focused and
  PR17-sensitive matrices passed as recorded in `progress.md`.

### SCOPE-019 - Broad unit baseline failures during Gate 2 verification

- **Status:** Controlled, non-blocking for this integration slice.
- **Observed behavior:** `tests/unit` produced `18` failures alongside `2776`
  passes, `25` skips, and `1` xfail.
- **Root cause:** The failures reference missing release/debug scripts,
  stale top-level README contract text, unavailable `config_optimizer`, and
  optional analysis CLI dependencies already absent from post-PR17 main.
- **Impact:** A clean all-unit verdict is unavailable on this checkout, but
  the failures do not import or exercise the six typed-lane files changed in
  Gate 2.
- **Resolution:** Keep the failures out of the PR20/PR21 merge fix scope;
  preserve the exact failure list as baseline evidence and rely on the focused
  plus PR17-sensitive matrices for this change.

### SCOPE-024 - Post-PR17 EP lane contract split (formal RCA)

- **Status:** Open at the docs-freeze checkpoint; the approved A' design is the
  resolution target. The current six-file dirty diff is a behavior probe, not
  a mergeable implementation.
- **Observed behavior:** The physical EP protocol is already the same for
  `EP=1` and `EP>1`: materialize a layer workload, perform pre-dispatch and
  dispatch, run routed expert compute, combine, post-combine, and close a
  participant barrier. `EP=1` simply has one participant, full local expert
  width, and zero communication. The scheduler's canonical materializer and
  wave/barrier code demonstrate this symmetry in
  `frontier/moe_ep_workload.py` and
  `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`.
- **Observed software split:** Ordinary `Batch` callers carry a global expert
  map, while scheduler-created `EPBatchGroup` callers carry a local map. The
  predictor currently decides which domain it received from the entity type
  and reconstructs maps in several methods
  (`sklearn_moe_execution_time_predictor.py` and
  `sklearn_disaggregation_execution_time_predictor.py`).
- **Root cause:** A physical workload descriptor is missing at the boundary.
  The same untyped `dict` is used for aggregate routing, one lane's local
  routing, and a compatibility input. Consequently the predictor has to infer
  topology, lane identity, and token domain from incidental shape or class.
- **Width defect:** `len(per_expert_tokens)` is not the physical local expert
  width. Sparse maps omit zero-load experts, and a global map has width
  `total_expert_num` while a lane model has width
  `total_expert_num / moe_expert_parallel_size`. `EP=1` masks this because the
  two widths coincide; `EP>1` sends the wrong feature width to profiling and
  can select an incompatible model row.
- **Conservation defect:** Global routing conservation is over all experts and
  all lanes. Local predictor features and lane barriers are over one owned
  expert set. Reusing one raw map for both domains loses the lane partition and
  permits a partial map to be interpreted as a complete global workload.
- **Zero-lane defect:** A physical lane with a complete zero token vector is a
  required participant in dispatch/combine barriers, but routed-dependent
  shuffling lookup has no positive-load profiling row. Checking map presence or
  length does not distinguish a zero-routed lane; the predictor must return the
  routed phase's identity value (`0.0`) without model access.
- **Identity coupling defect:** The provisional probe puts `source_batch_ids`,
  `target_replica_id`, and `global_layer_id` into a predictor value object and
  also attempts to use them as scheduler provenance. PR #17 already assigns
  exact stage/KV provenance to `BatchStage.attach_runtime_identity()` and the
  scheduler ledger. Duplicating admission tickets, epochs, stale-wave IDs, or
  metrics operation IDs in the physical descriptor would create two owners and
  could break rollback and exact KV transfer matching.
- **Impact:** The current shape/type split is permissive for EP=1 and fails
  late or silently for EP>1. It can produce wrong profiling features, model
  lookup misses, zero-load model queries, incomplete participant barriers, and
  identity drift across PR #17's stage lifecycle.
- **Resolution target:** `LayerEPWorkload` remains the sole aggregate owner and
  exposes one canonical constructor for immutable `EPLaneWorkload`. The lane
  descriptor carries only physical topology and routing data; the EP plan and
  entity retain that descriptor; predictor and communication consumers read it
  through typed accessors; scheduler context owns all lifecycle identity and
  barriers. Sparse maps are densified at this seam, zero lanes remain in the
  participant set, and both EP modes use the same interface.

### SCOPE-025 - Provisional probe contains A' boundary violations

- **Status:** Open; must be removed during implementation.
- **Second owner:** The probe lets callers pass both `lane_workload` and a raw
  map, then compares them in predictor helpers. That keeps two mutable input
  contracts alive instead of making the descriptor the canonical state.
- **Escape hatch:** `validate_expert_width=False` and the EP=1 raw-map branch
  allow a map to bypass topology validation. This recreates the exact
  EP=1-only compatibility path that A' explicitly removes.
- **Runtime identity leak:** The probe's lane descriptor carries source,
  replica, and layer provenance and synthesizes those values from predictor
  call arguments. This is incompatible with the scheduler-owned PR #17 stage
  identity boundary.
- **Half-migrated interface:** The probe adds optional parameters to selected
  concrete methods while leaving base signatures, stage callers, communication
  payload readers, mocks, and entity plans on raw tuples/dicts. A new caller
  would need to know which sibling path accepts the descriptor.
- **Synthetic MTP entity:** The existing MTP path constructs an artificial
  `EPBatchGroup` and `Request` merely to reuse stage prediction. That gives a
  physical phase calculation a fake scheduler identity and makes MTP depend on
  entity constructors and lifecycle fields.
- **Resolution:** Implement one descriptor contract end to end, preserve only a
  read-only compatibility projection for existing communication/metrics code,
  keep `predict_stage_execution_time()` stable for dense callers, and move MTP
  to a pure descriptor/phase seam. Delete the escape hatch and reject a raw
  local map without an explicit typed lane.

### SCOPE-026 - Generic MTP synthetic-shape path exceeds the implemented A' seam

- **Status:** Decision required before the remaining implementation; this is a
  real scope fork, not a test-only discrepancy.
- **Evidence:** The MoE-specific decoder hook now materializes one
  `EPLaneWorkload` per participant and calls `predict_moe_lane_phase_times()`
  (`frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:2558-2670`).
  The outer target-embedded MTP replay still constructs a block-shaped
  `Batch` in `_build_mtp_synthetic_batch()`
  (`sklearn_execution_time_predictor.py:5107-5175`) and a terminal `Batch`
  plus copied request progress in `_get_mtp_terminal_overshoot_time()`
  (`sklearn_execution_time_predictor.py:5270-5347`). Those objects carry
  per-block token counts and speculative metadata into the existing predictor
  API; they are not admitted to the scheduler or event queue and do not carry
  EP lane identity.
- **Root cause:** The approved A' text used “pure MTP” for the whole structural
  replay, while the implementation only replaced the physical MoE lane
  sub-path. The generic predictor API still accepts `Batch` because attention,
  token-shape, terminal-progress, and metadata calculations are coupled to
  that existing entity protocol.
- **Option A - narrow A' seam (recommended):** Define the A' gate as the
  physical MoE/EP phase path. Keep the generic block-shaped `Batch` replay as
  an explicitly named, scheduler-independent MTP shape adapter; prohibit
  synthetic `EPBatchGroup`, lane identity, and lifecycle fields. Update the
  design/harness/plan wording and tests to distinguish these contracts. This
  keeps the repair within the already verified MoE predictor seam and avoids a
  new shared predictor interface.
- **Option B - full pure MTP interface:** Introduce an immutable MTP phase
  descriptor carrying token shape, active-request count, terminal progress, and
  metadata; refactor the generic and MoE predictor paths to consume it without
  any synthetic `Batch` or copied `Request`. This satisfies the broad wording,
  but changes shared predictor contracts and the terminal speculative replay
  path across more than the current A' file map. It needs a separate RED test,
  interface review, and a larger compatibility audit.
- **Trace finding:** `_log_ep_workload_trace()` currently accepts a raw map but
  only validates/serializes it for observability; the two scheduler calls and
  the direct event call pass the descriptor-backed projection. Converting this
  helper to accept `EPLaneWorkload` is a localized completion of the A' owner
  boundary and is independent of the MTP fork.
- **Escalation:** Do not change the shared MTP interface until the maintainer
  selects Option A or Option B. The recommendation is Option A, followed by
  the typed trace helper migration and focused verification.

## Explicitly deferred questions

- Whether runtime's non-KV-cache-memory estimator remains a permanent import
  exception or moves behind a file/owned runtime API.
- Whether missing exact D24 identity for registered MLA operators is addressed
  in this PR or only represented as an explicit owner state.
- Whether the full duplicated profiling-plan/trainer catalog migration follows
  immediately or is a small follow-up after the resolver lands.
- Whether any concrete timing-only owner remains unrepresentable by the existing
  `profiling_key`/family registry after the `bind_operator_query` slice; a
  minimal bridge is deferred until that evidence exists.

## Resolved principles ported from the old task

- No nearest-row substitution, clamp, default, silent fallback, or profiling
  CSV write-back.
- A legal miss may use the canonical estimator only after identity/schema/
  physical validation.
- Runtime cache stores only validated model predictions.
- Cache identity must include timing determinants and exclude non-performance
  publication metadata.
- Optional CPU/PP missing-profile zero behavior remains intentional.
