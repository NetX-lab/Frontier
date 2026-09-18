## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded PR33 remediation, explicit R01-A decision and final verification status. |
| 2026-09-17 | Recorded the new candidate code-quality task and preserved behavior boundaries. |

# Requirements

## [Original Request]

> this is a new task. You are working in the current Frontier feature worktree. Compare it against the current `main` branch and perform a deep code-quality review and cleanup of the code introduced or materially refactored by this worktree.

> This is a code inspection + refactoring + verification task. Do not stop after generating a review report.

> The target is clear invariants, explicit ownership, centralized policy, minimal accidental state, minimal defensive branching, reuse of existing Frontier abstractions, and code that can be understood locally without knowing a collection of hidden fallback rules.

The original request requires the following, in order:

1. Read AGENTS.md and current records; pin main and pre-refactor candidate; inventory every added/materially modified file from the actual three-dot diff and rank runtime impact and quality risk.
2. Review runtime-critical simulator, schedulers, predictors, attention/layer binding, GDN state/guards, memory planner, MoE, timing entities and configuration binding. Inspect duplicated state, aggregate/layer ambiguity, immutable property resolution, copying, recursive public prediction, representation conversions, dynamic attributes, incomplete mocks, fallbacks, repeated policy, unnecessary/unbounded caches and wrappers.
3. After runtime contracts, review metrics/traces/ledgers for Stage/legacy duplication, ownership switches, first-layer ambiguity, dynamic proxying, duplicated aggregation and private-field consumption.
4. Review standard and experimental profiling for repeated paths/platform policy/metadata, optional state and initialization dependencies. Preserve the standard versus experimental SGLang boundary.
5. Review training/model manager/configuration/artifacts for repeated paths/family policy, optional/lazy state, measurement-family handling and redundant conversions.
6. Establish construction/binding invariants when production guarantees a field. Remove defensive getattr/hasattr/None/fallback handling only with caller evidence. Retain genuine supported optionality and accurate lightweight doubles; correct incomplete test fixtures instead of adding production accommodations.
7. Reuse canonical architecture/family/operator/device/model/scheduler registries and contracts. No hard-coded categories, duplicate defaults, magic scales, sentinel patches, broad exception suppression, unsupported compatibility paths or invented duplicate helpers.
8. Preserve approved simulator, scheduling, numerical, memory, attention/MoE, profiling-schema and training-artifact behavior. Diagnose discrepancies before fixing. Do not change expected outputs to conceal a difference.
9. For each meaningful refactor, identify the preserved invariant, run focused tests and compare pre/post outputs. After each major phase, run broader regression. After runtime changes, run fidelity and Simulator.run() wall-clock checks. Performance must retain simulation and metrics semantics.
10. Maintain the issue table: Module / Problem / Root cause / Why unnecessarily complex / Refactor / Reused abstraction / Verification. Also record inspected patterns deliberately retained and their real runtime need.
11. Use grill-me only after investigation establishes a consequential unresolved interface/semantic choice; present current behavior, cause, paths, options, trade-offs and recommendation. Routine implementation choices proceed autonomously.

## Scope and inherited decisions

- New independent task; preceding PR31/PR33 task documents are historical behavior/evidence sources, not this task's completion record.
- Main: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`; pre-refactor candidate: `c288a19f59bec09529ee18d782fa57218da2c781`; merge-base equals main.
- Local edits, focused fixtures, verification and coherent commits are authorized. No remote publication, merge or history rewrite is included.
- D01: preserve uniform physical-layer/stage ownership and previously approved corrections. Main equality is not a replacement for this accepted candidate behavior.
- D03: preserve admission, continuation, completion and failure cleanup; do not create a request cancellation API.
- D02: prior measured overhead acceptance is historical evidence, not a universal performance budget. Measure this cleanup independently.
- Existing main debt changes only when directly extended or required for a small candidate contract correction. Preserve README.md.
- Native GPU/AMD evidence requires actual hardware; CPU contracts and synthetic profiles do not establish native or production-data parity.
- Use existing task document layout instead of skill-specific root planning files. Keep raw logs/caches under `/data/ycfeng/tmp` and selected final evidence here.

## [Original Request] Review remediation follow-up — 2026-09-17

> follow the doc guidence: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-17_candidate_quality_cleanup/Frontier_PR33_Review_Comments_2026-09-17_d43ae932_EN.md, fix bugs/problems on by one.

Address R01–R10 sequentially with direct regression evidence and individual commits; assess C01–C03 within local authorization. Preserve historical cleanup evidence. Remote PR edits require approval; prepare a local description first. The review explicitly requests corrections to copied README recipes (R09), authorizing those focused edits. Native hardware results must remain distinct from CPU contracts.

## [Original Request] R01 clarification and explicit temporary approximation

> 我并没有理解当前问题和选项的含义。这里提到的phase的含义是什么？当我们采用不同的调度器时，调度策略会不同，为什么会需要进行gdn情况下的策略选择呢，难道不应该根据当前调度器来决定吗？请你给出清晰的例子帮助我理解

> 允许暂时用 prfill batch来处理mixed batch（请你确认），并进行warning提醒。并在docs中补充该limitation（特别针对于co-location模式下会发生）

Decision R01-A: preserve the selected scheduler and real Batch/Request semantics; temporarily predict the entire mixed batch's GDN work with prefill estimators, preserving physical features and state ownership. Emit an explicit approximation warning and document the co-location limitation. This supersedes the review's mixed-rejection requirement and the unselected phase-pure scheduling proposals. No native numerical-equivalence claim is authorized or established. Native producer and training rows retain their pure-phase contracts.
