## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-15 | Completed the independent v1.1 audit; recorded item-level status, evidence limits, TODOs, and decision gates at source candidate `b8cecf53`. |
| 2026-09-15 | Opened the audit at `b8cecf53` and froze the initial Git/PR evidence. |

# Independent PR33 v1.1 Completion Audit

## 1. Executive conclusion

The v1.1 revision plan is **substantially incomplete**. The previous session delivered substantial implementation and CPU evidence, but no R01–R14 item satisfies both the implementation and verification acceptance criteria in the plan. R01–R05, R07–R12, and R14 are **PARTIAL**; R06 and R13 are **NOT DONE**. No item is classified COMPLETE.

The most consequential unresolved facts are:

- The real `BaseModelConfig` path can still classify a homogeneous MLA model as `dense_attention`; this is a production routing defect, not only a missing test.
- Automatic GDN memory reservation and slot wiring have implementation pieces, but the required capacity/block behavior and real `Simulator` lifecycle have not been demonstrated.
- `ExecutionTime`/`StageExecutionTime` still retain the permanent legacy/aggregate dual semantics and generic compatibility forwarding that the plan explicitly required to remove.
- The required unified operator-ownership enumeration seam is absent.
- The 58-case matrix is a useful dummy/fidelity result, but it contains no `operation_metrics.csv` or `op_traces.jsonl` outputs and does not replace the required non-dummy/golden lanes.
- The historical approximately 9.42× `Simulator.run()` result is not reproduced at the final candidate, but the current dense run-phase residual remains about 1.44–1.50× and has no completed causal ablation or user-approved acceptance.
- GPU evidence is correctly unavailable, but the plan also required runnable GPU entry points and a complete timer contract; CPU fake-runtime tests alone do not satisfy that requirement.

This report is authoritative for the audit and supersedes earlier task-record statements such as “implementation and CPU verification complete” or “only D02/D03 remain.” Historical reports are preserved; they are not deleted or rewritten.

## 2. Exact Git and PR state

The audit execution date is **September 15, 2026**. The
`completion_audit_2026-09-16.md` and
`test_report_2026-09-16_completion_audit.md` filenames are retained from the
existing task handoff and do not claim that a future run occurred.

The audit inspected the local candidate separately from the remote PR:

| State | Exact evidence |
| --- | --- |
| Worktree | `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn` |
| Branch | `feature-amd-sglang-gdn` |
| Local candidate HEAD | `b8cecf53f8b81ea8380238971277ba94c6fe4961` |
| `main` baseline | `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2` |
| Local relationship | `main...HEAD = 0 38` (38 ahead, 0 behind) |
| Remote PR #33 HEAD | `69d09305ca4d881ced4c7d1653ddc9c39f9d9701` |
| Local vs PR | local candidate is 15 commits ahead of the remote PR head |
| PR state | OPEN, non-draft, MERGEABLE, base `main`, no status checks in `statusCheckRollup` |
| Clean baseline worktree | `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915`, detached at `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`, clean |
| Audit-created source changes | none |

The 15 commits absent from PR #33 include the source/test repairs and performance changes from `7d4dec45` through `b8cecf53` (including GDN phase validation, slot integration, DEVICE_EVENT path helper, timing/cache changes, and the final stage aggregation cache). Therefore, the remote PR cannot be treated as containing the audited candidate. At the start of this audit, only previous-session task documents were dirty; the audit edits are documentation-only and do not change the tested source SHA.

The complete v1.1 plan was read: 503 lines, including R01–R14, all 38 finding dispositions, W0–W5 sequencing, acceptance invariants, and D01–D03 gates.

## 3. Audit method and evidence boundary

The audit treated prior progress files, checklists, commit messages, test reports, and summaries as claims. Current source, current tests, exact Git state, and retained raw artifacts were checked independently. The focused audit command passed 69 tests, but it ran in a CPU environment with Python 3.13.13 and no `torch`; it is not GPU evidence.

The principal retained artifacts are:

- 58-case matrix: `/data/ycfeng/tmp/pr33-r11-fidelity-20260915-b8cecf53/manifest.json` and `results.json`.
- Final paired performance: `/data/ycfeng/tmp/pr33-r12-paired-20260915-final-b8cecf53/`.
- Exact 104-event completion-audit comparison: `/data/ycfeng/tmp/pr33-completion-audit-exact104-restored-env-b8cecf53/`.
- Independent cProfile comparison: `/data/ycfeng/tmp/pr33-completion-audit-exact104-cprofile-b8cecf53/`.
- Focused pytest basetemp: `/data/ycfeng/tmp/pr33-completion-audit-20260916-focused-final/`.

The plan’s comparison tolerances remain `rel_tol=1e-12` and `abs_tol=1e-9`. No tolerance, golden result, or hardware status was changed to obtain a PASS.

## 4. Item-by-item completion matrix

Status meanings follow the request: COMPLETE requires both implementation and sufficient verification; PARTIAL means that implementation or evidence exists but the plan’s acceptance is incomplete; NOT DONE means the required implementation/evidence is absent.

| Item | Status | Required by plan | Current implementation/evidence | Missing work | Basis for conclusion |
| --- | --- | --- | --- | --- | --- |
| R01 | PARTIAL | One authoritative homogeneous family rule; real `BaseModelConfig` through the public predictor; MLA prefill/decode/profile-key assertions; MHA/GQA/MQA/MFA/DSA/exotic/Qwen3/Qwen3.5 regressions. | `frontier/attention/model_binding.py:124-199` can return `latent_mla_attention`; focused family tests pass. However, `frontier/attention/gdn/config.py:279-298` still returns `dense_attention` for every non-Qwen3.5 config. `frontier/config/model_config.py:537-542` uses that resolver for real layer specs, and `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:2290-2302` can consume the resulting layer spec for cache identity. | Repair the real resolver/binder dependency and add a real `BaseModelConfig -> public predictor -> MLA prefill/decode/profile-key` integration test plus the full boundary matrix. | Current production path still contains the source-confirmed homogeneous MLA misclassification. SimpleNamespace/helper tests do not establish real-config routing. |
| R02 | PARTIAL | Pass effective scheduler concurrency into automatic GDN planning and R03 slots; test both automatic modes, exact fixed-state reservation, remaining blocks, capacity monotonicity, ordinary OOM, and zero non-GDN reservation. | `frontier/scheduler/replica_scheduler/base_replica_scheduler.py:61-70` passes a capacity; `frontier/scheduler/utils/memory_planner.py:197-205` reserves fixed state and `:224-278` subtracts it before full-attention KV division. `tests/unit/test_gdn_scheduler_slots.py:118-188` constructs both automatic modes; tests at `:190-220` check reservation arithmetic and full-attention denominator. | Directly assert that increasing capacity changes `num_blocks` in the required direction, prove non-GDN automatic reservation is zero, and exercise structured automatic OOM without relying on explicit `num_blocks` or arithmetic-only tests. | Wiring exists, but the plan’s end-to-end automatic-capacity acceptance is not demonstrated. |
| R03 | PARTIAL | Real scheduler ownership through admission, chunks, waiting/resume, completion, reachable cancellation/termination, pool exhaustion, rollback, repeated cleanup, and slot-ID/ownership assertions for at least two requests. | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:129-148, 1399-1444, 2833-3027` contains manager construction and admission/release paths. `tests/unit/test_gdn_scheduler_slots.py:269-363` covers allocation, pool exhaustion, manual waiting/resume, rollback, and idempotent cleanup. | Drive two requests through the actual `Simulator`, including real continuation and waiting/resume; verify terminal cleanup and complete cancellation/termination call-site inventory. Add an E2E test through reachable public events/API, not private methods and manually edited queues. | Existing tests call private scheduler methods and manually move requests. They prove local behavior, not the required Simulator lifecycle or reachable cancellation. |
| R04 | PARTIAL | Explicit phase and prefill mask; cold one-token prefill, one-token continuation, ordinary decode, mixed lengths, all-one-token mixed, missing/conflicting metadata; rejected input must invoke no native work and emit no valid row. | `frontier/profiling/gdn/inputs.py:46-102` has explicit phase/mask validation. `tests/unit/test_gdn_profiler_cpu_increment7.py:17-100` covers continuation, decode, mixed, and metadata rejection. | Add an explicit cold one-token prefill case and negative-side-effect assertions: native call count must remain zero and rejected CLI input must not write a valid artifact row. | Positive and validation behavior exists, but the negative execution/artifact guarantees and cold-prefill branch are not evidenced. |
| R05 | PARTIAL | Make `ExecutionTime` one real layer; make `StageExecutionTime` the sole cross-layer aggregation owner; remove permanent legacy aggregate/scaling/sentinel/dynamic-forwarding semantics; preserve overlap, units, IDs, and all owner categories. | Some single-layer ownership/copy and stage-cache optimizations are present. But `frontier/entities/execution_time.py:139-143` retains `_legacy_aggregate`; `:555-568` retains `_aggregation_factor`; `:1639-1654` retains model scaling. `frontier/entities/stage_execution_time.py:339-409` exposes first-layer compatibility views and `:543-565` retains generic `__getattr__`. `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py:1348-1407` still constructs a legacy aggregate before Stage conversion; MoE code retains production sentinel pass-through. | Complete coordinated migration of all real consumers, remove unsafe production dual semantics/sentinels, and add ownership-category tests for N=1/N=2, mixed families, shared experts, PP/CPU/terminal work, overlap, units, timestamps, and IDs. | The measured 32× double-counting symptom was mitigated, but the plan explicitly requires removal of the underlying dual semantics; source inspection shows they remain. |
| R06 | NOT DONE | Add one explicit ordered `(layer_id, family, operator, scope, duration)` ownership enumeration seam; keep output adapters separate; prove no duplicate/omitted work and unchanged metrics-toggle behavior. | `frontier/metrics/metrics_store.py:3690+` independently builds operation metrics, `:4123+` independently builds the component ledger, and `frontier/metrics/op_trace_utils.py` supplies separate trace metadata. The 58-case artifact contains no `operation_metrics.csv` or `op_traces.jsonl` files. | Design and wire the shared enumeration seam, then add independent hybrid ownership oracle and differential request/system/operation/ledger/trace tests. | No common production seam or required matrix outputs exists; existing metric tests do not satisfy the package acceptance. |
| R07 | PARTIAL | One helper must govern all three DEVICE_EVENT resolvers, including explicit/legacy/extensionless/empty/template cases, with absent paths remaining absent. | `frontier/execution_time_predictor/measurement_input_paths.py` is used by manager loading (`shared_prediction_model_manager.py:633-650`) and predictor path handling (`sklearn_execution_time_predictor.py:851-868`). The public manager API at `shared_prediction_model_manager.py:4613-4663` still has independent resolution behavior. Tests cover a two-entry loading contract, not all three production entries in one parameterized table. | Route the public manager API through the same helper and add the complete three-entry parameter matrix, including cache/measurement-family separation and clear missing-input failures. | The helper is real progress, but the required three-way behavioral equivalence is not complete. |
| R08 | PARTIAL | Bound numeric attention reuse by stage/prediction context; include all estimator-relevant identity; isolate mutation; measure cache size/hits/misses, allocations, and long-run resource cost; do not cache routing/ownership. | Stage-local/predictor LRU behavior and capacity 64 exist; `tests/unit/test_attention_query_cache.py:89-156` covers reuse, context/phase/TP/family misses, capacity, and counters. `sklearn_moe_execution_time_predictor.py:2217-2250` still uses a generic recursive freezer with `repr(value)` fallback. The key can inherit R01’s incorrect real-model layer spec. | Replace unjustified generic fallback/exception behavior, add dtype/artifact/state-mutation and allocation/RSS/long-run evidence, and revalidate key semantics after R01/R05. | Numeric cache tests pass, but the plan requires stronger key/fallback/lifetime proof and measured construction-cost evidence; current implementation still contains the cited fallback. |
| R09 | PARTIAL | Real fit/save/load/manager/predictor/Simulator chain plus automatic capacity, one-token final chunk, waiting/release/reuse, shared experts, TP>1 communication, real MLA, homogeneous MoE EP>1/PP>1, and parameter/analytic oracles. | `tests/unit/test_gdn_hybrid_e2e_increment14ab.py:771-930` covers real synthetic GDN training/loading and a Simulator constructor path, request completion, metrics, GDN/dense identities. The case uses `num_blocks=100`, TP=1/EP=1/PP=1, one request, prefill 16/decode 2, and `max_tokens_in_batch=64`. JSON fixture has `shared_expert_intermediate_size=64`, but the Python fixture does not fully map/verify it. | Add the missing orthogonal cases and independent oracles, with real MLA public routing and homogeneous EP/PP/TP coverage. Keep GDN unsupported boundaries unchanged. | The existing E2E is genuine but materially narrower than R09’s acceptance and uses a prepared-predictor seam for some timings. |
| R10 | PARTIAL | Runnable GPU entries with real GDN state/output assertions, TP cases, continuation equivalence, timer contract coverage, and separate NVIDIA/AMD PASS/SKIP evidence. | `tests/unit/test_device_timer_contract.py` exercises fake torch/CPU contracts. `frontier/profiling/common/device_timer.py:36` still accesses `Singleton._instances` directly. No real GDN GPU pytest entry or initialized-method/no-store/standalone contract matrix was found. Hardware evidence is explicit: `SKIP: AMD/MI355X hardware unavailable`; `SKIP: no visible NVIDIA device`. | Add/retain executable GPU test entry points and complete CPU timer contract; run on an authorized GPU worker or preserve exact prerequisite SKIPs with test collection evidence. | The SKIP reasons are truthful, but CPU fakes do not establish the required GPU acceptance and timer cleanup remains incomplete. |
| R11 | PARTIAL | Clean pinned-main/candidate comparison for all 58 cases, available non-dummy/golden lanes, full unit failure-node comparison, key/order/ID/discrete checks, and D01 RCA for every stable gap. | `/data/ycfeng/tmp/pr33-r11-fidelity-20260915-b8cecf53/manifest.json` and `results.json` report 58/58 PASS with `rel_tol=1e-12`, `abs_tol=1e-9`; baseline `0515589a` and candidate `b8cecf53` are recorded. However, operation/trace artifact counts are zero, and the reported 197 “non-dummy/golden” passes are unit tests rather than the required available real E2E/golden lanes. | Run available non-dummy dense/MoE/MLA and golden/operator matrix through their real prerequisites; preserve operation/trace artifacts; produce auditable same-environment baseline-vs-candidate node-level comparison and RCA/disposition for every stable difference. | Dummy/fidelity matrix is complete as far as its outputs go, but it is not the complete R11 gate. |
| R12 | PARTIAL | Same-host interleaved before/after raw measurements, variability, run/init/total/event breakdown, hotspots, calls/allocations/RSS, causal ablation, optimization, and explicit disposition of historical 9.42×. | Historical exact workload: baseline `0.009217162s`, old candidate `0.086823318s`, ratio `9.419745×`, 104 events. Final exact 104-event unprofiled artifact: 5/5 paired runs, 2/2 requests and 104 events; main median `0.008927742s`, candidate `0.012872720s`, ratio `1.441878584×`; `total_proc_s` ratio `0.991089458×`, `init_s` ratio `0.989398790×`. Final broader workloads are `1.504303×`, `1.407252×`, and `1.011424×` for small dense, longer dense, and representative MoE. cProfile points to roughly 800 extra `ExecutionTime.as_single_layer()` calls and 20 `StageExecutionTime.from_execution_time()` calls, but no narrow ablation or allocation/RSS closure exists. | Perform controlled ablations for per-layer expansion/record construction, capture calls/allocations/RSS and post-fix measurements, explain the residual, and submit the measured residual to D02 rather than accepting it implicitly. | The rejected 9.42× result no longer reproduces, and one avoidable aggregation cost was removed, but the plan requires causal closure for the remaining material residual. |
| R13 | NOT DONE | Bounded cleanup plus before/after line counts, retention rationale, feasible split boundaries, and sequencing for touched >2,000-line modules; no mechanical refactor. | Current line counts remain: `execution_time.py` 1682, `stage_execution_time.py` 568, `sklearn_moe_execution_time_predictor.py` 3837, `sklearn_disaggregation_execution_time_predictor.py` 3075, `metrics_store.py` 5587, `shared_prediction_model_manager.py` 4716, `sklearn_execution_time_predictor.py` 8371. | Record before/after counts, explain retained large modules and split seams, and tie cleanup to R01/R05/R06/R07/R08/R10 without broad unrelated churn. | No bounded cleanup analysis or boundary/sequence record was produced; current counts alone do not satisfy the soft-limit requirement. |
| R14 | PARTIAL | Truthful exact-SHA handoff: every R item with commit/no-code-change, tests/results, risks, baseline failure comparison, hardware boundaries, raw artifact manifest, and PR wording matching the tested revision. | Task records contain exact candidate artifacts, 58-case results, broad CPU claims, and hardware SKIPs. Historical “complete” wording is retained for provenance, while this audit now adds superseding status sections in `progress.md`, `issues.md`, `review.md`, `summary.md`, and `revision_execution_checklist.md`. The remote PR is still 15 commits behind the local candidate. | After the functional/performance gates close, refresh the final manifest and any later-authorized PR wording; keep the local-vs-remote distinction and docs-only/no-rerun statement. | The documentation correction is now recorded, but R14 cannot claim final closure while R01–R13 remain open and the remote PR does not contain the audited source. |

**Overall matrix result:** no R item is COMPLETE. None is NOT APPLICABLE or SUPERSEDED; every package remains relevant, although some already-resolved subfindings are retained as evidence rather than reopened (for example actual Replica-ID routing and SGLang tuple normalization).

## 5. Dependency and work-package gate assessment

| Stage | Plan dependency | Audit result |
| --- | --- | --- |
| W0 | Pin clean main/candidate, preserve pre-repair facts, and start R12 measurement/RCA before broad refactoring. | **PARTIAL.** Clean baseline and final measurements exist; the historical dirty candidate cannot be reconstructed exactly, and the final residual lacks causal ablation. |
| W1 | R01, R02→R03, R04 direct functionality repairs; R07 may proceed independently. | **NOT CLOSED.** Focused tests exist, but R01 production routing, R02 automatic acceptance, R03 Simulator lifecycle, and R04 negative-side-effect evidence remain open. |
| W2 | Coordinate R05/R06 and narrow R08 under R12 profiling; establish single timing semantics and once-only ownership. | **NOT CLOSED.** R05 remains dual-semantics, R06 has no seam, and R08 has incomplete fallback/resource evidence. |
| W3 | R09 integration and R10 CPU/GPU contracts with separate hardware evidence. | **NOT CLOSED.** R09 coverage is narrow; R10 has CPU fake tests and explicit SKIPs but lacks the required runnable GPU lane/timer matrix. |
| W4 | Complete R11 matrix/non-dummy lanes and final R12 comparison; RCA and D01/D02 treatment before closure. | **NOT CLOSED.** 58 dummy cases pass, but required non-dummy/golden evidence is missing and D02 remains open. |
| W5 | Bounded R13 cleanup and truthful R14 handoff. | **NOT CLOSED.** R13 analysis is absent and earlier task records overstated completion; this audit begins but does not erase the remaining gaps. |

The dependency consequence is `R01 -> R02 -> R03`, while `R05 <-> R06 -> R08 -> R11/R12`, and `R09/R10 -> R11 -> R14`. R13/R14 are handoff gates, not substitutes for the earlier functional/performance gates.

## 6. All 38 finding dispositions

The following preserves the plan’s complete mapping. “Plan disposition” is the adjudication in v1.1; “audit state” is whether the mapped remediation is actually closed now.

| Finding | Plan disposition | Package | Audit state |
| --- | --- | --- | --- |
| A SP-01 | Adopt | R02 | PARTIAL: automatic capacity acceptance missing |
| A SP-02 | Adopt | R01 | PARTIAL: real MLA route still defective |
| A SP-03 | Adopt | R03 | PARTIAL: no real Simulator lifecycle E2E |
| A SP-04 | Adopt; execute early | R04 | PARTIAL: negative side effects/cold prefill missing |
| A SP-05 | Adopt | R09 | PARTIAL: branch coverage below acceptance |
| A SP-06 | Adopt | R10 | PARTIAL: GPU entry/evidence missing |
| A SP-07 | Adopt | R05 | PARTIAL: dual semantics remain |
| A SP-08 | Adopt | R11 | PARTIAL: dummy matrix only |
| A SP-09 | Adopt with qualification | R12 | PARTIAL: residual RCA/decision open |
| A SP-10 | Adopt | R10 | PARTIAL: NVIDIA/AMD evidence separated, runtime lane absent |
| A ST-01 | Merge duplicate | R01 | PARTIAL with SP-02 |
| A ST-02 | Adopt jointly | R05 | PARTIAL: scope cleanup not complete |
| A ST-03 | Adopt | R05 | PARTIAL: production sentinels remain |
| A ST-04 | Adopt, bounded | R06 | NOT DONE: no common ownership seam |
| A ST-05 | Adopt | R07 | PARTIAL: public manager resolver diverges |
| A ST-06 | Adopt; measure costs | R08/R12 | PARTIAL: cache exists, cost evidence incomplete |
| A ST-07 | Adopt documentation requirement | R13 | NOT DONE |
| B D1 | Adopt goal; merge | R05 | PARTIAL: debt documentation did not remove dual semantics |
| B D2 | Adopt gate; correct causality | R12/R08 | PARTIAL: hotspot evidence, no ablation/acceptance |
| B D3 | Partially adopt | R03 | PARTIAL: slots wired, cancellation entry unresolved |
| B D4 | Adopt | R10 | PARTIAL: hardware boundary recorded, GPU test lane absent |
| B T1 | Partially adopt | R06 | NOT DONE: no shared differential ownership seam |
| B T2 | Adopt for supported models | R09/R11 | PARTIAL: no complete EP/PP/non-dummy campaign |
| B T3 | Reject proposed invariant | R05 | PARTIAL: corrected invariant not fully migrated |
| B T4 | Merge duplicate | R10 | PARTIAL with D4/SP-10 |
| B T5 | Adopt, bounded extension | R09/R02 | PARTIAL: representative parameter/memory cases missing |
| B T6 | Retain resolved fix | R09 | COMPLETE subfinding only: actual-ID fix is present and separately tested; package R09 remains PARTIAL |
| B T7 | Retain accepted hardware boundary | R10/R14 | PARTIAL: SKIP is truthful, handoff is not yet fully synchronized |
| B Q1 | Partially adopt | R05 | PARTIAL: first-layer compatibility remains |
| B Q2 | Merge duplicate | R05 | PARTIAL with D1/SP-07 |
| B Q3 | Merge; qualify | R08/R12 | PARTIAL: optimization occurred, causal closure absent |
| B Q4 | Partially adopt | R06 | NOT DONE: adapters still infer ownership independently |
| B Q5 | Reject count-based mass deletion | R01/R05/R08/R13 | PARTIAL: formal APIs retained, but required fallbacks/semantics remain |
| B Q6 | Partially adopt | R01/R06/R07 | PARTIAL/NOT DONE across the three packages |
| B Q7 | Partially adopt; correct premise | R10 | PARTIAL: timer premise corrected, accessor/runtime proof absent |
| B Q8 | Split disposition | R01/R03/R13 | PARTIAL: slots wired; R01, lifecycle, and cleanup remain open |
| B Q9 | Adopt scope fix; defer cosmetics | R05/R13 | PARTIAL: scope fix incomplete; cosmetic deferral is correct |
| B Q10 | No fix needed | R13 | COMPLETE as a non-action: no unrelated constants/tolerances were reopened |

The two “COMPLETE subfinding/non-action” entries above do not make their parent R packages complete.

## 7. Remaining TODO list, ordered by dependency and severity

### TODO 1 — Repair authoritative real-model attention routing (P1; prerequisite for R09/R11)

- **Issue:** `resolve_layer_attention_specs()` still hardcodes dense family for non-Qwen3.5 real configs, while the binder can correctly identify MLA.
- **Files/modules:** `frontier/attention/model_binding.py`, `frontier/attention/gdn/config.py`, `frontier/config/model_config.py`, public predictor/cache-key path in `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`; `tests/unit/test_attention_family_binding.py` and a new real-config integration test.
- **Required work:** Establish one acyclic authoritative homogeneous binding seam; preserve explicit Qwen3.5 schedule resolution; propagate malformed supported configs.
- **Tests/experiments:** Real `BaseModelConfig` through public predictor for MLA prefill/decode/profile keys; MHA/GQA/MQA/MFA/DSA/exotic/Qwen3/Qwen3.5 regression matrix.
- **Acceptance:** Real and per-layer family IDs agree; MLA no longer enters dense cache/profile paths; all boundary tests pass at a newly pinned candidate SHA.

### TODO 2 — Complete automatic GDN memory and slot-capacity acceptance (P1; R02 -> R03)

- **Issue:** Capacity is wired, but required automatic block behavior and structured OOM/non-GDN proofs are missing.
- **Files/modules:** `frontier/scheduler/replica_scheduler/base_replica_scheduler.py`, `frontier/scheduler/utils/memory_planner.py`, `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`, `tests/unit/test_gdn_scheduler_slots.py`.
- **Required work:** Use one effective capacity for reservation and slot pool; test both automatic modes without explicit block substitution.
- **Tests/experiments:** Vary capacity and assert reservation, remaining blocks, fixed-state context independence, non-GDN zero reservation, and structured parameter/KV OOM.
- **Acceptance:** Exact expected bytes/blocks match; capacity cannot over-admit state owners; no fallback block count or parameter reduction is used.

### TODO 3 — Prove real GDN request lifecycle and finish termination inventory (P1; R03)

- **Issue:** Current slot tests call private methods and manually mutate queues; no real Simulator two-request lifecycle exists.
- **Files/modules:** `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`, scheduler events/Simulator integration, `tests/unit/test_gdn_scheduler_slots.py`, `tests/unit/test_gdn_hybrid_e2e_increment14ab.py`.
- **Required work:** Add real admission, two continuations, waiting/resume, pool exhaustion, completion/reuse, KV rollback, and terminal cleanup through reachable paths. Search all request cancellation/abort/termination APIs and call sites.
- **Tests/experiments:** Assert slot IDs, ownership maps, KV maps, request state transitions, and zero leaks after completion/cancellation. If no reachable cancellation exists, prepare the D03 options rather than faking one with `manager.release()`.
- **Acceptance:** At least two requests traverse the actual Simulator; same owner retains its slot across waits; reuse is observed after completion; reachable cancellation either passes or is explicitly dispositioned by the user.

### TODO 4 — Close explicit GDN phase side-effect contract (P1; R04)

- **Issue:** Positive phase/mask validation exists, but cold one-token prefill and “reject before native/artifact output” are not proven.
- **Files/modules:** `frontier/profiling/gdn/inputs.py`, native wrapper/CLI producer, `tests/unit/test_gdn_profiler_cpu_increment7.py`, relevant profiling artifact writer.
- **Required work:** Add cold one-token prefill and instrument native-call/artifact-row side effects for invalid mixed/conflicting inputs.
- **Acceptance:** Invalid inputs produce no native invocation and no valid output row; one-token prefill/continuation remain logically distinct from decode.

### TODO 5 — Finish coordinated timing semantics and operator ownership (P1; R05 -> R06 -> R08)

- **Issue:** Legacy aggregate/scaling, first-layer views, generic `__getattr__`, disaggregation adapters, and sentinels remain; no shared owner enumeration exists.
- **Files/modules:** `frontier/entities/execution_time.py`, `frontier/entities/stage_execution_time.py`, `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`, `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`, `frontier/metrics/metrics_store.py`, `frontier/metrics/op_trace_utils.py`.
- **Required work:** Migrate real consumers to one-layer records plus explicit stage/owner fields; introduce ordered owner enumeration and keep trace/metrics/ledger adapters separate; remove only unjustified compatibility/sentinel paths.
- **Tests/experiments:** N=1/N=2 distinct layer values, mixed families, shared experts, PP/CPU/terminal once-only counts, critical-path overlap, units, timestamps, IDs, metrics toggles, and operation/trace/ledger differential artifacts.
- **Acceptance:** No permanent dual semantics or generic dynamic forwarding remains on production paths; every output has one authoritative owner enumeration and no duplicate/omitted work.

### TODO 6 — Complete DEVICE_EVENT resolver equivalence (P2; R07)

- **Issue:** The public manager resolver still diverges from the helper-backed loading/predictor paths.
- **Files/modules:** `frontier/execution_time_predictor/measurement_input_paths.py`, `shared_prediction_model_manager.py`, `sklearn_execution_time_predictor.py`, `tests/unit/test_measurement_family_selector.py`.
- **Required work:** Route all three production entries through one helper; preserve empty absence and legacy/template precedence.
- **Acceptance:** One parameterized table passes identically for explicit, suffix, extensionless, empty, legacy, and template cases; missing inputs fail clearly without family fallback.

### TODO 7 — Expand real hybrid CPU integration and supported parallel oracles (P1; R09)

- **Issue:** Existing real constructor E2E is one request, explicit blocks, TP/EP/PP=1, and does not verify shared-expert mapping/MLA/continuation branches.
- **Files/modules:** `tests/unit/test_gdn_hybrid_e2e_increment14ab.py`, `tests/fixtures/pr31_hybrid/model.json`, Python fixture/profile loader, `tests/e2e/moe_ep_non_dummy_matrix.py`, communication/ParamCounter paths.
- **Required work:** Add both automatic modes, one-token final chunk, concurrent waiting/reuse, shared experts, valid TP>1 communication, real MLA, homogeneous EP>1/PP>1.
- **Acceptance:** Each case names its oracle (baseline, analytic, invariant, synthetic profile); GDN unsupported EP/PP/DP boundaries remain fail-fast.

### TODO 8 — Restore executable GPU acceptance and timer contract (P1; R10)

- **Issue:** CPU fake-torch tests and hardware SKIPs do not substitute for collected GPU tests; `DeviceTimer` still uses `Singleton._instances`.
- **Files/modules:** `frontier/profiling/common/device_timer.py`, `frontier/profiling/common/cuda_timer.py`, `frontier/profiling/common/timer_stats_store.py`, GPU test entry points and `tests/unit/test_device_timer_contract.py`.
- **Required work:** Add/retain real GPU tests for wrapper outputs/state/continuation/TP and complete CPU timer initialized/disabled/no-store/standalone matrix. Run on GPU worker when available; otherwise preserve exact prerequisite SKIPs.
- **Acceptance:** Test collection proves the GPU lane exists; available hardware yields numerical/state assertions; no CPU fake test is labeled GPU PASS.

### TODO 9 — Finish full fidelity and non-dummy/golden acceptance (P1; R11)

- **Issue:** 58 dummy cases pass, but required operation/trace outputs and available non-dummy/golden lanes are absent.
- **Files/modules:** `tests/integration/run_scheduler_refactor_fidelity.py`, `tests/e2e/operator_parity/run_golden_matrix.py`, `tests/e2e/moe_ep_non_dummy_matrix.py`, comparator/artifact writers.
- **Required work:** Re-run all required lanes from clean baseline/candidate worktrees at a newly pinned source SHA; record interpreter/dependencies, dirty state, per-case statuses, failure first divergence, and baseline unit node/cause comparison.
- **Acceptance:** 58 cases plus available non-dummy/golden lanes produce complete request/system/stage/operation/trace artifacts; every stable gap follows D01 before disposition; unavailable prerequisites are explicit SKIP/BLOCKED.

### TODO 10 — Complete performance RCA and post-fix measurement (P1; R12/D02)

- **Issue:** 9.42× no longer reproduces, but dense residual is material and cProfile is attribution only.
- **Files/modules:** `tests/performance/sim_walltime_scaling/run_case.py`, stage/layer construction paths, `frontier/entities/stage_execution_time.py`, `frontier/entities/execution_time.py`; retained performance scripts/artifacts under `/data/ycfeng/tmp`.
- **Required work:** Use same-host interleaved unprofiled runs; ablate repeated expansion/record validation/estimator calls; capture calls, allocations, RSS and run/init/total breakdown; optimize evidenced avoidable overhead and preserve IDs/numerics.
- **Acceptance:** Historical result is explicitly disposed; residual source and removable share are causally established; post-fix measurement is recorded. No implicit percentage budget or D02 waiver is used.

### TODO 11 — Perform bounded cleanup analysis (P2; R13)

- **Issue:** Seven touched critical modules exceed the 2,000-line soft limit and no before/after/rationale/split-boundary record exists.
- **Files/modules:** `frontier/entities/execution_time.py`, `stage_execution_time.py`, `sklearn_moe_execution_time_predictor.py`, `sklearn_disaggregation_execution_time_predictor.py`, `metrics_store.py`, `shared_prediction_model_manager.py`, `sklearn_execution_time_predictor.py`.
- **Required work:** Record line counts before/after, retained/deleted rationale, feasible functional seams, and sequencing; limit edits to touched correctness/duplication/fallback concerns.
- **Acceptance:** Analysis is auditable and no mechanical mass split or unrelated churn is introduced.

### TODO 12 — Final truthful handoff after all gates (P1; R14)

- **Issue:** Old checklist/review/summary claims conflict with current item status and the remote PR does not contain local candidate commits.
- **Files/modules:** `task_memory/task_2026-09-14_pr31_selective_integration_v2/{progress,issues,review,summary,revision_execution_checklist}.md`, authorized PR description if later requested.
- **Required work:** Keep historical reports intact; update current statuses, exact tested SHA, raw artifact manifest, baseline failure comparison, hardware SKIPs, decision gates, and local-vs-remote distinction.
- **Acceptance:** Every R item has truthful status, commit/no-code-change, test result, risk, and remaining evidence; docs-only changes explicitly state they do not rerun source validation. Do not merge without separate authorization.

## 8. User-decision items

### D02 — Pending decision on measured residual simulator cost

This decision gate is genuinely unresolved and should not be silently closed.

- **Known:** The user rejected the historical approximately 9.42× run-phase degradation. The exact final 104-event workload no longer reproduces it (1.441878584× `Simulator.run()` median ratio, with 0.991089458× `total_proc_s` ratio). Broader final runs still show 1.504303× and 1.407252× dense run-phase ratios. cProfile identifies a credible direction—per-layer expansion/identity record construction—with roughly 800 extra `as_single_layer()` calls and 20 `StageExecutionTime.from_execution_time()` calls in the candidate profile. Repeated stage aggregation was optimized in `b8cecf53`.
- **Unknown:** The causal share of each remaining path, allocation/RSS cost, removable fraction, and correctness trade-off of further sharing/record reuse. The existing profiles do not prove that all residual work is necessary.
- **Options:** (a) continue targeted semantics-preserving optimization and remeasure; (b) accept a specific measured residual only after a complete RCA and explicit bound; (c) defer/hold release acceptance until the residual is explained. The plan does not pre-approve a 3%, 5%, or any other universal budget.
- **Recommendation:** choose (a) first, because the remaining evidence is attribution rather than causal closure. Do not ask the user to accept the residual yet and do not relabel it as expected overhead.

### D03 — Conditional cancellation/termination scope decision, not yet escalated

The current inventory did not find a reachable request-level `cancel_request`, `abort_request`, or `terminate_request` path; `_free_request_resources_by_id()` handles completion/callback/queue-exit cleanup, while `StageExecutionContext.cancel(ticket)` cancels a stage admission ticket only. However, the final all-repository cancellation/termination inventory and real lifecycle E2E are not complete. Therefore D03 is a pending trigger, not a request for immediate approval.

If the final inventory confirms no reachable request entry point, present: the request state diagram, terminal hooks, missing boundary, narrow cancellation proposal and tests, and the scope loss of completion-only cleanup. Recommended option is narrow integration; alternative is an explicit user revision of the cancellation acceptance requirement. Do not call direct manager release a cancellation E2E and do not delete the slot manager.

### D01 — Not triggered by the completed 58-case matrix; global status remains incomplete

The 58-case comparator found no stable beyond-tolerance numerical/discrete gap using the original tolerances, so no semantic approval is requested from that result. This does not establish that D01 is globally clear because the required non-dummy/golden/operator lanes were not completed. If any later lane produces a stable gap, the required order remains `stable gap -> first divergence/RCA -> options and consequences -> user decision -> revalidation`.

## 9. Final verification gaps

The following evidence is still missing or insufficient:

- **Clean baseline/candidate matrix:** 58 dummy/fidelity cases are present and pass, but operation metrics and op-trace artifacts are absent; available non-dummy dense/MoE/MLA and golden/reference lanes were not completed at the audited source SHA.
- **Full unit comparison:** a retained candidate run reports 3309 passed, 19 failed, 25 skipped, and 576 warnings, with the same 19 failure causes claimed for baseline. A complete auditable same-environment node-ID/cause manifest is still needed; equal counts alone are insufficient.
- **Real hybrid E2E:** no real Simulator lifecycle with two GDN requests, waiting/resume, slot reuse, automatic capacity, shared-expert timing, TP>1 communication, and real MLA public routing.
- **Metrics ownership:** no shared ordered ownership enumeration and no matrix proof of no duplicate dense projections, omitted shared experts/FFN work, or once-only PP/CPU/terminal work across all output adapters.
- **Performance RCA:** no narrow ablation, calls/allocations/RSS data, or post-fix causal measurement for the remaining dense residual.
- **GPU/timer:** no real GDN GPU test entry execution; AMD reason is `SKIP: AMD/MI355X hardware unavailable`; NVIDIA reason is `SKIP: no visible NVIDIA device`; CPU fake-torch tests must not be relabeled as hardware PASS.
- **Cleanup/handoff:** no R13 line-count/split-boundary analysis; historical completion claims are preserved with the synchronized current status recorded by this audit, while the remote/local SHA distinction remains relevant.

## 10. Finding, artifact, and status references

Primary source/test references used in this audit include:

- `frontier/attention/model_binding.py`
- `frontier/attention/gdn/config.py`
- `frontier/config/model_config.py`
- `frontier/scheduler/replica_scheduler/base_replica_scheduler.py`
- `frontier/scheduler/utils/memory_planner.py`
- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`
- `frontier/profiling/gdn/inputs.py`
- `frontier/entities/execution_time.py`
- `frontier/entities/stage_execution_time.py`
- `frontier/metrics/metrics_store.py`
- `frontier/execution_time_predictor/measurement_input_paths.py`
- `frontier/execution_time_predictor/shared_prediction_model_manager.py`
- `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`
- `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`
- `frontier/profiling/common/device_timer.py`
- `tests/unit/test_gdn_scheduler_slots.py`
- `tests/unit/test_gdn_profiler_cpu_increment7.py`
- `tests/unit/test_gdn_hybrid_e2e_increment14ab.py`
- `tests/unit/test_attention_query_cache.py`
- `tests/unit/test_device_timer_contract.py`
- `tests/integration/run_scheduler_refactor_fidelity.py`

The complete plan remains at `Frontier_PR33_review_revision_plan_2026-09-15_v1.1_en.md`. Original review files remain unchanged. This document and the companion `test_report_2026-09-16_completion_audit.md` are audit records only; they do not implement fixes, publish commits, or authorize merge.
