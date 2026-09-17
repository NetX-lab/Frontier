## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Created issue and intentionally-retained-pattern record. |

# Refactoring record

| Module | Problem | Root cause | Why current code is unnecessarily complex | Refactor | Reused existing abstraction | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| Simulator | Duplicate predictor construction and config reflection | Hybrid setup copied existing constructors | Two sites encode identical arguments | One loop and explicit capability method | Predictor registry and model topology | 78 tests; stable artifacts equal |
| ExecutionTime | Missing-component fallbacks in snapshot cloning | Required constructor state treated as optional | Constructed components cannot be absent | Copy required components and real optional maps | Existing component dataclasses/finalization | Isolation tests and E2E parity |
| StageExecutionTime | Repeated None filtering | Validation stages duplicated work | Prior check rejects incomplete identities | Check validated IDs directly | Physical-layer identity contract | Identity tests and E2E parity |
| MemoryPlanner / ParamCounter | Reclassified hybrid topology and probed required methods | Consumers bypassed canonical model contract | Full-layer scanning duplicated runtime-family selection and admitted an unreachable zero-KV fallback | Use model family/per-layer/GDN interfaces directly | BaseModelConfig, runtime family resolver | 47 tests both revisions; four exact memory snapshots |
| GDNBatchFeatures / runtime guards / preemption | Fabricated absent required fields and model state | Incomplete doubles and generic Any obscured production interfaces | Batch/Request and scheduler constructors already guarantee members | Typed direct access; valid preemption fixture; remove redundant zero-mean branch | Batch, Request, BaseModelConfig | 46 focused tests; full baseline and runtime campaign |
| MoE predictor | Alternate capacity state, generated-map revalidation, reflective subclass dispatch, obsolete private alias | Constructor and inheritance contracts were obscured | Internal generation already fixes map types; only replica config is populated in production | Direct initialized state, shared role getter override, one generator | Existing routing generator and disaggregated role getter | 116 tests; 90 stable artifacts and complete file inventories equal |
| ExecutionTime | Unread legacy stage-size attribute | Former aggregate semantics left bookkeeping behind | No consumer uses it after physical-layer transition | Remove stored state, retain input validation | Physical-layer/stage contract | Timing tests and non-dummy parity |
| GDN state slot manager | Front removal and sorting on each release | Ordered allocation implemented by repeatedly sorting storage | Required policy is minimum-free-ID selection | Standard heap operations; retain ownership map and lifecycle | Python heapq | 16 tests and exact 1,000-request old/new transcript |
| MetricsStore residual emitter (S1 correctness restoration) | Positive residual traces fail with unsupported operator errors | Complete explicit event metadata conflated with additive physical-layer identity | Unconditional derivation attempts to resolve non-tensor residuals | Explicit resolved metadata argument; preserve additive identity | Existing compute_op_trace_meta and residual event contract | Frozen 6 FAIL, native-main scalar 3 PASS, post-fix reporting 139 PASS |
| MetricsStore trace projection | Dead private overrides and dense annotations | Stage payload replaced annotation writers without removing readers | Reflection preserves an unreachable second source of timing | Remove orphan readers and unused family lookup | StageExecutionTime physical-layer payload | 69 focused reporting PASS; phase artifact comparison follows |
| Scheduler dense reporting | Test-only helpers, first-layer search, discarded copy, unused time arguments | Representative-layer reporting remnants survived the physical-layer transition | Model API already supplies layer counts; only live helper should define projection | Direct model/count contract; one Stage replacement; migrate tests to live helper | BaseModelConfig, StageExecutionTime and timing-copy helpers | 110 before / 114 after PASS; five byte-identical snapshots |
| EP lane traces / GDN trace metadata | Repeated phase-invariant construction and GDN name tuple | Context built inside operator loop; registry ownership copied locally | Same context/parallel values recur within phase | Hoist local context and lane sum; query GDN family; remove unreachable retired-DP shape labels | OpTraceContext and attention family lookup | 80 focused PASS; exact 64-case event/metric/ledger and four-operator metadata comparison |
| GDN profiling schema / precision / prediction tasks | Repeated literal field/operator lists and zero-mean fallback | Consumers copied declarative family data | Family and task registry already supply exact ordered data; query lengths are positive | Reuse registry tuples/sets and task selection; remove unreachable fallback | GATED_DELTA_NET_ATTENTION_FAMILY and GDN_TASKS | 184 focused PASS; exact 27-column, four-name and two-phase-order comparisons |

## Intentionally retained patterns

P4 M1: manager reimplemented hybrid-versus-MLA selection and silently defaulted a missing replica to CUDA. It now reuses runtime family binding and requires the production replica input. Verification: 127 before/after PASS, 120 exact measurement selections, four exact family classifications. Dataset-only whole-model None remains supported.

P3 J2: variadic profile_graph arguments and seven-position primitive tuples obscured ownership and re-derived GDN metadata. Explicit parameters and an internal named replay-call record retain the existing execution/reset/check owner and native builder contracts. Verification: 39 before / 40 after PASS; 12 byte-identical snapshots with 1,802 ordered events. Real recurrent/cache snapshots and optional trace-probe state remain.

P3 S7: experimental replay duplicated Hamilton allocation and omitted the runtime normalization step. The existing owner now exposes its pure histogram operation; both callers reuse it without a second registry or irrelevant EP state. Verification: 52 frozen / 60 cleaned PASS; 893 exact replay dictionaries, 67 unchanged infeasible rejections, 2,679 exact workloads; 3,407,872 generated histogram comparisons.

P3 S6: native norm/MXFP4 consumers repeated model-name classification. Exact type tuples now live in the existing architecture profiles, queried without broader profile resolution. Verification: 183 PASS and 330 exact admission/error-order comparisons. Native API/platform availability and the bootstrap-dependent local registry import are genuine boundaries and retained.

P3 J1: ROCm begin_forward built the admitted single-phase plan twice. Reusing its first `RocmSequencePlan` preserves metadata, slots, timer scopes and lifecycle outputs exactly; 5 before / 5 after tests and byte-identical snapshots. Two metadata fields remain because begin/end and inactive timer handling still use that supported state representation.

P3 S5: profiling ModelConfig duplicated canonical family dispatch to accommodate a monkeypatch and overlaid already-parsed model fields. It now delegates to `resolve_runtime_attention_family` and reuses BaseModelConfig parsing. Verification: 129 PASS; all 22 model snapshot entries exactly equal. Raw model_type and linear-shape overlays remain because explicit-null/string serialization is an actual compatibility concern.

Architecture identity cleanup: duplicated Qwen type/alias recognition in three gates is now owned by `model_architectures.py`; role-specific profile precedence and normalization remain explicit. Verification: 100 tests before/after, including nine identity cases. The GDN exported adapter retains a function-local import because direct re-export was observed to create an initialization cycle.

| Pattern | Supported runtime state / contract | Inspected evidence |
| --- | --- | --- |
| Optional shared manager/training paths | Ordinary monolithic trains independently; hybrid/disaggregated share artifacts | Simulator construction and non-dummy cases |
| Component copying and optional operator maps | Mutable physical-layer isolation required; unmeasured operator maps can be absent | ExecutionTime finalization tests |
| Missing GDN config and optional max_num_seqs | Dense models have no recurrent state; only GDN reservation requires capacity | Planner constructor/reservation, dense and GDN memory tests |
| GDN slot-manager None and rollback catch/rethrow | Non-GDN schedulers do not allocate slots; failed KV allocation must free the reserved slot and propagate the original error | Scheduler constructor, allocation transaction, failure cleanup tests |
| Explicit dense FFN metric projections | Historical metric label `mlp_activation` differs from family trace name `mlp_act`; direct enum conversion is invalid | Reconciled M3 proposal against constants.py, FFN family and existing scalar metric emitter; preserve output schema without a new alias layer |

## Large-module analysis

Completed in `large_module_review.md`: all seven critical modules above 2,000 lines already exceeded the soft limit on main. Candidate-specific cleanup precedes any extraction. The report records retained ownership and concrete functional split sequences for base/MoE/disaggregated predictors, shared manager, metrics store, vLLM V1 scheduler and aggregate config. No broad split is required to preserve this task's bounded scope. L2 private single-layer cleanup is committed; L1 manager family reuse remains in P4.
