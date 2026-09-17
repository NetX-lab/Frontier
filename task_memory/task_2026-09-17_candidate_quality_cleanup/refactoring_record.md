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

## Intentionally retained patterns

Architecture identity cleanup: duplicated Qwen type/alias recognition in three gates is now owned by `model_architectures.py`; role-specific profile precedence and normalization remain explicit. Verification: 100 tests before/after, including nine identity cases. The GDN exported adapter retains a function-local import because direct re-export was observed to create an initialization cycle.

| Pattern | Supported runtime state / contract | Inspected evidence |
| --- | --- | --- |
| Optional shared manager/training paths | Ordinary monolithic trains independently; hybrid/disaggregated share artifacts | Simulator construction and non-dummy cases |
| Component copying and optional operator maps | Mutable physical-layer isolation required; unmeasured operator maps can be absent | ExecutionTime finalization tests |
| Missing GDN config and optional max_num_seqs | Dense models have no recurrent state; only GDN reservation requires capacity | Planner constructor/reservation, dense and GDN memory tests |
| GDN slot-manager None and rollback catch/rethrow | Non-GDN schedulers do not allocate slots; failed KV allocation must free the reserved slot and propagate the original error | Scheduler constructor, allocation transaction, failure cleanup tests |

## Large-module analysis

Pending candidate-specific inspection. Remaining large modules require concrete scope, technical reason and a functional split proposal; no repository-wide split is implied.
