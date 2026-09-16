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

## Intentionally retained patterns

| Pattern | Supported runtime state / contract | Inspected evidence |
| --- | --- | --- |
| Optional shared manager/training paths | Ordinary monolithic trains independently; hybrid/disaggregated share artifacts | Simulator construction and non-dummy cases |
| Component copying and optional operator maps | Mutable physical-layer isolation required; unmeasured operator maps can be absent | ExecutionTime finalization tests |

## Large-module analysis

Pending candidate-specific inspection. Remaining large modules require concrete scope, technical reason and a functional split proposal; no repository-wide split is implied.
