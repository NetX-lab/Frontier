# CORE/SOURCE implementation plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Start sole-writer implementation. |

Dependency: count/source/tensor/verifier interfaces -> {scheduler/runtime, source package/producers, selected trainer/profiling consumers} -> fixture/runner -> root integration qualification.

1. Publish immutable count/source descriptors, canonical source tensors and full-receipt callable verifier with focused real consumer tests and an interface receipt.
2. Thread explicit sources through shared materialization and lane-aware cache; preserve ratio behavior and reject unsupported serving configurations before parent initialization.
3. Consume PRICING's selected specs/native schemas and NATIVE's factory/context contract in existing trainers/profiling consumers. No peer fallback.
4. Extend source packages/launcher/attention/linear/MoE identities without changing legacy outputs.
5. Add a fresh CPU-only actual-main fixture and runner integration using peer contracts; retain old fixture/workload/timeout.

Acceptance: exact P/D/chunk/mask conservation, source identity/cache binding, actual module/callable receipt negatives, selected frame/schema paired exact keys including empty physical scopes, real timer replay/restore/event preservation, meaningful source/consumer checks, owned-path/quota/AST/whitespace accounting. Native/runtime/E2E qualification remains root-owned.
