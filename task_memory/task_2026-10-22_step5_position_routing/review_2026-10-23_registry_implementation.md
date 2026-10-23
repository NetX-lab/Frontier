# Native binding registry implementation review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Parent scope/reachability review classifies independent public binding outside D29; mixed-view P2 remains open. |
| 2026-10-23 | Record mixed legacy/native enumeration P2 and independent public-consumer dependency; assign separate follow-up work. |

TaskID: D29-REGISTRY-IMPLEMENTATION-REVIEW. Reviewer:
ses_e9abd8364ffee5sQKdpiQh3yH4. Source8f96408c, documentationb6714757,
exact four-path patch and relevant frozen callers/specification. Parent independently
verified213 PASS/0 SKIP and a30-artifact CPU bundle with both real FP32 gate bindings.
Reviewer did not run tests or platform operations.

## Finding: P2, mixed-domain family enumeration

prediction_model_registry.py534–543 first emits native hash labels, then resolves
the legacy bare name through `_get_family_model()` without a selector. A same-name
legacy estimator plus native binding reaches the ambiguity error at447–451,
breaking `get_models()` and cluster admission. Lossless enumeration should retain
both native hash labels and the original legacy projection. Restrict bare projection
to the legacy/non-native domain; keep strict single-model ambiguity checks. Add
mixed family/cluster-view coverage. Owner: original REGISTRY implementer.

The explicit unknown native binding does not fall back to legacy. Tests cover
two actual FP32 gates, all-native enumeration/bundles, original scalar typed markers
and existing hashes. Those checks do not cover the mixed-view failure.

## Independent consumer dependency

This is pre-existing, not introduced by the four-path delta. Independent automatic
training merges dictionaries with `models.update()` in moe_dataset_training.py376–379,
overwriting same-name bindings. Its public constructor retains `_model_manager=None`,
but actual point/model consumers require a manager in moe_routing_workload.py137 and
sklearn_moe_execution_time_predictor.py128. Actual independent fit/cache PASS does
not qualify the independent public chain or complete D29 dual-path acceptance.

Root assigns a read-only bounded investigation of the actual independent public
configuration, approved capability, all affected retention/binding/consumer seams
and smallest correction reusing the existing registry. It must report paths, owners,
budget and real public-constructor/query checks before implementation allocation.
This dependency is not silently deferred or added to the small mixed-view repair.

Parent disposition after the bounded investigation: the actual approved P/D
product entrypoint always supplies the shared manager, and position routing rejects
monolithic mode. The design requires both trainer fit/cache paths, not a new
independent public consumer. Therefore the observed latent issue is outside D29
scope and is not a closing prerequisite. The earlier broader parent interpretation
is withdrawn; no such support is claimed. See repairs/independent_consumer_scope.md
and requirements.md D29-TRAINER-CONSUMER-BOUNDARY. The mixed-view P2 is unaffected.

## Root assignment and limits

D29-REGISTRY-VIEW-REPAIR uses the original registry and binding-test paths with
at most58 further gross lines, preserving the360 repair ceiling. Root transfers
20 unused predictor allowance to the binding test: predictor118 (actual113),
test200 (actual174), registry120 unchanged. CORE4404,109 paths and10000 overall
remain unchanged. See core_review_scope_amendment.json after the registry overlay.

Other CORE/PRICING repairs remain separate. Actual-main, native/GPU/graph/G3 and
numerical qualification are not established by this source/CPU review.
