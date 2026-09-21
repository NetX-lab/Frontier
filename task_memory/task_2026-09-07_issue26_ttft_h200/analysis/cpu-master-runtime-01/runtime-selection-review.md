## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Reviewed explicit uniform runtime selection and identified shared-model identity requirements. |

# Read-only runtime selection review

Reviewer: `/root/cpu_frontier_prepare`. Scope: `analysis/uniform_runtime_selection_proposal.md`, actual simulator config, resolver, shared manager, predictor, and focused test sources. No core changes or numerical simulation.

## Outcome

The explicit optional runtime selector is needed to preserve balanced expert loads while selecting measured uniform_topk routing latency. No existing simulator config provides the same result. The proposal's data-selection boundaries are correct, but a public per-replica selector also needs runtime-sensitive in-memory model identities. Merely changing the four resolver calls leaves a cross-cluster reuse/overwrite defect.

The standalone training CLI (`frontier/training/cli.py:152`) and `MoETrainer` (`frontier/training/moe_trainer.py:176`) already accept routing_runtime_path. They do not override simulator consumers, which independently resolve balanced to standard_fused_topk. Changing distribution to random generates nonuniform random expert weights in predictor lines794–803. Supplying uniform-only CSV with balanced currently fails the exact metadata filter; renaming CSV metadata would misrepresent the runtime.

## Config and consumer inventory

- `frontier/config/config.py`: ReplicaConfig distribution field1937 and validation2011. Add the optional selector beside this ownership. Preserve it through `_create_replica_config_from_fields` (4262–4285) and `_create_replica_config_copy` (5021–5034). The existing flat dataclass mechanism exposes the base ReplicaConfig field; verify the effective config rather than assuming a field declaration reaches every cluster.
- `frontier/moe_routing_runtime.py:24`: canonical resolver owns distribution-to-runtime default mapping. Its existing validator at14 owns accepted runtime membership. Optional empty/unset must preserve existing mapping; explicit values pass the same validator. Invalid distributions must not become accepted merely because an override is provided.
- `shared_prediction_model_manager.py:1235`: dataset preflight selection. `:1381`: training dataframe selection. Both must resolve the same override.
- `sklearn_moe_execution_time_predictor.py:271`: helper used by dataset preflight1230 and training1365. Constructor716 also resolves and stores `_moe_gating_routing_runtime_path` before parent initialization; this stored field currently has no subsequent read. Do not change only the stored value while leaving the helper to re-resolve the old default. Reuse this boundary to eliminate duplicate runtime resolution while retaining object.__new__ test compatibility where relevant.
- Predictor expert allocation at794–825 stays untouched: balanced uses equal weights. Runtime metadata selection is independent of allocation.
- Existing op filtering narrows runtime only for `moe_gating_routing_topk`, with both standalone_legacy and `__prefill_hot` contexts. Other operations keep their existing selection rules.

## Required shared identity correction

1. **Training deduplication:** manager `ffn_signature` at1318 and `model_signature` at1483 omit routing runtime. Two clusters with identical model/device/TP/typed contracts but different selected routing runtimes share the first FFN signature and skip the second. Include resolved runtime in the existing MoE FFN dedup identity. Keep non-MoE identities unchanged; preserve normal sharing for equal resolved runtimes.
2. **Model storage:** `_store_model_precision` at4172 stores a typed model under `(model_name, layer_identity)` and an untyped model under `model_name`. Distinct routing implementations with identical typed contracts therefore overwrite or alias even if training dedup is fixed. Extend the existing model identity for routing-topk family members, including `__prefill_hot`; do not build a parallel registry or put runtime into the architecture's physical layer contract.
3. **Model projection and retrieval:** `_get_family_model` at4197, `_models_view_for_family` at4313, and `get_model` at4345 must select the matching runtime dimension. Cluster projections derive the runtime from the cluster's ReplicaConfig. Unqualified lookup with several available runtime variants must reject ambiguity, consistent with the existing layer-contract behavior. Precision/eager/kernel-only isolation remains intact.
4. **Cached/new model storage paths:** `_train_single_model` at2706 calls storage on both cached and newly trained models. Carry the selected runtime through its existing training context and storage call paths so cache hits retain runtime identity. Current persistent hash at3949 includes the entire selected dataframe, including routing_runtime_path, so differing runtime rows already generate different disk hashes; no new standalone fingerprint mechanism is needed.

These corrections remain in the proposed four production modules, but exceed the original narrow resolver-only 40–70-line estimate in likely implementation size. Review the concrete diff before claiming that bound. `get_model` may need an optional runtime argument; current production search found no direct `.get_model(...)` callers, while `get_models_for_cluster` is the main runtime projection surface.

## Focused existing tests and missing coverage

Existing useful checks/templates:

- `tests/unit/test_moe_routing_runtime.py`: default structured distributions map to standard, random to uniform, removed distribution values fail. Preserve all five cases and add balanced+uniform override, invalid override, empty default cases.
- `tests/unit/test_moe_predictor_layer_id_semantics.py::test_global_routing_allocations_use_canonical_distribution_types`: balanced ratios remain equal; other modes remain nonuniform. Add independence assertion for the explicit runtime override.
- `tests/unit/test_pdaf_deferred_trace_contract.py::test_replica_config_copy_preserves_moe_routing_distribution_type`: template for selector copy propagation. Add direct/flattened effective config coverage.
- `tests/unit/test_profiling_governance_minimal_red.py::test_manager_typed_registry_is_canonical_and_requires_context_for_ambiguity` and `::test_manager_projection_selects_cluster_contract_and_rejects_unresolved_variants`: templates for two runtime variants under an identical typed contract, exact per-cluster projection, and unqualified ambiguity. Existing tests vary TP/layer contracts; they do not detect the runtime collision.
- Same file `::test_train_single_model_uses_typed_cache_hit_without_training`, `::test_manager_cache_hash_uses_selected_semantic_contract`, and `::test_manager_cache_marker_is_written_and_validated`: preserve typed cache semantics when adding runtime metadata.
- `tests/unit/test_moe_ep_non_dummy_matrix.py::test_pdd_profile_validation_uses_runtime_op_level_tp_policy` confirms routing-path mismatch fails under existing default behavior. Do not relabel that default to uniform globally.

A focused new shared-manager test must exercise two same-shape clusters with distinct selected runtime rows, prove both required routing models are selected/trained, and retrieve the correct one for each cluster. Neither fresh disk caches nor standalone resolver tests detect in-memory cross-cluster aliasing. Include both normal routing-topk and prefill_hot pseudo-models and ensure identical resolved runtimes still share.

## Practical limits

This is source review, not a test PASS or actual runtime-override implementation. Existing CPU environment and htsim viability evidence remain in the adjacent runtime report. Current single MONOLITHIC uniform experiment does not itself exercise heterogeneous-cluster model sharing, but the proposed public per-replica interface makes that case reachable; the complete bounded feature must include the identity correction.
