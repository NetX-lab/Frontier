## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Independent source review of W03 ownership, live producers and consumers, mutation boundaries, and legitimate fixture migration. |
| 2026-09-16 | Added independent finalized-stage contract tests and recorded the first implementation comparison: 13 passed, 2 failed. |

# W03 timing ownership contract review

## Review scope and evidence status

- Target Component/Phase: W03 in `Frontier_PR33_New_Execution_Plan_2026-09-16_EN.md`.
- Reviewer Agent Identity: `/root/w03_contract_review`.
- Inspected Artifacts: `frontier/entities/{execution_time,stage_execution_time,time_components}.py`; base, dense, MoE, and disaggregation predictors; replica stage scheduler; collective timing and metrics helpers; stage-aware operation metrics and trace/ledger emitters; relevant unit fixtures.
- Remediation/Verification Code Actions Taken: source inspection only; no production or test modifications by this reviewer. This document belongs to the reviewer; root owns implementation.
- Practical limit: source findings establish reachable structural contracts, not measured E2E correctness or performance. HEAD inspected is `41777755`; timing-entity files have no working-tree diff. The reviewer initially inferred source mutation from assignments in the stage getter without inspecting its callee, then incorrectly attributed defensive snapshots to concurrent edits. Both claims were withdrawn after checking `git show HEAD`. The accurate getter finding is recorded below.

## Required before/after contract

| Surface | Initially observed contract | Required coherent contract |
| --- | --- | --- |
| `ExecutionTime` | Identity absent means `_legacy_aggregate=True`; scalar accessors and `model_time` multiply by a constructor count. Identity present silently switches the same object type to one layer. | One real layer regardless of identity presence. Production supplies its actual complete identity. Stage count cannot change layer numerics. Characterize any retained external compatibility separately from production. |
| `StageExecutionTime` | Ordered layers plus an owner that is another `ExecutionTime`, often the first layer. Generic component/operator accessors return first-layer values while scalar public names sum layers. | Ordered distinct real-layer identities plus explicit once-only work. Generic stage operator/component views are aggregates; layer access is explicit and identified. Mixed MLP/MoE stages cannot masquerade as the first layer's component type. |
| Stage assembly | `from_execution_time` repeats payloads with default dense/unknown identity. MoE recursively builds one-layer stages and unwraps them. Disaggregation wraps an internal legacy-stage producer. | Direct one-layer prediction, explicit real identities, and one stage assembly. Reuse numerical attention lookup only when query equality is proven; independently compute layer-specific MoE routing. |
| Scheduling totals | Layer block formula plus owner PP/draft/terminal, then active CPU overhead. | Preserve this formula and seconds/milliseconds boundaries exactly. Do not replace critical-path totals with a flat diagnostic operator sum. |
| Mutation | Defensive component/operator snapshots, explicit setter mutation, copy-on-write flags, and per-layer mutation versions; aggregate cache polls versions. | A finalized publication boundary. Prefer finalized payload references and one-time stage totals. If explicit mutation is retained, one boundary must isolate inputs and returned values and keep aggregates current. |
| Metrics handoff | A private-field copier creates a new identity-free `ExecutionTime`, drops typed operator maps and MLA/identity fields, and adds trace override attributes. | Pass the authoritative stage or explicitly selected layer to an ownership-aware consumer. Metrics must not reinterpret stage totals as representative-layer scalars. |

## Field inventory using existing metadata

The existing component types and canonical operator mapping already provide the necessary classification. Do not add another complete list of all timing fields to `StageExecutionTime` or metrics.

1. **Layer work:** all fields of `AttentionTime`, `MLPTime`, `MoETime`, and `ResidualTime`; communication operators other than `pipeline_parallel_send_recv`. Scalar compatibility aliases (`attention_time`, `mlp_all_reduce_time`, `share_expert_time`, `moe_gating_time`, residual sums) are computed views of those owners, not independent values.
2. **Once-only stage work:** PP send/receive; the existing `OverheadTime` component; `decode_draft_proposer_time`; `mtp_terminal_overshoot_time`. Reuse `OverheadTime` directly instead of copying its complete list into each consumer. The two terminal scalars do not justify a second component hierarchy.
3. **Diagnostic-only overhead:** `OverheadTime.pp_stage_boundary_handoff_time`. `simulated_total_time()` omits it; `diagnostic_total_time()` includes it. All other current overhead fields participate in the active overhead sum.
4. **Identity:** global layer ID, attention family, and attention variant. Those are labels resolved from real model/config bindings, not aggregation switches.
5. **Operator maps:** `canonical_operator_execution_time_attrs()`, the four existing `build_*_operator_times_from_op_times()` functions, and family `e2e_trace_ops()` are the existing seam. PP is the one stage-owned physical communication operator. Aggregate layer maps once, include PP once, and project typed views through these existing builders.
6. **External units:** physical components and `*_time_ms` are milliseconds; `model_time`, `total_time`, and `diagnostic_total_time` are seconds. Collective event helpers convert layer or pipeline milliseconds with `1e-3`.

Preserve `ExecutionTime._get_block_execution_time()` and its established component ownership. Preserve the separate named EP dispatch/combine phases and explicit zero TP overrides; those are meaningful semantics currently covered by tests.

## Mutation findings and smallest defensible boundary

`StageExecutionTime.communication_time_component` fetches the first layer's component and assigns `pipeline_parallel_send_recv_time` and `operator_times` into it. **At current HEAD this component is already a detached `deepcopy` returned by `ExecutionTime.communication_time_component`.** The assignment therefore does not establish source, first-layer, or sibling mutation. The same defensive-snapshot mechanism applies to public component/operator getters. N02's initial causal claim must be corrected against current code; repeated-read/source-isolation tests should retain their oracle, rather than asserting an unobserved defect. The remaining observed issue is generic stage getter scope: it returns first-layer communication plus owner PP, while `op_times` uses stage aggregation. Repeated defensive copying is also an observed operation whose performance impact requires measurement.

Repository search found **no production callers** of `override_moe_grouped_gemm_time`, `override_moe_times`, or the four public operator-map setters. Their current callers are tests. Production does perform identity adoption/copying and attaches diagnostic trace attributes to metrics copies. Thus there is no demonstrated runtime need to carry arbitrary post-publication numerical mutation through every scheduler read.

Recommended boundary:

1. Keep predictor component construction mutable locally, because existing predictor code builds these dataclasses incrementally.
2. Publish a finalized numerical snapshot into one-layer `ExecutionTime`; identity records may reference that finalized snapshot. Reuse the existing component/operator types rather than inventing a parallel timing IR.
3. Publish a stage with finalized model, active-overhead, and diagnostic-overhead totals. Aggregate maps once or cache a finalized map lazily without per-read generation scans.
4. Public component/map views must be read-only or detached snapshots. A returned map must never mutate the published snapshot.
5. If old public mutation APIs are retained, make them an explicitly supported pre-publication or replacement boundary and test it. Do not keep copy flags, version scans, deep copies on every getter, and a separate immutable layer design simultaneously.

Characterize valid setter semantics before changing tests. Rejecting or replacing a published layer mutation is an API contract change that needs explicit migration evidence, even though production currently has no such callers.

## Real producer migration inventory

| File / functions | Required migration |
| --- | --- |
| `base_execution_time_predictor.py::_get_dummy_execution_time` | Currently constructs with `_num_layers_per_pipeline_stage`. Produce numerical one-layer data. Public abstract `predict_stage_execution_time` return annotation/docstring should describe stage result. |
| `sklearn_execution_time_predictor.py::predict_stage_execution_time` | Split direct layer prediction from stage assembly. Replace default dense/unknown identities with model-owned layer specs. Preserve pipeline and CPU calculation once per stage. Multi-layer dense numerical reuse must preserve every real layer identity and respect heterogeneous attention. |
| `sklearn_moe_execution_time_predictor.py::_get_dummy_execution_time`, the `ExecutionTime` assembly near initial line 2752, and `predict_stage_execution_time` | Both constructors initially pass `_num_layers_per_pipeline_stage`. Extract a direct layer result path; replace recursive public calls around initial lines 3561–3579. Keep `layer_id` in `_get_moe_tokens_input` and routing work for every actual layer. Remove dummy sentinel pass-through. |
| `sklearn_disaggregation_execution_time_predictor.py::_get_dummy_execution_time_for_cluster`, `_predict_attention_only_stage_execution_time`, `predict_stage_execution_time`, `_predict_stage_execution_time_legacy` | Replace internal legacy wrapper with direct layer prediction and stage assembly. Numerous constructor branches pass `num_layers`; move count ownership to stage. Remove sentinel pass-through. Audit dummy scaling branch: it scales component values by a layer ratio and constructs an aggregate count, a dangerous double-scaling combination; direct single-layer output removes that ambiguity. |
| Dense/MoE predictor helper calls near dense lines 5590/5605 and MoE line 3319 | Helpers currently call public stage prediction to obtain components; migrate helpers to the direct layer seam where they need one layer, preserving which subclass owns numerical prediction. |

The disaggregation role rules must survive: PREFILL and unified DECODE can include a full layer, DECODE_ATTN is attention only, and DECODE_FFN is FFN only. A layer identity is still required for filtered work. Repeated attention query keys do not establish repeated MoE routing numerics.

## Real consumer migration inventory

| File / functions | Observed dependency and migration |
| --- | --- |
| `replica_stage_scheduler/replica_stage_schduler.py::predict_and_create_stage` | Initially sets `layer_id=0` except DECODE_FFN. Ordinary PP stages must start at their actual global layer bound. Use the established stage-bound calculation rather than assigning identical IDs to every stage. PD-AF ping-pong remains one layer per call. |
| `events/replica_stage_schedule_event.py` | Several attention-only calls intentionally process one real layer. Their `get_single_layer_attention_scope_time()` access should select/assert that actual one-layer result, rather than rely on a generic stage first-layer facade. |
| `scheduler/utils/{collective_timing,prefill_collective,decode_collective,expert_parallel,ep_wave_schedule}.py` | Layer-local event paths need explicitly selected layer phases; final timing helpers need stage PP/CPU/draft values. `decode_collective` final full-stage call initially omits `layer_id`, so PP identity migration must include it. `prepare_decode_final_timing` treats terminal overshoot separately from its total and records completion delay; preserve that protocol. |
| `scheduler/utils/execution_time_metrics.py::build_single_layer_metrics_execution_time` | Replace the private-field reconstruction. It consumes the private catch-all for `_has_attn_tensor_parallel_allreduce_time`, `_has_moe_tensor_parallel_allreduce_time`, PP/CPU/draft fields; it omits canonical maps, identity, and MLA fields. Creating a new entity for an adapter is unnecessary. |
| `scheduler/utils/dense_metrics.py::{predict_dense_reference,build_prefill_metrics_execution_time}` | Explicitly unwrap/identify a one-layer result. The three private MLP projection probes should use public layer fields. Existing mixed-dense trace annotations must not become hidden numerical ownership overrides. |
| `cluster_scheduler/base_cluster_scheduler.py::_create_prefill_corrected_execution_time_for_metrics`, `_create_corrected_execution_time_for_metrics` | Migrate these delegating callers together with the adapter; their current `actual_execution_time_ms` argument does not determine numerical payload contents. |
| `metrics_store.py::_emit_stage_layer_traces`, `_push_stage_layer_operation_metrics` | Already iterate real layer records. Preserve and consolidate this path; it is evidence that a completely new reporting IR is unnecessary. |
| `metrics_store.py::_build_frontier_stage_batch_component_ledger` | Initially duplicates a long `owner_only_names` list and recursively rebuilds layer and owner ledgers. Use the common owner projection instead; keep export names and units stable. |
| `metrics_store.py::_emit_op_level_traces`, stage schedule/end callbacks, trace override handling | Preserve one stage overhead emission and physical layer identities. A first-layer metrics adapter must not route a real heterogeneous stage through homogeneous reconstruction. |

### Private delegates actually consumed

The stage's catch-all is not merely hypothetical: the metrics copy helper reads `_has_*` flags, pipeline, CPU, and terminal scalars that are absent from `_PER_LAYER_PRIVATE_NAMES`, and gets them via owner fallback. Listed private layer scalars are read by that same helper and `dense_metrics.py`. `_is_moe` is used in metrics and dense-reference validation; a mixed-stage `None` is not a useful substitute for explicit per-layer classification. `_num_layers_per_pipeline_stage` should not be an internal stage escape hatch. Dynamic `_trace_*` annotations belong to diagnostic handoff, not private numerical delegation.

## Legitimate fixture migration and invariant preservation

- `tests/unit/test_stage_execution_time.py`: remove identity-dependent aggregate expectations, copy-flag implementation assertions, and `test_stage_private_fields_remain_single_layer_for_metrics_adapter` after migrating that adapter. Keep numerical stage sums, nonzero IDs, ordered families, duplicate rejection, once-only PP/terminal/CPU, entity-ID stability, and isolation. If mutation is replaced, assert the explicit replacement contract and stability of already-published stages.
- `tests/unit/test_execution_time_op_times.py`: its common fixture initially creates count 2. Tests of layer operators should use one-layer fixtures; tests of two-layer totals should create a stage explicitly. Preserve all expected physical operator values, explicit-zero TP handling, named EP phase conservation, invalid-input atomicity, and canonical-map consistency. The corrected-copy tests should exercise the replacement handoff rather than resurrect private delegation.
- `tests/unit/test_moe_ep_aggregate_admission.py`: dummy sentinels near initial lines 325, 345, 363, 539, and the cluster helper `SimpleNamespace(num_layers=1)` near 589 must become valid numerical `ExecutionTime` fixtures. Continue asserting admission order and rejected-path helper non-invocation. Public successful prediction now returns a valid stage instead of sentinel identity.
- `tests/unit/test_spec_decode_mtp_structural_moe_replay.py`: sentinel return around initial lines 181–206 requires the same typed-fixture change; preserve structural replay call assertions.
- `tests/unit/test_pd_decode_moe_layer_accounting.py`, `test_moe_predictor_layer_id_semantics.py`, and disaggregation predictor tests: count-bearing fixtures must distinguish one-layer numerical payload from stage aggregation. Preserve original model/routing cardinality and PD/PDAF loop counts; do not merely reduce expected outputs.
- `tests/unit/test_attention_trace_mapping.py`: setter-based construction can move to constructor-supplied maps if finalized publication removes setters. This is a fixture assembly change; expected physical trace names/times must stay fixed.
- `tests/unit/test_metrics_stage_execution_time.py`: keep stage-aware tests and add nonzero PP offsets and mixed dense/MoE layer numerics, then cover real adapter handoff so helper-only success cannot conceal a live caller regression.

## Bounded acceptance evidence for implementation

1. For two layers with distinct nonzero IDs and different routing times, compute an independent oracle: sum the existing layer block formula, add PP/draft/terminal once, then active CPU once. Separately compare diagnostic overhead. Include explicit-zero TP and a named EP phase case.
2. Read every stage component/map repeatedly; mutate returned snapshots or verify read-only rejection. Assert source/layer/sibling values and finalized totals remain unchanged. Test a supported explicit replacement boundary if retained.
3. Exercise actual dense, MoE, and disaggregation public predictors through scheduler and metrics; count direct layer predictions and stage assemblies, confirm actual global IDs and family identities, and inspect selected metrics/trace rows.
4. Confirm no remaining production sentinel pass-through, `_predict_stage_execution_time_legacy`, identity-driven aggregation, or private stage catch-all.
5. Compare numerical outputs and entity-ID sequences against characterized valid behavior; then perform the plan's controlled allocation/copy/aggregation ablations. Passing source search alone does not establish performance or runtime correctness.

The smallest coherent implementation spans producers, stage entities, the two metrics adapters, and their real scheduler/metrics consumers. Fixing only getters is useful containment, but does not satisfy W03's required ownership migration.

## Independent contract tests against the developing implementation

Root subsequently authorized ownership of the new file `tests/unit/test_stage_finalized_contract.py`, with this concrete target: one-layer numerical `ExecutionTime`, finalized stage snapshots, aggregate generic getters, complete stage identities, single-layer-only probes, and no private owner catch-all. The reviewer added 15 independent cases without editing production code or existing tests.

Execution environment: `/usr/bin/python`, Python 3.12.3 (GCC 13.3.0); no conda activation was used. Exact command from the worktree root:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR python -m pytest tests/unit/test_stage_finalized_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w03-finalized-contract
```

Observed result after adding returned-layer and component isolation cases: **13 passed, 2 failed in 1.12 seconds**. This run was against root's developing working-tree implementation, not unchanged HEAD.

Independent numeric oracle: dense layer ID 7 has a 10 ms block; MLA/MoE layer ID 9 has a 228 ms block with 31 ms grouped GEMM, 37 ms routing, and 41/43 ms dispatch/combine. Stage owner adds 11 ms PP, 17 ms draft, and 19 ms terminal work: model = 285 ms, active CPU = 13 ms, total = 298 ms, diagnostic-only overhead = 23 ms, diagnostic total = 321 ms. Communication view = 147 ms. Non-owner once-only fields deliberately contain 999 ms to expose duplicate charges.

Passing criteria observed: identity-independent layer arithmetic; distinct routing values; aggregate attention/MLP/MoE/communication maps; aggregate attention/communication components; immutable stage arithmetic even when a source changes before the first read; finalized returned-layer mutation rejection; returned-map/component isolation; repeated communication reads preserve sources; owner PP scope; rejection of partly missing identity; removal of private catch-all; and entity ID stability.

Open failures delivered to root:

1. `test_published_stage_requires_complete_layer_identities[layer_ids0]`: constructing a stage containing one identity-free layer did not raise `ValueError`.
2. `test_single_layer_probe_requires_an_unambiguous_layer`: invoking `get_single_layer_attention_scope_time()` on a two-layer stage did not raise `ValueError`.

Both errors were `Failed: DID NOT RAISE <class 'ValueError'>`. The test expectations follow the implementation contract supplied by root. They should be resolved in production, not removed or weakened. These focused tests establish numerical publication behavior only; producer-to-scheduler-to-metrics validation and performance measurement remain with root. No commits were made by this reviewer. Root should include this evidence in the task's consolidated test report.
