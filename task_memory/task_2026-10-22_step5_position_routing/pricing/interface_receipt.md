# Pricing interface receipt

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Publish first selected-operator and immutable plan interfaces. |
| 2026-10-23 | Publish H04, selected native schema and reporting seams; record authorized imports. |
| 2026-10-23 | Publish immutable-list P2 repair and completed authentic dependency import. |
| 2026-10-23 | Correct the native prepare triple-return seam using published source execution. |
| 2026-10-23 | Deliver T2 optional constructor-limit binding and CPU invariants. |
| 2026-10-23 | Close selected producer/parent selection; archive model payload and real dependencies. |

TaskID: D29-PRICING-IMPLEMENTATION. Baseline: 1aee0af.

## Selected MoE operators

`frontier/operators/families.py:get_moe_profiling_operators(context=None)` returns
ordered OperatorSpec objects from MOE_FAMILY. Context is a mapping containing
`model_architecture_profile`, `routing_runtime_path` and, for Step position mode,
actual native `quant_dtype` (None or `float8_e4m3fn`) and `defer_input_quant` (bool).
The supplied context must come from native selection; weight FP8 alone is not
prepare evidence. No-context preserves the four legacy targets in their order.

- BF16/deferred prepare: gate linear, routing topk, inclusive grouped GEMM.
- Reached nondeferred FP8: gate linear, routing topk, input quantize, inclusive GEMM.
- No separately charged shuffle in position mode.

`moe_input_quantize`: MOE_FAMILY / ROUTED, RESHAPE, MEMORY, MOE_TP,
EP-agnostic, execution attr `moe_input_quantize_time`. Actual model resolver
at layer3/TP1/EP16 succeeds. Registry inventory includes it without requiring
it as a legacy profiling target.

`step5_operator_binding(..., context=...)` binds FP32 gate/router and FP8 prepare
for this selected path; context is part of fixed binding identity. Legacy
no-context binding remains unchanged. CORE must use the same context for target
selection, exact duration/count pairing and model/cache identities.

## Plan and backend capability

`frontier/cc_backend/collective_plan.py` defines immutable `CollectiveTensor`,
`CollectivePlan`, `CollectivePlanCost`. Plan includes ordered exact physical DP2
sizes, tensor geometry/dtypes/strides, all eight DP pairs or both TP8 groups,
actual graph mode, native/context and buffer receipts. `stages`, `to_dict()`,
`cache_key()`, `endpoint_bytes(rank)` preserve size/root/tensor order; no averaging.
Cost requires sixteen actual paired native kernel counts and a count receipt,
separate from algorithmic stages; duration must equal N+max(local)+remainder.
CPU fixtures remain explicitly CPU evidence.
Caller-owned list inputs (including nested groups and result arrays) are copied
to tuples before validation; subsequent caller mutation cannot alter identity.

CommOperatorSpec supports optional `plan_builder`; scalar payload/alias plus plan
is rejected. BaseCCBackend.predict_collective_plan raises explicit unsupported
capability; no other backend silently approximates a plan.

## Shared native-call planner

`forward_collectives(..., native_call_names=None)` is the sole selected planner.
Explicit Step subset must be nonempty/distinct/reached and retains native order.
Only explicit selection changes combine descriptor to
`reduce_scatterv_into_output` / `AgRsAll2AllManager.combine_into_output`.
No-option legacy list and endpoint bytes are unchanged.

## CORE-owned required numerical facts

Source gate/router/prepare/shared and post-combine TP widths are physical ni,
even with live0. Inclusive expert frame is gathered M=sum(ni), with local live
assignment counts separately retained. Every applicable physical scope needs
paired exact duration/count evidence; zero-live is not an early-return rule.
Native inclusive experts own permutation/GEMMs/intermediate work/unpermute and
weight finalization, excluding source prepare and all communication.

Native DEEPGEMM uses actual alignment/capacity/layout evidence, not Triton padded
all-expert sum. CPU-count rounded M and native physical-capacity bound remain
distinct. Actual native auto selection (including E8M0 OR shape/layout criteria)
must be retained rather than forcing TRITON.

## Dependencies not duplicated

CORE/SOURCE02f061d and Native2c4cc9e are now imported under root authority.
Pricing uses their immutable source, source_frame_tensors and actual callable
verifier; no replacement API. Authorized67d5b5e (accepted native repair and
authentic published image/source receipts) is imported as437b312 after511a3db.
This provides dependencies, not evidence of GPU qualification.

## Validation

Thread-capped frontier-py310, no bytecode/cacheprovider: selected pricing,
comm-family and Step binding-reuse suites: **60 passed** in10.81s. This establishes
CPU contracts, not measured GPU parameters/native-runtime qualification.

## Native prepared scope (second interface commit)

`NativeRoutedExperts.prepare_source(hidden_states)` calls the original
`naive_dp_ep._quantize_and_setup_dispatch` and returns `(activation, scales)`.
The native function actually returns `(activation, dispatch_scale_list, scale)`;
this seam returns only activation and scale, not the intermediate dispatch list.
`prepared_context(activation, scales, *, topk_ids=None, expert_map=None, source_tensors=None)` returns actual per-shape selected class,
quant_dtype, defer_input_quant, oracle backend and tensor shape/stride/dtype/size.
It executes the native hybrid selector, whose condition is E8M0 **OR** valid
DeepGEMM shape; backend labels alone must not infer the selected implementation.
`apply_prepared(activation, scales, weights, ids, expert_map, *, in_dtype)` calls
original modular `_fused_experts`, then original NoDPEP finalization into output.
It never invokes full `kernel.apply` or repeats source prepare/communication.
Manifest-bound construction passes actual config/oracle/conversion/construction,
returned kernel/expert/selector/finalizer callables to SOURCE's verifier.

CORE-owned schema/range facts: TritonExperts uses native config BLOCK_SIZE_M at
gathered physical M and `_prepare_expert_assignment(ignore_invalid_experts=...)`.
The flag is expert_map present, no LoRA, and no global a2 activation scale. The
accepted BF16/block-FP8 path drops remote/masked slots; record actual assignment
output, not all-expert padding. DeepGemmExperts calls
`compute_aligned_M_and_alignment(M,K,local_experts,actual_alignment,None)` because
this adapter supplies expert_tokens_meta=None. Capacity is native physical
`round_up(M*K + min(M*K,local_experts)*(alignment_used-1), alignment_used)`.
CPU-count rounded local-live M is a different diagnostic and not that capacity.
Pass actual gathered IDs/map to obtain native_padding and workspace_shapes.
Pass `(source_activation, source_scales)` as source_tensors to retain source
activation/scale layout separately from the gathered activation/scale layout.
Hybrid workspace_shapes comes from the outer hybrid object; selected class and
padding come from the actual child. Returned alignment_used, actual workspace_shapes, prepared input/scale layouts
and inclusive physical_scope must bind the feature schema/ranges. No fixed128
or universal Triton range applies to a native hybrid-selected DeepGEMM row.

## Finite model input and producer

`CollectiveSimCCBackend.predict_collective_plan(plan)` loads ONLY
`profiling_data_dir/<device>/collective_local.json`, schema
`step5.collective_local.v1`. It requires frozen scenario matching the configured
2-server/8-GPU TP8/DP2 attention layout, sum combine, native_measurement evidence,
independent PASS qualification with parent receipt, context/phase/graph/uniform
and tensor geometry match, and exact plan-key native count receipt. CPU evidence
is deliberately rejected by this production importer; CORE's CPU actual-main
fixture may construct an explicit cpu_fixture CollectivePlanCost, never relabel
it native_measurement. `tests/unit/test_collective_sim_cc_backend.py:setup_model`
is the precise fixture payload and negative-import example.

Exporter `export_finite_parameters(run_dir, output_dir)`
uses the frozen scenario and real HTSIM, fits independent per-rank copy/add then
nonnegative affine remainder from signed residuals, preserving rank/ranges,
held-out qualification and exact counts. It writes a candidate; no omitted
qualification becomes production evidence. Selected prepare/benchmark/
validator/fixture use the same forward_collectives native_call_names planner.

Validation: eight focused suites200 passed/1 skipped (torch unavailable in py310);
original native modular/finalize/E8M0 selector source methods also executed with
CPU tensors under asyncopd-py312. Real HTSIM completion was checked, not mocked.

## Third interface: physical queries, schemas and reporting

`MoeOperatorTimes._predict_step5_position_operator_times(lane_workload, *,
context, operator_features, collective_plans)` returns `(MoEOperatorTimes,
CommunicationOperatorTimes)`. Features cover exactly the selected registry.
Source scopes query actual ni, expert queries use gathered M and independent
total_routed_tokens; no live0 early return. It calls CORE's
`_predict_step5_ffn_operator(name, features, *, context)` and requires paired
positive integer kernel counts and exact_lookup. That keyword/context-aware
CORE implementation is NOT yet imported; peer b39fc51/e255ede now provides it,
but only02f061d is authorized/imported here. Do not treat H04 CPU fixtures as
verification of the real shared estimator. CORE must wire this H04 method into
the existing measured layer path and pass its return objects through the existing
ExecutionTime constructor/setters (component accessors return copies).

`step5_native_expert_features(context, *, physical_rows, live_assignments)`:
TritonExperts schema `(num_tokens,total_routed_tokens,num_tokens_post_padded)`;
DeepGemmExperts schema `(num_tokens,total_routed_tokens,native_capacity_rows,
native_alignment)`. Native capacity/alignment/assignment come from H17, not CPU
rounded live counts. Each row's actual native context/layout/padding and ranges
must be preserved by CORE-owned loaders/trainers. No universal backend bounds.

Frame keys: routing_source, ep_id, native_context, live_local_assignments,
operator_costs. The latter maps selected physical names to the paired CORE
proofs, including evidence_kind/kernel_count/exact_lookup. Comm objects carry
the complete plans/costs; finite duration is a full-device cost and does NOT add
another generic collective launch gap. Actual events/kernel counts and modeled
stage count remain separate trace/ledger fields. record_ep_wave was executed
with operation metrics, tracing and ledger enabled together; gathered17/source16,
live assignment0, 3ms joint dispatch and paired counts are preserved.

Finite template: tests/integration/step5_comm/step5_position_model_run.json.
Six eager calibration vectors, six independent qualification vectors, thirty
local cells, five selected labels. Graph is not selected without actual bucket
eligibility. Authentic manifest/input IDs/frozen configuration/auth inputs remain
required; prepare fails before staging when they are absent.

Validation: eleven affected suites335 PASS/71 SKIP (torch/native-runtime unavailable
in py310); isolated collective-sim7 PASS and actual HTSIM completion; native
original modular/finalizer/hybrid selector/GroupCoordinator methods executed with
CPU tensors under py312 AST isolation. No GPU leaf durations or qualification.

Latest validation:350 PASS/72 SKIP across eleven owned affected suites in24.39s.
Published source at `/data/ycfeng/tmp/d29_published_sources01/source/vllm` was
executed under py312 CPU tensors: original prepare triple-return/deferred branch,
modular inclusive invocation/finalization, E8M0 OR shape and coordinator ordering
all PASS. AST isolation avoids the unrelated missing py312 fasteners dependency;
this is method behavior evidence, not complete installed runtime qualification.

## T2 existing constructor seam

`frontier.profiling.collectives.step5_dispatch_probe.create_native_config(args,
*, engine_limits=None)` forwards only supplied `max_model_len`,
`max_num_batched_tokens`, `max_num_seqs` into native EngineArgs before
create_engine_config. Native should pass the mandatory cases.json mapping with
7168/8192/128 to this helper and its direct graph EngineArgs. No defaults are
injected when limits are omitted. Text-submodel with_hf_config and quant config
remain unchanged; HF max_position_embeddings remains1048576. There is no backend
override. Root decision D29-T2-ENGINE-LIMITS; delta48 gross in the existing two paths.

Focused CPU checks assert final post-with_hf_config model/scheduler limits,
unmodified HF metadata, excluded unrelated key, and exact legacy kwargs.
Three affected suites187 PASS/71 SKIP; parent must validate actual image config.
This helper delivery does not qualify T2 graphs or their case evidence.

## Final selected producer and qualification selection

prepare, benchmark, validator and fixture all use the selected
`forward_collectives(..., native_call_names=...)` owner. Valid complete subsets
pass; missing, duplicated, extra measurement/kernel/summary/probe keys fail even
when superficial counts are reduced. Selected worker persists job.json, forwards
one sealed profile manifest and effective environment, and supplies exact
moe_input_ids/layers/source inputs to the probe. CORE's source_frame_tensors is
called once before native timing; native exceptions are not converted to soft
success. Native source verification errors are advisory and mark artifacts
UNVERIFIED; collector gathers errors from all16 ranks. Auxiliary validation never
controls healthy workload execution. Legacy unbound plan metadata is unchanged.

Local copy/add observations require every rank, every repeat, correct source
layout and held queue. Native count receipts require actual per-rank events and
profiled_calls; graph launches and device events remain separate. The original
into-output native coordinator takes `(input_,output,dim,sizes)`; collector retains
supplied storage/registration/alias/lifetime/branch receipts and zero staging copy.

`export_finite_parameters(directory,output)` validates complete selected producer
evidence for native_measurement inputs before fitting. Raw CPU fixtures can be
exported but remain candidate_only/cpu_fixture. Cell identity, not equal aggregate
bytes, controls pass aggregation. Signed residuals and raw passes are retained.

`qualify_finite_parameters(candidate_path,receipt_path,output_path)` uses a
separate `step5.collective_qualification.v1` parent receipt with candidate_sha256,
status=PASS, independent=true, observations and exact one-row-per-model
model_checks(model_index,status,independent,observations). Every selected affine
fit has held-out observations and rank2. It changes qualification flags only,
preserving coefficients/ranges/raw/signed evidence. CLI:
`export_costs.py CANDIDATE_DIR --finite-model --qualification-receipt RECEIPT
--output-dir OUTPUT`. No CPU test receipt is runtime qualification. Production
import rejects candidate_only, CPU evidence and incompatible frozen scenario,
model/context/count/range; it never uses native full-plan duration lookup.

### Precise CORE actual-main CPU payload

Reuse `CollectivePlan`, `CollectivePlanCost` and existing ExecutionTime setters,
not a second pricing backend. CPU fixture cost fields are explicit:
`duration_ms=network_ms+max(local_rank_ms)+remainder_ms`,16 local_rank_ms entries,
16 positive kernel_counts, exact plan.cache_key(), model_identity and receipt
`{evidence_kind:cpu_fixture,kernel_counts,per_rank_events,source_receipt}`. Test
`test_step5_real_wave_metrics_trace_and_ledger_share_physical_proofs` in
tests/unit/test_ep_wave_trace_context.py is the actual metrics/trace/ledger
consumer example; test_step5_moe_pricing.py supplies H04 physical vectors/features.
The existing fixture may create explicit CPU costs, never relabel them native.

For real finite parameters, frozen_model_config contains device_type,
collective_sim_root and scenario. Scenario fixes cluster2x8, attention TP8/CP1/DP2/
EP1, placement TP,DP, kind=plan,tensor_bytes=0,no participant override,sum rule,
use_triggers=true and supported ring_steps/nccl_ring selectors/channels; network
and local inputs are frozen before fitting. Independent copy/add fits have
a_ms,b_ms_per_byte,range_bytes,design_matrix_rank2,distinct_sizes>=3,
signed_residuals_ms,calibration_receipt,qualification. Models retain actual
context/phase/graph/uniform/tensors; exact per-plan counts remain separate.
DP equations are the frozen N_joint+max(local rank)+effective remainder; TP uses
the separately fitted effective NVLink fields and observed counts. Existing
tests/unit/test_step5_comm_cost_export.py executes real HTSIM with explicit CPU
fixture records; tests/unit/test_collective_sim_cc_backend.py gives schema and
strict import negatives, not a production parameter bundle.

Remaining CORE-owned seams: import tested pricing commits via root, canonical
native_context JSON CSV equality/schema/range classification, wrapper source
prepare/gathered inclusive calls, shared norm/expert source-width exact pairs,
actual layer/main H04 invocation and H14/H15 fresh CPU fixture/reporting. The
context-aware exact lookup exists in peer b39fc51/e255ede but is not imported or
verified together with H04 here. Root owns final gitlink1a16426, integrated
full-unit/strict77/review, actual source/backend/T2/G3 and fresh native/E2E runs.

Final observed checks:399 PASS/89 SKIP across13 affected suites; isolated
collective-sim7 PASS with actual HTSIM completed flows. Published original native
prepare/modular/finalizer/selector/coordinator/capacity methods executed with CPU
tensors under Python3.12.13. Native capacity M17/K8/local22/alignment128 gives2944;
M1 gives1024, independently from live-route rounding. These are source-method
facts, not actual device-selected alignment/GPU durations or qualification.

Final delivery rerun:438 PASS/89 SKIP across14 affected suites, including legacy
ExecutionTime operation tests. Added the registry's missing source-prepare
ExecutionTime accessor; legacy MoE trace now retains zero without AttributeError.
Submodule7 PASS; published original source-method checks PASS. Exact commands,
limits and43-path/2626 gross budget are in test_report_2026-10-23_delivery.md.
