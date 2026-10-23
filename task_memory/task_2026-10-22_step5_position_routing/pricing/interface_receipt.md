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
| 2026-10-23 | Repair F3-F6 completed HTSIM, actual policy, exact replay and cold bootstrap evidence. |
| 2026-10-23 | Separate frozen fit context from complete per-cell policy and startup diagnostics. |
| 2026-10-23 | Require observed route values and exact per-cell compute/plan binding. |
| 2026-10-23 | Emit observed native compute points and associate exact buffer/count receipts. |
| 2026-10-23 | Deliver explicit unadmitted CPU plan consumer and numerical trace source. |

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
Final producer/qualified-import/legacy-trace code commit:940cc84. Complete code
sequence:7b86614,409093c,511a3db,f03286f,06bae09,3f54bc6,940cc84.
Isolated submodule commits:ae75af0,1a16426. Root integrates gitlink separately.

## Finite review repair F3-F6

F3/F6 early parent commit37ead7f; isolated submodule3dac53c supersedes1a16426.
F4/F5 tested parent commit181a534. No uncommitted code remains; root-owned
gitlink is intentionally unstaged. Summary/diff/AST/JSON/patch equality PASS.
`collective_sim_core.predictor.completed_plan_network_ms(payload)` requires
expected_flows>0, finished_flows==expected_flows and finite positive completed
makespan_us for new nonempty joint plans. Both prediction and exporter reuse it;
legacy scalar and valid TP-local no-network branches do not enter this gate.

F4 context now includes `execution_policy`, keyed by actual participating rank.
Each policy preserves actual runtime (image-resolved vllm defaults/overrides,
enabled communicator objects, PyNccl version, parallel flags, execution-affecting
NCCL environment), observed native branch sequence and input/output registration,
supplied output and alias receipts. NCCL_DEBUG/DEBUG_SUBSYS/DEBUG_FILE are retained
as diagnostics but excluded from model identity. Per-rank startup runtime is retained in
plan.json. Actual GroupCoordinator calls use the same untimed observation seam in
collector and eager module probe; no template routing flag infers a branch.
Eager timing issues the original operation only; captured branch observation is
inside capture, never graph replay. Auxiliary observation failures mark evidence
UNVERIFIED and retain the original native exception. No additional GPU probes run.
Validator requires collector/probe scope, tensor layout and execution policy
equality for each recorded rank; no missing policy can qualify. Export/model/
count keys retain this complete context, parent candidate_sha256 seals it and
production importer rejects missing/different policy. CORE must propagate the
retained measured policy into plan.context_json; no guessed defaults are supplied.

F5 records must exactly match selected call op/group/cuda_graph; native counts
require profiled_calls==configured calls_per_repeat, not just consistent division
of a relabeled event list. Legacy no-profile native-count semantics unchanged.
F6 tests use actual constructor bootstrap; no mock disables the loader, and cold
backend9 PASS with PYTHONPATH unset establishes the normal initialization path.

Final affected14 suites:456 PASS/89 SKIP (JUnit545/0 failure/0 error); isolated
submodule12 PASS with actual HTSIM; cold bootstrap9 PASS. Owned43 paths,
+2849/-95=2944/3956 (remaining1012), including333 submodule gross. Full pinned-to-tip
patch retained with sha256c96cace541b72f7627b5f429d55615a17be4cd6c86d1827ef53969d05d766a1d.
Exact commands/RED→GREEN/limits in test_report_2026-10-23_finite_repair.md.
Real GPU policy/parameters/source/backend/graph qualification remains unverified.

## F4a/F4b corrected identity boundary

Independent re-review closes F3/F5/F6; this section supersedes earlier F4 model
grouping and startup equality statements. No CORE plan payload migration:
`plan.context_json.execution_policy` keeps each participating rank's full actual
`runtime`, ordered `branches`, `buffers` and diagnostic errors. Full context and
buffer_receipt still determine plan.cache_key(), paired count receipt and cached
prediction; native timings are observations, never runtime duration lookup.

`frontier.cc_backend.collective_plan.collective_runtime_identity(runtime)` removes
only `enabled_communicators`, whose current enabled/disabled state native lazy
initialization can legally change. Actual image defaults/overrides, PyNccl versions,
parallel flags and execution-affecting NCCL settings remain the frozen policy.
`collective_model_context(context)` preserves other context fields and replaces
execution_policy with `runtime_policy={rank:collective_runtime_identity(runtime)}`.
Exporter model/context/parameters use that frozen identity across size-dependent
implementations. Every raw observation/pass and count retains full branch/buffer
evidence. Independent parent qualification still seals the whole candidate and
covers every model; its criteria/algorithm are unchanged.

Importer derives the same frozen model context from the supplied complete plan,
then separately requires the exact complete count key/context. It binds those
qualified parameters to the full context for the existing simulator invocation;
it does not modify the stored model, synthesize counts or ignore a branch switch.

Collector plan.json uses `runtime.per_rank_startup_runtime` explicitly as startup
diagnostics. Measurement policy samples runtime after the actual observed native
invocation. Selected eager module probe retains startup_runtime_policy diagnostics
and supplies DispatchObserver.runtime_policy as a callable, sampled after each
native operation. Validator compares only frozen startup/call fields, then exact
complete collector/native-probe call policy. Successful FlashInfer lazy-disable
and PyNccl fallback pass; stale fallback state/branch/buffer, changed image setting
or PyNccl version and missing policy remain UNVERIFIED. No retry, forced backend,
native exception suppression or timing observation overhead is added.

Targeted evidence: actual exporter single frozen TP8 BF16 policy, row anchors
16/256/8192 with custom/PyNccl switch and held-out64; one fit retains three anchors,
all raw/count branches. Published FlashInfer _ensure_workspace/should_use_fi_ar
and CUDA communicator all_reduce executed through actual DispatchObserver,
successful fallback state compared by real validator. CPU leaves/labels are not
GPU runtime qualification. Cold importer10 PASS also verifies separate count/cache
keys across a branch/buffer change.

Final14 affected suites552 JUnit:463 PASS/89 SKIP/0 FAIL/ERROR. Isolated actual
HTSIM12 PASS; independent cold importer10 PASS without PYTHONPATH. Owned43 paths,
+2972/-95=3067/3956 gross (remaining889), including333 unchanged submodule gross.
This repair touches10 existing parent source/test paths,+152/-29=181 versus27b3499;
original-baseline gross increases123. Submodule remains clean at3dac53c with no
new sub commit; full patch/sha256 unchanged. Exact commands and RED/GREEN in
test_report_2026-10-23_f4_boundaries.md. Verified scoped source/record commit506b43ac;
root re-review/global qualification remains open. CORE should retain the complete
plan policy contract above and derive no branch/defaults itself.

## C1/C2 compute-review interface

No change to frozen fit grouping or full execution_policy. For H04 the complete
point context returned for the selected native experts must also be retained as
`json.loads(plan.buffer_json)["compute_context"]`. H04 compares canonical JSON
with the current `context` (all fields, including backend/quant/deferral/source and
gathered layout/padding/workspace/source callable facts/collective_tensors), not
with the unrelated engine/environment/communicator envelope. Plan context.shared
must still equal context.native_facts.shared. Full buffer receipt remains part of
plan.cache_key, count proof and candidate seal; no full-point compute_context in
frozen model context and no new model grouping dimension or per-size fit.

H04 verifies exact ordered CollectiveTensor operands against
context.collective_tensors[dispatch/output], source/gathered physical width,
activation/scale dtype/element-size/contiguous stride, reached/deferred FP8 versus
BF16 activation, actual configured router_topk FP32 weights+INT32 IDs and BF16
output width. All plans validate before any backend invocation. Phase/size and
paired positive exact scope evidence remain required, including live0 lanes.

Producer closure: `DispatchObserver.compute_context(layer,text)` retains actual
router outputs, completed all_gatherv source/gathered operands and native expert
apply arguments. It requires the same tensor references and reuses
NativeRoutedExperts.prepared_context for selector, quant/layout, physical padding,
workspace and callable source verification, after native execution. Point output
geometry comes from the actual into-output combine, not a fabricated activation
dtype. Errors remain in diagnostic_scope and never replace native exceptions.

Rank layer report now emits `compute_context`. Validator requires observed shared
identity and local/gathered physical widths. `observed_collective_points(record,
probe,inputs)` associates exact physical/live vector, source frame, layer/input/rank,
complete call-time policy/group/layout and completed call. Ordered native TP
reductions retain shared-first/routed-second attribution from published Step5MoE
forward, not implementation-name inference. Returned native_point retains input/
output schemas, complete execution_policy and call_index. No new point campaign.

Exporter reads the existing native_dispatch report and routing inputs after its
actual selected validator passes. Each applicable eager count key contains
`buffer_receipt.compute_context` and `buffer_receipt.native_points`; both also live
in the exact plan.buffer_json and count.buffer_receipt. Raw frozen-model fit cells
remain unchanged. Same compute contexts merge all observed rank/layer associations;
distinct actual point contexts get distinct count keys, not new affine fits.
Calibration-only or captured cells without a corresponding eager point retain
unbound counts and cannot satisfy H04; no guessed GPU fields or graph qualification.

CORE's current `_step5_position_queries` in its development worktree already reads
buffer_json.compute_context, checks count.buffer_receipt equality and requires a
unique exact match (moe_routing_workload.py165-170 at inspection). No new shared API
is needed: carry this measured field unchanged, including source callable facts,
never synthesize it from the candidate prediction. Root owns integration and
actual-runtime collection; no peer writes/imports performed here.
Source/tests/records commit9dbca5b3 is ready for root-authorized CORE integration;
430 distinct affected cases350 PASS/80 SKIP, published native64 PASS,3606/3956.

Bound eager module observer gains `routing(router)` with the original named
signature hidden_states,router_logits,topk_indices_dtype=None,*,input_ids=None;
original selection called once, tensor/tuple references and exceptions unchanged.
The one source_forward_context materialization now yields (positions,is_padding)
while maintaining the existing native contexts/cleanup. No second source algorithm.
After native forward+synchronization, `compare_routing(source_tensors,hf_text_config)`
compares E/K/normalize/scale/EPLB/method/effective strategy+padding policy, selected
strategy source, actual positions/mask and exact IDs/weights from the same logits.
Checks use torch.equal, no tolerance. Full original select/_select/_compute and
actual strategy route/simulate callables verified before patching. There are no
timing rows/hotpath kernel additions, retries or input mutation. Auxiliary failure
marks UNVERIFIED while original native exceptions are retained.

Rank layers add `routing_comparison={status,differences,observations}`. Observation
retains actual_config, strategy/scoring contract, logits schema, explicit positions,
mask, returned IDs/weights and physical/live row+live assignment totals. Generic
comparison cannot be match on routing discrepancy. Selected validator requires
one observed routing match with source physical/live/assignment totals; missing/
wrong routing cannot be admitted by otherwise matching collective schemas.
Legacy unbound probe/validator paths unchanged; this is eager module evidence,
not T2/full serving/backend/GPU qualification.

Final affected682 JUnit=529 PASS/153 SKIP/0 FAIL/ERROR; actual published native
observer64 PASS; cold backend/importer10 PASS without PYTHONPATH. H0466 cases
cover BF16/FP8 and source/live0 lanes, all negatives assert no compute/backend
query. Owned43 paths,+3210/-95=3305/3956 (remaining651), unchanged sub3dac53c/
333 gross included. Six existing owned source/test paths repaired,+260/-22=282
against bfe6f2ae; approved original-baseline gross increases238. Exact commands
and numerical limits in test_report_2026-10-23_compute_repair.md. Scoped commit
989a883d delivered; observed compute binding/CORE and global qualification remain open.

## D29-CPU-PLAN-CONSUMER: explicit actual-main composition

Existing manifest.collective_plan_input_file -> profiling_data_dir/<device>/
collective_local.json selects the existing CPU fixture; no CLI/backend/config or
CORE runner API change. predict_collective_plan retains TP8/DP2 placement checks,
then routes artifacts OR plan.shared marked cpu_fixture through a gate that requires
BOTH to be cpu_fixture. Schema must be step5.collective_local.v1, device_type must
match, native_admitted is explicitly False, candidate_only explicitly True.

Exact plan.cache_key selects counts[key]. Its plan_key/context/buffer_receipt must
match the plan, buffer evidence_kind must be cpu_fixture, entry and embedded
count_receipt_json must declare cpu_fixture, and the embedded receipt must equal
the count entry excluding cost. CollectivePlanCost validates finite no-overlap
composition/count proof and its plan_key must also match. Return this cost ONLY
for unadmitted CPU composition; native count-entry cost remains ignored while the
existing qualified fit/HTSIM path predicts. Dual-direction CPU/native labels,
missing identity/admission and stale receipt/key/layout data fail.

Trace numerical_source is cpu_fixture_composition_not_native for explicit CPU
cost receipts; native receipts retain joint_htsim+measured_local+
effective_remainder_no_overlap. Tests execute CORE's actual
position_collective_costs function from read-only verify-step5-core-delivery-01,
with CPU synthetic input table, and use the normal backend constructor in a cold
process. All dispatch/combine/shared_tp/routed_tp costs preserve writer identities.
No numerical production defaults, fit per plan or native qualification introduced.

Affected JUnit68 PASS/0 SKIP/FAIL/ERROR; four-path patch+109/-2=111<=120,
original owned total43 paths,+3617/-96=3713/3956, remaining243. Producer9dbca5b3
remains completed. Submodule unchanged3dac53c; root owns final gitlink and full
integration/qualification. Exact commands in test_report_2026-10-23_cpu_plan.md.
