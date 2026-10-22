# D29 Step source-frame and collective pricing design

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | D29-PRICING-CORRECTION: retain simulated CC; compare the lossless existing-simulator extension with unapproved measured lookup, correct routed targets/contracts and reporting/collector scopes, and separate budgeted interfaces from unresolved algorithm work. |
| 2026-10-22 | Parent distinguishes the proposed measured-cost variant from the existing collective_sim decision, records reusable flow primitives, and corrects the table sum to 1987 gross lines. |
| 2026-10-22 | Trace native calls to NCCL primitives; define separate source/expert frames, native compute scopes, exact collective-plan requirements, implementation ceilings and acceptance limits. Design only. |

TaskID: **D29-PRICING-CORRECTION**, continuing **D29-STEP-PRICING-DESIGN**.

## 1. Result, scope and ownership

The existing scalar communication API **cannot faithfully price the accepted
Step TP8/DP2/EP16 no-SP path**. Its allgather/reduce-scatter methods support
uniform single-tensor payloads, not unequal vectors, grouped heterogeneous
tensors or the jointly executing eight DP groups. Calling those methods several
times and summing their answers is not the native execution plan.

The minimum complete repair has three inseparable parts:

1. Source-local GateLinear/router, shared branch and prepare work, versus
   gathered expert work; preserve physical rows even when live routes are zero.
2. Native inclusive compute-scope ownership and phase ordering; do not double-count a
   finalized expert invocation and a separate shuffling invocation.
3. A lossless collective-plan interface through the **existing** communication
   registry and backend. No average bytes, summed scalar calls, renamed EP16
   all-to-all or isolated-rail maxima substitute for that interface.

**Parent disposition:** the formal Step task explicitly retains simulation-based
`collective_sim` communication rather than GLM/V4's measured CC approach
(`task_2026-10-15_step5_preview_h800/requirements.md:1162,1182,1232`). Reusing the
backend class name does not authorize changing its numerical source. The
measured-plan variant requires an explicit modeling decision in addition to a
file/line scope approval. **Recommend Variant A: extend the existing simulator.**
Its plan transport and known flow-lowering interfaces are implementable without
a new backend/controller or C++ network engine. A faithful full-duration
algorithm and its complete implementation ceiling are **blocked by the exact
missing modeling inputs in section 3.2**. Do not approve the known-interface
subtotal as a complete numerical implementation estimate. Variant B is retained
only as an honestly labeled, explicitly unapproved comparison; it is not the
default or a fallback under the same backend class name.

The previous 37-path table was **+1802/-185 =1987**, with section D **+585/-14
=599**. Parent's correction is retained. The revised tables in section 6 add the
reviewed interfaces and distinguish common work, A's known work, and B's
lookup-only work. No assertion that HTSIM expansion is necessarily larger follows
from either the old table or the new partial subtotal.

All three parts, core `design.md`, derived-source `profile_source_design.md`,
native design and the actual-main fixture are mandatory for parent acceptance.
There is no counts-only completion option. Numerical scope approval remains a
parent/user decision before implementation.

The author edits only this document. This correction uses source inspection and
lightweight `python3` budget arithmetic; `python` was absent on the host. No code,
tests, GPU, Docker, platform, authentication, agents or commits were executed.
Source observations below are not fresh worker measurements. Companion files
remain owned by their authors; this correction reads the task requirements,
plan and progress, not their implementation/design bodies.

## 2. Native call trace: what is actually executed

Native paths below are relative to
`/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/`.

| Native location | Source-grounded consequence |
| --- | --- |
| `models/step5/nvidia/model.py:755-762,848-851,1751-1807,1814-1825` | No SP unless mega MoE is selected. At DP2, shared and routed branches reduce separately. Shared branch and FP32 GateLinear run on each source's physical hidden-state rows before routing/dispatch. |
| `model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:17-53,136-208` | Prepare quantizes locally unless the selected expert kernel defers quantization. Dispatch list order is activation, FP32 weights, signed IDs, then non-scalar activation scales. Static scalar scales are replicated, not gathered. No LoRA in the accepted path. |
| `distributed/device_communicators/all2all.py:76-97,128-161` | No-SP DP>1 selects `get_dp_group()`. Sizes are the explicit DP metadata vector; dispatch calls `all_gatherv(tensors, dim=0, sizes=sizes)`. EP16 is expert ownership, not this collective group. |
| `forward_context.py:129-145` | `sp_local_sizes` without SP retains the actual DP physical sizes. It does not calculate an average or substitute live routes. |
| `distributed/device_communicators/cuda_communicator.py:847-915` | Equal sizes become `sizes=None`; a tensor list is enclosed in an outer NCCL group. Unequal inputs each call PyNccl `all_gatherv`; equal inputs each call PyNccl `all_gather`. Uniform symmetric-memory selection is a real native branch, not inferred from an operation label. |
| `distributed/device_communicators/pynccl.py:324-357` | Unequal all-gatherv issues one `ncclBroadcast` per root per tensor, in size-vector order, inside a nested NCCL group. Broadcast element count is the destination slice's `numel`, not a common count. |
| `prepare_finalize/naive_dp_ep.py:210-245` | Native expert output is weighted/reduced into the physical gathered output frame before combine. `TopKWeightAndReduceNoOP` uses the already finalized expert output. |
| `all2all.py:193-209`; `cuda_communicator.py:526-621` | Combine uses the same DP size vector through `reduce_scatterv_into_output`. Unequal sizes cannot use symmetric memory. At DP2 the batch-invariant world-size>2 alternate is unreachable. |
| `pynccl.py:387-422` | Unequal combine issues one `ncclReduce(SUM)` per root, over that root's physical slice. Equal sizes reach `ncclReduceScatter`, or the actual uniform symmetric-memory branch. |
| `prepare_finalize/naive_dp_ep.py:109-110`; `runner/moe_runner.py:484-517` | Naive finalize reports output not reduced; after DP combine the routed output receives a source-local TP all-reduce. It is not an EP16 all-reduce or a pre-combine reduction. |

For rank `r = dp_index * TP + tp_index`, the active groups are:

- source owners: DP lane0 and lane1, once each for counts;
- expert owner: `ep_id=r`, 22 contiguous experts at E352/EP16;
- DP collectives: `(t, 8+t)` for each `t=0..7`, eight concurrent two-rank groups;
- TP reductions: `(0..7)` and `(8..15)`, each on its source-local output rows.

Use existing rank placement/parallelism metadata and existing
`group_members`/backend rank linearization, not a second placement scheme.
Source intervals are not replicated eight times. TP copies participate in
communication and compute, but do not multiply unique routing assignments.

### 2.1 Payloads and actual launch plans

Let source physical rows be `n=(n0,n1)`, gathered rows `M=n0+n1`, hidden width
`H`, top-k `K`, activation quantization group `G`. For the current Step config
H4096/K8/G128, the native FP8 expectation is:

| Tensor | Local shape on source i | dtype | Row bytes |
| --- | --- | --- | ---: |
| Activation | `[ni,H]` | `float8_e4m3fn` | H =4096 |
| Top-k weights | `[ni,K]` | `float32` | 4K =32 |
| Top-k IDs | `[ni,K]` | `int32` | 4K =32 |
| Activation scale | `[ni,H/G]` | `float32` | 4H/G =128 |

FP8 dispatch endpoint row bytes are4288. BF16 dispatch has the first three
tensors, activation `bfloat16`, and row bytes8256. Combine is one BF16
`[M,H]` input and `[ni,H]` output, row bytes8192. Shared and routed TP reductions
also operate on BF16 `[ni,H]` in this no-SP branch: `reduce_in_fp32=True` in
`Step5MLP` only casts the shard-SP output (`model.py:835-845`). Do not apply a
generic expert-communication quantization factor to weights, IDs, output or
scales. These constants are examples from inspected config; implementations
derive them from the selected model/quantization/native receipt.

FP8 DEEPGEMM + FP32 `[rows,32]` scales and BF16 TRITON without scales were
previously observed in the retained native module probes:
`../task_2026-10-15_step5_preview_h800/test_report_2026-10-20_step5_comm_preparation.md:182`.
Those probes are not derived-policy TP8/DP2 full-model evidence. Fresh selected
source receipts must confirm the tensor list, element sizes/strides, selected
expert implementation, `defer_input_quant`, scale layout and collective branch.
UE8M0 packing in `deep_gemm_utils.py:505-519` is a later expert-scatter layout;
it must not replace the dispatched FP32 scales with an invented byte count.

Native unequal FP8 dispatch has **8 broadcast primitive invocations** per DP
group (4 tensors x2 roots), BF16 has6; combine has2 reductions. All tensor/root
invocations are grouped. These are primitive counts, **not measured GPU kernel
launch counts**: NCCL may group/fuse kernels. Uniform dispatch has4 or3
`ncclAllGather` invocations in one outer group; uniform combine has one
reduce-scatter invocation. Actual kernel counts must come from paired evidence.

| Required case | Physical vector | Live vector | Dispatch inputs/outputs per rank | Combine inputs/outputs per rank |
| --- | --- | --- | --- | --- |
| P6128 plus idle lane | `(6128,1)` | `(6128,0)` | `ni * dispatch_row_bytes` / `6129 * dispatch_row_bytes` | `6129 *8192` / `ni *8192` |
| D transferred tail16 plus idle lane | `(16,1)` | `(16,0)` | `ni * dispatch_row_bytes` / `17 * dispatch_row_bytes` | `17 *8192` / `ni *8192` |
| q1 full graph with one idle lane | `(b,b)` from actual capture agreement | `(1,0)` | `b * dispatch_row_bytes` / `2b * dispatch_row_bytes` | `2b *8192` / `b *8192` |

Do not guess b: read the admitted `ForwardStartDescriptor` for both sources.
At b1 the uniform vector is `(1,1)`, still two physical rows and one live route.
Do not force the unequal P/tail cases into the uniform graph path. Retain all
padding/dummy tensors and participants; masked IDs/weights are `-1`/0, not absent
physical communication buffers. Endpoint bytes above are not NCCL wire bytes.

## 3. Existing Frontier capabilities and the exact missing capability

| Inspected code | Supported now | Missing for this native path |
| --- | --- | --- |
| `frontier/operators/spec.py:153-217` | Scalar payload builder, group/domain/participant-count builders, one collective alias, one fixed kernel_count | Sizes vector; ordered tensor descriptors; explicit multiple groups; grouped primitive plan; observed variable kernel-count proof |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py:6254-6363` | Registry-driven scalar allreduce/allgather/RS/alltoall/p2p | One lossless plan dispatch; returns only scalar duration through these methods |
| `frontier/cc_backend/base_cc_backend.py:108-151,184-339` | Scalar bytes/count/domain methods, including abstract broadcast | No reduce primitive; no roots; no uneven vector or grouped-call contract; no joint participant groups |
| `frontier/cc_backend/backends/collective_sim_cc_backend.py:341-348,447-512,627-683,789-810` | ATTN_DP chooses attention layout; backend rank builders preserve DP stride; scalar p2p exists | Default DP rank builder selects TP index0 only. `predict_broadcast` explicitly raises. Cache identity has one byte size/group, no vector/tensor/grouped-plan dimensions. |
| `frontier/cc_backend/backends/collective-sim/python/collective_sim_core/schema.py:55-87,273-291` | One scalar tensor_bytes, one participant list, one p2p direction | Unequal opposite-direction bytes; ordered heterogeneous tensor list; multiple concurrent explicit groups; grouped NCCL reductions/launch overhead |
| `frontier/cc_backend/backends/collective-sim/htsim_runner.py:565-574,913-940` | Domain enumeration can create all DP groups if no explicit override; p2p supports directions and equal-size bidirectional flows | Frontier always supplies one explicit group, overriding enumeration. Even direct p2p bidir uses the same byte size both ways and starts both at0; it cannot reproduce unequal grouped broadcast/reduce plans. |
| `tests/integration/step5_comm/export_costs.py:87-105,108-150,243-255` | Already preserves payload vectors, rails, tensor lists, graph/backend/kernel context | Existing exports explicitly remain candidate_only, production_consumable=false. Uneven/idle vectors are already marked scalar_api_representable=false. No admitted consumer. |

Therefore changing EP16 to ATTN_DP/DP2 is necessary but not sufficient. Summing
per-root scalar p2p predictions, using average ni, pricing a packed tensor of
4288 bytes/row, multiplying one rail by8, or taking8 isolated scalar maxima
would introduce unsupported assumptions about NCCL grouping and contention.
None is proposed. DSV4's SP/EP allgather identities are not reused for Step.

### 3.1 Common lossless interface, independent of numerical variant

Extend `CommOperatorSpec` with one optional `plan_builder`; scalar operators and
their existing behavior remain unchanged. A plan owns ordered tensor shapes,
dtypes/element sizes, sizes in **DP rank order**, root primitive order, grouping,
physical participants/placement, concurrent groups, graph/runtime/native source
identity, input/output alias/registration scope and native included duration
scope. It is an immutable value used by
the existing registry; no second operator registry or routing classifier.
Validation admits exactly one scalar or plan-backed payload contract; a
plan-only operator does not declare a dummy scalar payload/collective alias.
Retain all scalar validation for unchanged registrations.

Add `BaseCCBackend.predict_collective_plan(plan)` as a nonabstract method that
raises a capability error by default. Implement it only in the existing
`CollectiveSimCCBackend` for the selected solution. The other backends fail
new-mode admission before the run if they lack this capability; old policies
remain unchanged. The one plan call returns included duration, an explicitly
labeled numerical source/breakdown, actual per-rank kernel-count proof and plan
identity. Kernel counts are measured native evidence in either variant, not a
function of primitive invocations. Keep the complete ordered plan in cache keys;
equal aggregate bytes are not equal workloads. Use one joint duration per phase,
not one price multiplied by TP8. An unavailable algorithm/receipt is an explicit
prerequisite failure, not a scalar fallback.

### 3.2 Variant A: smallest faithful existing-simulator extension (recommended)

**What can already be reused.** `htsim_runner.py:319-379` has independent bytes
per `FlowDef`, starts and completion/dependency triggers, and one traffic matrix.
`:554-577` provides the placement map and all domain groups before an optional
singleton override. For ATTN_DP on TP8/DP2/EP1 it can enumerate all eight native
groups; EP16 is not a multiplicative dimension in that attention layout. The
existing network simulation can therefore run unequal directed transfers from
all groups together, with contention on the configured shared NIC/switch fabric.
No new generic simulator, controller, topology or placement registry is needed.
The legacy scalar DP method's singleton override must not enter the new plan.

**Known implementation seam.** Add serialized `plan` to the existing
`CollectiveConfig`/`Scenario.to_runner_spec`, and its strict plan branch to
`htsim_runner.CollectiveSpec`/spec merge/validation/`generate_tm_from_spec`.
Use the one common immutable plan in `frontier/cc_backend/collective_plan.py`;
the submodule receives its lossless serialized value, not another native-op
registry. All explicit groups must agree with the existing placement; reject
missing/duplicate ranks, wrong order, inconsistent tensor shapes/bytes, or a
scalar payload plus plan. Preserve existing scalar scenario generation verbatim.
`CollectiveSimCCBackend.predict_collective_plan` builds the scenario from existing
attention layout, invokes the existing Python predictor once and caches its full
plan/model/config identity. Its existing scalar `_predict` is not a loop used to
price the plan.

**DP2 logical transfer obligations, not a complete NCCL schedule.** For tensor j
of row bytes bj, unequal dispatch root i must deliver `ni*bj` bytes to the other
rank and retain its own root slice in the output. Unequal combine slice i must
deliver `ni*8192` bytes from the nonroot to owner i and reduce both contributions
there. The orders and nesting come directly from PyNccl. Equal vectors retain
the actual grouped AllGather/ReduceScatter calls. Dtypes distinguish activation,
weights, IDs and scales and determine element counts. `FlowDef.size` can express
these exact byte obligations without a DP average. This establishes the input
contract only: it does **not** establish one flow per call, wire bytes, chunk
size, all-start-at-zero concurrency, GPU reduction cost or native duration.

**Lowering once native schedule evidence exists.** For a qualified NCCL
algorithm/protocol/channel work schedule, emit each network chunk with its own
bytes and endpoints into the existing `flows` list; use existing `oneshot` and
`barrier` triggers for evidenced communication dependencies. Preserve all groups
in one matrix. Do not add a global barrier between tensors/roots/rails just
because Python calls are ordered, and do not start them all at zero because
`ncclGroupStart` groups them. Concurrent groups are a native participation fact;
their launch offsets/channel overlap still require evidence. Existing ring
lowering may be reused for a reached **uniform** ring only when its channels,
chunk semantics and ordering match; generic ring defaults are not that proof.
No ring/tree choice is inferred merely from the name `ncclBroadcast`/`ncclReduce`.

**Full device duration is not currently modeled by that matrix.** The existing
`collective_sim_core/predictor.py:24-38,94-118` runs HTSIM, estimates intra-server
time, then selects `max` or `sum`; `phase` currently also means sum, not an event
DAG. `intra_server_model.py:79-84` returns zero at one GPU per server per DP group.
That is correct for *intra-group NVLink traffic* in this layout, but says nothing
about root GPU copy/reduction and NCCL protocol/launch work. Its scalar
allreduce/alltoall launch constants cannot be repurposed as grouped broadcast or
reduce overhead. Do not silently charge these local costs as zero.

The complete A algorithm is blocked by the following exact inputs/capabilities:

| Missing input/capability | Existing mechanism and next decisive evidence |
| --- | --- |
| Loaded NCCL algorithm/protocol, channel allocation, tensor/root fusion, chunk sizes and stream work ordering for unequal FP8/BF16 and reached uniform graphs | Existing native collector, loaded communicator/environment and per-call profiler receipts can bind the native entry and observed kernels. They currently do not expose a lossless per-channel schedule. Obtain it from matching loaded NCCL source/debug/channel/graph evidence, not another vLLM version or hard-coded default. |
| Actual physical rank-to-host/NIC/transport placement for all eight rails | Existing backend layout and HTSIM network/topology config support it; collector `hosts`, group ranks and NCCL transport receipts must match the active native run. Historical Socket evidence is not an RDMA assumption. |
| Local root output copy, reduce accumulation, protocol work, grouped GPU launch overhead and overlap with network | Existing CUDA-event queued device timing and profiler kernels supply total device evidence, but not a separable local cost model. A measured local component requires a defined native-equivalent scope and verified non-overlap/overlap; neither residual subtraction nor arbitrary GPU bandwidth factors supply it. DP2 removes multi-hop reduce topology, not the local reduction. |
| Dependency-relative device delay/resource scheduling, if local work delays subsequent network chunks | `sim/trigger.cpp:22-72` activates targets immediately when a trigger fires; `FlowDef` has absolute start or trigger, not a duration/resource-consuming local task. Existing triggers suffice for a communication-only dependency DAG. If the reached algorithm needs a timed local dependency, the precise missing capability is delayed completion after that local work, potentially GPU/channel serialization. It cannot be honestly replaced by an absolute start or a final sum. |
| Full-duration combine rule and actual kernel-count normalization | Existing predictor breakdown and collector repeat/graph timing are reusable. An evidenced serial/overlap rule could use existing sum/max; an interleaved local/network DAG cannot. Count every rank's actual paired scope, not distinct kernel names or root count. |

Known interface ceilings are in section 6.A. **They are not a complete
implementable full-duration estimate.** Any timed-trigger/GPU resource extension
would need concrete native evidence before identifying necessary C++ hunks and
their ceilings. Potential seams are the existing submodule `sim/trigger.h`,
`sim/trigger.cpp` and `sim/datacenter/connection_matrix.{h,cpp}`; none is selected
or budgeted as a hypothetical addition. Stop the estimate here rather than
designing a generic GPU scheduler or a full-policy algorithm campaign. A native
schedule and local-duration decision can settle whether C++ changes are needed
at all. No statement about A being necessarily larger than B is supported.

### 3.3 Variant B: exact measured-plan lookup (explicitly unapproved)

B shares section 3.1 but changes the numerical source. After explicit approval,
the selected backend would strictly match a qualified artifact configured by
`CollectiveSimCCBackendConfig.collective_plan_input_file`, and return one jointly
measured full-plan device duration. It would not run HTSIM for those costs.
Implement only the selected variant; no automatic runtime fallback or two
permanent numerical routes are proposed. Under the prior decision B is not
eligible just because the class is still `CollectiveSimCCBackend`.

Reuse `step5_comm/export_costs.py`, the collector and existing admission receipts;
reject `candidate_only`/`production_consumable=false`. Match the entire vector,
tensor order/dtype/scale/stride, groups, graph mode, buffer alias/registration,
source/image, transport and duration/count scope. No interpolation, aggregate
byte key, isolated rail row or old policy alias. Synthetic rows remain CPU-only.
The collector correction in section 5.3 is mandatory before new combine rows
could qualify: old return-output timing is not proof of native into-output scope.
B avoids a new NCCL schedule/local-cost predictor, but still requires fresh
complete native plan timing/count/quality evidence and an explicit changed-CC
decision. Its full lookup implementation ceiling is separately shown in 6.B;
neither availability nor 10% E2E accuracy is established by inspection.

## 4. Required source-frame interface and pricing ownership

Consume the core-owned `WaveRoutingSource` / `SourceLaneRouting` in `design.md`
§4.1, preserved on parent and canonical EP lanes. Do not recreate source
intervals from a virtual expert Batch. Minimum frozen fields already proposed:
policy, lane/batch/global/group IDs, graph mode, physical/live rows, intervals,
gathered physical rows. Add the source's **gating runtime context** to that same
descriptor, resolved by the existing source-batch predicate before freezing;
an idle dummy source in a prefill wave is not a hot-prefill router frame.
No new position counter, count algorithm, cache or mask convention is owned here.

For each EP rank use its admitted source lane `dp_index=ep_id//attention_tp`.
The native accepted topology establishes this association; validate world and
contiguous placement rather than applying it to SP/PD-AF/MTP.

In the existing EP lane Batch builder, set new-mode pre-routing compute width
to the selected source physical rows, not the aggregate width. Preserve the
existing aggregate/gathered width in `routing_source` and use it explicitly for
expert features. This makes shared projections/norm/gate/TP output source-local
without constructing an averaged batch or adding a second scheduler. Retain
source graph/runtime context independently from fake expert Requests.

| Consumer | Required input frame/count |
| --- | --- |
| GateLinear and native tensor router | Source physical ni; router also live rows and real source positions/mask; source-specific hot-prefill context |
| Shared branch/norm and shared TP reduction | Source physical ni; native TP8 and actual dtype |
| Naive prepare activation quantization | Source physical ni and selected native quant configuration; executes before gather |
| Native expert permutation/alignment, expert kernels, output unpermutation/weight-reduction | Gathered M, actual global IDs/weights/mask and local expert map; live local counts, native-specific padding/layout |
| DP dispatch/combine | Exact n vector, tensor descriptors and eight concurrent DP groups; never live assignment counts |
| Routed TP reduction | Source physical ni after combine; TP8, BF16, two actual concurrent TP groups |

`moe_operator_times.py` currently derives gate widths from the virtual batch
(:405,:430) and EP payload from live local assignments (:244-272,540-555 in
`families.py`). Both paths need the new frame. Its `moe_shuffling` zero-load
shortcut cannot prove an empty physical native scope is inactive. Core-owned
exact duration/count coverage remains authoritative for these new queries.

Expose phase plans and proof data through existing operator-time/phase reporting,
not only a changed scalar. The registry selects Step's position plan
once; old four-policy alltoall operators and numbers remain unchanged. Register
`step5_moe_dispatch` and `step5_moe_combine` as distinct plan-backed operators;
do not fill `expert_parallel_alltoall_*` with differently named primitives.
Routed output can reuse the existing `moe_tensor_parallel_allreduce` label with
explicit source-local ATTN_TP8 plan/group/dtype, rather than interpreting
Frontier moe_tp1 as the native output-reduction group. Shared output uses its
existing shared-expert label with the same source physical frame.
Both source TP groups belong to the full reduction plan; do not multiply one
isolated group's cost by2 or assume its scalar time is the joint critical path.
The existing scalar allreduce/network/intra mechanism remains the legacy path.
A may reuse its lowering/model only with matching reached native communicator
and group concurrency evidence. In particular, the retained Step source can
select FlashInfer/custom TP allreduce by size; a NCCL-ring scalar estimate is
not proof of that branch's GPU work. This is part of A's native algorithm/local
model evidence boundary, not an invitation to implement every TP backend.

## 5. Native expert scopes: the additional necessary correction

The current profiler's `NativeRoutedExperts.apply` calls finalized
`kernel.apply` (`frontier/profiling/moe/fp8_block_experts.py:97-123`), and
`moe_vllm_kernel.py:899-958` times that entire call as `moe_grouped_gemm`.
It includes prepare/finalize/local movement. A separate generic `moe_shuffling`
sample uses `moe_align_block_size`, not necessarily the native DEEPGEMM
permutation. Counting both unchanged would not be a disjoint native breakdown.

Use the **selected native** modular/expert invocation with profiling-only timer
wrappers at the source-prepare and inclusive-expert boundaries; do not reproduce the expert math
or select TRITON just to obtain a familiar feature tuple. Record which imported
native function/class was wrapped. Time the original call exactly once; source
binding remains owned by `profile_source_design.md`.

Required disjoint scopes:

- `moe_input_quantize` (new registered compute scope): the source-local naive
  prepare quantization before dispatch. Reuse actual `_quantize_and_setup_dispatch`
  behavior, including reached dtype/scale construction. This is included only
  for the reached nondeferred FP8 quantization contract. For the BF16
  `quant_dtype=None` path, `fused_moe/utils.py:461-462` returns the same activation
  and scale without launching a kernel: there is structurally no separately
  included quantization scope. Do not fabricate a positive BF16 quantize row.
- `moe_grouped_gemm`: use one **inclusive native routed-expert scope** over
  already prepared gathered tensors, including native permutation/alignment,
  both GEMMs, activation/intermediate quantization and native output
  unpermutation/weight-reduction. Declare
  `physical_scope=native_experts_inclusive` in the new generation's typed
  identity; this is not the old finalized-kernel scope including input prepare.
  Invoke the selected original native expert implementation with the native
  workspace/quantization metadata, not a copied GEMM routine.
- `moe_shuffling`: the new mode has no separately charged instance; its actual
  native movement is owned by the inclusive expert root above. Child native
  movement timers may remain diagnostics, never another charge. A structural
  absence is not an empty-lane timing fallback: the inclusive expert root must
  retain its own exact positive duration/count coverage for empty local lanes.

This inclusive-root partition is smaller than introducing independent scatter,
GEMM, activation and unpermute models. Do not split them merely for prettier
operator reports, and do not obtain one duration by subtracting another
measurement. Paired CUDA_EVENT/record-function evidence identifies the original
native expert call, all included kernels and its excluded source prepare.
No new native source edit is proposed. Core's planned separately charged
shuffling rows must be reconciled to this ownership before implementation.
`modular_kernel.py:1287-1347` provides the native buffer allocation and original
`fused_experts.apply` seam. Include reached native top-k finalization before
DP combine in the expert root if that implementation delegates it, rather than
dropping it because another implementation already returned finalized output.
Register the FP8 prepare operator as Step/source-policy applicable, using the
existing registry/typed-contract mechanism. Do not make old-policy tables or
the BF16 identity require a newly invented positive prepare row. The previous
proposal to put it in `STEP5_FFN_FAMILY` is withdrawn: that family is bound to
SHARED (`model_architectures.py:710-714`), whereas the MoE wrapper resolves
ROUTED (`moe_wrapper.py:794-808`). A shared-family prepare would fail the real
resolver or use the shared FFN width. Section 5.1 specifies the routed contract.

Padding must follow the **selected native expert implementation**:

- TRITON: use `_prepare_expert_assignment` and the actual `ignore_invalid_experts`
  behavior and selected BLOCK_SIZE_M; do not pad invalid sentinel rows as live.
- DEEPGEMM: `deep_gemm_utils.py:28-82,467-555` can use CPU expert counts for
  exact per-expert rounded M, or the physical-M capacity bound
  `round_up(M*K + min(M*K,local_experts)*(alignment-1), alignment)` when CPU counts
  are absent. Its EP scatter/layout initializes capacity beyond live local
  counts. Global Triton padding is not the DEEPGEMM feature.

The existing padded-feature schema therefore needs a native padding/layout
contract selected by actual expert backend evidence. Reuse the existing
core-owned schema classifier/trainer/range machinery. Add the backend's reached
capacity/alignment/layout fields only when they change the expert query. Both
gathered physical M and live assignment shape remain in the key. The profile
producer supplies native padding/capacity observations; the simulator must not
infer a DEEPGEMM allocation from a Triton all-expert padded sum. H800 alignment
is obtained from the native helper/receipt, not a guessed literal128.

This is necessary pricing scope, not an unrelated optimization. E352/EP16 means
22 local experts, unlike retained EP32 module profiles with11; those historical
kernel/layout rows cannot be admitted by merely changing the policy label.

Native phase order is now:

```
pre_dispatch: source norm/shared branch/shared TP reduce/gate/router/input quantize
dispatch:    one joint physical DP gather plan
routed_compute: one inclusive gathered native expert scope (movement included)
combine:     one joint physical DP reduce-scatter plan
post_combine: source routed TP reduce/branch merge/residual
```

`ExecutionTime._MOE_PHASE_COMPONENTS` currently puts shuffle before dispatch
and routed TP reduction before combine (`frontier/entities/execution_time.py:33-54`).
Change only the selected Step new-mode phase projection. Preserve legacy and
DSV4 projections; do not globally reorder unrelated models. Device kernel-gap
counts are actual paired scope counts, never broadcast primitive counts.
Native CPU gap, PD-AF, MTP and SP expansion are outside this design.

### 5.1 Exact registry target set and routed prepare contract

The **one target-selection interface for parent/core reconciliation** is
registry-driven `get_moe_profiling_operators(context)` in existing
`frontier/operators/families.py`, returning ordered applicable `OperatorSpec`
objects. Context is the existing selected Step layer/runtime contract plus
core-owned policy/source identity and native quantization receipt. No second
target registry or new context controller is needed. Add one optional
applicability predicate to `OperatorSpec` and optional context-aware filtering
to existing `OperatorFamilySpec.profiling_ops`; absent context preserves the
legacy target tuple. A conditional operator is not globally required merely
because it is registered. `moe_input_quantize` belongs to **MOE_FAMILY**, not a
new family and not SHARED. Its conditional selection is Step+position_cyclic+
reached nondeferred FP8 source prepare. `moe_shuffling` remains selected for
legacy contexts, but is excluded from this new-mode target set. Declare both
conditions at the existing registry entry; do not repeat lists at callers.

| Selected Step position contract | Ordered charged MoE CSV/duration/count targets | Frames |
| --- | --- | --- |
| BF16, quant_dtype=None | `moe_gating_linear`, `moe_gating_routing_topk`, `moe_grouped_gemm` | source, source, gathered |
| Reached nondeferred FP8 block quantization | `moe_gating_linear`, `moe_gating_routing_topk`, `moe_input_quantize`, `moe_grouped_gemm` | source, source, source, gathered |

The FP8 set's fourth target is **prepare, not shuffle**. Neither set contains a
separately charged alignment/shuffle. Legacy policies retain their original
gate/router/shuffle/GEMM targets and numerical behavior. A future deferred-FP8
branch must use its actual contract; it does not gain a source prepare target
because its weight precision is FP8. Whether that branch is reached at EP16 is
native evidence, not a new-policy design expansion.

`moe_input_quantize` has role RESHAPE, trace COMPUTE, resource MEMORY,
`tp_mode=MOE_TP`, `ep_agnostic=True`, execution attribute
`moe_input_quantize_time`, and its own precision identity. Native expert TP is1
in Frontier's active expert contract; eight physical source copies are runtime
placement, not estimator TP8 sharding. The existing ROUTED layer contract owns
MOE_FAMILY (`model_architectures.py:331-346`). Reuse
`resolve_layer_contract(operator_name=..., layer_kind=ROUTED, moe_tp_size=1,
expert_parallel_size=16)`; the resolver already validates E divisibility and
then omits EP from an EP-agnostic estimator (`:898-922,986-998`). **No
`model_architectures.py` source hunk is required for this choice**; real resolver
checks are required. The profile wrapper must enumerate the selected specs, not
blindly `MOE_FAMILY.profiling_ops()` without context (`moe_wrapper.py:840-848`).

Prepare's contract includes BF16 input `[ni,H]`, FP8 output `[ni,H]`, dispatched
FP32 scale `[ni,H/G]`, quant dtype/group/per-token mode, `defer_input_quant`,
native callable and source physical scope. Its model identity is FP8 activation
prepare, not BF16 because the generic Step binding falls back to model dtype
when there is no projection (`step5_cost.py:50-67`). Gate and router remain their
actual FP32 native contract even in a BF16 expert layer. Extend the existing
Step binding/quantization registry inventory, not a dtype literal at a caller;
the core prepare schema uses source rows and excludes expert-load features. It
may retain the ROUTED width in typed identity without treating that width as
prepare tensor geometry. The gathered inclusive expert stays EP-sensitive and
retains native TRITON/DEEPGEMM padding/capacity distinctions from section 5.

Parent/core must use the same selected specs for output CSV required columns,
paired `_kernel_count` columns, typed identities, model names, exact coverage,
cache inputs, root aggregation and fixture generation. Replace the core draft's
unconditional four-target requirement in the **position path only**. Both fit
paths (`moe_dataset_training.py` and `prediction_family_trainers.py`) and
`profiling/moe/main.py` consume this registry selection. Their current family
enumerations are real affected code, not documentation-only changes. Keep the
existing `moe_input_file`, count suffix and ratio models. No second producer,
CSV selector, generic per-op layout or legacy pseudo-model campaign is added.
This is the proposed concrete interface delivered for parent agreement; the
companion authors have not been presumed to accept it.
Registration/precision inventory must inspect registered operators, including
the conditional prepare entry, rather than use no-context legacy profiling
selection to decide whether its name exists. Runtime/profile/fit selection still
uses the one applicable-target interface; this is not another registry.

### 5.2 Real operation metrics and trace consumers

`metrics/ep_wave_metrics.py:82` constructs `OperationMetrics(name)` for every
positive operator. Add `MOE_INPUT_QUANTIZE`, `STEP5_MOE_DISPATCH` and
`STEP5_MOE_COMBINE` to `metrics/constants.py`; enum values exactly match registered
physical names. Keep existing routed/shared TP labels. Do not disable operation
metrics to avoid these failures.

`ep_wave_metrics.py:66-75` currently uses live routed assignments as all routed
phase tokens and the source batch compute width as pre-routing rows. Those are
not the gathered inclusive expert's physical M. For the selected position path,
read existing layer/source frame metadata populated by Step pricing and pass
source ni, gathered M, live routed assignments, local expert counts and native
capacity separately. Preserve request/source-batch/global-layer IDs and DP/EP
association. `OpTraceContext` gains an optional reference to this existing
immutable Step frame/plan metadata; do not invent DSV4 `_source_rows` or fake
expert Requests. In source phases use ni. In routed_compute use physical M for
input/output geometry and a separate local live-assignment count; no division
by K to infer M. Dispatch/combine metadata uses the full plan, not one token
scalar. Lane ledger cost and wave barrier duration remain distinct quantities.

`op_trace_utils.py:295-319` otherwise supports only Step logical EP all-to-all
and a restricted TP label set; `:338-346` reconstructs routed shape from a
pre-routing scalar. Route new names through the existing registry binding and
the attached full Step plan; report actual tensor list, per-rank local/gathered
shapes/bytes, DP groups, roots/order/nesting and numerical-source/buffer receipts.
Support the already reachable shared TP reduction with source-local BF16 shape.
Replace the fixed-family precision mapping (`:172-183`) with the existing
binding/registry iteration so prepare's registered identity is consumed. No new
trace operation registry is added.

`frontier/attention/step5.py:954-955,979-983` must accept the registered prepare
and selected inclusive expert metadata. Preserve legacy structural scope output;
new-mode `moe_grouped_gemm.covered_scopes` includes selected native permutation,
GEMMs/intermediate work and unpermute/weight-finalization, excludes source
quantization/DP exchange/TP reduction, and records native capacity/scale layouts.
Mixed FP8/BF16/INT32 tensor byte metadata must come from the contract, not a
single output dtype. A registry label change without this hunk still fails the
actual trace path or reports the wrong scope.

Required verification runs the existing real `record_ep_wave` path and actual
main with **operation metrics, op-level trace and lane ledger enabled together**.
`run_step5_measured_main.py:153` currently disables op-level tracing; change that
for the required corrected fixture, not the simulator to hide unsupported ops.
Check emitted values/shapes/scopes/count proofs for P `(6128,1)`, tail `(16,1)`,
q1 reached uniform buckets, and empty local experts. Assert no separate shuffle
event/charge, ni versus M are not swapped, and the one joint CC duration is not
counted eight times in the wave critical path. Legacy trace expectations remain.

### 5.3 Collector: current reachable preallocated combine scope

Current inspected defaults and retained DP2 receipt
`../task_2026-10-15_step5_preview_h800/step5_comm_20261020/s2_20261020_01/collectives/plan.json:568-581`
both have `VLLM_USE_NCCL_SYMM_MEM=false`, `VLLM_BATCH_INVARIANT=false`.
`all_reduce_utils.py:157-166` therefore disables NCCL symm AG/RS. A TP
`symm_mem_comm` availability flag is a different all-reduce backend and does not
enable DP NCCL symmetric buffers. This is evidence for the current/default
branch, not fresh derived-image runtime qualification.

For unequal P/tail, `allocate_combine_input` returns None from unequal sizes
(`all2all.py:183-185`) regardless of symm enablement. For current uniform q1 with
NCCL symm disabled, `get_symmetric_memory_buffer` also returns None
(`cuda_communicator.py:710-727`). Native modular output is caller-owned
`torch.empty_like(hidden_states)` (`modular_kernel.py:1469`), and finalize calls
`combine_into_output` (`naive_dp_ep.py:241-245`). Native into-output receives
explicit sizes and contiguous gathered input; `movedim(0,0).contiguous()` retains
its storage. It calls grouped unequal Reduce or uniform ReduceScatter into the
provided output. **No symm staging copy is reached in either current case.**

The current benchmark calls `group.reduce_scatterv` with output=None
(`vllm_collective_benchmark.py:357-359`), whereas native calls into-output. With
current symm disabled this allocates output inside the call but does not add a
staging GPU copy; still, its output lifetime/capture/scope differs and cannot be
admitted as proof of native preallocation. Extend the existing collector only:

1. Add the actual Step call `reduce_scatterv_into_output` to existing
   `vllm_forward_collectives.OPS`/plan and byte accounting; Step's combine source
   becomes `AgRsAll2AllManager.combine_into_output`. Leave other models' return
   output entry unchanged.
2. In existing `_issue`, allocate ordinary contiguous BF16 `[M,H]` input and
   `[ni,H]` output before timed repeats/capture. Close over both; call the real
   group method with explicit ordered sizes and dim0. Fill/setup/warmup and
   pointer/registration inspection remain outside device events. Repeats/replay
   reuse those exact buffers as native caller-owned output reuse; no extra copy
   after the returned tensor. Inspect return storage alias to supplied output.
3. Add per-call receipt: entrypoint, explicit-size branch, shape/stride/dtype,
   input/output storage alias, supplied-output=true, ordinary/symmetric input
   and output registration status, runtime flags and loaded PyNccl version,
   captured/eager lifetime, staging-copy count/bytes and paired kernel scope.
   For the current branch expected input/output nonalias, return/output alias,
   no staging, no timed allocation/fill/copy. Fail qualification if the observed
   branch disagrees; do not force symm off in a run that enables it.
4. Extend the existing untimed profiler pass to retain **per-rank per-call
   multiplicity**, explicit calls_per_repeat/replay normalization and copy
   evidence alongside existing distinct names. Do not sum kernel counts over
   eight ranks and call that the per-device count, or infer a count from roots.
   Bind the timing and count receipt to the same prepared scope/config/plan.
   Reuse the same prepared issue/run buffers for the timing and count phases of
   that logical sample, rather than preparing a different memory branch in
   `kernel_names`; keep the profiler outside the device-duration repeats. This
   refactors only the existing collector preparation/measurement seam.

The registered no-staging path is source-reachable only with uniform sizes,
NCCL symm enabled and direct output support (registered output or loaded
NCCL >= native `NCCL_DIRECT_SYMM_RS_OUTPUT_MIN_VERSION`; successful native
combine allocator additionally checks that version), and a successful registered
combine input. `naive_dp_ep.py:125-134,222-239` writes or reuses that registered
buffer; explicit uniform sizes plus preallocated output cannot use symm merely
on an ordinary input (`cuda_communicator.py:588-596`). With a registered input,
`:692-707` skips copy; the benchmark's current output=None ordinary-input path
would stage via `copy_`. Thus a future receipt that enables that branch invalidates
ordinary-input collector admission and requires its exact allocator/registration
scope. **That additional branch is not demonstrated for the current active run,
so no implementation hunk or hypothetical GPU measurement is budgeted here.**
Record/inspect actual conditions now; request a scoped design extension if fresh
native evidence reaches it. No forced symm flag or artificial registration is
part of this proposed correction.

## 6. Proposed implementation paths and added/deleted ceilings

These are **additional pricing hunks**, not measured diffs or implementation
approval. Paths are repository-relative. Shared files must be unioned once with
other owners by actual nonoverlapping hunks; do not simply add whole-file totals.
No README, new backend/controller, native source edit, launcher, ratio-model
replacement or immutable historical plan edit is proposed. A has explicit
submodule paths; B does not. The old parent-corrected **37 paths +1802/-185
=1987** is historical, not the revised scope. No approved full union follows
from these tables.

### 6.Common — mandatory pricing, contracts, reporting and current collector

| Exact file | + / - ceiling | Required pricing hunk |
| --- | ---: | --- |
| `frontier/moe_routing_policy.py` (core-owned new file) | 12 /0 | Source runtime context and native DP/TP association validation; no duplicated interval math |
| `frontier/scheduler/utils/batch_builders.py` | 20 /4 | New-mode lane compute rows/source context, preserve explicit gathered descriptor |
| `frontier/scheduler/utils/expert_parallel.py` | 18 /4 | Bind native Step source lane/EP rank without invoking DSV4 dummy path |
| `frontier/execution_time_predictor/step5_cost.py` | 180 /0 | Source/gathered queries and plans; selected registry targets, prepare precision/typed identity and metadata; no second Step cost module |
| `frontier/execution_time_predictor/moe_operator_times.py` | 65 /25 | Source gate/router widths/context; explicit gathered expert inputs; plan-backed dispatch/combine |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 65 /18 | Include prepare scope, source-local shared/routed TP costs and phase/proof binding |
| `frontier/execution_time_predictor/moe_routing_workload.py` | 25 /8 | Selected native padding/capacity features, no generic all-expert padding assumption |
| `frontier/moe_load_imbalance.py` | 25 /5 | Shared native padding-schema classification alongside existing core schema work |
| `frontier/operators/spec.py` | 70 /8 | Optional plan builder/count-proof result; conditional applicability and context-aware existing profiling enumeration, legacy default preserved |
| `frontier/operators/families.py` | 95 /12 | One selected target enumeration, conditional ROUTED/EP-agnostic prepare and inclusive/no-shuffle targets, Step plan registrations/source reductions |
| `frontier/entities/execution_time.py` | 45 /6 | Selected Step phase projection and plan count/proof storage, existing scalar aggregation |
| `frontier/entities/time_components.py` | 25 /5 | Include source quantize in existing registered MoE time/attribute aggregation, preserve inclusive-root child ownership |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 18 /2 | Existing registry dispatcher invokes one backend plan call |
| `frontier/cc_backend/collective_plan.py` (new) | 110 /0 | One immutable native plan/result and lossless serialization/validation; no numerical lookup or second op registry in common hunk |
| `frontier/cc_backend/base_cc_backend.py` | 8 /0 | Explicit unsupported plan capability by default |
| `frontier/execution_time_predictor/step5.py` | 20 /3 | Selected target/prerequisite and native-scope admission; fail unsupported plan/branch |
| `frontier/profiling/collectives/vllm_collective_benchmark.py` | 60 /14 | Current preallocated ordinary into-output scope and per-rank multiplicity/alias/registration/no-copy receipts; no enabled-symm substitute |
| `frontier/profiling/collectives/vllm_forward_collectives.py` | 8 /2 | Actual Step into-output op/source and endpoint bytes in existing collector plan |
| `frontier/config/quantization_manager.py` | 25 /4 | Registered conditional prepare identity/precision inventory; actual mixed tensors, not default model dtype or generic CC factor |
| `frontier/profiling/moe/fp8_block_experts.py` | 90 /15 | Original inclusive expert invocation on prepared gathered tensors and backend/layout receipts |
| `frontier/profiling/moe/moe_vllm_kernel.py` | 70 /18 | Gathered native inclusive scope, actual counts, no finalized-call/shuffle double pricing |
| `frontier/profiling/moe/moe_impl.py` | 25 /8 | Original source prepare timer; no separately charged new-mode generic alignment invocation |
| `frontier/profiling/moe/moe_wrapper.py` | 55 /8 | Source/gathered scope and selected routed specs instead of static family enumeration; existing resolver reused |
| `frontier/profiling/moe/main.py` | 30 /4 | Precision/reached-prepare-specific CSV targets and metadata, no positive absent scope |
| `frontier/execution_time_predictor/moe_dataset_training.py` | 25 /4 | Selected specs/native padding/prepare schemas and count pairing in existing fit |
| `frontier/execution_time_predictor/prediction_family_trainers.py` | 25 /4 | Same selected specs/contracts/count pairing in shared fit, preserve ratios |
| `frontier/metrics/constants.py` | 8 /0 | Exact prepare/dispatch/combine OperationMetrics members |
| `frontier/metrics/ep_wave_metrics.py` | 50 /10 | Source ni/gathered M/live counts/plan metadata at real phase reporting, no DSV4 identity |
| `frontier/metrics/op_trace_utils.py` | 65 /20 | Existing registry mapping and full Step compute/comm frame/plan metadata; source shared TP scope |
| `frontier/attention/step5.py` | 55 /10 | Reached prepare tensors and selected inclusive expert covered scopes/capacity in existing descriptor |
| `tests/unit/test_step5_moe_pricing.py` (new) | 150 /0 | Production wave source binding, P/tail/q1 frames, scopes/phases/reductions, plan keys |
| `tests/unit/test_comm_operator_families.py` | 50 /0 | Registry plan dispatch, multiple tensors/groups, legacy scalar identity |
| `tests/unit/test_step5_moe_profiling.py` | 65 /0 | Native scope invocation/timer partition and padding/scale receipts; additional to core tests |
| `tests/unit/test_moe_grouped_gemm_padded_features.py` | 40 /0 | TRITON vs DEEPGEMM capacity feature distinction, gathered frame/empty lane |
| `tests/unit/test_step5_comm_cost_export.py` | 40 /0 | Existing candidate evidence stays unqualified and incompatible scope cannot admit by old label |
| `tests/unit/test_collective_benchmark_helpers.py` | 60 /0 | Real into-output branch/alias/no-staging scope and per-rank count/replay normalization |
| `tests/integration/step5_measured_compute_fixture.py` | 50 /10 | Same selected source/gathered/prepare targets and native scopes; variant-identified CPU evidence |
| `tests/integration/run_step5_measured_main.py` | 50 /6 | Variant-bound input and operation metrics/op trace/ledger all enabled; production scope/plan proof checks |
| `tests/integration/test_step5_measured_main.py` | 55 /0 | Mandatory offline/online native-sized workflow and actual reporting assertions |
| `tests/unit/test_ep_wave_trace_context.py` | 65 /0 | Existing record_ep_wave with all reporting on, M versus ni/live and selected scope conservation |
| `tests/unit/test_op_trace_utils.py` | 45 /0 | Actual Step prepare/plan/shared-TP tensor metadata and legacy scope preservation |
| `tests/unit/test_step5_contract_reuse.py` | 40 /0 | Actual ROUTED resolver/EP-agnostic prepare, BF16/FP8 selected target contracts, not SHARED |

**Common: 42 distinct paths; +2102/-237 =2339 gross lines.** No
`model_architectures.py` hunk is necessary because prepare stays in MOE_FAMILY;
the contract check is deliberately in the existing resolver test. If parent
chooses a separate family instead, that is a different registry/model-contract
budget, not an unrecorded alias of this one.

### 6.A — existing simulator, known interfaces only

| Exact file | + / - ceiling | Known hunk, not the unresolved native algorithm |
| --- | ---: | --- |
| `frontier/cc_backend/cc_backend_config.py` | 6 /0 | Explicit qualified plan-model/algorithm-receipt input; no full-duration measured-cost selector |
| `frontier/cc_backend/backends/collective_sim_cc_backend.py` | 65 /4 | One plan scenario invocation, complete model/config/plan cache key and truthful simulated breakdown; no singleton scalar DP override |
| `frontier/cc_backend/backends/collective-sim/python/collective_sim_core/schema.py` | 65 /4 | Lossless grouped plan input, strict shape/group/order/algorithm evidence validation and runner serialization |
| `frontier/cc_backend/backends/collective-sim/htsim_runner.py` | 115 /4 | Existing spec/merge/CLI plan branch and known admitted flow-DAG transport into existing FlowDef/TriggerDef writer; native schedule generation not included |
| `frontier/cc_backend/backends/collective-sim/python/collective_sim_core/predictor.py` | 25 /0 | Plan-aware invocation/result, reject absent complete local/network rule; do not pretend phase=sum is a DAG |
| `frontier/cc_backend/backends/collective-sim/python/collective_sim_core/intra_server_model.py` | 20 /0 | Explicit plan scope distinction and missing-local-model error, not g=1 total local zero |
| `frontier/cc_backend/backends/collective-sim/tests/test_grouped_plan_input.py` (new) | 60 /0 | Plan serialization/known per-flow bytes/groups/dependencies and malformed input rejection, not a native-equivalence test |
| `tests/unit/test_collective_sim_cc_backend.py` (new) | 75 /0 | Registry-to-one-scenario/full-key cache and missing-model rejection; preserve old scalar API |

**A known increment: 8 distinct paths; +431/-12 =443 gross lines. Common+A
known subtotal: 50 distinct paths; +2533/-249 =2782 gross lines.** Missing native
schedule generation, root/local-duration model and any evidenced delayed
local-task lowering have **no honest added/deleted ceilings yet**. A complete
implementation budget is **unknown**, not 2782. Existing submodule C++ trigger
and connection-matrix seams named in 3.2 have zero *selected* changes, not proof
that no change can ever be needed. Submodule changes, tests and eventual gitlink
update must be integrated by parent if this variant is approved; no submodule
commit/build was performed here.

### 6.B — measured lookup, only after an explicit changed-CC decision

| Exact file | Additional + / - ceiling | Lookup-only hunk |
| --- | ---: | --- |
| `frontier/cc_backend/collective_plan.py` (already in Common) | 80 /0 | Qualified exact row matching/admission atop the shared plan; combined ceiling +190/-0 |
| `frontier/cc_backend/cc_backend_config.py` | 6 /0 | Explicit qualified full-plan measured-cost artifact path |
| `frontier/cc_backend/backends/collective_sim_cc_backend.py` | 55 /4 | Exact measured-plan lookup/result, complete key and honest numerical-source label |
| `tests/integration/step5_comm/export_costs.py` | 20 /1 | Into-output/full-plan evidence and explicit admission separation through existing exporter |
| `tests/unit/test_collective_sim_cc_backend.py` (new) | 90 /0 | Exact full-plan row matching/quality/scope/alias/count rejection, including vector permutations |

**B increment: 5 hunks in 5 paths, of which 4 paths are new to Common;
+251/-5 =256 gross lines. Common+B: 46 distinct paths; +2353/-242 =2595 gross
lines.** This budgets a complete lookup implementation, not native measurement
availability or E2E accuracy. Do not sum A+B or retain both paths as alternatives
in implementation. B is explicitly unapproved and contradicts the retained
simulation numerical source unless YC changes that decision.

Arithmetic used host `python3` only, with no repository import/test. Table
ceilings and sums are planning estimates, not measured diffs. Parent still owns
native/core/profile-source reconciliation, document/module-size exceptions and
the final user scope request. The source owner's provenance/launcher hunks and
core owner's adapter/frame-row/schema/cache/coverage hunks are not removed by
these additional pricing estimates.
The final document-table arithmetic check parsed each exact path and +/− cell:
Common 42/+2102/−237; A 8/+431/−12; B 5/+251/−5. It confirmed 50 unique
Common+A paths/2782 gross and 46 unique Common+B paths/2595 gross. B's extra
`collective_plan.py` hunk is intentionally counted once as a path and added to
its common ceiling. No tests or native-equivalence verdict were produced.

## 7. Verification and acceptance boundaries

No check below was run in this design. Future checks must inspect actual
outputs/count/proof keys, not merely successful exits.

1. Through the existing `prepare_ep_wave_inputs` / `prepare_moe_wave_from_inputs`
   producer, assert P6128+dummy1, D context6128/query16+dummy1, q1 context6144
   and reached capture-bucket vectors. ni gate/router/prepare/shared widths;
   M expert widths; correct DP owner for every EP rank; no TP multiplication of
   unique assignments. Also test same M with swapped n vectors, and same local
   load with different physical M: neither may alias a plan or expert feature key.
2. Spy on the **existing registry dispatcher**, not a standalone plan builder:
   verify full native tensors/dtypes/scales, DP groups `(t,8+t)`, unequal
   broadcast/reduce roots, uniform native branch and actual output-reduction
   TP groups. Assert zero calls to scalar alltoall/average-byte/p2p substitutes
   for the selected new mode. Assert one jointly priced plan per phase.
3. Source/profile invocation checks confirm original native wrappers, source
   prepare before dispatch, gathered movement after dispatch, final TP reduce
   after combine. Paired kernel counts and durations come from the same scoped
   sample; no duplicate finalized scope. Empty local expert lanes retain M and
   native capacity, physical transfers and exact core-owned coverage rules.
4. A tests exercise exact serialized plan/vector/ordered tensor/groups and
   evidenced per-flow dependencies through the existing runner. One traffic
   matrix contains all rails. Missing native schedule/local rule is rejected;
   no fabricated zero local duration or root-derived kernel count is admitted.
   B, only if selected, rejects candidate-only costs, mismatched image/source,
   vector/tensors/groups/alias/registration/graph and duration/count. In either
   variant old scalar APIs, four ratio-routing policies and DSV4 are unchanged.
5. Required CPU fixture and paired actual-main offline/online run retain
   6144/1024 request, P6128/D16+1023q1 census, native capacity/READ ordering,
   all16 expert participants, source/live/physical conservation and positive
   included costs. OperationMetrics/trace/ledger run together through actual
   reporting and emitted gathered metadata. Synthetic compute/count/algorithm
   evidence is marked `cpu_fixture`, never native admission. A actual-main full
   completion needs the complete simulator model, not just known interfaces;
   B would need its approved exact-plan costs. The selected numerical source and
   scope/proof identity reach production reporting without class patches.
6. For native acceptance, retain fresh derived-image TP8/DP2/no-SP full-model
   source/collective/alias/registration receipts, simulated-model qualification
   and fresh native comparisons for A (qualified exact-plan costs only for B),
   native scope durations/kernel counts and count/sentinel/padding agreement.
   Complete numerical inputs and independently referenced E2E evidence are required;
   old module probes or an actual-main CPU PASS cannot establish E2E accuracy.

Affected future command, after implementation and core/native/source integration:

```bash
python -m pytest -q tests/unit/test_step5_moe_pricing.py tests/unit/test_comm_operator_families.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_moe_profiling.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_contract_reuse.py tests/integration/test_step5_measured_main.py
```

Then use the existing fixture generator and `run_step5_measured_main.py
--fixture-manifest <fresh-position-manifest> --output-root <fresh-output> --mode all`.
Record exact environment/commands/values and limits in the parent's test report.
For A, add the scoped submodule `tests/test_grouped_plan_input.py` and the
eventual evidence-bound algorithm checks once their design is complete. These
are future affected checks, not tests run in this delegation. A mathematical
flow-volume PASS, a CPU reporting PASS or a modeled-cost fixture cannot replace
native grouped-call/local-cost/transport qualification and the four E2E gates.
No GPU run, new native CPU-gap campaign or broad test suite is authorized here.

## 8. Parent reconciliation and remaining evidence limits

Decided from source: source/gathered split, DP2 groups, heterogeneous tensor
order, unequal primitive expansion, physical dummy/padding retention, native
TP output reduction and scalar API unrepresentability. Existing FlowDef bytes
and triggers are expressive for a communication DAG. No routing formula or
balancing question remains.

Parent must reconcile these additional interfaces before summing budgets:

- Core source descriptor: source-specific gating context plus source ownership;
  retain frozen cache-key completeness. Core input producer continues to own
  intervals/positions/masks, not this pricing module.
- Core schema/coverage: native padding-capacity classification and new local
  prepare scope; accept the exact selected registry interface in 5.1, replacing
  the position path's unconditional four-target CSV/count/coverage requirement.
  One inclusive expert root owns movement, with no charged shuffle child. Current
  generic Triton padded tuple must not be asserted universal for DEEPGEMM.
- Profile-source owner: stage the same native source binding for these added
  scopes and collector receipts; no second image/adapter selector. Existing
  launcher ownership does not transfer here.
- Numerical design: retain collective_sim, obtain the concrete native
  schedule/local-duration inputs in 3.2 and close A's missing implementation
  ceilings before a **complete** scope request. B needs a separate explicit
  changed-CC decision and must never become an automatic fallback. Actual
  new-mode main/E2E acceptance remains blocked while the selected full-duration
  capability is unavailable, not partially complete.

Not settled by inspection: fresh derived-image actual expert branch at22 local
experts; per-frame native capacity/alignment and zero-live included kernel
behavior; grouped NCCL actual algorithm/channels/kernel launches and local versus
network work; fresh derived runtime symm/registration/transport scope; and formal
cost quality/admission. The current/default ordinary into-output branch is
settled by source and retained receipts. The only unresolved blocking decisions
are A's evidenced schedule/local model (including whether timed local dependency
support is actually necessary), parent agreement on the target interface, and
any explicit decision to replace simulation with B. No broader policy math,
PD-AF/MTP/SP, native CPU-gap or hypothetical symm campaign is requested.

## 9. Current design checkpoint — D29-PRICING-CORRECTION

### 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 2｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Native 与 existing-simulator 能力检查 | PASS | Sections 2–3 区分 primitive plan、FlowDef 能力与 native algorithm 缺证。 |
| 四项 reviewer 接口纠正 | PASS | Sections 5.1–5.3 明确 targets、ROUTED contract、实际 reporting 与 current into-output。 |
| 两种预算与 current artifact | PASS | Section 6 记录 +1802/-185 历史纠正、common 和分方案预算。 |
| A 的完整算法/数值预算 | BLOCKED | Section 3.2 缺 native schedule/local model；2782 只是 known subtotal。 |
| 源码、测试、GPU/platform 执行 | SKIPPED | 本次仅设计/source inspection 与轻量预算算术。 |
| Parent 接口/union/scope 交接 | TODO | core targets 须确认；B 尚未获得 changed-CC 决定。 |

### 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** D29-PRICING-CORRECTION 至今｜**当前前沿：** 完成 source-grounded design，缺证处停止完整估算。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 不实现 collector/test；只界定 current 必要 scope。 |
| D2 细节反复重试 | CLEAR | `python` 缺失后一次 `python3` 算术；无行为修复循环。 |
| D3 支线问题修补 | CLEAR | 只检查 assigned pricing/simulator/四项 reviewer blockers。 |
| D4 可并行任务停滞 | CLEAR | 明确只有本文件 ownership，禁止 agents；parent 负责 companions。 |
| D5 超范围镀金 | CLEAR | 不增加 generic controller、policy campaign 或未到达 symm branch。 |
| D6 分析空转 | CLEAR | 输出两个区分完整度的方案；具体缺证即停止 algorithm estimate。 |
| D7 路线违背 | CLEAR | 当前建议保留 simulation；lookup 明确 unapproved，不利用类名改 numerical source。 |

**文档检查：** S5：旧本文件 sections 5/6 漏 ROUTED/reporting/into-output 接口且推荐未批准 lookup，已在本次更新；core 四-target 要求须由 parent/core 对齐 section 5.1。S1/S2/S3/S4/S6 未发现本次新增缺陷；plan 已拥有这些 blockers，设计不能宣称完整数值预算。
**归因：** 当前窗口无 HIT；旧 proposal 的 off-route 风险由本次明确纠正。

### 3. 纠正建议

**分支：** B：旧 pricing doc 存在缺陷，先纠正文档再继续。

**文档更新（先执行）：**
1. `pricing_design.md` sections 1/3/5/6：已纠正 retained numerical source、targets/contract/reporting/current collector 与分方案 ceiling（可直接执行，已完成）。
2. Parent/core 的 CSV/count/coverage targets：确认并采用 section 5.1 的同一个 registry selection（parent/core 接口合并；本 agent 不编辑 companion）。
3. 任何改用 B 的要求：先修改 retained simulated-CC 决定，再讨论 full lookup scope（需用户确认）。
**需停止的偏离行为：** 无。

**接下去的路线：**
1. Parent/core 对齐精度特定 targets、existing dict/frame metadata 与 provenance；完成标志：producer、fit/cache/coverage、actual metrics/trace 使用同一 scope。
2. Parent 获取 matching native NCCL schedule/local model 决定；完成标志：A 有完整必要接口及 ceilings，不把 known 2782 subtotal 当 full approval。
3. Parent 汇总批准范围；implementation、actual-main 和 independently referenced E2E acceptance 都保留，不能选 counts-only。

### 4. 最值得关注的 3 点

1. A 可复用网络 flow/trigger，但 missing local/schedule evidence 阻塞 honest complete estimate。
2. BF16 三 targets、reached FP8 四 targets；第四个为 source prepare，非 separately charged shuffle。
3. Real operation metrics/trace 与 current native into-output 不能被省略；B 2595 ceiling 不代表 changed-CC 已批准。
