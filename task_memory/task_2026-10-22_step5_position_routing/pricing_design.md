# D29 Step source-frame and collective pricing design

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | D29-FINITE-PRODUCER-CLOSURE: share selected-call planning across benchmark/prepare/validator and bind the actual communication worker's eager module probe to the derived runtime and explicit source positions/mask; add all producer/validator/test ceilings. |
| 2026-10-22 | D29-FINITE-COLLECTIVE-MODEL: replace the unsupported exact-NCCL pre-code prerequisite with one finite ordered-flow/local-service model, bounded calibration/qualification, and a complete model implementation ceiling. |
| 2026-10-22 | D29-PRICING-CORRECTION: retain simulated CC; compare the lossless existing-simulator extension with unapproved measured lookup, correct routed targets/contracts and reporting/collector scopes, and separate budgeted interfaces from unresolved algorithm work. |
| 2026-10-22 | Parent distinguishes the proposed measured-cost variant from the existing collective_sim decision, records reusable flow primitives, and corrects the table sum to 1987 gross lines. |
| 2026-10-22 | Trace native calls to NCCL primitives; define separate source/expert frames, native compute scopes, exact collective-plan requirements, implementation ceilings and acceptance limits. Design only. |

TaskID: **D29-FINITE-PRODUCER-CLOSURE**, continuing **D29-FINITE-COLLECTIVE-MODEL**, **D29-PRICING-CORRECTION** and **D29-STEP-PRICING-DESIGN**.

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
file/line scope approval. **Recommend the single finite simulator proposal in
section 3.2:** ordered per-communicator DP2 transfers, all eight rails in one
HTSIM matrix, measured copy/reduce proxies plus a calibrated effective protocol
remainder, and explicit phase-level **sum/no-overlap**. It preserves exact
workload inputs, not exact NCCL microarchitecture. No new backend/controller,
network engine, generic GPU DAG or lossless per-channel reconstruction is needed
before coding. Model assumptions need parent/YC acceptance; the complete scope
can be decided now, while numerical inputs and native/E2E qualification follow
implementation. Measured full-plan lookup remains unapproved and is not another
viable choice under the retained simulation decision.

The previous 37-path table was **+1802/-185 =1987**, with section D **+585/-14
=599**. Parent's correction is retained. The revised tables in section 6 add the
reviewed interfaces and now give the **complete** common plus finite-model
ceiling before producer closure: 54 unique paths, +3518/-294 =3812 gross lines.
Section6.Closure adds14 necessary producer/interface/test hunks, six paths new
to pricing, +708/-29=737 gross. The current complete pricing ceiling is
**60 unique paths, +4226/-323=4549 gross lines**. The numerical model is unchanged;
these increments close actual entrypoint and validator omissions, not another
model variant. The prior 50-path/2782 known subtotal is superseded, not relabeled complete. No assertion that a
simulator must necessarily be larger than lookup is made.

All three parts, core `design.md`, derived-source `profile_source_design.md`,
native design and the actual-main fixture are mandatory for parent acceptance.
There is no counts-only completion option. Numerical scope approval remains a
parent/user decision before implementation.

The author edits only this document. This producer correction uses source
inspection and document arithmetic, with no code/test/shell execution. The
previous correction used `python3` for table arithmetic (`python` was absent).
No tests, GPU, Docker, platform, authentication, agents or commits were executed.
Source observations below are not fresh worker measurements. Companion files
remain owned by their authors; only the current core registry/binding interface
draft was inspected alongside requirements, plan and review. Their files were
not edited.

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

### 3.2 Finite simulator proposal (the one recommended modeling choice)

**What can already be reused.** `htsim_runner.py:319-379` has independent bytes
per `FlowDef`, starts and completion/dependency triggers, and one traffic matrix.
`:554-577` provides the placement map and all domain groups before an optional
singleton override. For ATTN_DP on TP8/DP2/EP1 it can enumerate all eight native
groups; EP16 is not a multiplicative dimension in that attention layout. The
existing network simulation can therefore run unequal directed transfers from
all groups together, with contention on the configured shared NIC/switch fabric.
No new generic simulator, controller, topology or placement registry is needed.
The legacy scalar DP method's singleton override must not enter the new plan.

**Complete implementation seam.** Add serialized `plan` to the existing
`CollectiveConfig`/`Scenario.to_runner_spec`, and its strict plan branch to
`htsim_runner.CollectiveSpec`/spec merge/validation/`generate_tm_from_spec`.
Use the one common immutable plan in `frontier/cc_backend/collective_plan.py`;
the submodule receives its lossless serialized value, not another native-op
registry. All explicit groups must agree with the existing placement; reject
missing/duplicate ranks, wrong order, inconsistent tensor shapes/bytes, or a
scalar payload plus plan. Preserve existing scalar scenario generation verbatim.
`CollectiveSimCCBackend.predict_collective_plan` uses the existing attention
layout and Python predictor **once per joint plan**. Cache full workload,
configured model/topology/network, local calibration and source identities.
Its scalar `_predict` is not a loop used to price the plan. This is a finite
extension, not a requirement to discover the native implementation first.

#### 3.2.1 Facts, model configuration and what they do not prove

| Source facts / reusable mechanism | Explicit model choice, not native equivalence |
| --- | --- |
| Unequal DP2 dispatch is ordered tensor-major, then roots0/1 grouped Broadcast; combine is ordered roots0/1 grouped Reduce. Uniform input uses AllGather/ReduceScatter; physical dummy/padding rows remain. | Preserve that exact ordered program and all group metadata. Model each communicator's tensor/root stages as serial network stages; eight communicators enter at the same modeled phase start. This conservative ordered-flow schedule is an assumption, not the meaning of `ncclGroupStart`. |
| Existing `allgather_model`, `reducescatter_model`, `allreduce_model`, `nchannels`, triggers and topology/link configuration already exist. Backend defaults are `ring_steps`, submodule defaults `nccl_ring`; neither is a native receipt. | Use the explicitly frozen existing selector. `ring_steps` uses one logical channel; `nccl_ring` uses configured `nchannels`. DP2 needs only one direct peer hop per root; no tree/channel discovery or new broadcast algorithm selector is required. Reject unsupported selectors for this position path instead of silently switching them. |
| HTSIM FlowDef permits individual bytes, trigger dependencies and all eight physical groups in one matrix. | Use exact logical transfer bytes, with existing HTSIM packetization/queues/CWND and finite channel stripes. No NCCL LL/LL128/Simple wire-byte multiplier is invented. Configured model name/channel count denotes simulator behavior, not loaded NCCL behavior. |
| `htsim_runner.py:29` runs `htsim_ndp`, not NCCL's transport. Existing config exposes link speed, MTU, queue, CWND, ECMP paths and switch/link latency. | Treat HTSIM as the retained network service/contestion abstraction. Freeze actual rank/host/NIC/transport evidence and declared topology parameters; use bounded native comparisons to qualify this abstraction. Do not relabel NDP as IB/RDMA or Socket. |
| `device_sku_config.py:57-63` supplies H800 capacity/TFLOPS, not HBM bandwidth. `scenario_profiles.py:53-64` marks H800 NVLink200GB/s, latency0.6us, efficiency0.8 as starter values. | No HBM constant or unmeasured copy/reduce coefficient is guessed. Measure finite local proxies below. Existing NVLink fields may be calibrated for reached TP8 reductions; starter values alone do not qualify the run. |
| Existing predictor supports sum/max; `phase` currently equals sum. DP2's g=1 early return is only zero NVLink traffic. | Select the existing **sum** combine rule for the proposed coarse no-overlap model. Add explicit positive local-work service; do not use the g=1 return as complete GPU-local cost. No timed-local DAG or channel/resource scheduler is added. |

This document proposes **one** modeling choice. The already implemented `max`
rule is not selected here and is not an automatic fallback. Changing overlap or
grouped stage concurrency after observed errors is a material model decision,
not a retry knob. Full-plan measured lookup is not a second viable option under
the retained CC decision.

#### 3.2.2 Concrete network lowering, complete and bounded to TP8/DP2

Let `nr` be rank-r's physical source rows, `M=n0+n1`, and `bj` the exact bytes
per row of tensor j. In each physical group `(t,8+t)`:

- Unequal dispatch: tensor j root0 sends `n0*bj` to rank1; root1 sends `n1*bj`
  to rank0. Their native root slices and ordered grouping stay in the plan.
- Unequal combine: nonroot1 sends root0's output slice `n0*H*2` to root0;
  nonroot0 sends root1's slice `n1*H*2` to root1. Reduction ownership is explicit.
- Uniform dispatch/RS: one stage per tensor, the two peer directions carry
  their actual equal source/output-slice bytes. The stage is still labeled its
  native AllGather/ReduceScatter, not two native P2P calls.

For each stage split the exact **element count** over C model channels by
quotient/remainder; multiply each nonempty stripe by that tensor's element size.
Stripe sums equal the specified bytes exactly, including tiny ID/scale tensors;
do not use `ceil` for every stripe or create size0/unbounded HTSIM flows. C=1
for `ring_steps`, C=`nchannels` for `nccl_ring`. This deterministic whole-stripe
abstraction has no unprovided chunk-size parameter or per-channel reconstruction
requirement. Dtypes, scale stride and payload tensors stay separate in the plan
even though the network engine moves bytes.

At phase start all eight groups' first stages are released at modeled time0.
Each **group-local** stage has one barrier fed by `send_done_trigger` from every
nonempty stripe/direction in that stage, then releases that group's next stage.
Use the existing FlowDef/TriggerDef writer. No global tensor/root barrier waits
for another rail; no tensor/root starts before the previous stage on its own
communicator. All groups' lists inhabit **one** network matrix and share the
configured fabric. Root order matters for unequal opposite-direction volumes.
The returned network makespan `N` is not summed isolated-rail costs. Number of
flows/stages/barrier activations is **model machinery**, never native launches
or kernel counts. Group nesting/order is preserved as a source fact, while this
non-fused serial stage schedule is identified separately as an assumption.

This lowering is complete for current DP2. There is no arbitrary-DP tree/ring
generator, generic task DAG, new C++ trigger, GPU-resource queue or protocol
packet expansion in this proposal. Exact per-channel reconstruction becomes a
possible later remedy only if error evidence identifies this model as the cause.

#### 3.2.3 GPU-local service and protocol cost without invented coefficients

Use the existing native collector's prepared buffers, queued CUDA-event timing,
repeats/graph replay and per-rank gather for **separate labeled local probes**.
They do not replace native NCCL observations. For each reached dtype/context,
fit nonnegative affine functions in milliseconds, indexed by actual rank,
tensor layout/dtype and reached eager/captured modality (those indices are
omitted from the notation below):

`C_dtype(B)=a_copy_dtype+b_copy_dtype*B` from preallocated `dst.copy_(src)`;
`R_BF16(B)=a_reduce+b_reduce*B` from preallocated
`torch.add(local, peer, out=output)` on BF16 `[rows,H]`.

`B` means one operand's exact bytes. Local copies preserve actual tensor
shape/stride; an FP8 byte view may only be used for its copy with identical
storage/byte count and an explicit probe receipt. Reduction is actual BF16 add,
not int8 arithmetic. Buffers/fill/capture setup and alignment are outside device
timing; concurrent local probes run on all sixteen GPUs. Record per-rank fits
and actual local probe kernels separately from native counts. Copy/reduce fits
must have positive measured service/slope on the admitted range; a zero local
fit is invalid, not an excuse to remove GPU-local work. Affine intercepts may
be zero if supported by the fit and explicitly reported. No fabricated default
values are allowed; absent/invalid inputs fail admission.

The proposed **HBM service proxy**, not a claim about NCCL kernel decomposition,
is, for each physical rank r:

`L_dispatch,r = sum_j [C_dtypej(nr*bj) + b_copy_dtypej*(M*bj)/2]`

`L_combine,r = R_BF16(nr*H*2) + b_copy_BF16*(M*H*2)/2`.

Explanation: dispatch accounts for copying the rank's own root slice into its
gathered output, plus sender-read/receiver-write service for the peer transfer.
Combine accounts for adding local/received contributions into the caller's
output, plus peer-transfer memory service. A standalone memcpy reads and writes
one operand, so its slope corresponds to **two** operand-byte streams; the
division by2 converts modeled send-read plus receive-write bytes to that
measured proxy. It is an explicit symmetric read/write throughput assumption,
not a guessed bandwidth. No standalone staging copy is added: the current
native ordinary/preallocated branch has none. Network link/NIC timing has no
HBM cost in this abstraction, so the local memory term is not also billed there.
The local proxy intercepts denote local probe cost, not actual NCCL launch counts.

Use `L=max_r L_r` across the actual sixteen ranks. This reflects a coarse
parallel local service phase; summing sixteen GPUs is not its critical path.
For the selected no-overlap rule:

`T_DP_plan = N_joint8rails + L + P_q(S)`.

Here `S` is max per-rank logical send+receive bytes of the **full plan**; for
DP2 it is `M*sum_j bj` for dispatch and `M*H*2` for combine. The complete size
vector/root/tensor/group plan still determines N and L; S is only a protocol
model feature, not a surrogate plan key. `P_q(S)=a_protocol,q+b_protocol,q*S`
is a fitted **effective protocol remainder** for each reached phase,
dispatch dtype tuple, unequal/uniform collective branch, eager/captured scope
and frozen transport/communicator policy. Actual kernel/protocol observations
remain attached per cell; do not create a separate one-size parameter record
for every kernel name or reconstruct its channels. Combine uses its BF16 output
contract but retains the selected prepare/expert context in plan identity.
It is charged once per joint plan, not multiplied by roots, tensors, kernels or
rails. It represents grouped device protocol/setup and service not represented
by the fixed network/local proxies. It may also absorb this coarse abstraction's
systematic remainder; it is **not** a measurement of isolated NCCL protocol or
proof of non-overlap. That distinction is required in cost metadata/reports.

Fit this affine remainder jointly against native full-plan device durations,
using fixed N and independently fitted L. Preserve raw observations/residuals;
do not clip negative samples, scale network time, set L=0, or force-fit a positive
remainder when the fixed model overpredicts. A nonnegative fit is a defined
calibration algorithm, not invented coefficients. Report its error and use
independent qualification cells/repetitions. If a useful nonnegative fit cannot
pass the qualification, the model is rejected with its signed residual evidence;
changing overlap/stage scheduling requires a recorded parent/YC decision. There
is no hidden E2E correction factor and no direct native-duration lookup. A zero
fitted protocol coefficient is not a claim that native protocol work is absent;
its raw residuals and independent duration errors remain reported. No added
per-operator10% hard gate is introduced by this model: operator errors diagnose
the assumptions; scope/count/transport/quality must match, and the existing four
independent E2E10% criteria decide final acceptance. An invalid fit/range/quality
receipt is not admitted even if a synthetic fixture exits successfully.

**Reached source-local TP reductions.** Shared/routed output remains source
`[nr,H]` BF16 over two disjoint intra-node TP8 groups, after their actual phases.
Reuse existing `nvlink_analytic` transfer/step formula with each group's own B:
`V8(B)=1.75*B/(effective_nvlink_Bps)*1000 + 14*(latency_us+overhead_us)/1000`.
The numbers1.75 and14 derive from the selected eight-rank ring's
`2*(8-1)/8` bytes factor and `2*(8-1)` steps; they are modeled ring steps, **not
observed launches**. Fit effective bandwidth through the existing efficiency
field and device overhead through existing `nvlink_allreduce_launch_overhead_us`
from bounded native TP8 rows, with declared one-way bandwidth/latency inputs.
Record infeasible/negative fits instead of inventing an efficiency or overhead.
These **effective full-device** parameters include GPU-local reduction/protocol
service; do not add the DP copy/reduce proxy a second time to V8. Custom/FlashInfer
native identity remains in receipts, and the ring is explicitly its simulator
approximation, not a claim that NCCL ran. Each reached eager/captured runtime
policy has a fitted record, selected through the same input artifact. If that
policy chooses different actual implementations by size, keep the observed
implementation in every raw/count receipt and qualify the affine approximation
across the frozen range; do not hide the switch or demand three sizes per
kernel-specific identity before a model design is possible. Use
`max(V8(B0),V8(B1))` as the assumed parallel critical path for the two disjoint
nodes. Do not sum or multiply it by2. This is the finite current TP8 model;
other communicators/topologies are not implemented speculatively.

#### 3.2.4 Existing input/producer and bounded collection, after implementation

No new model selector, controller, path switch or local backend is needed.
Reuse `BaseCCBackendConfig.profiling_data_dir`, selected device and native runtime
identity to load `profiling_data_dir/<device>/collective_local.json`. That new
artifact contains the copy/reduce affine coefficients, protocol coefficients,
context-bound effective TP NVLink fields, calibration/model/config/source
identity, finite supported range, quality/qualification proof and actual paired
native count receipts. Reject missing, mismatched or unqualified records; old
policies do not load it. Synthetic values are allowed **only** in the explicitly
`cpu_fixture` artifact and never copied to production evidence.

Extend existing `tests/integration/step5_comm/export_costs.py` to fit/export these
finite parameters from existing collector output plus the frozen simulator
configuration, using nonnegative least squares and the existing descriptive
quality/source receipts. It remains a candidate exporter; a separate parent
qualification receipt selects the model. Preserve full native timings as
observations, not a runtime full-duration lookup table. Do not fit on the E2E
test result. No producer source receipt or actual integer count is synthesized
from a fitted coefficient.

The bounded collection definition is the new
`tests/integration/step5_comm/step5_position_model_run.json`, staged through the
existing `prepare.py`/worker/native collector. Do not edit historical S0/S1/S2
definitions or add a launcher. Required grid inputs are exact declared local
work rows and the frozen simulator config, not a runtime retry/model toggle.
`prepare.py` needs their propagation and correct expected local-evidence counts;
the existing worker invokes the same benchmark and hashes the changed sources.
The new definition declares `grid.native_call_names` (only reached dispatch,
into-output combine and shared/routed TP labels), `grid.local_work` (explicit
probe kind/layout/row anchors) and each scenario's calibration/qualification
membership. These are collector **workload inputs**, not new numerical backend
selectors. One shared `forward_collectives(..., native_call_names=...)` entry
selects calls for benchmark, prepare and validator; no caller-local filter or
unfiltered validator is allowed (section5.4). The legacy grid is unchanged. Local probe records
go to a separate `local_work.jsonl`, with scope/layout/dtype/rank/device/mode,
per-rank repeat times and actual probe events. Native measurements/counts retain
their existing records. `prepare.py` invokes that same selected-call planning and
propagates/hashes the external frozen model config so expected counts are not
those of the unfiltered forward. The budget includes these concrete producer
changes; no unchanged full-grid count or implicit worker input is assumed.

- Local calibration: four row anchors **1,16,256,8192**, only reached source
  tensor layouts/dtypes (BF16/FP8 activation H4096, weights K8, IDs K8, scales
  H/G32) and BF16 `[rows,4096]` reduce. At most20 copy cells +4 reduce cells per
  reached eager/captured modality, all sixteen devices; preserve actual graph
  eligibility and do not force a graph on a native-ineligible scope.
- Native parameter calibration: at most **nine** scenario vectors: unequal
  eager `(16,1),(256,1),(8192,1)`; uniform eager `(1,1),(64,64),(512,512)`; uniform
  captured the same three **only if those buckets are actually supported**.
  Use only reached BF16/FP8 dispatch, combine and source TP8 calls, not every
  forward collective. Each fitted context needs three distinct byte sizes; absent
  modality/precision is not invented as another target. If the fresh census's
  supported uniform buckets differ, freeze its smallest/middle/largest instead
  before collecting, keeping the same three-per-modality bound.
- Independent operator qualification: fresh repeats of `(6128,1),(16,1)` and
  their swapped owners, plus the smallest/largest **reached q1 physical uniform
  buckets** (at most six vectors). Do not feed these qualification observations
  or E2E results back into the fit. Tail16 is permitted as an independently
  repeated calibration-size check, explicitly not an unseen-size holdout.
- Reuse the current collector's **3 passes,30 repeats,5 calls per repeat** and
  queue-held evidence; all8 rails active, same derived source/runtime and ordinary
  preallocated combine scope. Separate local probe rows from native timing/count
  rows. These are proposed finite experiment settings, not executed measurements.

Calibration weights use per-cell repeat medians (same fixed model/context)
rather than giving repeated copies of one cell more fit weight. Preserve all
passes, raw repeats, queue-quality failures and signed residuals; fit only rows
qualified by the existing queue/scope checks, recording every excluded row and
reason. No statistical trimming, E2E tuning or automatic interpolation beyond
the frozen supported range. The exporter records design-matrix rank and units;
an unidentified slope/intercept or native policy/range switch is not solved by
inserting a default number.

The design allows one bounded calibration/qualification package after code and
run approval. It is not a full-policy/native microarchitecture campaign.
Durations/counts for actual-main/E2E still use actual reached scoped proof;
the calibration grid does not replace core exact empty/masked/source coverage.

#### 3.2.5 Exact inputs not provided, and decisions versus later validation

| Item | Status and bounded resolution |
| --- | --- |
| Ordered per-communicator stages, modeled simultaneous rail phase entry, exact channel striping and phase-level sum/no-overlap | **Explicit model assumptions, decision-ready now.** Parent/YC accepts or rejects this one finite proposal. They are not source observations and need no per-channel reconstruction prerequisite. |
| Frozen `allgather_model`, `reducescatter_model`, `allreduce_model`, `nchannels`, triggers; rank/NIC/topology/link/MTU/queue/CWND/paths/latencies; active device identity | **Unprovided selected values/qualification.** Existing config fields and collector `hosts`/groups/runtime plus `step5_comm/transport.py` inventory/logs supply them. Defaults/profiles are starter inputs, not verified hardware. No new arbitrary coefficient or forced transport. |
| Copy/reduce a,b per reached dtype/modality/device | **Unprovided measurements.** The bounded local probes and exporter above supply them; no HBM registry value currently exists. |
| Effective protocol a,b and TP NVLink efficiency/device-overhead records | **Unprovided fitted parameters.** Three native sizes per reached context, fixed network/local proxies and existing exporter supply them. Numeric calibration follows implementation and a run decision; absent parameters do not block a complete code ceiling. |
| Native actual scope, ordinary/registered buffer branch, count multiplicity and per-rank duration | **Post-implementation native evidence.** Existing corrected collector/provenance path supplies it; actual counts never come from model stages. Unexpected reached branches require bounded scope reconciliation, not hypothetical implementations now. |
| Four independently referenced E2E errors <=10% each, plus operator comparison and signed errors | **Post-implementation qualification**, mandatory. Coarse model accuracy is not claimed by this design. Finer concurrency/overlap/protocol machinery is considered only if observed errors locate a need. |

This is a complete implementable finite model **with unprovided numerical
inputs**, not a promise to meet 10% without validation. Its entire algorithm,
config/producer/consumer and affected test scope is counted in section6.Model.
No exact-NCCL reconstruction or full GPU/network DAG is a pre-code criterion.
The specific pending parent/YC decision is whether to adopt this fixed
ordered-stage/channel abstraction, symmetric copy/add HBM proxy, affine effective
protocol remainder and no-overlap sum, together with its bounded collection.
Choosing it authorizes neither arbitrary numerical parameter values nor a run;
parameters still need valid producer evidence and the run follows normal approval.

### 3.3 Measured full-plan lookup: retained as explicitly unapproved history

The previous lookup comparison changed the communication numerical source. It
remains unapproved; no lookup artifact/path/fallback is selected or budgeted in
the current proposal. Measured **local parameters/effective remainder** qualify
the finite simulator, while network predictions still come from joint HTSIM;
returning an observed full-plan duration would instead be measured CC and needs
a separate YC decision. The old lookup ceiling2595 is historical, not another
current approved/viable scope or a reason to prefer the class-name disguise.

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
Use the finite context-calibrated TP8 model and parallel-node critical-path
assumption in3.2.3, not the legacy scalar singleton as an undocumented estimate.
FlashInfer/custom native identity remains actual evidence, while the selected
ring/NVLink formula is explicitly a simulator approximation qualified against
those calls. Other TP backends and a new exact-channel implementation are not
required before coding this current bounded model.

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
This interface is now accepted at design level by the current core draft
(`design.md:1238-1244`) and reviewer closure. Core's bound `operator_facts` mapping
uses these specs through existing query/layer/precision/context resolvers before
loader/helper calls; it is not another target selector. Their remaining
constructor/trainer/timer-method seams stay core-owned. No runtime/test approval
is inferred from this design agreement.
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
   `vllm_forward_collectives.OPS`/plan and byte accounting; the new explicitly
   selected Step plan's combine source becomes
   `AgRsAll2AllManager.combine_into_output`. Leave legacy unselected Step and
   other models' return-output entries unchanged (shared entry in5.4).
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

### 5.4 Producer closure: one selected-call planner, including validation

**Observed omission:** benchmark `main:557-560`, package `prepare:58-59` and
`validate_result.validate:25-38` independently enumerate all forward calls.
The finite definition selects five reached call labels, but the prior design
only corrected the first two consumers. Worker `run_job:262-269` always runs
the validator and includes its result; bypassing that validator is not a repair.

Extend the existing public entry, not a second plan/registry/controller:

```python
forward_collectives(model, parallel, scenario, *, native_call_names=None)
    -> list[CollectiveCall]
```

For Step, obtain `_step5_collectives` first and apply selection **before** the
current early return at305-311. The Common8/2 plan hunk constructs the actual
into-output combine descriptor only for the explicit Step subset; the same
existing `_step5_collectives` construction retains its return-output descriptor
without selection. No second inventory or caller patch replaces the op afterward.
No-option returns the identical legacy list,
order and multiplicities; the non-Step default is also unchanged. An explicit
selector requires Step and a nonempty sequence of distinct existing call names;
unknown/missing names fail before execution. Preserve native plan order, not
selector order. Select actual `CollectiveCall` objects without modifying tensor,
size-vector, group, rails, captured flag or calls_per_forward. The current new
definition selects `moe_dispatch_all_gatherv_fp8`,
`moe_dispatch_all_gatherv_bf16`, `moe_combine_reduce_scatterv`,
`shared_expert_down_proj_all_reduce`, and `routed_expert_all_reduce`; the combine
name stays stable while its actual op/source becomes into-output as in5.3.
There is no duplicate op-label inventory in the selector.

Benchmark/list-cases, prepare expected counts and validator expected keys all
call this entry with `grid.get("native_call_names")`. The existing rail-isolation
transformation remains before the benchmark call; formal new collection requires
all eight rails. Prepare propagates the grid without dropping the selector.
The fixture producer in `test_step5_comm_transport.fixture_result:65-119` uses
this same entry for synthetic records/expected rows. Native cases and local-work
records stay in their separate collections; local rows are not counted as native
measurement/kernel/summary rows.
Add optional bound-definition/selector arguments to that existing fixture;
when trimming its scenario list at68, retain `native_call_names` and binding
fields instead of replacing the grid and losing selection. Existing stage-only
fixture calls retain their current defaults. No second artifact builder is added.

Validator requires exact selected `(pass, scenario, call)` keys and selected
kernel/summary keys plus the existing byte/group/queue/source checks. Compute
counts from the shared plan and also compare `job["expected"]`; merely lowering
the prepared expected count cannot bless missing selected evidence. Missing,
duplicate or unexpected native records must produce explicit UNVERIFIED key
checks, not a KeyError in `plan[key]` at45 or a summary lookup failure at68.
Continue independent valid-row checks after recording unknown/missing keys.
The advisory CLI still exits zero; qualification is false and the worker's
ordinary advisory receipt is retained. Do not suppress this check in worker.

**Affected checks, using the existing fixture, not another test controller:**
unfiltered S0/S1/S2 controls remain PASS; an explicitly selected proper subset
with all its rows/keys passes; remove all evidence for one selected call and
recompute the superficial job counts, and validation still fails selected-key
coverage; duplicate/extra/unselected and unknown requested labels fail. Existing
benchmark list-cases and prepare package tests compare the same selected count
and deterministic call order. These are CPU structure tests, not cost admission.

### 5.5 Actual pricing worker probe: derived-source and positions/mask binding

#### Source facts and the reason this old entrypoint now needs a pricing hunk

`prepare.py:61-89` currently forwards only the benchmark grid and layer/token
probe metadata. Its image is an unvalidated definition string. `submit.py:23`
already consumes `job["image"]`; its resource env contains only distributed/NCCL
settings. `run_on_worker.sh:19-23` sets source-root PYTHONPATH and runs worker;
neither process sets a routing binding. Worker `probe_command:87-100` passes
only model/output/topology/token counts; `run_job:248` runs that exact module.
Probe `create_native_config:328-345` builds auto native text/quant config with
enforce_eager=True. `load_native_layer:348-361` initializes Step5MoE directly.
The plain call at454-456 has neither `is_padding` nor logical positions and
never traverses `Step5ForCausalLM.forward`. Native runner629-640 selects experts
before gathered expert execution; adding an env label after the call fixes none
of these omissions.

Native19/T2, inspected at
`/data/ycfeng/Frontier/.worktrees/step5-d24-stage/task_memory/task_2026-10-22_step5_d29_native_balanced/native_design.md:18-36,86,142,154-160`,
already defines the actual `set_routing_positions` setter and explicitly binds
positions/mask for its separate module checks. T2 reuses the old probe's config,
loader and observer **components**, not `run_probe`, and budgets no old-probe
edit. That remains correct for T2. **This pricing proposal newly includes
worker248's probe as its collection identity prerequisite, so pricing owns the
additional old-entrypoint binding hunk below.** It is not charged to T2/native19
or hidden in the source owner's P0-launcher scope. Parent records this explicit
scope addition during union reconciliation; no native source/T2 edit is proposed.

#### Selection and propagation through the existing package

1. The new finite definition references the source-owned bound profile manifest
   and core-owned `routing_input_file` plus explicit `native_probe.moe_input_ids`.
   Reuse manifest fields/`verify_step5_profile_runtime` from the source design,
   including adapter `step5_position_cyclic`, exact `native_reference`, new
   generation, model/source revisions, installed root, pins/full receipt and
   derivation. For this existing source mount, require the bound manifest and
   referenced receipts to reside within the mounted repository and retain the
   exact bound generation across both producers/validator; the collection run_id
   stays its existing independent output identity. No remote/missing artifact or
   extra mount is substituted. Preserve the original sealed manifest and its reference
   resolution, not a relocated/relabelled copy. These are existing artifact locators, not another routing selector
   or manifest schema. Require a nonempty unique ID subset of the actual retained
   core inputs. No new interval/position grid is constructed by pricing.
2. Prepare uses the existing staged adapter/check_native_reference to validate
   that manifest and derives `job.image` from its immutable native_reference;
   reject a conflicting definition image. Copy the explicit routing input into
   the existing exclusive stage and hash it; include the exact existing bound
   manifest/receipt paths and their hashes in the same package inventory, without
   editing/resealing the source-owned stage. Use direct
   registry import with its existing vllm_burst_driver dependency. Add those
   actual repository paths, `runtime_identity.py`, and core `inputs.py` to the
   existing source_hashes selection, not a second copier/pin registry. External
   input files have package hashes, not a claimed Git revision. Freeze the same
   binding locator in benchmark and job/native_probe; copy byte-identical model
   metadata and require its SHA to agree with the bound model config. Record the
   selected input IDs and per-ID physical vectors from the core parser. The
   finite definition's `native_environment` mapping serializes existing native
   variables/values only; prepare copies it unchanged into benchmark/job and
   requires agreement with bound runtime policy. It is an environment input,
   not another backend toggle or strategy selector. Strategy always derives
   from the adapter, never from a competing map value.
3. In worker, before `runtime_manifest` imports torch/vLLM, resolve that single
   binding with `adapter_record`/`check_native_reference` and apply its validated
   `native_reference["routing_strategy"]`. This is the same registry-owned value
   native P3's server_env uses, not a copied adapter or strategy classification.
   Do not call serving `runtime_version`/`server_env` with the collective engine:
   their required serving tokenizer fields are intentionally different. Use
   the existing adapter/reference check, not a second family/runtime resolver.
   Remove inherited simulation strategy then set its validated value; pass
   `FRONTIER_STEP5_PROFILE_MANIFEST` to both torchrun subprocesses. Apply the
   frozen existing native environment inputs, including skip-padding=on,
   graph environment and the current symm/batch-invariant settings, exactly as
   declared in the bound collection. Preserve platform NCCL values. No new
   switch or forced registration is added; raw env absence and effective vllm.envs
   values are separate receipt fields. Effective mismatch is not ignored.
4. Extend `probe_command` only for the bound generation: pass
   `--routing-input-file <frozen-file> --moe-input-ids <selected IDs>` and retain
   the existing model/output/topology/custom-all-reduce arguments. Omit the
   single legacy token-vector option for this path, since each selected input
   supplies its own physical vector. Parse_args/probe_plan requires both new
   input arguments and a matching manifest together; native position mode without
   them fails, rather than using the old `[1]*DP` default. Worker/prepare topology
   checks consume the core parsed vectors in this branch. Legacy definitions have
   no binding, keep the old token option/defaults/command/report fields exactly.
5. Benchmark consumes the same environment/manifest before runner initialization
   and calls the source-owned verifier in that actual producer process. Bind
   the original loaded communication entrypoints/PyNccl callables and recheck
   imports after the untimed count pass; attach shared native facts to the
   existing plan/measurement evidence. Worker manifest alone cannot establish
   a later subprocess's native source. No new preflight launcher, shell change
   or ResourceConfig/env path is necessary: submit already uses prepared image,
   and the worker supplies both child processes' effective native environment.
   Require the frozen benchmark/job binding locator to equal the child process
   locator, and effective values to match the prepared mapping; inherited env
   alone cannot select a bound generation. With no declared binding, preserve
   legacy plan/record/report keys exactly, omit new facts, and never relabel
   native strategy-empty/old probe output as position_cyclic. An effective cyclic
   selector without the required explicit probe inputs/binding is rejected.
   The collector issues communication primitives, not GateLinear/router. Its
   source/setting agreement is not a claimed executed cyclic route; only the
   context-bound probe provides eager module-routing evidence, and T2/full serving
   provide graph/model evidence. Do not infer any of these from the other receipt.

The reusable verifier is a **planned source-owned dependency**, not currently
implemented code. Its full installed-source receipt and the native19 derived
adapter/setter must land before this branch can execute. The pricing hunk only
calls its existing agreed interface; it does not duplicate its hash/origin logic.
Byte-identical foreign-root imports, changed receipt/config/input hashes,
wrong generation/reference/image, missing bound artifacts, standard/uniform
strategy, wrong effective padding/backend settings or wrong actual router binding
cannot become accepted cyclic evidence. Declared config errors reject preparation
or selected invocation; runtime diagnostic failures retain UNVERIFIED/advisory
receipts and never replace native exceptions with a fabricated successful probe.

#### Minimal eager module execution and truthful proof scope

Use core `MoERoutingProfileInput.source_frames` parsing and its existing
positions/mask materialization contract (design6.2, inputs.py), not another
request/count implementation. Bind each native DP rank to its actual source
lane, replicating that lane's input on its eight TP devices. Validate DP2/TP8/PP1,
no-SP/mega/MTP/EPLB and gathered physical width. Empty intervals are all padding,
not position0/live1. Materialize signed integer device positions and boolean
mask once per supplied input before execution. Physical CPU sizes remain
positive and exact even for idle/all-padding lanes. Native setup/groups/workspace,
dummy weights, original GateLinear, auto oracle and inclusive experts are reused.
The eight-line `inputs.py` increment below exposes that already-planned
materialization as `source_frame_tensors(frame, *, device) -> (positions, mask)`
for both callers; it must share the core wrapper's implementation, not coexist
with a copied interval loop. This is an explicit core-shared seam for parent
reconciliation, not a claim that the function exists in the current tree.
The four-added/four-deleted `moe_impl.py` adapter connector below calls that same
function in core's planned preparation path; it does not introduce a second
materialization or count owner.

For every selected input and the existing two layer identities, call:

```python
with torch.inference_mode(), set_forward_context(
        None, config, num_tokens=frame.physical_rows,
        num_tokens_across_dp=physical_sizes_cpu, is_padding=local_mask):
    with set_routing_positions(local_positions):
        shared, routed = layer(inputs)
```

This is the real native setter, not assigning/cloning a context dictionary or
invoking a custom router. It binds the missing scope explicitly because a direct
Step5MoE call lacks the outer text-model wrapper. Existing native setter cleanup
owns normal/nested/exception restoration. Observe actual `layer.experts.router`
and original resolved select/compute-routing implementation, routing_method_type
Simulated, configured E/K/scoring/normalize/scale/EPLB and effective native env
**before** transparent diagnostic interception; check actual bound callables
with the shared verifier. A Simulated class with a different effective strategy
is not sufficient. Use full receipt checks for the actual defining implementation,
not the inherited base method alone or a constructed class name as execution proof.

Extend the existing DispatchObserver with an untimed transparent selection
observation: retain the returned physical IDs/weights and supplied logits/context
references after calling the original; pass the original results through unchanged.
Outside execution, compare live IDs with the explicit input positions and masked
IDs=-1/weights=0, and the configured selected unbiased sigmoid/normalization/scale
contract with the **same observed logits**. Retain discrepancies, observed route
and assignment totals and actual executed expert classes. Do not reroute with a
local implementation, force topk/weights, or use actual IDs as their own reference.
Observe dispatch physical schemas/ordered sizes/DP groups and native preallocated
combine as5.3, plus the two reached TP reductions. Existing output/scope observers
remain transparent; count any real native execution, not configured root counts.

New rank reports keep `layers` but entries are keyed by `(moe_input_id, layer_id)`;
validator requires exactly the prepared ID set times layers3/88 on all16 ranks,
each source physical/live/gathered vector, actual router/context observation,
source shared facts and comparison status. No unconditional `len(layers)==2`
for this new branch. Legacy report validation retains the two-entry rule.
Probe scope is explicitly `native_moe_layer`, **eager**, no timing rows, no outer
text-model or model-runner verification, no PW/FULL replay proof. A q1-shaped
eager module sample is not a captured native observation. Bound collection uses
only retained P6128+dummy, tail16+dummy, padded-q1 and owner-swapped source IDs
(at most six inputs/twelve layer invocations per rank); no new shape campaign.
Native T2 still supplies independent graph/binding/backend evidence; serving and
E2E scopes stay separately mandatory. This probe does not validate every
calibration modality or stand in for actual native per-sample count receipts.

**Affected checks:** extend existing `test_step5_dispatch_probe` plan, observed
schema and CpuCoordinator/transparent-observer fixtures for bound source inputs,
masked idle lane, original result/tensor identity, native context cleanup and
wrong/missing strategy/callable/source binding. Mock dependencies in CPU checks,
never mark them native source proof. Existing package tests inspect the real
prepared job, effective child env and probe_command (including legacy exact
argv), hash/model/reference rejection and no submit side effects. Extend the
same `fixture_result` for valid new `(input,layer)` receipts and missing/wrong
binding or layer/input keys yielding UNVERIFIED. Candidate exporter checks those
existing validation/source receipts before admitting calibration rows, reports
all rejection reasons, and remains candidate-only. This is the final producer
closure; the finite numerical model in3.2 is not reopened.

## 6. Proposed implementation paths and added/deleted ceilings

These are **additional pricing hunks**, not measured diffs or implementation
approval. Paths are repository-relative. Shared files must be unioned once with
other owners by actual nonoverlapping hunks; do not simply add whole-file totals.
No README, new backend/controller, native source edit, launcher, ratio-model
replacement or immutable historical plan edit is proposed. The finite model has
explicit submodule paths. The old parent-corrected **37 paths +1802/-185
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
| `frontier/execution_time_predictor/moe_dataset_training.py` | 25 /4 | Pricing native capacity/prepare feature columns and pairing at actual fit; CORE owns selected binding/pre-loader/helper seam, not a second selector |
| `frontier/execution_time_predictor/prediction_family_trainers.py` | 25 /4 | Same numerical schemas/paired targets; CORE supplies per-op binding facts before loader/helper calls, preserve ratios |
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

### 6.Model — complete finite algorithm, local calibration, source/config/tests

| Exact file | Additional + / - ceiling | Complete model hunk |
| --- | ---: | --- |
| `frontier/cc_backend/backends/collective_sim_cc_backend.py` | 120 /12 | One joint plan invocation; load/qualify contextual local parameters from existing profiling_data_dir; pass configured models/rule/local input; full cache and TP8 group costs, preserve scalars |
| `frontier/cc_backend/backends/collective-sim/python/collective_sim_core/schema.py` | 100 /8 | Lossless plan plus local parameters, strict device/context/range/model/channel validation and serialization; existing combine field, no duplicate native op registry |
| `frontier/cc_backend/backends/collective-sim/htsim_runner.py` | 185 /8 | Actual bounded DP2 tensor/root stage generator with exact element stripe remainders, group-local send barriers/all rails, existing parser/writer/result proof; no C++ change |
| `frontier/cc_backend/backends/collective-sim/python/collective_sim_core/predictor.py` | 65 /8 | Plan network+max-rank local+effective protocol sum; explicit assumption breakdown; finite local-only TP8 branch and old scalar rules preserved |
| `frontier/cc_backend/backends/collective-sim/python/collective_sim_core/intra_server_model.py` | 120 /8 | Exact DP copy/reduce proxy formulas/coefficients and per-rank critical path; reached context-calibrated TP8 native approximation, no g=1 total zero |
| `frontier/cc_backend/backends/collective-sim/tests/test_grouped_plan_input.py` (new) | 110 /0 | Serialize/generate actual unequal/uniform flows, exact tiny FP8/scale stripes, group-local barriers/all groups and numerical local/overlap/TP8 tests; legacy runner smoke reused |
| `tests/unit/test_collective_sim_cc_backend.py` (new) | 100 /0 | Real plan-to-predictor invocation/cache, contextual parameters/quality/range errors, actual sum/no-zero/TP8 projection, legacy scalar identity |
| `frontier/profiling/collectives/vllm_collective_benchmark.py` (Common path) | 140 /0 | Separate bounded local copy/add producer, same queued timing/graph/preallocated buffers/per-rank collection; joint plan support and raw local/native evidence without invented counts; combined +200/-14 |
| `tests/integration/step5_comm/export_costs.py` | 230 /5 | Read exact into-output/local/native observations and frozen simulator config, nonnegative affine copy/reduce/protocol and effective TP fit, residuals/context/ranges/candidate artifacts; no full-cost lookup or auto-admission |
| `tests/integration/step5_comm/prepare.py` | 18 /0 | Propagate scoped local/model grid and hash frozen simulator config, expected native/local evidence counts; reuse worker/launcher/source receipts |
| `tests/integration/step5_comm/step5_position_model_run.json` (new) | 70 /0 | Finite local/calibration/qualification definition using existing TP8DP2 resources/grid/runtime inputs, graph eligibility and frozen selected config; not a new platform path |
| `tests/unit/test_step5_comm_cost_export.py` (Common path) | 65 /0 | Identifiable affine/TP fits, nonnegative constraints, no fake residual clipping, held-out error and candidate/range/source rejection; combined +105/-0 |
| `tests/unit/test_collective_benchmark_helpers.py` (Common path) | 55 /0 | Real local copy/reduce scope construction, finite config/rows, preallocated graph/per-rank evidence and native count separation; combined +115/-0 |
| `tests/unit/test_step5_comm_package.py` | 30 /0 | Real prepare forwards model/local definition and config hash/expected counts; historical definitions unchanged |
| `tests/integration/step5_comm/analyze_result.py` | 8 /8 | New into-output endpoint semantics remain distinct from return-output entry; preserve source scope and byte vectors, no accidental old op interpretation |

**Model increment: 15 hunks /15 paths, 12 paths new to Common;
+1416/-57 =1473 gross lines. Common+Model before producer closure:54 unique paths;
+3518/-294 =3812 gross changed-line ceiling.** Arithmetic partitions:
backend+submodule algorithm/config 5 paths +590/-44; affected model tests 2 paths
+210/-0; local/native producer/export/definition 4 paths +458/-5; shared test
increments 2 paths +120/-0; package test +30/-0; endpoint analysis +8/-8.
This table bounds the finite algorithm and calibration work. Its previous claim
of complete producer transport missed validator and actual worker/probe bindings;
section6.Closure supplies those necessary hunks. The generated `collective_local.json` is a measured
artifact, not another source file; its numeric values cannot be supplied now.

### 6.Closure — actual planner/validator and worker/probe increments

These are **additional** to Common+Model, not a second charge for their listed
behavior. The six files new to the pricing table are marked; `inputs.py` is
core-owned and already belongs to the parent's core union. Source/core/native
authors retain their existing ownership; parent assigns the shared eight-line
extraction before implementation. All remaining changes are pricing collection
entrypoint consumers, not changes to the source verifier or native19/T2.

| Exact file | Additional + / - ceiling | Actual closure hunk / resulting pricing allowance |
| --- | ---: | --- |
| `frontier/profiling/collectives/vllm_forward_collectives.py` | 24 /0 | Keyword-only selection at the shared Step early-return branch; exact order/name validation and legacy default. Combined pricing32/2. |
| `frontier/profiling/collectives/vllm_collective_benchmark.py` | 16 /2 | Call shared selected planner, pre/post same-process source verifier and attach bound native facts to existing evidence; worker owns inherited child env. Combined216/16. |
| `tests/integration/step5_comm/prepare.py` | 60 /3 | Shared planner/counts; single bound manifest/input IDs/environment/source/receipt/config copy and hashes, adapter-derived image, per-input source vectors. Combined78/3. |
| `tests/integration/step5_comm/validate_result.py` **new to pricing** | 65 /6 | Shared selector and exact expected keys, safe mismatch reporting; bound native input×layer/source/env/router/context evidence; legacy two-layer rule retained. |
| `tests/unit/test_step5_comm_transport.py` **new to pricing** | 105 /0 | Reuse fixture_result: selected subset passes, absent selected call fails despite changed superficial counts, duplicate/extra keys and bound native input/layer/source/env controls; no new fixture controller. |
| `tests/integration/step5_comm/worker.py` **new to pricing** | 65 /3 | Binding before native imports, one effective child environment, actual probe command/input vector/topology branch and source/runtime manifest receipt; unconditional validator remains. |
| `frontier/profiling/collectives/step5_dispatch_probe.py` **new to pricing** | 160 /10 | Selected generation args/input plan and per-input loop; actual native setter/mask and router callable checks; transparent routing observation/weight-mask comparison; truthful eager-module input×layer receipts; legacy entrypoint untouched when unbound. |
| `frontier/profiling/moe/inputs.py` **new to pricing, core-owned new file** | 8 /0 | Expose the core-planned source-frame tensor materialization for reuse, no duplicate interval/count/tensor algorithm. Additive to core allowance only if not already subsumed by its same hunk. |
| `frontier/profiling/moe/moe_impl.py` (core/source shared path) | 4 /4 | Core adapter calls the one extracted source_frame_tensors implementation, rather than retaining an inline second materialization. Combined pricing29/12, separate from native timer/source callable checks. |
| `tests/integration/step5_comm/step5_position_model_run.json` | 8 /0 | Bound manifest/input IDs/actual native_environment fields and eager probe scope alongside existing finite grid. Combined78/0; actual generated digests/IDs come later, not invented constants. |
| `tests/integration/step5_comm/export_costs.py` | 12 /0 | Consume selected validation and actual benchmark/probe shared-source/binding receipts as calibration prerequisite, retain reasons and candidate status. Combined242/5. |
| `tests/unit/test_step5_comm_package.py` | 70 /0 | Actual prepared selected counts/image/hashes/child env/probe argv, missing/wrong binding controls and exact legacy command/default preservation. Combined100/0. |
| `tests/unit/test_collective_benchmark_helpers.py` | 15 /0 | Actual list-cases/shared-planner consumption and bound verifier evidence, no caller-local selector or foreign-source acceptance. Combined130/0. |
| `tests/unit/test_step5_dispatch_probe.py` **new to pricing** | 96 /1 | Existing plan/observer/context fixtures: explicit source input, original call/result, mask/weight contract, missing/wrong binding/router/source and legacy controls; no copied T2 graph suite. |

**Closure:14 paths/hunks, six new to pricing; +708/-29=737 gross.**
Arithmetic by purpose: planner/benchmark/prepare24+16+60=100 additions/5 deletions;
validator/transport65+105=170/6; worker/probe/inputs65+160+8=233/13;
definition/export8+12=20/0; package/helper/probe tests70+15+96=181/1;
shared core materialization connector4/4.
100+170+233+20+181+4=708,5+6+13+0+1+4=29.

**Current complete Common+Model+Closure:60 distinct pricing paths;
+4226/-323=4549 gross changed-line ceiling.** 54+14-8=60 unique paths;
3518+708=4226 added,294+29=323 deleted,3812+737=4549 gross.
Native19/T2, the source16 and core33 remain companion scopes, not included as
whole-file totals here. Parent's final union reconciles already-listed
overlapping paths and nonoverlapping hunks, especially core inputs.py, once;
this pricing number is not the final joint D29 budget or authorization.

No `cc_backend_config.py` edit/new file selector is needed: existing
`profiling_data_dir`, algorithm/channel/topology/network fields suffice. The
selected artifact's declared sum rule is passed through the submodule's existing
`intra_server_combine_rule`; no new overlap toggle is created. No edits to device
SKU/HBM inventory, scenario profile defaults, C++ triggers/network transport,
__init__ registry or StepMind launcher are required. The existing submodule
gitlink will need its recorded commit update during approved integration; it is
one repository pointer operation, not an omitted algorithm source hunk.

Ceilings count pricing-owned code/test/config hunks, including the prior common
interfaces. Parent still unions them with current core/native/profile-source
work by actual hunk. Core supplies its accepted sole registry binding bundle;
constructor/pre-loader/helper/timer-switch corrections are core-owned and are
not duplicated here. If some listed common consumer hunks are subsumed by core,
parent reconciles them by actual diff, not by dropping mandatory behavior.

The old **50-path/2782 known subtotal** and **46-path/2595 lookup** estimate are
superseded planning history. Lookup is unapproved and not included in current
4549. There is just one finite viable model proposal here, not two permanent
numerical routes or a known-subtotal disguised as a full algorithm ceiling.

Table arithmetic in this correction is document arithmetic; no command/test was
run. The prior `python3` check concerned the superseded tables only. Current
ceilings and sums are planning estimates, not measured diffs. Parent still owns
native/core/profile-source reconciliation, document/module-size exceptions and
the final user scope request. The source owner's provenance/launcher hunks and
core owner's adapter/frame-row/schema/cache/coverage hunks are not removed by
these additional pricing estimates.
No tests or native-equivalence verdict were produced. The complete model ceiling
does not imply calibration values, resource/run approval, native accuracy or
E2E acceptance.
Manual arithmetic: Common42 paths/+2102/-237 plus Model15 hunks/+1416/-57;
three Model paths already occur in Common, so42+15-3=54 unique paths. Closure14
adds six new pricing paths; total60 paths/+4226/-323/gross4549 as itemized above.
These counts include the new finite JSON definition and every specified producer/model/config/test hunk; measured data
volume, parent documentation and a gitlink pointer update are not source lines.

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
4. Exercise the actual finite stage generator and predictor through the existing
   runner: exact element/byte sums (including tiny scale/ID stripes), per-group
   barriers/directions/root order, all rails in one matrix; sum=N+L+P, no g=1
   local zero and context/range/quality/config rejection. Model flow counts are
   not native kernel counts. Run the actual bounded fit/export on synthetic
   known coefficients and deliberately incompatible residuals, showing no
   negative clipping/default parameters or direct full-cost lookup. Old scalar
   APIs, four ratio-routing policies and DSV4 remain unchanged.
5. Required CPU fixture and paired actual-main offline/online run retain
   6144/1024 request, P6128/D16+1023q1 census, native capacity/READ ordering,
   all16 expert participants, source/live/physical conservation and positive
   included costs. OperationMetrics/trace/ledger run together through actual
   reporting and emitted gathered metadata. Synthetic compute/count/algorithm
   evidence is marked `cpu_fixture`, never native admission. The fixture's finite
   network/local/protocol parameters reach the same real model/production path,
   not backend-class patches or a full-cost lookup. Native parameters/counts and
   numerical identity remain separate from structural synthetic proof.
6. Selected-call/package checks use the existing fixture_result and real prepare,
   benchmark list-cases and validator: the same proper subset passes, missing
   selected evidence fails even if superficial expected counts are changed, and
   duplicate/extra/unknown keys cannot pass. Exercise actual worker probe_command
   and binding-before-import env path with existing package fixtures; valid bound
   source IDs/context/mask and `(input,layer)` report keys pass, missing/wrong
   strategy/source/callable/input/mask/receipt fails without legacy relabeling.
   Old S0/S1/S2, exact old command/default fields and two-layer validator controls
   remain. Native observer failures remain advisory; native exceptions propagate.
7. For native acceptance, retain fresh derived-image TP8/DP2/no-SP full-model
   source/collective/alias/registration receipts, bounded model calibration,
   independent fresh native operator qualification and signed prediction errors,
   native scope durations/kernel counts and count/sentinel/padding agreement.
   Complete numerical inputs and independently referenced E2E evidence are required.
   Each of client TTFT, TPOT, total-token THPT and per-request E2E must satisfy
   `abs(predicted-actual)/abs(actual)<=0.10` independently; report actual/predicted
   values, absolute/relative errors and retained repetition protocol. A zero
   native value makes relative error undefined, not a passing zero substitute.
   old module probes or an actual-main CPU PASS cannot establish E2E accuracy.

Affected future command, after implementation and core/native/source integration:

```bash
python -m pytest -q tests/unit/test_step5_moe_pricing.py tests/unit/test_comm_operator_families.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_moe_profiling.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_step5_comm_package.py tests/unit/test_step5_comm_transport.py tests/unit/test_step5_dispatch_probe.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_contract_reuse.py tests/integration/test_step5_measured_main.py
```

Then use the existing fixture generator and `run_step5_measured_main.py
--fixture-manifest <fresh-position-manifest> --output-root <fresh-output> --mode all`.
Record exact environment/commands/values and limits in the parent's test report.
Also run the scoped submodule `tests/test_grouped_plan_input.py` for the actual
finite generator/local formulas and the existing real runner smoke, inspecting
the returned matrix/makespan and complete flow finish count. No C++ behavior is
changed. These are future affected checks, not tests run in this delegation. A mathematical
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
- Core schema/coverage: current core accepted `get_moe_profiling_operators(context)`
  in `design.md:1238-1244`; pricing owns its registry/precision/numerical hunks.
  Core owns constructor policy resolution, bound per-op facts before trainers'
  loaders/helpers and its local timer-method restoration; no duplicate selector.
  Its new-mode target/frame rules replace unconditional four-target coverage.
  One inclusive expert root owns movement, with no charged shuffle child. Current
  generic Triton padded tuple must not be asserted universal for DEEPGEMM.
- Profile-source owner: stage the same native source binding for these added
   scopes and collector receipts; no second image/adapter selector. Existing
   launcher ownership does not transfer here.
- Producer closure: pricing consumes the source verifier/manifest and core
   parsed inputs in the actual communication worker, benchmark, probe and validator
   (5.4-5.5/6.Closure). Core-shared inputs.py/moe_impl.py materialization extraction
   is assigned once by parent; no shared_scope.md edit here. Native19/T2's no-old-
   probe-edit allowance remains its own scope; pricing's newly included use now
   explicitly budgets that entrypoint's selected branch and existing tests.
- Numerical design: retain collective_sim; parent/YC decides the explicit
   scheduling/channel/local-proxy/protocol-remainder/no-overlap assumptions in3.2
   with the complete4549-line pricing ceiling. Lookup remains unapproved. Finite parameter
  collection/qualification follows implementation/run approval and does not
  require exact NCCL microarchitecture before a complete scope decision.
  Missing actual inputs/validation still prevents runtime/E2E completion.

Not settled by inspection: fresh derived-image actual expert branch at22 local
experts; per-frame native capacity/alignment and zero-live included kernel
behavior; actual grouped NCCL kernel multiplicities/scope/duration; fresh derived
runtime symm/registration/transport scope; local/effective model parameter
values; and formal
cost quality/admission. The current/default ordinary into-output branch is
settled by source and retained receipts. Exact NCCL algorithm/channel/fusion
discovery is **not** a model-admission prerequisite; current source does not
establish that this extra detail is necessary. The unresolved decision is whether
to adopt the one explicit finite assumption set; exact unprovided numeric
inputs and their bounded producer are in3.2.5. Target interfaces are already
accepted at design level. Any switch to measured lookup still requires a
separate changed-CC decision. No broader policy math,
PD-AF/MTP/SP, native CPU-gap or hypothetical symm campaign is requested.

## 9. Current design checkpoint — D29-FINITE-PRODUCER-CLOSURE

### 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 2｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| shared selected-call planner/validator 设计 | PASS | 5.4覆盖真实早返回/三caller及fixture_result正反控制。 |
| 实际 worker/probe source/positions/mask 设计 | PASS | 5.5显式native setter、eager scope、有效binding及legacy分支。 |
| finite model保持及complete delta预算 | PASS | 6.Closure14 hunks/+708/-29；总60 paths/4549 gross，模型未重开。 |
| 数值参数/native/E2E资格 | BLOCKED | 参数未提供；有明确 bounded producer，设计不能替代 native comparison。 |
| 源码、测试、GPU/platform 执行 | SKIPPED | 本次只读source、编辑唯一设计文件，不运行shell/tests/agents。 |
| Parent 假设决定/union/scope 批准 | TODO | ordered-flow/local proxy/protocol remainder/sum 假设未被本设计自动批准。 |

### 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** D29-FINITE-PRODUCER-CLOSURE 至今｜**当前前沿：** producer两seams已写入owned doc，parent union/批准。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 仅补两个实际consumer缺口；复用fixture_result和probe，不新增controller。 |
| D2 细节反复重试 | CLEAR | 无runtime/参数重试；patch定位错误已从文件实读修正，未绕开设计缺口。 |
| D3 支线问题修补 | CLEAR | 两seams由parent明确指派；不重开已settled numerical model。 |
| D4 可并行任务停滞 | CLEAR | 明确只有本文件 ownership，禁止 agents；parent 负责 companions。 |
| D5 超范围镀金 | CLEAR | 不增加 generic controller、policy campaign 或未到达 symm branch。 |
| D6 分析空转 | CLEAR | 有source trace、具体接口、逐hunk预算；停止于bounded closure。 |
| D7 路线违背 | CLEAR | 当前建议保留 simulation；lookup 明确 unapproved，不利用类名改 numerical source。 |

**文档检查：** S2/S5：旧3.2.4/6.Model称producer完整，却遗漏validator的未过滤plan和worker248的无positions/mask probe；5.4-5.5/6.Closure已补ownership/接口/预算。S1/S3/S4/S6未发现新增缺陷；父任务联合hunk须由parent去重。
**归因：** 当前窗口无 HIT；parent指派的旧文档缺口已修正。

### 3. 纠正建议

**分支：** B：旧 pricing doc 存在缺陷，先纠正文档再继续。

**文档更新（先执行）：**
1. `pricing_design.md`5.4/5.5/6.Closure：已补两个actual producer seams、existing fixture checks和全部delta（可直接执行，已完成）。
2. Parent在shared_scope.md去重core inputs/moe_impl extraction及pricing worker/probe新增ownership，再取得联合scope决定（需用户确认；本设计不授权新增code）。
**需停止的偏离行为：** 无。

**接下去的路线：**
1. Parent完成union/假设决定：60 paths/4549 pricing ceiling与core/native/source按实际hunks去重，明确全部数值假设/producer scope。
2. 批准后implementation与bounded参数收集：实际stage generator/local fit/native receipts通过affected checks，未提供数值不能默认补零。
3. actual-main加独立native/E2E qualification：四项误差各<=10%；失败先定位signed errors，再决定是否需要更细模型，不能选counts-only。

### 4. 最值得关注的 3 点

1. 同一shared planner决定selected keys；缺selected call不能靠更改expected数量通过。
2. worker实际probe显式bind native positions/mask和derived source；只证明eager module，不冒充T2/serving graph。
3. 完整delta737 gross、总4549；模型未重开，native数值/四E2E<=10%资格仍须后续验证。
