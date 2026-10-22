# D29 position_cyclic — Frontier and profiling design

TaskID: D29-CORE-SEAM-CORRECTION (continuation of D29-CORE-PRICING-INTERFACE,
D29-FRONTIER-DESIGN-CORRECTION and
D29-FRONTIER-POSITION-DESIGN; parent owns tree, commit and GPU work).

This correction rechecks the decisive live interfaces after the parent tracked
the first design at `0447681`. Line references are inspected source, not proposed
code. Recheck them against the integrated tree before implementation. Only this
document is edited; no child code step, test run or commit exists.

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Close b8be287 core delta: constructor resolves count registry before parent init; both fit paths use selected binding facts before loader/helper calls; locally switch/restore singleton RECORD_FUNCTION replay and require actual DeviceTimer annotation/parser tests. Accept pricing's sole get_moe_profiling_operators(context) selector and conditional MOE_FAMILY prepare; keep33 files, budget the missing hunks. |
| 2026-10-22 | Reconcile native applicable scopes: registry-derived gate/router plus reached source FP8 prepare and one gathered inclusive expert root; no separate shuffle target. Consume pricing-owned ROUTED quant/layout/context contracts; remove duplicate native invocation/identity hunks from CORE. Preserve collective_sim; measured CC variant unselected. |
| 2026-10-22 | Reconcile delivered profile-source design: CORE owns the actual native timer adapter; define bounded per-frame cells/CSV rows and shared admission; count kernel-input/identity seams and tests. Source S2 wholly other-owned; pricing union still pending. |
| 2026-10-22 | Address P1/P2: mandatory actual-main/pricing/launcher; strict source input; executable registry; immutable lane-aware cache; shared schema; paired exact Step coverage; separate source/gathered frames. Defer new-policy PD-AF/MTP/legacy expansion. |
| 2026-10-22 | Re-inspected the same-task untracked draft after predecessor failure; corrected producer seams, empty-load pricing, collective limitations, fixture consequences and gross line budget. Design only; parent owns Git. |

## 0. Approved decision and scope of this document

D29 approval (`task_memory/task_2026-10-15_step5_preview_h800/requirements.md`,
"D29 position policy approval"): canonical policy id `position_cyclic`,
`expert_id = (position * top_k + slot) % num_experts`, no per-wave reset, no
per-request phase and no per-layer rotation. Selected weights, renormalization
and scaling, and the native padding mask stay native behaviour. Both sides
count real logical position intervals; physical rows and dummy padding are
reported separately from valid expert routes. Same-position requests may
concentrate on the same experts.

Accepted serving scope: **Step P/D, PP1, attention TP8/DP2, MoE TP1/EP16,
no SP and no MTP**, per this task's `requirements.md:14-26`. Actual-main fixture
acceptance is mandatory. Mathematics can have broader domain tests, but they do
not admit broader serving modes. Old policies/generations remain unchanged.

This document owns Frontier routing descriptors, the actual importable native
router timer adapter in `moe_impl.py`, existing MoE profiler inputs/rows/metadata,
schema/coverage selection, and the CPU actual-main fixture.
Mandatory other-owned scopes, required before the combined scope decision:

- `pricing_design.md`: applicable registry/ROUTED quant contracts, original native
  prepare/inclusive expert invocation/layout/identity, source-local pricing,
  gathered physical sizes, phase/reporting and native DP collective plans/tests. This is
  required acceptance work, not an optional companion or counts-only alternative.
- `profile_source_design.md`: selection of the native-design runtime adapter/image,
  launcher/staging/pins/source binding and their tests. It does not implement the
  Frontier native router timer adapter. In particular `frontier/profiling/step5/job.py`
  belongs there; an 8-line routing flag edit cannot complete its launcher scope.
- Native router/context design: serving tensor router, selected weights/mask,
  graph binding and immutable derived generation.

Profile-source delivery has been read: its sections 2/4 and S1–S13 assign the
`moe_impl.py` factory/context/call seam to CORE and supply the binding interface
described in 6.4. Pricing's D29-PRICING-CORRECTION is now delivered; its section5.1
selection/conditional MOE_FAMILY prepare contract is accepted at the interface
in6.2.1. Its known collective_sim interfaces still lack a selected full-duration
model/complete ceiling; those remain pricing/parent work, not a CORE modeling task.
The subsequent parent plan:33-40 removes exact per-channel reconstruction as a
pre-code prerequisite and asks that owner for finite explicit model assumptions.
CORE neither reinstates that rejected gate nor solves those model alternatives.
Prior requirements retain simulation-based collective_sim. The measured lookup
variant is not approved; D30's separate7/750 approval does not authorize D29.
Sections 4.6 and 8 specify the remaining frame/kernel reconciliation without
duplicating the other owners' investigations. No implementation or GPU run exists.

Out of scope by instruction: native source edits, image/package publication,
GPU/platform/auth/cloud/SSH, CPU-overhead or wall-time work, new agents.

## 1. Why the existing representation cannot express the policy

Current Frontier routing is a per-expert *ratio table* plus one Hamilton pass
per EP wave:

- `frontier/moe_ep_workload.py:27` `generate_moe_routing_ratios` (equal
  `1/E` for `balanced`).
- `frontier/moe_ep_workload.py:774` `materialize_expert_token_counts`
  (floor quota, remainder to ascending expert IDs).
- `frontier/moe_ep_workload.py:810` `materialize_layer_ep_workload`
  (`total_routed_assignments = routing_token_count * router_topk`, then the
  EP ownership split).

That pipeline is a function of a single global scalar `N = routing_token_count`
and the ratio table. `position_cyclic` is a function of the *multiset of logical
token positions*, so no ratio vector and no scalar `N` can reconstruct it.
Two waves with the same `N` but different request position ranges have
different counts, and `(p*K+j) mod E` is not reproducible from any 1/E table.

Second gap: `routing_token_count` is today simultaneously the physical compute
width and the count of routed tokens. `position_cyclic` plus the native padding
mask separates these: `live_rows <= physical_rows`.

## 2. Policy mathematics (exact, closed form)

For one unmasked token at logical position `p`, slot `j in [0, K)`:
`e = (p*K + j) mod E`. For consecutive positions the slot ranges are
contiguous (`p*K+K-1 = (p+1)*K-1`), so the assignment stream of a contiguous
position interval `[a, a+L)` of length `L` is the `L*K` consecutive residues
starting at `r0 = (a*K) mod E`.

Per interval with `A = length * K`:

```
base, rem = divmod(A, E)
counts[e] += base                for every e in [0, E)
counts[(r0 + t) mod E] += 1      for t in [0, rem)
```

Sum over the request intervals of the wave. The direct implementation costs
`O(#intervals * E)`, not `O(#intervals + E)`. A residue difference array could
reduce this, but is unnecessary here. Counts are layer-independent; reuse the
existing bounded workload cache with interval identity rather than add a second
cache merely to promise once-per-wave computation.

Properties that follow and that the tests must pin:

- Conservation: `sum(counts) == live_rows * K`.
- Single interval: `max - min <= 1`; contiguous `E/gcd(K,E)` positions cover all
  experts exactly. For Step (`E=352, K=8`) that is 44 positions.
- Multi-interval waves need not be balanced (accepted by YC).
- Per-token distinctness holds for `K <= E` (`j` distinct mod `E`).
- Union/merge/split invariance: counts depend only on the interval multiset, so
  P/D split, prefix chunking, prefix-cache offsets, graph padding and dummy
  lanes cannot move another request's routing.
- Padding invariance: adding dummy/padding rows adds no assignments.

Worked contract case from the approved proposal (keep as a test):
positions `6128..6143` give IDs starting at `6128*8 mod 352 = 96`
(`96..103`, `104..111`, ...), whereas a per-forward Hamilton reset would start
at 0. q1 continuation at 6144 gives `224..231`.

## 3. Existing producers traced (no new scheduler)

Only the shared P/D layer-wave producer gains the new policy in this scope.
Other entry points explain why new-policy fallback inference must be rejected:

| Entry | File:lines | Wave identity available | Interval source available |
| --- | --- | --- | --- |
| Shared PREFILL / DECODE layer wave (in scope) | `frontier/scheduler/utils/ep_wave.py:99-114` calls `materialize_workload(batch=aggregate_batch, ...)`; base scheduler `_prepare_moe_ep_wave_plan` | `wave_inputs.source_batches` (`ep_wave_inputs.py:11-19,97-104`) | Explicit mapping holds actual request owners; aggregate loses their intervals |
| PD-AF DECODE_FFN EP wave (new policy deferred) | `frontier/scheduler/utils/expert_parallel.py:364` `materialize_wave_workload(group, ...)` | Original group batches | No new-policy producer/queue/transfer changes here |
| Predictor fallback / monolithic skew / MTP (new policy deferred) | `frontier/execution_time_predictor/moe_routing_workload.py:392-444`, callers in `moe_operator_times.py` and `moe_mtp_replay.py` | Real batch or shallow virtual aggregate | New policy requires an explicit source descriptor; infer neither case. Old policies unchanged |

Authoritative per-request progress already exists and must be reused, not
re-derived:

- `frontier/entities/batch.py:508-520`: `Batch._num_context_tokens` is captured
  at schedule time as "vLLM's `num_computed_tokens` in the scheduler output",
  index-aligned with `Batch._num_tokens`; mirrored by the
  `num_context_tokens` property (`batch.py:790`).
- `frontier/entities/request.py:719-722` supplies the active recompute cursor
  instead of semantic processed length when D recomputation is pending.
  `:1019-1056` verifies the received prefix against the scoped P compute target
  and sets that cursor to the transferred prefix. Thus P context0/query6128
  and D context6128/query16 already describe the approved P/D split; do not use
  `num_prefill_tokens=6144` or the later request processed count as D-tail start.
- `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py:232-248`
  preserves both lists when it splits a batch into live requests.
- `frontier/execution_time_predictor/step5_cost.py:155-175` already uses
  `sequence_lengths = context + query` and
  `physical_query_rows = get_effective_total_tokens_for_compute(...)`, and
  `frontier/attention/step5.py:788-794` names the split
  `valid_tokens` / `padded_tokens`. The MoE side is missing exactly that split.

Physical rows are also already produced:

- `frontier/entities/batch.py:367` `DecodeCudaGraphMetadata.get_effective_total_tokens_for_compute`
  (`padded_total_tokens` in FULL/PIECEWISE, `original_total_tokens` otherwise);
- `frontier/entities/batch.py:818` `Batch.get_effective_total_tokens_for_compute`
  (+`forward_start_descriptor` short-circuit at 836);
- `frontier/scheduler/utils/forward_start_agreement.py:12-87`
  `bind_forward_start` writes `ForwardStartDescriptor.num_tokens` (DP-agreed,
  capture-size padded) and the matching `DecodeCudaGraphMetadata`, so
  `physical - original` is the real dummy/DP padding;
- `frontier/entities/batch.py:1340` `DummyForwardBatch`: one local token, no
  owned request; `frontier/attention/step5_inputs.py:21-27`
  `MRV2_DUMMY_ROW_IDENTITY` states `dummy_is_padding: True`,
  `dummy_owned_requests: 0`, rows are physical InputBatch rows, and
  `MRv2DummyDescriptor` (`step5_inputs.py:49`) carries the agreed width;
- `frontier/scheduler/utils/ep_wave_inputs.py:72-80` already computes the
  shared DP padding (`max(lane_compute_tokens)` when any lane runs a graph
  mode) that the aggregate width uses.

Padding rows are therefore identifiable without any new readback: a row is live
only if it belongs to a request interval; `physical - sum(intervals)` is
padding, and a dummy lane contributes zero intervals.

Frozen mask source (read locally, not a platform observation):
`/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/v1/worker/gpu/input_batch.py:127-178`
defaults `make_dummy(...,is_padding=True)` and zeros its physical positions.
`gpu/model_runner.py:2042-2054` binds `input_batch.is_padding` into the forward
context; `model_executor/layers/fused_moe/router/fused_topk_bias_router.py:28-31,56-74`
passes it to native sigmoid top-k under unchanged default
`VLLM_MOE_SKIP_PADDING=1`. Physical dummy rows are not real position0 requests.

## 4. Frontier design

### 4.1 New module `frontier/moe_routing_policy.py` (at most 200 added lines)

One registry, one math owner:

```python
# Move the plain string-valued WorkloadDistributionType enum here; preserve
# .value and the old disaggregation import. One registry owns count behavior.
@dataclass(frozen=True)
class RoutingCountPolicy:
    runtime_path: str
    requires_source: bool
    count_experts: Callable

# All four old entries use one ratio callable, lazily delegating to the existing
# moe_ep_workload.materialize_expert_token_counts. The position entry uses the
# interval callable defined below. Do not move or duplicate Hamilton math.
ROUTING_COUNT_POLICIES: Mapping[str, RoutingCountPolicy]
def resolve_routing_count_policy(distribution_type: str) -> RoutingCountPolicy: ...

@dataclass(frozen=True)
class TokenInterval:            # start >= 0, length > 0
    start: int
    length: int

@dataclass(frozen=True)
class SourceLaneRouting:
    lane_id: int                # DP request owner, not an EP/TP copy
    source_batch_id: int
    source_global_id: int
    forward_group_id: int
    graph_mode: str
    gating_runtime_context: str  # pricing-owned source predicate/freeze hunk
    physical_rows: int          # source-local native router frame
    live_rows: int
    intervals: tuple[TokenInterval, ...]

@dataclass(frozen=True)
class WaveRoutingSource:
    policy_id: str
    source_lanes: tuple[SourceLaneRouting, ...]  # sorted source lane IDs
    gathered_physical_rows: int  # expert frame, not local router width

def build_wave_routing_source(source_batches, *, policy_id, cluster_type,
                              gathered_physical_rows) -> WaveRoutingSource: ...
def position_cyclic_expert_counts(intervals, top_k, total_expert_num) -> dict[int, int]: ...
```

`build_wave_routing_source` reads only existing metadata. Batches must be actual
request owners, never virtual aggregate or EP assignment batches. Keep count
mathematics free of scheduler state; defer the `Batch` import inside extraction
if needed to avoid an entities import cycle.

- `gathered_physical_rows` is the supplied aggregate compute width. Local rows
  come from each bound `ForwardStartDescriptor.num_tokens`
  (`forward_start_agreement.py:52-80`); those descriptors are mandatory for the
  new measured mode. Validate gathered width against their sum in the accepted
  no-SP scope. Do not derive local widths from aggregate/N, divide by EP16 or
  reproduce graph agreement. Unagreed legacy graphs remain old-policy paths;
- copy lane ID, source batch/global ID, forward group/mode, physical/live rows
  and intervals to frozen values. No mutable Batch/Request reference enters
  the descriptor/cache. The source mapping is the admitted scheduler mapping;
- `intervals` from `zip(batch.num_context_tokens, batch.num_tokens)`, skipping
  `DummyForwardBatch`; no request-id phase and no independently tracked position;
- validation: per real lane `len(requests) == len(num_tokens) == len(num_context_tokens)` and
  `sum(num_tokens) == batch.total_num_tokens` (live scheduled rows),
  `sum(interval.length) == live_rows <= local physical_rows`. Known dummy lanes give
  no intervals/live0, not their fake physical query list. All forward groups
  match the admitted wave. Extract only from the explicit source mapping,
  never a virtual lane's `batch.source_batches` or copied request/context lists.

No policy id is invented at this seam: the caller supplies the configured
policy, and the materializer rejects a source whose `policy_id` is not the one
the predictor was constructed with. The shared materializer invokes the
registry-owned `policy.count_experts(...)` once, then keeps its existing EP split.
Callers consult `requires_source`, not a chain of policy-name branches. The four
ratio entries delegate to the current Hamilton owner through one adapter. An
old public ratio-only materializer call uses the shared ratio behavior without
inventing a `balanced` policy identity. A new count policy requires one registry
entry/callable, not new count dispatch in every producer.

Count callable contract: keyword inputs `routing_ratios`, `routing_source`,
`routing_token_count`, `router_topk`, `total_expert_num`; result is the complete
logical-expert count map. The ratio adapter uses the old Hamilton owner with
physical_count*K. The position callable requires source and uses its live
intervals, never substituting physical rows for missing source. Only the shared
materializer invokes the callable and checks live_count*K before its EP split.
The count registry does not own a second list of compute targets or quantization
applicability. Section 6.2.1 derives those from existing operator registry/bindings,
with pricing-owned native applicability metadata. Old row filtering retains its
existing behavior rather than invent legacy frame labels.

### 4.2 Descriptor: physical rows vs live routes

`frontier/moe_ep_workload.py:134` `LayerEPWorkload` gains
`live_routing_token_count: int | None = None` (defaulting to
`routing_token_count`) and:

- `routing_token_count` keeps its shared-wave meaning: physical compute rows
  (existing tests at `tests/unit/test_moe_routing_conservation.py:121-175`
  assert exactly this and stay green);
- `padding_row_count = routing_token_count - live_routing_token_count >= 0`;
- invariant change: `total_routed_assignments == live_routing_token_count * router_topk`
  (today `routing_token_count * router_topk`); for all four ratio policies
  `live == physical`, so nothing changes;
- `EPLaneWorkload` (`moe_ep_workload.py:374`) keeps its count fields unchanged: its
  `routed_token_count`, `local_token_counts` and `global_token_counts` are live
  assignment counts and stay the routed-compute domain. Add optional
  `routing_source: WaveRoutingSource` to the parent and canonical lanes: timing
  consumers must retain the position-policy identity and both physical frames after
  lane construction. Do not infer these from artificial expert Requests.

`materialize_layer_ep_workload(..., policy_id=None, routing_ratios=None, routing_source=None)`
(`moe_ep_workload.py:810`): exactly one source is required.

Configured callers supply `policy_id`; source identity must agree with it.
Legacy direct calls supplying ratios but no policy id use the one shared ratio
behavior, without attaching a guessed distribution label. The materializer
resolves that behavior or the configured registry entry; it alone calls the
count callable. A missing source cannot invoke the ratio behavior for a
source-required policy.

- ratio path: unchanged code and numbers;
- position path: use the registry count callable on source intervals; validate
  `source.gathered_physical_rows == routing_token_count`, `live = sum(lane.live_rows)`;
  the EP ownership split and every existing conservation check are reused
  unchanged; a zero-live lane yields an all-zero lane that remains a valid EP
  participant (`tests/unit/test_moe_ep_aggregate_admission.py:570` already
  pins that).

### 4.3 Which consumer uses which frame/count

| Consumer | Count | Location | Change |
| --- | --- | --- | --- |
| `moe_gating_linear`, `moe_gating_routing_topk` lookup | source-local physical rows; router also live rows | `moe_operator_times.py:387-435` | mandatory pricing owner supplies source frame, not aggregate EP batch |
| inclusive routed-expert features | gathered physical rows, live assignments and selected native padding/capacity | `moe_routing_workload.py:577-604`, `frontier/moe_load_imbalance.py:33-56` | one shared selected-schema classifier; pricing supplies actual layout/features, not universal global Triton padding |
| grouped-GEMM `BLOCK_SIZE_M` when applicable | gathered expert physical frame | `block_size_m_for_tokens(model ranges, pre-routing tokens)` | ranges only for the selected native schema that uses BLOCK_SIZE_M |
| `moe_shuffling` | no separate position-policy charge | `moe_operator_times.py:447-500` | native movement belongs to the inclusive expert root; legacy shuffle stays unchanged, no zero row |
| source prepare when applicable | source physical rows and selected nondeferred FP8 quant contract | pricing-owned ROUTED operator binding | CORE consumes its registered target/features and paired exact keys; BF16/deferred structural absence is not a zero measurement |
| EP dispatch/combine payload | native DP physical sizes vector | `frontier/operators/families.py:540-555`, `moe_operator_times.py:244-282,512-585` | mandatory pricing scope (4.6); no assignment-cost fallback |
| EP-wave conservation trace | both, reported | `frontier/scheduler/utils/ep_trace.py` via `ep_wave.py:117-126` | add `live_routing_token_count`, `padding_row_count` |
| request-level MoE time | max over lanes | `moe_routing_workload.py:620-660` | none |

TP is not a factor anywhere: the source is built once per replica over the DP
lane set, counts are per EP domain, and `attn_tensor_parallel_size` /
`moe_tensor_parallel_size` never scale token counts. A test asserts identical
counts at TP 1 and TP 8.

### 4.4 Seams to thread the source (no new scheduler)

1. `frontier/scheduler/utils/ep_wave.py:109-114` passes
   `source_batches=wave_inputs.source_batches` to its existing
   materializer. `EPWaveInputs` stays unchanged: it already retains request
   owners and physical padding.
2. `base_cluster_scheduler.py:899-915` threads the optional source argument to
   `layer_workload.py:13-67`. Ratio policies do not inspect new metadata.
3. `layer_workload.py:52-67` resolves the configured policy, extracts intervals
   from real source lanes plus the existing aggregate width, and calls the
   shared materializer. Branch before `resolve_routing_details`; there is no
   ratio table for the position policy.
4. No new-policy source threading through PD-AF `expert_parallel.py:364-417`
   or base `:679-709`. Role initialization rejects the new policy for PD-AF,
   MTP and unsupported fallback modes before empty ratio-table presence guards
   can seem to admit them. Old policies/producers/transfer ownership unchanged.

Predictor boundary (`frontier/execution_time_predictor/moe_routing_workload.py:392`):
`_materialize_layer_ep_workload(batch, cluster_type, layer_id, routing_source=None)`
accepts the new mode only with an explicit frozen source. Missing source fails
before ratio resolution/cache lookup for **all** batch types, including real
single-request and dummy batches. It never reconstructs intervals from `batch`
or shallow-copied metadata. Old ratio fallback stays unchanged. No virtual
aggregate marker is needed because no retained new-mode fallback guesses
virtual-vs-real identity.

The existing per-layer LRU cache (`moe_routing_workload.py:392-444`) keeps its
ratio keys. When the explicit-source predictor entry caches a **descriptor**,
  the new key includes the entire frozen `WaveRoutingSource`: every lane ID,
  batch/global ID, group/mode, source gating context, local physical/live field and interval, plus gathered
width, policy, role, replica/layer and topology. Intervals/global width alone
can return a preceding wave's lane binding. Scheduler materialization adds no
cache. Test equal counts after lane swaps/padding moves and require a correctly
rebound descriptor. No new cache, fingerprint or inferred-source fallback.

### 4.5 Ratio table presence for `position_cyclic`

`position_cyclic` has no expert-ratio source, and none may be fabricated:

- `_init_global_routing_allocations` (`moe_routing_workload.py:268`) returns
  `{}` for position policies; `_build_shared_routing_details` (`:308`) yields
  `{replica_id: {}}` and the disaggregated counterpart
  (`sklearn_disaggregation_execution_time_predictor.py:430-489`) does the same;
  the role attributes stay non-`None`, so the existing presence guards
  (`frontier/scheduler/utils/layer_path.py:27-28`,
  `round_robin_cluster_scheduler.py:938-940`,
   `base_cluster_scheduler.py:1099-1160`) keep their meaning;
- Source-required policy admission runs before these empty maps are produced:
  predictor/role construction validates the accepted P/D architecture and
  role-owned PP1/attention TP8/DP2/MoE TP1/EP16/no-MTP configuration using
  `_get_cluster_replica_config`/the existing disaggregation role config. Shared
  producer additionally requires bound forward sources in this no-SP scope.
  Do not attach empty new-policy maps to DECODE_FFN, monolithic skew or MTP
  just to let presence guards pass. Count math itself remains model independent;
- position-aware callers branch before `resolve_routing_details`
  (`moe_ep_workload.py:649`), whose existing missing-layer error rejects an
  accidental lookup in an empty table. `_get_routing_details_for_cluster`
  (`moe_routing_workload.py:368-386`) can give the explicit policy error. Do not
  teach a policy-free ratio resolver to invent a new position table;
- `_emit_routing_details_snapshot` (`moe_routing_workload.py:102-113`) is
  skipped for position policies and replaced by a
  `[ROUTING-POLICY-SNAPSHOT]` line carrying `policy_id`, `router_topk`,
  `total_expert_num` and the interval contract, never a ratio map.

### 4.6 Mandatory pricing-owner interface; no counts-only outcome

The frozen Step dispatch is a DP `all_gatherv` + `reduce_scatter` over physical
rows, while Frontier's `expert_parallel_alltoall_*` operators price the
assignment-based all-to-all (`families.py:540-555` uses
`lane.routed_token_count`). This design must not silently change that identity
for the four existing policies, and must not pretend a padding-only lane costs
nothing to communicate. Therefore:

- keep legacy policies unchanged; do not claim assignment-based all-to-all is
  an aligned measured Step `position_cyclic` collective;
- expose the source-lane vector and gathered expert frame, not just
  `LayerEPWorkload.routing_token_count`;
- require `pricing_design.md` implementation/tests before actual-main and
  measured acceptance. No counts-only alternative or assignment-cost fallback.

Frozen `naive_dp_ep` gathers physical hidden-state, top-k weight, ID and scale
rows, then reduces physical output rows. Masked rows remain in these buffers.
Retain source-lane physical rows beside the live expert vector. TP8/DP2 has two
request owners, not sixteen copies of their intervals. E352/EP16 retains 22
contiguous experts per rank.

Related pricing limitation: `expert_parallel.py:90-139` builds each Step EP
timing lane from the virtual aggregate; `batch_builders.py:64-72` binds its
pre-routing compute width to the aggregate. Native routing precedes DP gather
(`proposal.md:42`), so its GateLinear/router physical width is source-local,
whereas expert alignment uses the gathered width. The current single width
cannot be described as already correct for gating. The mandatory companion needs
separate source-router and gathered-expert widths, using the existing rank
mapping/topology. Do not divide by EP16 or invent an averaged lane width. Keep
the source-lane map in the descriptor so this does not require a new scheduler.

Actual current Step calls are named all-to-all dispatch/combine in
`moe_operator_times.py:235-272`; payload comes from `families.py:540-555`.
DSV4 registrations at `families.py:811-829` show the existing physical
all-gather/reduce-scatter mechanism, but their SP/EP group contract is not the
Step contract. Do not relabel them or turn the observed DP2 all-gatherv group
into EP16. Exact group/payload selection belongs to the communication owner.

Interface supplied to that owner: immutable source identities, source-local
widths/live intervals, gathered physical width, global expert counts, EP-local
counts/rank. Interface needed back: operator-specific native frame sizes,
DP sizes/group/payload, native padded-feature scope, and source-to-EP/TP-rank
mapping. Gate/router queries use source-local features; expert queries use
gathered features. Neither owner silently sets both to one `num_tokens`.
Section 6 consumes these facts without repeating their native investigation.

## 5. Registry consolidation (no model-name switches)

One count registry replaces the duplicated value lists and one enum:

| Site | Current | Change |
| --- | --- | --- |
| `frontier/config/replica_config.py:91-97,178-192` | hardcoded `{balanced, random, skewed, zipf}` | validate against `ROUTING_COUNT_POLICIES`; help text updated |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:514-536` | separate four-value set at519-525 before parent init | call `resolve_routing_count_policy` on the existing normalized selector before parent init; remove the set/error branch and use the resolved runtime_path; A7 owns this constructor hunk |
| `frontier/moe_ep_workload.py:32-86,774-855` | ratio generator and Hamilton materializer | keep old ratio math; one materializer invokes registry callable; ratio generation rejects source-required policy |
| `frontier/moe_routing_runtime.py:6-43` | path constants + distribution→path map | map through the registry; add `position_cyclic` |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py:59-89` | `WorkloadDistributionType` enum | move enum declarations to the policy module and import them here; resolver delegates to the registry; `.value` and existing test imports remain unchanged |
| `tests/e2e/moe_ep_baseline_replay.py:43-56` | frozenset + old-runtime translation | unchanged for the four existing values; the old runtime has no position mode, so no translation is added (see 7.3) |

Selection is by configured `moe_routing_distribution_type` only. No
`model_type`/`model_name` branch is added anywhere, including
`_ROUTED_LAYER_GROUP_RESOLVERS` (`frontier/profiling/moe/routed_layer_groups.py:112`)
and the profiling model-type policy.

The registry is behavioral, not only descriptive metadata: its count callable
is the materializer's only count dispatch. Its source requirement also owns the
legacy-vs-explicit-source distinction. Runtime-path validation derives supported
paths from its entries. Supported Step P/D geometry is an admission constraint,
not a second registry or model-name switch for count mathematics.

Constructor ordering is observable: `__init__:516-518` normalizes the selector,
`:519-525` currently rejects position_cyclic, `:532-536` chooses runtime metadata,
then `super().__init__:538-547` can begin profile/model setup. Resolve the count
registry immediately after normalization and before that parent call; retain
the seed check at526-531 and all legacy defaults/normalization. Unknown names
fail in the existing registry resolver, not a second set. Store/use the returned
policy runtime_path for this early initialization; source-required policy skips
ratio initialization through the already planned seams. C6 tests the actual
constructor with its expensive parent/model setup stubbed, not only the resolver.

## 6. Profiling: what the new policy requires

### 6.1 Existing identity surface

- `frontier/profiling/moe/moe_impl.py:317-335` `get_routing_runtime_metadata`
  accepts exactly `standard_fused_topk` (`logit_topk`, `softmax_renorm`,
  logits used) and `uniform_topk` (`round_robin_uniform`,
  `uniform_1_over_topk`, logits unused). `uniform_topk`
  (`moe_impl.py:77-108`) builds `(token_index*K + k) % E` over a frame starting
  at index 0 with weights `1/K`.
- `frontier/moe_routing_runtime.py:14-43` maps distribution→path and
  `filter_moe_gating_routing_topk_rows` enforces the row filter at load time
  (`layer_contract_resolution.py:450-484`,
  `prediction_family_trainers.py:224-275`,
  `moe_dataset_training.py:214-243`).
- Row stamping: `frontier/profiling/moe/moe_wrapper.py:479-491` (gating) and
  `:829` (grouped GEMM) spread `routing_runtime_metadata`;
  `frontier/profiling/moe/moe_impl.py:252-311` `SigmoidBiasGatingNetwork`
  measures the native Step router (`GateLinear` + `fused_topk_bias`) and
  reports `routing_assignment_policy = "biased_score_topk"`.

### 6.2 Required source-frame/gathered-frame profiler interface

The runtime requests `routing_runtime_path="position_cyclic"`. One explicit
input module `frontier/profiling/moe/inputs.py` supplies cells to existing
`main.py`/`MoEWrapper.profile`; it does not expose a single ambiguous width:

```python
@dataclass(frozen=True)
class MoESourceFrame:
    source_lane_id: int
    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    physical_rows: int          # local GateLinear/router frame
    gating_runtime_context: str # frozen source context, not wave-role default

@dataclass(frozen=True)
class MoERoutingProfileInput:
    moe_input_id: str                         # retained census/forward identity
    source_frames: tuple[MoESourceFrame, ...]  # DP order, no TP copies
    gathered_physical_rows: int               # inclusive routed-expert frame

@dataclass(frozen=True)
class MoERoutingProfileCell:
    routing_input: MoERoutingProfileInput
    moe_frame_scope: Literal["source_local", "dp_gathered"]
    source_lane_id: int                       # owner/local or pricing-bound EP source
    ep_id: int | None                         # required only for dp_gathered
```

Each input requires a nonblank retained census/forward ID. Each source frame
requires equal arrays, positive queries, nonnegative contexts,
positive physical rows, live<=physical and unique source lane identity. Empty
arrays mean an all-padding frame, not implicit position0. Build device positions
as each real request's context+arange(query); padding values are not interpreted
as positions. Bool mask is false only for live intervals and true for physical
padding. Validate gathered width equals the sum of source widths for accepted
DP2/no-SP; this is validation of the supplied physical sizes, not a scheduler.
Retain the census/forward identity in the input artifact, not an invented
position grid. Attention dummy query slots are not MoE live intervals.

Gate/router and applicable native FP8 prepare execute on **source-local** frames.
The inclusive expert sample contains already prepared tensors concatenated in
native DP order plus the actual selected scale/quant/workspace metadata, weights
and IDs. One **gathered** inclusive native expert root consumes that sample for
the explicit EP owner; its native movement is included, not separately shuffled.
CORE passes these frames/rows to pricing's native adapter and consumes its actual
scope output; it does not implement prepare/expert kernels, collective group
selection or native alignment/layout.

Source gating context is frozen by pricing's existing
`should_use_prefill_hot_moe_gating_context(batch=actual_source, ...):179-196`
predicate, not the wave role or virtual expert batch. Pricing adds that field to
SourceLaneRouting; CORE's whole-source cache identity/inputs/row admission retain
it automatically and never default a missing new-mode context. For admitted
Step placement, pricing associates EP owner r with source lane r//attention_tp,
validates topology/rank placement and supplies that immutable association. CORE
uses it for row/model binding, not an independent rank mapper. An idle source in
a P wave does not inherit the active source's hot-prefill context.

Concrete wrapper seam: `_run_gating_with_runtime_context:278-284` currently reads
one constructor-wide `self.gating_runtime_context`, and result metadata at487/830
spreads that value. In new mode pass the selected frame's validated context
explicitly through gating preparation/calls and stamp that same frame context;
the existing hot-prefix branch uses that argument, never a preceding cell's
mutable state. For a gathered preparation loop, each source adapter receives its
own context. Keep constructor default/calls/metadata unchanged for legacy mode.
Source filtering and context-specific gate/router model/cache selection use the
same frozen value. B1/B3 and existing admission/cache budgets include this seam.

One required `--routing_input_file` replaces scalar token enumeration for the
new path. The input JSON has `schema_version="moe.position_cyclic.inputs.v1"`,
`routing_runtime_path="position_cyclic"` and `inputs`, the serialized input
objects above. Cells are derived from these real inputs, not persisted as a
second independent input grid. No implicit arange(num_tokens), Cartesian
token/load grid or supplied expert counts substitutes for request intervals.
Old scalar profile paths remain unchanged. The bounded cell/row protocol below
uses the three existing executors, not multi-row return compatibility code.

**CORE owns the actual native timer adapter, not source S1–S13 or native N/T/P.**
Extend existing `SigmoidBiasGatingNetwork` in `moe_impl.py:256-314` with the
selected path; its standard branch keeps the current direct fused_topk_bias
call. For position mode construct `create_fused_moe_router` once, forwarding the
layer group's top-k/E, sigmoid scoring, renormalization, routed scaling,
selection-bias tensor and frozen skip-padding setting, with EPLB disabled. Bind
the actual returned class, method and `routing_method_type=Simulated`; reject
a wrong factory result rather than use a local implementation. Native19 owns
this vLLM factory/router/context definition, not this Frontier integration.

Reuse `vllm_config_context` for the standalone config and native
`set_forward_context(attn_metadata=None, vllm_config=..., num_tokens=local_width,
is_padding=local_mask)` around the standalone profiler call. Inside that context
enter native `set_routing_positions(local_positions)` and call the actual
`router.select_experts(hidden_states, logits, topk_indices_dtype=...)` in the
existing routing timer. Preserve the FP32 GateLinear timer and configured weight
semantics. Device positions/mask are materialized before warmup, reused for all
active iterations, and never reconstructed from IDs/width inside the router.
The requested IDs dtype matches the agreed native modular input contract.
Native setters own context cleanup; do not clone/replace an existing context or
reimplement its DP coordination. Standalone groups stay one actual process;
the source-frame evidence is not a claimed distributed serving context.

Return full-physical selected weights/IDs; require masked weights0/IDs-1 and
live IDs/counts consistent with the policy. `_run_gating_with_runtime_context`
and `_prepare_routing_inputs` pass the explicit local tensors through this same
adapter. Source owner supplies `verify_step5_profile_runtime` and its bound facts;
CORE calls the delivered `verify_step5_profile_runtime(manifest_path) -> native
facts`, then requires its actual factory result and bound select_experts method
to belong to the checked native class/module recorded in those facts. Native
select_experts is inherited, so also bind the selected implementation's
`_select_experts` to the checked simulator module, not merely the base method or
unrelated imported module names. Reuse source validation results; add no second
pin/hash system or incompatible validator argument. No local tensor router or
fallback is permitted. All these hunks are counted in B2/B3, separately from
source-owner S8/S12 binding hunks.

### 6.2.1 One existing result row per actual frame

Decisive current interfaces:

- `moe_wrapper.py:810-869` builds one dict with one `num_tokens` and typed
  contracts only for measured MoE names. `profile:892-927` merges all timings;
  this merge is valid for legacy scalar frames, not DP-local/gathered frames.
- Each executor invokes one `profile` and collects one dict (`main.py:169-175,
  786-790,867-879,905-908,940-946`). Remaining Ray results at949-955 also use
  that contract. `pd.json_normalize(time_stats)` at957-964 naturally produces
  absent-target NaNs for per-frame rows in the same CSV.
- `_validate_canonical_moe_result_df:521-567` rejects any per-row missing target
  today. `_load_moe_df:695-727` validates every typed object before scalar filters;
  `:794-801` already removes rows without the selected typed operator because
  `typed_contracts.matches_resolved_layer_contract:440-443` returns false.
  Independent training's column-level check (`moe_dataset_training.py:137-158`)
  still derives the legacy four-column union; new mode must instead use the
  same applicable operator set as its producer and both training enumerators.

**Chosen protocol:** one source-local cell per actual DP owner and one gathered
cell per actual EP owner, per census input and routed layer group. Thus DP2/EP16
has exactly2 source cells plus16 expert cells before any explicitly proven
shape-equivalent measurement reuse; not2 source cells duplicated16 times. New
mode uses the accepted TP1 expert/EP16 domain only. The input's DP order defines
concatenation. Legacy token/TP/EP/load loops and combined return dict stay exact.

**One applicable target contract, derived from the existing registry.** Current
`MOE_FAMILY.profiling_order:85-90` has the legacy four targets.
`OperatorFamilySpec.profiling_ops:349-357`/`predictor_ops:359-360`,
`iter_operator_families` and `bind_operator_query` already own operator identity.
Accept **exactly** pricing section5.1's
`get_moe_profiling_operators(context)` in existing `frontier/operators/families.py`,
returning ordered applicable **OperatorSpec objects**, not enriched bindings.
Pricing owns that selector, optional applicability predicate/context-aware
profiling enumeration and the conditional `moe_input_quantize` entry in
**MOE_FAMILY**. ROUTED already owns that family; `model_architectures.py` needs
no prepare hunk. The entry is selected only for Step+position_cyclic+reached
nondeferred FP8 prepare. Legacy no-context enumeration stays the four targets.
The previous CORE `applicable_moe_profile_operators` name/second selection function
is withdrawn, not retained as a wrapper or an alternative interface.

CORE callers bind those selected specs through existing `bind_operator_query(spec.name,
family_id=MOE_FAMILY.family_id)`/ROUTED `resolve_layer_contract` and pricing's
Step precision/native frame contract, then pass their facts to the shared
`moe_routing_runtime.py` filter without rebinding there. `OperatorQueryBinding:32-40` already
returns operator/profiling name/family/TP mode, but **not** precision, context or
schema. Keep those derived facts as a local bound record/tuple for existing
filter/trainer calls, not new attributes silently assumed on OperatorSpec or a
new exported binding/controller/registry. Use spec.tp_mode/spec.ep_agnostic,
resolved contract TP/EP/identity, spec.precision_name() with the selected Step
precision binding, frozen source context, registered profiling target/frame and
selected native schema. Pricing owns quant/applicability/frame facts; CORE owns
binding/consumer integration (A3/A9/A10/A17). No name-based prepare exception or
new generic TP/family helper is needed.

All producer/validator/CSV admission/loader/fit/exact-query users consume that
same returned spec set and existing resolved facts. For legacy inputs it returns the unchanged four MOE_FAMILY
targets. For the accepted new contract its result is:

| Native applicable binding | Frame | Inclusion condition / physical scope |
| --- | --- | --- |
| moe_gating_linear, moe_gating_routing_topk | source_local | always; actual source's gate/router/context |
| moe_input_quantize (pricing registration) | source_local | only selected reached nondeferred FP8 input prepare; no BF16/deferred separate target |
| moe_grouped_gemm | dp_gathered | one native_experts_inclusive root on already prepared gathered tensors; native movement included |

There is **no applicable moe_shuffling binding** for new mode. The registered
legacy operator remains intact; do not require its column, train its model or
query its zero shortcut in this policy. Structural absence is a registry/quant
contract fact, never inferred from missing timing, empty load or source hashes.
An applicable target lacking a row/key still fails coverage, including in-bound
empty width. `STEP5_FFN_FAMILY` is currently SHARED (`model_architectures.py:711-712`);
Prepare uses the now-delivered conditional MOE_FAMILY entry and existing ROUTED
membership, not a new family/SHARED registration. Without its selected native
quantization/typed precision binding, new FP8 admission fails clearly.

`MoEWrapper.profile(..., routing_cell=cell)` returns **one dict** for either mode:

Its bounded signature is `profile(num_tokens=None, load_distribution="uniform",
seed=None, *, routing_cell=None)`. Position mode requires `routing_cell` and
rejects scalar num_tokens or load-grid/seed overrides; legacy requires its old
scalar arguments and rejects routing_cell. Existing executor callsites select
one profile-kwargs shape from the validated case and use `profile(**kwargs)`;
new cases supply only routing_cell, old cases retain their current arguments.
Width is taken once from the selected frame, not supplied a second time.

| Cell / row | `num_tokens` | Timed names / typed contracts | Rank fields |
| --- | --- | --- | --- |
| source_local | selected source's physical_rows | resolver's source bindings: gate/router plus reached FP8 prepare | moe_source_lane_id is actual DP owner; moe_ep_rank is null; association stays in receipt |
| dp_gathered | gathered_physical_rows | resolver's one inclusive expert binding | moe_ep_rank is actual0..15 owner; moe_source_lane_id retains pricing-supplied associated source |

Source cells run CORE's local gate/router adapter and call pricing's source
prepare adapter only when applicable. Gathered cells route/prepare the same
explicit source frames outside the inclusive root measurement, then pass
pricing's already prepared gathered sample/workspace/quant metadata to its
original native inclusive expert invocation. Its per-frame result stays one
dict. Clear preparation timer storage before measuring that root; neither
untimed router work nor excluded source prepare enters its duration/count.
Pricing owns the `fp8_block_experts.py`/`moe_vllm_kernel.py` invocation, supplied
sample, kernel identity and timer-boundary hunks; CORE consumes/validates them
in wrapper rows, not a second kernel adapter. Legacy random-A/full-kernel/shuffle
call paths remain unchanged. No extra sample cache, movement timer charge or
collectives implementation in CORE.

Exact new row fields, in addition to unchanged model/group/measurement identity:

- `moe_input_id`, `moe_frame_scope`, `moe_source_lane_id`, nullable `moe_ep_rank`,
  and source-specific `gating_runtime_context` (gathered means associated source,
  not a new expert feature or borrowed gate timing);
- `num_tokens`, `live_routing_token_count`, `masked_routing_token_count` refer
  only to this row's physical frame; physical=live+masked;
- `moe_gathered_physical_rows`, `moe_dp_size`, and canonical JSON
  `moe_source_frames` containing every source ID/query/context/physical/gating-context
  field; the row's source_lane_id/ep_rank retain pricing's validated association,
  not a second rank-map input or a duplicate scheduling scheme;
- existing routing path/assignment/weight/scoring/mask identity;
- native observed global expert counts as canonical JSON
  `moe_global_expert_token_counts`; source rows mean that source's contribution,
  gathered rows mean the DP sum. Gathered rows also have canonical JSON
  `moe_local_expert_token_counts`, actual EP ownership, existing load/padding
  selected native layout/features and inclusive kernel identity supplied by
  pricing. Source rows carry applicable prepare quant/scope evidence, not expert
  fields. Persist the native contract identity so resolver applicability can be
  revalidated; do not trust a row declaring whichever targets happen to exist.

Arrays are canonical JSON strings **before** DataFrame/CSV export, not Python
tuple repr or nested multi-row envelopes. Actual bool masks must match the
explicit live-prefix/physical-padding contract and recorded counts. They are
recoverable from these exact input fields; no invented positions fill padding.
The frame name fixes the meaning of num_tokens; it does not replace the number.
Source-local row fields must never be copied from the gathered expert row.

`main.py` builds/validates these cells once, then passes the same serialized
`routing_cell` keyword through multiprocessing, Ray and sequential calls. Each
still collects one row and updates progress once per cell. Dry-run/confirmation
and total work use cell count rather than scalar token-grid count. Existing
remaining-Ray collection, JSON flattening, typed serialization, metadata
attachment and the single `moe.csv` writer stay unchanged. No new output file,
schema registry, list-return adapter or legacy migration is needed.

Paired-count production must be real, not the timing sample `count` field:
`TimerStatsStore:39-51` and `_collect_cuda_event_stats:632-655` emit durations
without kernel_count; existing `RecordFunctionTracer:95-172` emits measured
kernel_count with scope correlation and rejects inconsistent launch counts.
For new mode reuse one separate untimed RECORD_FUNCTION pass per validated cell,
with the same native sample/scopes, after clean CUDA_EVENT timing. Attach only
its actual per-scope kernel_count to the event row; RECORD_FUNCTION rows already
have it. Count collection occurs outside event timing, never replaces its
duration, and its active replay/sample count is not a kernel count. Reuse existing
timers, tracer, scope names and failure rules; add no counting harness or synthetic
count derivation. B3/B4/C4 include CORE replay/row-attachment hunks; pricing owns
native prepare/inclusive-scope timer/count boundaries. Count
missing/inconsistent/zero for an included physical scope is unqualified evidence,
not coverage or automatic cost0. Source S2's existing two method commands remain
the generation's CUDA_EVENT/RECORD_FUNCTION output owners; CORE supplies the
paired keys in both. This is operator evidence, not wall or CPU-gap profiling.

**B3 owns the actual singleton switch.** `DeviceTimer:33-38` captures the existing
TimerStatsStore object; `:69-80` and `:91-105` read its current enum and emit
`vidur_*` record_function annotations only for RECORD_FUNCTION. Constructing
`RecordFunctionTracer` alone leaves CUDA_EVENT timers producing events with no
annotations. `Singleton.__call__:10-13` ignores later constructor arguments,
so `TimerStatsStore(profile_method="record_function")` does not switch it.
No common-timer/tracer/singleton file change is required.

For the CUDA_EVENT new-mode branch, materialize/copy the completed event stats
first (after sync), then clear the cell's event store. Preserve
`self.profile_method` and output measurement identity as CUDA_EVENT. Locally wrap
the existing direct native cell callable/annotations, **not** recursive
`profile()`/legacy methods that dispatch on self.profile_method:

```python
store = self.timer_stats_store
previous_method = store.profile_method
store.clear_stats()
try:
    store.profile_method = ProfileMethod.RECORD_FUNCTION
    tracer = RecordFunctionTracer(self.output_dir)
    with tracer:
        replay_cell()  # same tensors/scopes; no rebuild/prepare inside gathered root
    replay_stats = tracer.get_operation_time_stats()
finally:
    store.profile_method = previous_method
    store.clear_stats()
# Validate exact applicable scope/counts, then attach ONLY kernel_count to event_stats.
```

The method remains RECORD_FUNCTION throughout all timer enter/exit and tracer
exit/parser work; restore/clear also on callable, tracer-entry/export, parser or
validation failure. No partial row is returned on those failures. A count pass
must not mutate event min/max/mean/median/std/count or replace them with tracer
durations; those durations independently belong only to the RECORD_FUNCTION
campaign row. Warmup/preparation are outside the root's count replay. Existing
record-function mode already has paired counts; it uses the same selected cell
callable with a matching active store method, without another replay. Each
existing worker executes cells serially in its own process (main:149-175;
sequential/Ray actor methods also return one cell), so this local try/finally
does not authorize a second store, locking layer or cross-process controller.

Pricing owns replayable original prepare/inclusive-root callables and their
timer scopes; CORE's wrapper invokes that same callable under this method switch.
C4 must run real DeviceTimer.__enter__/__exit__ and torch.profiler.record_function
on CPU-safe operations with only CUDA-event endpoints/tracer transport stubbed.
Trace transport supplies correlated launch/kernel fixtures **from the captured
real annotation names**; call actual RecordFunctionTracer.get_operation_time_stats
and its scope correlation/consistency code, never stub its count dict. Verify
annotations are absent while the unchanged store is CUDA_EVENT, present in replay,
original event stats plus parsed actual count reach the same row, sample count
is not kernel_count, and store method/stats are restored after callable/parser
errors and before the next cell. CPU transport fixtures prove switch/annotation/
pair plumbing only, not GPU count fidelity, timing or native parity. Native T2
and actual paired scope traces remain the required runtime evidence.

CORE makes the canonical validator policy/contract-aware: old rows still require
all four targets; position rows require every applicable binding for their frame,
its positive finite median/integral kernel_count and resolved typed membership.
Structurally inapplicable bindings and opposite-frame targets must be absent,
not0 or copied. For new mode the generation's required column union is the union
of applicable bindings from its declared inputs/groups, **not four global names**.
FP8 prepare may be absent on BF16 rows (NaN after shared CSV flattening), and
moe_shuffling may be absent from the whole new CSV. Require declared
source/EP cells. The existing canonical validator receives the validated expected
cell identities from profile_model and verifies every input/group/source owner
and every input/group/EP owner was emitted, with no duplicate cell. It validates
median/count pairs before returning the DataFrame. Do not turn missing expected
cells into zeros after flattening; the loader reuses the same row checks but does
not try to infer an entire input inventory from partial timing rows.

Use one shared `filter_moe_operator_rows(df, *, operator_facts,
requested_runtime_path, source_name)` in `moe_routing_runtime.py`. Here
operator_facts is the local per-op resolved spec/query/typed contract/precision/
frame/context tuple or mapping described above, not a newly exported type.
The caller's pass-wide map keys these records by registered operator/model context.
It consumes
`get_moe_profiling_operators(context)`, not its own target/frame list. It checks row
policy/native contract/frame, applicable targets/fields and typed membership,
selects only the applicable operator's real frame, and fails if expected coverage
is absent. An intentionally inapplicable operator is not enumerated for training
or lookup; asking to bind one is a contract error, not a returned zero. Old policy filter behavior
stays unchanged. Call it from canonical admission, `_load_moe_df` after its
all-row typed validation, both trainer/contract branches and Step fit-row
selection. Cache keys include base operator/frame as well as runtime, selected
typed layer contract and gating context; source-local and gathered TP1/EP16
datasets must not reuse one cached frame just because scalar filters coincide.
Replace the independent new-mode column-level union/model-name enumeration with
that resolver; leave its legacy four-column rule. Both training enumerators
resolve bindings per routed precision/native contract before loader selection,
including conditional MOE_FAMILY prepare in the same moe_input_file.
No unconditional target append or separate prepare-file selector. Select rows
before features/metadata/pairs; gate-context suffixes come from the bound actual
source context, and no FP8 prepare pair is trained from BF16 structural absences.
Both duration and count maps
come from those selected rows. Never rely on sklearn dropping opposite-frame
NaNs or on a copied target to satisfy the legacy validator.

**Trainer closure must happen before any old pre-loader helper, not after load.**
For new mode enumerate pricing's selected specs and bind the local facts above
for each routed precision/group and actually required source gating context.
Reuse existing model/context suffix and typed contract serialization only where
`spec.precision_name() == "moe_gating"` (the existing registry classification,
not `_is_moe_gating_family_model_name`'s repeated family lookup); prepare stays its registered
name/FP8 precision, never `moe_input_quantize__prefill_hot`. For each selected
spec use the resolved ROUTED contract's tensor_parallel_size and optional EP
field (validated against spec.tp_mode/ep_agnostic), actual precision/native schema,
frame and source context directly. At accepted TP1/EP16, prepare's estimator is
TP1/EP=ANY although the row/native receipt still records physical EP16 placement;
inclusive expert is TP1/EP16. Gate/router retain their actual FP32 contract.
Missing selected typed/precision/context facts fail before loader/fit.

Reachable callsites and CORE hunks:

- Independent `moe_dataset_training.py`: `_validate_moe_dataset_contract`
  `:219-254` calls TP/EP/gating helpers even before `_get_moe_df_for_op`; new-mode
  validation instead receives the selected bound facts and calls the shared
  frame/target/typed/precision/context filter. Enumeration/metadata at320-345 and
  `_register_additional_profiling_metadata_from_files:444-466` use those same
  specs/contexts. `_get_moe_df_for_op:357-396` branches on source-required policy
  **before** `_get_moe_op_tp_key`, `_is_moe_op_ep_agnostic` or
  `_is_moe_gating_family_model_name`, selects by the bound TP/EP/frame/context/
  precision facts and returns them with the rows. `_train_moe_models:420-437`
  uses the bound profiling target/schema and registered precision metadata for
  both duration/count fits; no generic model-dtype prepare identity.
- Shared `prediction_family_trainers.py`: enumeration/validation at197-225
  passes those bound facts into the existing validator. `_get_moe_df_for_op`
  `:249-302` new-mode branch happens before unbound `_resolve_typed_layer_contract`
  or the old TP/EP/gating helpers at256-280. Pass its resolved operator-specific
  ROUTED contract, actual tp_key/ep_key and profiling name through existing
  `_load_moe_df(..., tensor_parallel_size=..., expert_parallel_size=...,
  operator_facts=op_facts,
  **_layer_contract_kwargs(contract, operator_name=...))`. Existing loader
  `_load_moe_df:695-727` gains that optional keyword (legacy None) and forwards
  it into the planned shared filter after all-row typed validation; it does not
  rebind through old helpers. The A18 +25/-2 allowance includes this argument/
  filter forwarding, so no ninth delta path or second binding work is counted.
  Apply shared filter
  before returning; attach precision/frame/context/typed identity to existing
  `op_training_context:345-376`, and use the same fields for both fitted maps.
  Cache/model signatures include selected operator/group precision, source
  context, runtime and frame, not the first generic ffn_signature alone.
- Shared preflight `layer_contract_resolution.py:451-484` must branch before
  helpers457/462, using passed per-op bound facts and the shared filter. It must
  not validate every selected operator under the first primary_contract or call
  no-context MOE_FAMILY.profiling_ops() to rediscover conditional prepare.
  Both existing `_validate_moe_dataset_contract` implementations take one optional
  keyword-only `operator_facts` mapping produced for that training pass from
  selected specs and existing binding/contract/precision/context results. It is
  a local immutable fact bundle, not a new registry/class/public selection
  function. New mode requires it; legacy defaults to None and takes the old
  path. Each validator selects that operator's record for the shared filter;
  `_get_moe_df_for_op` closes over the same mapping rather than recalculating
  family applicability. This is A17/A9/A10; old helper352-381 stays unchanged.

Legacy branches keep existing TP/EP/gating helpers and optional hot-row warnings.
New-mode required source context/count coverage cannot be silently skipped by
the legacy PrefillHotRowsUnavailableError catch. No broad family registry or
helper generalization is added. C5 exercises both real trainer entrypoints plus
the shared preflight/loader on a minimal scoped FP8+BF16/context CSV; stub only
the expensive fit, and make old helper calls raise if reached in new mode.
Assert source prepare TP1/EP=ANY and FP8 identity survive until fit, BF16 does not
enumerate it, expert uses gathered M/EP16, actual gate contexts remain distinct,
and legacy calls still take their original helpers. C6 covers constructor/order
and paired lookup after that admission. These are additions to existing tests,
not a new grid, runtime campaign or extra file.

Required metadata:

- `routing_runtime_path="position_cyclic"`,
  `routing_assignment_policy="position_modulo_experts"`; retain configured
  `sigmoid_renorm_scaled`/`sigmoid_scaled` weight identity.
- `routing_assignment_uses_positions=true`,
  `routing_assignment_uses_scores=false`, `routing_weight_uses_scores=true`,
  `routing_uses_router_logits=true` (logits still compute weights).
- Router features are exactly `("num_tokens", "live_routing_token_count")`,
  meaning source-local physical/live rows. Expert `num_tokens` means gathered
  physical rows. Persist the agreed frame identity as operator metadata, not
  another width inference. The concrete row identity is `moe_frame_scope` plus
  `moe_source_lane_id` or `moe_ep_rank` in 6.2.1; pricing must consume that same
  contract, not invent another ambiguous width.
- Exact source query/context arrays, source-local physical/live/masked rows,
  source lane/frame identities, DP sizes, gathered rows, profiled EP rank and
  observed local/global counts, native padded shape and source identity.
  Scope labels distinguish source-router rows from gathered-expert rows.
- `_get_profiling_ep_rank` (`moe_wrapper.py:304-306`) uses the explicit cell's
  EP owner, not fixed rank0. Collect every distinct reached local shape; retain
  which owners reached it. A duplicate feature vector cannot erase incompatible
  source/kernel/mask scope identity.

`standard_fused_topk` and `uniform_topk` retain their real semantics and labels;
neither is renamed for this policy. Admit the new path through the registry at
`moe_wrapper.py:104-114` and `main.py:340-348`. Routed layer groups still classify
precision/activation, not policy. `examples/profiling/profile_moe.sh` already
passes the runtime path; its old default remains, no companion mode is added.

Two mandatory mask-consumer fixes are within profiling scope:
`moe_wrapper.py:336-338` must filter global IDs-1 **before** indexing expert_map
(Python negative indexing otherwise aliases the last expert); and
`load_distribution.py:178-181` must exclude the masked sentinel before bincount.
The inclusive expert sample retains its full gathered physical row dimension;
the unchanged legacy shuffler keeps its current input contract.
The routed-layer-group classifier stays unchanged: policy selection is input,
not a new Step model or precision group.

### 6.2.2 Actual expert identity versus source identity

The delivered source design's isolated-expert note (previously lines275-281,
now moved by its owner's correction) is correct: the native router's `Simulated`
classification and isolated expert builder's `Unspecified` are different facts.
The former cannot be stamped onto the latter to claim numerical/kernel parity.
The current core plan otherwise left this builder implicit; this correction
makes its actual identity and qualification boundary explicit.

Observed decisive source:

- `fp8_block_experts.py:200-228` builds a one-process TP1/DP1 expert config with
  supplied EP/rank and `RoutingMethodType.Unspecified` for precomputed IDs.
  `:245-252,277-282,290-304` calls the current automatic FP8/BF16 oracle and
  modular kernel construction, with existing `allow_vllm_cutlass=False` intact.
- `modular_kernel.py:546-580` tests routing method and parallel/quant/activation
  support; its inherited method check at636-658 accepts all methods/logit dtypes.
  The inspected FlashInfer CUTLASS, Triton and TritonOrDeepGemm expert classes do
  not override those routing-method checks. Thus changing Unspecified to
  Simulated alone is not a demonstrated reason to change these modular classes.
  This static fact is **not** executed-backend parity: oracle priority and shape
  support have other inputs, and a constructed fallback class can select an
  actual subkernel at execution.
- Existing `NativeRoutedExperts.apply:97-116` calls the real modular
  `kernel.apply`; `moe_vllm_kernel.py:906-936` exposes only selected backend,
  constructed experts class and optional tuning config. It does not prove the
  actual E8M0/valid-shape subkernel or whole distributed module scope.

**No speculative backend/config change.** The observations above describe the
current finalized kernel.apply, not proof it is the agreed new inclusive root.
Pricing owns the selected original inclusive-expert invocation over prepared
gathered tensors, native scope boundaries/quant/layout and truthful expert
config/kernel/callable identity, including any difference between its actual
routing method and router Simulated. CORE propagates/validates those returned
fields through existing rows/measurement identity; it does not force Unspecified
or stamp that old builder label onto a new callable. CORE's router separately
records `moe_router_routing_method="Simulated"` and its actual select_experts
implementation. Pricing owns any KERNEL_IDENTITY_COLUMNS/native builder hunks;
no duplicate CORE allowance remains in those files. Current modular API is
required; no legacy low-level fallback for new mode.

Before **timing reuse/qualification**, compare the selected/constructed identity
and actual executed kernel/module/shape/activation/quant evidence with native
T2's executed FP8/BF16 evidence for the accepted source/gathered contract. Native
T2 already owns that execution observation; do not duplicate it in source pins.
Matching package/source hashes or equal expert counts are insufficient. Any
Expert-method/Simulated compatibility claim must state the native method-support
evidence and observed callable/subkernel agreement. Preserve both labels even
after qualification. If the native backend/scope actually differs, leave rows
unqualified and ask the parent to reconcile a measured mismatch with pricing;
do not force Triton/DeepGEMM or silently change routing_method here. Real timings
may be collected as unqualified evidence; they are not admitted to the new
measured serving bundle until this comparison passes. CPU fixture kernel labels
remain explicitly synthetic and never satisfy this requirement.

### 6.3 Shared feature schema and exact typed coverage: no empty-load credit

Observed current behavior, not a timing claim:

- `moe_operator_times.py:479-480,497-498,653-654,660-662` returns0 for an empty
  local expert lane before measured lookup. With the position policy this must
  instead query the applicable positive-physical-row inclusive expert scope;
  separate shuffle is structurally inapplicable, not a lookup producing zero. Old policies
  and explicitly synthetic dummy mode remain unchanged.
- An empty **local** lane is not an all-padding wave. Current
  `moe_routing_workload.py:595-603` pads `global_token_counts` over all experts,
  so local routed0 can coexist with padded>0. This is an observed current
  behavior, not proof of the correct new native expert scope. Preserve the
  local/global distinction and consume the pricing owner's actual padded-frame
  contract; do not force padded0 from local live0 or call current global padding
  aligned without that companion.
- `moe_dataset_training.py:50-81` selects two padded grouped-GEMM features,
  or the full load feature set; neither contains physical width. Positive
  physical frames of different widths can alias at empty load. For rows
  explicitly selected as `position_cyclic`, add `num_tokens` to the existing
  inclusive expert feature selection, retain the selected native layout/live
  features, and use `num_tokens` plus live rows for routing-topk. Validate that
  the selected new-policy rows contain all required fields; do not pad a
  partially stamped table with defaults or alter old feature schemas.
  Filter position rows for affected MoE operators before deriving schema, not
  just routing-topk; otherwise mixed legacy/new tables can contaminate feature
  and measurement identity. Keep this in existing MoE training owners.
- `step5.py:495-566` requires measured median duration>0 and actual integral
  kernel_count>=1 for an included physical operator. Physical dimension
  metadata (`operators/typed_contracts.py:31-51,269-319`) is not a live-route
  count and need not be weakened. `sklearn_moe_execution_time_predictor.py:106-165`
  preserves duration/count identity and feature bounds; kernel-only mode also
  requires the existing GPU kernel-gap table. No new CPU-gap mechanism.
- Bounds are **not coverage**: `sklearn_execution_time_predictor.py:4915-4979`
  uses an exact row if present, otherwise runtime cache/sklearn.predict even for
  an unmeasured feature inside bounds. A zero-local-load width between two
  measured empty widths can therefore interpolate today. New-policy Step
  lookup must reject that miss, not merely record `exact_lookup=False`.
- `frontier/moe_load_imbalance.py:41-56` rejects physical width above the
  profiled BLOCK_SIZE_M range. Do not widen the range by copying the last row.

Shared recognition is mandatory. The proposed position **Triton-compatible** tuple
`("num_tokens", "total_routed_tokens", "num_tokens_post_padded")` alongside the
existing two-column tuple in `frontier/moe_load_imbalance.py` is not a universal
DEEPGEMM padding claim. Keep one shared exact schema classifier in that module:
the old/new Triton-compatible tuples plus the pricing owner's actual selected
native layout/capacity schemas. Pricing owns those added native schema entries
and feature construction; CORE owns common recognition integration across
selection/runtime/two trainers. Recognize required field order and native layout
identity, not arbitrary supersets or an invented BLOCK_SIZE_M for another backend.
Replace all three old exact-tuple tests with that classifier:

1. Runtime `moe_routing_workload.py:593-604` builds padded/capacity features for each
   recognized selected native schema, retaining gathered physical rows for the
   new mode. Pricing supplies non-Triton capacity/alignment fields at this seam.
   In measured Step context it must obtain schema/ranges from the same typed
   per-layer/per-precision model selected by `step5_operator_binding` and the
   existing family-model lookup (`sklearn_moe_execution_time_predictor.py:110-119`),
   not assume the generic `_predictions["moe_grouped_gemm"]` entry is the actual
   FP8/BF16 query model. Reuse this existing model selection for both feature
   construction and exact lookup; a default/last-trained model is not a source
   for the current layer's native ranges.
2. `moe_dataset_training.py:429-439` selects the same native schema and attaches
   `_frontier_block_size_m_ranges` only if that schema uses them. No fabricated
   Triton ranges on DEEPGEMM; native layout metadata comes from pricing.
3. `prediction_family_trainers.py:354-380` does the same, including every fitted
   Step precision's duration/count pair, not just the last returned model.

Selection remains in existing `select_moe_operator_features` (`:50-81`): resolve
applicable binding and requested path first; inclusive expert root requires its
complete selected native schema with gathered num_tokens, router uses source-local
num_tokens/live rows, gate uses local num_tokens. Applicable prepare uses local
num_tokens plus the pricing-registered quant contract/schema, not an invented
feature list. No new-mode shuffle features/model. Missing required native
quant/layout binding is an admission error, never a legacy schema fallback.
Existing load features already contain `total_routed_tokens`, but not physical
width. Old tuples and old selection fallback remain unchanged. New rows never
fall back to scalar/load schemas if a required padded/mask column is absent.
Both trainer branches and validation (`layer_contract_resolution.py:451-484`,
`moe_dataset_training.py:214-243`) call the same routing-row filter/selector for
affected operators; trainer cache keys include runtime path, not router only.
Use the registry requirement, not operator-local new-policy string lists.
The affected row selection covers GateLinear/router source frames and
applicable source prepare and inclusive gathered expert frames; frame identity distinguishes them even when scalar
widths coincide. Reusable unchanged gate measurements require matching real
source/kernel identity, not relabeling old-policy rows into a new table.

Exact Step query contract, in existing
`sklearn_moe_execution_time_predictor.py:106-165`:

New-mode dispatch at `_get_prediction_for_features`/`_get_on_demand_prediction`
`:168-176` must consult that same applicable binding set, not the old hardcoded
`_STEP5_FFN_OPERATORS` membership. Pricing supplies source gate/router-context/
prepare and gathered expert feature queries/bindings; CORE ensures each included
operator (including context-specific gate/router aliases) reaches the same typed
paired exact lookup, with the source lane's local width or gathered root's M.
No generic sklearn gate/router bypass or default feature schema for new mode.
Legacy dispatch/membership remains unchanged. This integration is in A7's ceiling,
not a duplicate source-local pricing model implementation.

- First resolve the applicable registered operator and its source/gathered
  feature schema/physical scope from the selected native contract. Non-applicable
  BF16/deferred prepare and separate shuffle are not duration models; do not look
  them up or record zero timing/count. Missing applicability prerequisites fails
  binding. For an applicable source-required routing policy, normalize the
  requested feature tuple
  in the trained schema order. Require that exact key in **both** duration and
  kernel-count maps before calling the parent prediction/cache path. Count
  model absence, duration-only row or count-only row is coverage failure,
  including CUDA_EVENT (not only KERNEL_ONLY) for new-policy MoE scopes.
- Require matching pair schemas and canonical measurement identity, target
  columns, routing policy, operator precision/layer group, runtime/source
  signature and physical scope/frame contract. Source proofs remain in the
  existing measurement identity/bundle chain; no unbound custom receipt is an
  identity substitute. `step5.py:603-625` must propagate the selected routing
  and frame/source metadata to that identity instead of losing it after fit.
  Invariant policy/backend/image/runtime/source signature and operator-frame
  contract belong in model identity. Per-point lane widths, request intervals,
  masks and counts stay in the coverage/source rows; do not require all rows in
  one model to have the same position interval or gathered size. Compatible
  duplicate feature rows may share timing only after their source/kernel/frame
  shape equivalence is established by the owning producer/pricing contract.
- Existing `_train_single_model` (`prediction_family_trainers.py:1591-1643`)
  fits/pairs counts from the same selected rows. Make the new-policy paired-count
  requirement explicit for both measurement families; persist actual counts,
  never derive them from assignment load. Both pairs share schema/ranges/source.
- Included scope duration must be finite>0, measured count integral>=1. Require
  every requested empty/local-concentrated/all-padding feature key; an in-bound
  missing empty width fails without model.predict or runtime-cache credit.
  Preserve existing Step measurement validation (`step5.py:495-566`) and old
  policy prediction behavior. KERNEL_ONLY still needs existing measured GPU
  kernel-gap coverage; this adds no wall/CPU-gap campaign.
- Retain the current `_step5_ffn_costs` proof fields for precision, layer,
  measurement, duration/count model hashes and evidence kind; add/validate
  bound routing/frame/source identity and report exact_lookup=true only after
  both maps passed. CPU fixtures remain synthetic, never native qualification.

Required future native measurements at each reached physical width/precision
and compatible kernel scope:

1. Router/gate for live, mixed-padding and all-padding source-local frames,
   executing the new native tensor router with observed positions/mask.
2. One inclusive native routed-expert root for empty local lanes in a live
   global wave, concentrated nonempty lanes, and globally all-padding waves.
   Record actual IDs/local/global counts and the selected backend's reached
   ranges/alignment/padding/capacity/layout, durations and actual kernel counts.
   BLOCK_SIZE_M is required only when that selected schema uses it. Applicable source-local
   nondeferred FP8 prepare has its own positive physical-row keys; BF16/deferred
   prepare and separate shuffle have no fabricated rows. E352/EP16 all22-expert
   lanes participate. Exact scope identity includes excluded prepare and included
   native movement/top-k finalization from pricing's actual invocation.
3. Physical dispatch/combine and source-router-width companion in4.6, from the
   actual TP8/DP2/no-SP route. Existing isolated TP1/EP16 profile points do not
   establish whole serving collective identity.

The actual empty execution may launch a positive-duration scope or may omit a
particular subkernel. Source inspection alone does not settle that. If the
included operator really has zero kernels, the current positive-scope contract
cannot admit it: parent must obtain measured evidence and decide an explicit
inactive-scope contract. That is not permission to invent duration/count0.

### 6.4 Mandatory launcher/source-owner consequences

`profiling/step5/job.py:197-204` hardcodes `standard_fused_topk`/scalar MoE tokens;
`:210-226` copies a fixed model config, fixes offline_site and checks immutable
P0 workload hashes. Thus complete new-policy source/launcher plumbing belongs
to mandatory `profile_source_design.md`, not this document's small flag budget.
Supply it the explicit frame-input interface in 6.2, schema/policy identity in
6.3 and a required new input path in the new generation; keep old P0 immutable.
A new JSON MoE input file, built
from the census, is necessary. `p0_prefill_inputs.json` and
`p0_decode_inputs.json` contain actual query/context pairs but representative
attention layers, not the EP wave's lane ownership or full MoE census. They
must not be relabeled complete new-policy coverage.

This document proposes no new policy classifier in `routed_layer_groups.py`,
P0 relabeling or shell-wrapper mode. Physical-dimension changes necessary for
actual collectives/expert scopes belong to pricing design and are not claimed
unnecessary here. FP8 layers3–87,
BF16 layers88–90 and swiglustep stay exactly the same group identities. Native
DeepGEMM/Triton dispatch/E8M0 still requires actual receipts; count equality is
not an expert-kernel reuse proof.

Reconciled source-owner interface (delivered, not a guessed future adapter):

| Seam | CORE | profile-source S1–S13 / native owner |
| --- | --- | --- |
| Native router timer | B2 constructs actual factory result/context/select_experts and binds _select_experts to S5's checked native facts; B3 supplies local tensors | Native N1/N3/N4 supply factory/router/position setter; S5 returns verify_step5_profile_runtime(manifest_path) facts. No adapter implementation is deferred to source. |
| New inputs / rows | B1 input schema/validation and frame cells; B3/B4 actual rows, dry-run, executors/CSV; A owners scoped admission/pairs | S2 passes/stages `roles[role].moe_input_file` as `--routing_input_file`, with generated plan path and derived routing strategy. |
| Source checks | B2 supplies actual router callable; B4 and C tests cover integration | S5/S8 verify binding before timing/identity, S12 supplies adversarial source tests. Their disjoint hunks are additive on shared files. |
| Image / launcher / pins | consumes `FRONTIER_STEP5_PROFILE_MANIFEST` and existing RuntimeIdentity fields; no duplicate producer selector | Source owns all S2 `job.py` hunks, S1/S3/S4 launch and S5/S6/S7 source identities; native owns immutable adapter/reference/image receipts. |
| Kernel parity | 6.2.2 propagates/validates pricing's actual inclusive expert identity; do not stamp constructed classes as execution evidence | Native T2 supplies executed backend/subkernel/module evidence; pricing owns real inclusive invocation, quant/frame/layout. Source hash equality alone proves neither. |

Source S2 is **wholly other-owned**: this core table contains no `job.py`
allowance to subtract. Source S8/S12 add their own binding work to shared
`moe/main.py` and `test_step5_moe_profiling.py`. Do not reuse source document's
old overlap totals (+662/10-files or61-file union): they refer to the superseded
core snapshot. Parent recomputes distinct paths and adds disjoint hunks after
all current tables, including pricing, arrive. No final union is claimed here.

Reconciled pricing boundary for this focused correction:

| Dependency supplied by pricing owner | CORE consumer / exact seam | Scope accounting |
| --- | --- | --- |
| get_moe_profiling_operators(context) → ordered OperatorSpec; conditional MOE_FAMILY prepare and real ROUTED precision/quant contract | CORE callers bind selected specs using existing query/layer/Step precision interfaces before loader/helper calls; shared runtime filter consumes those facts without rebinding | Pricing owns families.py/spec.py selector/conditional entry and Step precision binding; no model_architectures.py prepare hunk. CORE owns binding/consumer integration; no second target selector/family. |
| Source gating context and validated source-to-EP association | SourceLaneRouting frozen identity in4.1; inputs/source rows/context-specific models and gathered rank receipts | Pricing owns predicate/mapping/freeze implementation; CORE preserves fields in existing cache/input/selection/row protocol, no second mapper. |
| Native source prepare and inclusive gathered expert invocation/returned scope stats and identity | existing wrapper per-frame dict, native sample/target validation, paired rows | Pricing owns moe_impl prepare, fp8_block_experts/moe_vllm_kernel invocation/layout/identity/count timer hunks. CORE owns only gate/router timer adapter and consuming wrapper/main/fit hunks. |
| Selected native padded feature schema/layout/capacity and reached ranges | existing moe_load_imbalance shared classifier → runtime feature construction and both fit paths | Pricing owns added backend-specific schemas/features; CORE common classifier and actual typed-model/schema selection, no global-Triton fallback. |
| Scope phase/reporting and faithful collective_sim plan contract | parent integration/actual-main acceptance | Pricing owns costs/metrics/traces/CC; CORE fixture uses selected interface, never measured-plan artifact as an approved substitute. |

Native prepare/inclusive-root/registry/layout/reporting hunks are mandatory
pricing allowances on the shared paths, not zero-cost work and not duplicated in
CORE. Pricing's corrected section6 common/Variant-A known-interface tables are
delivered, but a finite selected full-duration simulation model/complete ceiling
remains unresolved. Parent plan:33-40 supersedes its exact NCCL per-channel
reconstruction prerequisite: required model inputs/assumptions and their scope
belong to pricing/parent. Variant-B measured lookup is explicitly unapproved.
Parent must combine the common/selected work only after remaining model/scope
are settled; it cannot mechanically union the lookup variant or call known
interface ceilings the full numerical repair. This document's table is complete for CORE
consumer/count/adapter/fixture ownership, not the full integrated repair.

## 7. Fixture and runner consequences

- `tests/integration/step5_measured_compute_fixture.py:119-137` explicitly
  materializes Hamilton-balanced scalar widths and skips every empty EP lane.
  Its config at218 and manifest at251-252 also say `balanced`. New-policy
  count/lookup integration **requires** explicit P/D source intervals/masks,
  all EP lanes including empty, separate source/gathered physical shapes,
  the position identity and a fresh fixture generation. Its synthetic times
  must stay `evidence_kind=cpu_fixture`, never native timing proof.
- `tests/integration/run_step5_measured_main.py:73-74,127-128` requires and
  passes `balanced`/seed42. It cannot validate a position fixture unchanged.
  Preserve old fixture cases/schema; add a separate version/identity for the
  required position fixture. Runner forwards the manifest's explicit policy,
  not hardcoded `balanced`, and rejects schema/policy/input disagreement. This
  is required actual-main acceptance, not optional. The native owner still
  owns serving selector/reference generation.
- `tests/unit/test_step5_measured_ffn.py:50-105` builds synthetic
  `standard_fused_topk`/uniform load rows. Keep old fixtures intact and add
  explicitly labeled position/empty cases. Do not change a string label on the
  old rows to manufacture coverage.
- Existing conservation fixtures remain valid for the four old policies.
  `ep_trace.py:521-524`, however, enforces physical*K conservation and must be
  amended to live*K for a descriptor carrying the position source, reporting
  both physical/live and padding. A comment or a new counter alone is not enough.
- Public old-runtime matrices (`tests/e2e/moe_ep_baseline_replay.py:43-56`,
  `moe_ep_non_dummy_matrix.py:4635-4800`) remain old-policy regressions. Do not
  add the new policy to them or translate it to random. Their ratio snapshots
  cannot prove position behavior. New campaign proof consumes policy id plus
  intervals/global expert counts from the actual shared materializer. Preserve
  `[ROUTING-SNAPSHOT]` for old tables; emit a separate policy snapshot for new
  tables. A full old-matrix schema migration is not needed.

The position fixture builds rows from the same canonical source/materializer
interface as production, with native forward binding for two source DP owners;
it does not generate counts from scalar aggregate widths. Reached fixture work
covers scoped P targets (including P6128), transferred D-tail context6128/query16,
q1 positions6144..7166, physical capture buckets and dummy lanes. Include all
EP owners and empty source/expert rows with explicit CPU durations/counts and
cpu_fixture source identity. Any synthetic kernel choice remains synthetic.
Do not refresh an immutable historical fixture into the new policy.

Required runner acceptance is the existing paired offline/online full
`python -m frontier.main` path (`run_step5_measured_main.py:228-286`), retaining
2400s timeout, 6144/1024 request, P6128/D16+1023q1 census, capacity/READ ordering
and positive measured-scope costs. Its output now also proves policy id,
live/physical conservation, correctly bound source frames and empty-lane exact
coverage via production descriptors/logs, not a manually attached expert map.
This does not create a new wall-time or CPU-gap campaign. Pricing owner's costs
and profile-source owner's source identities must integrate before this passes.

## 8. Decisions and blockers for parent

Already decided: `position_cyclic`, mask preservation, selected native weights,
no request phase/layer rotation, same-position concentration, independent G3.
There is no remaining label or per-wave-balance choice.

1. **Repair expansion:** combine section 9's own mandatory scope with native,
   `pricing_design.md` and `profile_source_design.md` ceilings, union overlapping
   paths without dropping their added/deleted work, then obtain parent/user
   scope approval. Own-side subtotal is not a complete implementation estimate.
2. **Mandatory pricing interface:** consume its confirmed source/applicable-FP8-
   prepare/inclusive-expert partition and exact section5.1 selector/conditional
   MOE_FAMILY prepare. Actual selected native quant/layout/source-to-EP evidence
   and full-duration communication inputs still need the existing owners.
   Existing all-to-all numbers do not prove the accepted Step DP identity.
   collective_sim remains the approved numerical approach; measured exact-plan
   lookup is unselected. Pricing/parent must settle a faithful simulation design
   and its separate allowance, not replace it by retaining the backend class name.
   There is no counts-only or optional-pricing completion path.
3. **Measured empty scope:** schedule the stated native rows through the parent.
   Until present, fail coverage; a measured zero-output scope requires explicit
   inactive-scope design after observing its actual launch behavior. No fallback
   or arbitrary cost is proposed.
4. **Mandatory source/launcher interface (reconciled):** CORE implements the
   actual native adapter/call seam and passes it to the delivered source-binding
   interface. Source owns image/runtime/source signatures, new input path,
   all `job.py`/staging/pins propagation and launcher/source tests. Its S8/S12
   are additive on shared files; no core job allowance remains to deduct.
   Required native executed-kernel parity in 6.2.2 remains a measurement
   prerequisite, not a source-equality shortcut.
5. **Mandatory actual-main fixture:** section 9.D is included unconditionally;
   accept only after shared schema, paired exact coverage and mandatory pricing
   integrate. Preserve old policy fixture support; no old-row relabeling.
6. **Deferred new-mode scope:** PD-AF/MTP/legacy virtual fallback implementation
   and their tests are excluded. The strict new-mode admission/source guards
   reject them, while existing policy behavior stays intact.

## 9. Exact minimum code/test paths and gross line ceilings

Ceilings are planned added+deleted lines, **not** net growth or measured diff.
No implementation has occurred. A-D are this owner's required code/test scope,
including actual-main fixture. Mandatory pricing, native and profile-source
launcher/image work is additional other-owned scope. This subtotal is not the
final combined estimate; parent must integrate delivered companion tables and
reconcile shared files before approval. Documentation is separately identified.

### A. Frontier count source, descriptor and measured-coverage seam

All paths relative to repository root:

| Exact file | Existing size | + / - ceiling | Reason / decisive seam |
| --- | ---: | ---: | --- |
| `frontier/moe_routing_policy.py` (new) |0|200 /0| callable registry, frozen source lanes, strict source validation, interval math |
| `frontier/moe_ep_workload.py` |898|75 /12|134-181,374-388,810-855: live ledger, single registry-callable materializer, shared EP split |
| `frontier/moe_routing_runtime.py` |67|105 /18|6-43: count-registry paths and shared frame/target/typed filter; consume pricing's sole selected-spec interface and locally bind existing TP/EP/precision/context facts, no second selector |
| `frontier/config/replica_config.py` |269|6 /10|91-97,178-192: registry validation/help |
| `frontier/execution_time_predictor/moe_routing_workload.py` |671|90 /22|268-444,593-604: strict source, whole-source descriptor cache, shared padded schema/runtime features |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` |3019|18 /20|59-89,430-514: import enum, skip ratio initialization |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` |1793|86 /19|106-176: applicable dispatch/paired exact keys;514-536: resolve count registry before parent init instead of four-name set, preserve seed validation |
| `frontier/execution_time_predictor/moe_operator_times.py` |744|35 /10|479-508,653-668: applicable inclusive-root no-zero/exact-query integration; no new shuffle query. Source/prepare/native scope/collective pricing hunks other-owned |
| `frontier/execution_time_predictor/moe_dataset_training.py` |493|109 /26|50-81,137-158,219-254,320-437,444-466: selected specs/metadata; preflight and row-cache binding facts before TP/EP/gating helpers; typed precision/frame/context into both fits. Legacy branches unchanged |
| `frontier/execution_time_predictor/prediction_family_trainers.py` |1849|101 /28|197-225,249-325,345-380,1591-1643: bind selected specs before loader/helper paths; actual TP/EP/precision/context/contract and frame caches/signatures/pairs |
| `frontier/scheduler/utils/ep_wave.py` |164|20 /2|109-131: actual source lanes and live conservation trace |
| `frontier/scheduler/utils/layer_workload.py` |67|35 /8|13-67: role policy/source into core |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` |1993|8 /3|899-915: shared P/D source threading only, no new PD-AF path |
| `frontier/scheduler/utils/ep_trace.py` |542|20 /8|496-542: live ledger and physical/padding trace |
| `frontier/moe_load_imbalance.py` |195|25 /1|33: one common exact schema classifier used by runtime/selection/trainers; pricing-owned native capacity/layout entries are additive |
| `frontier/execution_time_predictor/step5.py` |627|50 /8|495-625: scoped operator rows and routing/frame/source/backend/callable identity propagation, paired-row prerequisites |
| `frontier/execution_time_predictor/layer_contract_resolution.py` |495|37 /8|451-484: pass per-op bound facts and branch before TP/EP helpers457/462; shared filter/actual ROUTED contract, legacy352-381 helpers untouched |
| `frontier/execution_time_predictor/profiling_dataframe_loaders.py` |979|25 /2|695-727,794-811: after all-row typed validation, same applicable/scoped row filter; preserve per-frame num_tokens and legacy typed/scalar filters |

A: **18 files, +1045/-205 =1250 gross lines**.

### B. Existing MoE profiling interface and producer

| Exact file | Existing size | + / - ceiling | Reason |
| --- | ---: | ---: | --- |
| `frontier/profiling/moe/inputs.py` (new) |0|130 /0| exact input schema, DP frames/source gating context and supplied association, bounded cells, canonical JSON and mask/position validation |
| `frontier/profiling/moe/moe_impl.py` |658|95 /18| CORE-owned factory construction, native forward/position context/select_experts, existing timer/metadata/callable integration; no source-owner substitute |
| `frontier/profiling/moe/moe_wrapper.py` |929|184 /37|118-119/443-495/872-929: preserve event dict, locally switch existing singleton to RECORD_FUNCTION around direct cell replay and restore/clear in finally; selected targets/widths/counts/native adapter consuming hunks |
| `frontier/profiling/moe/main.py` |1218|115 /30| input/cell count, three executor keyword paths; same applicable target/column set for scoped validation and CSV admission; one dict, existing flattening and legacy rule |
| `frontier/profiling/moe/load_distribution.py` |275|3 /2| skip IDs-1 before existing bincount |

B: **5 files, +527/-87 =614 gross lines**. Native expert sample/invocation/layout/
kernel-identity hunks in fp8_block_experts.py and moe_vllm_kernel.py are wholly
pricing-owned, mandatory and excluded from this subtotal; no duplicate CORE hunk.
New census-derived JSON is a mandatory
generation input artifact, not an invented grid. Its exact path/source and
launcher/staging binding belong to `profile_source_design.md`, not zero-cost
omissions from the complete task. No dependency/package edit in this subtotal.

### C. Minimal persistent tests

| Exact file | + / - ceiling | Acceptance |
| --- | ---: | --- |
| `tests/unit/test_moe_position_routing.py` (new) |160 /0| math/registry/input proof, strict missing-source rejection, source-lane-aware descriptor cache/invariance |
| `tests/unit/test_moe_routing_conservation.py` |70 /1| actual public wave producer + physical/live trace |
| `tests/unit/test_moe_routing_runtime.py` |60 /0| registry-derived applicability and one shared frame/typed/target set; FP8 reached/deferred/BF16 and legacy unchanged; no guessed ROUTED family |
| `tests/unit/test_step5_moe_profiling.py` |175 /0| C4: CORE adapter/mask plus real DeviceTimer/record_function annotations and actual tracer correlation parser/paired count; immutable event durations, callable/parser failure restore/clear. CPU trace transport only; source/pricing tests additive |
| `tests/unit/test_moe_grouped_gemm_padded_features.py` |140 /0| C5: both real pre-loader trainer/preflight paths consume selected prepare TP1/EP=ANY/FP8 and context/frame facts without old helpers; existing scoped CSV/schema/cache tests, legacy helpers unchanged |
| `tests/unit/test_step5_measured_ffn.py` |108 /0| C6: actual constructor admits registered policy before parent init and rejects unknowns; typed paired exact empty keys and source/target proof; no structural zero |
| `tests/unit/test_moe_profiling_output_metadata.py` |85 /0| mixed FP8/BF16 source/gathered cells, optional prepare and no new shuffle; missing applicable/opposite/fabricated target rejection; legacy four targets; real executors/CSV flattening |

C: **7 files, +798/-1 =799 gross lines**.

### D. Mandatory actual-main fixture/runner acceptance

| Exact file | + / - ceiling |
| --- | ---: |
| `tests/integration/step5_measured_compute_fixture.py` |140 /30|
| `tests/integration/run_step5_measured_main.py` |25 /10|
| `tests/integration/test_step5_measured_main.py` |90 /8|

D: **3 files, +255/-48 =303 gross lines**, unconditionally included. Covers
required position-generation inputs/all EP lanes/source+gathered shapes,
manifest-driven CLI, fresh schema, source/conservation/empty exact coverage
proofs and the same registry-applicable scoped CSV protocol (reached FP8 prepare,
no BF16/deferred prepare or separate shuffle zero); preserves legacy fixture
expectations. CPU proof is not native timing.

**Own mandatory A+B+C+D:33 distinct code/test files, +2625/-341 =2966 gross
changed-line ceiling.** This is not the final complete task budget: additional
mandatory native, pricing and profile-source/launcher paths/line ceilings must
be imported from their owning designs. `job.py` is expressly mandatory there.
Shared `moe_operator_times.py`, descriptor, profiler and fit/query files may
overlap other-owner work; union paths but add all disjoint changed hunks. Do not
count a file twice or omit its additional work to keep this subtotal unchanged.
This already exceeds 5 files/500 lines; implementation awaits combined scope
approval. The old 29/1556 and 32/1699 estimates are superseded, not accepted
complete alternatives;31/+1817/-251/2068,35/+2365/-311/2676 and
33/+2420/-315/2735 are also superseded.
Source S2 is wholly outside this table. Source S8/S12 remain additive on shared
files, with no prior S2 allowance to deduct. Pricing's registry/native scope/
layout/phase/reporting hunks are mandatory additional allowances; its delivered
measured-CC variant is still unselected; delivered collective_sim known-interface
tables do not yet budget a selected finite full-duration model. The parent owns
that correction and has removed exact per-channel reconstruction as a pre-code gate.
Neither this complete CORE subtotal nor old companion arithmetic is
a final union claim. Parent must reconcile their corrected tables and count
shared paths once with each owner's disjoint hunks, not import a second CORE
expert adapter or pretend the unselected CC allowance is the complete repair.

**This focused delta stays inside the same33 paths.** Only eight existing
ceilings increase: A7 +6/-7 constructor; A9 +24/-6 independent pre-loader/
metadata branches; A10 +26/-8 shared pre-loader/context facts; A17 +12/-3 shared
preflight; B3 +24/-2 singleton replay; C4 +65 actual annotations/parser/pair and
cleanup; C5 +30 actual trainer pre-loader admission; C6 +18 constructor/order.
Total delta **+205/-26 =231 gross lines** over2735, justified by real caller
branches/record_function activation omitted from the preceding table. A3 uses
its existing binding/filter allowance rather than counting pricing's selector
twice. No common timer/tracer/singleton file, new registry generalization,
campaign/grid or CC algorithm hunk is added.

Unchanged: `ep_wave_inputs.py` already retains actual lane batches/padding;
`round_robin_cluster_scheduler.py` already calls the base producer;
`moe_mtp_replay.py` is not admitted for the new policy; `routed_layer_groups.py`
classifies precision, not routing; no new E2E matrix, README or GLOSSARY edits.
Implementation task records and the required size-exception update are parent
work; separately allow `docs/general/module-size-exceptions.md` at most+12/-0.
`expert_parallel.py`/`test_decode_ep_wave_materialization.py` new-policy PD-AF
work and `batch_builders.py` virtual marker are removed from this owner's scope.
That does not exempt mandatory pricing-owner changes to accepted shared waves.

## 10. Tests that settle the contract

1. Pure math vs direct `(p*K+j)%E` over interval multisets. E352/K8 plus K1,
   K=E and gcd>1; conservation, distinctness and per-interval max-min<=1.
   Split/merge/order invariance. Equal N at positions0 and6128 differ. The
   existing cache must distinguish those source keys; layers3 and88 agree.
2. Actual source extraction: frozen Batch context/query lists; request mutation
    after scheduling cannot change the Batch interval; DummyForwardBatch gives
    no interval. A real/dummy/one-request virtual batch without explicit source
    is rejected before cache/ratio lookup; no marker/inferred-source distinction.
    Same intervals/gathered width but swapped source IDs, local widths or live
    ownership must not reuse a cached lane-bound descriptor. TP1/TP8 produces
   identical assignment counts from the same2 DP owners, not8x assignments.
3. Through `prepare_ep_wave_inputs` then `prepare_moe_wave_from_inputs`, not
   by attaching a manually prebuilt count map: P6128 and D16 contexts6128,
   chunks[2048,2048,2032,16], full6144 reference, q1 context6144 and dummy lanes.
   Assert exact per-expert counts, all16 EP participants, live*K conservation,
   lane ownership and physical rows retained. D-tail counts begin96; q1 begins224.
   Two same-position q1 requests have count2 on experts224–231, intentionally.
4. Registry dispatch is called by the shared materializer; the four ratio
   policies preserve counts/imports/path labels. New-mode unsupported
   PD-AF/MTP/legacy admission fails without changing their old policy behavior.
   No new-policy tests/implementation of those deferred producers.
5. Native profiler adapter test stubs actual create_fused_moe_router,
    set_forward_context/set_routing_positions and select_experts: prove exact
    settings, tensors, requested dtype, factory result and nested/exception
    cleanup. Reject a standard/wrong router in position mode. Preserve the
    legacy direct fused_topk_bias branch. Negative masked IDs must not count or
    index experts; CORE passes the explicit source/prepared gathered samples to
    pricing's adapters and propagates returned native scopes/config/callable
    identity while router truthfully says Simulated. Pricing owns the original
    prepare/inclusive-root invocation and layout tests; compare native T2 later.
    CPU stubs prove integration, not parity or a forced expert routing method.
    In real main.profile_model executor/CSV stubs, one DP2 wave produces2 local
    rows plus16 EP rows; local widths8/16 and gathered24 remain8/16/24 after
    json_normalize/CSV roundtrip. Source rows carry gate/router plus prepare only
    for reached nondeferred FP8; BF16/deferred rows lack that target and all new
    gathered rows have one inclusive root, no separate shuffle. Spy on the
     pricing-owned get_moe_profiling_operators(context): producer, validator,
     loader, both enumerators and exact-query users consume the same spec set.
     C4 additionally exercises actual DeviceTimer record_function annotations and
     actual correlated-trace parser/count pairing, event-dict preservation and
     exception-safe singleton restoration as specified in6.2.1, not a count stub.
     C5 tests real trainer preflight/cache/loader paths with legacy helper traps
     and actual bound TP/EP/precision/source-context facts. Wrong context/source-rank
    association, missing applicable target/count, fabricated BF16 prepare/shuffle,
    copied opposite timing or omitted EP owner fails. Legacy dict still contains
    all four timings. Exercise sequential, multiprocessing worker and reachable
    Ray dispatch/remaining-result stubs without actual Ray/GPU. Load the resulting
    typed CSV through the existing loader and both fit branches: GateLinear/
    router/prepare never see24 from this wave, expert never sees8/16; cache cannot alias
    these frames. Duration/count exact maps retain their selected source rows.
6. Typed coverage: a positive physical frame/empty local lane reaches lookup,
    C6 first exercises actual constructor519-525 replacement before stubbed
    parent init: registered position policy passes; unknown names fail without
    invoking parent; all old selectors still normalize/validate as before.
   missing empty/new-policy rows raise, two physical widths do not alias at
   `(total_routed=0,padded=0)`. Put empty rows at widths8/24 and query missing
   width16 inside their bounds; require coverage error and no predict/cache
   fallback. Duration-only/count-only key, mismatched pair schema/source/target,
   negative duration and zero included-kernel count are rejected. Exercise the
   classifier through runtime and both trainer branches, verifying selected
   native schema and only applicable ranges on every precision pair. Query a
   missing in-bound empty source prepare/router and gathered root key; all fail.
   BF16/deferred prepare or separate shuffle is not enumerated or a zero row.
   Old four-policy maps/features/fixtures unchanged.
7. Generate a fresh position CPU fixture and run the required paired actual-main
   offline/online command. Assert section 7's 6144/1024/P6128/D16/q1/READ/capacity
   contracts plus canonical source/live/physical/empty-lookup proofs and the
   same applicable-target set across fixture/production. Historical
   balanced fixture/schema tests stay separate, never relabeled.

Targeted future command (not run in this design):

```bash
python -m pytest -q tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_moe_profiling.py tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_measured_ffn.py tests/integration/test_step5_measured_main.py
```

Parent also runs existing `step5_measured_compute_fixture.py` for a fresh
explicit position generation, then the existing `run_step5_measured_main.py`
with `--fixture-manifest <new-manifest> --output-root <new-output> --mode all`.
Generation/paths and native/pricing companion verification are required,
not replaced by the unit command. These are future commands, not design evidence.

Reuse existing admission/descriptor tests only if an affected check fails;
no broad tests-unit scans, new timing suites or wall/CPU-gap campaign.

## 11. Module depth, size, gain and recorded design checks

The new module's interface is a supplied immutable source plus existing
topology; its implementation owns validation, behavioral policy classification
and exact counts. The shared producer and explicitly sourced predictor entry
use one materializer; no guessed fallback is added. Deleting this module would
scatter math, source rules and policy dispatch into its callers. Source frames
and gathered frames are necessary distinct inputs to the profiler adapter,
not a second scheduler. The shared schema classifier earns its place by keeping
selection, two trainer branches/ranges and runtime construction consistent.

Each core edit earns its place: real intervals deliver reference-safe P/D/chunk
counts; live-vs-physical descriptor supports native padding; registry prevents
callers inventing policy aliases; profile interface allows truthful measurement;
empty-load lookup prevents unsupported free work. No second scheduler, extra
cache, model-name switch, hash, retry or compatibility mode is added.

Size inventory observed with Python `Path.read_text().splitlines()`, not guessed:
base scheduler1993, disaggregation3019, MoE predictor1793, trainer1849,
moe_ep_workload898. Base scheduler proposed+8/-3 gives1998; new math
and extraction must stay in `utils`/policy module. Disaggregation removes its
local enum and keeps role initialization/source-policy admission; net growth is
nonpositive within +18/-20. Its existing size exception remains justified, with
enum cleanup recorded. Current CORE ceilings: MoE predictor1793+86/-19 gives1860;
trainer1849+101/-28 gives1922; loader979+25/-2 gives1002;
moe_impl658+95/-18 gives735; wrapper929+184/-37 gives1076;
MoE main1218+115/-30 gives1303. Kernel adapter/native identity files are pricing-
owned, not zero-cost or included in these CORE net sizes. These are upper net sizes from this owner's hunks,
not the final sizes after other-owner additions. The small same-dict frame
protocol avoids a new serializer/output layer; the native router adapter belongs
beside its existing timers. CORE consumes pricing's actual native expert identity
instead of adding a second builder/identity hunk or altering backend selection. Do not
move a large unrelated mixin in this repair. If the scheduler bound cannot
hold, update the size/split analysis and ask parent before expanding.

Historical first-design verification (not rerun for this correction): an inline
`python3 - <<'PY'` probe in repository
root, Python3.10.6, compared closed-form
counts with direct per-position enumeration for200 deterministic random cases
(seed29; E/K352/8,7/3,8/8,11/1,10/4). PASS conservation, merge/split,
44-position cycle, distinctness and exact P6128+D16/full6144/chunks/q1 cases.
This is mathematics only; no producer code, native graph, timing or G3 run.
Example physical/live numbers in that probe were test inputs, not observed
worker widths.

Current arithmetic: A18/+1045/-205/1250; B5/+527/-87/614;
C7/+798/-1/799; D3/+255/-48/303; total33/+2625/-341/2966. Relative to
33/+2420/-315/2735, this focused delta adds +205/-26=231 gross across eight
existing paths, itemized in9; no new file. Closure evidence is inspected actual
constructor514-536, both trainer pre-loader branches and shared preflight,
DeviceTimer's singleton enum dispatch/annotations, and RecordFunctionTracer's
correlation parser. These are design closures, not executed implementation tests. This is a complete
CORE subtotal, not a complete task estimate. Source S2 has no overlap deduction;
S8/S12 and pricing native/registry/layout/reporting work remain mandatory additive
allowances; pricing/parent still must select finite collective_sim assumptions
and a complete numerical ceiling, following the latest plan33-40, not impose
exact native schedule reconstruction. D30's separate approval grants no D29 work.
Read-only document validation checks this arithmetic, scoped exact paths/code
fences and git diff --check; no new math campaign, tests, profiling, platform
operation, code edit or commit. The document is now
tracked by the parent; obsolete first-design force-add instructions do not apply.

## 1. 规划完成情况

**汇总：** PASS3｜BLOCKED/FAILED/SKIPPED1｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| constructor/两trainer/计数replay seam设计 | PASS | design5/6.2.1：实际constructor、pre-loader helper路径、singleton切换和实际annotation/parser测试已明确 |
| pricing selected-spec/typed schema/exact pair接口 | PASS | 接受唯一get_moe_profiling_operators；条件MOE_FAMILY prepare，local/gathered与typed precision/context不重查旧helper |
| 本侧mandatory文件/行数/fixture设计 | PASS | design9/11；fixture必需，33文件+2625/-341=2966gross CORE subtotal；8路径补齐231gross |
| implementation/tests/native/platform/commit | SKIPPED | 本child只修改design；未获实施范围批准 |
| pricing有限CC模型/联合scope/native证据 | TODO | pricing已交付compute接口；parent要求有限模型和完整ceilings，native/image/count/runtime均未验证 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** D29-CORE-SEAM-CORRECTION请求至今｜**当前前沿：** parent验收本侧3个seams、pricing有限CC模型及完整scope整合

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只有decisive接口核实和design算术/格式检查，无test campaign |
| D2 细节反复重试 | CLEAR | 无重复behavior fixes/test reruns；一次patch错误尾部上下文去除后落地，未形成连续失败循环 |
| D3 支线问题修补 | CLEAR | 所有改动对应review correction；未重查其他owner native scope |
| D4 可并行任务停滞 | CLEAR | companions由父端另行委派，child禁止新agents |
| D5 超范围镀金 | CLEAR | 仅3个已请求seams/既有测试8路径补齐ceilings，不增加文件、selector或CC算法 |
| D6 分析空转 | CLEAR | design为所请求交付；已写明确seams/ceilings/dependencies |
| D7 路线违背 | CLEAR | 无D29实施；D30独立批准不外推；collective_sim保留且接受parent有限模型/非exact重建的plan纠正 |

**文档检查：** S2/S5：前版design5/6/9遗漏constructor/pre-loader/store切换，与review166-187及实际源码矛盾；已补明确所有权与8路径ceilings。S6：旧本侧第二selector及prepare family表述与pricing5.1不一致，已接受唯一selected-spec接口/条件MOE_FAMILY。旧CC exact重建前置门与最新plan33-40冲突，已移除本侧重述；其模型仍由pricing/parent纠正。S1/S3/S4未见新增缺陷。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：补齐3个真实seams及pricing接口后再联合审批；CC有限模型/假设由已有owner处理，不实施measured-lookup替代。

**文档更新（先执行）：**
1. 本design：完成constructor、实际pre-loader binding facts和singleton replay/实际annotation测试设计；33/+2625/-341/2966，fixture必需（已完成）。
2. 本task `progress.md`/review和parent联合scope：记录3个设计closure和231gross增量，按当前companion表计数；parent更新，本child无写权限。
3. 联合scope：父端整合各owner文件/行数、共享接口及明确所有权，获得完整扩展批准（需用户确认具体数值，不重开已决定的fixture或position政策）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. parent核对本侧3个seams的设计closure；并行由pricing既有owner补有限collective_sim模型/显式假设及完整ceilings，D30继续其独立获批scope → 完成标志：本侧无caller gap、模型scope可有限批准；不猜native parity或measured-CC替代。
2. 联合scope审批 → 完成标志：合并真实distinct文件和所有必需改动ceilings，fixture保留，用户批准。
3. 获批后实施/验证 → 完成标志：聚焦producer/schema/exact-coverage与paired actual-main通过，再由父端取native/G3新证据；无缺测量fallback。

## 4. 最值得关注的 3 点

1. Constructor在parent init前解析count registry；两trainer在任何旧family helper/loader前消费同一selected spec的TP/EP/typed precision/context事实。
2. Count replay真实切换已有singleton至RECORD_FUNCTION并在finally还原/clear；测试实际annotation和trace parser配对，不以stub counts冒充验证。
3. 本侧33文件+2625/-341=2966gross含mandatory fixture；完整联合scope和finite collective_sim假设仍需parent审批，native/runtime未验证；D30批准不外推D29。
