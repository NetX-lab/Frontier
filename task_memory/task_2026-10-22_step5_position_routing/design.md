# D29 position_cyclic — Frontier and profiling design

TaskID: D29-FRONTIER-POSITION-DESIGN (child of D29; parent owns tree, commit and GPU work).

Source references are to the inspected live tree initially supplied at
`fa5bae2`; parent advanced HEAD to `9ed57630f2f564498f9706c7a5af0243928d1370`
while this read-only design ran. Implementation must recheck its listed seams
against the integrated tree. No child code step or commit exists.

## Modification History

| Date | Change |
| --- | --- |
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

The existing file had this task ID, although the work order reported no retained
design. It was not in the Git index. This revision is confined to that draft.

This document owns only the Frontier simulator plus the existing MoE
profiling/producer metadata. The native router/context implementation, image,
pins and generation are the native agent's design. Nothing here is
implemented, run on GPU, or committed.

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

There are exactly three materialization entry points, all already holding the
source batches:

| Entry | File:lines | Wave identity available | Interval source available |
| --- | --- | --- | --- |
| Shared PREFILL / DECODE / MONOLITHIC layer wave | `frontier/scheduler/utils/ep_wave.py:96-105` calls `materialize_workload(batch=aggregate_batch, ...)`; `_prepare_moe_ep_wave_plan` at `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py:1061` | `wave_inputs.source_batches` (built by `frontier/scheduler/utils/ep_wave_inputs.py:20`) | Not passed today; `aggregate_batch` has lost per-request identity |
| PD-AF DECODE_FFN EP wave | `frontier/scheduler/utils/expert_parallel.py:364` `materialize_wave_workload(group, ...)`; caller `frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py:957` | `group: list[(Batch, transfer_info)]` | Batches are in hand but only `total_num_tokens` is summed (line 383-394) |
| Predictor fallback (monolithic decode-sync skew, MTP replay, `_simulate_routing_per_layer`) | `frontier/execution_time_predictor/moe_routing_workload.py:392`, callers `moe_operator_times.py:110-181`, `moe_mtp_replay.py:53`, `moe_routing_workload.py:632` | Real batch for single-lane decode; aggregate otherwise | Real batch only; aggregate has no intervals |

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

### 4.1 New module `frontier/moe_routing_policy.py` (at most 180 added lines)

One registry, one math owner:

```python
# Move WorkloadDistributionType here; each enum entry declares its string value,
# profiling_runtime_path and requires_logical_positions in one place.
# Existing imports from the disaggregation module still see the imported enum.
# BALANCED = ("balanced", "standard_fused_topk", False)
# RANDOM   = ("random", "uniform_topk", False)
# SKEWED   = ("skewed", "standard_fused_topk", False)
# ZIPF     = ("zipf", "standard_fused_topk", False)
# POSITION_CYCLIC = ("position_cyclic", "position_cyclic", True)
ROUTING_COUNT_POLICIES = {entry.value: entry for entry in WorkloadDistributionType}
def resolve_routing_count_policy(distribution_type: str) -> WorkloadDistributionType: ...

@dataclass(frozen=True)
class TokenInterval:            # start >= 0, length > 0
    start: int
    length: int

@dataclass(frozen=True)
class WaveRoutingSource:        # one Replica-local EP wave, all DP lanes
    policy_id: str
    physical_rows: int          # agreed padded compute width of the wave
    intervals: tuple[TokenInterval, ...]
    # Retain source lane identity, actual bound local width and live row count;
    # required for local-router pricing and physical collectives, not routing.

def build_wave_routing_source(source_batches, *, policy_id, physical_rows) -> WaveRoutingSource: ...
def position_cyclic_expert_counts(intervals, top_k, total_expert_num) -> dict[int, int]: ...
```

`build_wave_routing_source` reads only existing metadata. Batches must be actual
request owners, never virtual aggregate or EP assignment batches. Keep count
mathematics free of scheduler state; defer the `Batch` import inside extraction
if needed to avoid an entities import cycle.

- `physical_rows` is supplied by the existing aggregate width, not computed by
  another DP-padding algorithm. The shared path uses the aggregate batch's
  compute getter; PD-AF uses its existing `pre_routing_tokens` sum. For an
  unagreed legacy graph cohort, `ep_wave_inputs.py:80-86` applies a maximum, so
  summing the original lane getters alone is not always the aggregate width;
- keep source lane metadata from the existing mapping/forward descriptors.
  In measured Step, those descriptors contain post-agreement local widths.
  Do not infer an unbound lane's native local width from aggregate/N or duplicate
  legacy graph agreement; mark it unavailable for measured companion pricing;
- `intervals` from `zip(batch.num_context_tokens, batch.num_tokens)`, skipping
  `DummyForwardBatch`; no request-id phase and no independently tracked position;
- validation: per real lane `len(requests) == len(num_tokens) == len(num_context_tokens)` and
  `sum(num_tokens) == batch.total_num_tokens` (live scheduled rows),
  `sum(intervals) <= physical_rows`. Explicitly mark and reject virtual batches:
  a shallow copy of a one-request batch can otherwise pass length checks.

No policy id is invented at this seam: the caller supplies the configured
policy, and the materializer rejects a source whose `policy_id` is not the one
the predictor was constructed with.

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
  consumers must retain the position-policy identity and physical width after
  lane construction. Do not infer these from artificial expert Requests.

`materialize_layer_ep_workload(..., routing_ratios=None, routing_source=None)`
(`moe_ep_workload.py:810`): exactly one source is required.

- ratio path: unchanged code and numbers;
- position path: `global_per_expert_tokens = position_cyclic_expert_counts(
  source.intervals, router_topk, total_expert_num)`, validate
  `source.physical_rows == routing_token_count`, `live = sum(lengths)`;
  the EP ownership split and every existing conservation check are reused
  unchanged; a zero-live lane yields an all-zero lane that remains a valid EP
  participant (`tests/unit/test_moe_ep_aggregate_admission.py:570` already
  pins that).

### 4.3 Which consumer uses which count

| Consumer | Count | Location | Change |
| --- | --- | --- | --- |
| `moe_gating_linear`, `moe_gating_routing_topk` lookup | source-local physical rows | `moe_operator_times.py:387-435` via the batch compute getter | current EP virtual batch uses aggregate width; companion needed (4.6) |
| grouped-GEMM features `total_routed_tokens`, `num_tokens_post_padded` | local live assignments; global expert-block padding | `moe_routing_workload.py:577-604`, `frontier/moe_load_imbalance.py:33-56` | retain these; add physical `num_tokens` for the position path |
| grouped-GEMM `BLOCK_SIZE_M` | physical | `block_size_m_for_tokens(model ranges, pre-routing tokens)` | none |
| `moe_shuffling` | physical rows + live/global assignment shape | `moe_operator_times.py:447-500` | bypass its zero-load shortcut only for position policy; require measured features |
| EP dispatch/combine payload | current live assignments, required physical source rows | `frontier/operators/families.py:540-555`, `moe_operator_times.py:244-282,512-585` | communication companion required (4.6) |
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
4. `expert_parallel.py:364-417` receives policy and role through
   `base_cluster_scheduler.py:679-709`, extracts intervals from original group
   batches, and uses physical compute widths for the position policy. Keep the
   old `sum(total_num_tokens)` ratio path. At `:512-516`, compare position live
   count with `total_tokens`, and physical width with `pre_routing_tokens`
   (`:489-490`). The old check is against query rows, not padded physical rows.
   Do not alter PD-AF transfer or queue ownership.

Predictor boundary (`frontier/execution_time_predictor/moe_routing_workload.py:392`):
`_materialize_layer_ep_workload(batch, cluster_type, layer_id, routing_source=None)`
builds a source locally only for a genuine request batch or known dummy. It
rejects a marked virtual batch and an EPBatchGroup without an explicit source.
This covers the monolithic decode-sync caller
(`moe_operator_times.py:110`) and `moe_mtp_replay.py:53` without touching the
scheduler. Length checks alone do not prove a genuine batch; the marker described
below is required for the reachable one-request virtual aggregate case.

The existing per-layer LRU cache (`moe_routing_workload.py:392-444`) keeps its
ratio keys. Append exact immutable intervals and physical width for the new
policy; equal scalar N at position0 and6128 must not reuse counts. No new cache.
`batch_builders.py:95-98` marks virtual inputs aggregate-only; fallback cannot
infer request positions from their shallow-copied context list. Missing source
raises, never resets positions or substitutes Hamilton.

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
- position-aware callers branch before `resolve_routing_details`
  (`moe_ep_workload.py:649`), whose existing missing-layer error rejects an
  accidental lookup in an empty table. `_get_routing_details_for_cluster`
  (`moe_routing_workload.py:368-386`) can give the explicit policy error. Do not
  teach a policy-free ratio resolver to invent a new position table;
- `_emit_routing_details_snapshot` (`moe_routing_workload.py:102-113`) is
  skipped for position policies and replaced by a
  `[ROUTING-POLICY-SNAPSHOT]` line carrying `policy_id`, `router_topk`,
  `total_expert_num` and the interval contract, never a ratio map.

### 4.6 Physical collectives: required companion, not zero-load credit

The frozen Step dispatch is a DP `all_gatherv` + `reduce_scatter` over physical
rows, while Frontier's `expert_parallel_alltoall_*` operators price the
assignment-based all-to-all (`families.py:540-555` uses
`lane.routed_token_count`). This design must not silently change that identity
for the four existing policies, and must not pretend a padding-only lane costs
nothing to communicate. Therefore:

- keep legacy policies unchanged; do not claim assignment-based all-to-all is
  an aligned measured Step `position_cyclic` collective;
- expose `LayerEPWorkload.routing_token_count` (physical) so an approved
  physical-payload/collective-identity change has the data it needs;
- require a communication-owner companion before measured campaign acceptance,
  or explicitly leave this outcome count-only; no silent cost patch.

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
cannot be described as already correct for gating. A measured companion needs
separate source-router and gathered-expert widths, using the existing rank
mapping/topology. Do not divide by EP16 or invent an averaged lane width. Keep
the source-lane map in the descriptor so this does not require a new scheduler.

Actual current Step calls are named all-to-all dispatch/combine in
`moe_operator_times.py:235-272`; payload comes from `families.py:540-555`.
DSV4 registrations at `families.py:811-829` show the existing physical
all-gather/reduce-scatter mechanism, but their SP/EP group contract is not the
Step contract. Do not relabel them or turn the observed DP2 all-gatherv group
into EP16. Exact group/payload selection belongs to the communication owner.

## 5. Registry consolidation (no model-name switches)

One registry replaces four duplicated value lists and one enum:

| Site | Current | Change |
| --- | --- | --- |
| `frontier/config/replica_config.py:91-97,178-192` | hardcoded `{balanced, random, skewed, zipf}` | validate against `ROUTING_COUNT_POLICIES`; help text updated |
| `frontier/moe_ep_workload.py:32-86` | ratio generator with its own string dispatch | keeps the four ratio branches; `position_cyclic` raises "no expert ratio table; supply a wave routing source" |
| `frontier/moe_routing_runtime.py:6-43` | path constants + distribution→path map | map through the registry; add `position_cyclic` |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py:59-89` | `WorkloadDistributionType` enum | move enum declarations to the policy module and import them here; resolver delegates to the registry; `.value` and existing test imports remain unchanged |
| `tests/e2e/moe_ep_baseline_replay.py:43-56` | frozenset + old-runtime translation | unchanged for the four existing values; the old runtime has no position mode, so no translation is added (see 7.3) |

Selection is by configured `moe_routing_distribution_type` only. No
`model_type`/`model_name` branch is added anywhere, including
`_ROUTED_LAYER_GROUP_RESOLVERS` (`frontier/profiling/moe/routed_layer_groups.py:112`)
and the profiling model-type policy.

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

### 6.2 Required position/mask profiler interface

The runtime requests `routing_runtime_path="position_cyclic"`. Add one explicit
input in `frontier/profiling/moe/inputs.py`, passed by the existing `main.py` to
`MoEWrapper.profile` and its three existing operations:

```python
@dataclass(frozen=True)
class MoERoutingProfileInput:
    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    physical_rows: int
    ep_id: int               # explicit owner whose local expert shape is timed
    # Empty query/context tuples mean all-padding, never implicit position0.
```

Require equal array lengths, positive query lengths, nonnegative contexts,
positive physical width and `sum(query_lens)<=physical_rows`. Flatten
`context+arange(query_len)` in request order and append masked rows to the
physical width. Dummy frames have no intervals and an all-true mask. Existing
Step profile inputs (`profiling/step5/inputs.py:16-25`) and Batch context/query
lists supply the metadata; no cursor. Attention dummy descriptors contain
fake physical query slots, which must not become real MoE intervals.

One MoE input-file option is required for the new path; no guessed
`arange(num_tokens)` default. Old scalar profiling paths remain unchanged.
Thread the input through multiprocessing (`main.py:169-173`), Ray (`:867-873`)
and sequential (`:940-945`) calls. Extend existing `_prepare_routing_inputs`,
whose one sample is already shared by shuffling and GEMM, rather than add a
second routing generator. The native owner supplies the actual tensor router;
the Frontier wrapper adapts its interface and timers:

- a gating module that takes the **positions tensor** and the **padding mask**
  in addition to hidden states, selects IDs by
  `(positions.unsqueeze(1)*K + arange(K)) % E`, sets masked IDs to-1 and weights
  to0 under `VLLM_MOE_SKIP_PADDING=1`, and computes weights from
  the unbiased sigmoid scores of the selected experts with the configured
  renormalization and `routed_scaling_factor` — i.e. the same operator the
  serving graph runs, not a re-implementation;
- row metadata: `routing_runtime_path="position_cyclic"`,
  `routing_assignment_policy="position_modulo_experts"`,
  `routing_weight_policy` = the existing `sigmoid_renorm_scaled` /
  `sigmoid_scaled` value from `SigmoidBiasGatingNetwork.routing_weight_policy`,
  and a new `routing_assignment_uses_positions: true` plus
  `routing_weight_uses_scores: true`; keep `routing_uses_router_logits=true`
  because it consumes logits for weights, and explicitly declare
  `routing_assignment_uses_scores=false` rather than overload that boolean;
- retain exact query/context arrays, physical/live/padding rows, profiled EP
  rank and observed expert counts. `moe_wrapper.py:304-306` currently fixes
  representative EP rank0; collect every distinct reached local load shape,
  not merely rank0. Counts may match `uniform_topk` at frame0, but timing reuse
  also requires matching actual expert kernel, mask, physical shape and scope;
- `standard_fused_topk` (generic metadata says logit-topk/softmax; Step routed
  groups override this with their actual biased-sigmoid semantics)
  and `uniform_topk` (uniform `1/K` weights, gate result unused) must not be
  relabelled: the new policy differs from the first in assignment and from the
  second in weights, mask semantics and position domain.
- CLI/wrapper plumbing: `frontier/profiling/moe/main.py:340-348` choice list,
  `frontier/profiling/moe/moe_wrapper.py:104-114` (the routed-layer-group guard
  currently forbids any non-`standard_fused_topk` path, because the group times
  its own router; the new path must be admitted through the registry, and the
  group router metadata must carry the policy id), and
  `examples/profiling/profile_moe.sh:23,75,105` already passes the selected path
  through; its old default stays unchanged and it needs no edit.

Two mandatory mask-consumer fixes are within profiling scope:
`moe_wrapper.py:336-338` must filter global IDs-1 **before** indexing expert_map
(Python negative indexing otherwise aliases the last expert); and
`load_distribution.py:178-181` must exclude the masked sentinel before bincount.
GEMM/shuffling still receive tensors with the full physical row dimension.
The routed-layer-group classifier stays unchanged: policy selection is input,
not a new Step model or precision group.

### 6.3 Typed duration/count/feature coverage: no empty-load credit

Observed current behavior, not a timing claim:

- `moe_operator_times.py:479-480,497-498,653-654,660-662` returns0 for an empty
  local expert lane before measured lookup. With the position policy this must
  instead query the actual positive-physical-row operator scope. Old policies
  and explicitly synthetic dummy mode remain unchanged.
- An empty **local** lane is not the same as an all-padding wave.
  `moe_routing_workload.py:595-603` pads `global_token_counts` over all experts.
  An empty local lane may therefore have `total_routed_tokens=0` but
  `num_tokens_post_padded>0`; only a wave with no global live routes has both0.
  Do not force padded0 from local live0.
- `moe_dataset_training.py:50-81` selects two padded grouped-GEMM features,
  or the full load feature set; neither contains physical width. Positive
  physical frames of different widths can alias at empty load. For rows
  explicitly selected as `position_cyclic`, add `num_tokens` to the existing
  grouped-GEMM and shuffling feature selection, retain the native padded/global
  features, and use `num_tokens` plus live rows for routing-topk. Validate that
  the selected new-policy rows contain all required fields; do not pad a
  partially stamped table with defaults or alter old feature schemas.
  Filter position rows for all3 affected operators before deriving schema, not
  just routing-topk; otherwise mixed legacy/new tables can contaminate feature
  and measurement identity. Keep this in existing MoE training owners.
- `step5.py:495-566` requires measured median duration>0 and actual integral
  kernel_count>=1 for an included physical operator. Physical dimension
  metadata (`operators/typed_contracts.py:31-51,269-319`) is not a live-route
  count and need not be weakened. `sklearn_moe_execution_time_predictor.py:106-165`
  preserves duration/count identity and feature bounds; kernel-only mode also
  requires the existing GPU kernel-gap table. No new CPU-gap mechanism.
- `sklearn_execution_time_predictor.py:4840-4849` rejects out-of-bound feature
  values before prediction. The current nonempty-only fixture has a positive
  minimum routed count. Let missing empty/new-policy coverage fail clearly;
  do not extrapolate, insert a fake zero row, or reuse another routing identity.
- `frontier/moe_load_imbalance.py:41-56` rejects physical width above the
  profiled BLOCK_SIZE_M range. Do not widen the range by copying the last row.

Required future native measurements at each reached physical width/precision
and compatible kernel scope:

1. Router/gate for live, mixed-padding and all-padding source-local frames,
   executing the new native tensor router with observed positions/mask.
2. Shuffling/alignment and grouped expert scope for empty local lanes in a live
   global wave, concentrated nonempty lanes, and globally all-padding waves.
   Record actual IDs, local/global counts, BLOCK_SIZE_M/ranges, native padded
   count, durations and kernel counts. E352/EP16 all22-expert lanes participate.
3. Physical dispatch/combine and source-router-width companion in4.6, from the
   actual TP8/DP2/no-SP route. Existing isolated TP1/EP16 profile points do not
   establish whole serving collective identity.

The actual empty execution may launch a positive-duration scope or may omit a
particular subkernel. Source inspection alone does not settle that. If the
included operator really has zero kernels, the current positive-scope contract
cannot admit it: parent must obtain measured evidence and decide an explicit
inactive-scope contract. That is not permission to invent duration/count0.

### 6.4 Existing producer consequences

`profiling/step5/job.py:197-204` currently hardcodes `standard_fused_topk` and
only scalar `moe_tokens`. Add an explicit routing argument/input path for the
new generation; keep old P0 plans immutable. A new JSON MoE input file, built
from the census, is necessary. `p0_prefill_inputs.json` and
`p0_decode_inputs.json` contain actual query/context pairs but representative
attention layers, not the EP wave's lane ownership or full MoE census. They
must not be relabeled complete new-policy coverage.

No edit to `routed_layer_groups.py`, existing P0 JSONs, shell wrappers or typed
physical-dimension contracts is necessary for this policy. FP8 layers3–87,
BF16 layers88–90 and swiglustep stay exactly the same group identities. Native
DeepGEMM/Triton dispatch/E8M0 still requires actual receipts; count equality is
not an expert-kernel reuse proof.

## 7. Fixture and runner consequences

- `tests/integration/step5_measured_compute_fixture.py:119-137` explicitly
  materializes Hamilton-balanced scalar widths and skips every empty EP lane.
  Its config at218 and manifest at251-252 also say `balanced`. New-policy
  count/lookup integration requires new interval/mask inputs, all EP lanes,
  the position identity, and a fresh fixture generation. Its synthetic times
  must stay `evidence_kind=cpu_fixture`, never native timing proof.
- `tests/integration/run_step5_measured_main.py:73-74,127-128` requires and
  passes `balanced`/seed42. It cannot validate a position fixture unchanged.
  Preserve legacy fixture support; accept a separately identified position
  fixture only after its owning inputs and coverage tests exist. Native agent
  owns actual serving selector/reference generation, not this CPU runner.
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

## 8. Decisions and blockers for parent

Already decided: `position_cyclic`, mask preservation, selected native weights,
no request phase/layer rotation, same-position concentration, independent G3.
There is no remaining label or per-wave-balance choice.

1. **Repair expansion:** approve the bounded file/line list below together with
   the native design. Even count-only scope exceeds5 files/500 gross lines.
2. **Measured companion:** assign actual source-local router width and physical
   DP gather/combine mapping to the communication/pricing owner, or explicitly
   make the first deliverable counts/metadata only. Existing calibrated
   all-to-all numbers do not establish the new Step identity. This companion is
   necessary for measured end-to-end acceptance, not a routing improvement.
3. **Measured empty scope:** schedule the stated native rows through the parent.
   Until present, fail coverage; a measured zero-output scope requires explicit
   inactive-scope design after observing its actual launch behavior. No fallback
   or arbitrary cost is proposed.
4. **Existing fixture/runner inclusion:** if parent needs
   `run_step5_measured_main` acceptance in this repair, include its3-file
   companion below. Otherwise report it as intentionally still a legacy fixture,
   and use focused producer/typed-coverage proofs only.

## 9. Exact minimum code/test paths and gross line ceilings

Ceilings are planned added+deleted lines, **not** net growth or measured diff.
No implementation has occurred. Documentation here is not included in the code
ceiling. These seams are the minimum complete count plus profile interface;
source-local router/physical communication behavior is not estimated as though
settled. Native router/context/image files belong to the other design.

### A. Frontier count source, descriptor and measured-coverage seam

All paths relative to repository root:

| Exact file | Existing size | + / - ceiling | Reason / decisive seam |
| --- | ---: | ---: | --- |
| `frontier/moe_routing_policy.py` (new) |0|180 /0| registry, source validation, interval math |
| `frontier/moe_ep_workload.py` |898|65 /10|134-181,374-388,810-855: live ledger and shared split |
| `frontier/moe_routing_runtime.py` |67|25 /15|6-43: registry paths, shared row selection |
| `frontier/config/replica_config.py` |269|6 /10|91-97,178-192: registry validation/help |
| `frontier/execution_time_predictor/moe_routing_workload.py` |671|80 /20|268-444: no ratio fabrication, source-aware cache/fallback |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` |3019|18 /20|59-89,430-514: import enum, skip ratio initialization |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` |1793|25 /8|106-165,516-524: policy validation/new measured features |
| `frontier/execution_time_predictor/moe_operator_times.py` |744|55 /14|387-435,479-508,653-668: physical features, no new-policy zero shortcut |
| `frontier/execution_time_predictor/moe_dataset_training.py` |493|20 /3|50-81: physical feature/selected new-policy rows |
| `frontier/execution_time_predictor/prediction_family_trainers.py` |1849|12 /3|273-377: same feature selector and row identity |
| `frontier/scheduler/utils/ep_wave.py` |164|20 /2|109-131: actual source lanes and live conservation trace |
| `frontier/scheduler/utils/layer_workload.py` |67|35 /8|13-67: role policy/source into core |
| `frontier/scheduler/utils/expert_parallel.py` |823|55 /14|364-417,489-516: PD-AF source/physical count |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` |1993|12 /5|679-709,899-915: threads source/role policy |
| `frontier/scheduler/utils/ep_trace.py` |542|20 /8|496-542: live ledger and physical/padding trace |
| `frontier/scheduler/utils/batch_builders.py` |111|2 /0|95-98: reject virtual fallback; propagated source stays in descriptor |

A: **16 files, +630/-140 =770 gross lines**.

### B. Existing MoE profiling interface and producer

| Exact file | Existing size | + / - ceiling | Reason |
| --- | ---: | ---: | --- |
| `frontier/profiling/moe/inputs.py` (new) |0|65 /0| explicit query/context/physical frame |
| `frontier/profiling/moe/moe_impl.py` |658|65 /12| native router adapter/timer/metadata, not duplicate tensor math |
| `frontier/profiling/moe/moe_wrapper.py` |929|95 /20| input/mask, one shared native sample, correct sentinel counting |
| `frontier/profiling/moe/main.py` |1218|65 /12| input option, all3 execution paths, registry choice |
| `frontier/profiling/moe/load_distribution.py` |275|3 /2| skip IDs-1 before existing bincount |
| `frontier/profiling/step5/job.py` |279|8 /3| explicit new routing input instead of relabeling P0 command |

B: **6 files, +301/-49 =350 gross lines**. The new census-derived JSON is a
future data artifact, not an invented input grid; parent chooses its path with
the generation after actual census. No package/dependency edit.

### C. Minimal persistent tests

| Exact file | + / - ceiling | Acceptance |
| --- | ---: | --- |
| `tests/unit/test_moe_position_routing.py` (new) |180 /0| math, input proof, split/cache/concentration invariance |
| `tests/unit/test_moe_routing_conservation.py` |70 /1| actual public wave producer + physical/live trace |
| `tests/unit/test_decode_ep_wave_materialization.py` |45 /0| PD-AF source producer and physical widths |
| `tests/unit/test_moe_routing_runtime.py` |20 /0| canonical path and old values unchanged |
| `tests/unit/test_step5_moe_profiling.py` |55 /0| native adapter metadata, mask sentinel, input plumbing |
| `tests/unit/test_moe_grouped_gemm_padded_features.py` |30 /0| local-empty/global-live vs all-padding; physical feature |
| `tests/unit/test_step5_measured_ffn.py` |35 /0| missing/zero-duration/count/feature coverage rejection |

C: **7 files, +435/-1 =436 gross lines**.

**A+B+C:29 code/test files, +1366/-190 =1556 gross-line ceiling**, before
native design, communication companion, campaign artifacts and task records.
A plus its first4 focused tests is20 files/1086 gross lines. Neither fits the
5-file/500-line repair bound; do not implement without parent's expansion.

### D. Optional existing-fixture acceptance companion

| Exact file | + / - ceiling |
| --- | ---: |
| `tests/integration/step5_measured_compute_fixture.py` |70 /20|
| `tests/integration/run_step5_measured_main.py` |12 /6|
| `tests/integration/test_step5_measured_main.py` |35 /0|

D:3 files/143 gross lines. If included, **32 code/test files/1699 gross lines**.
This remains CPU contract verification, never native timing evidence.

Unchanged: `ep_wave_inputs.py` already retains actual lane batches/padding;
`round_robin_cluster_scheduler.py` already calls the base producer;
`moe_mtp_replay.py` already passes the real batch; `routed_layer_groups.py`
classifies precision, not routing; no new E2E matrix, README or GLOSSARY edits.
Implementation task records and the required size-exception update are parent
work; allow `docs/general/module-size-exceptions.md` at most+12 lines/-0.

## 10. Tests that settle the contract

1. Pure math vs direct `(p*K+j)%E` over interval multisets. E352/K8 plus K1,
   K=E and gcd>1; conservation, distinctness and per-interval max-min<=1.
   Split/merge/order invariance. Equal N at positions0 and6128 differ. The
   existing cache must distinguish those source keys; layers3 and88 agree.
2. Actual source extraction: frozen Batch context/query lists; request mutation
   after scheduling cannot change the Batch interval; DummyForwardBatch gives
   no interval; one-request virtual aggregate is rejected. TP1/TP8 produces
   identical assignment counts from the same2 DP owners, not8x assignments.
3. Through `prepare_ep_wave_inputs` then `prepare_moe_wave_from_inputs`, not
   by attaching a manually prebuilt count map: P6128 and D16 contexts6128,
   chunks[2048,2048,2032,16], full6144 reference, q1 context6144 and dummy lanes.
   Assert exact per-expert counts, all16 EP participants, live*K conservation,
   lane ownership and physical rows retained. D-tail counts begin96; q1 begins224.
   Two same-position q1 requests have count2 on experts224–231, intentionally.
4. PD-AF `materialize_wave_workload` gets source batch intervals; physical
   width uses the existing DECODE_FFN getter. Same live intervals yield the
   same expert counts as shared P/D. No request or transfer queue mutation.
5. Native profiler adapter test stubs the actual router interface to prove
   positions and bool mask are supplied, selected score settings are forwarded,
   negative masked IDs are not counted/indexed as experts, and the shared sample
   reaches both shuffling and GEMM. No stand-in tensor router is accepted as
   native timing proof.
6. Typed coverage: a positive physical frame/empty local lane reaches lookup,
   missing empty/new-policy rows raise, two physical widths do not alias at
   `(total_routed=0,padded=0)`, negative duration/zero included-kernel count are
   rejected. Old four-policy maps/features and fixtures remain unchanged.

Targeted future command (not run in this design):

```bash
python -m pytest -q tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_conservation.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_moe_profiling.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_measured_ffn.py
```

Reuse existing admission/descriptor tests only if an affected check fails;
no broad tests-unit scans, new timing suites or wall/CPU-gap campaign.

## 11. Module depth, size, gain and recorded design checks

The new module's interface is source intervals/physical rows plus the existing
topology; its implementation owns validation, policy classification and exact
counts. This seam gives both real producers and predictor fallback the same
behavior. Deleting it would scatter position math and validation into all3
paths. The profile input module owns a genuinely different adapter: an explicit
frame is turned into native tensors, not another simulator scheduler.

Each core edit earns its place: real intervals deliver reference-safe P/D/chunk
counts; live-vs-physical descriptor supports native padding; registry prevents
callers inventing policy aliases; profile interface allows truthful measurement;
empty-load lookup prevents unsupported free work. No second scheduler, extra
cache, model-name switch, hash, retry or compatibility mode is added.

Size inventory observed with Python `Path.read_text().splitlines()`, not guessed:
base scheduler1993, disaggregation3019, MoE predictor1793, trainer1849,
moe_ep_workload898. Base scheduler proposed+12/-5 stays exactly2000; new math
and extraction must stay in `utils`/policy module. Disaggregation removes its
local enum and keeps role initialization; net growth is nonpositive. Its
existing size exception remains justified, with enum cleanup recorded. Do not
move a large unrelated mixin in this repair. If the scheduler bound cannot
hold, update the size/split analysis and ask parent before expanding.

Read-only design verification: a `python3 - <<'PY'` inline probe in repository
root, Python3.10.6, compared closed-form
counts with direct per-position enumeration for200 deterministic random cases
(seed29; E/K352/8,7/3,8/8,11/1,10/4). PASS conservation, merge/split,
44-position cycle, distinctness and exact P6128+D16/full6144/chunks/q1 cases.
This is mathematics only; no producer code, native graph, timing or G3 run.
Example physical/live numbers in that probe were test inputs, not observed
worker widths.

The same inline Python mechanism verified every table's added/deleted sums,
existing code/test path existence, and absence of trailing whitespace. PASS:
A16/+630/-140/770; B6/+301/-49/350; C7/+435/-1/436; D3/+117/-26/143.
`git ls-files --error-unmatch <design-path>` exited1 (not tracked);
`git check-ignore -v <design-path>` showed `.gitignore:171:task_memory/*`.
Parent must force-add only the owned design when recording the approved task.

## 1. 规划完成情况

**汇总：** PASS3｜BLOCKED/FAILED/SKIPPED1｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 实际 source/interval/physical producer trace | PASS | design3–4记录真实调用与缺失的source seam |
| profiling identity/mask/coverage设计 | PASS | design6–8记录empty/physical区别与缺测量阻塞 |
| 文件/行数/最小验证交付 | PASS | design9–11；scope算术/path检查通过 |
| implementation/native测量/platform操作 | SKIPPED | 本child明确仅设计，无其他文件权限 |
| 父端联合范围确认 | TODO | 待合并native设计与通信计费companion范围 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** fresh child开始至当前设计交付｜**当前前沿：** 联合范围确认

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只有定位/设计验证，无测试平台扩建 |
| D2 细节反复重试 | CLEAR | patch上下文失配已按实际文件修正，无行为fix循环 |
| D3 支线问题修补 | CLEAR | 所有发现都影响所请求的source/计数/profile可信度 |
| D4 可并行任务停滞 | CLEAR | native由父端独立协调；child禁止新agent |
| D5 超范围镀金 | CLEAR | 未实施任何代码；必要范围与companion分开列出 |
| D6 分析空转 | CLEAR | design本身是交付；已形成确定seams/预算/阻塞 |
| D7 路线违背 | CLEAR | 保留position/mask/权重/G3，无reset或替代路由 |

**文档检查：** S5：旧草稿有不成立的零计费、frame0复用与净行数结论，已在本design修正；D29旧proposal/progress仍说decision pending，与root requirements191–215矛盾。S1–S4、S6未发现明显缺陷。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：spec/tickets存在陈旧记录，先更新文档再继续。

**文档更新（先执行）：**
1. 本design：修正旧结论并按gross行数列scope（已完成）。
2. `task_memory/task_2026-10-22_step5_d29_native_balanced/progress.md`：父端记录已批准position政策及当前scope阶段（可直接执行；child无写权限）。
3. 父端联合plan：加入本design9的repair扩展与4.6必要计费companion，明确首次交付是否仅counts（需用户确认具体扩展范围）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 联合native+Frontier范围确认 → 完成标志：明确文件/行数上限、companion所有者与验收范围。
2. 并行：native tensor/context实现 ‖ Frontier source/count/profile接口实现 → 完成标志：聚焦producer/invariance/coverage测试通过。
3. 父端collective/测量阶段 → 完成标志：真实rows、kernel/count/physical identity与独立G3标准一致；无缺测量fallback。

## 4. 最值得关注的 3 点

1. 计数由Batch冻结的context/query区间产生；不能用scalarN/equal-ratio或TP复制替代。
2. zero-local-load、globally all-padding与positive physical width不同；现有测量数据与计费均不能凭0.0准入。
3. 29文件/1556gross行是本侧count+profile预算；通信/source-local计费companion与native范围尚须父端整合。
