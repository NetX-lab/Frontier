# Issue 26 Correctness PR — Design Records

One section per work package. Each records the source-backed reasoning, the
scope decisions and the pre-measurement expectation for that package.

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | Added "Design checkpoint D9-2: the key from the fourth shape": reference lockstep facts, seven-shape P1(b) scores, the proposed group-anchored key rule with its invariant argument and residuals, W9-03 pointer. Proposal only; awaits the user's decision. |
| 2026-09-22 | Added the design checkpoint section: D9-1 payload settled from the P1 oracle; D9-2 key left open because the `ForwardSyncState` candidate fails invariant I5 at PP>1 and the I1/I5 trade-off is only observable on the shape blocked by W9-01. |
| 2026-09-22 | W9 second review (user-directed quality gates): section "What the code already provides" added; planned-edits rows for the hook payload, the call site and the CPU oracle amended; plan §18.12 R9-01..R9-08. |
| 2026-09-22 | Created. Source-backed design for Checkpoint D's W3 half, with the scope decisions and their evidence. |
| 2026-09-22 | Restructured into per-work-package sections and added the W4 design, with the report-key identity measured at the emission boundary. |
| 2026-09-22 | Added the W9 design: vLLM 0.10.2 count publication under pipeline parallelism, the equivalence argument, the schedule-only gap, the report-key options, and the discriminating scenario. Analysis only; implementation pending. |
| 2026-09-22 | W9 corrected per the external review (P9-01..P9-06): the steady-state claim withdrawn in favor of stated preconditions and the engine-iteration state table; K1, K3-as-written and stride keys rejected as acceptance basis, with the reproduced counterexample and six invariants; the discriminating scenario qualified as a conditional witness; planned edits extended (observation record, instrumentation chain, PP3 fixture, CPU reference loop, controls). Still analysis only. |

# W3 — one shared monolithic forward

## The defect, restated from source

In `MONOLITHIC`, a batch takes the prefill or the decode per-layer sync path by
one rule, at `frontier/events/replica_stage_schedule_event.py:157-167`:

```python
is_monolithic_prefill_moe = ... and batch.num_prefill_tokens > 0
is_monolithic_decode_moe  = ... and batch.num_prefill_tokens <= 0
                                and batch.num_decode_tokens > 0
```

`initialize_sync_waiting_rooms` (`frontier/scheduler/utils/sync_state.py:29-34`)
then allocates **two disjoint waiting-room trees** for that one cluster, and
`ForwardSyncState` (`forward_sync_state.py:38-41`) keeps **two open-step
namespaces**, `"prefill"` and `"decode"`.

So when lane 0 holds a batch with prefill tokens and lane 1 holds a pure-decode
batch, lane 0 waits in the prefill room and lane 1 in the decode room. Each room
requires `expected_lanes == _replica_dp_size` members. Neither reaches it. The
idle-lane filler cannot rescue either room: `_can_supply_idle_lane`
(`sync_entry.py:8-14`) refuses a sibling whose stage `is_busy`, and the sibling
is busy holding the other phase. Both entries return `[]` and the forward never
dispatches.

This needs `attn_dp >= 2`. With one lane a cohort has one batch and therefore one
phase, which is why the defect has never been observed on the shipped recipes.

## Reachability, and what that means for acceptance

The public MoE wrappers enforce `ATTN_TP == MOE_TP * MOE_EP` while the runtime
enforces `attn_tp * attn_dp == moe_tp * moe_ep`; these have no common solution
above one lane. **No case in the 71-case fidelity matrix can reach a multi-lane
monolithic MoE forward at all.** That is why R35-02 requires a
direct-construction integration fixture rather than another wrapper case, and it
is also why most of this change is expected to be fidelity-neutral.

The expectation is therefore stated in two parts, before measuring:

- The shared-room, shared-identity, one-restoration and per-source-continuation
  changes cannot move any matrix case, because every matrix case runs one lane,
  where the cohort is a single batch and `sample_batch` *is* that batch.
- The decode-layer advance for decode requests carried inside a prefill batch
  (I7) is reachable under chunked prefill at one lane, so it may move MoE cases.
  It is the one part of this change with a real blast radius; see below.

## Scope decisions

| Invariant | In scope | Why |
| --- | --- | --- |
| I1 one shared step-id namespace | yes | The defect. |
| I2 one waiting room per monolithic forward | yes | The defect. |
| I3 a request may not occupy two non-idle lanes | yes | Group-formation guard the review asks to validate at the owning boundary. |
| I4 one ownership restoration per cohort | yes | "one shared completion, one ownership restoration". |
| I5 source-local continuation | yes | "Do not choose a single sample batch and apply its predicted duration to all source lanes." |
| I7 decode-layer advance inside a mixed source | yes, measured | The prefill entry path never advances a decode request's layer counter. |
| I11 per-source dense-layer label | yes | A decode source inside a prefill-mode group is mislabelled today. |
| I6, I9, I10 | already hold | I6 follows from one room; I9/I10 verified unchanged in the audit. |
| **I8 per-source decode component ledger** | **no** | Checkpoint D says "current-main metrics ownership". Introducing the donor's `_decode_model_execution_components_ms_by_stage` would replace main's decode final-timing computation and move every reachable MoE decode case. Excluded deliberately, not overlooked. |

## Event-type determinism: the constraint that shapes the design

Events order by `(time, event_type, id)` (`frontier/events/base_event.py:66-72`),
and `EventType` values are the priority (`PREFILL_SYNC_COLLECTIVE = 10`,
`DECODE_SYNC_COLLECTIVE = 24`). Two consequences:

1. **A new `EventType` value must not be given to a cohort shape that already
   works.** A pure-prefill monolithic MoE cohort emits
   `PrefillSyncCollectiveEvent` today and is exercised by the matrix. Moving it
   to a new event type would reorder it against `REPLICA_STAGE_SCHEDULE = 11` at
   equal timestamps and change results.
2. **The class must not depend on which lane completed the cohort.** Choosing it
   from the triggering entry would make a mixed cohort's priority depend on
   arrival order, which is nondeterminism introduced by the fix.

Resolution: the collective event class for a monolithic cohort is chosen from
the **cohort's contents**, by the same rule the entry event uses —
`PrefillSyncCollectiveEvent` if any non-idle source has `num_prefill_tokens > 0`,
`DecodeSyncCollectiveEvent` otherwise. A pure-prefill cohort and a pure-decode
cohort therefore keep exactly the event type they have today; only the mixed
cohort, which currently deadlocks, is new. No `EventType` value is added.

## Planned edits

| File | Change |
| --- | --- |
| `forward_sync_state.py` | Add the `"forward"` kind to the open-step namespaces and `_validate_kind`. |
| `sync_state.py` | For `MONOLITHIC` + MoE, allocate one room and bind `_forward_sync_waiting_room` plus both legacy names to it. `PREFILL`/`DECODE` keep their own single room and get `_forward_sync_waiting_room = None`. |
| `sync_entry.py` | Collapse the two ~140-line near-duplicate entries into one `enter_layer_sync(..., mode)` with thin `enter_prefill_sync` / `enter_decode_sync` wrappers, mirroring the existing `schedule_layer_wave(mode=...)`. Use `sync_kind="forward"` and the shared room when the cluster is monolithic. |
| `ep_wave_inputs.py` | Reject a request appearing in two non-idle source lanes. |
| `ep_wave_schedule.py` | Per-source mode for the dense-layer event label and the prefill-only component ledger; choose the post_moe collective class from cohort contents. |
| `forward_collective.py` (new) | Pop the shared room, advance the decode-phase subset once, restore full-stage owners once, then run each non-idle source through its own existing helper with `direct_batch`. |
| `prefill_collective.py`, `decode_collective.py` | Accept `owners_restored` and `layer_advance_done` so the shared actions are not repeated; predict each source's continuation from that source's own batch instead of `sample_batch`. |

Routing a mixed cohort through per-source `direct_batch` calls is what makes I5
fall out: with `direct_batch`, the helper's `sample_batch` *is* the source batch,
so each lane already predicts its own continuation. The remaining `sample_batch`
uses inside the multi-lane loops are corrected in place.

## What the tests must distinguish

Per the plan and R35-02, a helper-level stub is not acceptable as the only
evidence. At least one test builds a runtime configuration directly, runs the
real event loop, real admission, ownership and completion code, and injects
deterministic durations only at the predictor boundary. It must fail both on the
old mixed-phase deadlock and on a deliberately wrong source-timing
implementation.

## Fidelity expectation, stated before measuring

`completed_layer_count` is read on the monolithic path only by admission guards
and diagnostics. Its two arithmetic consumers,
`ClusterBatchEndEvent._get_current_layer_id_from_batch` call sites
(`cluster_batch_end_event.py:174` and `:354`), are both PD-AF
`DECODE_ATTN`/M2N paths. No monolithic predicted duration is derived from it.

The prediction for the 71-case matrix is therefore:

- **All 71 cases stay exactly equal.** At one attention-DP lane every cohort
  holds one source batch, so the shared room, the shared identity, the single
  restoration and the per-source continuation all reduce to today's
  single-source behavior; `sample_batch` *is* the only batch.
- The one behavior change reachable at one lane is the I7 layer credit for an
  already-decoding request carried inside a prefill-mode batch. It changes a
  counter that no monolithic timing reads.
- **If a MoE co-location case does move, the I7 credit is the cause**, and it is
  the approved fidelity fix that plan section 9 requires ("advance the decode
  subset exactly once per completed layer").

Recorded before the measurement so the result can falsify it.

---

# W4 — opt-in vLLM-style DP request placement

## What this adds, and what it is not

A new **opt-in** cluster-scheduler policy, `vllm_load_balancing`, that routes
requests inside one serving Replica the way vLLM V1's frontend does: pick the
engine with the lowest `4 * waiting + running` from a **delayed** snapshot of
engine counts, reserving local waiting load between snapshots. Nothing selects
it by default; every existing policy and default is untouched.

This models vLLM's *source-level* selection and count-publication behavior. It
does **not** claim placement or timing equivalence with a real vLLM deployment,
and it does not model IPC transport latency, multiple frontends, elastic
scaling, or warm-start publication phase.

## Reference facts, verified in the pinned checkout

All from `.real-engine/vLLM-BS` at `ea95f571`, read rather than assumed:

| Fact | Location |
| --- | --- |
| `score = waiting * 4 + running`, scanned from `eng_start_index`, strict `<` so the first minimum wins | `vllm/v1/engine/core_client.py:1139-1150` |
| Local reservation after selection: `current_counts[eng_index][0] += self.client_count` — with one frontend, `waiting + 1` | `core_client.py:1151-1153` |
| A published snapshot **replaces** the frontend estimate wholesale; it is not merged into local reservations | `core_client.py:1073-1078` (`self.lb_engines = sliced_counts`) |
| Publish interval: `stats_update_interval_ms` when changed, else `5000` ms | `coordinator.py:195-198` |
| `min_stats_update_interval_ms` default `100` | `coordinator.py:116`, `:122`, `:130` |
| Minimum collection wait `50` ms while a previous-step snapshot is pending | `coordinator.py:201-203` |
| Poll timeout `max(min_timeout, wait_for - elapsed)` | `coordinator.py:205-206` |
| On timeout: publish the pending previous-step snapshot if present, else the current counts and clear `stats_changed` | `coordinator.py:207-218` |
| The order key is the pair `(wave, step)` against one shared `(last_stats_wave, last_stats_step)` | `coordinator.py:156-157`, `:293-300` |
| A strictly newer key latches the prior counts **only when** `stats_changed`; an **equal** key takes neither branch; an out-of-order key logs a **warning** and the counts are still applied | `coordinator.py:293-310` |

So the constants are confirmed as 4, 50 ms, 100 ms and 5000 ms, and the
out-of-order case is a warning, never an abort. Per decision D1, W4 must not add
a runtime assertion on report order that the reference does not have.

## The report key, measured at the emission boundary

Decision D1 asks for exactly this: inspect the identity **at the load-report
emission boundary**, reuse it only where its equality and ordering hold, and
reject the rest explicitly rather than adding a second counter.

The boundary is `GlobalBatchEndEvent`, whose `_replica_local_id` and batch reach
the new `on_replica_batch_end` hook. A probe patched that handler and recorded
`(replica_local_id, ForwardSyncState.get_step_id(batch))` for every completion
of a real `Simulator` run, in four monolithic shapes:

| Shape | Observed `(lane, step)` sequence | Verdict |
| --- | --- | --- |
| MoE `attn_dp=2`, all arrivals at once | `(1,3) (0,3) (1,7) (0,7) (1,11) (0,11) (1,15) (0,15) (1,19) (0,19)` | **valid** — both lanes of one forward share one key; distinct forwards are strictly increasing |
| MoE `attn_dp=2`, staggered online arrivals | `(0,3) (0,7) (0,11) (1,15) (1,19) (1,23) (0,27) (0,31) (0,35) (1,39) (1,43) (1,47) (0,51) (0,55) (0,59)` | **valid** — strictly increasing across the whole run whichever lane reports |
| MoE `attn_dp=1` | `3 7 11 15 19 23 27` | valid, degenerate |
| Dense `attn_dp=1` | `0 1 2 3 4 5 6` | valid, degenerate |
| Dense `attn_dp=2`, all arrivals at once | `(1,0) (0,0) (1,1) (0,1) (1,2) (0,2) (1,3) (0,3) (1,4) (0,4)` | **incidentally** paired; nothing enforces it |
| Dense `attn_dp=2`, staggered online arrivals | `(0,0) (0,1) (0,2) (1,0) (1,1) (1,2) (0,3) (0,4) (0,5) (1,3) (1,4) (1,5) (0,6) (0,7) (0,8)` | **INVALID** — lane 1's step `0` arrives after lane 0's step `2` |

Two things this settles that the audit could only argue:

1. **MoE monolithic is valid because W3 landed.** The ids come from the
   Replica-scoped `_next_step_id_by_replica` counter that W3's single shared
   room resolves, so they are monotonic per Replica regardless of which lanes
   are live. Both lanes of one forward report the same key, which is precisely
   the reference's "equal key, peer engine" path.
2. **Dense with more than one lane is invalid, and observably so.** Each dense
   lane emits its own `0, 1, 2, …` from a per-lane creation counter that no
   shared forward ever promotes, because a dense monolithic Replica has no
   per-forward DP collective to keep the lanes in lockstep. Under staggered
   arrivals the counters interleave out of order against one shared
   `last_report_step`, so the previous-step snapshot latch would be wrong.

Also observed: `replica_local_id` at this boundary is **always an exact `int`**,
never the full-stage `None`, in all four shapes; and an idle batch never reaches
the hook. So the engine index needs a type/range check, not a `None` branch.

Numeric caveat, unchanged from the audit and kept in a code comment: the value
advances per layer, so consecutive forwards are about `num_layers` apart. Only
its ordering and equality are used. The reference key is a `(wave, step)` pair;
Frontier has no wave reset, so a Replica-scoped monotonic counter collapses the
pair to a single scalar.

## Scope decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Topology guard | `MONOLITHIC` + one Replica + `PP1` + `vllm_v1` + (**MoE or `attn_dp == 1`**) | The first four are the candidate's. The fifth is decision D1's "reject unsupported configurations explicitly", and the dense multi-lane row above is the measurement behind it. |
| Second step counter | **no** | D1 forbids broadening W4 with a new counter. W3's identity is reused where it holds. |
| Runtime order assertion | **no** | The reference warns and applies the counts. W4 mirrors the warning. Key equality per shared forward is a test invariant, not a runtime abort. |
| `waiting + 1` reservation | keep | One modeled frontend, so `client_count == 1`. The one-frontend restriction is stated in the class docstring. |
| Heartbeat events | **none** | Timers advance lazily inside `select`/`report`. The policy creates no events, so it cannot keep a drained simulation alive — asserted in a test rather than assumed. |
| `get_request_load` placement | next to the existing `_get_num_waiting_reqs_for_decision_log` in `vllm_v1_iteration_policy.py`, delegating to it | The donor deleted that helper and broke `sglang_style_replica_scheduler.py`; this branch's split already moved it into the shared policy mixin and pinned it with a boundary test. Delegating gives one definition of "waiting" for the balancer and both decision-log emitters, without churning that boundary. |
| Config module extraction | **already done** | The merged module split created `frontier/config/cluster_scheduler_config.py`. W4 only adds one dataclass to it, which is a registry entry through an unchanged mechanism. |

## Planned edits

| File | Change |
| --- | --- |
| `frontier/types/cluster_scheduler_type.py` | `VLLM_LOAD_BALANCING = 5`. |
| `frontier/config/cluster_scheduler_config.py` | `VllmLoadBalancingClusterSchedulerConfig`, no fields. |
| `frontier/scheduler/cluster_scheduler/cluster_scheduler_registry.py` | One registry entry. |
| `frontier/scheduler/request_load.py` (new) | `RequestLoad(NamedTuple)` with `waiting`, `running`. |
| `frontier/scheduler/utils/vllm_dp_load_balancer.py` (new) | The pure state machine: `report`, `select`, lazy timer advance, reference-mirrored warning. |
| `frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py` (new) | Topology guard, `schedule_at`, `schedule`, `on_replica_batch_end`. |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | Two inert default seams: `schedule_at(time)` delegating to `schedule()`, `on_replica_batch_end(...)` returning `None`. |
| `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | `get_request_load()` raising `NotImplementedError` with the concrete class name. |
| `frontier/scheduler/replica_scheduler/vllm_v1_iteration_policy.py` | `get_request_load()` delegating to the existing waiting helper; both decision-log payloads consume it. |
| `frontier/events/cluster_schedule_event.py` | `schedule()` → `schedule_at(self.time)`. |
| `frontier/events/global_batch_end_event.py` | Call `on_replica_batch_end` after the request-state transition in `on_batch_end`. |

## Fidelity expectation, stated before measuring

The policy is opt-in and no existing configuration selects it. The two seams are
inert for the five existing policies: `schedule_at` delegates to the same
`schedule()` the event called before, and `on_replica_batch_end` returns `None`.
The decision-log payloads keep identical values because `get_request_load`
delegates to the same helper they already called.

**Prediction: all 71 fidelity cases stay exactly equal.** Anything that moves is
a defect in the seams, not an approved behavior change — unlike W2 and W3, W4
has no reachable fidelity fix, so a single mismatch falsifies the change rather
than confirming it.

# W9 — schedule-time reports under pipeline parallelism

Analysis recorded before implementation (plan §18; corrected 2026-09-22 per the external review, see plan §18.11). Everything here is read from
the pinned checkout `.real-engine/vLLM-BS` at `ea95f571e` and from the current
branch; nothing has been measured on a GPU yet.

## Reference facts, verified in the pinned checkout

| Fact | Source |
| --- | --- |
| `max_concurrent_batches = pipeline_parallel_size`; a value above one builds `batch_queue` and selects `step_with_batch_queue`. | `vllm/v1/executor/multiproc_executor.py:325-329`, `vllm/v1/engine/core.py:147-157` |
| One iteration schedules first (`waiting → running` for admitted requests; in-flight requests are skipped because `num_new_tokens == 0`), appends, and returns without completing anything when the queue still has room and the oldest batch is not done. Otherwise it pops the oldest, waits, and applies `update_from_output`. | `core.py:318-370`, `vllm/v1/core/sched/scheduler.py:436-441` |
| `_maybe_publish_request_counts()` runs after every iteration and publishes `(running, waiting)` when it changed, with `step_counter` and `current_wave`. `step_counter` increments afterwards, so the publication of iteration `k` carries `k-1`. | `core.py:1075-1087, 1089-1137` |
| Coordinator: a strictly greater `(wave, step)` latches the previous counts when `stats_changed`; an equal key applies without latching; a smaller key warns. Publications every `min_stats_update_interval_ms=100` while changed, 5000 ms otherwise, 50 ms first-collection wait. | `vllm/v1/engine/coordinator.py:116, 196-227, 280-312` |
| Frontend: `score = waiting * 4 + running`, first minimum from `eng_start_index`, `+client_count` waiting reservation until the next publication replaces `lb_engines`. Only `vllm serve` builds `DPLBAsyncMPClient`. | `vllm/v1/engine/core_client.py:85-103, 1131-1156` |
| `get_request_counts()` is `(len(running), len(waiting))`; preempted requests are back in `waiting`. | `scheduler.py:1502-1504` |
| No DP+PP prohibition; per-engine `world_size = PP*TP`. | `vllm/config/parallel.py:314`, `vllm/engine/arg_utils.py:1221-1269` |

## Frontier facts on the current branch

| Fact | Source |
| --- | --- |
| `on_schedule` admits while `_num_running_batches < _num_stages`; the counter falls in `on_batch_end`, called just before the cluster scheduler's completion report. | `base_replica_scheduler.py:1052-1057`, `global_batch_end_event.py:180-185` |
| `_running_requests` grows at admission; `get_request_load()` returns `(queue + preempted, len(_running_requests))`. | `vllm_v1_engine_replica_scheduler.py:948`, `vllm_v1_iteration_policy.py:528-545` |
| At admission a batch carries the provisional per-lane creation counter; the Replica-scoped key is assigned when the sync room opens during execution. | `base_replica_scheduler.py:460-467`, `forward_sync_state.py:152-158` |
| The only report boundary today is `on_replica_batch_end`, keyed by `ForwardSyncState.get_step_id(batch)`. | `vllm_load_balancing_cluster_scheduler.py`; W4 above |

## Where the steady state matches, and on what conditions

(Corrected 2026-09-22, P9-01. The earlier text here said "no change is needed
for the steady state"; that sentence is withdrawn.)

The reference decides each engine iteration by a conjunction, not by queue room:

```python
model_executed = scheduler_output.total_num_scheduled_tokens > 0
if (model_executed
        and len(batch_queue) < batch_queue_size
        and not batch_queue[-1][0].done()):
    return None, True          # admission-only iteration
# otherwise: pop the oldest output, wait, update_from_output, then publish
```

With queue depth `P`, appending `B_k` to a queue holding `P-1` earlier outputs
completes `B_(k-P+1)`; the `B_(k-1)` algebra of the first draft is the `P=2`
case. Frontier's completion report at `B_(k-P+1)`'s end shows the same combined
state only when `B_k` was admitted at `B_(k-P)`'s end, no empty iteration
intervened and no completion became visible between the two boundaries. Equal
queue occupancy does not prove equal `waiting`/`running` populations; request
membership, empty schedules, completions and their visibility times must
correspond. Those preconditions are a test table (plan §18.11), not an
assumption, and the PP=1 equivalence remains the degenerate case in which the
conjunction is never true because `step()` is atomic.

## The gap: iterations the completion report cannot represent

| State after a scheduling attempt | Reference | W9 must represent |
| --- | --- | --- |
| Nonzero tokens, room remains, oldest not ready | Return, publish changed counts | Admission-only observation |
| Nonzero tokens, room remains, oldest already ready | Apply oldest, publish | One combined observation, not an extra admission-only report |
| Nonzero tokens, queue full | Wait/apply oldest, publish | Completion-path observation |
| Zero-token schedule, work queued | No early return | Explicit mapping of the empty iteration |
| No new work, output queued | Drain, publish | Completion-only observation |

The admission-only row is the visible one: after idle (or an empty-batch
iteration) vLLM publishes `waiting -n, running +n`, score `-3n`, with a key
strictly greater than the last completion's, while Frontier stays silent until
that batch ends and, when a single `on_schedule` call admits two batches, never
exposes the state after the first. But `pipeline_room_remaining` tests only the
second conjunct, so a hook keyed on it has no representation for the other
rows; the hook (name fixed by D-d) must pass an observation classified by this
table, captured where the state changes, with the publish decision kept in the
existing load-report owner. At PP=1 none of these iterations occur, which is
what makes a byte-identical PP=1 fidelity check the right acceptance test for
the unchanged path.

## The report key at the admission boundary

The W4 key is valid because both lanes of a forward share the Replica-scoped
id and distinct forwards are strictly increasing. That id does not exist yet at
admission — only the provisional per-lane counter does, and the dense
multi-lane row in W4 shows what per-lane counters do to the latch. Options:

| Option | Rule | Verdict (2026-09-22, P9-02) |
| --- | --- | --- |
| K1 | Schedule-time report reuses `last_report_step` (equal key). | Rejected as acceptance basis: applies counts but never latches, so it misses the reference latch of the pre-admission state when the previous completion is still unpublished. |
| K3 as written | A completion mints a label on first sight of its cohort; every schedule-only admission mints a fresh label. | Rejected as written: a fresh label per admission *callback* equates callback order with iteration order and gives two peer lanes of one logical iteration different keys. Counterexample below. If "fresh label" was meant per shared logical iteration, the rule must first say how that iteration is identified. |
| Stride keys (`2*cohort±1`) | Arithmetic room between completion keys. | Rejected: arbitrary factor; no room for consecutive admission-only iterations at PP≥3. |

**Counterexample, reproduced on this branch's `VllmDPLoadBalancer`** (zero
initial counts; lane 0 reports `waiting=0, running=3` at 10 ms, lane 1 reports
the same at 20 ms; a request is placed at 80 ms):

| Key assignment | Frontend snapshot before the 80 ms placement | Last publication | Selected lane |
| --- | --- | --- | --- |
| One key for both reports | `[(0,3), (0,3)]` | 70 ms | 0 (first minimum) |
| Fresh key for the second report | `[(0,3), (0,0)]` — the partial snapshot latched at 20 ms | 20 ms | 1 |

Identical inputs and loads, different published state. A second interleaving
must also be handled: one lane completes cohort `C`, proceeds to an
admission-only observation, and the peer's completion of `C` arrives later;
reusing `C`'s old label after minting the next one breaks strict emission
order, and the native coordinator's out-of-order warning is evidence to
analyze, not something to suppress by inventing newer identities.

The rule is chosen at the design checkpoint (plan §18.5) after the P1 probes
establish logical-iteration membership: first identify the reference-equivalent
engine iteration, then reuse an existing scheduler iteration/forward identity if
it represents it, else a derived identity or a small report-state field. It
must satisfy: (1) peer observations of one logical iteration compare equal in
any callback order; (2) a new iteration orders after the previous one, with the
key captured at the observation boundary; (3) an iteration that schedules and
completes owns one report decision; (4) suppressed unchanged reports create no
fictitious messages; (5) PP3 allows consecutive admission-only iterations
without spacing constants; (6) bookkeeping is released with the in-flight work.

## What the code already provides (second review, 2026-09-22)

Read against `c231322` with the user's gates for core-module changes (plan
§18.12). Three facts narrow the design.

**A completion is atomic in the DES.** `GlobalBatchEndEvent` runs
`replica_scheduler.on_batch_end` and then `cluster_scheduler.on_replica_batch_end`
at the batch's end time, and the lane's next `ReplicaScheduleEvent` follows at
the same time. There is no "oldest output ready but not yet applied" state, so
the hook has nothing to classify: it carries `(time, replica_id,
replica_local_id, batch)` exactly as the completion hook does, and the policy
scheduler reads the post-admission load through `get_request_load()`, which
already reflects the admission because `_running_requests` grows inside
`_get_next_batch`. The reference's other conjunct, room in the batch queue, is
real and is the lane's `num_running_batches < num_pipeline_stages` after the
admission; the policy reads both from existing getters and publishes the
admission on its own only while room remains, otherwise the admission is
folded into the completion the engine then blocks on. At PP=1 the single slot
is always filled, so this one rule reproduces today's completion-only
reporting without a PP special case. The zero-token iteration publishes nothing in the reference
(`_maybe_publish_request_counts` emits only changed counts) and needs no
observation here. Of the five reference rows only the admission-only row is new.

**The completion key names the wrong iteration at PP>1.** `on_replica_batch_end`
keys by `ForwardSyncState.get_step_id(batch)`, assigned when the batch's own
forward opened. The reference publishes a completion under the iteration that
applied the output, which at depth `P` is `P-1` iterations later. At PP=1 the
two are the same iteration; at PP>1 the batch key would order every completion
before the admissions emitted while it ran. The key rule therefore applies to
both observation kinds and is read at the observation boundary.

**A Replica-scoped, monotonic, lane-equal value already exists.**
`ForwardSyncState._next_step_id_by_replica[replica_id]` is the id the next
forward on the Replica will take: equal for all lanes between room openings,
strictly greater than every open or completed step, advanced only by
`resolve_step`/`close_step`. As the key for both kinds it meets invariants 1–4
and 6 with no new bookkeeping. Its gap is invariant 5: several admissions on
one lane while stage 0 is busy read one value, where the reference gives
strictly increasing steps and latches the intermediate state for one publish
interval. P1 measures whether that case occurs in the target scenario (one
admission per lane) and whether the reference actually keeps peer steps equal
there — `_has_global_unfinished_reqs` all-reduces every 32 steps, so it may
not. The decision stays at the design checkpoint; if a derived identity is
needed, the reason is this measurement, not a preference.

**Call site and layering.** `self._cluster_scheduler` is constructor-required
(`TypeError` when `None`), so the call in `on_schedule` is unconditional; the
two existing `getattr(self, "_cluster_scheduler", None)` reach-ups in
`_create_batch` are the pattern not to repeat. The completion hook is invoked
by an event and the admission hook by the replica scheduler; the asymmetry is
deliberate, because the per-admission state after the first of two admissions
in one call is visible only inside the loop.

**The CPU oracle models the engine loop only.** `VllmDPLoadBalancer` already
reproduces the coordinator latch/publish and the frontend score with cited
constants; `reference_loop.py` scripts the `step_with_batch_queue` conjunction,
the changed-count emission and the per-iteration step counter, and feeds the
real balancer. Comparing a Frontier-driven balancer against it isolates the
one mapping W9 changes.

## The discriminating scenario, derived

Let engine `e` receive `k_e` requests in a burst and admit `a_e` of them in
its first batch. With the frontend reservation only, its score is `4k_e`;
after vLLM's schedule-only publication it is `4k_e - 3a_e`. The two views
order the engines differently when `k_0 > k_1` but
`4k_0 - 3a_0 < 4k_1 - 3a_1`, i.e. `3(a_0 - a_1) > 4(k_0 - k_1) ≥ 4`, so
`a_0 - a_1 ≥ 2`. The smallest instance: a burst of five whose reservation
alternation gives `e0: r1, r3, r5` and `e1: r2, r4`; `r2` has a long prompt that
fills `e1`'s chunk budget so `r4` waits (`a_1 = 1`); `r1, r3, r5` are short
(`a_0 = 3`). Reference publication after the schedule-only iteration:
`e0 = 3`, `e1 = 5`, so a probe request `r6` arriving after the first
coordinator publication (≥50 ms) and before `e1`'s first chunk completes goes to
`e0`. Current Frontier never reports before the first completion, the
reservations persist (`e0 = 12`, `e1 = 8`) and `r6` goes to `e1`. The fixed
module reports the admission and sends `r6` to `e0`. The scenario is robust to
timing as long as `e1`'s first chunk exceeds the window on both sides (vLLM:
chunk budget and prompt length; Frontier: `dummy_execution_time_ms`, decision
D-e). Three warmups and an idle gap of at least 5 s precede the burst so both
systems start from the quiet state the argument assumes.

**Conditional witness (2026-09-22, P9-04).** The algebra holds for a state with
`k_e` assigned, `a_e` admitted, no intervening completion and no further
published state; `k=(3,2), a=(3,1)` is the target, not a guaranteed live
outcome. The trace must show the actual routing order of the burst, the
admissions, the in-flight requests, later scheduling attempts and the snapshot
applied at the frontend before `r6` was routed; otherwise the slice is
`SCENARIO_NOT_REACHED`, not a scheduler mismatch. HTTP concurrency does not fix
engine-receipt order, so request ids, dispatch order and receipt order are
recorded and qualified. Chunk sizes and decode lengths are frozen explicit
values; the balancer's constants are never tuned to make the witness occur. The
pre-change control for `r6` is the explicit test-only completion-reporting
baseline (guard lifted only), because the unmodified constructor rejects PP2
and can only report that rejection.

## Planned edits

| File | Edit |
| --- | --- |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | `on_replica_batch_scheduled(time, replica_id, replica_local_id, batch)`, the completion hook's signature, inert default. No readiness field or observation record (second review, R9-01). |
| `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | Call the hook once per admitted batch in the MONOLITHIC/PREFILL admission loop, after `_num_running_batches += 1`, through the constructor-required `self._cluster_scheduler` without `getattr`/`hasattr` (R9-04). |
| `frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py` | Drop the PP1 clause of the guard and its error text; implement `on_replica_batch_scheduled`, publishing an admission on its own only while `num_running_batches < num_pipeline_stages` (existing getters); key both observation kinds by the observing iteration per the design checkpoint (first candidate: the Replica's next forward id through a plain `ForwardSyncState` accessor), satisfying invariants 1–6. |
| `tests/comparison/dp_placement_pp/reference_loop.py` | CPU oracle of the engine iteration only (scripted admissions, empty schedules, completions, controllable readiness; conjunction, changed-count emission, step counter), feeding the real `VllmDPLoadBalancer`; not a second coordinator/frontend model (R9-05). |
| `tests/unit/test_vllm_dp_load_balancer.py`, `tests/integration/test_vllm_dp_placement_runtime.py` | Guard case inverted; the plan §18.11 behavioral matrix (PP2 both callback orders, oldest-ready, PP3 consecutive admission-only iterations on a 6- or 12-layer fixture, full queue, empty schedule, drain, idle peers, bounded bookkeeping); the discriminating scenario against the test-only control; the C35-01 hybrid-layer credit case at PP2. |
| `.real-engine/vLLM-BS` (local branch only, D-b) | Case-gated event chain: iteration result, emitted report, coordinator receive/publish with snapshot id, frontend application, frontend routing; named `waiting`/`running`; correlation ids; buffered per-process JSONL. |
| `AGENTS.md:620`, this file, `plan.md`, `progress.md`, `validation.md`, `review.md` | Wording and records. |

## Fidelity expectation, stated before measuring

- Every PP=1 `vllm_load_balancing` scenario: `request_metrics.csv` value-identical and `system_metrics.json` identical after removing timestamps and run ids.
- Every scenario of every other cluster scheduler: identical (the hook's default is inert; the only added work is one method call per admission).
- PP=2 and PP=3 with `vllm_load_balancing`: runs complete with request/token/owner conservation; on the qualified discriminating scenario the corrected module sends `r6` to `e0` while the test-only completion-reporting control sends it to `e1` (round-robin inequality alone proves nothing, since PP1 `vllm_load_balancing` already differs from round-robin).
- Ground truth: on a controlled or causally matched history, emitted loads, key equality/order, coordinator snapshots and frontend-visible counts agree with the reference; natural divergence is labeled by first cause; T2 matches the corrected module in a trace-qualified slice or is recorded as `SCENARIO_NOT_REACHED`.

## What this adds, and what it is not

It adds vLLM's schedule-time publication to the existing report path and
removes a capability boundary that the candidate had inherited. It does not add
an event type, a flag, or a constant; it does not model multiple frontends,
hybrid or external load balancing, wave resets, or elastic EP; and it does not
claim latency equivalence of placements — the comparison is by boundary index,
with timing controlled only where the discriminating scenario needs it.


## Design checkpoint: what P1 settled and what it did not (2026-09-22)

### D9-1 payload — settled

The oracle confirms there is nothing to classify at the boundary. In the
reference an iteration's branch depends on whether the oldest queued output is
already done; in the DES a completion is atomic at its end event, so the
admission hook needs the same four values the completion hook already takes,
`(time, replica_id, replica_local_id, batch)`, and the policy reads the
post-admission population through `get_request_load()`. The pipeline-room test
stays, computed by the policy from `num_running_batches` and
`num_pipeline_stages` (R9-01); it is what makes PP=1 report nothing extra.

### D9-2 key — not settled, and the reason is structural

The first candidate was the Replica-scoped forward id that `ForwardSyncState`
hands out, read at the boundary. Measured on the three shapes that run:

| Invariant | `attn_dp=2, PP=1` | `attn_dp=1, PP=2` | `attn_dp=1, PP=3` |
| --- | --- | --- | --- |
| I1 peers of one iteration compare equal | holds (both lanes read the same value at both boundaries) | not observable (one lane) | not observable |
| I2 new iteration orders after the previous | holds (0, 6, 12, 18, 24) | holds after the fill | holds after the fill |
| I3 one decision per iteration | holds (completion and the admission it triggers share the value) | holds | holds |
| I5 consecutive admission-only iterations stay distinct | vacuous (none occur) | **fails**: both cold-fill admissions read 0 | **fails**: all three read 0 |
| I6 bookkeeping released | holds (no new state) | holds | holds |

The failure is not incidental. The forward id advances when a forward room
opens, which happens once per executed forward, while the reference key advances
once per engine iteration. At PP>1 a lane admits up to `num_pipeline_stages`
batches before the first forward completes, so every one of those admissions
reads the same id. That is precisely the cold fill the discriminating scenario
in plan §18.6 turns on, so the candidate cannot be accepted.

The obvious repair, a per-lane observation counter, restores I5 by construction
and breaks I1 by construction: two lanes advance independently, so peer
observations of one logical iteration no longer compare equal, which is the
counterexample recorded above. Any rule that satisfies both has to derive a
shared identity that still advances per observation, and whether a given rule
does can only be decided by observing a shape with `attn_dp > 1` **and**
`PP > 1`. That shape deadlocks today (W9-01 in `issues.md`).

The checkpoint therefore closes with D9-1 fixed and D9-2 open. Implementing a
key rule now would mean choosing between two invariants with no way to test the
choice, which is the kind of unfalsifiable design the gates exclude.

## Design checkpoint D9-2: the key from the fourth shape (2026-09-23)

W9-01 is fixed on `main` (PR 36) and merged forward, so the shape that decides
D9-2 now runs. This section records what the probe shows and the proposed key
rule. The rule is a proposal: it needs the user's decision before P2.

### The reference key is the shared forward index

R9-08 said DP engines are not iteration-lockstep, because the unfinished-work
all-reduce runs only every 32 steps. The pinned source shows lockstep by
another route:

| Fact | Source |
| --- | --- |
| An iteration that schedules no tokens enqueues the empty output, then blocks on the oldest queued output. | `core.py:364-420` (`step_with_batch_queue`) |
| After such an iteration the busy loop runs `execute_dummy_batch`, a blocking collective RPC to every worker, each of which runs `_dummy_run(1)`. | `core.py:1170-1195`; `multiproc_executor.py:197-199`; `gpu_worker.py:556-557` |
| An empty scheduler output runs no forward. | `gpu_model_runner.py:2250-2254` |
| Every real or dummy forward on every stage joins its peers' DP all-reduce in `get_dp_padding` (CUDA graphs enabled). MoE layers add EP collectives. | `gpu_model_runner.py:1904-1925`; `forward_context.py:72-85` |

So each iteration of a running engine launches exactly one forward, real or
dummy, and on every stage the k-th forward of one engine pairs with the k-th
forward of each peer. The `step_counter` a report carries is therefore the
index of that shared forward. Peers reporting the same forward compare equal.
The next iteration is strictly greater. The 32-step all-reduce only decides
when a wave ends.

Frontier's shared stage-0 forward group is the analog of that index.
`StageExecutionContext.bind_forward_group` gives every MoE lane batch that
starts one stage-0 forward the same id, and it hands out ids in forward order.
Idle lanes take part in the same forward's layer rooms.

### What each report describes

The D9-1 rule sorts every report into one of three kinds:

| Kind | When | Forward it describes |
| --- | --- | --- |
| admission | admitted with `running_after < stages`, published on its own | the admitted batch's stage-0 group |
| completion_folded | completion whose lane holds an admission made with a full pipeline | the folded batch's group |
| completion_only | completion with no folded admission | the lane's next Frontier forward (the reference runs a dummy forward here) |

The probe (`step9_p1b/probe_boundaries.py`) records these boundaries with the
stage-0 state at each one, and records every batch's stage-0 group and each
room's group. `step9_p1b/analyze_keys.py` derives each report's group after
the fact and scores the candidate keys against it in two ways:

- pairwise order: peer splits (the same group gets different keys, I1); merges
  (different groups get one key; on one lane this is I5); inversions (the
  order is reversed);
- replay: the reports drive a fresh `VllmDPLoadBalancer`, and the score is the
  milliseconds in which the frontend-visible counts differ from the replay
  keyed by the groups.

Shapes: 6-layer synthetic MoE model, analytical CC backend, round-robin
placement, dummy predictor, prefill 16 and decode 3 tokens, 6 requests. The
burst shapes are offline at 1e6 QPS; the staggered shapes are online Poisson at
20 QPS. Every shape completes 6/6.

| Shape | Reports (adm / fold / only) | A: `ForwardSyncState` next id | Lane report counter | Group-anchored (proposed) |
| --- | --- | --- | --- | --- |
| MoE dp2 PP1 burst | 0 / 12 / 0 | 0 / 0 / 0, 0 ms | 0 / 0 / 0, 0 ms | 0 / 0 / 0, 0 ms |
| MoE dp1 PP2 burst | 1 / 13 / 1 | 0 / 0 / 0, 0 ms | 0 / 0 / 0, 0 ms | 0 / 0 / 0, 0 ms |
| MoE dp1 PP3 burst | 7 / 9 / 7 | 0 / 1 / 0, 400 ms | 5 / 0 / 0, 0 ms | 0 / 0 / 0, 0 ms |
| MoE dp2 PP2 burst | 4 / 12 / 4 | 0 / 0 / 0, 0 ms | 4 / 0 / 0, 0 ms | 0 / 0 / 0, 0 ms |
| MoE dp2 PP2 staggered | 5 / 12 / 5 | 0 / 0 / 0, 0 ms | 6 / 0 / 0, 0 ms | 0 / 0 / 0, 0 ms |
| MoE dp2 PP3 burst | 4 / 14 / 4 | 0 / 4 / 0, 148 ms | 0 / 0 / 0, 0 ms | 0 / 0 / 0, 0 ms |
| MoE dp2 PP3 staggered | 7 / 11 / 7 | 4 / 1 / 0, 200 ms | 15 / 7 / 5, 0 ms | 0 / 0 / 0, 0 ms |

Cells are splits / merges / inversions, then the replay mismatch. The scored
reports are those whose forward exists in the run; 12 final drain completions
have no later forward and are left out.

- **A** merges the PP3 cold fill. Several admissions on one lane read one
  value, because the id advances when a room opens, not per iteration. The
  same happens in the dp2 PP3 burst, where the two lanes' first and second
  admissions all read 0. The replay differs for 148–400 ms.
- **The lane report counter** is exact whenever every lane runs the same
  sequence of report kinds. With staggered arrivals the lanes run different
  numbers of completion-only iterations and the counters drift apart. In dp2
  PP3 staggered, 6 splits, 5 merges and 5 inversions fall on reports that do
  carry a real forward. Nothing brings the counters back together: in the
  reference the dummy forwards keep lanes aligned, and Frontier has no such
  forwards. A variant of the proposed rule that also advances on every
  completion-only report drifts in the same way, with the same 6/5/5.
- **Group-anchored** matches every scored report in all seven shapes, and on
  every report it equals `predicted_group`. That reference predictor needs
  the count of the lane's admitted batches that have not started stage 0,
  which a policy cannot read.

### Proposed rule (D9-2)

For lane `l` of the Replica, with `C` the Replica's stage-0 context:

```
key(l) = max(C.joinable_forward_group_id, last_admitted_key[l] + 1)
```

`joinable_forward_group_id` is the bound group's id while it is not sealed.
Otherwise it is the next id. A batch admitted now joins exactly that group,
unless its lane already has a batch in it, in which case the `+ 1` term takes
over.

- At an admission: `last_admitted_key[l] = key(l)`. The admission is
  reported now while `num_running_batches < num_pipeline_stages`. Otherwise
  the key is held in `held_key[l]`.
- At a completion: report `held_key.pop(l)` if a key is held. Otherwise report
  `key(l)` and store nothing.

Against the invariants:

| Invariant | How the rule meets it | Evidence |
| --- | --- | --- |
| I1 | Peers of one forward read one Replica-scoped group id. | 0 splits in 7 shapes. |
| I2 | Group ids only grow, and `+ 1` orders a lane's later forwards. | 0 inversions. |
| I3 | A folded admission reports once, with the completion it joins. | D9-1 folding. |
| I4 | Unchanged counts are still dropped inside `VllmDPLoadBalancer.report`. | Unchanged code. |
| I5 | Each consecutive cold-fill admission on a lane gets `+ 1`. No spacing constant. | dp1 and dp2 PP3 cold fills. |
| I6 | Two integers per lane, overwritten in place. | Bounded by `attn_dp`. |

At PP=1 every admission is folded, and the held key is the group of the
forward that completes. Peers compare equal and later forwards compare
greater, so the comparisons match the current `ForwardSyncState.get_step_id`
key. C2's byte-identical PP=1 check tests exactly that. The policy's
`ForwardSyncState` import and use are then superseded and removed.

New surface, compared with the planned edits above: one read-only property on
`StageExecutionContext` over two existing fields, in place of the planned
`ForwardSyncState` accessor, and two per-lane dicts in the policy scheduler.
The policy reaches the context through the base class's
`_stage_execution_contexts[(replica_id, 0)]`. For a dense model the group is
never bound (`bind_forward_group` is MoE-only), so the key falls back to the
lane's admission counter. The policy admits a dense model only at
`attn_dp=1`, where placement has one choice.

### What the rule does not reproduce

1. **The reference's dummy iterations.** A completion-only report shares its
   key with the lane's next report whenever Frontier runs no forward between
   them. The reference gives them keys `k` and `k+1`. There are 17 such pairs
   in the seven shapes, 3 of them with changed counts (two in the dp2 PP3
   burst drain, one in dp1 PP3). The effect: the coordinator does not latch
   between the two reports, so the frontend can see the later counts one
   publication interval (100 ms) early. No Frontier forward exists to anchor
   a separate key. Inventing one would bring back the drift measured above.
2. **The reference's forward pairing when lanes diverge.** In the reference,
   an engine with nothing new to schedule blocks and then runs a dummy
   forward, and its peers' next forward waits for it. Frontier lanes do not
   block. In the dp2 PP3 staggered probe, lane 1's first batch joins the group
   that lane 0 opened 1.3 ms earlier. At 154.3 ms lane 1's batch `b26` joins
   lane 0's open group 2; there, the reference would pair engine 1's dummy
   forward with that forward and `b26` with the next one. This is a
   forward-model difference, not a key defect: the key follows Frontier's own
   grouping. It is recorded as W9-03 in `issues.md` for a separate decision.
   Only source reading supports it; G4's `engine_iteration` records would
   measure it.

The analytical backend runs MoE `attn_dp=2, PP=3`. W9-02's rejection is the
collective-sim topology rule. The P3/P4 matrix can therefore restore the
multi-lane PP3 row that I5 needs (C1 amendment, part of this proposal).
