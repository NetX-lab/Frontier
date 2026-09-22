# Issue 26 Correctness PR — Design Records

One section per work package. Each records the source-backed reasoning, the
scope decisions and the pre-measurement expectation for that package.

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created. Source-backed design for Checkpoint D's W3 half, with the scope decisions and their evidence. |
| 2026-09-22 | Restructured into per-work-package sections and added the W4 design, with the report-key identity measured at the emission boundary. |

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
