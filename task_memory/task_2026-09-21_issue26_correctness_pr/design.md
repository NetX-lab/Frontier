# W3 design — one shared monolithic forward

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created. Source-backed design for Checkpoint D's W3 half, with the scope decisions and their evidence. |

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
