# Stage admission ordering under pipeline parallelism — Design

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created: defect restated from source on `origin/main` `1f694f7`, what the FIFO guarantees today, four options, recommended rule with its invariants, fidelity expectation. For review before implementation. |

All line references are to `origin/main` at `1f694f7`, checked out in
`/data/ycfeng/Frontier/.worktrees/stage-admission-ordering`.

## The defect, restated from source

One `StageExecutionContext` owns each physical `(replica, stage)`. All
attention-DP lanes of that stage share it. Each lane has its own
`ReplicaStageScheduler` with its own batch heap and its own `_is_busy` flag, so
**a lane consumes at most one ticket at a time**.

Admission is a two-step handshake:

1. `ReplicaStageScheduler.add_batch` (`replica_stage_schduler.py:145-162`) mints
   a `StageAdmissionTicket` at batch **arrival** through
   `StageExecutionContext.enqueue_full_stage`, which appends it to the shared
   `_ready_fifo` (`stage_execution_context.py:188`).
2. `pop_batch_if_not_busy` (`:286-359`) takes the lane's own heap head and asks
   the context to `try_acquire` its ticket. `try_acquire` (`:322-343`) refuses
   when an EP wave is active, when active full-stage owners already fill
   `full_stage_capacity`, when the forward group is sealed, and finally when the
   ticket **is not the FIFO head**:

   ```python
   if not self._ready_fifo or self._ready_fifo[0] != ticket:
       return False
   ```

`full_stage_capacity` is `attn_dp` for `MONOLITHIC`, `PREFILL` and `DECODE`
(`stage_contexts.py`), so up to one ticket per lane may be active at once: that
is how the lanes of one forward co-own the stage.

`BaseReplicaScheduler.on_schedule` admits up to `num_pipeline_stages` batches
per lane in one round (`base_replica_scheduler.py:893-901`). At
`num_pipeline_stages > 1` a lane therefore holds **several queued tickets** while
being able to consume only one. The FIFO head can then be a ticket whose own
lane is busy, and every other lane is refused although capacity is free.

### Observed state at the drain (MoE `attn_dp=2, moe_ep=2, PP=2`, 4 requests)

Both lanes admitted two batches each. Lane 1 scheduled first.

| Where | State |
| --- | --- |
| Stage `(0,0)` context | `capacity=2 sealed=False group=0`; active `{seq0: batch 0 (lane 1)}`; FIFO `[seq1: batch 1 (lane 1), seq2: batch 2 (lane 0), seq3: batch 3 (lane 0)]` |
| Lane 1 stage 0 | `busy=True`, heap `[(batch 1, global_id 3, seq1)]` |
| Lane 0 stage 0 | `busy=False`, heap `[(batch 2, global_id 0, seq2), (batch 3, global_id 2, seq3)]` |
| Prefill sync room step 0, layer 0, `pre_moe` | `lanes_present=[1]`, waiting for lane 0 |
| Event queue | empty |

The wait is circular:

- Lane 1 holds `seq0`, has bound forward group 0 and sits in the sync room
  until lane 0 joins.
- Lane 0 presents `seq2`; the FIFO head is `seq1`, which belongs to lane 1, so
  `try_acquire` refuses.
- Lane 1 cannot consume `seq1` because it is busy.
- The room does not stand in an idle batch for lane 0: `_can_supply_idle_lane`
  (`sync_entry.py:8-14`) returns `False` when the lane has queued work and the
  group is unsealed, precisely because such a lane is expected to join.

The last point is the crispest statement of the defect: **the sync room's
"this lane can still join" predicate and the context's `try_acquire` disagree
about the same lane.** The room is right about the model; the context's FIFO
position test is the part that has no counterpart in the system being
simulated.

Two orderings also disagree with each other. The lane heap orders by
`global_id = counter * lane_count + lane_id` (`batch_ids.py:19`), which puts
lane 0's batch 2 (`global_id 0`) ahead of lane 1's batch 0 (`global_id 1`),
while the ticket FIFO orders by arrival, which puts lane 1 first. Within one
lane the two agree; across lanes they need not.

### Why dense completes and MoE does not, and why PP=1 is untouched

- Dense never calls `bind_forward_group` (the call at
  `replica_stage_schduler.py:347-356` is MoE-only), so it never seals and has no
  sync room. A refused lane is simply re-woken at the next release
  (`batch_stage_end_event.py`, `stage_wakeup.py`), so dense finishes, but the
  lanes run **one after the other**: while lane 1 works through both of its
  batches, lane 0's slot stays empty. Measured: dense `attn_dp∈{2,4}, PP=2`
  completes 6/6 and 8/8; the overlap loss is the softer form of the same defect.
- At `num_pipeline_stages = 1` a lane never holds more than one ticket, tickets
  are minted just before the lane attempts to use them, and every refusal
  condition other than FIFO position is lane-independent. Measured: MoE
  `attn_dp∈{2,4}, PP=1` completes 6/6, 12/12; dense likewise.

| Shape (origin/main, fresh process each) | Requests | Result |
| --- | --- | --- |
| MoE `attn_dp=2, moe_ep=2, PP=2` | 3 | completes |
| MoE `attn_dp=2, moe_ep=2, PP=2` | 4, 6 | **drained** |
| MoE `attn_dp=4, moe_ep=4, PP=2` | 8 | **drained** |
| MoE `attn_dp=2, moe_ep=2, PP=1` | 6, 12 | completes |
| MoE `attn_dp=4, moe_ep=4, PP=1` | 8, 12 | completes |
| MoE `attn_dp=1, PP=2` / `PP=3` | 6 | completes |
| Dense `attn_dp=2, PP=2`, `attn_dp=4, PP=2` | 6, 8 | completes (lanes serialized) |
| Dense `attn_dp=2, PP=1`, `attn_dp=4, PP=1` | 6, 8 | completes |

The threshold at 4 requests is where both lanes first hold more than one batch.

## What the FIFO position test guarantees today

Read from the unit tests that pin it:

| Test | Guarantee |
| --- | --- |
| `test_admission_fifo_cannot_skip_an_earlier_ready_wave` | Two queued EP waves are admitted in queue order. |
| `test_ep_wave_owns_stage_before_dense_can_start` | A full-stage ticket queued after an EP wave waits for it. |
| `test_started_group_blocks_new_lane_through_ep_restore_and_partial_release` | A lane arriving after a group started waits until every owner releases (seal), not FIFO. |
| `test_next_group_queue_does_not_block_current_group_idle_participation` | A lane whose queued work is refused by the **seal** is stood in as idle. |
| Comment at `replica_stage_schduler.py:301-303` | The lane must admit the same heap head it inspected (a bypass fix unrelated to cross-lane order). |

None of them require that two full-stage tickets from **different lanes** be
admitted in arrival order. That ordering is the one piece with no stated
purpose, and it is the one that fails.

## Options

| Option | Rule | Verdict |
| --- | --- | --- |
| A. Skip busy owners | Ticket carries its lane; an earlier queued full-stage ticket blocks admission only while its lane holds no active ticket. Provably a no-op wherever a lane holds at most one ticket. | **Rejected.** It leaves a lane waiting for a *peer's acquisition*, and the DES wakes lanes only at *release* (`build_stage_wakeup_events` runs from `BatchStageEndEvent`). Traced on the drain scenario: after the first cohort releases, lane 0 presents `seq3`, lane 1's `seq1` is queued ahead with no active owner, lane 0 is refused, lane 1 then acquires and enters the room, and nothing retries lane 0. It would need a second wake path on acquisition, which is new machinery for a state the model does not have. |
| B. Order only exclusive operations | A full-stage ticket is refused only when an **EP wave** is queued ahead of it; earlier full-stage tickets never block it. EP waves keep the strict head rule. Capacity, seal and EP-active checks unchanged. | **Recommended.** No new field, no interface change, one predicate. Every remaining refusal (capacity, seal, EP active, EP wave ahead) is cleared by a release, which already wakes siblings, so no new wait state exists. The room predicate and the context now agree. |
| C. Mint the ticket at the admission attempt instead of arrival | A busy lane never holds a queued ticket. | Rejected. Changes ordering semantics for every path and breaks the stale-drop logic, which relies on the ticket attached at arrival (`_discard_stale_ticket`, `_drop_queued_lanes_for_ticket`, sibling tickets in `DECODE_FFN`). |
| D. Stand in an idle lane when a lane is blocked by admission order | Change `_can_supply_idle_lane`. | Rejected. Lane 0 has real work for this forward; modelling it as absent skips that work into a later forward. An error-suppressing fallback in the sense of the working gates. |

## Recommended rule

In `StageExecutionContext.try_acquire`, replace the head test for full-stage
tickets with:

> A full-stage ticket may be admitted when no EP wave is queued ahead of it.
> An EP wave may be admitted only as the FIFO head.

Sketch (final wording at implementation; the existing scope, capacity and seal
checks above it are unchanged):

```python
if ticket.scope == EP_WAVE:
    if not self._ready_fifo or self._ready_fifo[0] != ticket:
        return False
elif any(queued.scope == EP_WAVE for queued in self._ready_fifo
         if queued.admission_seq < ticket.admission_seq):
    return False
self._ready_fifo.remove(ticket)
```

The FIFO stays one deque so that an EP wave still sees every full-stage ticket
ahead of it. The scan is bounded by `lanes × num_pipeline_stages` tickets.

Files touched: `stage_execution_context.py` (rule and the two docstrings that
describe admission as "FIFO-head"), no other source file. `_can_supply_idle_lane`
is left as is; it becomes consistent rather than changed.

### Invariants after the change

1. At most one active full-stage ticket per lane per stage (unchanged; from
   `_is_busy`).
2. Within one lane, batches enter a stage in heap order (unchanged; the lane
   presents only its heap head).
3. Exclusive operations (EP waves) are admitted in queue order and never
   overtaken by full-stage work queued behind them (unchanged; pinned by the
   two EP tests).
4. A lane with queued work, free capacity and an unsealed group can be admitted
   at its next attempt (new; this is the property the sync room already
   assumes).
5. Every refusal is cleared by a release event, which wakes idle non-empty
   sibling lanes (unchanged mechanism, now sufficient).

### Where the rule is a no-op by construction

- `full_stage_capacity = 1` contexts (`DECODE_ATTN`, `DECODE_FFN`): with one
  owner slot, a second full-stage ticket is refused by capacity whenever the
  first is active; when nothing is active, letting a later full-stage ticket
  pass an earlier one is the only new behaviour, and it can arise only if the
  later ticket's lane attempts first while both are queued. In `DECODE_FFN`
  the shared groups carry one ticket for all sibling lanes, so two distinct
  full-stage tickets queued at once means two successive groups, which are
  produced and attempted in order. Verified by byte comparison in the plan, not
  assumed.
- Dense `attn_dp > 1, PP = 1` and MoE `attn_dp > 1, PP = 1`: one ticket per lane,
  minted immediately before the attempt; the only refusals are lane-independent.
  Verified by byte comparison.

## Fidelity expectation, stated before measuring

| Scenario class | Expected after the change |
| --- | --- |
| Every `num_pipeline_stages = 1` scenario (co-location, PDD, PD-AF examples; Step 8 regression set) | Byte-identical `request_metrics.csv` and `system_metrics.json`. |
| PD-AF `DECODE_ATTN` / `DECODE_FFN` (capacity 1) | Byte-identical. |
| MoE `attn_dp > 1`, `PP > 1` | From drain to completion with request and token conservation. |
| Dense `attn_dp > 1`, `PP > 1` | Completes before and after; lane stage-busy intervals overlap after the change where they were serialized before, so makespan and per-request latencies **change**. This is the same defect's softer symptom and is proposed as an accepted fidelity fix (decision D-2 in the plan). |
| `attn_dp = 1` any PP | Byte-identical (one lane, FIFO order equals heap order). |

Any difference outside the two "changes" rows is a defect in this change and
stops the work.

## What this is not

- Not a change to `full_stage_capacity`, the seal, the EP wave protocol, the
  sync rooms, or the wake-up helper.
- Not a new flag or configuration field.
- Not the Step 9 report key (D9-2 in the parent plan); that design resumes once
  this lands and the `attn_dp=2, PP=2` shape runs.
