# Stage admission ordering under pipeline parallelism — Design

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | Applied the round-1 plan review (`review.md`). Changes: the admission-loop anchor now points to the `MONOLITHIC`/`PREFILL` path; the drain condition is stated as a queued-ticket arrangement, not a shape; the shape table is marked author-reported until P0; added where queued EP waves exist; `remove(ticket)` made explicit; option A's stall trace labelled an unverified hypothesis; the capacity-1 section rewritten as a caller-level condition; the queue bound narrowed; added the mixed-phase scope boundary; the dense "lanes serialized" label withdrawn as unmeasured and replaced by the admission sequence read from source. |
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
2. `pop_batch_if_not_busy` (`:286-359`) returns at once when its lane is busy;
   otherwise it takes the lane's own heap head and, unless the context already
   `owns` that ticket, asks `try_acquire` (`:329`). `try_acquire` (`:322-343`)
   refuses when an EP wave is active, when active full-stage owners already
   fill `full_stage_capacity`, when the forward group is sealed, and finally
   when the ticket **is not the FIFO head**:

   ```python
   if not self._ready_fifo or self._ready_fifo[0] != ticket:
       return False
   self._ready_fifo.popleft()
   ```

`full_stage_capacity` is `attn_dp` for `MONOLITHIC`, `PREFILL` and `DECODE`
and 1 otherwise (`stage_contexts.py:59-81`), so up to one ticket per lane may
be active at once: that is how the lanes of one forward co-own the stage.

`BaseReplicaScheduler.on_schedule` admits batches while
`num_running_batches < num_stages`. The co-location reproduction runs the
`MONOLITHIC`/`PREFILL` branch (`base_replica_scheduler.py:1037-1054`, loop at
`:1039`); the unified `DECODE` branch (`:893`) has the same bound. At
`num_pipeline_stages > 1` a lane can therefore hold **several queued tickets**
at one stage while consuming one.

The drain needs a specific arrangement, not merely `attn_dp > 1` and `PP > 1`:

- a lane's queued ticket is at the FIFO head while that lane is busy with an
  active ticket of an open (unsealed) forward group, and
- another lane with queued work presents a ticket behind it and is refused,
  while the busy lane waits in a sync room for that refused lane.

With too little queued work the same shape completes (the 3-request row
below).

### Observed state at the drain (MoE `attn_dp=2, moe_ep=2, PP=2`, 4 requests)

Author-run on 2026-09-22 with the session scripts. P0 republishes it from the
published case inputs (`plan.md` §4, group R0) and stores the state report.

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

### Why dense completes and MoE does not, and why PP=1 is not expected to drain

- Dense never calls `bind_forward_group` (the call at
  `replica_stage_schduler.py:347-356` is MoE-only), so it never seals and has no
  sync room. A refused lane is re-woken at the next release
  (`batch_stage_end_event.py:139-158`, `stage_wakeup.py:8-43`), so dense
  finishes, but with lost overlap. From source, with every request at `t=0`:
  the lane that schedules first mints two tickets before the other lane mints
  any; the other lane is then refused while the first lane's first batch holds
  stage 0, although capacity is free, and is admitted only at that release.
  The first draft called this "lanes serialized". That was not measured: the
  earlier runs checked completion only. P0 measures the loss with the `plan.md` §4.5
  ledger metric. The loss is the softer form of the same defect.
- At `num_pipeline_stages = 1` a lane admits its next batch only after the
  previous one leaves the only stage, so it never holds a queued ticket while
  busy, and the first bullet of the drain arrangement cannot form.

Author-reported shapes (2026-09-22; default `astra_sim_analytical` backend,
Poisson `qps=1e6`, prefill 16 / decode 3; logs
`/data/ycfeng/tmp/w10_repro/case_*.log`). They remain author-reported
evidence until P0 reruns them as group R0 from the published inputs.

| Shape (origin/main, fresh process each) | Requests | Result |
| --- | --- | --- |
| MoE `attn_dp=2, moe_ep=2, PP=2` | 3 | completes |
| MoE `attn_dp=2, moe_ep=2, PP=2` | 4, 6 | **drained** |
| MoE `attn_dp=4, moe_ep=4, PP=2` | 8 | **drained** |
| MoE `attn_dp=2, moe_ep=2, PP=1` | 6, 12 | completes |
| MoE `attn_dp=4, moe_ep=4, PP=1` | 8, 12 | completes |
| MoE `attn_dp=1, PP=2` / `PP=3` | 6 | completes |
| Dense `attn_dp=2, PP=2`, `attn_dp=4, PP=2` | 6, 8 | completes (overlap not measured) |
| Dense `attn_dp=2, PP=1`, `attn_dp=4, PP=1`, `attn_dp=1, PP=2` | 6, 8, 6 | completes |
| MoE `attn_dp=2, moe_ep=2, PP=3` | 6 | rejected at construction by the Replica-pod node-size rule (6 devices against node size 4; parent task W9-02) |

In this table the drain first appears at 4 requests, where both lanes first
hold more than one batch.

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
purpose, and it is the one that fails. This is a reading of the tests, not a
test result; C4 in the plan settles it.

### Where queued EP waves exist

`enqueue_ep_wave` has one caller, the `DECODE_FFN` M2N group path
(`round_robin_cluster_scheduler.py:1052-1057`). On `MONOLITHIC`, `PREFILL` and
`DECODE` contexts, `EP_WAVE` appears only as an active-scope transition of
owners already admitted (`transition_active_scope`,
`replace_full_stage_owners_with_ep_wave`,
`replace_ep_wave_with_full_stage_owners`, driven by
`forward_step_admission.py`); it never enters the FIFO there. So the FIFO of a
shared-lane context holds only full-stage tickets, and a mixed FIFO of
full-stage tickets and EP waves exists only on `DECODE_FFN` contexts, whose
capacity is 1.

## Options

| Option | Rule | Verdict |
| --- | --- | --- |
| A. Skip busy owners | The ticket carries its lane; an earlier queued full-stage ticket blocks admission only while its lane holds no active ticket. | **Rejected on design grounds.** It adds lane identity to tickets and makes one lane's admission depend on a peer lane's *acquisition*. Acquisition emits no retry; only `BatchStageEndEvent` wakes siblings (`batch_stage_end_event.py:148-158`), so A would need a new wake path on acquisition. The first draft also sketched a specific second-cohort stall. That trace is an **unverified hypothesis**. It did not account for the releasing lane's own retry, which is emitted before sibling retries (`:139-146`). It did not account for retries from several same-time releases, and the reviewer notes that prefill participants can release at one shared predicted time. And the DES orders equal-time events by `(time, id, event_type)` (`base_event.py:63-64`, `simulator.py:1268`), not by `BaseEvent.__lt__` (`:66-70`), which compares type before id. B does not depend on that trace, so it is not pursued. |
| B. Order only exclusive operations | A full-stage ticket is refused only when an **EP wave** is queued ahead of it; earlier full-stage tickets never block it. EP waves keep the strict head rule. Capacity, seal and EP-active checks unchanged. | **Recommended (adopted as D-1).** No new field, no interface change, one predicate. Every remaining refusal (capacity, seal, EP active, EP wave ahead) is cleared by a release, which already wakes siblings, so no new wait state exists. The room predicate and the context now agree. |
| C. Mint the ticket at the admission attempt instead of arrival | A busy lane never holds a queued ticket. | Rejected. Changes ordering semantics for every path and breaks the stale-drop logic, which relies on the ticket attached at arrival (`_discard_stale_ticket`, `_drop_queued_lanes_for_ticket`, sibling tickets in `DECODE_FFN`). |
| D. Stand in an idle lane when a lane is blocked by admission order | Change `_can_supply_idle_lane`. | Rejected. Lane 0 has real work for this forward; modelling it as absent skips that work into a later forward. An error-suppressing fallback in the sense of the working gates. |

## Recommended rule

In `StageExecutionContext.try_acquire`, replace the head test for full-stage
tickets with:

> A full-stage ticket may be admitted when no EP wave is queued ahead of it.
> An EP wave may be admitted only as the FIFO head.

Sketch (final wording at implementation; the existing scope, capacity and seal
checks above it are unchanged, and the EP-wave line is today's line):

```python
if ticket.scope == EP_WAVE:
    if not self._ready_fifo or self._ready_fifo[0] != ticket:
        return False
else:
    for queued in self._ready_fifo:
        if queued == ticket:
            break
        if queued.scope == EP_WAVE:
            return False
self._ready_fifo.remove(ticket)
```

- The admitted ticket is removed with `remove(ticket)`, not `popleft()`: once a
  non-head ticket can be admitted, `popleft()` would dequeue a different
  ticket. `cancel` already removes a queued ticket the same way
  (`stage_execution_context.py:456`).
- The single caller reaches `try_acquire` only with a queued ticket: it checks
  `owns` first (`replica_stage_schduler.py:328`), and `_validate_ticket`
  rejects a ticket that is neither queued nor active. No branch is added for
  other states.
- The FIFO stays one deque so that an EP wave still sees every ticket ahead of
  it.
- Queue length: on the shared-lane contexts this change targets, the FIFO holds
  at most `attn_dp × num_pipeline_stages` tickets, because each lane runs at
  most `num_pipeline_stages` batches (`base_replica_scheduler.py:893,1039`).
  `DECODE_FFN` contexts are fed by M2N groups and have no such bound; the scan
  there is linear in the queue, as `cancel`'s `remove` already is. No index or
  second queue is added.
- On shared-lane contexts no EP wave is ever queued (previous section), so the
  loop only walks to the ticket; its EP clause acts on `DECODE_FFN`.

Files touched: `stage_execution_context.py` (rule and the two docstrings that
describe admission as "FIFO-head"), no other source file. `_can_supply_idle_lane`
is left as is; it becomes consistent rather than changed.

### Invariants after the change

1. At most one active full-stage ticket per lane per stage (unchanged; from
   `_is_busy`).
2. Within one lane, batches enter a stage in heap order (unchanged; the lane
   presents only its heap head).
3. Exclusive operations (EP waves) are admitted in queue order and never
   overtaken by full-stage work queued behind them; an EP wave still waits for
   every earlier queued ticket and every active owner (unchanged; pinned by the
   two EP tests and extended by P2(a)).
4. A lane with queued work is admitted at its next attempt when capacity is
   free, the group is unsealed, no EP wave is active and no EP wave is queued
   ahead of its ticket (new; this is the property the sync room already
   assumes).
5. Every refusal is cleared by a release event, which wakes idle non-empty
   sibling lanes (unchanged mechanism, now sufficient).

Invariant 4 removes the admission-order refusal. It is not a whole-run
liveness proof; see the scope boundary below.

### Where behaviour is expected to stay unchanged, and why

The rule is not a no-op at the context API. With an idle capacity-1 context
and FIFO `[full0, full1]`, `try_acquire(full1)` is refused today and admitted
under B. Capacity prevents two simultaneous owners but does not preserve
arrival order when nothing is active. Unchanged behaviour is therefore a claim
about the callers, under this condition:

> For a scheduler that presents only its heap head, B and today's rule make the
> same decision whenever no full-stage ticket of **another** scheduler is
> queued ahead of the presented ticket, and each scheduler's heap order agrees
> with FIFO order among its own full-stage tickets. Then only EP waves can be
> ahead of the presented ticket, and both rules refuse exactly when something
> is ahead.

| Context | Why the condition is expected to hold | Evidence planned |
| --- | --- | --- |
| `DECODE_FFN` (capacity 1) | Every `DenseFFNBatchGroup` gets `global_id = _batch_group_creation_counter` (`round_robin_cluster_scheduler.py:1097,1118`) and one full-stage ticket (`:1138`), and is queued on the one full-stage scheduler of its replica (`:1100`), so its heap and FIFO both follow the group counter. EP child batches hold no full-stage ticket; the group shares one `EP_WAVE` ticket (`:1052-1057`). Shared EP sibling tickets are not multiple full-stage owners. | P2(a′) control with two successive dense FFN groups and a neighbouring EP group through the real full-stage scheduler; G6 byte comparison. |
| `DECODE_ATTN` (capacity 1) | `attn_dp=1` with `replica_local_id=None` (AGENTS.md): one scheduler per stage, so no other scheduler's ticket can be ahead. | G6 byte comparison. |
| Shared-lane contexts, `PP = 1` | A lane never holds a queued ticket while busy. A cross-lane inversion needs one release to wake two or more idle siblings whose tickets are queued in the opposite order to the wake order: wake-ups follow lane-key order (`stage_wakeup.py:30-32`), and today's rule refuses the first sibling woken. The releasing lane has no queued ticket at `PP=1` and is excluded, so this needs `attn_dp ≥ 3`. | G1, G3 and G4 `PP=1` byte comparison. `attn_dp=2` is expected unchanged. `attn_dp=4` is expected, not guaranteed, unchanged, and a difference stops the work for diagnosis (plan P3). |
| `attn_dp = 1`, any PP | One lane, so FIFO order equals heap order. | G5 byte comparison. |

No capacity-1 or `PP=1` special case is added: no supported caller has been
shown to need arbitrary cross-lane full-stage FIFO order. An unexpected
difference in any of these classes stops the work and is reported.

## Scope boundary: mixed-phase forwards

This branch is based on `main`, where prefill and decode source lanes still
enter separate synchronization paths. The shared forward across mixed prefill
and decode lanes is PR 35 W3 (`65ed8a7`), not on `main`. Fixing admission does
not fix that. A shape that deadlocked at admission may, once admitted, reach a
mixed-phase cohort and fail another way. Such a failure is recorded and
diagnosed separately; it is not repaired by widening this one-file change.

Consequences for verification:

- The C1 witnesses are phase-controlled. All requests arrive at `t=0` with
  equal prompt lengths, and the primary group is prefill-only
  (`decode_tokens=1`). A `MONOLITHIC` request of that shape completes at the
  prefill boundary, which grants its one decode token
  (`request.py:1286-1293,1379-1384`), so no decode batch forms.
- Composition with PR 35 is validated in the parent task after this branch is
  merged forward, before Step 9 is declared unblocked.

## Fidelity expectation, stated before measuring

| Scenario class | Acceptance path (plan §4.2) | Expected after the change |
| --- | --- | --- |
| Declared `PP = 1` scenarios (release examples, synthetic `PP=1` cells) | U | Byte-identical metrics files. `attn_dp=4` cells carry the caveat in the table above. |
| PD-AF `DECODE_ATTN` / `DECODE_FFN` (capacity 1) in the declared recipes | U | Byte-identical, for the caller-level reason above. |
| `attn_dp = 1`, any PP | U | Byte-identical. |
| MoE `attn_dp > 1`, `PP > 1`, cells P0 classifies as admission deadlock | L | Completes, with request and token conservation. |
| MoE `attn_dp > 1`, `PP > 1`, cells that complete on base | T | Byte-identical, or a difference explained with the stage ledger. |
| Dense `attn_dp > 1`, `PP > 1` | T | Completes before and after. A lane refused only by FIFO position is admitted at once, so lane overlap increases and makespan and per-request latencies may **change**. This is the same defect's softer symptom and was accepted as a fidelity fix (D-2). |

Any outcome outside its row stops the work.

## What this is not

- Not a change to `full_stage_capacity`, the seal, the EP wave protocol, the
  sync rooms, or the wake-up helper.
- Not a new flag or configuration field.
- Not a fix for mixed-phase forwards on `main` (PR 35 W3).
- Not the Step 9 report key (D9-2 in the parent plan); that design resumes once
  this lands and the `attn_dp=2, PP=2` shape runs.
