# Issues

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | W9-05 review addendum: F-R1..F-R4 at PP>1 fixed in `c647e95`; F-R5 and F-R6 recorded as fidelity proposals. W9-03 transferred to the S42 task. |
| 2026-09-23 | W9-05 diagnosed and fixed in `75c1140` (user direction "授权上述1-2，推进W9-05"): mechanism and Resolution with checks B1–B8 added. |
| 2026-09-23 | W9-04 fixed in `2ffb062` under option 1 (user decision); Resolution added with checks A1–A7. W9-05 deferred as a separate item (user decision). |
| 2026-09-23 | W9-04 (placeholder/join deadlock at `attn_dp=4`, root cause and prototype) and W9-05 (requests lost under KV pressure, also on `main`) recorded from Step 9 P5. |
| 2026-09-23 | W9-01 remaining step 3 done (P1(b) complete, D9-2 proposed); W9-02 narrowed to the collective-sim backend; W9-03 recorded (reference DP lockstep under PP, observation). |
| 2026-09-23 | W9-01: merged forward (`dd9b8d9`); composition check passes on `03d5f24`. |
| 2026-09-23 | W9-01: PR 36 ran its pre-merge untrack (P6, `4d08c5d`); the copies here are now the only published records of that task. |
| 2026-09-23 | W9-01: PR 36 round-2 review remediation recorded; the composition check now also reruns PR 36 groups G9 and G10. |
| 2026-09-23 | W9-01: fixed on `fix/stage-admission-ordering` (draft PR 36) under option 2; resolution recorded, summary and test report copied to `w9_01_stage_admission_ordering/`. |
| 2026-09-22 | Created; recorded W9-01 (stage-admission deadlock at PP>1 with attn_dp>1) and W9-02 (PP=3 x attn_dp=2 topology rejection) found during Step 9 P1(b). |

## W9-01 Stage admission deadlocks when `num_pipeline_stages > 1` and `attn_dp > 1`

Status: fixed on `fix/stage-admission-ordering` (draft PR 36), not yet on this
branch. Step 9's PP>1 packages stay paused until PR 36 merges into `main`, is
merged forward here, and passes the composition check in Resolution below.
Found: 2026-09-22, Step 9 package P1(b) boundary probe.

### Symptom

A MONOLITHIC MoE replica with `attn_dp=2`, `moe_expert_parallel_size=2` and
`num_pipeline_stages=2` drains the event queue with scheduler state still
non-empty. Requests neither complete nor raise; the simulation ends early and
reports fewer completed requests than were generated.

Reproduced in fresh processes, deterministic in the request count:

| Shape | Requests | Result |
| --- | --- | --- |
| `attn_dp=2, moe_ep=2, PP=2` | 2 | completes |
| `attn_dp=2, moe_ep=2, PP=2` | 3 | completes |
| `attn_dp=2, moe_ep=2, PP=2` | 4 | drained, 0 completed |
| `attn_dp=2, moe_ep=2, PP=2` | 6 | drained, 0 completed |
| `attn_dp=2, moe_ep=2, PP=2` | 8 | drained, 0 completed |
| `attn_dp=1, moe_ep=1, PP=2` | 6 | completes |
| `attn_dp=1, moe_ep=1, PP=3` | 6 | completes |
| `attn_dp=2, moe_ep=2, PP=1` | 6 | completes |

The threshold is the point at which both lanes can hold more than one batch in
flight at once, which is what `num_pipeline_stages > 1` permits.

### Mechanism

`ReplicaStageScheduler.add_batch` mints a `StageAdmissionTicket` through
`StageExecutionContext.enqueue_full_stage` at batch *arrival*, and
`try_acquire` admits a ticket only when it is the strict head of the
per-`(replica, stage)` `_ready_fifo`:

```python
if not self._ready_fifo or self._ready_fifo[0] != ticket:
    return False
```

At `num_pipeline_stages = 1` each lane has at most one batch in flight, so
tickets interleave one per lane and the head is always acquirable. At
`num_pipeline_stages > 1` `BaseReplicaScheduler.on_schedule` admits up to
`num_pipeline_stages` batches for one lane in a single scheduling round, so
that lane enqueues `num_pipeline_stages` tickets before its peer enqueues its
first. The peer's ticket then sits behind a ticket whose own lane scheduler is
already `_is_busy`, so that head can never be acquired, and the shared forward
cohort never assembles. Neither lane can progress and no event is left.

Observed event sequence (MoE `attn_dp=2, moe_ep=2, PP=2`, 4 requests):

```
ReplicaStageScheduleEvent(lane 1) -> PrefillSyncEvent
ReplicaStageScheduleEvent(lane 1) -> []      # scheduler busy
ReplicaStageScheduleEvent(lane 0) -> []      # head ticket belongs to busy lane 1
ReplicaStageScheduleEvent(lane 0) -> []
```

Stage `(0, 0)` state at drain:

```
capacity=2  active_seqs=[0]  waiting_fifo_seqs=[1, 2, 3]
sealed=False  ep_active=False
```

`full_stage_capacity` is `replica_dp_size` (2 here), so capacity is not the
constraint; strict FIFO head ordering is.

### Scope

Pre-existing, not introduced by this PR. The three files involved are
byte-identical to `main`:

- `frontier/scheduler/replica_stage_scheduler/stage_execution_context.py`
- `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py`
- `frontier/scheduler/utils/stage_contexts.py`

It is unobserved today because nothing exercises the combination: every
Simulator-level test with `attn_dp > 1` uses `num_pipeline_stages = 1`, and no
shipped example sets `attn_dp > 1` at all.

### Why it blocks Step 9

The vLLM DP placement policy is only meaningful when `attn_dp > 1`; with
`attn_dp = 1` there is a single engine and `select` is degenerate. Step 9's
acceptance criterion C1 requires MoE `attn_dp=2` to complete at PP2, and the
discriminating scenario in plan section 18.6 is
`--data-parallel-size 2 --pipeline-parallel-size 2`. Both require this shape to
run. The design checkpoint is blocked for the same reason: invariant I1 (peers
of one forward share the report key) can only be observed on a shape with both
`attn_dp > 1` and `PP > 1`.

### Options

1. Fix the ordering in the shared stage-admission path as part of Step 9.
   Acquisition would have to consider the first ticket that is actually
   acquirable for its lane rather than the global FIFO head, keeping the
   existing anti-starvation intent. This touches infrastructure shared by
   co-location, PDD and PD-AF, so it needs its own regression matrix.
2. Fix it as a separate correctness item with its own validation, and pause
   Step 9's PP>1 packages until it lands.
3. Restrict Step 9 to `attn_dp = 1` at PP>1. This satisfies nothing: the policy
   has no effect at `attn_dp = 1`, so it would ship a PP>1 claim with no
   evidence for the only configuration the policy affects.

Recommendation: option 2. The defect is independent of the placement policy,
predates both PRs, and changing shared admission ordering under a feature branch
would mix an infrastructure fidelity fix into a feature PR.

### Resolution

Option 2 was taken. The fix lives on its own branch and PR:

| Item | Value |
| --- | --- |
| Branch / PR | `fix/stage-admission-ordering`, draft https://github.com/NetX-lab/Frontier/pull/36 |
| Rule commit | `dac4e69`: `StageExecutionContext.try_acquire` refuses a full-stage ticket only when an EP wave is queued ahead of it. EP waves keep the strict FIFO-head rule. |
| Acceptance rules | `aeeca93` (plan D-9) |
| Round-2 review fixes | `1661bf1` (rule refactor: an active ticket is refused), `a8e8d8a` (PDD, online and PD-AF matrix groups G8–G11), `e35242f` (comparison tools) |
| Records | `fc34341`, `4bcd616`, `ecff89a`, `1218ba6`, `7a7c22e`; copies in `w9_01_stage_admission_ordering/` (`summary.md`, `test_report_2026-09-23_stage_admission_ordering.md`) |
| Pre-merge untrack | `4d08c5d` (PR 36 plan P6): the branch restores `main`'s `.gitignore` and no longer tracks its task directory, so the copies here are the published records |

Observed on that branch (details in the copied test report):

- The 18 base admission deadlocks complete with requests and tokens
  conserved.
- 50/50 unchanged cases are byte-identical.
- G2 shows no regressions.
- The Step 9 probe shape, MoE `attn_dp=2, moe_ep=2, PP=2`, completes 6/6.
- vLLM DP=2/PP=2 on 4×H800 gives 50 MATCH, 0 MISMATCH and 2 INFORMATIONAL
  (dense V5, per D-9); the 4 base negative-control rows hold.
- Round 2 added PDD offline and online, co-location online and PD-AF
  `PREFILL_PP=2` cells: 12 more base deadlocks complete, 0 STOP. On `main`
  the online Poisson cells of MONOLITHIC and PREFILL run on lane 0 only,
  because `main` lacks this branch's W2 lane rotation; the online multi-lane
  coverage there comes from burst cells.

Remaining here, in order:

1. After PR 36 merges into `main`, merge `main` forward into this branch.
   Done: PR 36 squash `4ab1964`, merge `dd9b8d9`.
2. Rerun the PR 36 matrix groups on the merged tree as the composition check:
   G3b (mixed prefill/decode, `attn_dp > 1`, `PP > 1`) with W3, and G9 and
   G10 (online), whose Poisson cells reach every lane only with W2.
   Done: K1–K4 pass on `03d5f24` (`test_report_2026-09-23_w9_01_composition_check.md`),
   after a harness drain-reader fix. With W2 two Poisson PDD PP3 cells drain
   under the pre-merge rule; both complete on the merged tree.
3. If they pass, resume Step 9 P1(b) and the design checkpoint D9-2.
   Done: the fourth shape and three more multi-lane PP shapes complete 6/6;
   the D9-2 proposal is in `design.md` ("Design checkpoint D9-2") and waits
   for the user's decision.

Step 9 C1's PP3 row stays on `attn_dp=1` (W9-02).

## W9-02 `attn_dp=2, moe_ep=2, num_pipeline_stages=3` is rejected at config time

Status: open, expected behavior, affects plan wording only.
Found: 2026-09-22, Step 9 package P1(b).
Narrowed 2026-09-23: the rejection is the collective-sim topology rule. With
the analytical CC backend, MoE `attn_dp=2, PP=3` constructs and completes 6/6
(P1(b) probe, `moe_dp2_pp3_burst` and `moe_dp2_pp3_staggered`).

Construction fails with:

```
collective-sim physical topology requires cluster_total_devices 6 to be
divisible by node size 4
```

The rejection is correct: 2 lanes x 3 stages is 6 devices against a node size of
4. Plan acceptance criterion C1 lists MoE `attn_dp=2` at both PP2 and PP3; the
PP3 row must be `attn_dp=1`, or must choose a device count that divides the node
size. C1 was amended accordingly.

## W9-03 Frontier does not model the reference's DP engine lockstep under PP

Status: open, observation from source reading, not measured. Outside Step 9's
scope. **Transferred 2026-09-24** to
`task_memory/task_2026-09-24_s42_dp_wave_idle_forward/` by the user's decision
that S42 becomes its own calibration-and-repair task; this section is its
source record.
Found: 2026-09-23, Step 9 package P1(b), while deriving the D9-2 key.

### Reference (pinned `.real-engine/vLLM-BS`)

- An iteration that schedules no tokens enqueues the empty output. It then
  blocks on the oldest queued output (`core.py:364-420`).
- The busy loop then runs `execute_dummy_batch` (`core.py:1170-1195`). This
  is a blocking collective RPC: every worker runs `_dummy_run(1)`
  (`multiproc_executor.py:197-199`, `gpu_worker.py:556-557`).
- Every real or dummy forward on every stage joins its peers' DP all-reduce
  in `get_dp_padding` when CUDA graphs are enabled
  (`gpu_model_runner.py:1904-1925`, `forward_context.py:72-85`). MoE layers
  add EP collectives.

The k-th forward of one engine therefore pairs with the k-th forward of every
peer, per stage. An engine that is only waiting for an output delays its
peers' next forward until its own wait and dummy forward finish.

### Frontier (probe `moe_dp2_pp3_staggered`)

- Lanes never block on their oldest output.
- A lane's batch can join a group that a peer's batch has already opened.
  Lane 1's first batch arrives at 52.3 ms and joins group 0, which lane 0
  opened 1.3 ms earlier.
- At 154.3 ms, lane 1's `b26` joins lane 0's open group 2. The reference
  would pair engine 1's dummy forward with that forward, and `b26` with the
  next one.

### Effect

The difference affects the timing of MoE `attn_dp > 1, PP > 1` runs
(co-location and PDD) when lanes have unequal work. Its size is unknown. The
G4 trace would measure it: `engine_iteration` records carry
`(engine, wave, step)`.

It is not a report-key defect. The proposed D9-2 key follows Frontier's own
grouping, and its residual 1 in `design.md` is the part of this difference
that reaches the report stream.


## W9-04 A lane with a first-layer placeholder can join the forward and deadlock it

Status: fixed in `2ffb062` under option 1, by the user's decision of
2026-09-23 (see Resolution below). Not caused by Step 9: the pre-P2 tree
`d1a2a06` and `bacdbb4` stop at the same state.
Found: 2026-09-23, Step 9 package P5, in the C2 PP=1 policy matrix.

### Symptom

MoE co-location, `attn_dp=4, moe_ep=4, PP=1`, `vllm_v1`, 24 Poisson requests
of 8–96 tokens: the event queue drains at 0.1201 s with no request complete
and `RuntimeError: Sequential simulation ended with non-empty scheduler state`.
Lanes 0–2 hold stage 0; lane 3 has a queued batch and is not busy.

Reachability sweep (`step9_p5/deadlock_sweep.py`; MoE `attn_dp` ∈ {2, 4},
Poisson qps ∈ {50, 100, 200, 400}, seeds {42, 7, 123}; 24 cells per row):

| Tree | Cluster scheduler | Stuck cells |
| --- | --- | --- |
| this branch `bacdbb4` | `round_robin` | 5 (all `attn_dp=4`, qps ≥ 200) |
| this branch | `lor` | 1 (`attn_dp=4`, qps 400, seed 42) |
| this branch | `random` | 0 |
| `origin/main` `4ab1964` | `round_robin` | 0 (every request is placed on lane 0: the W2 defect) |
| `origin/main` | `random` | 0 |
| `origin/main` | `lor` | 24 (a different failure: stuck with 2–12 requests done in every cell, `attn_dp=2` included; not diagnosed) |
| prototype below | `round_robin`, `lor`, `random` | 0 of 72 |

`attn_dp=2` never stalled on this branch. `vllm_load_balancing` stalls the
same way as `round_robin` on the same cells.

### Mechanism (trace `step9_p5/evidence/w9_04_trace.txt`)

1. t=12.10 ms: lane 0 reaches layer 0 `pre_moe` of forward group 0. Lane 1 is
   bound to group 0 and still in attention. Lanes 2 and 3 have empty, idle
   stages, so `_can_supply_idle_lane` (`sync_entry.py:9`) gives each an idle
   placeholder in that room.
2. The same instant: request 2 reaches lane 2. The group is not sealed, so
   `try_acquire` admits lane 2's batch and `bind_forward_group` binds it to
   group 0.
3. t=13.71 ms: lane 1 arrives. The room holds four entries (two real, two
   placeholders) and dispatches the layer-0 EP wave. The group is sealed.
4. t=20.38 ms: lane 2's real batch reaches layer 0. That room is closed, so
   `resolve_step` opens a new step for layer 0. Lanes 0 and 1 are busy in
   group 0 and are not given placeholders.
5. t=27.71 ms: lanes 0 and 1 reach layer 1 and wait for lane 2, which waits
   for them at layer 0. Lane 3's new batch cannot join the sealed group.

The join rule and the placeholder rule disagree. A lane may join a forward
until its first EP wave dispatches, and a real batch replaces the lane's
placeholder only if it reaches the room first. A join after the placeholder
but before dispatch, whose attention outlasts the last peer's arrival, loses
that race.

### Options

1. **Drop a stale placeholder (prototype, `step9_p5/w9_04_prototype.patch`,
   8 lines in `enter_layer_sync`).** An idle entry whose lane's stage is busy
   belongs to a lane that has since joined this forward with real work, which
   will enter the room. It is removed before the room is counted, so the
   room waits for the real batch. This is the replacement the code already
   performs when the real batch arrives first, applied to the other order.
   Measured in `trees/proto`: 72 of 72 sweep cells drain. The 22 C2 cases that
   drained before are identical: `request_metrics.csv`, `system_metrics.json`,
   the report stream and every selection. The two stalled cases now finish
   (24/24, and 23/24 with W9-05). Not yet run: the 71-case fidelity matrix,
   the stage-admission groups and the suites.
2. **Keep a lane's placeholder binding (reference behavior).** A lane that has
   been given a placeholder does not join that forward; its batch waits for
   the next one. This is what the pinned vLLM does: an idle engine is already
   inside its dummy forward (W9-03). It changes the timing of every run
   where a late join currently succeeds, so it is a fidelity change that
   needs its own measurement and approval.
3. Record only and defer.

Recommendation: option 1 in this PR, because this PR's W2 is what makes the
multi-lane Poisson path reachable under `round_robin`. Option 2 belongs with
W9-03 and the G4 lockstep measurement.

### Resolution (2026-09-23)

Decision: "本 PR 修复 (Recommended)", which is option 1 (`requirements.md`).

Change (`2ffb062`). `sync_entry.py` gains
`_withdraw_idle_batches_of_joined_lanes`, which `enter_layer_sync` calls after
storing the arriving batch. It removes the room's idle batches whose lane's
stage is now busy. It is the prototype's rule, extracted into a named function
next to `_can_supply_idle_lane`, whose `is_busy` test it reuses. Why such a lane
has joined this forward is argued in `review.md` F9-03 and plan §18.17.

Regression case. `tests/integration/test_vllm_dp_placement_runtime.py` adds
`moe_dp4_late_join`: MoE `attn_dp=4, moe_ep=4`, three online requests at 0, 2
and 8 ms, run under `vllm_load_balancing` and under `round_robin`. The trace
was reduced from the stalled C2 case with `w9_04_fix/extract_trace.py` and
`trace_probe.py`. Before the fix it stalls under both policies with 0 of 3
complete. The test asserts four things:

- the placements are lanes 0, 1 and 2;
- each lane's first stage-0 batch is bound to forward 0;
- the withdrawn placeholder is lane 2's;
- the run conserves work.

Checks, against the criteria fixed in plan §18.17 before measuring:

| Id | Result | Evidence |
| --- | --- | --- |
| A1 | Test passes; the module gives 10 passed. With only the call removed, it fails with `RuntimeError: Sequential simulation ended with non-empty scheduler state`, imported from the control tree. | `w9_04_fix/evidence/negative_control_pytest.txt` |
| A2 | 72 of 72 sweep cells drain, 24 under each of `round_robin`, `lor` and `random`. The branch before the fix stalled in 5 and 1 of those cells. | `w9_04_fix/evidence/sweep_fix_*.txt` |
| A3 | 22 of 22 identical. The two stalled cases now finish, at 24/24 and at 23/24 (`tight_kv`, no error; evidence byte-identical to the prototype's; the loss has the W9-05 signature, not diagnosed). | `w9_04_fix/evidence/c2_bacdbb4_vs_fix.txt` |
| A4 | 71 of 71 identical; 0 provenance findings; `complete_comparison` and `predictor_cache_populated_cleanly` true | `w9_04_fix/evidence/fidelity_comparison.json` |
| A5 | G3b, G9 and G10: 51 of 51 PASS; each cell's `sha256sums.txt` is identical | `w9_04_fix/evidence/stage_admission_compare_w904.json` |
| A6 | unit 84 failed / 3829 passed / 51 skipped / 10 errors; integration 5 errors / 27 passed / 22 skipped. 0 regressions, 0 new failures, 0 skip changes. The only new test id is the regression test. | `w9_04_fix/evidence/{unit,integration}_compare.json` |
| A7 | 16 of 16 examples pass and are identical to `bacdbb4` | `w9_04_fix/evidence/examples_bacdbb4_vs_fix.txt` |

Limits:

- The fix follows Frontier's join rule, under which a late join into an
  unsealed forward succeeds. The reference would pair a dummy forward instead
  (option 2, W9-03). That remains a separate fidelity question, and the fix
  does not settle it.
- All checks are CPU runs with dummy or trained predictors on this host.

## W9-05 Requests disappear mid-decode under KV pressure

Status: fixed in `75c1140` (see Resolution below). Present on `origin/main`
`4ab1964`. Outside Step 9. First deferred by the user's decision of 2026-09-23
("暂缓，单独立项 (Recommended)"), then scheduled the same day ("授权上述1-2，推进W9-05").
Found: 2026-09-23, Step 9 package P5, in the C2 PP=1 policy matrix.

`vllm_v1` with `num_blocks=12, block_size=16`, 24 Poisson requests of 8–96
tokens at qps 200. MoE `attn_dp=2` and dense `attn_dp=1` both reproduce it,
under `vllm_load_balancing` and under `round_robin`, on this branch and on
`origin/main`. The run ends normally with 20 of 24 requests complete and
exit code 0. Requests 10, 16, 17 and 21 decode a few tokens (for example
request 10: 3 of 14), then leave every queue. No log line mentions
preemption, and the drain check passes although `is_empty` also counts the
preempted queue, so the requests are held by no queue it reads. Their
`request_metrics.csv` rows have empty latency fields.

### Mechanism (`w9_05/membership_trace.py`, `w9_05/lost_request_probe.py`)

Preemption was not the missing step; it was the start of the loss.

1. `KvBlockAllocation._preempt_request` (`vllm_v1_kv_allocation.py`) reset
   `_num_processed_tokens` to 0 for every cluster type except DECODE and
   DECODE_ATTN, including a MONOLITHIC victim that had finished its prefill.
2. `is_prefill_complete` stays `True`, so the request keeps its decode phase.
   `_get_request_next_num_tokens` (MONOLITHIC branch) then returns
   `max(processed - computed, 0) = 0`.
3. Phase 2, `_schedule_waiting_requests`, takes the `num_new_tokens <= 0`
   branch: it pops the request, and `_set_waiting_queues_from_ordered_requests`
   rebuilds the queues without it.
4. No queue holds the request, so the drain check passes and the run exits 0
   with it incomplete.

Traced on the C2 dense `tight_kv` case: request 7 is preempted at t=1.093 with
45 of 55 tokens processed and leaves the waiting queue at t=1.157. The
KV-pressure sweep below shows the loss is exactly the decode-phase preemption:
before the fix every one of the 176 decode-phase preemptions lost its request.

The disaggregated DECODE and DECODE_ATTN roles were exempt from the reset since
`35eb631`, which is why only MONOLITHIC lost requests.

### Resolution (2026-09-23)

Rule. vLLM v1 preemption discards a request's computed KV but keeps its output
tokens (`num_computed_tokens = 0`; the prompt and output are recomputed). A
victim still in prefill has no output, so it restarts its prompt as before. A
victim past prefill keeps its Request-level progress; only the scheduler's
computed frontier and the KV allocation restart. On the next step it asks for
one token and allocates KV for its whole context. The one rule covers DECODE and
DECODE_ATTN too, whose requests always arrive past prefill, so the cluster-type
set `_REQUEST_PROGRESS_PRESERVING_PREEMPTION_CLUSTER_TYPES` is deleted.

Change (`75c1140`, `vllm_v1_kv_allocation.py`): the reset of `_num_processed_tokens` is
guarded by `not victim.is_prefill_complete` instead of the cluster-type set.

Tests.
- New `tests/integration/test_vllm_v1_decode_preemption_runtime.py`: three
  requests of 30+30 tokens with eight 16-token blocks through the real
  `Simulator`. It asserts that the run reaches a preemption past prefill, that
  the victim keeps its progress, and that every request completes all of its
  decode tokens.
- `tests/unit/test_pdaf_decode_attn_preemption.py`: the MONOLITHIC reset test is
  replaced by `test_monolithic_preemption_restarts_a_victim_still_in_prefill`,
  on a real `Request` in prefill; the disaggregated-decode fixture request
  states `is_prefill_complete=True`.

Checks, against the criteria fixed in plan §18.18 before measuring (baseline
`2ffb062`):

| Id | Result | Evidence |
| --- | --- | --- |
| B1 | Test passes with the fix. At `2ffb062` it fails on the progress assertion (`assert 0 == 34`); the same configuration there leaves request 1 incomplete with 0 of 30 decode tokens and exit 0. | `w9_05/evidence/negative_control_pytest.txt`, `symptom_probe.txt` |
| B2 | The three C2 `tight_kv` cases complete 24 of 24 with all decode tokens (before: dense 20, MoE `attn_dp=2` 20, MoE `attn_dp=4` 23). KV-pressure sweep, 72 cells: before, 41 cells lose 176 requests; after, 72 of 72 complete, 0 short outputs. The 41 lossy cells are exactly the cells with a decode-phase preemption after the fix. | `w9_05/evidence/c2_preemption_counts.txt`, `kv_pressure_sweep_summary.txt` |
| B3 | 21 of 24 identical. The 3 differing cases are the `tight_kv` cases, the only ones with any preemption (total 5, 6 and 2 after the fix); the other 21 have none on either side. | `w9_05/evidence/c2_2ffb062_vs_fix.txt`, `c2_preemption_counts.txt` |
| B4 | 72 of 72 deadlock-sweep cells drain with 24 of 24 requests, identical cell by cell to W9-04's result. | `w9_05/evidence/sweep_fix_*.txt` |
| B5 | 71 of 71 identical. No MONOLITHIC or PD-AF case preempts (`request_total_preemption_count` 0); the preemptions in the matrix are PDD PREFILL victims, still in prefill, and PDD DECODE victims, whose rule did not change. | `w9_05/evidence/fidelity_comparison.json` |
| B6 | G3b, G9 and G10: 51 of 51 PASS, `sha256sums.txt` identical. | `w9_05/evidence/stage_admission_compare_w905.json` |
| B7 | unit 84 failed / 3829 passed / 51 skipped / 10 errors; integration 5 errors / 28 passed / 22 skipped. 0 regressions, 0 new failures, 0 skip changes; the id changes are the renamed unit test and the new integration test. | `w9_05/evidence/{unit,integration}_compare.json` |
| B8 | 16 of 16 examples pass and are identical to `2ffb062`. | `w9_05/evidence/examples_2ffb062_vs_fix.txt` |

Limits and follow-ups:

- The replay cost is not modeled. vLLM recomputes the prompt and the generated
  output of a resumed request; Frontier resumes it with a one-token step. The
  DECODE and DECODE_ATTN roles have had the same simplification since `35eb631`.
  Modeling it would change how a resumed request's work is counted in
  Request/Batch and is a fidelity change that needs its own approval.
- Phase 2 still drops a waiting request silently when `num_new_tokens <= 0`,
  where vLLM asserts `num_new_tokens > 0`. With this fix no measured case reaches
  that branch with an incomplete request; turning it into an error is a
  separate change.
- `request_decode_preemption_count` and the `request_decode_tokens_at_preemption_*`
  columns cover only the PDD DECODE role. A MONOLITHIC preemption appears only
  in `request_total_preemption_count`.
- CPU runs with dummy or trained predictors on this host.

### Review addendum (2026-09-24)

The fix review (`test_report_2026-09-24_fix_review.md`) found that the
`75c1140` rule held at PP=1 but not at PP>1, where an earlier batch can still
carry the victim through a later stage when the preemption happens.

| Id | Defect at PP>1 | Fixed in `c647e95` by |
| --- | --- | --- |
| F-R1 | The victim kept its active-batch mark; the running phase skipped it for good and the run stalled. | Preemption drops the mark. |
| F-R2 | The stale step still credited its layers; the resumed step overran the layer counter. | Preemption resets the layers of a decode step the victim no longer runs. |
| F-R3 | The old batch's end released the victim again after its new batch admitted it. | Batch release iterates the batch's live requests only. |
| F-R4 | At PP>=4 a finished victim waiting for its terminal release re-entered the waiting queue. | Preemption retires it, as vLLM does when that output arrives. |

`tests/integration/test_vllm_v1_decode_preemption_runtime.py` now runs dense
PP4, MoE DP2 EP2 PP2 and a PP4 finished-victim case besides the original; each
fails on a tree without the fix its comment names
(`/data/ycfeng/tmp/issue26-correctness-pr/review_20260924/w9_05_regress/negctl/matrix.txt`).

Two fidelity differences remain, recorded as proposals because they change
results for any preempting run:

- F-R5. vLLM chooses the victim from `running[-1]` or the lowest priority,
  which can be the requesting request itself, and then stops scheduling it
  (`scheduler.py:470-548`). Frontier excludes the requester
  (`vllm_v1_kv_allocation.py:412-465`, `:661`).
- F-R6. vLLM's `update_from_output` applies the token that a preempted
  request's in-flight step sampled, and retires the request from waiting if
  that token stops it (`scheduler.py:1278-1324`). Frontier discards the token.

