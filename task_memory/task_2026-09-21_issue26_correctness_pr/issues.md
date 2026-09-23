# Issues

## Modification History

| Date | Change |
| --- | --- |
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
   In progress.

Step 9 C1's PP3 row stays on `attn_dp=1` (W9-02).

## W9-02 `attn_dp=2, moe_ep=2, num_pipeline_stages=3` is rejected at config time

Status: open, expected behavior, affects plan wording only.
Found: 2026-09-22, Step 9 package P1(b).

Construction fails with:

```
collective-sim physical topology requires cluster_total_devices 6 to be
divisible by node size 4
```

The rejection is correct: 2 lanes x 3 stages is 6 devices against a node size of
4. Plan acceptance criterion C1 lists MoE `attn_dp=2` at both PP2 and PP3; the
PP3 row must be `attn_dp=1`, or must choose a device count that divides the node
size. C1 was amended accordingly.
