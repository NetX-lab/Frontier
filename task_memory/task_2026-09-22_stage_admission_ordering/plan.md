# Stage admission ordering under pipeline parallelism — Plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | D-1..D-5 adopted by the user; D-3 executed now (push + draft PR for remote review); D-5 adjusted so the records travel with the branch. Work packages P0–P4 unblocked. |
| 2026-09-22 | Created for user review. Scope, acceptance criteria, work packages P0–P4 with dependencies, verification matrix, decisions D-1..D-4. No source change yet. |

Diagnosis, options and the recommended rule are in `design.md`. This file is the
executable plan.

## 1. Scope

Fix the pre-existing stage admission deadlock (W9-01 in the parent task) on its
own branch:

- Branch `fix/stage-admission-ordering`, base `origin/main` `1f694f7`, worktree
  `/data/ycfeng/Frontier/.worktrees/stage-admission-ordering`.
- Source change confined to
  `frontier/scheduler/replica_stage_scheduler/stage_execution_context.py`:
  the full-stage admission predicate in `try_acquire` and the two docstrings
  that describe admission as "FIFO-head".
- Out of scope: `full_stage_capacity`, the forward-group seal, EP wave
  protocol, sync rooms, wake-up helper, any configuration field, and the Step 9
  report key.

## 2. Acceptance criteria

| Id | Criterion | Settled by |
| --- | --- | --- |
| C1 | MoE `attn_dp∈{2,4}`, `moe_ep = attn_dp`, `PP∈{2,3}` (layer count divisible by PP), offline, 4/6/8/12 requests: every request completes; request count, prefill tokens and decode tokens conserved; no "non-empty scheduler state" drain. | P2(c) integration test; P3 matrix. |
| C2 | Every `PP = 1` scenario and every PD-AF scenario in the matrix produces byte-identical `request_metrics.csv` and `system_metrics.json` before and after. | P0 baseline vs P3 rerun, SHA-256 per file. |
| C3 | Dense `attn_dp∈{2,4}`, `PP∈{2,3}`: completes before and after; the number of simulated intervals during which two lanes of one stage are busy at once is greater after than before; every metric difference is attributable to that overlap. | P3, with the lane-overlap count from `metrics_ground_truth.jsonl` or the batch-stage trace. |
| C4 | All existing unit tests pass without modification. If any existing test must change, stop and report; that is a design signal, not a test fix. | P2. |
| C5 | The predicate is one readable condition; module and method docstrings state the ordering contract as implemented; no flag, no config field, no `getattr` fallbacks. | Review of the diff against the quality gates. |
| C6 | The Step 9 boundary probe on MoE `attn_dp=2, moe_ep=2, PP=2` runs to completion on this branch (informational: it unblocks the parent task's design checkpoint). | P3, scratch probe rerun. |

## 3. Work packages

```text
P0 baseline capture (no source change)
    -> P1 rule change
    -> {P2 tests, P3 fidelity rerun and comparison}
    -> P4 records, commit, push and draft PR (D-3)
```

| Package | Content | Acceptance |
| --- | --- | --- |
| P0 Baseline | On `1f694f7`, run the verification matrix of §4 and store outputs under `/data/ycfeng/tmp/stage_admission_ordering/baseline/<scenario>/` with a `sha256sums.txt` per scenario. Store the drain evidence already collected (`design.md` tables) and the reproduction scripts (`repro_main.py`, `drain_state.py`, `drain_lanes.py`, currently in the session scratchpad) under `tests/integration/stage_admission/` as the seed of P2(c). | Every scenario has a hash file; the four drain shapes are recorded as `DRAINED`. |
| P1 Rule | Implement the recommended rule from `design.md`: full-stage tickets are refused only by an EP wave queued ahead; EP waves keep the strict head rule. Update the `StageExecutionContext` class docstring ("A complete operation first enters the ready FIFO, then the owner admits it atomically") and the `try_acquire` docstring ("Acquire the FIFO-head ticket if this stage is currently idle") to describe what is now true. | Diff touches one source file; `python -m pytest tests/unit/test_stage_execution_context.py tests/unit/test_shared_forward_group_admission.py -q` passes unchanged. |
| P2 Tests | (a) Context-level, in `tests/unit/test_stage_execution_context.py`: a full-stage ticket queued behind another lane's queued full-stage ticket is admitted while capacity remains; a full-stage ticket queued behind an EP wave is refused; the two existing EP-order tests stay as they are. (b) Scheduler-level, in `tests/unit/test_shared_forward_group_admission.py` using its `make_stage`/`make_batch` helpers: rebuild the drain state (lane 1 active with a second ticket queued, lane 0 with two queued) and assert lane 0's `pop_batch_if_not_busy` returns its head and binds the same forward group. (c) Simulator-level, `tests/integration/test_stage_admission_pipeline_lanes.py`: one child process per shape (`IS_MOE` is process-global), MoE `attn_dp=2, moe_ep=2, PP=2` at 4 and 6 requests and `attn_dp=4, moe_ep=4, PP=2` at 8, asserting completion and conservation; dense `attn_dp=2, PP=2` asserting completion and lane overlap > 0. Expected values written from the scenario, not copied from a run. | (a) and (b) fail on `1f694f7` and pass after P1; (c) drains on `1f694f7` and passes after P1. |
| P3 Fidelity | Rerun the §4 matrix on the P1 revision into `.../after/<scenario>/`; `diff` the hash files; for every differing scenario, confirm it is in the C3 class and record the overlap count before/after and the makespan delta. Rerun the Step 9 boundary probe for C6. | C1, C2, C3, C6 tables in `test_report_2026-09-22_stage_admission_ordering.md`. |
| P4 Records | Test report, `progress.md`, `summary.md`; commit P1+P2 as one code commit and records as one docs commit; then, under D-3, push the branch and open a draft PR against `main` whose body carries the C1–C3 tables and the W9-01 cross-reference. Note in the parent task (`issues.md` W9-01) the branch and commit. | Pushed and verified, or explicitly left local if D-3 is not granted. |

## 4. Verification matrix

Environment: `/data/ycfeng/envs/frontier-py310/bin/python`, `PYTHONPATH` = the
worktree, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. Each Simulator run in a
fresh process.

| Group | Scenarios | Expected |
| --- | --- | --- |
| G1 release examples | The 30 `examples/architecture/{co-location,pdd,pd-af-disagg}/{offline,online}/*.sh` recipes (all `PP=1`), metrics dir redirected per scenario | Byte-identical (C2) |
| G2 Step 8 regression set | `pytest tests/unit -q --continue-on-collection-errors` and `pytest tests/integration -q --continue-on-collection-errors`; compare the FAILED **set** to the recorded `1f694f7` baseline (84 entries) | Same set (C4, C2) |
| G3 lanes × stages, MoE | `attn_dp∈{2,4}`, `moe_ep=attn_dp`, `PP∈{1,2,3}`, 6-layer tiny model (all three PP values divide 6), offline, requests ∈ {4, 6, 8, 12} | `PP=1` byte-identical; `PP∈{2,3}` drain → complete (C1) |
| G4 lanes × stages, dense | `attn_dp∈{2,4}`, `PP∈{1,2,3}`, same model with `is_moe=False`, requests ∈ {6, 8} | `PP=1` byte-identical; `PP∈{2,3}` overlap increases (C3) |
| G5 single lane | `attn_dp=1`, `PP∈{1,2,3}`, MoE and dense | Byte-identical |
| G6 PD-AF capacity-1 | The 10 PD-AF recipes from G1 plus `tests/unit/test_mixed_layer_decode_ffn_scheduling.py`, `test_decode_ep_wave_materialization.py`, `test_prefill_ep_wave_materialization.py` | Byte-identical; tests pass (C2, C4) |

Scenario count: 30 + 2 suites + 24 (G3) + 12 (G4) + 6 (G5) = 72 Simulator runs
plus the two pytest suites, satisfying the AGENTS.md gate of at least 50
concrete settings.

Note on G3 `PP=3` with `attn_dp=2`: the collective-sim topology check rejects
`attn_dp=2, moe_ep=2, PP=3` (6 devices against node size 4). Those cells use
`attn_dp=4, moe_ep=4, PP=3` (12 devices) or are marked "rejected at
construction" and excluded; either way the rejection itself is byte-identical
before and after.

## 5. Decisions (adopted by the user on 2026-09-23: "采纳你d1-d5的推荐决策")

| Id | Question | Recommendation | Outcome |
| --- | --- | --- | --- |
| D-1 | Adopt option B from `design.md` (full-stage tickets are ordered only behind EP waves) rather than option A (lane-aware skip) or C/D. | B. A needs a wake path on acquisition that the DES does not have; C and D are rejected on the working gates. | Adopted. |
| D-2 | Accept that dense `attn_dp>1, PP>1` timelines change (lanes overlap instead of serializing). | Accept as a fidelity fix; the serialization is the same defect. | Adopted. |
| D-3 | Authorize pushing `fix/stage-admission-ordering` and opening a draft PR against `main`. | Grant at P4; until then everything stays local. | Adopted and brought forward: the user reviews on the remote, so the branch is pushed and a draft PR opened with the plan itself (2026-09-23). Code commits follow per package. |
| D-4 | Baseline for byte comparison is `origin/main` `1f694f7`. PR 35 will merge this branch later instead of carrying the fix itself. | Confirm. | Adopted. |
| D-5 | `task_memory/` is ignored by `.gitignore` on `main` (line 171), so these records are local to the worktree unless force-added. Keep them local and archive the outcome in the parent task, or track them on this branch as PR 35 does? | Keep local; copy `summary.md` and the test report into the parent task at P4. | Adopted with one adjustment required by D-3: remote review needs the records on the branch, so `.gitignore` gets the same narrow exception PR 34/35 use (`task_memory/*` plus `!task_memory/task_2026-09-22_stage_admission_ordering/`). The copy into the parent task at P4 stands. |

## 6. Dependencies and risks

- The reproduction and boundary scripts live in the session scratchpad
  (`w10/repro_main.py`, `w10/drain_state.py`, `w10/drain_lanes.py`); P0 moves
  them under `tests/`.
- Risk: a `DECODE_FFN` path that queues two distinct full-stage tickets from
  different sibling lanes at the same time would see admission order change.
  `design.md` argues this does not occur; G6 measures it. If G6 differs, stop
  and report before adjusting anything.
- Risk: the C3 overlap count needs a per-lane busy-interval source. If
  `metrics_ground_truth.jsonl` lacks stage-level lane intervals, P3 derives
  them from the batch-stage trace; adding metrics is out of scope.
- The parent task's Step 9 resumes only after this branch is merged into `main`
  and merged forward into `fix/issue26-correctness-pr`.
