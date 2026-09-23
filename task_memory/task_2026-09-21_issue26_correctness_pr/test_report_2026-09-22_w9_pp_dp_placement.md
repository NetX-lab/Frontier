# Test report — Step 9, PP>1 support for the vLLM DP placement policy

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created. Covers packages P1(a), P1(b) and G1. |
| 2026-09-23 | §2 addendum: P1(b) rerun after the W9-01 merge-forward on seven shapes, with key scoring. §4 updated. |
| 2026-09-23 | §4–§6 added: P2/P3 pointer, P4 real-loop integration evidence, P5 fidelity and regression evidence, W9-04/W9-05 found in P5. Summary renumbered to §7. |
| 2026-09-23 | §8 added: the W9-04 fix `2ffb062` and its checks A1–A7; §6.4 and §7 updated. |
| 2026-09-23 | §9 added: the W9-05 fix `75c1140` and its checks B1–B8; §6.4 and §7 updated. |

Environment for every CPU check below:

| Item | Value |
| --- | --- |
| Host | `kun-workspace-vgen2` |
| Worktree | `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`, branch `fix/issue26-correctness-pr` |
| Interpreter | `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10, no `torch`) |
| `PYTHONPATH` | the worktree root |

## 1. P1(a) — reference-loop oracle

Command:

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_dp_placement_reference_loop.py -q -p no:cacheprovider
```

Expected: every row of the plan's §18.11 state table is reproduced by a model
of the vLLM engine iteration alone, and the model drives the real
`VllmDPLoadBalancer` rather than a second copy of it.

Actual: `9 passed in 1.21s`. **PASS.**

Together with `tests/unit/test_vllm_dp_load_balancer.py`:
`70 passed in 1.44s`. **PASS.**

Row-by-row result:

| Scripted iteration | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Depth 1, three iterations | Each schedules and applies; loads (0,2), (0,1), (0,0) | as expected | PASS |
| Depth 2, cold fill | First `scheduled=True, applied=False`, load (0,3), published; second applies | as expected | PASS |
| Depth 3, oldest already ready | One combined publication, load (0,4) | one publication | PASS |
| Depth 3, zero-token schedule | No early return; applies; counts unchanged so nothing published | `published=False` | PASS |
| Depth 2, drain | `scheduled=False, applied=True`, published, load (0,0) | as expected | PASS |
| Depth 3, three admissions | Steps 0, 1, 2; first two do not apply; all published | as expected | PASS |
| Engine stepped with no work | `ValueError` | raised | PASS |
| Two peers at the same index | Equal step, both counts visible to the frontend | equal, frontend sees `[(0,3), (0,3)]` | PASS |
| One peer one iteration ahead | Counters diverge | 2 against 0 | PASS |

Conclusion carried into the design: at depth 1 the admission-only iteration
cannot occur, so Frontier's completion-only report is already exact at PP=1;
above depth 1 it is not, and no fixed stride between completion keys can hold
because depth 3 produces two consecutive admission-only publications.

## 2. P1(b) — Frontier boundary probe

Scratch driver: `$SCRATCH/w9/probe_frontier_boundaries.py` (one process per
shape; `IS_MOE` is a process global in this codebase). Each shape builds a
`MONOLITHIC` single-Replica `vllm_v1` configuration on a 6-layer model so that
PP 1, 2 and 3 all divide the layer count, runs the Simulator, and records at
every admission and completion the lane, the slot occupancy and the value of
`ForwardSyncState._next_step_id_by_replica`, the candidate report key.

Expected: every shape completes all six requests, and the candidate key is
equal for peers of one forward and distinct for distinct engine iterations.

| Shape | Requests completed | Boundaries | Candidate key behavior | Verdict |
| --- | --- | --- | --- | --- |
| `attn_dp=2, moe_ep=2, PP=1` | 6/6 | 24 | Peers always equal; values 0, 6, 12, 18, 24; each completion shares a value with the admission it triggers | PASS |
| `attn_dp=1, moe_ep=1, PP=2` | 6/6 | 28 | Both cold-fill admissions read 0 | FAIL (invariant I5) |
| `attn_dp=1, moe_ep=1, PP=3` | 6/6 | 32 | All three cold-fill admissions read 0 | FAIL (invariant I5) |
| `attn_dp=2, moe_ep=2, PP=2` | 0/4 | — | Event queue drained with requests unfinished | FAIL (W9-01) |
| `attn_dp=2, moe_ep=2, PP=3` | — | — | Rejected at construction | FAIL (W9-02) |

Request-count sensitivity of the deadlock, fresh process each time:

| Requests | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- |
| `attn_dp=2, moe_ep=2, PP=2` | completes | completes | drained | drained | drained |

Scheduler state at drain, stage `(0, 0)`:

```
capacity=2  active_seqs=[0]  waiting_fifo_seqs=[1, 2, 3]
sealed=False  ep_active=False
```

W9-02 rejection message:

```
collective-sim physical topology requires cluster_total_devices 6 to be
divisible by node size 4
```

Both are recorded in `issues.md`. The three source files involved in W9-01 are
byte-identical to `origin/main`, verified with `git rev-parse HEAD:<path>`
against `origin/main:<path>`.

### 2.1 Addendum (2026-09-23): the multi-lane PP shapes after W9-01

Revision: `d3e6e78` plus the untracked probe files. The rule file equals
`origin/main`, PR 36 `4ab1964`.

Driver: `step9_p1b/probe_boundaries.py`. It builds each shape with the
`tests.e2e.stage_admission_matrix` fixture: a 6-layer synthetic model,
analytical CC backend, round-robin placement, dummy predictor, prefill 16 and
decode 3 tokens, 6 requests. The driver records admissions, completions,
stage-0 starts with their forward group, and layer rooms, together with the
stage-0 state at each boundary. The scorer is `step9_p1b/analyze_keys.py`,
with method and targets in its docstring.

Commands, with `PYTHONPATH=<worktree>`, `WANDB_DISABLED=true`,
`VIDUR_DISABLE_WANDB=1`, `FRONTIER_TMP_ROOT=/data/ycfeng/tmp` and
`/data/ycfeng/envs/frontier-py310/bin/python`:

```bash
python task_memory/.../step9_p1b/probe_boundaries.py <shape> /data/ycfeng/tmp/issue26-correctness-pr/step9_p1b/<shape>
python task_memory/.../step9_p1b/analyze_keys.py /data/ycfeng/tmp/issue26-correctness-pr/step9_p1b <key_scores.json>
```

Expected results:

1. Every shape completes 6/6.
2. For the key to be accepted, it has 0 peer splits, 0 merges and 0
   inversions against each report's actual stage-0 group, and 0 ms of replay
   mismatch.

| Shape | Completed | Records | A (split/merge/inv, ms) | Lane counter | Group-anchored | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| MoE dp2 PP1 burst | 6/6 | 108 | 0/0/0, 0 | 0/0/0, 0 | 0/0/0, 0 | PASS |
| MoE dp1 PP2 burst | 6/6 | 140 | 0/0/0, 0 | 0/0/0, 0 | 0/0/0, 0 | PASS |
| MoE dp1 PP3 burst | 6/6 | 176 | 0/1/0, 400 | 5/0/0, 0 | 0/0/0, 0 | PASS for group-anchored |
| MoE dp2 PP2 burst (the W9-01 shape) | 6/6 | 160 | 0/0/0, 0 | 4/0/0, 0 | 0/0/0, 0 | PASS for group-anchored; was FAIL (W9-01) |
| MoE dp2 PP2 staggered | 6/6 | 176 | 0/0/0, 0 | 6/0/0, 0 | 0/0/0, 0 | PASS for group-anchored |
| MoE dp2 PP3 burst | 6/6 | 198 | 0/4/0, 148 | 0/0/0, 0 | 0/0/0, 0 | PASS for group-anchored; was FAIL (W9-02, collective-sim only) |
| MoE dp2 PP3 staggered | 6/6 | 210 | 4/1/0, 200 | 15/7/5, 0 | 0/0/0, 0 | PASS for group-anchored |

Split/merge counts for the lane counter on burst shapes fall on
completion-only rows. There the Frontier target is the lane's next forward,
while the reference would count a separate dummy iteration. The
real-forward-only counts are in `key_scores.json` (`forward_reports`). The
group-anchored key equals the stage-0 predictor on every report: 139 of 139.

Limits:

- Targets are Frontier's own forward grouping, not vLLM measurements.
- 12 final drain completions have no later forward and are unscored.
- The replay metric depends on this small workload's timing. The pairwise
  counts are the primary evidence.
- W9-03 (reference lockstep) is a source-reading observation.

## 3. G1 — ground-truth instrumentation

Checkout `/data/ycfeng/Frontier/.real-engine/vLLM-BS`, local branch
`feature/frontier-comparison-instrumentation` created at the remote tip
`ea95f571e` and committed as `494b9f327`. Tree clean, nothing pushed (D-b).

Changed files, 212 insertions and 10 deletions:

| File | Change |
| --- | --- |
| `vllm/v1/frontier_trace.py` | Buffered per-process JSONL writer gated by `VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR` |
| `vllm/v1/engine/core.py` | Iteration classification at both step methods; publish helper returns whether it published; the DP busy loop emits the record |
| `vllm/v1/engine/coordinator.py` | Receive disposition and publication snapshot id, sent on to the front ends |
| `vllm/v1/engine/core_client.py` | Applied snapshot and routing decision |

`vllm/v1/core/sched/scheduler.py` is not changed. The plan expected a `dp_rank`
column there; it is unnecessary because `SchedulerOutput` already carries the
scheduled request ids and the record is written by the engine that owns the
rank.

Writer check, since the module is pure standard library and can be loaded
without `torch`:

```bash
/data/ycfeng/envs/frontier-py310/bin/python $SCRATCH/w9/check_trace_writer.py
```

Expected: nothing is written while the gate is off; records buffer until the
bound or an explicit flush; one file per process id; `seq` is dense and
ordered; a second flush is a no-op; the warmup gate suppresses records.

Actual: `frontier_trace DP placement writer: all checks passed`. **PASS.**
Sample record:

```json
{"engine": 1, "kind": "engine_iteration", "monotonic": 549631.140559739,
 "pid": 3180674, "published": true, "running": 0, "seq": 0, "step": 0,
 "waiting": 0}
```

Syntax check of all four changed files with `python -m py_compile`: **PASS.**

Verification limit: the instrumented engine paths have not been executed. They
need a GPU host with the compiled vLLM extensions, which is package G3. What is
established here is that the writer behaves as specified and that the four
files parse; that the records are emitted at the right points is asserted from
source reading, not from a run.

## 4. P2 and P3 — implementation and unit tests

Recorded in `progress.md`, "Step 9 P2 and P3". 132 targeted tests pass; the
19 new or changed cases fail on the pre-P2 tree.

Before P2, the constructor rejected every PP>1 shape. The evidence is the old
guard case `pipeline_parallel`, message "one co-location Replica", which passed
on the pre-P2 tree and was inverted into construct cases by P3.

## 5. P4 — real event loop (`tests/integration/test_vllm_dp_placement_runtime.py`)

Commit `bacdbb4`. Command:

```bash
PYTHONPATH=$PWD python -m pytest tests/integration/test_vllm_dp_placement_runtime.py \
  tests/integration/test_monolithic_mixed_forward_runtime.py -q -p no:cacheprovider
```

Each child run wraps the two policy seams, `VllmDPLoadBalancer.report` and
`.select`, and stage-0 `pop_batch_if_not_busy`. The parent then checks every
report against the reference engine iteration it stands for:

- An admission reports on its own if and only if `running_after < PP`, and its
  key is the batch's own forward.
- A completion that follows a held admission reports under the held batch's
  forward.
- A completion with nothing held reports a key greater than the lane's last
  forward.
- Keys never decrease per lane.

Report kinds per case, from the child evidence, in
`/data/ycfeng/tmp/issue26-correctness-pr/step9_p4`:

| Case | PP | Admission-only | Held | Folded into a completion | Completion-only |
| --- | --- | --- | --- | --- | --- |
| `moe_dp2` | 1 | 0 | 10 | 10 | 0 |
| `dense_dp1_pp2` | 2 | 1 | 9 | 9 | 1 |
| `moe_dp2_pp2` | 2 | 2 | 10 | 10 | 2 |
| `moe_dp2_pp2_online` | 2 | 45 | 0 | 0 | 45 |
| `moe_dp1_pp3` | 3 | 3 | 8 | 8 | 3 |
| `moe_dp2_pp3` | 3 | 12 | 0 | 0 | 12 |
| `moe_dp2_pp2_discriminating` | 2 | 8 | 18 | 18 | 8 |

Every run conserves requests and tokens and releases every lane and stage
context. Each policy run adds no event type, and its makespan equals its
comparison run's.

Discriminating case (plan §18.6). The comparison run differs from the policy
run in its two seams only: there is no admission report, and each completion
is keyed by `ForwardSyncState.get_step_id`. A burst at 1.0 s is routed
0, 1, 0, 1, 0. A probe arrives at 1.1 s, and the first completion follows at
1.68 s.

| Run | Snapshot at the probe | Probe lane |
| --- | --- | --- |
| policy | `[[0, 3], [1, 1]]` (both lanes' first admissions published) | 0 |
| completion-reporting control | `[[3, 0], [2, 0]]` (reservations only) | 1 |

`tests/integration/test_monolithic_mixed_forward_runtime.py` gains a PP2 run
under `vllm_load_balancing`, `hybrid_layers_pp2_dp_placement`. It has 3 mixed
dense completions, and all 15 decode tokens peak at 4 dense layers; 6 of 6
requests complete. The PP1 values are unchanged: 4 mixed-phase cohorts out of
24.

Result: `tests/integration/test_vllm_dp_placement_runtime.py` 9 passed,
`test_monolithic_mixed_forward_runtime.py` 3 passed (JUnit of the §6.3 run). **PASS.**

## 6. P5 — unchanged behavior (C2) and the Step 8 regression set

Before: `d1a2a06`, whose `frontier/` equals the pre-P2 source. After:
`bacdbb4`. Scripts and evidence are in `step9_p5/`; raw runs are in
`/data/ycfeng/tmp/issue26-correctness-pr/step9_p5`.

### 6.1 PP=1 `vllm_load_balancing` scenarios

`step9_p5/c2_pp1_policy_matrix.py` runs 24 scenarios on `git archive` exports
of both commits:

- shapes: MoE `attn_dp=2, moe_ep=2`, MoE `attn_dp=4, moe_ep=4`, dense
  `attn_dp=1`;
- workloads: offline bursts of 4, 16 and 24 requests (fixed and uniform
  lengths); online Poisson at qps 20, 50 and 200; qps 200 with 12 KV blocks;
  and the asymmetric trace of the integration test.

Each child imports `frontier` from its tree only and writes metrics. It also
records every report `(time, engine, key, load)` and every selection
`(time, snapshot, engine)`.

Pass rule (Q11 plus C2):

- Every artifact is equal after path substitution (the refactor-fidelity
  comparator).
- The report stream is equal without the key.
- The keys are order-isomorphic: every pair compares the same way on both
  sides.
- The selections are equal.

| Result | Count |
| --- | --- |
| identical | **24 of 24** |
| drained on both sides | 20 |
| incomplete on both sides (20 of 24 requests; W9-05) | 2 (`*_tight_kv`, MoE `attn_dp=2` and dense) |
| stopped without draining on both sides, identical diagnostics (W9-04) | 2 (MoE `attn_dp=4`, qps 200, both KV sizes) |
| additional admission-only reports | 0 (report counts equal in every case) |

The key values differ for MoE: `3, 3, 7, 7, …` before and `0, 0, 1, 1, …`
after. For dense `attn_dp=1` they are equal. The balancer uses only key
comparisons, so the order-isomorphism check is the one that matters.

### 6.2 Other cluster schedulers

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Refactor fidelity matrix | `step9_p5/run_fidelity.sh` (clean detached worktrees at both commits, harness from `bacdbb4`, `--clean-cache`) | 71 of 71 identical | 71 of 71 identical; 0 provenance findings; `complete_comparison` and `predictor_cache_populated_cleanly` true | PASS |
| 16 Step 8 architecture examples | `step9_p5/run_examples.sh` on both exports, then `compare_examples.py` | 16 pass on both; artifacts identical | 16 / 16 pass on both; 16 of 16 identical | PASS |

### 6.3 Suites

`w9_01_stage_admission_ordering/composition_run_suites.sh` on the clean
`bacdbb4` worktree. It is compared by test id with `composition_compare_junit.py`
against the K4 JUnit of the merged tree `03d5f24`. `frontier/` and `tests/`
are unchanged between `03d5f24` and `d1a2a06`.

| Suite | Before (`03d5f24`) | After (`bacdbb4`) | Regressions / new failures / skip changes |
| --- | --- | --- | --- |
| unit | 84 failed, 3814 passed, 50 skipped, 10 errors | 84 failed, 3829 passed, 51 skipped, 10 errors | 0 / 0 / 0 |
| integration | 5 errors, 19 passed, 22 skipped | 5 errors, 26 passed, 22 skipped | 0 / 0 / 0 |

Tests found on one side only:

- Unit, before only (12): 8 are the old `test_vllm_dp_load_balancer` ids that
  P3 renamed or inverted.
- Unit, before only: the other 4 are `test_collective_sim_zero_payload`. The
  detached worktree has no initialized collective-sim submodule, so that module
  skips as a whole; this is the added skip. In the development worktree it
  gives 4 passed.
- Unit, after only (28): the P3 cases and that module-level skip.
- Integration, after only (7): P4's new cases.

The 5 integration errors are the absent PD-AF Reference checkout, as before.

### 6.4 Found during P5

Both defects are pre-existing; neither is caused by Step 9. See `issues.md`.

- **W9-04**: MoE `attn_dp=4` online runs can deadlock when a lane joins a
  forward after it was given a first-layer placeholder. It is reachable on this
  branch under `round_robin` in 5 of 24 sweep cells. A scratch prototype
  (`step9_p5/w9_04_prototype.patch`) drains all 72 sweep cells and leaves
  every previously drained C2 case identical. The user chose to fix it in
  this PR; see §8.
- **W9-05**: under KV pressure, `vllm_v1` loses requests mid-decode without
  an error. It is also present on `origin/main`. The user scheduled it the same day;
  see §9.

## 7. Summary

| Package | Verdict |
| --- | --- |
| P1(a) | PASS |
| P1(b) | PASS on seven shapes after the W9-01 merge-forward (§2.1) |
| Design checkpoint D9-1 | Settled |
| Design checkpoint D9-2 | Settled: group-anchored key (user decision 2026-09-23) |
| G1 | PASS for what is testable without a GPU |
| P2, P3 | PASS (§4) |
| P4 | PASS (§5) |
| P5 / C2 | PASS (§6): 24 of 24 PP=1 policy scenarios, 71 of 71 fidelity cases and 16 of 16 examples identical; 0 suite regressions |
| W9-04 fix (§8) | PASS: A1–A7 |
| W9-05 fix (§9, `75c1140`) | PASS: B1–B8 |
| G3–G5 | BLOCKED on GPU authorization |

## 8. W9-04 fix (`2ffb062`)

Change, decision and argument: `issues.md` W9-04 "Resolution". Criteria:
plan §18.17, fixed before measuring. The baseline is `bacdbb4`, whose
`frontier/` differs from `2ffb062` only in `sync_entry.py`.

| Id | Check | Result |
| --- | --- | --- |
| A1 | New case `moe_dp4_late_join`; control tree with the call removed | 10 passed; control fails with non-empty scheduler state |
| A2 | Sweep, `round_robin` / `lor` / `random` | 72 of 72 drain (before: 5 and 1 stalled) |
| A3 | C2 PP=1 policy scenarios | 22 of 22 identical; the 2 stalled cases finish at 24/24 and 23/24 (W9-05 signature) |
| A4 | Refactor fidelity matrix | 71 of 71 identical, 0 provenance findings |
| A5 | Stage-admission G3b, G9, G10 | 51 of 51 PASS, hashes identical |
| A6 | Unit and integration suites against P5 | 0 regressions, 0 new failures, 0 skip changes; +1 test id |
| A7 | 16 architecture examples | 16 of 16 identical |

What the regression case shows, from its evidence:

- Requests go to lanes 0, 1 and 2.
- Each lane's first stage-0 batch is bound to forward group 0.
- The one withdrawn placeholder is lane 2's.
- Both policy runs complete 3 of 3, conserve tokens, and release every lane
  and stage context.

Limits are in `validation.md` "W9-04 fix".

## 9. W9-05 fix (`75c1140`)

Mechanism, change, decision and limits: `issues.md` W9-05. Criteria: plan
§18.18, fixed before measuring. The baseline is `2ffb062`; `frontier/` differs
only in `vllm_v1_kv_allocation.py`.

| Id | Check | Result |
| --- | --- | --- |
| B1 | New `test_vllm_v1_decode_preemption_runtime.py`; the same test and a probe on `2ffb062` | 1 passed; at `2ffb062` `assert 0 == 34`, and request 1 ends incomplete with exit 0 |
| B2 | C2 `tight_kv`, three shapes; KV-pressure sweep, 72 cells | 24/24 in each (before 20, 20, 23); sweep 72 of 72 complete (before 31 of 72, 176 requests lost) |
| B3 | C2 PP=1 policy scenarios | 21 of 24 identical, all without preemption; the 3 `tight_kv` cases differ and complete more |
| B4 | Deadlock sweep, three cluster schedulers | 72 of 72 drain, equal to W9-04's cells |
| B5 | Refactor fidelity matrix | 71 of 71 identical |
| B6 | Stage-admission G3b, G9, G10 | 51 of 51 PASS, hashes identical |
| B7 | Unit and integration suites against the W9-04 JUnit | 0 regressions, 0 new failures, 0 skip changes; one test renamed, one added |
| B8 | 16 architecture examples | 16 of 16 identical |

What the sweep shows, from `w9_05/evidence/kv_pressure_sweep_summary.txt`:

- Before the fix, each of the 176 decode-phase preemptions lost its request.
- The 41 lossy cells are exactly the 41 cells in which the fixed tree records a
  decode-phase preemption.
- Losses fall as KV grows: at `num_blocks=24` no shape has a decode-phase
  preemption, and no cell lost a request.

The fidelity matrix has no MONOLITHIC preemption, so B5 shows that the other
paths are unchanged; B1–B3 cover the changed branch. Run notes (exported tree
versus git worktree for B7, the symlink path in B8) are in `validation.md`
"W9-05 fix".
