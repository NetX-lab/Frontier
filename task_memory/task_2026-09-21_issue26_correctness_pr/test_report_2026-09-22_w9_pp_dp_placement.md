# Test report — Step 9, PP>1 support for the vLLM DP placement policy

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created. Covers packages P1(a), P1(b) and G1. |
| 2026-09-23 | §2 addendum: P1(b) rerun after the W9-01 merge-forward on seven shapes, with key scoring. §4 updated. |

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

## 4. Summary

| Package | Verdict |
| --- | --- |
| P1(a) | PASS |
| P1(b) | PASS on seven shapes after the W9-01 merge-forward (§2.1) |
| Design checkpoint D9-1 | Settled |
| Design checkpoint D9-2 | Proposed (group-anchored key); awaiting the user's decision |
| G1 | PASS for what is testable without a GPU |
