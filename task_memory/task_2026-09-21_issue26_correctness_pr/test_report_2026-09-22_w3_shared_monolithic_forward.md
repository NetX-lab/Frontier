# Test report — W3, one shared monolithic forward (2026-09-22)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created. Acceptance evidence for the shared monolithic forward lifecycle, including four deliberate-defect controls. |
| 2026-09-22 | Fidelity matrix recorded (71 of 71 identical). Controls rebuilt against the final test file and re-run; counts and failure messages updated. |

## Environment

| Item | Value |
| --- | --- |
| Worktree | `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr` |
| Branch / commit under test | `fix/issue26-correctness-pr` @ `65ed8a7` |
| Baseline | `/data/ycfeng/Frontier/.worktrees/w3-baseline`, detached at `3d47417` (this branch's parent) |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python` (3.10) |
| Invocation | `PYTHONPATH=$PWD python -m pytest … -p no:cacheprovider` from the worktree root |
| Absent optional deps | `torch`, `matplotlib` — 10 GPU/plot-only unit modules are excluded on both sides by the same `--ignore` list |

Control trees live under the session scratchpad
(`.../scratchpad/w3-control-*`) and are driven by `run_against_tree.py`, which
removes the editable-install meta-path finder and prepends the chosen tree, so
each control imports exactly the source it names.

## What was measured

| # | Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | W3 behavior matrix | `pytest tests/unit/test_monolithic_mixed_forward_sync.py` | all pass | 23 passed | PASS |
| 2 | Real-runtime acceptance | `pytest tests/integration/test_monolithic_mixed_forward_runtime.py` | 1 pass, a mixed-phase cohort actually reached | 1 passed; 24 cohorts, 4 mixed-phase, 4/4 requests completed | PASS |
| 3 | Forward-sync regression set (17 files) | see list below | all pass | 342 passed | PASS |
| 4 | Full unit suite vs baseline | `pytest tests/unit` both sides | identical failure set | 84 failed on both, identical identities; 3717 vs 3694 passed (+23 new) | PASS |
| 5 | Integration suite vs baseline | `pytest tests/integration` both sides | identical error set | 5 errors on both (PD-AF Reference checkout absent); 12 vs 11 passed (+1 new) | PASS |
| 6 | Fidelity matrix | `run_matrix.py run/compare` | all 71 cases exactly equal (expectation recorded in `design.md` before measuring) | 71 of 71 compared, 71 identical, 0 mismatched | PASS |

### Check 3 — the forward-sync regression set

```
tests/unit/test_decode_ep_wave_materialization.py
tests/unit/test_forward_sync_state.py
tests/unit/test_mixed_layer_decode_ffn_scheduling.py
tests/unit/test_moe_routing_conservation.py
tests/unit/test_pdaf_parity_harness_wave2_events.py
tests/unit/test_pdaf_prefill_model_time.py
tests/unit/test_pd_decode_moe_layer_accounting.py
tests/unit/test_prefill_ep_wave_materialization.py
tests/unit/test_review_comment_fixes.py
tests/unit/test_shared_ep_layer_protocol_guard.py
tests/unit/test_shared_forward_group_admission.py
tests/unit/test_stage_reporting_contract.py
tests/unit/test_cluster_scheduler_dp_lanes.py
tests/unit/test_collective_timing.py
tests/unit/test_dense_layer_complete_event.py
tests/unit/test_ep_wave_trace_context.py
tests/integration/test_online_pdd_forward_groups.py
```

These cover forward identity, EP-wave materialization for both phases, stage
ownership and admission, stage execution reporting, and the sequential PDD /
PD-AF layer accounting that shares the same call chain.

## Controls — each defect fails for its own reason

A test that cannot fail proves nothing, so the suite was run against four
trees that each carry one specific defect.

Each tree carries the final `frontier/` and the final `tests/`, differing from
the delivered source by exactly one edit — except the baseline tree, whose
`frontier/` is the pre-fix parent in full.

| Tree | Defect | Unit result | Integration failure |
| --- | --- | --- | --- |
| `w3-control-baseline` | none — the pre-fix source at `3d47417` | 22 of 23 fail | `RuntimeError: Sequential simulation ended with non-empty scheduler state` |
| `w3-control-borrowed-timing` | every source continues on `live_batches[0]` | 12 of 23 fail | `ValueError: one attention-DP lane cannot occupy two open sync cohorts: replica=0, stage=0, lane=1, layer=1, sync_stage=pre_moe` |
| `w3-control-double-layer-advance` | the decode helper advances layers again | 2 of 23 fail | `ValueError: Decode post_moe layer counter cannot advance: request_id=1, completed_layer_count=4, total_layers=4` |
| `w3-control-double-owner-restore` | the per-source helper restores owners again | 16 of 23 fail | `ValueError: operation_id is already queued or active in this stage context: ('shared_layer', 0, 1, 0, 1, 'attention', 'FULL_STAGE_WORLD')` |

Two observations worth keeping:

- On the pre-fix tree the **same-phase** pairs still complete, so the suite is
  not failing wholesale for an unrelated reason. `decode-decode` and
  `prefill-prefill` pass every assertion about the forward itself — one shared
  identity, one collective event, two per-source continuations — and fail only
  at the last line, which inspects a shared room the pre-fix cluster does not
  have (`AttributeError: 'RoundRobinClusterScheduler' object has no attribute
  '_forward_sync_waiting_room'`). Every **mixed** pair instead fails earlier, at
  `assert len(collective) == 1` with an empty list. That gap between the two
  groups is the deadlock.
- The deadlock reproduces in the **real event loop**, not only in the fixture.
  With `attn_tp=1, attn_dp=2, moe_tp=1, moe_ep=2` on a monolithic MoE Replica,
  the pre-fix sequential run drains its event queue with a non-empty scheduler
  state.

## The integration fixture, and why it is not a wrapper case

The public MoE wrappers enforce `ATTN_TP == MOE_TP * MOE_EP` while the runtime
enforces `attn_tp * attn_dp == moe_tp * moe_ep`; those have no common solution
above one attention-DP lane, so no wrapper and no fidelity-matrix case can
reach a multi-lane monolithic MoE forward. `tests/integration/
test_monolithic_mixed_forward_runtime.py` therefore assembles the
configuration directly and runs the real `Simulator`:

- real admission, ownership restoration, synchronization and completion code;
- deterministic durations from constant profiling targets, plus an observer
  that *wraps* `predict_stage_execution_time` rather than replacing it, so
  source attribution is visible without changing any prediction;
- four requests of unequal prefill and decode length arriving together under
  chunked prefill, which is what puts one lane in prefill while the other
  decodes.

Asserted: every request completes, exactly once, with every token accounted
for (`request_metrics.csv` has one row per request and the token sum matches);
each live member of a mixed cohort appears in the prediction log slice that
belongs to that cohort's completion; no waiting-room leaf still holds a batch;
every `StageExecutionContext` is idle with no queued ticket; and the run is
identical with reporting on and off (same makespan, same cohort counts).

## Fidelity matrix

Both sides were run from **clean detached worktrees** so no untracked draft in
the development worktree could dirty the recorded provenance, and both were
driven by **one harness revision**, the candidate's.

```bash
PYTHONPATH=/data/ycfeng/Frontier/.worktrees/w3-candidate \
  python .worktrees/w3-candidate/tests/e2e/refactor_fidelity/run_matrix.py run \
  --repo-root .worktrees/w3-<side> --label <side> \
  --output-root /data/ycfeng/tmp/issue26-correctness-pr/w3-fidelity \
  --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 8 --clean-cache

python tests/e2e/refactor_fidelity/run_matrix.py compare \
  --output-root /data/ycfeng/tmp/issue26-correctness-pr/w3-fidelity \
  --baseline-label baseline --candidate-label candidate
```

| Side | `source_revision` | `source_dirty` | `git_dirty_paths` | `harness_revision` | Executed | Cache files |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | `3d474170a5df` (this branch's parent) | `False` | empty | `65ed8a76055e` | 71 of 71 | 426 |
| candidate | `65ed8a76055e` | `False` | empty | `65ed8a76055e` | 71 of 71 | 426 |

No case filter, cache cleaned before each run, 0 cases dropped as stale.

| Metric | Value |
| --- | --- |
| cases compared | 71 of 71 in the case table |
| identical | **71** |
| mismatched | 0 |
| baseline failures / candidate-only failures | 0 / 0 |
| cases missing from one side / with missing evidence / with differing definitions | 0 / 0 / 0 |
| provenance findings | 0 |
| predictor cache: baseline-only / candidate-only / findings | 0 / 0 / 0 |
| `complete_comparison`, `predictor_cache_populated_cleanly` | `True`, `True` |

Report: `/data/ycfeng/tmp/issue26-correctness-pr/w3-fidelity/comparison.json`.

### Judged against the expectation recorded before measuring

`design.md` predicted, before the run: all 71 cases stay exactly equal, because
at one attention-DP lane every cohort holds one source batch, so the shared
room, the shared identity, the single restoration and the per-source
continuation all reduce to today's single-source behavior. It further predicted
that the one behavior change reachable at one lane — the I7 decode-layer credit
for an already-decoding request carried inside a prefill-mode batch — changes a
counter no monolithic timing reads, and that **if** a MoE co-location case
moved, that credit would be the cause.

The measurement matches the primary prediction exactly: 71 identical, 0 moved.
The conditional branch was therefore not taken, which also confirms the source
reading behind it — on the monolithic path `completed_layer_count` feeds only
admission guards and diagnostics, and its two arithmetic consumers
(`cluster_batch_end_event.py:174` and `:354`) are PD-AF `DECODE_ATTN`/M2N.

A null result is the correct outcome here and is **not** evidence that the fix
works: no matrix case can reach a multi-lane monolithic MoE forward at all (see
below). The matrix answers "did anything else move", and the answer is no. The
evidence that the defect is fixed is checks 1 and 2 with their controls.

## Verification limits

- The multi-lane monolithic MoE shape has no released wrapper, so its evidence
  is the direct-construction integration test rather than a matrix case.
- The integration profiles are constant synthetic targets. They make timing
  deterministic and source attribution checkable; they are not trained
  numerical parity.
- I8, the per-source decode component ledger, is deliberately out of scope.
  See the scope table in `design.md`.
