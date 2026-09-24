# Test report — W4, opt-in vLLM-style DP request placement (2026-09-22)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created. Acceptance evidence for the opt-in `vllm_load_balancing` cluster scheduler, including five deliberate-defect controls. |

## Environment

| Item | Value |
| --- | --- |
| Worktree | `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr` |
| Branch / commit under test | `fix/issue26-correctness-pr` @ `10dd474` |
| Baseline | `cdfcdf5`, this commit's parent (the published W3 state) |
| Reference read | vLLM v0.10.2 at `/data/ycfeng/Frontier/.real-engine/vLLM-BS` @ `ea95f571` |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python` (3.10) |
| Invocation | `PYTHONPATH=$PWD python -m pytest … -p no:cacheprovider` from the worktree root |
| Absent optional deps | `torch`, `matplotlib` — the same 10 GPU/plot-only unit modules are excluded on both sides by one `--ignore` list |

Control trees live under the session scratchpad (`.../scratchpad/w4-control-*`)
and are driven by `run_w4_tree.py`, which strips the editable-install
meta-path finder, prepends the chosen tree and asserts that
`frontier.scheduler.utils.vllm_dp_load_balancer` resolved inside it. `frontier`
is a namespace package here, so without that step the development worktree
would win and a control would silently test the delivered source.

## What was measured

| # | Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | Balancer and policy behavior | `pytest tests/unit/test_vllm_dp_load_balancer.py` | all pass | 61 passed | PASS |
| 2 | Real-runtime wiring | `pytest tests/integration/test_vllm_dp_placement_runtime.py` | all pass, and the policy must actually diverge from round-robin | 3 passed; placements `[0,1,1,0,1,1]` vs round-robin's `[0,1,0,1,0,1]` | PASS |
| 3 | Focused regression set (46 files touching cluster scheduling, the decision log, and the two edited events) | see below | no failure outside the known baseline set | 51 failed / 1432 passed; all 51 are in the 84-failure baseline | PASS |
| 4 | Full unit suite vs baseline | `pytest tests/unit` | identical failure set | 84 failed on both, identical identities; 3778 vs 3717 passed (+61 new) | PASS |
| 5 | Integration suite vs baseline | `pytest tests/integration` | identical error set | 5 errors on both (PD-AF Reference checkout absent), 21 skipped on both; 15 vs 12 passed (+3 new) | PASS |
| 6 | Fidelity matrix | `run_matrix.py run/compare` | all 71 cases exactly equal (expectation recorded in `design.md` before measuring) | see the fidelity section | PASS |

### Check 3 — the focused regression set

Every `tests/unit` file mentioning `cluster_scheduler`, `ClusterScheduler`,
`decision_log`, `ClusterScheduleEvent`, `GlobalBatchEndEvent` or
`get_num_waiting_reqs`, minus the 10 GPU-only modules: 46 files, recorded in
`.../scratchpad/w4_regression_set_run.txt`.

One real defect surfaced here and was fixed rather than explained away. Running
the W4 unit file after a MoE configuration in the same process tripped the
process-global `IS_MOE` latch:

```
RuntimeError: IS_MOE already initialized to True, cannot change to False
```

The file builds both dense and MoE shapes, so it now resets the simulation
globals around each test with `global_vars.reset_global_vars()`, which is what
`tests/unit/test_config_owned_contracts.py` already does. Checks 1, 3, 4 and 5
were re-run afterwards; the numbers above are the post-fix ones.

## Check 2 — what the event loop showed

Each case runs the same configuration twice, under `vllm_load_balancing` and
under `round_robin`, in one child process per case; the only difference between
the two runs is the policy. Execution time comes from the dummy predictor, so
both policies see identical durations. Recorded evidence:
`.../scratchpad/w4-evidence/<case>_evidence.json`.

| Evidence | `moe_dp2` (offline) | `moe_dp2_online` | `dense_dp1` (offline) |
| --- | --- | --- | --- |
| lanes / requests / completed | 2 / 4 / 4 | 2 / 6 / 6 | 1 / 4 / 4 |
| cluster schedule times | `[0.0]` | `[0.0, 0.4, 0.6, 0.8, 1.0]` | `[0.0]` |
| routing times | identical to the above | identical to the above | identical to the above |
| placements, policy | `[0,1,0,1]` | **`[0,1,1,0,1,1]`** | `[0,0,0,0]` |
| placements, round-robin | `[0,1,0,1]` | **`[0,1,0,1,0,1]`** | `[0,0,0,0]` |
| report keys | `3,3,7,7,11,11,15,15,19,19` | `3,3,7,11,…,159` (44) | `0,1,2,3,4,5,6` |
| keys non-decreasing | yes | yes | yes |
| keys with a repeated lane | none | none | none |
| reports / after the lane released the batch | 10 / 10 | 44 / 44 | 7 / 7 |
| reports matching post-step / pre-step load | 10 / 6 | 44 / 38 | 7 / 3 |
| reports where the release moved the load | 4 | 6 | 4 |
| event types produced | `ReplicaScheduleEvent` | `ReplicaScheduleEvent` | `ReplicaScheduleEvent` |
| makespan (s) | 0.31 | 2.48 | 0.448 |

Four things in that table are load-bearing.

- **The routing time is the event's time.** Recorded routing times equal the
  recorded `ClusterScheduleEvent` times exactly, at five distinct instants in
  the online case. The policy's own `schedule()` raises, so a stale or retained
  time could not have been substituted silently.
- **The policy diverges from round-robin.** In the online case the two runs see
  the same arrivals, the same durations and the same lane capacity, and the
  policy places strictly fewer requests on lane 0 — the lane still draining the
  one 40-token request. Round-robin cannot do that; it alternates.
- **The reported load is the post-step one.** The lane's own `on_batch_end` is
  bracketed, so pre- and post-step load are distinct values for the batches
  that released work. Every report matches the post-step value, and strictly
  fewer match the pre-step value (6 of 10, 38 of 44, 3 of 7) — so the reports
  are not merely the pre-step state relabelled.
- **The report key ordering the guard assumes actually holds.** MoE keys are
  non-decreasing and every equal-key pair carries two distinct lanes, which is
  one shared forward observed twice. Keys advance by `num_layers = 4` per
  forward in the MoE cases and by 1 in the dense single-lane case, which is the
  measured identity recorded in `design.md`; only ordering and equality are
  used.
- **No event type is introduced and the run drains.** Both policies produce the
  same event-type set and both complete every request.

## Controls — each defect fails for its own reason

Each tree is the delivered source and the delivered tests with exactly one
edit; `diff -r frontier/` against the delivered source reports exactly one
differing file per tree.

| Control | The one edit | Tests that fail | Why that is the right failure |
| --- | --- | --- | --- |
| `baseline` | none | 0 of 64 | the harness itself passes in a control tree |
| `no-time-plumbing` | `ClusterScheduleEvent` calls `schedule()` again | 3 integration | the policy refuses to route without a time, so the whole real-runtime surface fails rather than routing from a stale snapshot |
| `pre-step-report` | the hook moves above `replica_scheduler.on_batch_end` | 2 integration | `reports_after_the_lane_released_the_batch` drops to `0 == 10`, and the online placements collapse onto round-robin's |
| `unweighted-waiting` | `WAITING_SCORE_WEIGHT = 1` | 2 unit | the weight and its exact boundary are the two tests that pin `waiting * 4 + running` |
| `no-local-reservation` | `select` stops reserving the chosen lane | 4 unit + 2 integration | without the reservation every request in one routing burst lands on the same lane; the snapshot-replacement and latch tests also lose their discriminating state |
| `no-dense-lane-guard` | the `is_moe or attn_dp == 1` rejection is deleted | 1 unit | only the `dense_multi_lane` construction case, which is exactly the shape whose report-key ordering was measured to interleave |

Controls were rebuilt against the final test files after the `IS_MOE` fixture
was added and re-run; the counts above are from that run
(`.../scratchpad/w4_ctrl_*.log`).

## Fidelity matrix

Both sides ran from **clean detached worktrees** so no untracked draft could
dirty the recorded provenance, and both were driven by **one harness
revision**, the candidate's.

```bash
PYTHONPATH=/data/ycfeng/Frontier/.worktrees/w4-candidate \
  python /data/ycfeng/Frontier/.worktrees/w4-candidate/tests/e2e/refactor_fidelity/run_matrix.py run \
  --repo-root /data/ycfeng/Frontier/.worktrees/w4-<side> --label <side> \
  --output-root /data/ycfeng/tmp/issue26-correctness-pr/w4-fidelity \
  --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 8 --clean-cache

python tests/e2e/refactor_fidelity/run_matrix.py compare \
  --output-root /data/ycfeng/tmp/issue26-correctness-pr/w4-fidelity \
  --baseline-label baseline --candidate-label candidate
```

| Side | `source_revision` | `source_dirty` | `git_dirty_paths` | `harness_revision` | Executed | Cache files |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | `cdfcdf54b545` (this commit's parent) | `False` | empty | `10dd4745b9a7` | 71 of 71 | 426 |
| candidate | `10dd4745b9a7` | `False` | empty | `10dd4745b9a7` | 71 of 71 | 426 |

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

Report: `/data/ycfeng/tmp/issue26-correctness-pr/w4-fidelity/comparison.json`.

### Judged against the expectation recorded before measuring

`design.md` recorded, before the matrix ran:

> **Prediction: all 71 fidelity cases stay exactly equal.** … W4 has no
> reachable fidelity fix, so a single mismatch falsifies the change rather than
> confirming it.

71 of 71 identical, so the prediction held. What that does and does not
establish:

- It **does** establish that nothing in the 71 cases moved. All three of the
  reachable edits are inert on those paths: `schedule_at` defaults to
  `schedule()`, `on_replica_batch_end` returns `None` on every policy the
  matrix selects, and `get_request_load()` recomputes the decision-log payload
  from the same two accessors it replaced.
- It **does not** establish anything about the new policy. No matrix case
  selects `vllm_load_balancing`; the matrix cases all use the default
  `round_robin`. The policy's own evidence is checks 1 and 2 and the controls.

## Verification limits

- **No vLLM equivalence is claimed or measured.** The selection rule, the tie
  break, the reservation, the publication intervals and the collection wait are
  each cited against vLLM v0.10.2 source, and the unit tests pin them against
  values derived from that source by hand. Nothing here was compared against a
  running vLLM deployment, and the object deliberately omits IPC latency, more
  than one frontend, elastic scaling and the coordinator's warm-start phase.
- **The report key is not a vLLM step counter.** It is
  `ForwardSyncState.get_step_id(batch)`, which advances once per layer, so
  consecutive forwards are roughly `num_layers` apart. Only its ordering and
  equality are used, which is all the reference coordinator uses its
  `(wave, step)` pair for. Frontier has no wave reset, so the pair collapses to
  one scalar.
- **The dense restriction is measured at one shape, not proved in general.**
  The `attn_dp=1` requirement for dense models comes from a probe showing dense
  dp2 keys interleaving under staggered online arrivals while MoE dp2 keys stay
  ordered. That is one measurement per shape, recorded in `design.md`, not an
  exhaustive argument; the constructor rejects the shape rather than relying on
  the observation holding everywhere.
- **Integration timing is dummy-mode timing.** The three real-runtime cases use
  the dummy predictor, so the placements they produce are not latency-realistic.
  They are still load-sensitive and deterministic, and both policies in a case
  see identical durations, which is what those checks depend on.
- **The online divergence is one scenario.** `[0,1,1,0,1,1]` versus
  `[0,1,0,1,0,1]` shows the policy reads published load; it does not
  characterise the policy's behavior across arrival patterns, and the test
  asserts the direction (fewer requests on the busy lane) rather than the exact
  sequence.
- **Five controls, not a proof.** Each control shows that one specific piece of
  the change is load-bearing for a specific test. They do not exclude a defect
  that no control models.
