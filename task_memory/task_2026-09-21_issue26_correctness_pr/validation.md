# Issue 26 Correctness PR — Validation Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created. Environment recorded; baseline results recorded in the refactor task's Step 0 report because both branches share the same base commit. |
| 2026-09-21 | Step 1 recorded: audit spot checks and the vLLM reference identity check. |
| 2026-09-21 | Step 2 recorded: unit sensitivity and the fidelity measurement against a stated expectation. |
| 2026-09-22 | Step 2 re-measured with harness and source at one revision, after the gate corrections. Same expectation, same result, recorded provenance. |
| 2026-09-22 | Step 3 recorded: the shared monolithic forward, its direct-construction runtime evidence, four deliberate-defect controls, and a 71-of-71 identical fidelity matrix. |
| 2026-09-22 | Step 4 recorded: the opt-in vLLM-style DP placement policy, its real-runtime wiring evidence including a divergence from round-robin, five deliberate-defect controls, and a 71-of-71 identical fidelity matrix. |
| 2026-09-22 | Step 6 native parity test recorded and submitted as `exp-0922-140423-075005`; artifact identity closed as document-only. |

## Environment

| Field | Value |
| --- | --- |
| Host | `kun-workspace-vgen2` (CPU master) |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python` (uv-managed CPython 3.10.6) |
| Install | `uv pip install -e ".[test]"` from the active worktree |
| Scratch root | `FRONTIER_TMP_ROOT=/data/ycfeng/tmp/issue26-correctness-pr` |
| vLLM reference | `.real-engine/vLLM-BS` at `ea95f57` |

## Per-checkpoint evidence

| Step | Purpose | Source under test | Command | Outcome | Baseline comparison | Limits |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Baseline for touched areas | `1f694f7` | See `task_memory/task_2026-09-21_oversized_module_split/test_report_2026-09-21_step0_baseline.md` | 84 passed / 10 failed (pre-existing or environmental); co-location and PDD dense dummy smokes PASS | Base itself | Shared with the refactor branch (same base). |
| 1 | Source audit only; no code executed | `1f694f7` vs `a7b3320`; vLLM `ea95f57` | `git diff` / `git show` at the three pinned revisions; `git fetch` of upstream `v0.10.2` into the reference checkout | Recorded in `review.md`, `audit_scheduler.md`, `audit_predictor_profiling.md`, `reference_vllm_0_10_2.md` | n/a | An audit establishes intent and current behavior from source. It does not establish runtime behavior; every disposition still needs its own test in the step that implements it. |

## Step 1 spot checks

Five consequential audit claims were re-verified directly against the source before being recorded as project facts.

| # | Claim | Verification | Result |
| --- | --- | --- | --- |
| 1 | Main's `_schedule_batch_mode` restarts DP-lane assignment at zero on every call | Read `round_robin_cluster_scheduler.py:365-389`: the persistent `_request_counter` drives `replica_idx`, while `dp_id = local_idx % self._replica_dp_size` uses the per-call index | CONFIRMED |
| 2 | Main already implements the intended formula for the DECODE role only | Read `:432-444` (`dp_id = (counter + idx) // num_replicas % dp_size`) and the dispatch at `:74-86` (`ClusterType.DECODE` only) | CONFIRMED |
| 3 | The candidate removes a method the SGLang scheduler still calls | `git grep _get_num_waiting_reqs_for_decision_log` at `a7b3320` returns only the caller at `sglang_style_replica_scheduler.py:65`; at `1f694f7` it returns the caller plus the definition at `vllm_v1_engine_replica_scheduler.py:2136` | CONFIRMED |
| 4 | `ffn_signature` on main carries no routing-runtime term | Read `shared_prediction_model_manager.py:1377-1382`: device, model, TP, is-MoE, architecture profile, typed contract hash, measurement family only | CONFIRMED |
| 5 | The vLLM selection weight and the coordinator intervals | Read `core_client.py:1146` (`score = waiting * 4 + running`), `coordinator.py:116` (`min_stats_update_interval_ms: int = 100`), `:198` (`5000`), `:202` (`50`) | CONFIRMED, with the correction that the weight lives in the frontend, not the coordinator |

## Limits of Step 1

The audit reads source at pinned revisions. It does not run the candidate, does not measure timing, and makes no accuracy claim. The reference checkout was fetched read-only; an `upstream` remote and the tag `upstream-v0.10.2` now exist in `.real-engine/vLLM-BS`, which is outside both PR branches.

## Step 2 — round-robin DP rotation

### What the fidelity matrix can and cannot show here

Unlike the module splits, this is a behavior fix, so the matrix is expected to report mismatches. A result of zero mismatches would mean the fix did not reach the path. The gate is therefore an expectation stated **before** measuring, not a comparison against zero.

The expectation was derived from the arithmetic: cases with `_replica_dp_size == 1` cannot change, because the old lane expression was `local_idx % 1` and the new one is `(ordinal // num_replicas) % 1`, both always zero. Within a single scheduling call starting at counter zero the two expressions also agree, so a case that admits its whole stream in one call cannot move either.

Four cases were added to the matrix specifically to sit in the defective regime, and their regime was verified by measuring lane occupancy on the base commit rather than by reasoning.

| Case | Base-commit lane occupancy | Expected |
| --- | --- | --- |
| `dp_dense_online_lanes2` | all 45 records on lane 0 | must move |
| `dp_dense_online_lanes4` | all 45 records on lane 0 | must move |
| `dp_dense_online_lanes2_replicas2` | replicas split 28/28 while every record is lane 0 | must move |
| `dp_dense_offline_lanes2_replicas2` | lanes split 70/70 | must not move |
| `coloc_dense_offline_attn_dp2` | lanes split 39/39 | must not move |

`dp_dense_online_lanes2_replicas2` is the sharpest single case: both rotations are visible in one run, and only the replica index was rotating.

### Result

| Field | Record |
| --- | --- |
| Source under test | `6ab521d`, measured from a detached checkout |
| Cases compared | 71 of 71 |
| Identical | 68 |
| Mismatched | exactly `dp_dense_online_lanes2`, `dp_dense_online_lanes4`, `dp_dense_online_lanes2_replicas2` |
| Expected to move but did not | none |
| Moved but was not expected to | none |
| Predictor cache names | no differences |
| Result | **PASS against the stated expectation** |

Direction, from the stage ledger's `replica_local_id`:

| Case | Baseline | Candidate |
| --- | --- | --- |
| `dp_dense_online_lanes2` | `{0: 45}` | `{0: 44, 1: 45}` |
| `dp_dense_online_lanes4` | `{0: 45}` | `{0: 43, 1: 43, 2: 44, 3: 45}` |
| `dp_dense_online_lanes2_replicas2` | `{0: 56}` | `{0: 55, 1: 56}` |
| `dp_dense_offline_lanes2_replicas2` (control) | `{0: 70, 1: 70}` | unchanged |

Collapsed onto lane zero before, evenly spread after, and the control did not budge.

### Unit evidence

Comparison against the refactor tip `db15e64` over 73 files: identical failure identities, 1808 to 1812 passing, the four new tests being the difference.

Sensitivity was verified by stashing the fix and rerunning, not by reasoning. Three of the four new tests fail on the pre-fix code for the right reason, alternating lanes collapsing to lane 0. The fourth, which asserts the per-replica grouping of the returned mapping, passes on both, which is what confirms the ordering was preserved.

### Limits

The prefill role reaches the same placement path, but **no shipped recipe can give it more than one lane**, so the matrix cannot cover that half. A dense model in a disaggregated architecture is rejected outright, and the MoE wrappers enforce `ATTN_TP == MOE_TP * MOE_EP` while the runtime enforces `attn_tp * attn_dp == moe_tp * moe_ep`, which have no common solution above one lane. Both routes were attempted and both were rejected, so this is measured rather than inferred. The unit test is the only evidence for the prefill half of this fix, and the PR says so.

## Step 2 re-measured — harness and source at one revision (2026-09-22)

### Why it was repeated

The original W2 measurement ran the `6ab521d` source against the refactor tip's
case table and comparator, and it was taken with the pre-correction harness.
Two things were wrong with that as a record, neither of them a defect in the
result: the harness revision was disclosed in prose but not recorded as a field,
and the harness itself could report success without comparing anything
(review comments R34-01, R34-02). Repeating it is cheaper than arguing about it.

### Setup

Both sides were driven by one harness, running from the candidate checkout, with
each side's production tree supplied as `--repo-root` from its own clean
detached checkout.

```bash
# from .worktrees/w2-candidate-ceac2b4, PYTHONPATH=$PWD
OUT=/data/ycfeng/tmp/issue26-correctness-pr/w2-remeasure
python tests/e2e/refactor_fidelity/run_matrix.py run \
  --repo-root .worktrees/w2-baseline-6ef0a3c  --label w2_baseline_6ef0a3c \
  --output-root "$OUT" --jobs 6 --clean-cache --continue-on-failure
python tests/e2e/refactor_fidelity/run_matrix.py run \
  --repo-root .worktrees/w2-candidate-ceac2b4 --label w2_candidate_ceac2b4 \
  --output-root "$OUT" --jobs 6 --clean-cache --continue-on-failure
python tests/e2e/refactor_fidelity/run_matrix.py compare --output-root "$OUT" \
  --baseline-label w2_baseline_6ef0a3c --candidate-label w2_candidate_ceac2b4
```

| Field | Baseline | Candidate |
| --- | --- | --- |
| Source revision | `6ef0a3c` (refactor tip, no W2) | `ceac2b4` (this branch, with W2) |
| Source working tree | clean | clean |
| Harness revision | `ceac2b4` | `ceac2b4` |
| Case filter | none | none |
| Cases executed | 71 | 71 |
| `case_count` / result lines | 71 / 71 | 71 / 71 |
| Cache cleaned first | yes | yes |
| Cache files produced | 426 | 426 |

### The expectation, unchanged from the first measurement

Exactly three cases move: `dp_dense_online_lanes2`, `dp_dense_online_lanes4`,
`dp_dense_online_lanes2_replicas2`. The two offline DP cases and the remaining
66 do not. A result of zero mismatches would mean the fix never reached the
path; a mismatch anywhere else would mean it reached more than the path.

### Result

| Measure | Expected | Actual | Result |
| --- | --- | --- | --- |
| Cases compared | 71 | **71** | PASS |
| Identical | 68 | **68** | PASS |
| Mismatched | the 3 named above | **exactly those 3** | PASS |
| Expected to move but did not | none | **none** | PASS |
| Moved but was not expected to | none | **none** | PASS |
| Baseline failures | 0 | **0** | PASS |
| Candidate-only failures | 0 | **0** | PASS |
| Missing / missing evidence / differing definitions | 0 | **0 / 0 / 0** | PASS |
| Cases not compared | 0 | **0** | PASS |
| Provenance findings | none | **none** | PASS |
| Predictor cache differences | 0 | **0**, compared cleanly | PASS |

Comparison exit code 1, which is correct here: the gate reports inequality, and
the acceptance criterion is the stated expectation, not exit 0.

### Direction, from the stage ledger's `replica_local_id`

| Case | Baseline | Candidate | |
| --- | --- | --- | --- |
| `dp_dense_online_lanes2` | `{0: 45}` | `{0: 44, 1: 45}` | moved |
| `dp_dense_online_lanes4` | `{0: 45}` | `{0: 43, 1: 43, 2: 44, 3: 45}` | moved |
| `dp_dense_online_lanes2_replicas2` | `{0: 56}` | `{0: 55, 1: 56}` | moved |
| `dp_dense_offline_lanes2_replicas2` | `{0: 70, 1: 70}` | `{0: 70, 1: 70}` | control, unchanged |
| `coloc_dense_offline_attn_dp2` | `{0: 39, 1: 39}` | `{0: 39, 1: 39}` | control, unchanged |

Every number reproduces the first measurement exactly. The re-measurement
changed the provenance of the evidence, not the evidence.

### Relationship between the measured commit and the branch tip

The candidate measured is `ceac2b4`. Commits after it on this branch touch only
`tests/` and `task_memory/`; `git diff --stat ceac2b4..HEAD -- frontier/` is
empty, so the production tree that produced these artifacts is the branch tip's
production tree. This is the "prove the relationship" path the review prefers to
a rerun, and here it is genuinely available because no production file changed.

### Unit evidence for the strengthened tests

`tests/unit/test_cluster_scheduler_dp_lanes.py`, 23 tests, all passing. Against
the pre-fix `_schedule_batch_mode` taken verbatim from `6ab521d^` and installed
in memory, 12 of 23 fail; the grouping control and the eight unrelated lane
tests still pass. What the failures show is recorded in `review.md`: the old
code produces the expected sequence exactly for a single burst and collapses to
lane 0 only for incremental arrival, so the fix restored a rotation that already
existed rather than introducing one.

Related modules, whole files: 51 failed / 1356 passed / 19 skipped, and all 51
failures are `test_pdaf_parity_reference_observer_bootstrap.py`, which needs the
pinned PD-AF reference checkout that is absent on this host. That count matches
the inherited-failure inventory recorded for the refactor branch.

### Limits

- The matrix still validates only the monolithic half of W2. The prefill role is
  now covered at the scheduler level, through the public `schedule()`, for both
  roles that reach batch-mode placement; that is a scheduler test, not a run of
  the event loop. Full runtime coverage needs the direct-construction fixture
  described under R35-02, which is W3 acceptance work.
- Lane occupancy is read from the stage ledger, which records scheduled stage
  executions. It shows where work was placed, not that placement is optimal.

## Step 3 — one shared monolithic forward (2026-09-22)

### What the fidelity matrix can and cannot show here

The defect needs at least two attention-DP lanes on a monolithic MoE Replica.
The public MoE wrappers enforce `ATTN_TP == MOE_TP * MOE_EP` while the runtime
enforces `attn_tp * attn_dp == moe_tp * moe_ep`; those have no common solution
above one lane. **No case in the 71-case table can reach the defect**, so the
matrix here answers only one question — did anything *else* move — and a null
result is the pass condition, not the proof of the fix.

The proof of the fix is the direct-construction runtime test plus its controls.

### Setup

Both sides ran from clean detached worktrees, driven by one harness revision,
so nothing in the development worktree could dirty the recorded provenance.

```bash
# harness = the candidate checkout, for both sides
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

### The expectation, recorded before measuring

From `design.md`, written before the run:

- All 71 cases stay exactly equal. At one attention-DP lane every cohort holds
  one source batch, so the shared room, the shared identity, the single
  ownership restoration and the per-source continuation all reduce to today's
  single-source behavior; `sample_batch` *is* the only batch.
- The one behavior change reachable at one lane is the I7 decode-layer credit
  for an already-decoding request carried inside a prefill-mode batch. It moves
  a counter that no monolithic timing reads.
- If a MoE co-location case does move, the I7 credit is the cause, and it is the
  approved fidelity fix that plan section 9 requires.

### Result

| Metric | Value |
| --- | --- |
| cases compared | 71 of 71 in the case table |
| identical | **71** |
| mismatched | 0 |
| baseline failures / candidate-only failures | 0 / 0 |
| missing from one side / missing evidence / differing definitions | 0 / 0 / 0 |
| provenance findings | 0 |
| predictor cache: baseline-only / candidate-only / findings | 0 / 0 / 0 |
| `complete_comparison`, `predictor_cache_populated_cleanly` | `True`, `True` |

PASS against the stated expectation: the primary prediction held exactly and the
conditional branch was not taken. That also confirms the source reading behind
it — on the monolithic path `completed_layer_count` feeds only admission guards
and diagnostics, and its two arithmetic consumers
(`cluster_batch_end_event.py:174` and `:354`) are PD-AF `DECODE_ATTN`/M2N.

### Runtime evidence, which the matrix cannot supply

`tests/integration/test_monolithic_mixed_forward_runtime.py` builds
`attn_tp=1, attn_dp=2, moe_tp=1, moe_ep=2` on a monolithic MoE Replica directly
and runs the real `Simulator`: real admission, ownership restoration,
synchronization and completion, with deterministic durations injected only at
the predictor boundary and an observer that *wraps* `predict_stage_execution_time`
rather than replacing it. Four requests of unequal length arrive together under
chunked prefill, which is what puts one lane in prefill while another decodes.

Observed: 24 cohorts, 4 of them mixed-phase, 4 of 4 requests completed, every
token accounted for, no waiting-room leaf left holding a batch, every
`StageExecutionContext` idle with no queued ticket, and an identical run with
metrics reporting on and off. On the pre-fix source the same configuration ends
with `RuntimeError: Sequential simulation ended with non-empty scheduler state`.

### Controls — four trees, four distinct failures

Each control tree carries the final `frontier/` and the final `tests/`,
differing from the delivered source by exactly one edit; the baseline tree's
`frontier/` is the pre-fix parent in full. They are driven by
`run_against_tree.py`, which strips the editable-install meta-path finder and
asserts the resolved source path, because an earlier attempt to select a tree
with `PYTHONPATH` alone silently kept importing the installed package.

| Tree | Defect | Unit | Runtime failure |
| --- | --- | --- | --- |
| baseline | none — pre-fix `3d47417` | 22 of 23 fail | `RuntimeError: Sequential simulation ended with non-empty scheduler state` |
| borrowed timing | every source continues on `live_batches[0]` | 12 of 23 fail | `ValueError: one attention-DP lane cannot occupy two open sync cohorts: replica=0, stage=0, lane=1, layer=1, sync_stage=pre_moe` |
| double layer advance | the decode helper advances layers again | 2 of 23 fail | `ValueError: Decode post_moe layer counter cannot advance: request_id=1, completed_layer_count=4, total_layers=4` |
| double owner restore | the per-source helper restores owners again | 16 of 23 fail | `ValueError: operation_id is already queued or active in this stage context: ('shared_layer', 0, 1, 0, 1, 'attention', 'FULL_STAGE_WORLD')` |

On the pre-fix tree the same-phase pairs pass every assertion about the forward
itself and fail only at the last line, which inspects a shared room the pre-fix
cluster does not have. The mixed pairs fail earlier, at an empty collective
list. That gap between the two groups is the deadlock, isolated.

### Regression comparison

| Suite | Baseline `3d47417` | Candidate `65ed8a7` | Verdict |
| --- | --- | --- | --- |
| `tests/unit` | 84 failed / 3694 passed | 84 failed / 3717 passed | identical failure identities; +23 are the new tests |
| `tests/integration` | 5 errors / 11 passed | 5 errors / 12 passed | identical errors (PD-AF Reference checkout absent on this host); +1 is the new test |
| forward-sync set, 17 files | — | 342 passed | no regression in the shared call chain |

### Limits

- The multi-lane monolithic MoE shape has no released wrapper, so its evidence
  is a direct-construction test rather than a matrix case. Anything that only a
  released recipe would exercise is therefore still unmeasured for this shape.
- The integration profiles are constant synthetic targets: deterministic and
  source-attributable, not trained numerical parity.
- I8, the per-source decode component ledger, is deliberately out of scope; see
  the scope table in `design.md`.

## Step 4 — opt-in vLLM-style DP request placement (2026-09-22)

### What the fidelity matrix can and cannot show here

The policy is opt-in and no case in the 71-case table selects it; every case
uses the default `round_robin`. **The matrix therefore answers one question
only — did anything else move — and a null result is the pass condition, not
evidence about the policy.** The policy's own evidence is the balancer unit
suite, the real-runtime integration cases, and five deliberate-defect controls.

Three edits are reachable from the existing matrix paths, and all three are
inert there: `BaseClusterScheduler.schedule_at` defaults to `schedule()`,
`on_replica_batch_end` returns `None` on every policy the matrix selects, and
`get_request_load()` rebuilds the decision-log payload from the same two
accessors it replaced.

### Setup

Both sides ran from clean detached worktrees, driven by one harness revision.

```bash
# harness = the candidate checkout, for both sides
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

### The expectation, recorded in `design.md` before measuring

> **Prediction: all 71 fidelity cases stay exactly equal.** … W4 has no
> reachable fidelity fix, so a single mismatch falsifies the change rather than
> confirming it.

| Metric | Value |
| --- | --- |
| cases compared / identical / mismatched | 71 of 71 / **71** / 0 |
| baseline failures / candidate-only failures | 0 / 0 |
| missing from one side / missing evidence / differing definitions | 0 / 0 / 0 |
| provenance findings / predictor-cache findings | 0 / 0 |
| `complete_comparison`, `predictor_cache_populated_cleanly` | `True`, `True` |

Prediction held. Report:
`/data/ycfeng/tmp/issue26-correctness-pr/w4-fidelity/comparison.json`.

### Runtime evidence for the policy itself

Each case runs one configuration twice — under `vllm_load_balancing` and under
`round_robin` — in a single child process, with the dummy predictor so both
policies see identical durations.

| Evidence | `moe_dp2` | `moe_dp2_online` | `dense_dp1` |
| --- | --- | --- | --- |
| routing times == cluster schedule times | `[0.0]` | `[0.0, 0.4, 0.6, 0.8, 1.0]` | `[0.0]` |
| placements, policy vs round-robin | `[0,1,0,1]` / `[0,1,0,1]` | **`[0,1,1,0,1,1]` / `[0,1,0,1,0,1]`** | `[0,0,0,0]` / `[0,0,0,0]` |
| report keys | `3,3,7,7,…,19,19` | `3,3,7,…,159` (44) | `0,1,2,3,4,5,6` |
| ordered / no repeated lane per key | yes / yes | yes / yes | yes / yes |
| reports after the lane's release | 10 of 10 | 44 of 44 | 7 of 7 |
| matching post-step / pre-step load | 10 / 6 | 44 / 38 | 7 / 3 |
| event types vs round-robin | equal | equal | equal |

The online row is the discriminating one: identical arrivals, durations and
lane capacity, and the policy still places strictly fewer requests on the lane
draining the one long request. Round-robin cannot, because it cannot see load.

### Controls

Each tree is the delivered source and tests with exactly one edit.

| Control | Fails | Where |
| --- | --- | --- |
| `baseline` | 0 of 64 | — |
| `no-time-plumbing` (`schedule()` restored in the event) | 3 | integration |
| `pre-step-report` (hook moved above `on_batch_end`) | 2 | integration; `reports_after_the_lane_released_the_batch` is `0 == 10` |
| `unweighted-waiting` (`WAITING_SCORE_WEIGHT = 1`) | 2 | unit |
| `no-local-reservation` (`select` stops reserving) | 6 | 4 unit + 2 integration |
| `no-dense-lane-guard` (guard deleted) | 1 | unit, the `dense_multi_lane` case |

### Regression comparison

| Suite | Baseline `cdfcdf5` | Candidate `10dd474` | Verdict |
| --- | --- | --- | --- |
| `tests/unit` | 84 failed / 3717 passed | 84 failed / 3778 passed | identical failure identities; +61 are the new tests |
| `tests/integration` | 5 errors / 12 passed | 5 errors / 15 passed | identical errors (PD-AF Reference checkout absent on this host); +3 are the new tests |
| focused set, 46 files | — | 51 failed / 1432 passed | all 51 are in the known 84-failure baseline |

### Limits

- No vLLM equivalence is claimed or measured. Every constant is cited against
  vLLM v0.10.2 source; nothing was compared against a running deployment. IPC
  latency, multiple frontends, elastic scaling and the coordinator's warm-start
  phase are deliberately absent.
- The dense `attn_dp=1` restriction rests on one probe per shape (recorded in
  `design.md`), which is why the constructor rejects the shape instead of the
  code relying on the observation holding everywhere.
- Integration placements are dummy-mode placements: deterministic and
  load-sensitive, not latency-realistic.
- The online divergence is one arrival pattern. The test asserts the direction
  — fewer requests on the busy lane — not the exact sequence.

## Step 6 — legacy fused-MoE expert arithmetic (2026-09-22)

Full record: `test_report_2026-09-22_w6_fused_expert_arithmetic.md`.

| Item | Result |
| --- | --- |
| Reachability | Confirmed. The repaired path is selected when vLLM exposes the low-level API, which the pinned reference v0.10.2 and the `environment_profiling.yml` pin `vllm>=0.10,<0.11` both do. Neither Torch environment on this host reproduces it (vLLM 0.11.0 and 0.28.0 both select the functional path). |
| Magnitude, stated before implementing | Estimated 16.5% of the corrected `moe_grouped_gemm` time at 4096 tokens on `a800/qwen3-a3b-30b-moe`; 6.8% median, 26.3% max over its rows; 1.1-1.4% on the two 64-token h800 datasets. Analytical, at 80% of peak HBM. |
| CPU tests | `tests/unit/test_moe_fused_expert_arithmetic.py`, 7 new tests, all passing under `/data/ycfeng/envs/openmopd-py312/bin/python` (Torch 2.8.0, vLLM 0.11.0). Composition only: the native calls are replaced by plain-Torch references, and the file says so. |
| Discriminating check | `test_a_gated_activation_is_not_the_first_half_of_the_projection` computes the old slice-only arithmetic and asserts the repaired result differs, so the reference-equality test cannot pass against the unrepaired path. |
| Regression, Torch environment | Against a detached worktree at `HEAD` (`bbbfcaa`), same four existing files: 1 failed / 136 passed at HEAD, 1 failed / 143 passed with the repair. Same failure identity on both sides — `test_functional_vllm_kernel_exposes_mxfp4_switch_without_importing_vllm` needs Torch without vLLM, which no local environment provides. |
| Regression, default environment | `pytest tests/unit --continue-on-collection-errors` under `frontier-py310`: 84 failed, 3778 passed, 49 skipped, 11 errors — the W4 baseline of 84 failures and 3778 passing, unchanged. Collection errors 10 -> 11 because the new file imports Torch at module level, as the seven existing profiling test files in that list already do; verified by re-collecting with it ignored. |
| Fidelity matrix | Not run, deliberately. The change is confined to a module the simulator cannot import (it requires Torch, absent from the simulator environment), and the matrix consumes checked-in CSVs rather than fresh profiling. The unchanged default-environment suite is the evidence. The repair changes what a future profiling run measures, not any simulation from existing data. |
| Native GPU parity | **SUBMITTED, result pending.** `tests/integration/test_moe_fused_expert_numerical_parity.py`, 8 cases at `rtol=0, atol=0` against `fused_experts`: Qwen3-A3B-30B shapes from the checked-in model config at 4096 and 4097 tokens on EP ranks 0 and 1; a 257-token, 16-expert case at top-k 2 and 4 whose routing leaves two local experts empty; repeated invocation with different inputs; and the FP8 path as a structural check. Job `exp-0922-140423-075005`, creator `i-fengyicheng`, `steptron_ci` / `H800`, 1 GPU, image `artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2`, NFS source `100.96.128.195:/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`. |
| Artifact identity | **CLOSED as document-only** by user decision. The finding stands: `resolve_grouped_gemm_backend` labels both vLLM paths `vllm_fused`, and `profiling_patch_tag` holds three historical free-text values in `a800/qwen3-a3b-30b-moe/moe.csv` while nothing in the source writes it. No column was added. `docs/profiling/README.md` records the operator's scope, the size of the pre-repair gap, and that a row cannot be checked for completeness from its own metadata. |
| Suite re-run after the documentation and record commits | `pytest tests/unit -q --continue-on-collection-errors` under `frontier-py310` at `cad3afd`: 84 failed, 3778 passed, 49 skipped, 11 errors. Identical to the W4 baseline and to the earlier W6 measurement. |
