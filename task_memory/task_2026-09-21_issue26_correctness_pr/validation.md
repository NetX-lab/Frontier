# Issue 26 Correctness PR — Validation Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | W9-05 fix `75c1140` validated: B1–B8 pass (regression test and its control, `tight_kv` 24 of 24 in three shapes, KV sweep 72 of 72, C2 21 of 24 identical with the 3 preempting cases explained, deadlock sweep 72 of 72, fidelity 71 of 71, stage-admission 51 of 51, suites 0 regressions, examples 16 of 16). |
| 2026-09-23 | W9-04 fix `2ffb062` validated: A1–A7 pass (regression test and its control, sweep 72 of 72, C2 22 of 22 plus both stalled cases finishing, fidelity 71 of 71, stage-admission 51 of 51, suites 0 regressions, examples 16 of 16). |
| 2026-09-23 | Step 9 completed on CPU: P1(b) on seven shapes after the W9-01 merge-forward, P2/P3 unit tests, P4 real event loop, P5 unchanged behavior (C2 24 of 24, fidelity 71 of 71, examples 16 of 16, 0 suite regressions), and the two pre-existing defects W9-04 and W9-05 found during P5. |
| 2026-09-22 | Step 9 partial validation added: reference-loop oracle, Frontier boundary probes on three shapes, the two probe failures (invariant I5, W9-01), and the ground-truth writer checks. |
| 2026-09-21 | Created. Environment recorded; baseline results recorded in the refactor task's Step 0 report because both branches share the same base commit. |
| 2026-09-21 | Step 1 recorded: audit spot checks and the vLLM reference identity check. |
| 2026-09-21 | Step 2 recorded: unit sensitivity and the fidelity measurement against a stated expectation. |
| 2026-09-22 | Step 2 re-measured with harness and source at one revision, after the gate corrections. Same expectation, same result, recorded provenance. |
| 2026-09-22 | Step 3 recorded: the shared monolithic forward, its direct-construction runtime evidence, four deliberate-defect controls, and a 71-of-71 identical fidelity matrix. |
| 2026-09-22 | Step 4 recorded: the opt-in vLLM-style DP placement policy, its real-runtime wiring evidence including a divergence from round-robin, five deliberate-defect controls, and a 71-of-71 identical fidelity matrix. |
| 2026-09-22 | Step 6 native parity test recorded and submitted as `exp-0922-140423-075005`; artifact identity closed as document-only. |
| 2026-09-22 | Step 6 native GPU parity recorded: 8 of 8 at `rtol=0, atol=0` on H800 under `codesign`. |
| 2026-09-22 | Step 7 recorded: the companion backend fix, both negative controls, clean-checkout validation, and the governance-scan repair the gitlink bump exposed. |
| 2026-09-22 | Step 8 §14.1 recorded: combined unit/integration suites, 16 architecture examples, four PP=2 cases, and the cold-then-warm predictor-cache pair. |

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
| Native GPU parity | **PASS. 8 passed in 13.70 s** on `NVIDIA H800` (`gpu-h800-0110`), job `exp-0922-145047-660565`, creator `i-fengyicheng`, charged group `codesign` per the user's 2026-09-22 instruction, 1 GPU, image `artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2`, NFS source `100.96.128.195:/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`. Worker: Python 3.12.11, torch 2.8.0+cu128, vLLM 0.10.2, `VLLM_API_VERSION=0.10.x`, `FP8_AVAILABLE=True`. All 8 cases of `tests/integration/test_moe_fused_expert_numerical_parity.py` compare at `rtol=0, atol=0` against `fused_experts`: Qwen3-A3B-30B shapes from the checked-in model config at 4096 and 4097 tokens on EP ranks 0 and 1; a 257-token, 16-expert case at top-k 2 and 4 whose routing leaves two local experts empty; repeated invocation with different inputs; and the FP8 path as a structural check. Three earlier attempts and their causes are in `test_report_2026-09-22_w6_fused_expert_arithmetic.md` section 8. |
| Artifact identity | **CLOSED as document-only** by user decision. The finding stands: `resolve_grouped_gemm_backend` labels both vLLM paths `vllm_fused`, and `profiling_patch_tag` holds three historical free-text values in `a800/qwen3-a3b-30b-moe/moe.csv` while nothing in the source writes it. No column was added. `docs/profiling/README.md` records the operator's scope, the size of the pre-repair gap, and that a row cannot be checked for completeness from its own metadata. |
| Suite re-run after the documentation and record commits | `pytest tests/unit -q --continue-on-collection-errors` under `frontier-py310` at `cad3afd`: 84 failed, 3778 passed, 49 skipped, 11 errors. Identical to the W4 baseline and to the earlier W6 measurement. |
| Native rerun with the corrected FP8 wiring (external review C35-03) | `exp-0922-202645-561899`, `codesign` / H800 (`gpu-h800-0095`), vLLM 0.10.2 image, worktree `c231322`: **8 passed in 14.27 s**, exit 0. Seven zero-tolerance comparisons plus the FP8 structural check, now with `block_shape` reaching both GEMMs. FP8 numerics still not compared against a reference. |

## Step 7 — Zero-payload collective through the collective-sim backend

Full record: `test_report_2026-09-22_w7_collective_sim_zero_payload.md`.

| Item | Result |
| --- | --- |
| Reachability | Confirmed by source. `moe_operator_times.py:512` computes `data_size_bytes = embedding_dim * 2 * routed_tokens` and hands it to `predict_all_to_all`, so an EP lane routing no token in a step asks for an empty transfer; `predict_reduce_scatter` floor-divides by the device count and reaches zero for any payload below it. `base_cc_backend._validate_data_size` rejects only negative sizes, so zero reaches the runner. |
| Defects, confirmed by execution against published `main` (`b8518af`) | (1) an explicit zero and a deleted field produce the identical `exit=2, Error: missing required fields: ['tensor_bytes']`; (2) `-1` passes validation and reaches flow generation; (3) `--tensor-bytes 0` against a spec of 32768 yields 32768. |
| Companion fix | `fwyc0573/frontier-htsim` branch `fix/zero-payload-input-handling`, commit `eb7bc4f`, draft PR 1. `tensor_bytes` merges through `set_if_none_or_empty`; the required-field check became a table carrying per field whether zero is legal; a negative payload is rejected in the runner and in `Scenario.validate()`. No change to flow generation or latency arithmetic. |
| Companion tests | `tests/test_zero_payload_input.py`, 9 tests: **9 passed** against `eb7bc4f`. Negative control with `htsim_runner.py` and `schema.py` restored to `HEAD`: **6 failed, 3 passed**, the failures reporting the missing-field error. |
| Frontier tests | `tests/unit/test_collective_sim_zero_payload.py`, 4 tests on the canonical `TP=4 x DP=2, EP=8` pod with `intra_server_model=nvlink_analytic`: **4 passed**. An empty all-to-all and an empty reduce-scatter each price at `7 x 0.5 us = 0.0035 ms`; a 1 MiB all-to-all prices at `0.0060486222 ms`; a negative payload raises from Frontier's own guard. Negative control at gitlink `b8518af`: **3 failed, 1 passed**. |
| Clean-checkout validation | Fresh clone of the branch: the module skips with the submodule absent (**1 skipped**); `git submodule update --init` checks out `eb7bc4f` from `https://github.com/fwyc0573/frontier-htsim.git`; `make -j` returns `build_exit=0`; the four tests then pass. The gitlink therefore resolves from the published remote, not from anything local to this host. |
| Governance scan exposed by the bump | The first post-bump suite gave 85 failures. Diffing the `FAILED` lists named one new failure, `test_model_architecture_registry.py::test_raw_model_profile_resolution_callsites_are_allowlisted`: it `ast.parse`s every file under `frontier/`, and the initialized submodule adds 38 vendored files, one of which raises `IndentationError`. Two other scans walked the same tree and tolerated it while silently measuring vendored files. Repair: `tests/frontier_sources.iter_frontier_sources()` yields the 413 Frontier-owned files and skips the vendored subtree; all three scans use it. The three modules then give **88 passed**. |
| Regression, default environment | `pytest tests/unit -q --continue-on-collection-errors` under `frontier-py310` with the submodule initialized and built: **84 failed, 3782 passed, 49 skipped, 11 errors**. Diffing the `FAILED` lists against the 84-failure baseline gives an empty set in both directions; the four extra passes are the new module. |
| Fidelity matrix | Not run. No Frontier source file changed, and the backend fix is unreachable without `--cc_backend_config_type collective_sim`, which no matrix case selects. The unchanged failure set and the pre-fix/post-fix negative controls are the evidence. |
| Limits | The `7 x 0.5 us` figure is the analytic NVLink model's own arithmetic, not a hardware measurement. Under the default `legacy_fabric` model with `collective_exclude_intra_server=True`, a single-server pod prices both an empty and a 1 MiB all-to-all at `0.0`, because all traffic is intra-server and excluded; that is the configuration's semantics, unrelated to the payload. The companion PR is draft, so the gitlink points at a branch commit and must be re-pointed at `main` once it merges. |

## Step 8 — Combined regression on the integrated branch (§14.1)

Full record: `test_report_2026-09-22_w8_combined_regression.md`.

| Item | Result |
| --- | --- |
| Revision under test | `d881357`, 32 commits ahead of the PR base `refactor/oversized-module-split` @ `6ef0a3c`. |
| Base re-fetch before handoff | `git fetch origin main` gives `1f694f7`, unchanged since Step 0 and an ancestor of `HEAD`. No integration merge was needed and none was made. |
| Unit suite | `pytest tests/unit -q --continue-on-collection-errors` under `frontier-py310`: **84 failed, 3782 passed, 49 skipped, 11 errors**. Diffing the `FAILED` list against the recorded `origin/main` baseline list gives an empty set in both directions. |
| Integration suite | `pytest tests/integration -q --continue-on-collection-errors`: **15 passed, 22 skipped, 5 errors** against a baseline of 15 / 21 / 5. The one added skip is this branch's W6 parity module, which needs a GPU. All 5 errors are `test_pdaf_reference_lifecycle_observer.py` reporting the absent pinned Reference checkout at `/data/ycfeng/stepfun-performance-optimization/Frontier/worktrees/ref-afd-readonly`; identical on the base and environmental. |
| Architecture examples | All 16 release-supported scripts run end to end: 5 co-location offline, 2 co-location online, 2 PDD offline, 2 PDD online, 4 PD-AF offline, 1 PD-AF online. **16 passed, 0 failed**, each writing `request_metrics.csv` and `system_metrics.json`. |
| Pipeline cases | Four `PP=2` runs, all PASS: co-location offline dense `TP=2`; co-location offline MoE `Attn_TP=4, MoE_TP=2, MoE_EP=2`; sequential PDD offline dense with `PREFILL_PP=2, DECODE_PP=2`; co-location online dense `TP=2`. Each script's echoed topology confirms the intended values in its log. The changed cluster-scheduling, stage-dispatch, and metrics code runs on every stage, so these exercise it on the multi-stage path even though the new DP placement itself stays PP1-only. |
| Dense-layer coverage | From the dense examples, and from the MoE examples whose shared-expert work uses the separate `dense_mlp_hidden_dim` width through the ordinary linear-op path. Frontier has no "first `k` dense layers then MoE" model field, so there is no third heterogeneous shape to exercise. |
| Cold predictor cache | The same trained-predictor simulation run twice against an empty scratch cache via `--metrics_config_cache_dir`, so the repository `cache/` was neither moved nor deleted. Cold: 0 entries before, **26.9 s**, 63 artifacts written. Warm: 63 before, **2.1 s**, nothing new written. Their `request_metrics.csv` outputs are byte-identical (`request_e2e_time = 16.77818517187422 ms`, `ttft = 8.430025150867172 ms`), so the persisted-cache path reproduces the freshly trained path exactly. |
| Working tree | `git status --porcelain` empty afterwards. The one leftover `outputs/examples/` tree was removed after confirming 0 tracked files there; the 110 tracked files under `outputs/` are all still present. |
| Pre-existing defect, deferred | `AGENTS.md` §Tests names `comm_backend_tests/`, `debug/`, and two `bash tests/debug/e2e-level/monolith_mode/scripts/*.sh` commands. `tests/debug/` exists neither here nor on `origin/main`. The same missing tree causes 10 of the 84 baseline unit failures in `test_colocation_release_review_contracts.py`, and a docstring at `vllm_v1_engine_replica_scheduler.py:16` still points into it. One pre-existing defect class from the release scrub, unrelated to Issue 26; recorded in `future.md` and not repaired here. The PP2 coverage was obtained through the example scripts instead. |
| Limits | CPU only. No native profiling suite and no vLLM serving or TTFT comparison, both excluded by §14.1. The PD-AF Reference-checkout tests could not run on this host. The example runs use dummy execution time except for the CSV smokes, so they validate structure, lifecycle, and conservation rather than latency accuracy. |

## Step 9 — PP>1 support for the vLLM DP placement policy (2026-09-22 to 2026-09-23)

| Item | Command | Expected | Actual | Outcome |
| --- | --- | --- | --- | --- |
| Reference-loop oracle | `pytest tests/unit/test_dp_placement_reference_loop.py -q` | Every §18.11 state-table row reproduced from a model of the engine iteration alone | `9 passed in 1.21s` | PASS |
| Oracle with the existing balancer suite | `pytest tests/unit/test_dp_placement_reference_loop.py tests/unit/test_vllm_dp_load_balancer.py -q` | No interference with the shipped balancer tests | `70 passed in 1.44s` | PASS |
| Frontier boundary probe, `attn_dp=2 PP=1` | scratch `probe_frontier_boundaries.py` | 6/6 requests complete; peer keys equal | 6/6, 24 boundaries, peers equal at every boundary | PASS |
| Frontier boundary probe, `attn_dp=1 PP=2` | same | Consecutive admissions carry distinct keys | Both cold-fill admissions read key 0 | FAIL, invariant I5 |
| Frontier boundary probe, `attn_dp=1 PP=3` | same | Same | All three cold-fill admissions read key 0 | FAIL, invariant I5 |
| Frontier boundary probe, `attn_dp=2 PP=2` | same | 6/6 requests complete | Event queue drained with requests unfinished | FAIL, W9-01 |
| Ground-truth writer | scratch `check_trace_writer.py` | Gate off writes nothing; buffering, per-process file, dense `seq`, idempotent flush, warmup gate | All checks passed | PASS |
| Ground-truth syntax | `python -m py_compile` on the four changed vLLM files | Parse | All four parse | PASS |
| P1(b) after the W9-01 merge-forward | `step9_p1b/probe_boundaries.py` on seven shapes, scored by `step9_p1b/analyze_keys.py` | 6/6 per shape; the accepted key has 0 peer splits, merges and inversions against each report's stage-0 group | 6/6 on all seven; the group-anchored key scores 0/0/0 and 0 ms everywhere and equals the stage-0 predictor on 139 of 139 reports; the lane counter fails on four shapes | PASS; D9-2 decided as group-anchored |
| P2 implementation, existing suites | `pytest` on the DP-placement, reference-loop, stage-context and forward-group test modules | Only the intended PP rejection case fails | 1 failed (`pipeline_parallel` guard case), 112 passed | PASS |
| P3 unit tests | same modules after P3 | All pass; each new or changed case fails on the pre-P2 tree | 132 passed; on the pre-P2 tree 19 failed, 72 passed, and the 19 are exactly the new or changed cases | PASS |
| P4 real event loop | `pytest tests/integration/test_vllm_dp_placement_runtime.py tests/integration/test_monolithic_mixed_forward_runtime.py` | Each report maps to one reference iteration kind under its forward's key; conservation; no added event type; the discriminating probe is placed differently only because of what was published | 9 passed, 3 passed; probe lane 0 under the policy, lane 1 under the completion-reporting control | PASS |
| P5 C2, PP=1 policy scenarios | `step9_p5/c2_pp1_policy_matrix.py`, 24 scenarios on exports of `d1a2a06` and `bacdbb4` | Artifacts identical after path substitution; report stream identical without the key; keys order-isomorphic; selections identical; no added admission-only report | 24 of 24 identical (20 drained on both sides; 2 incomplete on both sides, W9-05; 2 stalled on both sides with identical diagnostics, W9-04) | PASS |
| P5 C2, other cluster schedulers | `step9_p5/run_fidelity.sh` on clean detached worktrees, `--clean-cache` | 71 of 71 identical | 71 of 71 identical; 0 provenance findings; `complete_comparison` true | PASS |
| P5 architecture examples | `step9_p5/run_examples.sh` on both exports, `compare_examples.py` | 16 of 16 pass on both; artifacts identical | 16 of 16 pass and identical | PASS |
| P5 suites | `composition_run_suites.sh` at `bacdbb4`, compared by test id with `composition_compare_junit.py` against `03d5f24` | 0 regressions, 0 new failures | unit 84 failed / 3829 passed / 51 skipped / 10 errors; integration 5 errors / 26 passed / 22 skipped; 0 regressions, 0 new failures, 0 skip changes | PASS |
| W9-04 prototype (not applied) | `step9_p5/deadlock_sweep.py` on an export with `step9_p5/w9_04_prototype.patch` | Cause established before any change | Sweep 72 of 72 drain (branch 66 of 72); 22 of 22 previously drained C2 cases identical | Finding; fixed in `2ffb062` (next section) |

Limits. The instrumented vLLM paths have not been executed; that needs a GPU
host and is package G3, still blocked with G4 and G5. The three probe FAIL rows
of 2026-09-22 are findings, not regressions: the I5 failures are the
measurement the design checkpoint asked for, and W9-01 was a pre-existing
defect on `main`, fixed by PR 36 and merged forward before P1(b). The P1(b)
targets are Frontier's own forward grouping, not vLLM measurements. W9-04 and
W9-05 are pre-existing and recorded in `issues.md`; neither is repaired here.
Evidence: `test_report_2026-09-22_w9_pp_dp_placement.md`, `step9_p1b/`,
`step9_p5/evidence/`.

## W9-04 fix (2026-09-23)

Commit `2ffb062`, compared with `bacdbb4`. The criteria were fixed in plan
§18.17 before measuring. Clean detached worktrees `.worktrees/w9-04-{before,after}`
were used for A4–A6, and `git archive` exports for A2, A3 and A7. Environment
as in Step 9. Scripts are in `w9_04_fix/`, and evidence in
`w9_04_fix/evidence/`.

| Id | Command | Expected | Actual | Outcome |
| --- | --- | --- | --- | --- |
| A1 | `pytest tests/integration/test_vllm_dp_placement_runtime.py`, then the new case on a control tree with only the call removed | Pass; the control fails with non-empty scheduler state | 10 passed; control `1 failed` with `RuntimeError: Sequential simulation ended with non-empty scheduler state` | PASS |
| A2 | `run_a2_a3.sh` (sweep, three cluster schedulers) | 72 of 72 drain | 72 of 72 drain, 24 of 24 requests each | PASS |
| A3 | `run_a2_a3.sh` (C2); the 22 cases that finished before compared with the matrix comparator | 22 identical; the 2 stalled cases finish apart from W9-05 losses | 22 of 22 identical; 24/24 and 23/24 without error, the latter byte-identical to the prototype | PASS |
| A4 | `run_a4_a5_a6.sh` (refactor fidelity, `--clean-cache`) | 71 of 71 identical | 71 of 71; 0 provenance findings; `complete_comparison` true | PASS |
| A5 | `run_a4_a5_a6.sh` (`stage_admission_matrix` G3b/G9/G10, sets `w904-before`, `w904-after`) | 0 STOP; every cell identical | 51 of 51 PASS; 51 of 51 `sha256sums.txt` identical | PASS |
| A6 | `run_a4_a5_a6.sh` (`composition_run_suites.sh`), `composition_compare_junit.py` against the P5 JUnit | 0 regressions, 0 new failures | 0 / 0 / 0 skip changes in both suites; only new id is the regression test | PASS |
| A7 | `run_examples.sh`, `compare_examples.py` against P5's `bacdbb4` outputs | 16 of 16 pass and identical | 16 of 16 | PASS |

Limits:

- CPU only.
- The fix keeps Frontier's late-join rule. Whether a late join should instead
  wait for the next forward, as in vLLM, is W9-03 and is not measured here.
- The one request missing in the `tight_kv` case is attributed to W9-05 from
  its signature: tight KV, a drained queue, and no error. It is not traced.
  The W9-05 fix below confirms it: that case completes 24 of 24 with the fix.

## W9-05 fix (2026-09-23)

Commit `75c1140` compared with `2ffb062`; the commits between them change only
records. Criteria fixed in plan §18.18 before measuring. The tested trees were
`33a1c8e` plus the uncommitted fix, whose three files are byte-identical to
`75c1140`. `git archive` exports under
`/data/ycfeng/tmp/issue26-correctness-pr/w9_05/trees/{before,after}` were used
for B2–B5 and B8. Detached worktrees `.worktrees/w9-05-{before,after}` were used
for B6 and B7, because the stage-admission matrix and the git-provenance unit
tests need a git checkout. Environment as in Step 9. Scripts are in `w9_05/`,
and evidence in `w9_05/evidence/`.

| Id | Command | Expected | Actual | Outcome |
| --- | --- | --- | --- | --- |
| B1 | `pytest tests/integration/test_vllm_v1_decode_preemption_runtime.py`; the same file from the `2ffb062` tree root; `w9_05/symptom_probe.py` on both trees | Pass; fails at `2ffb062` on the progress assertion | 1 passed in 1.44 s; at `2ffb062` `assert 0 == 34`; probe: request 1 incomplete (0 of 30 decode tokens, exit 0) at `2ffb062`, 3 of 3 complete with the fix | PASS |
| B2 | `run_validation.sh` (C2 and `kv_pressure_sweep.py` on both trees) | `tight_kv` 24 of 24 in all three shapes | 24/24 each (before 20, 20, 23). Sweep: 72 of 72 complete, 0 lost, 0 short outputs (before 31 of 72 complete, 176 lost); lossy cells = cells with a decode-phase preemption, 41 = 41 | PASS |
| B3 | `c2_pp1_policy_matrix.py compare` | Cases without a decode-phase MONOLITHIC preemption identical; each differing case has one and completes at least as many | 21 of 24 identical, all with 0 preemptions; 3 differing `tight_kv` cases with 5, 6, 2 preemptions and more completions | PASS |
| B4 | `deadlock_sweep.py`, `round_robin` / `lor` / `random` | 72 of 72 drain; completion does not fall | 72 of 72 drain with 24 of 24; every cell equal to W9-04's | PASS |
| B5 | `tests/e2e/refactor_fidelity/run_matrix.py run/compare --clean-cache` | Identical except decode-phase MONOLITHIC preemption cases | 71 of 71 identical; no such case in the matrix (see Limits) | PASS |
| B6 | `tests.e2e.stage_admission_matrix` G3b/G9/G10, sets `w905-before`, `w905-after` | 0 STOP; cells identical | 51 of 51 PASS; hashes identical | PASS |
| B7 | `composition_run_suites.sh` on `.worktrees/w9-05-after`, `composition_compare_junit.py` against the W9-04 JUnit | 0 regressions, 0 new failures | unit 84/3829/51/10 and integration 5 errors/28/22, as before plus the new test; 0 / 0 / 0 skip changes | PASS |
| B8 | `run_examples.sh`, `compare_examples_symlink.py` | 16 of 16 pass and identical | 16 of 16 | PASS |

Notes on the runs:

- B7 was first run on the exported tree. 88 unit tests failed there, all on
  git provenance (`git rev-parse HEAD`, `unable to capture git provenance`), a
  property of the export. On the git worktree the counts equal W9-04's. Running
  `tests/unit/test_moe_ep_non_dummy_matrix.py` alone fails collection on both
  `2ffb062` and the fix (`step3_text` references an unknown operator family);
  it passes inside the full suite. This is an existing import-order dependence,
  recorded here and not changed.
- B8's first comparison reported one path difference in `config.json`
  (`trace_file`), because `trees/before` is a symlink to the W9-04 export and
  the comparator substituted only the resolved path.
  `compare_examples_symlink.py` also substitutes the literal path; the
  artifacts are then identical.

Limits:

- CPU only, with dummy or trained predictors.
- B5 does not exercise the changed branch: no fidelity case has a MONOLITHIC
  preemption. The changed behavior is covered by B1, B2 and B3.
- The resumed request's replay (recomputing prompt and output) is not modeled
  (`issues.md` W9-05, Limits).
