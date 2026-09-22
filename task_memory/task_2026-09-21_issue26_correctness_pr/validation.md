# Issue 26 Correctness PR — Validation Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created. Environment recorded; baseline results recorded in the refactor task's Step 0 report because both branches share the same base commit. |
| 2026-09-21 | Step 1 recorded: audit spot checks and the vLLM reference identity check. |
| 2026-09-21 | Step 2 recorded: unit sensitivity and the fidelity measurement against a stated expectation. |
| 2026-09-22 | Step 2 re-measured with harness and source at one revision, after the gate corrections. Same expectation, same result, recorded provenance. |

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
