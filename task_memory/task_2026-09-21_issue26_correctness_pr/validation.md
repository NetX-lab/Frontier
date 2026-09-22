# Issue 26 Correctness PR — Validation Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created. Environment recorded; baseline results recorded in the refactor task's Step 0 report because both branches share the same base commit. |
| 2026-09-21 | Step 1 recorded: audit spot checks and the vLLM reference identity check. |
| 2026-09-21 | Step 2 recorded: unit sensitivity and the fidelity measurement against a stated expectation. |
| 2026-09-22 | Step 2 re-measured with harness and source at one revision, after the gate corrections. Same expectation, same result, recorded provenance. |
| 2026-09-22 | Step 3 recorded: the shared monolithic forward, its direct-construction runtime evidence, four deliberate-defect controls, and a 71-of-71 identical fidelity matrix. |

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
