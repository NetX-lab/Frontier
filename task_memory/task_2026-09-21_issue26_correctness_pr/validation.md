# Issue 26 Correctness PR — Validation Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created. Environment recorded; baseline results recorded in the refactor task's Step 0 report because both branches share the same base commit. |
| 2026-09-21 | Step 1 recorded: audit spot checks and the vLLM reference identity check. |
| 2026-09-21 | Step 2 recorded: unit sensitivity and the fidelity measurement against a stated expectation. |

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
