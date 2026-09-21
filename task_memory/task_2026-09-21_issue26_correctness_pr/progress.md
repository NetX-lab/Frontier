# Issue 26 Correctness PR — Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Step 0 started: records landed, environment created, baseline pending. |

## Status

| Field | Value |
| --- | --- |
| Correctness branch | `fix/issue26-correctness-pr` (worktree `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`) |
| Base at creation | `refactor/oversized-module-split` @ `41dabfb9d5ef3b51cdf3009d486450515d9a8d2d` (itself on `origin/main` `1f694f7`) |
| Prerequisite | MET. All four modules this PR edits are under the 2,000-line gate; the split measured 67 of 67 fidelity cases identical with no predictor cache name differences. |
| Current step | Step 1 complete; Step 2 unblocked |
| Publication | PUSHED_VERIFIED (records) |
| Next action | Step 2: preserve round-robin DP rotation across scheduling calls. Decisions D1 and D2 in `review.md` remain open and are the user's; neither blocks Step 2. |

## Step status

| Step | Work package | Status | Test | Publication | User review |
| --- | --- | --- | --- | --- | --- |
| 0 | Worktree, references, baseline | PASS | PASS (baseline recorded) | PUSHED_VERIFIED | NOT_REVIEWED |
| 1 | Candidate/vLLM audit | PASS | n/a (source audit) | LOCAL_ONLY | NOT_REVIEWED |
| 2 | RR DP rotation | NOT_STARTED | — | — | — |
| 3 | Shared monolithic forward | NOT_STARTED | — | — | — |
| 4 | Opt-in vLLM DP placement | NOT_STARTED | — | — | — |
| 5 | Routing implementation identity | NOT_STARTED | — | — | — |
| 6 | Legacy fused-MoE profiling | NOT_STARTED | — | — | — |
| 7 | Optional zero-payload backend | NOT_STARTED (facts in `plan.md` A7) | — | — | — |
| 8 | Combined regression, PR hand-off | NOT_STARTED | — | — | — |

## Chronological updates

- 2026-09-21: Baseline on the shared base recorded in the refactor task's Step 0 report; vLLM reference cloned (no tags in the fork; upstream `v0.10.2` comparison deferred to Step 1).
- 2026-09-21: Draft specification analyzed; eleven facts verified against main, the candidate, the submodule remote, and the host; planning interview settled twelve decisions (see `requirements.md`). Records landed under `task_memory/`, `.gitignore` narrowed, `plan.md` carries the Amendments table.
- 2026-09-21: Step 1 audit complete. Three pinned-source audits landed as `audit_scheduler.md`, `audit_predictor_profiling.md`, `reference_vllm_0_10_2.md`; dispositions and two decision checkpoints recorded in `review.md`. Execution order corrected to W2 first, then W3, then W4, because the candidate's report-order key depends on the shared forward identity. The vLLM fork was confirmed to be a direct descendant of upstream v0.10.2 with the DP placement files byte-identical to the tag.
- 2026-09-21: Merged the completed oversized-module split into this branch. Merge rather than rebase, so the published review anchors stay valid. The modules this PR edits are now `config.py` 788 with `cluster_config.py` 1888, the vLLM V1 replica scheduler 1386, the prediction model manager 722 and the MoE predictor 1557, each with named child modules that give the planned fixes a clear owner. Step 2 is unblocked.
