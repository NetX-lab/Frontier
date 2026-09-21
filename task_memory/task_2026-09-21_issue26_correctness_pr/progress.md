# Issue 26 Correctness PR — Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Step 0 started: records landed, environment created, baseline pending. |

## Status

| Field | Value |
| --- | --- |
| Correctness branch | `fix/issue26-correctness-pr` (to be created from `refactor/oversized-module-split` after the refactor branch's Step 0 push) |
| Base at creation | pending |
| Prerequisite | Refactor PR fidelity matrix PASS (see `task_memory/task_2026-09-21_oversized_module_split/progress.md`) |
| Current step | Step 0 (records landed; branch creation pending) |
| Publication | LOCAL_ONLY |
| Next action | Create the correctness worktree from the refactor branch, push, and stop for user review (Q6=a). |

## Step status

| Step | Work package | Status | Test | Publication | User review |
| --- | --- | --- | --- | --- | --- |
| 0 | Worktree, references, baseline | IN_PROGRESS | NOT_RUN | LOCAL_ONLY | NOT_REVIEWED |
| 1 | Candidate/vLLM audit | NOT_STARTED | — | — | — |
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
