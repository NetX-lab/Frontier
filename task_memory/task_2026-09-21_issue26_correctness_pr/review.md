# Issue 26 Correctness PR — Review Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created with the pinned source snapshot. Dispositions are filled in Step 1. |

## Pinned source snapshot

| Repository / reference | Revision | Role |
| --- | --- | --- |
| `NetX-lab/Frontier` `main` | `1f694f7c549aa3aeeb7c5bbae04e119c09167a77` | Integration baseline (verified 2026-09-21 after `git fetch`). |
| `NetX-lab/Frontier` `bug/ttft-check` | `a7b3320fe9b8b083ee86b91dae3d6838f4443d91` | Candidate. Final commit touches only `task_memory/`. |
| Candidate parent before evidence import | `b7f8d055461d9208b8ae57eceeb6c0246cbc8d3c` | Source comparison point (verified equal source tree to `a7b3320`). |
| Merge base | `d71ad80b0800880808a0857fd30477e6d96592c6` | Verified with `git merge-base`. |
| `fwyc0573/vLLM-BS` | `ea95f571e20937c7c908c6d59ddd1cd6bf9268f1` | vLLM reference, `.real-engine/vLLM-BS`. |
| `fwyc0573/frontier-htsim` main gitlink | `b8518afcc310f0fe0e3ce52ba6b4f0bf57a3be04` | Current optional backend. |
| Candidate gitlink | `e564935d3874d8c71b52a554ab7c9a72e5e19f68` | Not reachable on the configured remote (HTTP 422, 2026-09-21). |

## Candidate change dispositions

To be completed in Step 1. Format: path, donor hunk summary, old defect, main behavior, reference behavior, disposition (`PORT`/`ADAPT`/`ALREADY_PRESENT`/`DROP`/`BLOCKED`), chosen owner, planned test.

| Path | Disposition | Notes |
| --- | --- | --- |
| `tests/e2e/issue26_*`, `tests/integration/issue26_dp_*rca*`, `tests/performance/issue26_*` | DROP (default, per `plan.md` A11) | Experiment scripts; re-evaluate only a named helper. |

## Design decisions

Pending Step 1.

## Final code-review findings

Pending Step 8.
