# Joint CORE and producer closure review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Preserve independent joint and registry-view review findings and parent disposition. |

## CORE and producer phase

TaskID D29-CORE-PRODUCER-JOINT-REVIEW; reviewer session
`ses_e9afdb7c4ffejb3hbkjZCeHFKS`. Frozen bfbc87ad; complete CORE seven-path
d96b3279..e49b26f8 and PRICING five-path84e66115..a7e2a3ca diffs, related callers,
validators and deleted hunks. Verdict: **No findings**.

- Producer P1 closed: full ambient validation remains; stable context uses reached
  callable provenance. Wrapper keeps full diagnostics and row provenance separately
  (`fp8_block_experts.py:124,160–167`, `moe_wrapper.py:998–1013`).
- Producer P2 closed: current checkout writer is the default; external references
  require an explicit file and matching complete-file SHA
  (`test_collective_sim_cc_backend.py:105–127`).
- CORE R1 closed: actual bound compute, simulator and strategy callables are checked
  through real published inheritance/delegation (`moe_impl.py:307–311`).
- CORE R2 closed: source descriptor required; existing validator always runs on
  selected rows before both trainers/shared preflight (`moe_routing_runtime.py:91–130`).
- CORE R3 closed: actual source owner/live0 establishes coverage; all applicable
  compute, including supplied duration0, requires exact positive duration/count
  proofs (`run_step5_measured_main.py:235–257`).

Reviewer traced wrapper→normal CSV, exact buffer/count export, CORE recovery and
H04 context/layout comparisons. Evidence is segmented, not a single full native
producer→CSV→export→H04 execution. Full points stay outside fit grouping. Reviewer
did not run tests or platform operations. Five known fixture failures are assigned
separately and were not duplicated as findings.

## Registry-view phase

TaskID D29-REGISTRY-VIEW-REPAIR-REVIEW; reviewer session
`ses_e9abd8364ffee5sQKdpiQh3yH4`. Exact8f96408c..a5189e19 two-path diff.
Verdict: **P2 closed; No findings**. Bare-name family projection alone excludes
native candidates; native hash labels remain. Public ambiguity and explicit
unknown-binding rejection are unchanged. Both insertion orders, legacy/typed
domains and real family/cluster views are covered. No cache/typed/caller changes.
Owner217PASS is distinct from parent combined regression, which still has five
CORE fixture failures. Independent public mode is not an added D29 prerequisite.

## Parent verification and disposition

Frozen worktree `verify-step5-core-closure-01/core_parent` under the task directory
retains exact diffs, JSON/XML/logs and `test_report_2026-10-23_joint_parent.md`.
Parent independently observes717 unique cases:693PASS,5FAIL,19SKIP. The three
previously dependency-blocked wrapper cases now pass using an isolated extra-site.
All five failures match the assigned two-file fixture repair. Source review
findings are closed; full regression and full runtime qualification remain open.
No native accuracy, GPU/backend/graph, G3 or four-metric acceptance is implied.
