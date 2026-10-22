# D29 Frontier position-routing progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Receive six design findings and start disjoint core, pricing and profile-source corrections. |
| 2026-10-22 | Receive the first design and request independent scope/correctness review. |

- completed: `design.md` traces actual Batch context/query snapshots, source
  lanes, shared materializers and profile consumers. Author reports 200
  mathematical cases and path/budget/format checks passing; these are not
  production tests and the parent has not independently rerun them.
- completed: initial estimate is 29 code/test files / 1556 gross changed lines,
  plus 3 required fixture files / 143 lines. Communication/pricing corrections
  are not included, so this is not a complete implementation budget.
- completed: reviewer `ses_e9e0793b2ffe5YWSjciM0uRzna` identifies six findings;
  the initial estimate is not acceptable as complete/minimal. `review.md`
  records source references, required corrections and unnecessary new-policy scope.
- in-progress: three disjoint design corrections cover count/schema/cache/exact
  coverage, source-local/physical collective pricing, and derived profile-source
  entry. The actual-main fixture is mandatory. No source implementation started.
- pending: settle required companions, combine corrected native design and
  user scope approval, implement, verify and collect fresh evidence.

No simulator/profiler source was changed and no GPU/platform operation was run.
