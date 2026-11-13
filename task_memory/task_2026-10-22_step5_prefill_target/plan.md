# D26-PREFILL-TARGET Plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Define source, regression, fixture and handoff milestones. |
| 2026-10-22 | Mark verified scoped milestones complete and identify parent acceptance. |

1. Inspect registered transfer semantics and real shared-Request lifecycle;
   establish red scheduler→P completion→READ→D tail→output boundary regressions.
2. Implement family-scoped target selection, P completion and source ownership;
   preserve semantic lengths and make mismatched exact transfer coverage fail.
3. Update the three measured-main fixture/runner/test files and create a fresh
   unadmitted CPU fixture, validating exact keys and truthful provenance.
4. Run targeted and adjacent regressions, checkpoint progress, write test report,
   commit source/tests/records, and hand off diff and remaining fullCLI/native work.

Dependency:1 ->2 ->3 ->4. No parallel work or further delegation.
Acceptance includes canonical6144/1024 and6143/6144/6145 boundaries, exactly-once
completion/transfer/read/release, all output credits and zero dummy credits.
CPU fixtures prove lifecycle/lookup integration only, never native fidelity.

## Final status

Steps1–4 completed: source locked, red/green lifetimes and affected regressions
passed, new fixture generated/reloaded, source and fixture changes committed,
test report/receipt/archive/diff prepared for the final records commit.
FullCLI, D27 physical geometry, native profile/fidelity and strict77 acceptance
remain outside this slice and parent-owned, as recorded in `summary.md`.
