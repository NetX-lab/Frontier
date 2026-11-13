# D29 performance final regressions plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Define fresh suites, exact-ID comparisons and conditional affected coverage. |

Dependency: preflight -> {strict77, full-unit, supplement101, metadata selection,
catalogue selection} -> audits/conditional uncovered affected checks -> archive.

1. Read current root requirements/plan and retained previous commands; verify new
   HEAD/tree/gitlink, clean status, nonexistent output paths, unchanged fidelity
   harness/scenarios and locked baseline XML/IDs. Record seven changed paths.
2. Run five required suites independently in background without polling. Preserve
   original absolute source paths through the previous read-only bubblewrap layout;
   generated repo/cache goes to unique scratch, never the frozen worktree. Do not
   modify source, scenarios or gates. Each suite has separate logs/JUnit/receipts.
3. Reuse unchanged strict comparator and retained hash/cache/provenance auditor.
   Require77/77 equality, complete scenario table and cache filename equality.
4. Compare exact unit IDs against actual locked baseline, latest overlay and e41.
   Check all new cases and failures, baseline loss, PASS-to-SKIP and affected
   coverage. Reconcile only actual missing dependencies with fresh exact-ID/module
   py312 results. Governance52 uses explicit existing extras only if full-unit
   and required supplement leave this affected coverage unsettled.
5. Recheck immutable source/tree/submodule and recorded hashes. Write proportionate
   independent reports, JSON receipts, final summary and progress checkpoint here.
   Raw large XML/logs stay in scratch, referenced by exact hashes. No commits.

Acceptance: fresh strict77 equality; no unexplained new code failures, baseline
case loss or PASS-to-SKIP; all new regression cases pass; required affected checks
settled without duplicate suites. Primary main08 and wider runtime/GPU acceptance
remain separate. Counts use actual new XML and are never summed across suites.
