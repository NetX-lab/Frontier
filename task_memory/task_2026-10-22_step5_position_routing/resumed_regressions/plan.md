# D29-E41 final regressions plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Establish isolated, independent regression executions. |

1. Recover authoritative requirements, previous commands, baseline IDs and hashes;
   verify frozen HEAD, submodule and dirtiness. PASS requires exact declared source.
2. Run independently: strict77 final04; full-unit parent04 (1800s); supplement101
   (180s); metadata196 (180s); catalogue53 (120s). Use existing interpreters and
   exact suites, new basetemps/JUnit/output roots and single-thread BLAS.
3. Compare strict77 through unchanged harness; rehash both sides and check full
   scenarios, provenance, cache inventories and clean-cache capture.
4. Compare full-unit exact IDs with locked baseline and latest baseline overlay;
   reconcile every added failure/error against explicit same-source supplement.
   Check baseline disappearance/PASS-to-SKIP and affected suite membership.
5. Recheck source/dirtiness, checkpoint progress, archive independent suite reports
   and concise summary; commit only this new record directory.

Dependency: 1 -> {2a,2b,2c,2d,2e} -> {3,4} -> 5.

The harness creates `repo/cache`; bubblewrap keeps every pre-existing source
entry read-only at its original absolute path while a scratch-backed namespace
root receives generated cache/outputs. No harness/scenario/source edit is needed.
Each suite uses a distinct namespace and generated-file directory. This is
filesystem isolation, not a dependency/import-path workaround. Retained baseline
artifacts and caches stay read-only. No broadened suites or repeat repair57/6.
