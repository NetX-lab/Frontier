# D18-NATIVE progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded source implementation, HTTP regressions and staging handoff pending verification. |

- **completed:** read primary D18/spec/plan; preserved shared/historical workload bytes and parent R2 ownership.
- **completed:** confirmed locked Step choice-level token-ID streaming. Obtained explicit authorization for the minimal shared-client extension; added stable prompt identity, exact native token counts/timestamps and per-request failure retention in the existing replay client.
- **completed:** added canonical phase consumer (16+128 requests ×3), complete-population metrics, actual per-lane idle/fresh-completion checks and soft observation/error diagnostics. Phase-boundary snapshots avoid rewriting large token-delta records on every dispatch.
- **completed:** source-derived full-role H200 limits 7168/8192/128 and native capture ceiling256. Built-in native graph statistics are enabled; no custom R2 engine hooks or cache overrides.
- **completed:** initial affected baseline 46 PASS. New suite iteration: 14 PASS/1 FAIL because the injected failure fixture used the wrong template-ID suffix; fixed the fixture to select the actual frozen template. Joint follow-up: 425 PASS. A broader run hit a collection-only environment error (`plotly` absent in asyncopd-py312); no implementation behavior was changed. Reran harness tests in Python3.12 and the narrow simulator metric regression in the existing frontier Python3.10 environment: **574 PASS +9 PASS**, before the final native graph-flag assertion.
- **completed:** final574-test affected suite and9 narrow simulator metric tests passed; after narrowing to the source-confirmed choice-level prompt-ID schema,160 overlapping affected tests passed again. `git diff --check` passed.
- **in-progress:** committed-source stage build, isolated stage verification and exact resource/submission receipts.
- **pending, parent-only:** H200 runtime qualification, per-lane/rank evidence, parent R2 verdict and formal native admission. No GPU/auth/platform/cloud/SSH/Docker operations performed here.

Raw test outputs are under `/data/ycfeng/tmp/d18_native_tests01` through `d18_native_tests04`; test reports below record the commands and observed totals. No full simulator sweep is needed for these HTTP harness-only edits.
