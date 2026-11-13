# D29-E41 final regressions requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record delegated final-source regression scope and frozen baseline. |

## [Original Request]

TaskID D29-E41-FINAL-REGRESSIONS. Close required regression gates on read-only
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05` at
`e41abc073500d61f224eda4fca1123e74e40544d`, submodule
`3dac53c544e546574cb6e8078cd14053d0916113`.

Run strict77 final04 against the unchanged retained baseline; full-unit parent04
against locked failure/error IDs; same-source Python3.12 supplement101;
metadata196 and catalogue53 affected suites. Keep suites separate. Preserve raw
exits, ID additions/removals, missing baseline IDs, PASS-to-SKIP transitions,
environment differences, source identity, scenario/cache/artifact comparisons.

No source edits, checkout, source commits, further delegation, full CLI/GPU or
external-network operations. Do not repeat source-equivalent repair57/6. Only new
records here and unique scratch under `/data/ycfeng/tmp` are writable. Commit
these records only after completion. Root owns separate main06 CLI acceptance.

Acceptance: strict77 equality and no new code failure after explicit, evidenced
environment reconciliation. Failures must remain visible, without exclusions,
changed gates or ambient PYTHONPATH masking.
