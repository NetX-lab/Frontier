# Startup census plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Define a scoped RED/GREEN observer correction and unchanged native acceptance. |

1. Reproduce native `Worker.compile_or_warm_up_model` calling its wrapped `execute_model` at the existing CPU host seam.
2. Bind startup lifetime to the native method boundary, including exceptional return; do not classify request-name prefixes.
3. Run focused census/observer and source-pin/stage tests. Verify native calls and their results/exceptions remain unchanged.
4. Record exact evidence and limits, commit the CPU fix. Preserve original capture01 verdicts.

Dependency: 1 -> 2 -> 3 -> 4. Independent streams: typed MRv2 input parent verification, D23 forward-start implementation and HTTP502 diagnosis.
