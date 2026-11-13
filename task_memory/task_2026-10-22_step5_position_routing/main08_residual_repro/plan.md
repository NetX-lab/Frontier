# D29-MAIN08-RESIDUAL-REPRO plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record and complete the bounded actual decode-wave diagnosis route. |

Dependency: offline scoped sample -> actual layer61/16-EP wave and trace loops
-> ranked hypotheses -> isolated variable tests -> exact artifact comparison
-> diagnosis handoff. No implementation or full CLI is in this slice.

1. PASS: read requirements/plan, GLOSSARY and the dummy-weight ground-truth ADR.
2. PASS: stream the 259555-byte offline scheduler log and bounded 4MiB head/
   8MiB tail trace samples; do not load the 1.448GB trace or touch online.
3. PASS: replay actual complete decode trace events through real TraceStore.
4. PASS: replay the actual position6147/live1+idle0, layer61, all16 EP lanes
   through public predictor, real wave phases, metadata emission and flush.
   Match actual captured model/count/plan evidence before final conclusions.
5. PASS: rank four falsifiable hypotheses before single-variable tests;
   preserve all point checks and compare complete cost/proof/plan payloads.
6. PASS: retain valid RED loop, causality evidence, setup failures and limits.
7. SKIPPED: production repair, full CLI, strict77/unit/GPU/native qualification.

The final deterministic RED signal is repeated unchanged-artifact reads plus
more than20000 actual JSON decodes per one16-EP wave. This is a diagnostic
work-count threshold, not a new acceptance gate or replacement for2400 seconds.
The final original wave actually measured28003 warm parses and80 reads.
