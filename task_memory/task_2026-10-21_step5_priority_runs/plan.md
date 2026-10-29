# D21-RUNS plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Created scoped dependency and verification plan. |

1. P0: parameterize native32/64 topology and preemptibility through four case entries; correct source-supported first-start diagnostic; run real CPU HTTP qualification for both DP sizes; commit and seal complete H80032 qualification first.
2. Release: use existing clean-run and four-metric reading schemas for formal prerequisites, including own readiness and same-hardware32 three-repeat30% dependency. Bind Step hardware/shape identity; retain final10% comparison.
3. Seal remaining basic qualification stages, CPU dry-runs, cost projections and compact parent handoff; no platform submission.

Dependency: `source inspection -> P0 implementation -> CPU checks -> committed source -> P0 sealed handoff -> {formal release check, other basic stages} -> verification -> final commit`.

Acceptance: real HTTP tests exercise16 warmups/verified idle/8 simultaneous completions at DP2 andDP4; frozen resource payloads have4/8 nodes and correct preemptible boolean; complete package standalone verification and dry-run; absent/foreign/incomplete/>30% dependency blocks formal64 while equality passes, qualification independent; final gate unchanged. Source, original model metadata, probe helpers and immutable image pins verified locally; runtime truth remains unverified until parent inspects actual native evidence.
