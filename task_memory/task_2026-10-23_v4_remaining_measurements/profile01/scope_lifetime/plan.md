# Source slice plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded source/CPU ownership and stop gate. |

Dependency: S0 frozen inputs -> S1 callback/window/lifetime/replay implementation and CPU RED/GREEN -> S2 exact source/static/CPU/budget/immutable closure -> primary independent dual review/replay -> separately allocated S3 package -> S4 bounded GPU -> S5 actual audit.

1. S0: read applicable instructions, parent records, design, actual shared source and official CUPTI header/sample; verify hashes and clean base1566b359.
2. S1: reuse frozen ABI/IDs/subscriber/stage/target extraction and real range; extend only callback13 and copy every payload while valid. Test deliberately different source/clone IDs, freshness/TLS, invalid joins and lifecycle.
3. S2: asyncopd-py312 CPU checks under single-process60s/30G; retain raw RED and GREEN, full diff/numstat/gross budgets/source/test hashes and immutable closure. No CUDA call on host.
4. Commit only owned source/tests after verification, then owned records; stop. Missing scope or budget blocks delivery rather than weakening predicates.

Acceptance: P1–P6 remain independently auditable even for negative scope delivery; setup/query/trace errors are UNVERIFIED. Reuse the existing 77-case binding, without new capture or borrowed GPU-pin remeasurement claims.
