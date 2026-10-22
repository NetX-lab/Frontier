# Pricing implementation plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Start scoped interface-first implementation. |

1. Publish selected targets, ROUTED prepare binding, immutable plan/result.
2. Extend isolated collective-sim with strict plan serialization, joint ordered
   channel lowering, actual HTSIM invocation and measured-parameter composition.
3. Integrate source/gathered queries, inclusive adapter and metrics after peers
   deliver immutable source/materializer/provenance APIs.
4. Close selected collector/prepare/validator/probe/exporter and qualified import.
5. Verify scoped consumers and budget; commit and archive receipts.

Dependencies: approval -> {1, peer source/native APIs} -> 2 -> {3,4} -> 5.
Root integrates gitlink/peers and runs full-unit/strict77/GPU acceptance.
CPU structural/numerical tests are not GPU calibration or native source proof.
