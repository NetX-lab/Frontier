# Pricing implementation plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Start scoped interface-first implementation. |
| 2026-10-23 | Close owned delivery and distinguish peer/runtime acceptance boundaries. |

1. **completed:** Publish selected targets, ROUTED prepare binding, immutable plan/result.
2. **completed:** Extend isolated collective-sim with strict plan serialization, joint ordered
   channel lowering, actual HTSIM invocation and measured-parameter composition.
3. **completed owned seams / blocked peer acceptance:** Integrate source/gathered
   query contract, inclusive adapter and actual metrics/trace/ledger. CORE's newer
   exact estimator/layer/main implementations are not authorized/imported here.
4. **completed implementation / pending runtime inputs:** Close selected collector,
   prepare/validator/probe/exporter and qualified import. Native measurement,
   frozen device/network parameters and parent qualification are not fabricated.
5. **completed:** Verify scoped consumers,
   source methods and budget; commit and archive receipts.

Dependencies: approval -> {1, peer source/native APIs} -> 2 -> {3,4} -> 5.
Root integrates gitlink/peers and runs full-unit/strict77/GPU acceptance.
CPU structural/numerical tests are not GPU calibration. Published source-method
execution establishes only those methods' CPU behavior, not installed GPU runtime.
