## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Recorded execution order and acceptance gates. |
| 2026-09-05 | Added Replica-local CC backend materialization gate. |

# Plan

Execution chain: approved semantic scope -> co-location repair -> pd-disagg repair -> pd-af-disagg repair -> direct smoke validation -> issue/PR publication.

Acceptance gates:

- shared-domain roles admit positive `attn_dp` values;
- one non-FFN scheduler exists per `(replica_id, dp_id)` lane;
- PD-AF `DECODE_ATTN` retains full-stage `None` identity and `attn_dp=1`;
- unit tests and one-request dummy smoke pass for all three architectures;
- issue and PR are open on the target repository and linked.
- collective-sim and ASTRA-Sim analytical backends materialize one Replica pod
  and keep outer Cluster replicas out of collective participant dimensions.
