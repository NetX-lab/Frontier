## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Captured the approved scheduler and topology design. |
| 2026-09-05 | Added the Replica-local collective backend boundary. |

# Design

One Frontier Replica represents one complete GPU pod. Shared-domain non-FFN roles own one logical request scheduler per `(replica_id, dp_id)` lane, while TP remains implicit in operator execution. MoE EP remains a collective participant cardinality. PD-AF `DECODE_ATTN` is intentionally full-stage with `replica_local_id=None` and `attn_dp=1`; `DECODE_FFN` keeps explicit EP lanes.

Each Cluster `CCBackend` models the rank space of one complete Replica pod. The
outer Cluster `num_replicas` value controls scheduler capacity only; it never
multiplies attention-DP or MoE collective dimensions. Thus `TP4 x DP2 x EP8`
materializes as local attention `TP4 x DP2` and local FFN/MoE `TP1 x EP8`.
