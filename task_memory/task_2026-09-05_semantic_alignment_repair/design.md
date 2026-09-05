## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Captured the approved scheduler and topology design. |

# Design

One Frontier Replica represents one complete GPU pod. Shared-domain non-FFN roles own one logical request scheduler per `(replica_id, dp_id)` lane, while TP remains implicit in operator execution. MoE EP remains a collective participant cardinality. PD-AF `DECODE_ATTN` is intentionally full-stage with `replica_local_id=None` and `attn_dp=1`; `DECODE_FFN` keeps explicit EP lanes.
