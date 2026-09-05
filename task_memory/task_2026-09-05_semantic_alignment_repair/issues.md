## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Recorded and resolved the PD-AF runtime blockers found by direct smoke testing. |
| 2026-09-05 | Resolved outer-Replica collective contamination in runtime backends. |

# Issues

Resolved: `_replica_load_tracker` used integer keys while the repaired update path unpacked `(replica_id, dp_id)` keys. PD-AF now tracks full-stage Replica keys as integers and maps them to `(replica_id, None)` scheduler entries.

Resolved: utilization metrics allocated no local meters for non-FFN DP lanes, while repaired events supplied DP lane IDs. Metrics now size non-PD-AF non-FFN local scopes by `attn_dp`; `DECODE_ATTN` continues to use full-stage meters.

Resolved: PD-AF return batches do not carry `decode_attn_original_dp_id`. The return path now treats the absent field as the required full-stage `None` identity and rejects unexpected non-`None` values.

Open: documentation contract tests still report two pre-existing README marker failures; they are unrelated to this runtime repair and remain outside the approved scope.

Resolved: collective-sim and ASTRA-Sim analytical runtime materialization
treated outer Cluster replicas as attention-DP and MoE-DP participants. Each
Cluster backend now materializes one Replica pod (`world_size=8` for the
validated `TP4 x DP2 x EP8` case); outer `num_replicas` remains scheduler
capacity and no workload-level cross-Replica collective is modeled.
