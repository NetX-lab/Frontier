## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Recorded user-approved semantic-alignment repair scope and code-change marker. |

# Approved Repair Scope

- **Case:** `issue26-ttft-calibration`
- **User authorization:** User confirmed the revised repair design: restore the
  legacy `(replica_id, dp_id)` per-Replica scheduler lanes, retain Frontier's
  logical TP abstraction, and represent MoE EP8 as shared collective
  cardinality without adding per-rank identities.
- **Branch:** `feat/semantic-alignment-repair-20260905`, based on `main`
  commit `3b564fe1`.
- **Ordered phases:** `co-location -> pd-disagg -> pd-af-disagg`.
- **Scope:** Config admission/materialization, per-Replica DP scheduler
  construction and request mapping, DP batch synchronization, shared MoE EP
  accounting, and predictor/communication assumptions that reject valid
  `attn_dp>1`. Preserve existing `replica_id` and `dp_id` identities and the
  logical TP abstraction.
- **Out of scope:** Groundtruth rerun, timing comparison, legacy H800 CSV
  reuse, per-TP-rank schedulers, per-EP-rank schedulers, and unrelated release
  refactors.

## Code-change marker

`SEMANTIC_ALIGNMENT_REPAIR_START: legacy Replica-local DP lanes with logical TP and shared EP cardinality`

The marker is recorded immediately before the first Frontier source edit.
