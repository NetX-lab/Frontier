# Measured NIXL read input

The measured PD consumer uses an explicit `nixl_read_profile` from the execution-time predictor configuration. It has no default dataset. The input prices READ through GPU readiness, in milliseconds, for an exact prompt prefix and per-TP-rank payload. The existing analytical path continues to own structural timing.

## JSON contract

`frontier/kv_cache_transfer/nixl_read_cost.py` reads one hardware/layout profile:

| Field | Required value |
| --- | --- |
| `schema` | `nixl_read_cost.v1` |
| `model_architecture_profile` | Existing model registry profile ID |
| `profile_identity` | Nonempty identity of the frozen supplied profile |
| `source_device`, `target_device` | Exact replica device labels |
| `source_attention_tp`, `target_attention_tp` | Positive integer TP widths |
| `duration_scope` | `read_to_gpu_ready` |
| `cache_capacity` | JSON serialization of the existing `AttentionKVCacheCapacity` dataclass, including components and groups |
| `measurements` | Nonempty list of unique prompt-length records below |

Each measurement contains `num_prefill_tokens`, `num_computed_tokens`, `transfer_bytes_per_rank`, and `time_ms`. The computed prefix must be nonempty, shorter than the prompt, and aligned to the declared cache block size. Payload bytes are a positive integer; duration is positive and finite. Multiple raw repetitions must be judged and aggregated by their producer before creating a unique cost record.

The loader matches prompt length exactly. It checks both roles' architecture, devices and TP, plus the full static cache layout. The existing reservation-backed payload must match the recorded per-rank byte count before time is returned. Unknown lengths, another layout or hardware, incomplete readiness scope, and missing input raise errors. No interpolation or bandwidth substitute is used.

## Ownership and evidence

The predictor delegates its existing `get_nixl_transfer_coverage` and `get_nixl_transfer_time_ms` hooks to `NixlReadCost.get_coverage` and `get_time_ms`. The transfer event and scheduler retain ownership of reservation, readiness, source release, tail recomputation and emitted-output counting.

Schema validity is a consumable-input contract, not native qualification. A profile identity string alone proves neither measurement quality nor provenance. Formal canonical comparison still requires the case-bound profile/native receipts described by the task and `tests/comparison/e2e_gate/client_metrics.md`. CPU contract fixtures must be identified as fixtures; they cannot be published as qualified H200 observations.
