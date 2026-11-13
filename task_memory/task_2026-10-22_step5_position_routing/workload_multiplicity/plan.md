# D29-FULL-WORKLOAD-MULTIPLICITY plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Plan the bounded source/binary evidence collection and the projection. |

## Scope

Read-only determination of decode-time EP-wave multiplicity, persisted trace
volume and projected wall time for the unchanged main08 workload on the
accepted source. No implementation, no new run, no GPU.

## Steps

| # | Step | Evidence source | State |
| --- | --- | --- | --- |
| S1 | Real argv and mode/workload facts | `d29-parent-position-main08/offline/{command,execution}.json`, `scheduler.jsonl` | done |
| S2 | Decode layer-advance and wave-entry call chain | `decode_collective.py`, `sync_entry.py`, `base_cluster_scheduler.py`, `ep_wave_schedule.py` | done |
| S3 | Per-wave event composition and trace persistence | `ep_wave_metrics.py`, `trace_store.py`, `metrics_store.py`, `metrics_config.py` | done |
| S4 | Layer set (MoE vs dense) from the real model config | `data/config/models/stepfun-ai__Step-5-Preview.json` plus a real `ModelConfig.create_from_name` count | done |
| S5 | Bounded proof in the retained trace | 8 MiB tail of main08 `op_traces.jsonl` | done |
| S6 | Per-wave phase timings | retained `parent-red01/result.json`, `wave-final-*/result.json`, `trace_loop.py` shallow run | done |
| S7 | Arithmetic: waves, events, bytes, floor/range, startup | this report | done |
| S8 | Identify the mechanism worth a future RED probe (no implementation) | this report | done |

## Verification

- Whole-file byte closure must reproduce the observed `op_traces.jsonl` size
  from the derived per-event byte size and event multiplicity; 1.448 GB is
  reproduced within 0.1 %.
- Two complete waves in the 8 MiB tail must show 16 lanes x 13 operators = 208
  events and ~2.905 MB.
- No arithmetic term may rely on a phase median that was not measured at the
  accepted per-wave seam; extrapolation factors are stated explicitly.
