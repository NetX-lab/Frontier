# D29-FULL-WORKLOAD-MULTIPLICITY summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Parent corrects clock origins and replaces hard-floor/slowdown claims with implementation-specific projections. |
| 2026-10-23 | Archive the multiplicity/volume determination and the projected floor. |

## Overview

Read-only determination of how many measured 208-operator EP waves and trace
bytes the unchanged main08 workload (one request, 6144 prompt / 1024 output,
PD disaggregation, TP8/DP2/EP16/PP1, op-level tracing on) must execute on the
accepted source, and the resulting current-implementation CPU estimate.

## Deliverables

| Path | Content |
| --- | --- |
| `requirements.md` | Request, constraints, questions, acceptance. |
| `plan.md` | Steps and verification shape. |
| `report.md` | Call graph, multiplicities, byte and wall arithmetic, evidence, limits. |
| `evidence.json` | Machine-readable counts, timings and projection inputs. |
| `progress.md` | Reads, the one resolver probe, derived checks. |
| `summary.md` | This archive. |

## Result

- The 208-event wave runs once per MoE layer, 88 times per decode token inside
  92 layer steps; dense layers `0,1,2,91` execute no wave.
- 208 events = 16 EP lanes x 13 non-zero operators (144 COMPUTE + 64 COMM),
  measured at 2,905,920 B per wave in the retained trace.
- Per decode token: 18,304 events, 255.7 MB. For 1024 tokens: 90,112 waves,
  18,743,296 events, 261.9 GB, plus 0.26 GB for the 88 prefill waves; ~524 GB
  for both modes.
- Persistence is not sampled or capped: all tokens x all layers are written.
  The per-request expansion quota only changes the `layer_id` field value.
- Projected per mode: 24-28 h (86.3-101.4 ks), 36-42x the 2400 s cap;
  post-repair startup carried separately at 0.84-1.29 ks.
- Holding current emission/flush costs fixed, zero pricing still projects
  8.7-9.1ks with the historical startup sensitivity. This motivates an output
  optimization; it is not an impossibility claim about future implementations.
- P runs6128prompt tokens; the first of1024 D batches replays16prompt tokens
  and commits one output token. Total1025layer sets produce90200MoE waves.

## Verification observed

- The offline file size is roughly consistent with5.66representative layer-set
  equivalents. Different row sizes prevent treating this as an exact census.
- Two complete 208-event waves were measured directly in the bounded tail.
- The MoE layer set was counted with the production resolver, not assumed.

## Open / deferred

- The iteration-4 slowdown claim is withdrawn: its1085s offset starts at the
  first scheduler event, approximately1112.345s after process launch. About
  202.789s remained before timeout, not1315s. No write stall is established.
- No re-measurement of the accepted 1.021 s pricing, the post-repair startup or
  a single-token end-to-end path; all are labelled as inputs or extrapolations.
- No allocation, source change or new run was requested or made.
