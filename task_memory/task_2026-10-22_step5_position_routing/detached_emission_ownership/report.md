# D29 detached emission ownership report

| Date | Change |
| --- | --- |
| 2026-10-24 | Read-only ownership-boundary diagnosis from existing profiles; no new measurement. |

TaskID `D29-DETACHED-EMISSION-OWNERSHIP`; session
`ses_e96a5c1c7ffeoPBK7V08CMas7v`. Frozen source `75404143`. No production edit, campaign or allocation.

## Verdict

**No materially useful safe seam.** After the required getter snapshots, each remaining
derived object is per-event output a tested alias boundary keeps independent. The
largest removable duplicate bounds at 1.7–3.0 ms/wave against the 39.956 ms
diagnostic baseline (≤7.5%), far from the 12–17 ms whole-output target.

## Ranked hypotheses

**H1 — second traversal of detached getter data: largely absent.** Emission
deep-copies nothing itself. Every `deepcopy` in the wave is one of the 32 required
getter calls (`execution_time.py:734,781`) or `asdict`'s leaf copies inside
`to_dict`. `ep_wave_metrics.py:29-31` calls each getter once per lane and reads
`physical_frame` as a field.

**H2 — per-lane common metadata repeated per operator: absent in the step5 path.**
`common` is built once per lane (`ep_wave_metrics.py:52-64`) and merged per event by
shallow `**common`; `parallel_context` is six integers; layer metadata is
warm-cached.

**H3 — residual is required reconstruction: confirmed.** `to_dict` runs 128 times
after the key memo (64 meta + 64 ledger) and must return fresh lists and dicts.
`asdict` costs 47 µs per call (measured: removing it from 192 calls saved 9.11 ms):
output construction, not duplicate copying.

## The one real duplicate

`op_trace_utils.py:303` and `ep_wave_metrics.py:121` derive identical dictionaries
from one already-detached plan, since lines 87 and 121 select the same object
(`ctx.collective_plans is communication.plans`). A private per-lane prepared dict,
handed to the ledger and copied for the trace, removes one derivation:
64 × 47 µs = 3.0 ms gross minus a ~20 µs copy, giving **1.7–3.0 ms**. It needs a
prepared-dict field on `OpTraceContext`, whose consumer `compute_op_trace_meta` has
four production call sites and three test files.

## Second candidate (not ownership)

`op_trace_utils.py:302` parses a 13.4 KB `count_receipt_json` per collective event
to read one field: 64 parses, 8 distinct values. `to_dict` also parses 2141 B
(1 distinct) and 1468 B (2 distinct) receipts per call. Bound ≈2.5 ms, but that is
an exact-value memo of a scalar, not a transferred subtree, needing its own control.

## Alias and escape evidence

Getter snapshots are detached per call, yet one lane's events share its plan,
cost and `native_context`. Emitted plan and cost dictionaries are
fresh per event and independent of the ledger proof (`probe_body.py:83-92`);
cross-lane maps are independent (`diagnose.py:198-207`). A shared prepared dict
would break that tested independence, so one consumer must receive a copy.

## Next control

Build the per-lane prepared dict, give it to the ledger, deep-copy it for the
trace meta, and rerun the retained probe (208 events, JSONL/ledger bytes, 64 wrong
keys, 3902 B prefix). Expected ≤3 ms; I recommend not allocating.

## Limits

Parse figures come from aggregate profiled `json.loads`, unmeasured for 13.4 KB
receipts. The 39.956 ms baseline is diagnostic only; its key source is absent.
Parity is the canonical 208-event wave, never a public-domain proof; the 2400 s CLI
still fails.
