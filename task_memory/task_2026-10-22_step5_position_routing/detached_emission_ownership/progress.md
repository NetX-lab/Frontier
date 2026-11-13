# Progress — D29-DETACHED-EMISSION-OWNERSHIP

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Completed the read-only ownership diagnosis and the records. |

## completed

1. Read the existing profile `baseline/emission.pstats` (208 events, 212,789 calls,
   0.122 s profiled against a 49.601 ms unprofiled median). Decisive caller rows:
   `to_dict` 192 calls / 0.0532 s cumulative, of which 64 from `cache_key`
   (0.0190), 64 from `ep_wave_metrics.py:121` (0.0180) and 64 from
   `op_trace_utils.py:303` (0.0170); `asdict` 208 calls / 0.0430 s (192 via
   `to_dict`); getters `moe_or_mlp_time_component` 16 / 0.0148 and
   `communication_time_component` 16 / 0.0195; `describe_step5_operator_tensors`
   144 / 0.006; `json.loads` 448 / 0.017 (384 via `to_dict`, 64 via
   `op_trace_utils.py:302`). Evidence: `evidence/profile_callers.txt`.
2. Confirmed the getters are called once per lane, not per event
   (`ep_wave_metrics.py:29-31`), and that `common`, `parallel_context` and the
   warm layer metadata are not re-derived per operator.
3. Found the single real duplicate: `op_trace_utils.py:303` and
   `ep_wave_metrics.py:121` call `to_dict()` on the same detached plan object
   (lines 87/121 select it). Bound: 64 × 47 µs = 3.0 ms gross minus a ~20 µs copy,
   i.e. 1.7–3.0 ms/wave, where 47 µs is calibrated from the measured H3 delta
   (9.11 ms over 192 calls).
4. Bounded the parse candidate from the retained artifact: 64 `count_receipt_json`
   parses of ~13.4 KB over 8 distinct values, plus 2141 B (1 distinct) and 1468 B
   (2 distinct) per `to_dict` call. Estimate ≈2.5 ms; unmeasured for the 13.4 KB
   strings.
5. Recorded alias/escape facts: within-lane sharing of plan, cost and
   `native_context`; per-event fresh plan/cost dictionaries independent of the
   ledger proof (`probe_body.py:83-92`); cross-lane independence
   (`diagnose.py:198-207`).
6. Wrote `report.md` (499 words), `evidence/source_refs.md` and
   `evidence/artifact_facts.json`.

## in-progress

None.

## pending

None. A bounded control exists on paper only; the parent decides whether to spend
allocation on a ≤3 ms expected gain.
