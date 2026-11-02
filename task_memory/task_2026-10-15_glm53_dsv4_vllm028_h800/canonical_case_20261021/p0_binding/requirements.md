# P0 native workload and metric binding

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded W128-V4-P0-CASE-BINDING and its CPU-only ownership boundary. |

## [Original Request]
Task W128-V4-P0-CASE-BINDING: verify actual P0-H80032 workload and metric bindings from shared68f891b and ACCEPTED_NATIVE_REFERENCE_R4 native06 original data. Own only this directory and ticket17; no delegation, GPU/cloud/platform operations, source/package edits or Git/index/commit operations. Do not create another manifest or production checker. Keep ticket17 delta <=60 lines and its all-four-case criteria open.

## Acceptance and explicit decisions
- Parent D-FIXED-WORKLOAD: 128 formal requests, arrivals0..127s,6144/1024 tokens,786432/131072/917504 totals per repeat;16 disjoint same-shape warmups before each of THREE repetitions. Stable prompt template IDs and model-valid native prompt IDs; exact returned counts.
- Parent D-Q10-CLIENT-METRICS: means of client TTFT, request TPOT `(E2E-TTFT)/1023`, request E2E; total THPT `917504/(last formal last-token completion-first formal submission)`. Full432 records, no filtering.
- Parent D-CALIBRATION-PRIORITIES: this evidence is P0-H80032 only. Bind runtime/graph/quantization/memory only to admitted R4 receipts; retain unverified memory-budget/profiling/Frontier prerequisites explicitly. All-hardware32/64, independent10% and30% numerical acceptance remain open.
- Parent owns requirements/plan/progress and platform operations. This work is independent of queued32GPU profile02. Historical R3 FAIL/native05 UNVERIFIED and SharedGPT artifacts remain immutable.
