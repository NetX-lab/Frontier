# WARM-READING completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Archived the verified D-WARM-TARGET CPU deliverable. |

Task ID: WARM-READING.

Implemented a 100-line reusable analysis script at `tests/analysis/e2e_warm_reading.py`. It retains the unchanged client harness reading and its verbatim text/SHA-256 alongside repeat-specific paired warm readings, reusing the existing admission, client_reading, metric_values, metric_row and verdict functions. No simulator or gate behavior changed.

Canonical P2-64 has128 compatibility log lines forming16 eight-TP events:11 warmup events and5 formal events, all in repeat01. Formal requests r006/r017/r027/r035/r036 bind uniquely to their events and first-contact lane pairs. Canonical P2-32 has32 lines/four events, all in repeat01 warmup. Later canonical repeats have no handshake events. The identical server/inventory/proxy method exactly reproduces the lifecycle run's r002/r004/r006/r009/r053 and the wave run's r005/r017/r037 ground-truth sets.

Deliverables, relative to `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/warm_target_20261101/`:

- `identify_openers.py`, `openers.json`: source identity, clock anchors/windows, every event and source line, lanes, candidate and bound request IDs, ground-truth validation, source SHA-256.
- `p2_32_repeat_openers.json`, `p2_64_repeat_openers.json`: normalized per-repeat opener IDs consumed by the reusable script.
- `readings/B29_p2_32_warm_reading.json`, `readings/B29_p2_64_warm_reading.json`.
- `readings/pass1_all_p2_32_warm_reading.json`, `readings/pass1_all_p2_64_warm_reading.json`.
- `readings/pass2_all_p2_32_warm_reading.json`, `readings/pass2_all_p2_64_warm_reading.json`.
- `readings/*_empty_reading.json`, `readings/verification.json`: six exact no-op value reproductions.
- `readings/metrics.csv`:192 metric rows with units, absolute values, signed/absolute errors, signed/absolute relative errors, and PASS/FAIL.
- `verify_outputs.py`, `output_verification.json`: independently checked paired populations, pooled totals, source hashes, verbatim original bytes, arithmetic and verdicts.
- `test_report_2026-11-01_warm_reading.md`: complete opener and96-row paired metric tables, commands and limitations.
- `run_analysis.py`, `command_receipts.json`, command logs: exact execution and hashes.
- `checkpoint.md`: route self-check; all four scoped plan steps complete.

Observed validation: all six warm readings pass all four metrics in each of three native repeats and pooled (96/96 PASS). All-request readings remain unchanged, including three P2-64 repeat01 TTFT FAILs. Empty opener sets reproduce original pooled/per-repeat values exactly; explicit pooled totals sum both sides while preserving original ratio rounding. P2-32 retains384 request-repeat rows; P2-64 retains379 (123/128/128). B29 rows already existed and matched retained reading hashes. All selected source hashes and original reading bytes remained unchanged. Python3.10.6, CPU only, no confidence intervals, no new replay/normalization/GPU operations, no commit.

Root-level first-iteration reading JSONs and metrics.csv are retained unchanged for execution history and are superseded by `readings/`.

Open/deferred: warmup-only cohorts can have multiple candidate request/engine bindings because log resolution is1s; no such ambiguity affects formal exclusions. This population reading does not simulate counterfactual stronger-warmup scheduling and does not qualify every remaining input/domain. The primary agent owns the separately assigned calibration skill and parent task-record updates.
