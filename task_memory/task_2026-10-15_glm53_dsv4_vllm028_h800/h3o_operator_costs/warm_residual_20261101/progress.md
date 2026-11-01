# WARM-RESIDUAL progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | Completed current-run CPU span attribution and uncertainty report. |

- completed: decisions D-WARM-TARGET, D-NO-NEW-MODELING,43/46 and prior A10 sections read.
- completed: six replay joins, native379/384 warm pools and lifecycle32 structural join.
- completed: exact current FIFO waits reproduced within0.103 ms; price-neighborhood689 ms estimate distinguished from fixed-price long-window mean.
- completed: two span tables and coarse error-budget closure; P2-32 clean–lifecycle handoff difference retained as -11.06 ms unallocated.
- completed: ranked causes and correction/validation proposals in test_report_2026-11-01_warm_ttft_residual.md.
- completed: scope checkpoint in checkpoint.md; its parent-doc recommendation returned to primary, not applied to primary-owned records.
- pending approval: any new mechanism, measurement or configuration change; not part of this read-only assignment.
- verification: validate.py checks populations, conservation, current FIFO errors and scratch-compatible smoothing; validation.json is the decisive receipt.
- inspection correction: config.json is under metrics/.../, not the replay root; discovery resolved the read error without changing files.
- no GPU, core/test code edits, commit, official gate rewrite or table promotion.
