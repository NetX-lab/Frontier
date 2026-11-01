# Historical P2-64 TTFT decomposition

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Verified the existing scratch archive and linked the event-based warm reading and complete residual analysis. |

This directory preserves the eight initial CPU investigation scripts, their retained stdout, and the FIFO sweep result. `archive_manifest.json` records all 18 file hashes and comparisons with `/data/ycfeng/tmp/p2_64_ttft_decomposition_20261101`. Two previously applied archival adjustments were inspected: `fifo_sweep.py` uses its own directory for imports/output; `prefill_wait.py` omits an unused placeholder variable. No analysis was rerun during this archival verification.

The current comparison is [the warm-reading report](../warm_target_20261101/test_report_2026-11-01_warm_reading.md), with final numerical products under that directory's `readings/`. It binds canonical first-contact events using server logs, request ownership, and proxy identities. Earlier slow-handoff heuristics in this archive are historical diagnostics and do not select the comparison population.

The complete six-replay decomposition is [the residual report](../warm_residual_20261101/test_report_2026-11-01_warm_ttft_residual.md). Its price-neighborhood FIFO average is a sensitivity diagnostic, not a fixed-price long-run prediction. Formal warm readings use repeat-specific matching on both sides; the preliminary all-Frontier-request arithmetic here is superseded for gate reporting.

Archival check: the first write correctly refused an existing directory. A subsequent byte-identity check exposed the two archival adjustments above; their exact diffs were inspected before recording them. Existing scripts, stdout, and numerical products were preserved.
