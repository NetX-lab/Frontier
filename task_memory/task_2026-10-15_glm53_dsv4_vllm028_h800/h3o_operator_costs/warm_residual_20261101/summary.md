# WARM-RESIDUAL completion

## Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | Archived the completed CPU-only warm TTFT attribution. |

All assigned analysis deliverables are complete. No GPU, core/comparison edits,
commit, input promotion or official gate rewrite occurred.

- Report: `test_report_2026-11-01_warm_ttft_residual.md` in this directory.
- Exact replay/native spans: `summary.json`, `span_tables.csv`, per-request JSON rows.
- FIFO/sensitivity evidence: `sweep_summary.json`; observed wait errors<=0.103 ms.
- Validation: `validation.json`,9/9 checks PASS; command/environment record in report section5.
- P2-64 residual is31.1–31.8 ms under the scratch-compatible689 ms price-neighborhood average; deterministic replay alignment explains the16 ms cross-table change.
- P2-32 clean P-side shortage is13.4–13.8 ms; fine host/device attribution is incomplete. Its lifecycle handoff exceeds clean by11.06 ms, so a13–14 ms clean KV deficit is not established.
- Existing K19, eager EP aligned-call pricing, READ/depth2 and ingress/handoff fields already model the formerly missing terms. Host-zero, late handles and egress remain candidates; new mechanisms require YC approval.

Task WARM-RESIDUAL; session `ses_e68fcce2effep415tOL36M7v4g`.
Primary owns subsequent warm-gate integration, decisions and any GPU authorization.
