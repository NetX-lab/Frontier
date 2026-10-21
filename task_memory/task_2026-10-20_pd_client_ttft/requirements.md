# D5 T1: PD client TTFT

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-20 | Recorded the parent's finish-to-commit instruction and exact known-failure reference. |
| 2026-10-20 | Recorded the delegated D5 T1 implementation scope. |

## Request and decisions

[Original Request, verbatim] Implement the D5 T1 alignment for the Frontier LLM inference simulator: for prefill/decode disaggregated (PD) execution, the client-visible TTFT must end at the D-side first token instead of at P prefill completion. Write code and tests; no GPU job.

[Original Request] Implement D5 T1: for offline/online Step and existing GLM/DeepSeek PD, client-visible TTFT ends at the D first output token, not P completion. Preserve non-PD behavior and historical prefill latency. Write code/tests, run structural PD cases and the 77-case fidelity matrix against `8585748`. No GPU job. Do not edit scheduler, cluster-scheduler or ticket-20 driver files. Commit only owned files. Update the parent task's progress and ticket 14/17 notes.

D5 evidence: parent task `task_2026-10-15_step5_preview_h800/test_report_2026-10-20_native_pd_h800_r1w_preparation.md`, H800 execution result; `/data/ycfeng/tmp/step5_native_pd_h800_r1w/verdict02.json`. The measured T1 share is 6.7–62.0%, median 20.5%, over sixteen requests, exceeding the parent's 1-percentage-point bound.

The existing public `ttft` / `ttft_ms` comparison contract remains arrival to P-prefill completion. Add explicit `client_ttft` through the existing metric distribution/CSV path for `pd-disaggregation` only. No gate rule change, host-time estimate or T2 tail pricing is authorized here.

Inspected timestamp correction: `first_decode_token_completed_at` is the first emitted output in PD; `decode_first_token_completed_at` is stamped when output count crosses two, so using it would select the second output. Preserve both historical timestamps.

[Original Request, follow-up, verbatim] Your last turn ended with an interim status ("GLM comparison still running; then fixture fixes, full unit rerun and commit"). Continue to the end without stopping for status updates: finish the GLM/DeepSeek PD comparison, fix the fixtures, run the affected unit tests, the full unit suite (compare failing IDs with the 132 known pre-existing IDs in TASK/test_report_2026-10-20_stream_consolidation.md), the structural PD cases and the 77-case legacy fidelity comparison against 8585748, then commit and deliver everything listed in the original task (commits, before/after semantics, counts, ticket 14/17 status, progress entry, open issues). Return only when the work is committed or blocked.
