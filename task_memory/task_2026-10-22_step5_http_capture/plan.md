# D23-G3-HTTP-CAPTURE plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Defined narrow instrumentation and actual HTTP acceptance tests. |

`a source + clean worktree -> b red actual-proxy tests -> c Step-local safe capture -> d focused verification -> e commit + handoff`

1. Inspect actual source and existing adapter/tests; create clean requested branch
   from7fa73f91606841b069b5bd7cdb6ae410a93be8c8.
2. Add proportional local HTTP tests for missing D phase/failure traceback,
   healthy fresh/reused transport, unchanged body/headers/counts and non-Step
   receipt exclusion. Run before implementation to record missing evidence RED.
3. Add public TraceConfig callbacks and same-clock phase/times inside existing
   proxy. Bound events/chain/frames and redact diagnostic messages; isolate any
   auxiliary capture failure. No new module: source237 lines can own this seam.
4. Run affected existing proxy/canonical/client tests and new injected-failure
   checks. Native disconnect remains unresolved, original EOF remains502.
5. Review scoped diff without delegation; checkpoint, code commit, English summary
   and exact test report/receipt commit. Parent owns integration/fresh package.

Acceptance: actual failed D POST has phase/submit/error/public fresh-reuse signal
and bounded symbolic exception chain; callback failure cannot change success or
native failure. No retries/behavior change, no other-owner edits, clean branch.

Verification budget: CPU-only actual127.0.0.1 ephemeral HTTP; focused tests,
wall<120s and peakRSS<2GiB per suite command. Temporary pytest files/logs under
`/data/ycfeng/tmp/d23_step5_http_capture01/`; no GPU run or sealed-stage edits.
