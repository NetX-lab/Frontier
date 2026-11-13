# D23-G3-HTTP-CAPTURE requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded parent approval for observability only and explicit behavior/ownership limits. |

## [Original Request]

Parent accepts D23-G3-DISCONNECT recommendation for OBSERVABILITY ONLY. Implement
narrow phase/submit/exception/connection reuse evidence in worktree
`/data/ycfeng/Frontier/.worktrees/step5-http-capture`, branch
`feat/step5-http-capture`, from7fa73f9. Preserve completed reports/sealed packages.
No GPU/platform/auth/cloud/SSH/push/delegation. Own
`tests/comparison/calibration/pd_proxy.py` and minimal nearby tests; no
r2_observation/r2_runtime/core/graph/input edits. No change to pooled connections,
timeout, retries, bodies/headers, arrivals, status or timing subtraction. There is
NO Step independent-connection decision.

Record failure phase, same-monotonic submit/error times, bounded redacted exception
chain/traceback frames. No token bodies/auth/secrets/locals/environment. Prefer
public aiohttp TraceConfig fresh/reuse signals; mark unavailable socket identity.
Capture failures are soft and cannot affect healthy HTTP or mask native failures.
Scope extra receipts to Step through the existing adapter classification.

Feedback: actual proxy EOF stays502 with correct D evidence; healthy stays200;
exactly1P/1D POST, no retries, original endpoints/tokens/request headers, strict
redaction, callback-failure isolation and distinct fresh/reused local connections.
Run focused existing proxy/canonical/client checks. Commit verified code and
maintained task records. Completion is diagnosis-capable instrumentation, NOT a
native-disconnect fix.

## Explicit decisions

- Use unchanged pooled `TCPConnector(limit=0)` and `ClientTimeout(total=None)`.
- Only add Step-local receipt fields; legacy and v0.28 receipts stay unchanged.
- Public trace events prove fresh/reused selection, not exact TCP socket identity.
- Preserve authoritative existing error string/status. Added diagnostic exception
  messages are redacted wholesale; traceback has symbolic frames only, no code,
  locals or full paths.
