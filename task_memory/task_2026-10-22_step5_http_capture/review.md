# Step HTTP capture scoped review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Self-reviewed owned diff against parent observability-only scope. |

Reviewer: D23-G3-HTTP-CAPTURE executing agent (no delegation permitted). Inspected
base7fa73f9, changed`pd_proxy.py`, new`test_step5_http_capture.py`, requirements
and plan; final focused execution reported separately. This is an internal scoped
review, not parent acceptance or proof of native correctness.

## Findings and evidence

- Scope: only proxy/test plus task records; no r2/core/graph/input edit. Proxy
  grows237→418lines, well below2000; no diagnostic module or extra CLI is needed.
- Existing HTTP semantics: original connector`TCPConnector(limit=0)`, timeout
  `ClientTimeout(total=None)`, POST paths, data bytes and two explicit headers
  retained. No retry introduced. Existing502 and streaming re-raise remain.
  `except BaseException` adds evidence and immediately re-raises unchanged.
- Classification: Step registry record selects capture; legacy/v0.28 receipt
  exclusion checked through actual HTTP. No version ordering ladder.
- Trace callbacks use public`TraceConfig`only, never connector/socket internals
  or callback params. Record fresh/reused/fresh_connecting/unknown; socket ID
  explicitly unavailable. Event timestamp shares`time.monotonic`with submit/error.
- Redaction: added exception messages always`[REDACTED]`;4exceptions with at most
 12 symbolic module/function/line frames,128character symbols;16events per P/D,
 4capture errors. No token body/header/env/local/traceback source or full path.
  Existing authoritative error/status/kv_transfer_params remain unchanged.
- Soft capture: constructor/setup/callback/frame extraction failures are isolated
  and mark capture UNVERIFIED; original HTTP continues. Warning messages report
  only operation/type, at most first failure per receipt. No native error masked.
- Phase: request JSON, P/D response headers/body, handoff validation, existing
  cancel-control, downstream prepare/write/eof differentiated. Received HTTP
  errors without an exception keep original status; no fabricated exception.

## Corrections made during verification

The newly added direct-public-trace redaction test accidentally retained an EOF
assertion after test insertion (HttpCapture object treated as dict). Source-backed
test-only correction moved the assertion back to the actual EOF test; retained
`capture17.log`records16PASS/1FAIL before correction. Production code unchanged.

No actionable in-scope behavior defect remains from the inspected diff. Native
disconnect cause, exact socket identity and deployment image aiohttp capability
remain outside local verification. Parent owns integration and any fresh package.
