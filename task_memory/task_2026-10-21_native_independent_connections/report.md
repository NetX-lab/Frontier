# W128-NATIVE-INDEPENDENT-CONNECTIONS report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded scoped implementation, actual CPU evidence, and native verification limits. |

## Outcome

`tests/comparison/calibration/pd_proxy.py:70` now creates its existing session with `TCPConnector(limit=0, force_close=True)`. This is the only production change. The fixed approved policy applies to every upstream P/D POST, including buffered and streaming forwarding; no flag, wrapper, retry, or new package is introduced.

Six real-proxy localhost CPU regressions in `tests/unit/test_calibration_pd_harness.py` verify server-observed TCP identity and exact request multiplicity. Baseline code failed five independent-connection assertions. After the change, the complete proxy and native-streaming suites passed: **96 passed in 28.82s** on Python 3.12.13 / aiohttp 3.14.3 / pytest 9.1.1.

## Requirements covered

1. Independent connections: two consecutive successful client requests create four distinct upstream TCP transports, on legacy buffered, native buffered, and native streaming paths. Failure followed by success uses a distinct transport for every reached POST.
2. No POST retry: exact server-received POST counts remain one per request/role. P and D pre-header closes remain 502; mid-stream close remains a visible truncated response. Next request succeeds without replaying the failed request. Installed aiohttp's persistent retry excludes POST, and proxy call paths contain no retry loop.
3. Protocol and measurement preservation: native pull metadata and legacy decode body/ID contracts remain intact. Streaming token-ID event bytes and DONE reach the client incrementally. Existing replay tests preserve first/last token timestamp semantics. No arrival, token length/count, timestamp, or normalization code changed; TCP setup remains inside native measured client TTFT.
4. Historical preservation: no V4 artifacts were edited or requalified. Local proxy/client records still report `fixed-6144-1024-req128-qps1-warmup-r000-repeat02`, P=200, absent decode status, `ServerDisconnectedError: Server disconnected`, client=502.

## Limits

- This CPU policy regression does not prove that stale reuse caused the historical native failure. Actual connection age and close timing were not captured. D non-receipt and prior 15s-client/5s-server reproduction were supplied by the parent; the referenced `task/warmup_disconnect_20261021/report.md` was not located here.
- No native GPU rerun, device-readiness qualification, platform action, external-checker edit, simulator matrix, merge, push, or delegation was performed. Next-generation native acceptance remains with the parent session.
- Ordinary pytest collection hit the environment's unrelated regular `verl.tests` package. A command-local test namespace bootstrap resolved it without packaging changes. Exact commands and results are in `test_report_2026-10-21_native_independent_connections.md`.

## Deliverables

- Production change: `tests/comparison/calibration/pd_proxy.py`.
- Regression coverage: `tests/unit/test_calibration_pd_harness.py`.
- Task records: this directory's `requirements.md`, `plan.md`, `progress.md`, `check_progress.md`, test report, and `summary.md`.
- Diff and evidence: `code.diff`, `evidence.json`, `pytest_red_connections.log`, `pytest_green.log`.
