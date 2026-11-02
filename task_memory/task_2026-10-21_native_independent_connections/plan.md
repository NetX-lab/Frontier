# Plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Established the minimal proxy-policy and CPU verification plan. |

## Scope and dependency graph

`A -> B -> C -> D`

- A: Inspect the base proxy/tests, historical local evidence, and available Python/aiohttp environment. Create the isolated worktree from `e55c83c` and task records.
- B: Add real-proxy CPU regressions in the existing proxy test file. Record a red baseline proving current connection reuse; verify disconnect request counts before the production change.
- C: Add only `force_close=True` to the existing connector. Run new regressions and focused legacy/native streaming suites. Independent environment inspection and historical evidence reads may run concurrently; edits and red/green verification are serial because they share the same files.
- D: Inspect the final diff, check route progress, record tests/limitations, reread requirements, and commit owned files. Return commit/diff/evidence without merge or push.

## Verification

- Preferred interpreter: `/data/ycfeng/envs/oneshot-opd-py312/bin/python` (inspect actual versions).
- Real localhost aiohttp servers retain transport objects, record request identities/bodies, and expose each TCP connection identity. No mocked connector verdicts and no native GPU claims.
- Buffered/streaming success: exactly two P and two D POSTs for two consecutive client requests; distinct transports for every upstream POST; bodies and response bytes preserved.
- Disconnect injection: close P or D transport before response headers, or D after a token event. Verify visible failure, no duplicate request IDs/POSTs, skipped D after P failure, and a successful subsequent request.
- Focused suites: `tests/unit/test_calibration_pd_harness.py` and `tests/unit/test_native_streaming_replay.py`; no external-checker or platform suite.
- Store temporary pytest artifacts/logs under `/data/ycfeng/tmp/opencode-v2-tmp/opencode`; retain compact evidence in this task directory.

## Done conditions

All four requirements' acceptance criteria verified within the CPU scope; one production-line change; no owned-file overlap; historical evidence unchanged; scoped commit and completion summary available. Native next-generation E2E remains the parent session's responsibility.
