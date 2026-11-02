# Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded baseline inspection and bounded CPU implementation scope. |

## A — completed

- Created branch/worktree `native-independent-connections` at `e55c83c263f86f48972f137858723a07c6f5e691`; initial `git status --short --branch` was clean.
- Inspected `pd_proxy.py`, existing legacy/native streaming proxy tests, repository guidance, glossary, and native-reference ADR. The actual connector is `TCPConnector(limit=0)` and both buffered and streaming upstream POSTs share its session.
- Inspected local historical `run/proxy.log:1`, `run/proxy_requests.jsonl:145`, and `run/client_requests.jsonl:145` beneath `/data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_results`. P returned 200; no decode status was recorded; the error is `ServerDisconnectedError: Server disconnected`; client status is 502. D not receiving this request and the prior 15s-client/5s-server CPU reproduction are supplied parent diagnostic context, not independently reverified here.
- The supplied relative `task/warmup_disconnect_20261021/report.md` was not located in the inspected worktrees/root. No continued broad search is needed to verify the approved connection policy; use the available raw local logs and disclose this limit.
- `/data/ycfeng/envs/oneshot-opd-py312/bin/python`: Python 3.12.13, aiohttp 3.14.3, pytest 9.1.1. Alternative asyncopd environment: Python 3.12.13, aiohttp 3.13.3, pytest 9.0.2. No dependency installation.

## B — completed

- Added six cases to the existing proxy test file, retaining transport objects and request counts at real localhost P/D HTTP servers. Three success cases cover legacy buffered, native buffered, and native streaming paths; three fault cases inject P pre-header close, D pre-header close, and D mid-stream close, followed by a healthy request.
- First ordinary pytest invocation could not collect because the environment's editable `verl` package supplies a regular `tests` package that shadows Frontier's namespace. Error: `ModuleNotFoundError: No module named 'tests.comparison'`. No packaging change is allowed or needed; a command-local `sys.modules['tests']` namespace bootstrap selects this worktree's existing test tree.
- Red baseline command/log: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-native-connections-red-namespace.log`, exit 1: **5 failed, 1 passed, 62 deselected in 0.97s**. Three consecutive-request cases observed 2 distinct transports rather than 4. D-header and D-stream disconnect cases observed 3 rather than 4 because P was reused. All exact POST-count assertions passed before those connection assertions. P-disconnect case already passed. This proves base connection reuse, not the native failure's connection age.
- Cause investigation phases beyond the connection-policy seam are intentionally omitted: the user explicitly approved the fixed no-reuse policy after the parent diagnosis. The CPU regression tests do not identify the historical native root cause.

## C — completed

- Applied only `force_close=True` to the existing connector to prevent upstream TCP reuse without changing POST control flow or timestamps. Gain: each approved P/D POST establishes an independent connection. The existing session/connector API already implements this; no flag or replacement path is required.
- Full focused verification log `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-native-connections-green.log`, exit 0: **96 passed in 28.82s**, running the entire `test_calibration_pd_harness.py` and `test_native_streaming_replay.py` files. All six added CPU cases passed, including exact POST counts and independent transports after failure.
- Inspected installed aiohttp 3.14.3 `client.py:256,703-704,867-873`: transparent persistent-connection retry is enabled only for `IDEMPOTENT_METHODS`, which excludes POST. The production proxy calls `session.post` once per reached phase and has no retry loop. No private retry setting was changed.
- `git diff --check` passed. Code diff contains exactly one production-line replacement and 169 added test lines in the existing proxy test file; no changes to arrivals, workload inputs, native observers, external checker, core simulator, or timestamp code.

## D — completed

- Reread `requirements.md` and inspected the exact owned-file diff. All CPU acceptance criteria are covered; native E2E is explicitly deferred to the parent, not represented as complete.
- Retained actual red/green logs, server connection/request-count summaries, selected historical records, and a code diff under this task directory. `evidence.json` records SHA256 for all five historical `run` files; re-reading them during extraction showed no changes. No historical file was written.
- Final route checkpoint records A–D complete with no scope drift. No additional tests or simulator matrix were added after the focused suite passed.
- The repository ignores `task_memory/*`; explicitly force-stage only this task's named files so the required records accompany the scoped code commit, without modifying `.gitignore` or staging other tasks.
- Initial staged whitespace check flagged blank context markers in the generated diff artifact, not code. Regenerated the exact artifact with `git diff --cached --unified=0` so it retains all edits without whitespace-only context lines; compressed JSON evidence formatting without dropping observations. Final staged whitespace check passes.
- Deliver the commit on branch `native-independent-connections`; retrieve its exact hash using `git log -1 --format=%H`. No merge, push, delegation, GPU/platform action, or other owner edits.

## Final acceptance

- Completed: fixed independent connection policy; six real-proxy CPU regressions; 96 focused tests; historical failure preservation; durable diff/evidence/task records; scoped commit.
- Failed attempt resolved: initial pytest collection shadowing by installed `verl.tests`, addressed only in the command-local bootstrap.
- Unfinished within this task: none. Outside this task: parent-owned next-generation native E2E and historical connection-age investigation remain unverified.
