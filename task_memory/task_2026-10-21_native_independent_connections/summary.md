# Completion summary

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archived W128-NATIVE-INDEPENDENT-CONNECTIONS implementation and CPU validation. |

## Overview

Implemented the approved fixed independent-upstream-connection policy using the existing aiohttp connector's `force_close=True`. No retry, workload, arrival, streaming timestamp, token-ID, metric-normalization, or simulator-core behavior was added or changed. Historical V4 failed-run artifacts remain preserved.

## Deliverables

- `tests/comparison/calibration/pd_proxy.py`: one connector-line replacement.
- `tests/unit/test_calibration_pd_harness.py`: six real-localhost proxy regressions, covering legacy/native buffered and native streaming success plus P/D pre-header and D mid-stream disconnects followed by healthy requests.
- `task_memory/task_2026-10-21_native_independent_connections/requirements.md`, `plan.md`, `progress.md`, `check_progress.md`, `report.md`, and `test_report_2026-10-21_native_independent_connections.md`: scope, acceptance, commands, evidence, and verification limits.
- `task_memory/task_2026-10-21_native_independent_connections/code.diff`, `evidence.json`, `pytest_red_connections.log`, `pytest_green.log`: exact code diff and selected actual CPU evidence.

## Observed validation

- Python 3.12.13; aiohttp 3.14.3; pytest 9.1.1 from the existing oneshot-opd environment.
- Red baseline: 5 failed connection-identity assertions, 1 passed, 62 deselected, 0.97s; POST-count assertions remained exact.
- Green focused suites: 96 passed, 28.82s; each upstream POST receives its own TCP transport; no duplicate retry after injected disconnects; payload/token-ID streaming and existing timestamp contracts pass.
- `git diff --check`: PASS. Only owned files are staged for the small scoped commit; no merge/push.

## Open and deferred work

- Native next-generation E2E validation belongs to the parent session; no GPU/platform action was authorized here.
- Actual historical failed-run connection age and server close timing remain unknown. CPU evidence demonstrates the approved connection policy, not a proved historical stale-connection root cause.
- The supplied prior diagnostic report path was not located; available raw local proxy/client logs were inspected without modification. Parent-reported D non-receipt and prior timeout-race reproduction were not independently verified.
