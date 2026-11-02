# Diagnostic DP sequence source review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded both independent source reviews and root closure of the restored-fixture verification gap. |

Source `bb50186e`, owner records `178c1123`, base `44f2ddb9`; inspected four-file change is156gross lines (18production/138tests). Authoritative requirements are parent `requirements.md`, diagnostic DP sequence allocation, including direct dummy execution and detached sampling context.

## Standards

Reviewer `ses_e95dda618ffedYE6fgSm8ztXb3`, task W128-DP-SEQUENCE-STANDARDS: **No findings**. Actual explicit-group getter follows successful native reduction; existing soft errors preserve native behavior. Worker/V2 reset covers early failure and direct dummy execution. Context selects the active dispatch counter or saved detached step counter as appropriate, before capacity admission. No new hook, base schema, collective or core interface.

## Spec

Reviewer `ses_e95dda60fffePmYNdrGdL3m4lG`, task W128-DP-SEQUENCE-SPEC: **No findings**. Required source-backed DP/native Worker and direct-dummy/stale-context controls are present. Frozen dispatch keys, base rows/dedup, hooks and calibration predicates remain unchanged.

Both reviews were independent source/record inspections, not test executions or hash reruns. Root subsequently ran the fixture-enabled five-suite battery and verified source/frozen/native/matrix hashes, AST boundaries and the32-rank baseline/candidate evidence independently (`root_verification.json`,94checks PASS).

## Verification limitation

Owner194PASS/21SKIP became root214PASS/1FAIL/0SKIP after restoring actual C26 and legacy fixtures. The remaining failure reproduces at baseline44f2ddb9 and candidate178c1123 with identical32 counter/hook dictionaries: the old positive groundtruth fixture omits dp_utils and selects an older exact-hook checker. This is retained as a baseline fixture limitation; no test, checker or native source was changed to hide it. Native cross-DP pairing still requires fresh collection.
