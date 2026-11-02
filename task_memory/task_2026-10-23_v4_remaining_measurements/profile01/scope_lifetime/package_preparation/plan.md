# Package preparation plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Set the bounded preparation dependency and acceptance conditions. |

1. P1: Inspect inherited decisions, frozen source/imports, accepted templates and image receipts; pin inputs and budgets. No accepted source edits.
2. P2: Add the three allocated files; verify an in-memory candidate's five payloads, mirrored CPU imports, real limits reader and launcher argv under60s/30GiB. Compile image_check without claiming actual-image execution.
3. P3: Generate the official package exactly once with exclusive creation after P2 PASS; verify manifest closure and unchanged original inputs. Commit source plus generated package.
4. P4: Retain commands/diffs/numstat/hash inventory/budget/test report/checkpoint and English summary; commit records and STOP clean.

Dependency: `P1 -> P2 -> P3 -> P4`; no independent owner work or delegation. Root gates are downstream and outside this allocation.
Acceptance: five exact-byte payload copies; unchanged source_binding; repository-relative graph/calibration resolution; no `run` subcommand; explicit digest/elements/warmup/limits/diagnostics; inherited image and binary/ABI pins;1H800/600s/attempt1 recorded; all caps met; no platform/GPU/seal action.
Verification: asyncopd Python3.12 `-B`, at most two60s/30GiB CPU checks; report Python/version, exact stdin/commands, PASS/FAIL and CUDAfalse. Generation is not a retryable check. The root image script must fail on missing/wrong image pins, never substitute host binaries.
