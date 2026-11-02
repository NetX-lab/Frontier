# P0 GC observation and one post-load freeze

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded same-limit GC telemetry and one-variable freeze experiment; both P0 audits remain incomplete. |

## Scope and commands
Primary executed `D-V4-P0-AUDIT-GC-OBSERVATION`, then the separately recorded
`D-V4-P0-AUDIT-GC-FREEZE`; see [commands](gc_command.md) and `gc_evidence.json`.
Python3.10.6, one P0rank0 input,60seconds/30GiB per invocation; no GPU operation.
Frozen auditor7ed0ea45/readerfe4c3f82 and the128-line stage observer remain unchanged.
New telemetry43lines/80cap; freeze26lines/30cap; exact control comparison12lines/20cap.

## Observed results
| Check | Telemetry baseline | Post-load freeze | Verdict |
| --- | --- | --- | --- |
| Completed decode control |142567rows/61useful steps|Exact payload/index equality|PASS control only|
| P0 exit / wall seconds |124 /60.374559|124 /60.374535|INCOMPLETE both|
| Decoded load seconds |44.459|47.129|No loading change/speedup proof|
| Completed gen2 seconds |15.835609|16.132583|Material measured GC cost|
| Completed post-load gen2 seconds |2.701915|1.942750|Reduced in this comparison|
| Partial complete JSONL lines |89878|363354|Not completed audit output|

GC and stage rows use the same monotonic clock. The baseline's collection starting
58.669870seconds was cut off by timeout; its duration is unknown. No completed
gen2collection occurs inside either retained kernel-loop interval. The freeze run
has no open final collection; its kernel loop continues toidx2275000 at59.789seconds.
Loading still dominates. One post-load freeze is not a complete timeout repair.

No complete gzip/final audit JSON exists for either P0run. Partial lines are counted
through the available gzip prefix and never admitted as kernel coverage. Prior errors,
limits and original outputs are preserved. No whole32-rank, dependency, ownership,
reference/cost admission, limit increase or GPU retry follows.
