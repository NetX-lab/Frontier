# P0 trace loading GC comparison

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded one separately bounded loading-GC comparison after the incomplete post-load freeze experiment. |

## Decision and scope
[Decision — primary under D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY]
The preceding one-variable freeze run still exits124 at60seconds. Loading takes47.129seconds;
completed gen2 scans take16.132583seconds, including14.162647during loading, collecting0objects.
The reader parses retained JSON trees; suspending cyclic GC does not change JSON decoding.
Test one new variable relative to that retained freeze run: disable cyclic GC only across loading,
freeze the completed tree as before, then re-enable GC before annotation/index/kernel work.
Keep JSON parser, predicates, events, inputs and all frozen source bytes unchanged.

Allocate <=40new scratch source lines, <=20control lines and <=80maintained Markdown lines.
First require exact decode-dp1 payload/kernel-index equality at60seconds/30GiB. Then run
at mostone P0rank0 whole-input comparison at the same limit, new outputs only. No additional
P0rank run, parser replacement, GC disabling outside loading, timeout increase or GPU operation.
Full output,>=3useful kernel-bearing steps and exact controls—not exit0 alone—settle the pilot.
Any later all-rank audit requires a separate recorded allocation and unchanged-output checks.
