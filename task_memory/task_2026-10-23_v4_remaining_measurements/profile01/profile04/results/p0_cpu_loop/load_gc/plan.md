# Loading-GC comparison plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Established the one-input hypothesis test and acceptance limits. |

Dependency: source -> exact decode control -> one P0pilot -> output inspection -> decision.
The source imports the unchanged43-line observer and128-line stage instrument.
Record the GC-disabled load interval, one freeze, GC re-enable, completed collections,
actual memory controller, exact command/source/input/output hashes and process exit.
Compare loading, later stages, full or partial index and useful-step count against the
retained freeze run. Faster loading is an observation only if this run directly measures it;
no extrapolation substitutes for completed P0/full32 qualification.
