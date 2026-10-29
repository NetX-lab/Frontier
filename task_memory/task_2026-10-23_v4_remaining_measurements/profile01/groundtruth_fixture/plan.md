# CPU fixture restoration plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Defined the source-bound RED/GREEN sequence and immutable checks. |

Dependency: `instructions/source/input verification -> original RED -> bounded fixture repair -> {full groundtruth suite, V2 inventory suite} -> immutable/matrix checks -> source commit -> records commit -> STOP`.

1. Verify clean base and original failure, using only existing legacy/C26/native inputs. Preserve exact commands and logs.
2. Reuse real frozen R3 classifier/dispatch bodies through CPU stand-ins and explicit DP group; select R3 only for V2 evidence checks. Preserve V1 and legacy controls.
3. Require the real32-rank positive fixture and all retained negative/oracle tests to pass, with no skipped tests.
4. Rehash frozen dependencies, runtime modules, native source and sealed profile04 inputs; compare simulator/input domain trees with the existing77-case capture.
5. Check gross source/Markdown budgets, commit source and records separately, verify clean worktree. Root review and integration remain pending.

No independent agent work is authorized. The two suites may run in independent CPU processes; all writes remain in the owned records or approved local temporary storage.
