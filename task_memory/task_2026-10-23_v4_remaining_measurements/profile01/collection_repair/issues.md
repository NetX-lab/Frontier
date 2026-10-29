# Collection repair verification issues

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Preserved raw pytest whitespace after the records-only staged check reported two console-output lines. |
| 2026-10-24 | Recorded the original RED signals and the auxiliary verification snapshot interruption. |

## Source-backed RED controls — resolved

Unchanged worker SHA256 `9ec701ec12f7a5da0f442f994a353f98c5ae1d2880b9feca38106ea536fd5cab`, source base `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`:

- Role override assertions saw `[3,3,3,3]` instead of `[7,7,5,5]` or `[7,7,3,3]`.
- Six malformed role mappings did not raise `ValueError`.
- Natural and active headless marker20 controls observed `head_stopped_before_headless_stop=false` under stop10/peer60.
- With all four direct drain paths visible, stale enumeration20 yielded fake clock23 instead of expected0+3 (wait20 plus the stop3 stand-in).
- Adjacent timeout/native-error/partial-peer controls already preserved their business outcomes; they failed the newly requested timing assertions because the diagnostic field did not exist.

Original log: `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE03-COLLECTION/red_worker.log`; command and source-preservation evidence: `red_command.json`. Summary:19PASS/16FAIL. No production source was changed during RED. Post-repair whole worker suite:36PASS.

## Auxiliary final-check snapshot interruption — rerun with frozen inputs

The first separate-process batch completed all eight suites with passing JUnit summaries. Before its final snapshot assertion, the owner strengthened only `test_native_pd_worker.py` assertions for unchanged profiler fields and role overrides outside the local Torch window. The batch then raised:

```text
Traceback (most recent call last):
  File "<stdin>", line 24, in <module>
AssertionError
```

The failed assertion was `assert before == after`, after all suite processes completed. Production worker bytes did not change. The batch summary was not written, so the individual process exit codes were not retained and are not invented from JUnit counts. `verification_interruption.json` records the original commands, passing XML/log evidence, aggregate exit1 and this verification limit.

Correction: freeze both allocated source files, rerun the worker suite with before/after hashes, and rerun the seven other affected suites independently to retain actual exit codes and complete final records. The ledger is now written before any final exit decision so evidence cannot be lost by its final verdict. This is a temporary CPU verification command, not a production gate or new repository harness. No behavior fix or budget expansion was needed.

## Raw evidence whitespace — preserve bytes

The first records-only staged whitespace check exited2 before the commit. It reported `red_worker.log` lines23 and48, both original pytest console lines containing `E` followed by spaces. These are raw failure evidence, not code or maintained prose. The retained log remains byte-identical to the original temporary log; no trimming or replacement is performed.

The final ordinary staged check excludes only the two raw-format artifacts: `source_changes.diff` (separately validated against the base in a temporary Git index) and `red_worker.log` (separately checked against the original SHA256 and the exact reported lines). All Markdown, JSON and source contents remain under ordinary whitespace checks. `artifact_checks.json` records the original diagnostic and byte-identity validation. No business code or acceptance predicate changed.

## Remaining external acceptance

Root independent reviews, integration, actual-image/package checks and GPU/runtime verification remain outside this owner scope. Finite peer wait does not guarantee completion; local monotonic times do not establish cross-host order. No GPU or cloud failure cause is claimed.
