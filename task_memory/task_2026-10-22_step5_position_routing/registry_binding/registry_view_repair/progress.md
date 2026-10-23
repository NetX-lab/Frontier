# D29 registry mixed-view repair progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Four-case RED/GREEN and217 affected checks complete; exact incremental29/58 and total331/360. |
| 2026-10-23 | Confirm isolated branch and root allocation; prepare tight same-name RED. |

TaskID: D29-REGISTRY-VIEW-REPAIR; source8f96408c, recordsb6714757.

- completed: inspected actual family/cluster view entry points and the shared
  family resolver. Root's diagnosed P2 is the only in-scope defect.
- completed: actual fitted native gate/non-native coexistence RED in
  both insertion orders and legacy/non-native typed domains. Diagnosis phases2–4 are shortened because root already
  provided independently reproduced exact cause; persistent RED remains required.
- completed: add `include_native=True` to the existing family resolver; only the
  bare-name view projection requestsFalse. Native hash enumeration is untouched;
  public get calls retain the defaultTrue and strict ambiguity behavior.
- completed: RED02 four failures/zero errors in36.54s; GREEN01 four passed in35.85s.
  Failure is the exact mixed-domain family-view ambiguity, not setup or fitting.
- completed: final01 217 passed/zero failures/errors/skips in213.85s: original213
  acceptance checks plus the four coexistence controls. Public family and real
  cluster views retain both entries; explicit native lookup still resolves.
- completed: exact source budget3added/0deleted and test26added/0deleted,
  incremental29/58; total331/360 vsd96b3279. Registry96/120, test200/200,
  predictor113/118 and workload136/145 vs1aee0af. Caller delta35/60 unchanged.
- completed: verified implementation and acceptance; local records accompany the
  separate repair commit on `feat/step5-registry-bindings` for root re-review.
  Root-owned gitlink is excluded. Exact commit receipt is in the session handoff.
- pending(root-owned): independent re-review/integration and global gates.

The legacy-domain test registers before attaching the existing typed marker;
cluster lookup therefore validates its real matching typed identity without a
new fallback. Typed-domain control attaches the marker before registration.
Both use an actual trained native FP32 gate, real public views, and the existing
Step cluster resolver. Model hashes and global typed serialization are unchanged.

An isolated operator-import diagnostic encountered a pre-existing circular import;
normal `frontier.config` import order succeeded. No import/dependency behavior
was changed or pursued; this is not evidence of a registry-view failure.
