# Balanced routing phase reviews

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Original Spec reviewer closed both B P2s atc491e6a; root133PASS and bounded builder0832200 accepted,23/952. |
| 2026-10-23 | W128-V4-BALANCED-STATUS-SYNC: closed C runner status, recorded fixed39b2bf8 source/CPU/fidelity acceptance and current23/950 allocation; B corrections and package/GPU admission remain open. |
| 2026-10-23 | Archived B/C Standards review and three Spec P2 findings; assigned bounded corrections before package admission. |
| 2026-10-22 | Archived A shared-caller source APPROVE and independent primary hash/test verification. |

## Current review status (2026-10-23)

A186dfef is accepted after root218 checks, independent review and fixed-clean strict77, merged292c489. C8c2502c closes the runner-values P2, merged84c806f and finalized39b2bf8 with five files/202 lines. Root accepted C source/CPU/fidelity:184/184 tests with zero failures/errors/skips, merged A/C26PASS and strict77/77 identical with451 artifact entries. Evidence: `prefill_primary_fidelity.json` and `test_report_2026-10-23_prefill_acceptance.md`; this is not GPU or serving admission.

B c491e6a closes both schema-origin and effective-DP P2s: root133PASS, independent targeted APPROVE from the original Spec reviewer. Builder0832200 supplies the existing immutable commit and binds payload3600/replay1800/readiness32GPUh. Root's delegated two-line deadline correction gives the current23/952 ceiling, allocated A432/B318/C202. Combined-source verification, single-writer checker freeze and native06 package/image/runtime checks remain open. C passed package/image/installed-adapter checks and its approved H800 run Succeeded; local/cloud integrity passes while row acceptance proceeds independently.

GPU budgets remain prefill1H800/900s and native32H800/60min. GLM stays paused; workflow/GPU/operator-gap alignment precedes CPU-overhead calibration. Source review and fidelity acceptance do not change this phase order.

## A: historical source review, W128-V4-BALANCED-CALLERS-REVIEW

Reviewer: native reviewer session `ses_e9cec2592ffesauaz3LsPAE7kJ`. Reviewed full9d44675..186dfef5503ecb5329bd0fa96f9e978cc813276a diff in v4-balanced-callers, originating requirements, current approval/allocation, source inventory and test records. Verdict **APPROVE, source scope only**, no actionable P1/P2 findings. B/C WIP was excluded.

Verified by source review: all four resolvers carry the explicit selector; PDD/PD-AF reconstruction/copy preserve it; existing typed contract and serializer separate selected routing without parallel registry keys; producer uses the original routing timer; missing/wrong timing identity rejects; no unrelated guard expansion. Actual V4 FFN controls cover both training orders, cold/warm and public role views, with controlled standard2/balanced7 values. Ten default identities were recorded byte-equal to9d44675. Scope12 files/432 changed lines stays within A's450-line allocation.

Reviewer read root218PASS XML but did not run tests or independently execute Git/hash checks. Primary independently verified every changed working file against186dfef, the complete12/432 diff, unchanged policy SHA6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea and reviewed diff SHAf307d4957d048027f3c01fe3eb32edf1ce260428e5939e0f598dbc59e58ddc9f. Receipt: `callers_primary_source_receipt.json`. Primary's218-test run has0 failures/errors/skips; exact command/log/XML in `test_report_2026-10-22_primary.md`.

At this historical review entry, final186dfef strict77 capture/comparison was still pending. Historical410f062 fidelity could not substitute. The then-current23/900 union and source acceptance were separate from runtime activation, GPU timing/backend/capture, new-package freeze and native admission; current acceptance/allocation is recorded above.

Root subsequently accepted A and merged292c489 after exact final77-case checks; see `callers_primary_final_acceptance.json` and `test_report_2026-10-23_shared_acceptance.md`.

## Historical B/C joint review: fixed65743ea andac21f88

Artifacts: B `evidence/native_final_source.delta.patch`, C `source.diff`, both retained builder baselines, actual source and primary requirements/plan. Existing A source review was not repeated.

### Standards

Reviewer `ses_e9b6e20d3ffet3va8CzJ5pKQT5`, task `W128-V4-BALANCED-BC-STANDARDS`: no new actionable implementation-standard violations or bounded-scope refactoring recommendation. Final combined-source scenario evidence remains required by AGENTS.md799–801; use the existing strict77 matrix and fixed baseline rather than constructing a new broad suite. Known unfinished builder constants/timeouts were not classified as an incorrect launch claim.

### Spec

Reviewer `ses_e9b6e7bcfffevLPIR9BYvjfRmy`, task `W128-V4-BALANCED-BC-SPEC`: REQUEST_CHANGES, three P2 findings.

| Finding | Evidence | Required correction | Owner/status |
| --- | --- | --- | --- |
| Ambient Frontier schema import | R3 `verify_balanced_frontier` imports live `frontier.config`; B774/A775 preload difference already occurred | Bind actual schema import/origin to the pinned harness; retain full-flat validation | B, in-progress |
| Missing attention-DP/effective role check | R3 checks TP/EP/PP but not `replica_config_attn_dp`; rehashed DP1 input could pass nativeDP2 | Validate DP and reconstructed/materialized P/D topology; wrong-DP negative | B, in-progress |
| Wrong runner class admitted to CSV | Original `dsv4_attention_rows.routing_columns` compared runner keys only | Exact runner key/class mapping and direct row/consumer negative | C, closed by8c2502c targeted recheck below |

Both reviewers were read-only: no independent test execution, full hash recomputation, build or platform operations. They distinguished source/CPU reports from actual native GPU/backend/capture qualification. Root confirmed these corrections are on the existing acceptance path. At this historical joint-review entry, the874-line delivered total left26 lines under the then-current900 ceiling before final constants; this is not the current balance. Current23/950 allocation and expected937 total are recorded above. Owners still report exact minimum scope before exceeding their local allocations; optional new persistence must not displace mandatory P2 controls or expand a harness.

### C targeted recheck

`W128-V4-BALANCED-PREFILL-R1-REVIEW`, same Spec reviewer, APPROVE for8c2502c: exact runner names-to-MoERunner mapping closes the original P2; existing consumer converts the rejection to `DeepSeekV4CoverageError`. No new actionable finding in the narrow change. B's two findings remain independently open.

Root verified every one of C's five source files against8c2502c and recomputed190 cumulative routing lines. Exact report-command execution passed all five actual producer/consumer controls, including balanced and legacy positives; root affected XML records22PASS/0failure/0error/0skip. A warning-prefix parsing error occurred only in the outer receipt reader and was corrected by inspecting the retained output, without rerunning controls. C was then merged84c806f and finalized39b2bf8 at202 lines; root's184/184 and clean strict77 complete its source/CPU/fidelity acceptance. Evidence and exact commands: `test_report_2026-10-23_prefill_acceptance.md` and `prefill_primary_fidelity.json`. C package/image/runtime acceptance remains pending.

### B targeted recheck — c491e6a

Original Spec reviewer `ses_e9b6e7bcfffevLPIR9BYvjfRmy`, task `W128-V4-BALANCED-NATIVE-R1-REVIEW`: **APPROVE**, both earlier B table findings CLOSED, no new actionable finding. R3:48–98 requires a full immutable commit, fresh isolated interpreter, actual loaded origins within the declared harness and source bytes equal to that revision. R3:37–47,60–81 verifies attention-DP plus complete reconstruction/materialized P/D TP/DP/EP/PP/world/model/routing. Existing reader tests:496–535 retain rehashed-DP, ambient-schema and source-byte negatives. Six non-graph predicate AST controls and frozen R2 dependency remain unchanged. Reviewer inspected source/receipts without rerunning Git/tests; root separately read133PASS/zero failures/errors/skips from its own XML. Exact command and final builder checks are in `test_report_2026-10-23_native_primary.md`. This closes source findings only; native05 is not upgraded and native06 has not run.
