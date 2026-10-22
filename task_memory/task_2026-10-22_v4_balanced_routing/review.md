# Balanced routing phase reviews

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived B/C Standards review and three Spec P2 findings; assigned bounded corrections before package admission. |
| 2026-10-22 | Archived A shared-caller source APPROVE and independent primary hash/test verification. |

## A: shared callers, W128-V4-BALANCED-CALLERS-REVIEW

Reviewer: native reviewer session `ses_e9cec2592ffesauaz3LsPAE7kJ`. Reviewed full9d44675..186dfef5503ecb5329bd0fa96f9e978cc813276a diff in v4-balanced-callers, originating requirements, current approval/allocation, source inventory and test records. Verdict **APPROVE, source scope only**, no actionable P1/P2 findings. B/C WIP was excluded.

Verified by source review: all four resolvers carry the explicit selector; PDD/PD-AF reconstruction/copy preserve it; existing typed contract and serializer separate selected routing without parallel registry keys; producer uses the original routing timer; missing/wrong timing identity rejects; no unrelated guard expansion. Actual V4 FFN controls cover both training orders, cold/warm and public role views, with controlled standard2/balanced7 values. Ten default identities were recorded byte-equal to9d44675. Scope12 files/432 changed lines stays within A's450-line allocation.

Reviewer read root218PASS XML but did not run tests or independently execute Git/hash checks. Primary independently verified every changed working file against186dfef, the complete12/432 diff, unchanged policy SHA6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea and reviewed diff SHAf307d4957d048027f3c01fe3eb32edf1ce260428e5939e0f598dbc59e58ddc9f. Receipt: `callers_primary_source_receipt.json`. Primary's218-test run has0 failures/errors/skips; exact command/log/XML in `test_report_2026-10-22_primary.md`.

Final186dfef strict77 capture/comparison is still pending at this entry. Historical410f062 fidelity cannot substitute. Runtime activation, GPU timing/backend/capture, complete23/900 union, new-package freeze and native admission remain separate root gates.

Root subsequently accepted A and merged292c489 after exact final77-case checks; see `callers_primary_final_acceptance.json` and `test_report_2026-10-23_shared_acceptance.md`.

## B/C joint phase: fixed65743ea andac21f88

Artifacts: B `evidence/native_final_source.delta.patch`, C `source.diff`, both retained builder baselines, actual source and primary requirements/plan. Existing A source review was not repeated.

### Standards

Reviewer `ses_e9b6e20d3ffet3va8CzJ5pKQT5`, task `W128-V4-BALANCED-BC-STANDARDS`: no new actionable implementation-standard violations or bounded-scope refactoring recommendation. Final combined-source scenario evidence remains required by AGENTS.md799–801; use the existing strict77 matrix and fixed baseline rather than constructing a new broad suite. Known unfinished builder constants/timeouts were not classified as an incorrect launch claim.

### Spec

Reviewer `ses_e9b6e7bcfffevLPIR9BYvjfRmy`, task `W128-V4-BALANCED-BC-SPEC`: REQUEST_CHANGES, three P2 findings.

| Finding | Evidence | Required correction | Owner/status |
| --- | --- | --- | --- |
| Ambient Frontier schema import | R3 `verify_balanced_frontier` imports live `frontier.config`; B774/A775 preload difference already occurred | Bind actual schema import/origin to the pinned harness; retain full-flat validation | B, in-progress |
| Missing attention-DP/effective role check | R3 checks TP/EP/PP but not `replica_config_attn_dp`; rehashed DP1 input could pass nativeDP2 | Validate DP and reconstructed/materialized P/D topology; wrong-DP negative | B, in-progress |
| Wrong runner class admitted to CSV | `dsv4_attention_rows.routing_columns` compares runner keys only | Exact runner key/class mapping and direct row/consumer negative | C, in-progress |

Both reviewers were read-only: no independent test execution, full hash recomputation, build or platform operations. They distinguished source/CPU reports from actual native GPU/backend/capture qualification. Root confirmed these corrections are on the existing acceptance path. The874-line delivered total leaves26 lines before final constants; owners must report exact minimum scope before exceeding their revised allocations. Optional new C persistence must not displace mandatory P2 controls or expand a harness.

### C targeted recheck

`W128-V4-BALANCED-PREFILL-R1-REVIEW`, same Spec reviewer, APPROVE for8c2502c: exact runner names-to-MoERunner mapping closes the original P2; existing consumer converts the rejection to `DeepSeekV4CoverageError`. No new actionable finding in the narrow change. B's two findings remain independently open.

Root verified every one of C's five source files against8c2502c and recomputed190 cumulative routing lines. Exact report-command execution passed all five actual producer/consumer controls, including balanced and legacy positives; root affected XML records22PASS/0failure/0error/0skip. A warning-prefix parsing error occurred only in the outer receipt reader and was corrected by inspecting the retained output, without rerunning controls. Evidence and exact commands: `test_report_2026-10-23_prefill_acceptance.md`. C is accepted for source integration; package/image/runtime acceptance remains pending.
