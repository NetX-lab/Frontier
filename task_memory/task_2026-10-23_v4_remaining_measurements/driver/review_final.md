# G2 repaired-source independent review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Both independent R1 reviews closed all original findings with no new actionable findings. |

Component: G2 first-formal native profiling driver, collector window retention and the cold-start test fixture correction. Reviewed source: all seven paths in `source_manifest_r1.json` and `change_r1.diff`,433/750gross lines from313c8167. Initial artifacts and findings are retained separately.

## Standards

Reviewer `ses_e9841fef9ffePvnX6RHfChfqKD`, task `W128-G2-DRIVER-STANDARDS-R1`: **original2P2 CLOSED; No new findings**.

- `native_pd_worker.py:159–165` preserves CUDA/URI configurations and transforms only absolute local torch paths; incompatible auxiliary windows remain UNVERIFIED without blocking replay.
- `native_serve_profile.py:113–118` isolates archive emission, then establishes the new budget/window even if archival fails. The strengthened control proves retained steps, automatic stop and window snapshot.
- Also checked the Spec checkpoint-error repair and the16-line cold-start fixture change. No new documented-rule violation or actionable heuristic smell.

## Spec

Reviewer `ses_e984289acffe3o6TiR1NxOFgNA`, task `W128-G2-DRIVER-SPEC-R1`: **original2P2 CLOSED; No new actionable findings**.

- Startup archive failure no longer prevents the fresh window allowance.
- `pd_replay.py:426–435` routes checkpoint errors carrying the profiling receipt into its own `receipt_errors`, covering active/persistent post-stop failures. Unselected checkpoint behavior and request/idle/metrics validation remain intact.
- Also confirmed native CUDA/URI preservation and that the additional cold-start test stages/hash-binds both current modules while retaining existing admission assertions. No scope creep.

## Evidence and limits

Both reviewers inspected the complete seven-file change, current requirements/plan/design, scope amendment and RED/GREEN receipts. They reported source-review approval only; neither executed tests, Git, platform or GPU commands. Root independently rehashed the seven files and established equality with `review_green.json`, whose four suites separately pass32/19/39/18 with0errors/failures/skips.

Final required regression, matrix binding and commit acceptance remain root's next step. These reviews do not establish useful three-batch GPU coverage, native-reference admission, CUPTI/graph correctness or numerical qualification.
