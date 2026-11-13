# D29 post-named-query final parent audit requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record the independent final receipt audit, immutable source, raw gates, and records-only ownership. |

## [Original Request]

Task: D29-POST-NAMED-QUERY-FINAL-PARENT-AUDIT. Independently verify the completed
owner receipts, raw JUnit deltas, decisive strict evidence, and final
source/input/script bindings. Publish a compact parent acceptance receipt,
reproduction script, and report. No tests/probes rerun, output masking, source
edits, delegation, GPU, network, or platform operations.

Owner: `/data/ycfeng/Frontier/.worktrees/d29-post-named-query-regressions`;
records commits `0a42e56ef53769a85a999b652cd9d03973d1f727` and
`39f0c1000a96c10f186cabe5f77dfc3058f86549`.
Frozen source: `dbe19f531ff68d9edafc480418e8927c40eb1c60`, tree
`6c43090097e92c6fc0c33aa7358e69c2cf092705`, at
`verify-step5-d29-integrated-08`, with submodule `3dac53c`.

Reported hypotheses to check, not assumed observations: strict 77/77 with
451 artifact hashes and 519 cache filenames per side; canonical
8028 PASS / 279 FAIL / 32 ERROR / 281 SKIP, actual exit 1. All 46fe IDs/statuses
must be retained plus eight genuinely new passing IDs. Against 79, preserve the
one approved missing old ID and eleven new passing IDs, with no shared-status
or failure/error/skip-set delta. Do not alias IDs or sum overlapping suites.
The independent 101/197/53/52 shorts already have primary admission through
27 hashes at `86b9923f`; do not overwrite that partial acceptance.

Read root requirements/plan, owner docs/evidence, and existing post-key parent
acceptance mechanisms. Reuse existing comparators where appropriate. Independently
stream-hash listed source/input/script artifacts, without pandas CSV reads or
large-trace parsing. Every executable inspection uses timeout 60s, MemoryMax 30G,
runuser brainpp, and Python -B. Scratch belongs under `/data/ycfeng/tmp`.

## [Ownership / Acceptance]

- Records worktree: `/data/ycfeng/Frontier/.worktrees/d29-post-named-query-final-parent-audit`.
- Branch: `d29-post-named-query-final-parent-audit`, based on
  `86b9923f8c29f1c58ba9f8df23a396c70a4d1caa`.
- Write only this `resumed_performance/post_named_query_final_parent/` subtree.
  Preserve primary/index, owner, frozen source, partial acceptance, and root docs.
- Confirm 2755 source-file, 14 input, 12 script bindings and 124/120 scope bindings;
  source 9565/global 9579 remain accounting-only totals.
- Commit the owned subtree. Parent merges owner/audit and updates current
  acceptance/root docs separately. Whole CLI and native acceptance remain open.
