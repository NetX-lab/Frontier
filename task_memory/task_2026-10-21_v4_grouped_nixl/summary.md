# Grouped V4 structural NIXL

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded source/CPU implementation; post-commit fidelity comparison pending. |

TaskID: W128-V4-GROUPED-NIXL. Branch/worktree: `v4-grouped-nixl`, based on6d88700.

## Delivered

- Existing allocation manager accepts native zero-local-work external KV slots;
  existing P/D ownership, reservation, transfer and release owners now carry V4
  grouped structural payloads. Full source, target and selected transfer group
  snapshots remain separate, with no scalar grouped byte shortcut.
- `frontier/attention/memory.py` is unchanged: existing component-page accounting
  already expresses the payload. Graph/measured execution admission stays closed.
- At6144 input/1024 output, initial P2520 pages become46 held pages before
  exchange; D reserves46 pages and transfers39521664 B per worker lane. D starts
  at computed frontier6143 and releases all request pages after1024 outputs.

## Evidence paths

All paths below are relative to this task directory:
- `native_retention.md`, `native_rules.json`: pinned native callsites and actual
  CPU source-body controls at191/6144/6145 tokens.
- `cpu_tests.xml`: **371 PASS**, including three complete V4 loops, actual FIFO
  release/readmission, invalid-group controls and unchanged GLM/context/P2P tests.
- `test_report_2026-10-21_grouped_nixl.md`: exact commands, outputs and limits.
- `source.diff`: pending exact source/test diff snapshot before implementation commit.
- `fidelity_receipt.json`: pending unique clean-source strict77 comparison.

Reusable checks: `tests/unit/test_dsv4_structural_pd.py` and
`tests/integration/verify_dsv4_native_nixl_rules.py`.

## Open / deferred

Clean strict77 comparison and final receipt are pending. Actual multi-owner
execution, graph/measured consumption (slice A), native wire descriptors/timing,
notification/readiness ordering, contention, formal128-request repetitions and
four-metric accuracy remain outside this CPU structural acceptance. No GPU,
network, cloud, Docker, platform, authentication, delegation, push or merge ran.
