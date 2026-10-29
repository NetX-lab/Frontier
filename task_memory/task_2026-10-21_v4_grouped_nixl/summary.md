# Grouped V4 structural NIXL

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Closed primary integration at2a84c58 after clean merged strict77/77 identical; retained exact primary receipts. |
| 2026-10-21 | Primary re-review and independent470/28 checks passed; merge2a84c58 is awaiting its clean merged strict result before publication. |
| 2026-10-21 | Completed source/CPU implementation and clean strict77 comparison; retained final receipts. |
| 2026-10-21 | Closed reviewed P2 metadata repair: actual callback RED/GREEN,470 focused PASS and clean strict77/77 identical. |

TaskID: W128-V4-GROUPED-NIXL. Branch/worktree: `v4-grouped-nixl`, based on6d88700.
Original implementation commit: `1a65e4985091f5c85c14830f86385800a614e4e6`.
P2 repair commit: `5428bd7803137cac1de5c691409534af1e68888e`.
Primary merge: `2a84c58`, accepted after independent P2 closure, 470 focused
checks, 28 combined checks and a clean merged strict comparison with all
77 cases identical. Failure, missing-evidence, provenance and cache-name
findings are zero. This closes B's bounded CPU structural integration.

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
- `primary_receipt.json`, `primary_fidelity_comparison.json`, `primary_focused.xml`,
  `primary_merged.xml`: primary acceptance, exact hashes and merged evidence.
- `native_retention.md`, `native_rules.json`: pinned native callsites and actual
  CPU source-body controls at191/6144/6145 tokens.
- `cpu_tests.xml`: **371 PASS**, including three complete V4 loops, actual FIFO
  release/readmission, invalid-group controls and unchanged GLM/context/P2P tests.
- `test_report_2026-10-21_grouped_nixl.md`: exact commands, outputs and limits.
- `source.diff`: cumulative zero-context production/test diff from6d88700,
  including P2. `p2_source.diff` isolates the repair from17e3b42.
- `cpu_receipt.json`: actual group, byte, context, output and free-pool observations.
- `fidelity_comparison.json`, `fidelity_receipt.json`: the unique clean-source
  strict77 run at1a65e49: **77/77 identical**,0 failures/missing/provenance/cache
  differences. Every per-case source stamp is clean; reference worktree retained.

Reusable checks: `tests/unit/test_dsv4_structural_pd.py` and
`tests/integration/verify_dsv4_native_nixl_rules.py`.

## Open / deferred

No unfinished item remains in this bounded CPU structural slice. Actual multi-owner
execution, graph/measured consumption (slice A), native wire descriptors/timing,
notification/readiness ordering, contention, formal128-request repetitions and
four-metric accuracy remain outside this CPU structural acceptance. The owner
performed no GPU, network, cloud, Docker, platform, authentication, delegation,
push or merge operations; primary integration is recorded separately above.

## P2 repair

The original fixture metrics missed a reachable start-callback failure. Actual
MetricsStore with op tracing reproduced the mixed-compression ValueError before
the fix. Grouped metadata now consumes validated P/D/selected snapshots and
existing component page sizes before scalar binding; MetricsStore only forwards
existing transfer info and scheduler block size. No model_binding guard changed.

`p2_red.xml`, `p2_green.xml`, `p2_focused.xml`, `p2_receipt.json`: RED1 failure,
GREEN1 PASS, focused470 PASS. Actual callback writes/reads TraceStore JSONL,
returns the end event, completes1024 outputs and frees both pools. An existing
GLM test compares every prior metadata field to the real callback. The retained
native record includes12 exact method bodies and9 callsite/spec excerpts.

The unique new clean strict77 candidate `candidate_v4_grouped_nixl_p2_5428bd7`
compared **77/77 identical** against the same absolute baseline, with zero failed,
missing, definition, provenance or cache identity findings. All77 source stamps
are clean at5428bd7. `p2_fidelity_comparison.json` preserves the exact comparator;
`p2_receipt.json` records all checked observations and external artifact paths.
The cumulative and P2-only diff artifacts exactly match that commit. The final
records-only commit leaves the measured frontier/tests trees unchanged. No P2
acceptance blocker remains; the bounded native/full-path limits above still apply.
