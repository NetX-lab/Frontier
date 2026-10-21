# W128-V4-GROUPED-NIXL (B)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded the delegated approved CPU-first grouped NIXL slice. |
| 2026-10-21 | Added reviewed P2 real MetricsStore/op-tracing metadata repair. |

## [Original Request]

Implement approved Q10 slice B from Frontier 6d88700 in its own worktree/branch.
Authority: parent task requirements D-Q10-V4-GRAPH-MEASURED-AND-NIXL, approved
2026-10-21. First establish native group retention and connector selection at
2cf0a6915ce544dc493a0990f2ea38d81601128a; source-held pages, transferred pages,
and destination reservations are different quantities. Reuse existing grouped
allocator and capacity byte accounting; preserve full prompt handoff, all 1024
outputs, prompt-1 frontier, GLM 96 pages / 569106432 bytes, PP1/no prefix/no MTP,
capacity checks and unsupported-path failures. Leave graph/measured execution
guards untouched. No GPU, platform, auth, SSH, cloud, network, Docker, delegation,
push or merge operations.

## Acceptance

1. Native source references and reachable CPU controls for each group.
2. Real scheduler allocation -> reservation -> transfer -> completion -> release
   conservation at 6144/1024, with zero/missing/incorrect groups rejected.
3. Existing GLM structural PD and NIXL context regressions preserved.
4. Explicit remaining typed execution failures; no full-path claim.
5. Meaningful focused tests before commit; one clean-source strict 77-case
   comparison against /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline.

Owned code: existing allocation and transfer modules plus scoped regression tests.
Other owners' GLM H5 and V4 profiling files are excluded. Use frontier-py310 and
PYTHONDONTWRITEBYTECODE=1; logs stay under /data/ycfeng/tmp.

## [Original Request] P2 repair

Follow-up W128-V4-GROUPED-NIXL P2 repair, same branch/worktree; no merge/push/platform.
Reviewer ses_ea2a9124cffe8PElQc7qwMEzQn found concrete reachable failure:
events/kv_cache_transfer_start_event.py calls real MetricsStore.on_kv_cache_transfer_start();
with op-level tracing this enters metrics/op_trace_utils.py whole-model
bind_attention_family and rejects V4 mixed compression variants after P claim/D
reservation committed, so no transfer-end event. The _Metrics fixture did not
exercise it. Primary independently reran371PASS and native-source extraction for
191/6144/6145, confirming exact groupcounts; no other review finding.

Fix existing grouped transfer metadata consumption before scalar binding, reusing
grouped snapshots/component bytes. This necessary metrics metadata seam is within
approved B (not timing/graph admission A); may edit frontier/metrics/op_trace_utils.py
only for transfer metadata plus existing B tests. Do not edit model_binding or
broadly relax guards; no synthetic timings or error suppression. Reproduce actual
MetricsStore/optracing callback failure before fix, then successful scheduling/
completion and resource conservation through that callback after fix; retain GLM
metadata behavior. Keep disabled execution guards.

Update own records/diff/receipts; commit fix after focused meaningful checks; use
one new clean strict77 only for the changed shared consumer, no dirty run/repeats.
No broad test additions. Retain exact pinned method bodies or git show evidence in
existing native_retention records so reviewer need not use the differing native
working checkout. Return fix commit, RED/GREEN, real callback evidence,
focused/strict results and bounded remaining limits. No delegation; A/H5 files
remain owned by others.
