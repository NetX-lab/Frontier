# W128-V4-GROUPED-NIXL (B)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded the delegated approved CPU-first grouped NIXL slice. |

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
