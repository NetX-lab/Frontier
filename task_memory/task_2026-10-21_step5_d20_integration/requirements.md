# D20 integrated measured execution

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Resume the approved D20 implementation and close cross-owner input seams. |

## Original request and authority

- [Original Request] YC: "continue" after the D20 implementation correction.
- [Original Request] D20 approval: "批准 CPU 实现（推荐）". The primary source is `../task_2026-10-15_step5_preview_h800/requirements.md` in the step5-preview worktree, lines1088–1104.
- Integrate existing measured attention/FFN/cache/bundle, PD and client gate. CPU implementation can precede native qualification; no fixture establishes native timing or four-metric acceptance.
- Preserve fixed6144/1024,128@QPS1,three repetitions,collective_sim and four10% criteria. Native/R2/profile prerequisites remain open.

## Ownership and completion

- Compute owner finishes normal manager/FFN/bundle and additionally owns Step rows.py and attention/profiling_mapping.py.
- PD owner seals9625984 and existing review follow-up. Primary owns transfer input/provider integration from that point.
- Primary adds explicit READ cost input through the existing cost-provider hooks, then executes a small full measured path, required affected checks and one integrated fidelity matrix. No analytical timing substitutes for missing NIXL data.
- All code/records converge onto feat/step5-preview before final acceptance. Unfinished producer/native coverage is reported explicitly.
