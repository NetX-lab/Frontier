# Calibration warm-target skill requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded inherited D-WARM-TARGET authority and the six workflow-review acceptance criteria. |

## [Original Request]

> 采纳你的方案，继续下一步；并请你修正完善skill： /data/ycfeng/Frontier/skills/frontier-calibration，确保比较、校准范畴为充分warmup条件下的（对于kernel编译、通信握手等warmup开销，不建模）

Authority and comparison evidence are maintained in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/requirements.md`, D-WARM-TARGET. This directory records the repository-root skill edit and its validation, not a separate calibration decision.

## Acceptance

1. Define the warm comparison scope once in the contract; entry and reference documents point to it.
2. Preserve original all-request readings and all valid repeats. Describe reproducible event-bound, repeat-specific symmetric selection, retained windows and bilateral pooling; the deciding analysis reading is explicit.
3. Separate clean-run validity from warm readiness, allowing valid incompletely warmed runs with complete event evidence into the warm-reading path.
4. Before launch require a coverage plan and verification method; require observed coverage and drain at the formal-window boundary and post-run.
5. Restrict the simple NIXL compatibility-log counting shortcut to its verified PP1, same-TP pull path. Other layouts require identity-based coverage and completed setup.
6. A warmup-policy-only correction starts a new run generation and retains the unchanged serving-build identity.
7. Existing structure validation, skill validation and phase review pass. Preserve README and unrelated work. No simulator or harness runtime change belongs to this skill slice.
