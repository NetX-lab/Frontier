# D29 Frontier position-routing design delivery

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive the first design delivery with its unclosed review and pricing scope. |

Artifact: `task_memory/task_2026-10-22_step5_position_routing/design.md`.
It describes interval-based expert counts, physical/live separation, source
propagation, native-router profiling and new-policy coverage requirements.
The author reports 200 math cases passing; no production or GPU result follows.

Initial source/test/fixture estimate: 32 files / 1699 gross changed lines.
This excludes the required source-local router and physical-collective pricing
work and is under independent review for correctness and minimality.
Implementation, final numerical scope approval and all runtime acceptance remain
open. The parent owns integration, fixture/native operations and final acceptance.
