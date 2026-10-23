# D29-CLI-TIMEOUT-REPRO requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record the diagnosis-only assignment and parent-verified main07 evidence. |

## [Original Request]

Diagnose both unchanged 2400-second actual-main07 timeouts on source
`e41abc073500d61f224eda4fca1123e74e40544d`, submodule
`3dac53c544e546574cb6e8078cd14053d0916113`, without another complete CLI run.
Build a seconds-scale deterministic RED-capable timing/call-count loop at the
actual dominant seam. Use the actual fixture03 workload and retained inputs;
profile first, state falsifiable hypotheses, and change one diagnostic variable
at a time. Deliver exact commands, measured results, source locations, limits,
and a proposed repair owner/path/budget. This is not implementation authority.

Read-only inputs:
- `/data/ycfeng/tmp/d29-parent-position-main07/{offline,online}/`.
- `/data/ycfeng/tmp/d29-parent-position-fixture03/`.
- `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05/`.
- Current requirements/plan, `GLOSSARY.md`, ADR0001, and the diagnosing-bugs skill.

Write only probes under `/data/ycfeng/tmp/d29_cli_timeout_repro/` and records in
this directory. Do not edit production/tests, frozen trees, old artifacts,
workloads, gates, semantics, timeouts, or source. No dependency installation,
GPU/platform/network operations, extra agents, or full CLI retry. Relocate every
CLI output/cache path before config parsing.

## [Follow-up Decision] Parent evidence, 2026-10-23

Parent verified exactly seven PREFILL-only scheduler records in each mode: two
iteration starts, two iteration ends, one grouped cache allocation, one decision,
and one grouped cache batch. Last simulation times are offline0.0/online0.25;
there are no final request/system metrics or stage/lane/transfer ledgers. Do not
equate this with the historical D23 multi-decode state.

Parent streamed all nine launch-recorded inputs, 9,900,444,171 bytes, in12.356s;
all hashes are unchanged, and e41/3dac are clean. Do not repeat those checks.
Evidence: root `resumed_parent/main07_timeout_receipt.json`, records-only commits
`33288957` and `44c3008d`. Primary owns next hypothesis ranking and repair
authorization. Return the diagnosis phase once the real-path loop exists.
