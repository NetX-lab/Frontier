# D29 trace serialization repair completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive the single-path implementation and focused evidence; keep parent gates open. |

TaskID: `D29-TRACE-SERIALIZATION-REPAIR`.
Branch: `feat/step5-d29-trace-serialization`.
Code commit: `390bb5a2` (`perf(metrics): serialize declared trace fields at flush`).

## Deliverables

- `frontier/metrics/trace_store.py`: remove the undocumented one-caller
  `TraceEvent.to_dict`; replace `asdict` with `fields`; write top-level non-None
  declared fields at the existing flush site. No new helper or exposed return
  interface. `BaseEvent.to_dict` and failure restoration remain unchanged.
- `task_memory/task_2026-10-22_step5_position_routing/trace_serialization_repair/`:
  requirements, plan, progress/checkpoints, this archive,
  `test_report_2026-10-23_trace_serialization_repair.md`,
  `trace_serialization.diff`, `budget.json`, and selected `evidence/` receipts.
- `/data/ycfeng/tmp/d29_trace_serialization_repair/`: source-bound scratch
  scripts, raw logs, full captured replay/wave JSONL traces and physical proofs.

## Observed validation

- Exact Git consumption: 7 insertions + 7 deletions = 14 gross, below20.
  The donor remains41 below45; no test/config/schema/cache path changed.
- Both focused suites: 98 passed / 1 unchanged known mapper-fixture failure;
  all99 test statuses match. Eight real-disk root events match byte for byte.
- 598 captured events: identical8,353,308-byte payload and required
  SHA256 `3ea6239cea9295e5d532722200c78238c9e01ecb58b32383fe8c8aaa20b7e004`.
- Real EP producer: 7 events,21,051 bytes, exact identity; no unsupported leaf.
- Complete actual layer60 wave:208/208 duration and full-proof matches;
  1,040 events over five iterations have identical baseline/candidate bytes.
- Contracts, exact allowed AST delta, unchanged error handlers/BaseEvent, and
  whitespace checks pass. Injected OSError/TypeError propagate and restore the
  buffer in the original order.
- Unprofiled598-event serialization median:0.373688s ->0.066070s.
  Complete-wave warm flush median:0.131004s ->0.023990s, a0.107013s reduction.
  Pricing variation is not attributed to this repair; no full-CLI claim.

## Open work and limits

This is the allocated source repair and focused verification only. Parent-owned
independent review, final union freeze/accounting, strict77/full-unit, full CLI,
and native/GPU/workflow acceptance remain pending. No scope expansion, network,
GPU/platform action, peer edit or new agent was performed. The existing known
mapper failure is preserved, not repaired or suppressed.
