# W128-GROUNDTRUTH-FIXTURE completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Closed root review, independent merged verification andCPU-only acceptance. |
| 2026-10-24 | Archived the bounded source restoration and verified CPU evidence. |

Source commit `6abd2d8e` restores the V2 fixture with the actual frozen R3 classifier/DP-dispatch bodies, an explicit CPU group, complete source/hook checks and an explicit existing C26 fixture location. Only `tests/unit/test_native_groundtruth_check.py` changed:32 additions/5 deletions,37/60gross lines. No test assertions or frozen/runtime/native/C26/core/package/evidence bytes changed.

Observed original RED:1failure with32 identical missing-dp_utils setup errors. Observed GREEN: all99original groundtruth nodes and46existing V2 inventory/oracle tests passed with zero failures/errors/skips. The32-rank positive path retained six serving phases/rank,14hooks and expected3456per-role forward counts, with3456decode replays and no prefill replays.

Deliverables in `task_memory/task_2026-10-23_v4_remaining_measurements/profile01/groundtruth_fixture/`: requirements/plan/progress/checkpoint, this English summary, `test_report_2026-10-24_groundtruth_fixture.md`, `source.diff`, `source_budget.json`, `source_binding.json`, `immutable_before.json`, `verification.json`, `matrix_binding.json`, exact command receipts and preserved logs/XML.

Final immutable checks rehashed189files, including13runtime modules/87native sources/45sealed profile04 inputs and frozen dependencies. The existing77-case comparison and unchanged simulator/input trees are explicitly source-bound; no new capture was made.

The one-process full-suite attempt exceeded the fixed60-second CPU ceiling; exhaustive disjoint invocations covered every original test node without changing that ceiling. Initial coordinator/root-selection failures and original RED remain preserved. Image binaries were not re-executed/rehashed.

Root integrated source/records as0dbb617f/3c3fdce4 and accepted [primary_acceptance.json](primary_acceptance.json) after Standards/Spec No findings and independent145-test owner/main executions. Both rehashed189protected files;15main equivalents/matrix tree match and all32synthetic serving snapshots retain14hooks/six phases without losses. See [root report](test_report_2026-10-24_root_acceptance.md). Note9's repair is complete. GPU/runtime/reference/operator-cost/serving acceptance and full ticket closure remain excluded; original failures and the60-second whole-suite timeout stay archived.
