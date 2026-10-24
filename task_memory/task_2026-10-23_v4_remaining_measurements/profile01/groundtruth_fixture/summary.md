# W128-GROUNDTRUTH-FIXTURE completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archived the bounded source restoration and verified CPU evidence. |

Source commit `6abd2d8e` restores the V2 fixture with the actual frozen R3 classifier/DP-dispatch bodies, an explicit CPU group, complete source/hook checks and an explicit existing C26 fixture location. Only `tests/unit/test_native_groundtruth_check.py` changed:32 additions/5 deletions,37/60gross lines. No test assertions or frozen/runtime/native/C26/core/package/evidence bytes changed.

Observed original RED:1failure with32 identical missing-dp_utils setup errors. Observed GREEN: all99original groundtruth nodes and46existing V2 inventory/oracle tests passed with zero failures/errors/skips. The32-rank positive path retained six serving phases/rank,14hooks and expected3456per-role forward counts, with3456decode replays and no prefill replays.

Deliverables in `task_memory/task_2026-10-23_v4_remaining_measurements/profile01/groundtruth_fixture/`: requirements/plan/progress/checkpoint, this English summary, `test_report_2026-10-24_groundtruth_fixture.md`, `source.diff`, `source_budget.json`, `source_binding.json`, `immutable_before.json`, `verification.json`, `matrix_binding.json`, exact command receipts and preserved logs/XML.

Final immutable checks rehashed189files, including13runtime modules/87native sources/45sealed profile04 inputs and frozen dependencies. The existing77-case comparison and unchanged simulator/input trees are explicitly source-bound; no new capture was made.

The one-process full-suite attempt exceeded the fixed60-second CPU ceiling; exhaustive disjoint invocations covered every original test node without changing that ceiling. Initial coordinator/root-selection failures and original RED remain preserved. Image binaries were not re-executed/rehashed.

These are CPU fixture results only. GPU/platform/auth/Docker/network/control operations and further delegation were excluded. Root owns independent review/rerun/integration and note #9 closure; no GPU/runtime/reference/operator-cost/serving acceptance or ticket closure is claimed. Source and completion records are committed separately before the clean-state handback and STOP.
