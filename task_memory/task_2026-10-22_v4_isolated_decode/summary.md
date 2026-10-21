# W128-V4-ISOLATED-DECODE-CONSUMER

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archived scoped implementation and CPU validation; strict completion pending. |

## Deliverables
- `frontier/execution_time_predictor/deepseek_v4.py`: existing isolated attention reader consumes exact ongoing q1 decode with semantic/geometry/native-layout/provenance checks. No serving path changes. Round-trip parsing is decode-only to preserve existing prefill numerical behavior.
- `tests/unit/test_dsv4_graph_measured.py`: real-table positives for two methods, six layer roles and12 contexts, all scopes; supported-scope and source/layout negatives.
- `task_memory/task_2026-10-22_v4_isolated_decode/`: requirements, plan, progress, test report, inventory and checkpoint.

## Validation
207 focused CPU tests PASS;12 uncached q1 observations exact; unchanged real P6144 integration102 positive/72 typed negative PASS. Strict77 inventory PASS with no GLM-specific cases. Strict execution and final committed-source archive pending.

## Qualifications
Receipts are isolated eager physicalTP1 observations of the six-layer dummy model, target per-rankTP8/H800 dimensions. Native effective max_num_seqs1 is not CSV planner128 or serving capacity. No full-P0/E2E, graph/compiled, first-tail, PD, native admission, independent reference/repeated CV or timing correction claim. Event excursions and median differences remain unexplained. Primary owns acceptance/review; no merge or push performed.
