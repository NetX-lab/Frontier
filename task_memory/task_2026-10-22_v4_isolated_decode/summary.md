# W128-V4-ISOLATED-DECODE-CONSUMER

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Completed isolated q1 consumer, CPU checks and clean strict77 IDENTICAL archive. |

## Deliverables
- `frontier/execution_time_predictor/deepseek_v4.py`: existing isolated attention reader consumes exact ongoing q1 decode with semantic/geometry/native-layout/provenance checks. No serving path changes. Round-trip parsing is decode-only to preserve existing prefill numerical behavior.
- `tests/unit/test_dsv4_graph_measured.py`: real-table positives for two methods, six layer roles and12 contexts, all scopes; supported-scope and source/layout negatives.
- `task_memory/task_2026-10-22_v4_isolated_decode/`: requirements, plan, progress, test report, inventory and checkpoint.

## Validation
R1 review found unbound input acceptance in631250b despite its preserved207-test and strict77 results. Repair now requires independently supplied source/runtime/model/input-byte provenance in the existing attention_layout seam; expected values come from the parent accepted-source decision/root report, not candidate rows.225 focused checks and12 uncached accepted observations/two real cold-table rejections PASS; oldP6144 regression102/72 PASS. Fixed-source strict comparison pending.
207 focused CPU tests PASS, including1128 exact real-table scope comparisons across both methods/six layer roles/12 contexts.12 additional uncached q1 observations exact; unchanged real P6144 integration102 positive/72 typed negative PASS. Strict77 inventory has no GLM-specific cases. Sole clean strict77 on631250bd86b93b154d98bc148b72a78a62ba7884 passed77 executed/compared/IDENTICAL, zero missing evidence/cache/provenance differences; retained checkout clean. Selected `strict77_receipt.json` pins raw result hashes/paths. Code diff: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_isolated_decode_code.diff` (202 additions/9 deletions). Only task archive records changed after the strict-tested code commit.

## Qualifications
Receipts are isolated eager physicalTP1 observations of the six-layer dummy model, target per-rankTP8/H800 dimensions. Native effective max_num_seqs1 is not CSV planner128 or serving capacity. No full-P0/E2E, graph/compiled, first-tail, PD, native admission, independent reference/repeated CV or timing correction claim. Event excursions and median differences remain unexplained. Primary owns acceptance/review; no merge or push performed.
