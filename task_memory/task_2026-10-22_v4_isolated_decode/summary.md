# W128-V4-ISOLATED-DECODE-CONSUMER

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Completed R1 binding repair, separate clean strict77 and final original-base diff archive. |

## Deliverables
- `frontier/execution_time_predictor/deepseek_v4.py`: existing isolated attention reader consumes exact ongoing q1 decode with semantic/geometry/native-layout/provenance checks. No serving path changes. Round-trip parsing is decode-only to preserve existing prefill numerical behavior.
- `tests/unit/test_dsv4_graph_measured.py`: real-table positives for two methods, six layer roles and12 contexts, all scopes; supported-scope and source/layout negatives.
- `tests/fixtures/deepseek_v4_decode_inputs.json`: expected acceptance identity and method-specific byte pins independently supplied by parent's accepted-source instruction/retained root receipt; no candidate self-qualification.
- `task_memory/task_2026-10-22_v4_isolated_decode/`: requirements, plan, progress, test report, inventory and checkpoint.

## Validation
R1 review found unbound input acceptance in631250b despite its preserved207-test and strict77 results. Repair102c0c9e503548640e86e18be8374faac7f06917 now requires independently supplied source/runtime/model/input-byte provenance in the existing attention_layout seam.225 focused checks (1128 exact real-table scope comparisons),12 uncached accepted observations/two real cold-table rejections PASS; oldP6144 regression102/72 PASS. Separate clean fixed-source strict77 **PASS77 executed/compared/IDENTICAL**, zero missing evidence/cache/provenance differences; selected `r1_strict77_receipt.json` pins raw paths/hashes and clean retained checkout. Original-base code/test/fixture diff302 additions/11 deletions (313 changed lines), full code+record artifact `/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_isolated_decode_r1_full.diff` includes final archive-only records with no change to102c0c9 verified code.
Historical631250b only:207 focused tests,12 uncached observations and P6144 102/72 passed. Its sole clean strict77 passed77 executed/compared/IDENTICAL; `strict77_receipt.json` pins original results. These results do not establish safe input admission and are not attributed to R1. Original code-only artifact `/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_isolated_decode_code.diff` remains preserved.

## Qualifications
Receipts are isolated eager physicalTP1 observations of the six-layer dummy model, target per-rankTP8/H800 dimensions. Native effective max_num_seqs1 is not CSV planner128 or serving capacity. No full-P0/E2E, graph/compiled, first-tail, PD, native admission, independent reference/repeated CV or timing correction claim. Event excursions and median differences remain unexplained. Primary owns acceptance/review; no merge or push performed.
