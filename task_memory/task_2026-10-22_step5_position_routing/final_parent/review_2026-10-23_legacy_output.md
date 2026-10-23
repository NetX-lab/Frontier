# Legacy output repair review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Persist independent P2 finding and owner repair allocation; retain strict77 PASS as separate evidence. |

TaskID D29-LEGACY-OUTPUT-REPAIR-REVIEW. Reviewer session
ses_e99338171ffelDu0PCiJJJhChG;read-only,no tests or writes. Frozen integrated
source60ba4c7569b7c9fd4c8f9f43a16f92249fa76181 and exact
final_parent/legacy_output_b60dfddd.diff (four paths).

## Finding P2 — selected prepare metadata has inconsistent numerical source

quantization_manager.py837-838 reports registered FP8 but retains
data_source=approximation and approximation_factor2.0 computed against default
BF16. The actual writer CSV
`/data/ycfeng/tmp/d29_legacy_output_green/test_precision_csv_requires_ac1/op_precision_metadata.csv`
contains this row. Actual position preparation uses the FP8 exact measured
duration/count, not that BF16 approximation. The added test asserts precision
but does not check the other fields.

Required correction:derive precision,source and factor coherently from the
actually selected scope. This case must report FP8/profiling/empty factor.
Retain genuine approximation,including legitimate precision/quant-signature
mismatch;do not globally clear approximation or change precision lookup,timing,
prediction/cache hashes. Test the complete metadata row and real approximation.

Reviewer confirmed omission of empty collective proofs preserves nonempty
plan/cost serialization;inventory priority and model reset had no other finding.
It inspected the four source/test paths,related applicability/registration/reset,
actual FP8 query/plan-cost pairing,root requirements/plan/progress,repair allocation,
owner records,RED log/GREEN CSV,and parent XML194PASS. It did not inspect running
CLI/strict77 outputs or claim GPU/native qualification.

## Disposition and allocation

CLOSED by the re-review below;originally returned to PRICING as D29-PREPARE-METADATA-REPAIR. Additional
delta<=40,only quantization_manager.py and test_comm_operator_families.py,inside
remaining100 at approved test commit047f7313 (PRICING3856/3956). Quantization cap29
remains;test cap rises50→70 funded by step5_cost120→100,zero-sum. No new paths,
timing flags,baseline filters or runtime resources. Re-review required.

Root independently compared the repaired legacy workflow:77/77 identical against
713e6a8,zero failures/missing evidence/definition or cache-name differences.
It rehashed451 artifacts and verified519 cache filenames per side. Receipt:
final_parent/fidelity_repair02_audit.json. This PASS covers the legacy matrix;
it does not close the Step-specific P2 above. The active CLI remains frozen60ba.

## Re-review: P2 closed on1be91bad

TaskID D29-PREPARE-METADATA-REPAIR-REVIEW,same independent reviewer. It checked
the complete four-path prepare_metadata_review_delta.diff and actual files at
1be91bad431d3bd886de7f5d80c19bd7e6c54a25,including the two catalogue fixtures.
Verdict:No findings;original P2 closed. Matching selected prepare reports the
complete FP8/profiling/empty-factor row; genuine precision/signature approximation
remains. Source inventory and explicit operation precision retain precedence;
new registration state resets with load_config/model configuration. The last
two properties were inspected statically,not newly tested by the reviewer.

Catalogue changes preserve assertions and V4 applicability;no fallback,skip or
timing/lookup/hash change. Reviewer read owner196PASS XML and both actual CSVs;
it did not run tests or inspect the active CLI. Parent separately completed
196PASS and CSV/legacy-byte checks in test_report_2026-10-23_final_source.md.
Source finding closure permits final-code freeze;it does not certify final
strict77/full-unit/CLI or GPU numerical gates.
