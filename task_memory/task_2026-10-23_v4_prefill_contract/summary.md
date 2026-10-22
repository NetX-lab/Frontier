# Isolated V4 prefill contract completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived scoped implementation, final CPU validation and remaining qualification boundaries. |
| 2026-10-23 | Recorded completed source/evidence commit2f87828 and pending root acceptance. |

Task ID: `W128-V4-PREFILL-CONTRACT-DQ10`. Base `b62c155abd981a71e12e62aa0c0fe3990741d2f8`; source/evidence commit `2f87828`; worktree `/data/ycfeng/Frontier/.worktrees/v4-prefill-contract`. Primary source remains untouched pending root review.

## Deliverables

- `frontier/execution_time_predictor/deepseek_v4.py`: production-owned independent prefill layout builder, explicit accepted-file/runtime provenance consumption and typed isolated PD rejection. Reuses existing producer metadata/model derivation and reader contract.
- `tests/integration/check_dsv4_measured_scopes.py`: explicit `--accepted-prefill` plus `--prefill-package` input path, hash-pinned producer plan/raw/identity inputs and independent scope enumeration; legacy H200 defaults unchanged.
- `tests/unit/test_dsv4_graph_measured.py`: real accepted contract comparisons and24 typed negative cases across both methods.
- Task records in this directory: `requirements.md`, `plan.md`, `progress.md`, `checkpoint_2026-10-23.md`, `test_report_2026-10-23_v4_prefill_contract.md`, `comparisons.json`, `source.diff`, and selected `verification_logs.json`.

## Observed verification

- Final affected unit module144PASS in69.46s,0FAIL/0SKIP; final prefill subset27PASS.
- Actual accepted tables94/94 exact median comparisons: CUDA_EVENT50 plus KERNEL_ONLY44. All relative/absolute errors0; serving_admission=false.
- Legacy checker102positive/72typed negative controls unchanged, including absent-selector and valid historical non-balanced behavior.
- Exactly3 maintained source/test files197added+deleted lines,core55; whitespace check passes. Original accepted files and frozen package remain unchanged.
- Core SHA-256 `45d30210ffc34a1538e86eb3305a8cd5ef9960122c4b3c5d4c6b26df12f4429c`; exact diff SHA-256 `aa3450a980a046184cb2e4410cae9552b6646d22d19b2c98b977e6e3924e3fbf`.

## Open/deferred work

Root review/acceptance and primary integration are pending. No full predictor construction, serving/PD/graph/physicalTP8/first-tail/balanced-decode/histogram/CPU-overhead/E2E admission. No GPU or external operation occurred. A strict77 rerun is unnecessary for this isolated-only seam and was not performed; root retains the final regression decision if integration changes source.
