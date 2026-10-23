# V4 kernel-only route: completed prerequisites and remaining work

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived the accepted input component and current collector/measurement dependencies. |

## Completed

- Decision, retained-trace audit and native profiling seam investigation are complete. The audit reproduces1815kernels/132annotations/366owned kernels and identifies missing outer boundaries and dependency evidence; it does not establish an additive serving cost. Tool: `tests/analysis/v4_kernel_coverage.py`; evidence: `audit/` and `native_profile_seam.md` in this directory.
- Private isolated attention input source `4a1b53e6` is integrated at `732406aa235ff63c97e7b9bbd6d5836fe1881de7`. It discovers `dsv4_attention.json` beside the selected CSV, requires separate external admission and preserves explicit-layout compatibility.
- Root accepted the actual candidate at `/data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_candidate02/`. All twelve table/layer layout maps were reconstructed from seven non-timing records; original evidence and both tables remain unchanged.
- Root validation:194affected unit tests on final source, then194on merged source; both exact-input paths yield94medians/layouts; legacy102positive/72negative controls remain valid. Counts describe separate checks and are not summed. Independent Standards and Spec reviews returned no findings.
- Strict matrix:77/77identical against the fixed baseline,451artifacts rehashed, no cache-name or provenance findings. The merged source explicitly reuses this capture, with a recorded byte-equivalence binding.

Acceptance and exact artifact paths: `input_primary/primary_acceptance.json`, `input_primary/input_admission.json`, `input_primary/merged_matrix_binding.json`, and `input_primary/test_report_2026-10-23_primary.md`. Owner delivery records are in `input_bundle/`.

## Remaining

H collector implementation is active in `/data/ycfeng/Frontier/.worktrees/v4-native-serve-profile`. Its first delivery found and reproduced three launch omissions; root authorized the two additional launch paths, bringing allocation to five non-core files/787changed lines. Collector source/CPU verification, installed hooks, independent review and integration remain required.

G2 driver requirements are recorded in `driver_design.md`. The real replay callsite must activate profiling on both role heads before selected formal dispatch and stop after drain, preserving native behavior and auxiliary soft failures. Implementation and bounded resource/window planning precede any new GPU request.

Serving totals, complete kernel attribution, actual stream dependencies, graph/TP8/PD cost equivalence, balanced decode coverage and four-metric calibration are still open. Native06 remains the admitted R4 reference. GLM remains paused, and CPU-overhead work stays last. No new GPU allocation was used or authorized in this phase.
