# V4 kernel-only route: completed prerequisites and remaining work

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived the accepted input component and current collector/measurement dependencies. |
| 2026-10-23 | Archived accepted H collector integration64f3c950 and the final G2 continuation boundary. |

## Completed

- Decision, retained-trace audit and native profiling seam investigation are complete. The audit reproduces1815kernels/132annotations/366owned kernels and identifies missing outer boundaries and dependency evidence; it does not establish an additive serving cost. Tool: `tests/analysis/v4_kernel_coverage.py`; evidence: `audit/` and `native_profile_seam.md` in this directory.
- Private isolated attention input source `4a1b53e6` is integrated at `732406aa235ff63c97e7b9bbd6d5836fe1881de7`. It discovers `dsv4_attention.json` beside the selected CSV, requires separate external admission and preserves explicit-layout compatibility.
- Root accepted the actual candidate at `/data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_candidate02/`. All twelve table/layer layout maps were reconstructed from seven non-timing records; original evidence and both tables remain unchanged.
- Root validation:194affected unit tests on final source, then194on merged source; both exact-input paths yield94medians/layouts; legacy102positive/72negative controls remain valid. Counts describe separate checks and are not summed. Independent Standards and Spec reviews returned no findings.
- Strict matrix:77/77identical against the fixed baseline,451artifacts rehashed, no cache-name or provenance findings. The merged source explicitly reuses this capture, with a recorded byte-equivalence binding.

Acceptance and exact artifact paths: `input_primary/primary_acceptance.json`, `input_primary/input_admission.json`, `input_primary/merged_matrix_binding.json`, and `input_primary/test_report_2026-10-23_primary.md`. Owner delivery records are in `input_bundle/`.

- Native collector source `f8621a8258ffa445abfb51f5ea73258ad271494e` is integrated at `64f3c9509b09279f321d237e3a95cbe904c6dd22`. All five initial review findings are closed by independent Spec/Standards re-reviews. Scope: five non-core files,880/987changed lines. Status: `ACCEPTED_CPU_COLLECTOR_COMPONENT`.
- Root verified final-source and merged suites separately: collector28, inventory28, V2inventory45, reader135, worker16, DPharness75, P/Dharness68 and manifest14, all PASS with zero failures/errors/skips. Merged automatic94 remains exactly equal to the prior accepted input receipt. The H default matrix gate explicitly reuses the4a1b53e6 capture with77/77identical; no new capture was run.
- Final H evidence: `collector_primary/primary_acceptance.json`, `collector_primary/review_2026-10-23_final.md`, `collector_primary/test_report_2026-10-23_final.md`, `collector_primary/merged_matrix_binding.json`. Initial failures/RED controls are retained. Source and task records are consolidated on `feat/glm53-dsv4-vllm028`; the continuation prompt is `handoff_prompt.md` and pending actions are in `progress.md`.

## Remaining

H source/CPU closure is complete. Its emitted profiles remain UNVERIFIED: real GPU stream mapping, per-kernel PDL, compile/breakable survival, captured-segment ownership and GPU CUPTI restart require runtime evidence. Installed-image31/31source symbols are static evidence, not successful native executable hook resolution.

G2 driver requirements are recorded in `driver_design.md`. The real replay callsite must activate profiling on both role heads before selected formal dispatch and stop after drain, preserving native behavior and auxiliary soft failures. Implementation and bounded resource/window planning precede any new GPU request.

The collector currently records from startup and emits without resetting its buffers; `/start_profile` does not reset its retention budget. G2 must align records/bytes capacity with the actual selected profiling windows and preserve their required evidence.

Serving totals, complete kernel attribution, actual stream dependencies, graph/TP8/PD cost equivalence, balanced decode coverage and four-metric calibration are still open. Native06 remains the admitted R4 reference. GLM remains paused, and CPU-overhead work stays last. No new GPU allocation was used or authorized in this phase.
