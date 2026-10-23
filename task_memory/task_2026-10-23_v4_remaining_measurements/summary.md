# V4 kernel-only route: completed prerequisites and remaining work

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | W128-H800-AUTHORITY-DOCS: archived approved first32-H800 profiling plan and standing task-required H800 resource authority; package/runtime/GPU qualification remains pending and no new run is submitted. |
| 2026-10-23 | Archived the accepted input component and current collector/measurement dependencies. |
| 2026-10-23 | Archived accepted H collector integration64f3c950 and the final G2 continuation boundary. |
| 2026-10-23 | Recorded G2 source/CPU implementation and review repairs; independent follow-up and final acceptance remain pending. |
| 2026-10-23 | Accepted G2 CPU source 05952396 and completed its proposal; GPU decision and runtime work remain open. |

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

G2 driver source `05952396` is **ACCEPTED_CPU_PROFILE_DRIVER_COMPONENT**, with 433 changed lines in seven existing non-core files. `driver/primary_acceptance.json` binds the source, two independent re-reviews, final separate-suite checks and explicit 77/77 matrix reuse. The real replay activates both role heads around the first formal window, uses native iteration limits and separates selected-window capacity from startup records.

Both R1 reviews closed all three distinct defects with no new findings. Final suites separately pass: collector 32, worker 19, streaming 39, inventory 28, V2 inventory 45, reader 135, DP harness 75, P/D harness 68, manifest 14, cold-start 18, client metrics 24 and canonical HTTP 1; all have zero errors/failures/skips. The cold-start repair corrects a baseline-reproduced staged dependency mismatch. `driver/review_initial.md` and `driver/test_report_2026-10-23_driver.md` retain findings and RED/GREEN evidence. The old startup-only retention statement describes H64f3c950 and is superseded by G2.

[requirements.md](requirements.md) records YC's exact response: “批准；h800集群的的运行无需向我过问，允许你提交符合任务需求的任意规模的rjob（在h800上）”. D-V4-H800-RJOB-AUTHORITY approves the first32-H800/codesign/non-preemptible run in [proposal.md](proposal.md), at most60minutes/32GPUh, P16+D16/TP8/DP2/EP16/PP1 and first-formal64worker steps. Task-required H800 RJobs of any necessary scale need no repeated resource approval; subsequent runs record their concrete purpose, resources and evidence and respect actual quota/capacity. This approval does not alter calibration predicates, native/C26 source authority, publication, broad CUPTI investigation, CPU-overhead collection or paused GLM scope.

Next: fresh diagnostic package/source/image and quota verification, then seal/submit/monitor and retained trace/inventory/client/terminal checks. Accepted CPU source05952396 and recordsb8fe6b2d remain unchanged; no new package is sealed, run submitted or allocation used for this collection. Package/runtime/GPU qualification is pending. Full runtime/coverage/stream/PDL/graph checks, serving totals and numerical qualification remain open; auxiliary observation failures stay soft and affected evidence UNVERIFIED.

Serving totals, complete kernel attribution, actual stream dependencies, graph/TP8/PD cost equivalence, balanced decode coverage and four-metric calibration are still open. Native06 remains ACCEPTED_NATIVE_REFERENCE_R4; old R3 FAIL/native05 UNVERIFIED and official mean/30%/10% gates are unchanged. GLM remains paused, CPU-overhead work stays last, and the handoff is informational while work continues in the same session. Historical approval-pending entries in Modification History preserve their original phase, not current authority.
