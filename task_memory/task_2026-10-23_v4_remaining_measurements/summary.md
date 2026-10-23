# V4 kernel-only route: completed prerequisites and remaining work

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | W128-PROFILE02-STAGE-DOCS factual delta: job exp-1023-180821-373637 created10:08:21Z, verified SUBMITTED_PENDING/queuing at10:10:16Z with personal creator/resources/image/NFS/RDMA; I3 monitoring then I4 verification remain open. |
| 2026-10-23 | W128-PROFILE02-STAGE-DOCS: archived preserved profile01 entry FAIL, repaired/re-reviewed cc0227e9 profile02, separate final root tests and 45-input diagnostic seal; launch started without evidenced platform job identity. |
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

## Diagnostic profile02 package — current (2026-10-23)

- Initial profile01 entry FAIL (`ModuleNotFoundError: No module named 'vllm_replay'`) remains preserved, unchanged, unsealed and unsubmitted. Owner repair `d87f4c74`, records `a90c9dbb`, merged `cc0227e9`, is complete at cumulative152/300 changed lines in two maintained source/config files. Both independent re-reviews have no open findings. `profile01/review.md` records the closure; `profile01/delivery.json` retains the historical owner-unsealed snapshot.
- Final root tests after source/package binding passed separately: actual StepMind-interpreter launcher; real locked-image bootstrap/fresh child with87native/23tool/20binary pins; four isolated hook/role-path probes, each PASS. No actual GPU constructor/model forward/replay was exercised. Root sealed45 inputs, manifest `51e39a42ec863e496f20be8a165230e0dee17c34906ad6bd4fade14bf898ee91`, status SEALED_READY_DIAGNOSTIC. Evidence: `profile01/test_report_2026-10-23_root.md`, `profile01/profile02_root_readiness.json`.
- Current status: SUBMITTED_PENDING, job `exp-1023-180821-373637`, created2026-10-23T10:08:21Z. Snapshot2026-10-23T10:10:16Z is Pending/`RJob is queuing`, `replicas=[]`, not Running or success. Verified creator `i-fengyicheng`, `codesign`, preemptible=no, four replicas each8GPU/64CPU/500Gi (SDK mem_gb=512 converts to500Gi), locked image, exact local NFS `100.96.128.195:<profile02-package>:<profile02-package>` and RDMA1/8; launcher PID3236764 was alive then. Evidence: `profile01/profile02_platform_launch.json`; `profile01/profile02_root_submission.json` retains the prelaunch-gates snapshot. GPU/runtime qualification is not granted by submission or the diagnostic seal.

## Remaining

H source/CPU closure is complete. Its emitted profiles remain UNVERIFIED: real GPU stream mapping, per-kernel PDL, compile/breakable survival, captured-segment ownership and GPU CUPTI restart require runtime evidence. Installed-image31/31source symbols are static evidence, not successful native executable hook resolution.

G2 driver source `05952396` is **ACCEPTED_CPU_PROFILE_DRIVER_COMPONENT**, with 433 changed lines in seven existing non-core files. `driver/primary_acceptance.json` binds the source, two independent re-reviews, final separate-suite checks and explicit 77/77 matrix reuse. The real replay activates both role heads around the first formal window, uses native iteration limits and separates selected-window capacity from startup records.

Both R1 reviews closed all three distinct defects with no new findings. Final suites separately pass: collector 32, worker 19, streaming 39, inventory 28, V2 inventory 45, reader 135, DP harness 75, P/D harness 68, manifest 14, cold-start 18, client metrics 24 and canonical HTTP 1; all have zero errors/failures/skips. The cold-start repair corrects a baseline-reproduced staged dependency mismatch. `driver/review_initial.md` and `driver/test_report_2026-10-23_driver.md` retain findings and RED/GREEN evidence. The old startup-only retention statement describes H64f3c950 and is superseded by G2.

[requirements.md](requirements.md) records YC's exact response: “批准；h800集群的的运行无需向我过问，允许你提交符合任务需求的任意规模的rjob（在h800上）”. D-V4-H800-RJOB-AUTHORITY approves the first32-H800/codesign/non-preemptible run in [proposal.md](proposal.md), at most60minutes/32GPUh, P16+D16/TP8/DP2/EP16/PP1 and first-formal64worker steps. Task-required H800 RJobs of any necessary scale need no repeated resource approval; subsequent runs record their concrete purpose, resources and evidence and respect actual quota/capacity. This approval does not alter calibration predicates, native/C26 source authority, publication, broad CUPTI investigation, CPU-overhead collection or paused GLM scope.

Next: I3 scheduling/business monitoring, then I4 retained-data verification of trace/inventory/client/terminal evidence under the approved plan, without repeated H800 resource approval. Accepted CPU source05952396 and existing77 binding remain unchanged. GPUs/runtime/operator coverage/stream/event/graph/TP8 costs and full05/09/19 remain unfinished; submission and the diagnostic seal are not runtime or measured-cost qualification. Serving totals and numerical qualification remain open; auxiliary observation failures stay soft and affected evidence UNVERIFIED. Earlier package-pending, approval-pending and launch-started/no-job-identity snapshots are historical.

Serving totals, complete kernel attribution, actual stream dependencies, graph/TP8/PD cost equivalence, balanced decode coverage and four-metric calibration are still open. Native06 remains ACCEPTED_NATIVE_REFERENCE_R4; old R3 FAIL/native05 UNVERIFIED and official mean/30%/10% gates are unchanged. GLM remains paused, CPU-overhead work stays last, and the handoff is informational while work continues in the same session. Historical approval-pending entries in Modification History preserve their original phase, not current authority.
