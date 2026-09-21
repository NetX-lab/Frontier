## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Recorded profiler failures, standalone validation and queued same-node normal/skip replay. |
| 2026-09-08 | Recorded bounded environment, diagnostic contract, and independent zero-payload execution-repair reviews. |


| Target Component / Phase | Reviewer Agent Identity | Inspected Artifacts | Identified Issues / Anomalies | Remediation / Verification Code Actions Taken |
| --- | --- | --- | --- | --- |
| Simulator runtime preflight | /root/frontier_runtime_recovery | fresh-frontier-01/frontier.log; environment probe; pyproject; environment handbooks | Profiling environment lacks plotly; GPU probe did not import Simulator | Added task runtime audit and isolated simulator interpreter selection; Actual generation02 full Simulator import and RF fit/predictPASS in vidur_te; interpreter selection committedbdd261ab. |
| Diagnostic worker joins | /root/diagnostic_join_review | vLLM GPUModelRunner/utils; task identity helper | Total prefill count did not prove unique TP coverage; detail/batch fields unchecked | Added unique TP worker and metadata checks in helper; Both actual400-request operator/routing checksPASS, including100formal and8worker coverage; committede87b0b54. |
| Operator scope boundaries | /root/diagnostic_join_review | Qwen3/FusedMoE/parallel_state sources; first warmup schema only | Duplicate nested dispatch/combine, norm/add overlap, attention projection includes TP AR, grouped GEMM contract discrepancy | Recorded analysis/operator_scope_contract.md; no source change or numeric attribution; independent MoE contract review requested. |
| Zero-payload execution repair | /root/diagnostic_join_review | Frontier 69764e50; collective-sim e564935; generation02 failure/config; real-runner tests and reported 26 PASS | Scoped PASS: positive integer normalization and PREDICT latency remain unchanged; exact one-server EP8 path filters all network pairs. Off-case zero payload with multiple network phases still divides by zero at htsim_runner.py:773. | Independently reproduced the off-case limitation with Python 3.12.3, without GPU or htsim execution; no source changes. Full H200 completion and empirical latency calibration remain unverified by this review. |
| Clean TTFT engine-interval decomposition | /root/moe_profile_contract_review | vLLM 46f7b179f `output_processor.py`, `metrics/stats.py`, and `engine/__init__.py`; clean 100-request server metrics; `analysis/clean_engine_prefill_decomposition.json` | Algebra and ms units PASS: mean 115.982880592 = 80.554247736 + 35.428632856. The 80.554-ms term is first-SCHEDULED to first EngineCoreOutput, not pure GPU/model time; the remainder mixes queue/front-end/IPC boundaries and is neither CPU-only nor an additive correction. | Independently recomputed all 100 rows with zero per-row algebra residual, unique request identities, finite nonnegative components, and fixed 4096/1024 shapes. Accepted as a D005-independent request-level diagnostic only; no GPU, source edit, operator averaging, or TTFT attribution. |

The execution-repair review accepts the fixed one-server H200 case. `htsim_runner.py:1453` preserves explicit zero and CLI precedence; lines 2332–2343 distinguish missing and negative payloads. Positive integer CLI/spec values retain their former merge and execution path. The unused converters have no remaining Python references. `predictor.py:24–36` still invokes the real runner and combines its result with the unchanged intra-server model; zero bytes therefore retain seven 0.5-us steps (0.0035 ms). This establishes model-contract preservation, not measured NCCL latency.

The tests use default all-to-all channel/chunk settings, whereas the actual case uses channels=8, chunk_bytes=8000000, and inflight=2. This difference does not affect the fixed case: `htsim_runner.py:755–759` removes every same-server pair before phase or chunk calculations. The tests consequently do not establish general zero-payload support for inter-server traffic. A direct CPU counterexample used `generate_tm_from_spec` with nodes=8, servers=2, gpus_per_server=4, TP1/CP1/DP1/EP8, EP domain, TP/CP/DP/EP placement, exclude_intra_server=True, pairwise_steps, channels=1, tensor_bytes=0, and seed=0. `/usr/bin/python3` 3.12.3 with `PYTHONDONTWRITEBYTECODE=1` observed `ZeroDivisionError: integer division or modulo by zero` at line 773, before traffic output or binary execution. Existing arithmetic is newly reachable through the corrected CLI boundary; this is outside the active one-server case and is not a positive-payload regression. Broadening the repair or its acceptance claim requires a separate scoped decision.

## Minimal bypass source review — 2026-09-10

- Target Component/Phase: isolated single-call bypass and standard H800 source selection.
- Reviewer Agent Identity: root/bypass_audit (read-only), independently checked by root.
- Inspected Artifacts: fused_moe/layer.py, all2all.py, parallel_state.py, _C.py loaders, H800 official/diagnostic workers.
- Identified Issues/Anomalies: hard-coded worker source paths; inherited boundary logger independent of instrumentation flag; no source evidence requiring clone. Audit missed ignored FlashAttention files, exposed by actual -02 failure.
- Remediation/Verification Code Actions Taken: select isolated path in both workers, unset boundary/old mode in launch, retain DP combine, copy eight missing runtime assets unchanged after failure. Static diff/AST/shell and asset byte comparison PASS. GPU validation pending -03.

## 2026-09-10 paired replay and capture audit

- Target Component/Phase: H800 same-node normal/skip standard suite; preceding capture failures.
- Reviewer Agent Identity: /root/pairing_audit; independent code/evidence recheck by /root.
- Inspected Artifacts: H800 replay/official/diagnostic workers, paired worker, standard result analyzer, canonical source commits, three failed preflight logs, gpu_model_runner capture conditions.
- Identified Issues/Anomalies: historical reference/source hardcoding blocks fresh standalone analysis; profiler counter and prefix defects; DP1 dummy model call bypasses capture hook. Agent assertion that every arm cache is unique was rejected: cache paths use fixed probe basenames runtime/clean; existing standard cache convention remains unchanged, full warmup is required. Agent assertion that scheduler logging cannot affect span is unsupported; retain batch-only diagnostic scope and host-gap caveat.
- Remediation/Verification Code Actions Taken: explicit source and standalone analyzer mode; two existing-artifact direct runs PASS (normal77.483665466,skip77.935920715ms), committed a27b80f5. Canonical source pair and same standard logging held fixed; capture path deferred. GPU ABBA queued, execution pending.
