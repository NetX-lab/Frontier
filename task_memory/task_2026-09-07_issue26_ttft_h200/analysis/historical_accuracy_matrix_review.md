## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded historical harness review and current replay preparation. |

# Historical accuracy-matrix harness review

## Provenance and scope

Reference clone: /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908, branch `v0.3-hopper-testbed`, HEAD `9fd7fea584dba998d3b8c12af7233c6f0c7deeed`, shallow single-branch clone, clean tracked tree. The historical task is `task_memory/task_2026-04-19_frontier_vllm_v1_accuracy_matrix`. Its vLLM gitlink is `1109c4f16e2e4c565fd5c85564f2a6eacc164393`; that submodule has not been initialized or executed. Current Frontier remains the issue26 worktree at `9e3d1874fca51491d7f14b4788394811d3e48ffa`.

The historical task uses README.md, task_plan.md, notes.md and progress.md; requirements.md and summary.md do not exist. Initial reads of those absent filenames failed, then investigation followed the actual layout. The README explicitly excludes raw artifacts and generated profiling from its curated evidence pack. Historical acceptance is reported history, not reproducible numerical evidence by itself.

## Reviewed surfaces

- Task README, task_plan, notes, progress, issues and mainline status after A24.
- MoE co-location prefill-heavy eager and remote25 CUDA-graph manifests; decode-heavy eager acceptance report.
- `scripts/run_vllm_colo_clean.py`, `leaf_runtime.py`, `compare_case_metrics.py`.
- `tests/comparison/chunked_prefill_online/replay_openai_workload.py`, `extract_vllm_online_request_metrics.py` and their regression suites.
- Current vLLM batch-log producer and D007 worker identity extension, plus current uniform run artifacts.

## Reusable components and limits

| Component | Existing behavior | Replay disposition |
| --- | --- | --- |
| Workload replay | Ordered planned dispatch; concurrent HTTP streams; integer prompt tokens; records dispatch lag | Reuse mechanics after mapping the approved workload and preserving full 4096/1024 token counts |
| Warmups | Three drained replays; IDs offset by 1,000,000 per replay; prompt-token offset changes | Reuse isolation and drain validation; current string request namespace already provides equivalent isolation |
| Batch producer | Instrumentation enabled; batch IDs, request IDs and token vectors; current implementation synchronizes CUDA after each forward | Use a separately labeled diagnostic execution; its metrics cannot replace uninstrumented clean values |
| Worker identity | Current D007 producer emits DP/TP/PP identity and per-worker files | Prefer the existing identity fields; historical DP-local batch IDs alone do not identify collective rounds |
| Historical extractor | TTFT = latest prefill batch timestamp minus reconstructed client planned arrival | Retain as a separately named historical diagnostic metric, never silently replace official server TTFT |
| Percentile helper | Linear interpolation at rank `(n-1)*p` | Directly reusable; invoked for the current baseline re-analysis |
| Historical comparison CLI | Uses old Frontier schemas, reconstructs arrivals, compares p90/p95 and makespan | Map explicit current request IDs/actual arrival endpoints; retain current 10% mean-TTFT criterion |
| Historical orchestration | Hard-coded two-host Docker/SSH paths, container removal, branch-local imports | Do not execute wholesale; reuse vLLM controls through the already verified H200 worker mechanism |

## Metric contract findings

1. `run_vllm_colo_clean.py:540-561` enables instrumentation and batch output, plus server request metrics. Current vLLM `gpu_model_runner.py:2418-2459` calls `torch.cuda.synchronize()` in this path before obtaining elapsed time and writing batch timestamps. Calling it clean historically does not make it equivalent to current clean execution.
2. `extract_vllm_online_request_metrics.py:111-118` resolves arrival as `dispatch_epoch_s - dispatch_lag_ms/1000` when present. This is planned client dispatch time, not the measured engine queue arrival. Lines 243-245 subtract it from the last prefill timestamp. The comment describing queue-visible arrival is not sufficient evidence for that boundary.
3. The extractor overrides TPOT and E2E from server metrics when available but deliberately retains batch-derived TTFT (`metric_sources`, lines 376-380).
4. Historical duplicate-rank collapse keys on batch ID, ordered request IDs and token vector within 12 ms. Current DP/TP/PP identity should remain available; do not treat this time-window heuristic as proof of one global round.
5. Historical final co-location gates are <=11% and PD-only gates <=12%; earlier leaves used <10%. These historical thresholds and p90/p95 criteria do not change D006/current mean-TTFT <=10%.
6. Historical reports include calibrated Frontier surfaces (for example grouped-GEMM surrogate scale 3.03, EP communication scale 2.0 and the A24 short-output completion addition of 475 ms). The user requests only vLLM control reuse; none of those Frontier changes/data are imported.

## Concrete scope choice presented to YC

Recommended: adapt the historical vLLM replay/identity/batch tools to current Qwen3 4096/1024, 100 formal requests, QPS2, TP4/DP2/PP1/EP8, H200 step_main, FLASHINFER, eager, prefix OFF, uniform routing. Current Frontier and fresh approved H200 profile inputs stay authoritative. Run uninstrumented clean and isolated batch diagnostics; map fresh observed arrivals into a fresh CPU-master Frontier execution, then report official metrics plus separately labeled historical diagnostics and first-batch membership.

Alternative requested-scope interpretations: reproduce the original 2048/256 MoE eager leaf (64 requests, QPS4, PP2, FLASH_ATTN, 16 H200 GPUs and matching fresh profiling), or expand to a staged co-location matrix. The original single-case boundary makes this a material scope choice; no dependent GPU allocation has started while YC's answer is pending. Historical H800 hosts, old profiling and Frontier patches are not proposed options.

## First-batch status from the current uniform pair

Frontier directly records batch 0 with request IDs [0], tokens [4096], prefill [4096], decode count 0, stage interval [0, 65.790076904] ms. The first-prefill 48-layer evidence retains only request 0. The clean vLLM run has no scheduler/batch records because instrumentation was explicitly disabled. All 300 warmup requests drained before formal replay. Formal request 1 enters the engine queue 81.169424579 ms after request 0; request 0 official TTFT is 119.173765182 ms from a different start boundary. Those numbers alone do not prove the contents of the first vLLM scheduler batch or the peer DP dummy population. Cross-side request-composition status is INSUFFICIENT_EVIDENCE, not MATCH or MISMATCH. Old standard-routing diagnostic batches are not substituted.

## Current result and next action

Historical component checks: 22 PASS. Fresh historical-harness GPU/Frontier replay: NOT_STARTED_PENDING_CASE_SCOPE. Existing current uniform pair re-analysis is saved in `historical_replay_current_baseline.json` and `.md`; it is explicitly not a new run. The main mean-TTFT gap remains 23.870110536 ms (18.739217017%). Next action: receive case-scope selection, freeze compatible controls, validate semantic/preflight predicates and execute fresh isolated runs.
