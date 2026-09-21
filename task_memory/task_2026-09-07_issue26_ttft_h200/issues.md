## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-11 | Recorded H200 local-stage repeat completion, formal identity, and rank-varying RCA. |
| 2026-09-10 | Recorded profiler failures, standalone validation and queued same-node normal/skip replay. |
| 2026-09-10 | Recorded successful standard H800 replay reproduction and corrected baseline provenance. |
| 2026-09-10 | Recorded the ten-warmup H800 bounded formal-span result and its validation limits. |
| 2026-09-09 | Recorded the recovered H200 batch-only baseline and separated the pending exact source-pin job from the completed authoritative artifact. |
| 2026-09-09 | Recorded synchronization-boundary RCA and remaining diagnostic A/B blockers. |

## 2026-09-10 active failures and queue state

Three profiler attempts ended before vLLM and generated zero formal timing/trace. nsys02 explicitly reports unavailable executable in worker image; Torch01 reports dubious ownership. Diagnostic source f827a4ecc additionally crosses CUDA/Torch counters and reads CUDA prefix while wrapper exports Torch prefix. These defects are confirmed from full condition lines, not only agent claims. Capture path is deferred and excluded from approved canonical pair source. Any future hook must qualify target identity and DP0 and account for the separate DP1 dummy forward path before shared-forward claims.

Paired RJob yc26-h800-paired-normal-skip-20260910-01 is Pending at15:28UTC (codesign-default position4, H800 quota0). This is allocation waiting; no runtime result. Worker execution receipt and arm validators determine completion independently of platform Succeeded.

# Open Issues and Root-Cause Analysis

## I-2026-09-09-04 — Exact source-pin H200 repetition is platform queued

**Observed:** the completed historical H200 artifact and fresh validator rerun provide a strict batch-only baseline at `76.9348--77.1880 ms` for the first formal DP0 TP0--TP3 boundary. The independent source-pin RJob `yc26-h200-batch-repro-historical-361-20260909-02` is still `Starting`; its worker is pending because the H200 scheduler reports insufficient matching GPU/CPU/memory capacity.

**Root cause:** this is a platform allocation condition, not a vLLM or measurement failure. No runtime rows exist for the pending job, so it cannot be used as a second timing sample.

**Resolution/status:** the authoritative completed artifact remains the valid baseline. The task-local source-pin worker now exports `VLLM_MOE_UNIFORM_ROUTING=1`, matching the historical replay wrapper. Preserve the queued job and recheck it later; keep H800, `normal/scalar_sync/skip`, post-MoE RCA, and span reconciliation paused until the H200 baseline decision is made.

## I-2026-09-09-01 — Late TP participant cause is not yet fully decomposed

**Observed:** `combine` is a DP-pair all-reduce and the first following TP-wide collective is post-MoE AR. Existing traces match the latest combine arrival to the shortest AR scope in 48/48, 45/48 and 30/48 layers across three batches.

**Root cause established:** the DP combine does not align TP0–TP3. Early TP ranks enter the TP AR and record inclusive waiting for the late participant; rank-local AR scope sums are therefore not additive.

**Root cause still open:** the late DP-pair arrival may contain local routed workload, host-side enqueue delay or queued CUDA/NCCL work. Existing event scopes cannot separate these sources.

**Next evidence:** normal versus one-element TP `scalar_sync` is complete and rejects payload size as the primary cause. A bounded `skip` run is queued to observe the upstream path; record payload shape and boundary timestamps before any production correction or span reconciliation.

## I-2026-09-09-02 — Post-combine AR payload shape is not persisted

Existing routing rows record dispatch population and local routed counts but not the post-combine `states.shape`, `numel` or `dtype`. The fixed hidden size is 2048 BF16, but DP-local token slices vary with mixed batches and dummy padding. A fresh boundary probe must record shape before asserting that all AR calls carry the same 16 MiB payload.

## I-2026-09-09-03 — GPU A/B allocation is serialized

Baseline job `yc26-h200-rank-stability-20260909-01` and scalar job `yc26-h200-ar-scalar-20260909-04` succeeded. Bounded skip `yc26-h200-ar-skip-20260909-01` failed before requests because `warmups=0` violates the client precondition. Matching normal/scalar/skip jobs were resubmitted on H800 with corrected `warmups=3`; preserve all jobs and output directories. Do not stop or delete either job during this RCA.

- 2026-09-09 H800 rerun execution RCA: the first H800 normal RJob (`yc26-h800-ar-normal-20260909-01`) failed at platform pod admission because the selected node exposed an unhealthy `mellanox.com/mlnx_rdma` device; the worker command never ran. The first H800 scalar/skip jobs reached the worker but reused the H200 probe, which raised NVML `NVMLError_NotSupported` on H800 and then hit a `set -u` `CUDA_HOME` expansion bug. These are execution failures with no timing evidence. The corrected H800 probe is isolated at `/data/ycfeng/tmp/issue26_h800_environment_probe.sh`; rerun jobs use `codesign` + `h800` and remain separate from H200's `step_main` + `h200` recipe.
- H800 `-02/-03` diagnostic runs are invalid despite boundary JSONL presence: all server logs show EngineCore KV-cache OOM from inherited H200 `num-gpu-blocks-override=310809` (H800 79.19 GiB, 477 MiB free, attempted 2.37 GiB). Boundary rows were emitted during initialization profiling before any request. H800-specific reruns use override 4096; H200 keeps 310809. Cluster recipes and artifact directories remain separate.

## H800 ten-warmup evidence limits — 2026-09-10

Completed bounded formal span remains519.799-522.551ms after ten drained warmups. Standard-suite gate remains unfulfilled (one request per round), and batch worker lacks an explicit uniform-routing flag. Manifest formal_requests100 was incorrect (actual1); future worker metadata fixed, raw manifest retained. Rank-max decrease versus prior run is confounded by node and DP lane. See analysis/h800-ar-176000-normal-10warm-01/report.md.

## H800 standard baseline recovered — 2026-09-10

Fresh standard H800 DP0 first formal CUDA median77.483665ms/max77.486847ms reproduces the historical80.654881/80.687454ms scale. Both400-row modes and standard identity validation PASS. Shutdown-only TCPStore Broken pipe warnings occurred after client completion; RJob Succeeded. Prior claim that H800 lacked valid70-110ms records is corrected. The500ms diagnostic excess remains unattributed; source/config/logging differences require causal discrimination.

- 2026-09-11: Switched ABBA pairing to H200 using step_main + h200 + 310809. RJob yc26-h200-paired-normal-skip-20260911-01 submitted; initial phase Starting/Scheduled with no machine available messages, then still starting. Output analysis/h200-paired-normal-skip-01. H200 worker script committed faedd1aa; no timing rows yet. H800 queued job remains separate and untouched.

- 2026-09-11: H200 paired job -01 completed clean normal_1 (400 rows, 3 warmups + formal drain PASS) then failed at diagnostic source/commit assertion because H200 diagnostics worker hardcoded old checkout. Root cause fixed in commit 7b462dfa using explicit SOURCE/COMMIT override for both H200 official and diagnostic workers. New job yc26-h200-paired-normal-skip-20260911-02 is Running; no timing yet.

## H200 same-node normal/skip ABBA result — 2026-09-11

**Observed:** RJob `yc26-h200-paired-normal-skip-20260911-02` completed on one H200 node (`step_main + h200 + 310809`) with worker exit 0. The four arms ran `normal_1 -> skip_1 -> skip_2 -> normal_2`; every arm passed 3×100 warmup, 100 formal requests, full drain, identity validation, and the first-formal predicates (`cmpl-pf4096_dc1024:0-0`, 4096 prefill, DP0 TP0–TP3, `[4096,1]`).

**Measured:** normal_1 median `81.084751129 ms`, skip_1 `80.058879852 ms`; skip_1−normal_1 `-1.025871277 ms` (`-1.2652%`). Skip_2 median `80.260639191 ms`, normal_2 `84.223361969 ms`; skip_2−normal_2 `-3.962722778 ms` (`-4.7050%`). Rank spreads were `0.101020813`, `0.510169983`, `0.630340576`, and `0.473953247 ms` respectively. All four medians remain in the established H200 70–110 ms normal scale.

**RCA boundary:** both paired deltas are negative, so commenting only the post-MoE TP AR invocation does not cause the earlier 5× slowdown under the standard warmed chain. The delta magnitude is not stable: normal_2 is `3.138610840 ms` slower than normal_1, larger than the first pair delta. This run therefore rejects bypass-induced span inflation, but it cannot identify the residual completion/queue cause and cannot justify clean/diagnostic reconciliation or a production profiling correction.

**Next evidence:** retain the same standard chain and add completion-based timestamps only after a stable repeated delta is obtained. Any attribution must join the exact first-formal batch across `combine_return`, `tp_ar_call`, CUDA completion, and next-layer start; rank-local inclusive AR scopes remain non-additive.

## 2026-09-11 completion capture launch failure and correction

- `yc26-h200-completion-normal-20260911-01` failed after the H200 runtime probe. The worker's strict source check compares `git rev-parse HEAD` (40 characters) with `ISSUE26_DIAGNOSTIC_VLLM_COMMIT`; the launch supplied short `ab6cb0f97`, so the diagnostic chain exited before server startup. No timing artifact exists and the run is not evidence.
- Review also found a metadata bug in the diagnostic source: `get_dp_group().rank` is the global process rank, while the field is named `dp_rank`. The source now uses `get_dp_group().rank_in_group` in `f35033e05a5fa5632ac4a4952981cb8c14e9f346`.
- Corrected job `yc26-h200-completion-normal-20260911-02` uses the full pinned commit and remains queued under `step_main`; no duplicate or cross-cluster job was submitted.

## I-2026-09-11-02 — Late local-MoE participant varies by run

**Observed:** The second completion capture `yc26-h200-completion-local-stage-20260911-03` passed all standard gates and selected DP0 `batch_id=3882`. TP2 local `quant_method.apply()` CUDA sum was `47.142784 ms` and latest on 37/48 layers; its TP AR sum was only `6.539872 ms`. TP3, which was locally earlier, recorded `32.188768 ms` in the inclusive TP AR scope. Local-to-combine gaps were approximately `0.003 ms` CUDA and `0.031 ms` host on every rank.

**Cross-run evidence:** The preceding valid capture `-02` had TP1 as the late local stage on 47/48 layers, with TP1 local sum `47.901280 ms` and TP1 TP AR sum `4.770560 ms`. The late rank changed from TP1 to TP2 while the outer first-forward spans stayed near 90–91 ms.

**RCA status:** The evidence supports a per-run rank-local local-MoE device execution or queued-device-work variation before combine. It does not support a fixed TP0/TP1/TP2/TP3 hardware cause. The short AR on the late rank is the expected inclusive-rendezvous signature; cross-rank AR sums remain non-additive.

**Open decomposition:** The current envelope does not identify grouped-GEMM versus activation/reduction, expert token population, host enqueue delay, stream idle, or NCCL internal behavior. A production correction or clean/diagnostic reconciliation remains unauthorized until those sources are separately measured and the clean span gate is closed.

## Native Nsight run-03 post-client worker failure — 2026-09-13

**Observed:** `yc26-h200-nsys-profiler-20260913-03` reached the complete data gate (10 drained warmups, 100 formal requests, 1100 client rows, first-formal markers, and a 1,994,203-byte `.nsys-rep`) but the RJob phase is `Failed`. The final worker log is `tests/e2e/issue26_h200_nsys_profiler_worker.sh: line 152: ttp://127.0.0.1:8000: No such file or directory`.

**Boundary:** `runtime/phase_records.json` records formal completion and the start/stop capture messages are present, so the failure is after the client/capture work. Treat platform status as FAIL and preserve the artifact; do not report a successful RJob or silently discard the status.

**Postprocess limitation:** the worker-created Nsight directory is owned by root. Stats were generated from a byte copy in `analysis/nsys-profiler-20260913-run03-postprocess/`. Nsight timestamps are process-relative and control markers are wall-clock, so the report uses a trace-domain profiler API interval and labels it diagnostic. It is not a clean 78–79 ms span.
