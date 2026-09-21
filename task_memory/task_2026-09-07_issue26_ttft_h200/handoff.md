## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-11 | Recorded H200 local-stage repeat completion, formal identity, and rank-varying RCA. |
| 2026-09-10 | Recorded profiler failures, standalone validation and queued same-node normal/skip replay. |
| 2026-09-10 | Added completed standard minimal-bypass result and exact continuation point. |
| 2026-09-09 | Added synchronization-anchored RCA, scalar payload-isolation result, and bounded skip queue status. |
| 2026-09-08 | Captured cross-session handoff, verified progress, live CPU run and remaining execution gates. |
| 2026-09-12 | Recorded the completed H200 warmup-10 alternative all2all backend comparison and DeepEP LL RCA. |
| 2026-09-13 | Added the clean PPLX boundary-only replay and naive Kineto profiling continuation; both require the ten-warmup/1100-row gate. |
| 2026-09-13 | Recorded completed PPLX runs 03/05 and native CUDA-only Nsight run-04; retained the profiler result as diagnostic because it did not reach the clean 78–79 ms scale. |
| 2026-09-13 | Recorded native NVTX run-03 completed-workload capture failure; no additional identical retry is justified. |

## Latest resume checkpoint — 2026-09-13 NVTX run03 failure

The native/naive H200 NVTX RJob `yc26-h200-nsys-nvtx-20260913-03` completed
the standard artifact gate: ten drained 100-request warmups, 100 formal
requests, 1100 client rows, formal first request `pf4096_dc1024:0`, start/stop
markers, and a passing process-tree watcher status. It then failed during
Nsight shutdown with `Generated: No reports were generated`; its `nsys/`
directory has no `.nsys-rep`, SQLite, or stats output.

This repeats the same completed-workload NVTX trigger failure as run02. The
watcher status proves that an NVTX range was emitted by the child watcher, but
Nsight emitted no `Capture range started/ended in the application` messages.
The remaining cause is trigger delivery or process-scope handling, not a
vLLM timing result. Do not submit another identical NVTX process-tree retry.

The accepted native clean reference remains `78.118782043--79.307357788 ms`.
The existing CUDA-only Nsight run04 and Kineto run03 windows are above that
scale and diagnostic only. No pure-kernel 79 ms decomposition is available,
and no Frontier predictor/backend/CPU correction or clean/diagnostic
reconciliation is authorized from these profiler artifacts.

Report: `test_report_2026-09-13_nsys_nvtx_run03_failure.md`.

## Latest resume checkpoint — 2026-09-13 PPLX and native CUDA-only profiling completed

The two user-authorized H200 lanes have completed their artifact collection.
`yc26-h200-pplx-clean-20260913-03` and `yc26-h200-pplx-clean-20260913-05`
are both platform `Succeeded` and pass the ten drained 100-request warmup,
100 formal request, 1100-row, first-formal identity/predicate, and DP0
TP0–TP3 boundary gates. With `VLLM_ALL2ALL_BACKEND=pplx` and controlled
`VLLM_MOE_DP_CHUNK_SIZE=4096`, their boundary medians are `182.860077 ms`
and `183.436272 ms` (0.315% apart). These are independent PPLX model-forward
CUDA-event envelopes and remain a backend semantic result; they are not a
native clean reference or a Frontier correction input.

The native CUDA-only Nsight RJob `yc26-h200-nsys-cuda-only-20260913-04` is
platform `Succeeded`. Its standard client artifact has ten complete drained
100-request warmups, 100 formal requests, 1100 unique rows, and first formal
request `pf4096_dc1024:0`; `.nsys-rep`, SQLite, CUDA reports, and the
rank-aware postprocess are present. The corrected SQLite classifier removes
the generic `collective` token so FlashInfer `CollectiveEpilogue` remains
compute; the repair is committed as `8d5dfc5f`.

The four selected CUDA-context processes have bounded Nsight windows
`106.824398/102.003857/99.196101/101.641481 ms`, with compute unions
`35.519020/33.800442/32.840313/32.726633 ms`, communication unions
`17.580257/60.052668/58.671559/61.517546 ms`, memory unions
`2.272524/2.172709/2.245958/2.114272 ms`, and idle/non-activity
`50.369607/5.163441/4.674389/4.336521 ms`. The fourth process has no
`cudaProfilerStop` API row and uses the earliest-stop fallback, so this is
not a strict common-window measurement. Nsight did not produce a clean
79 ms decomposition; all values are diagnostic and cannot replace the
accepted native clean `78.118782043–79.307357788 ms` span or authorize
Frontier predictor/backend/CPU corrections or clean/diagnostic reconciliation.

Reports and artifacts:

- `test_report_2026-09-13_pplx_clean_boundary_runs03_05.md`
- `test_report_2026-09-13_nsys_cuda_only_run04.md`
- `analysis/pplx-clean-20260913-run03/runtime/boundary_analysis.json`
- `analysis/pplx-clean-boundary-20260913-run05/runtime/boundary_analysis.json`
- `analysis/nsys-cuda-only-20260913-run04-postprocess/nsys_sqlite_rank_breakdown_reclassified.json`

No new RJob is active. A profiler retry centered on a 79 ms window remains an
optional follow-up only; the current evidence is sufficient to reject the
existing Nsight window as a clean reference.

## Latest resume checkpoint — 2026-09-13 PPLX repair and naive Kineto run-03

The PPLX analyzer repair is committed as `ad0f6f0d`. The existing PPLX
boundary-only replay was re-analyzed using the actual selected DP lane: the
first formal request `pf4096_dc1024:0` landed on DP1, and DP1 TP0--TP3 are
complete. The repaired artifact is
`analysis/pplx-boundary-repaired-20260913-02.json` with median
`542.338012695 ms`, P90 `542.361901855 ms`, rank max `542.371643066 ms`, and
spread `0.390197754 ms`. This is a fused PPLX CUDA-event envelope, not a
native clean DP0 reference and not a Frontier correction input.

The naive Kineto RJob `yc26-h200-naive-kineto-20260913-03` completed its data
collection but the platform phase is `Failed` because the worker's final
post-client Bash row-count expression was syntactically invalid. The client
and profiler artifacts were already complete before that check: ten drained
100-request warmups, 100 formal requests, 1100 client rows, eight GPU traces,
one async CPU trace, first formal request `pf4096_dc1024:0`, and complete DP0
TP0--TP3 48-layer marker groups. The worker arithmetic was fixed and committed
as `f1f5ba74`; `bash -n`, arithmetic execution, and `git diff --check` pass.

The run-03 parser output is
`analysis/naive-profiler-20260913/run-03_kineto_breakdown_formal.json`; the
independent gate audit is
`analysis/naive-profiler-20260913/run-03/run-03_gate_audit.json`; the report is
`test_report_2026-09-13_naive_kineto_run03.md`. DP0 marker spans are
`113.293890625 / 110.268503906 / 114.339026367 / 112.168838867 ms` for
TP0--TP3, giving median `112.731364746 ms`, P90 `114.025485645 ms`, max
`114.339026367 ms`, and spread `4.070522461 ms`. Category unions are compute
`27.821732--28.975674 ms`, communication `22.931490--72.972239 ms`, memory
`3.674188--3.908914 ms`, all-kernel `55.816078--105.084194 ms`, and
idle/non-kernel `7.084645--58.522948 ms`.

Kineto run-03 is diagnostic only. Its median is approximately 42--44% above
the accepted clean native `78.118782043--79.307357788 ms` span, and the rank
communication/idle redistribution matches the prior profiler pattern. Do not
sum rank-local communication, infer a missing 20 ms operation, modify
Frontier predictors/backends, add CPU overhead, or perform clean/diagnostic
reconciliation from this run. No H200 RJob is active. Optional PPLX
chunk-size/backend attribution remains deferred until a new scoped decision.

## Latest resume checkpoint — 2026-09-12 H200 all2all backend experiment completed

The alternative communication-backend experiment is complete on the approved
H200 `step_main + h200 + num_gpu_blocks_override=310809` recipe. The native /
naive RJob `yc26-h200-all2all-naive-20260912-01`, DeepEP high-throughput RJob
`yc26-h200-all2all-deepep-ht-20260912-02`, and DeepEP low-latency RJob
`yc26-h200-all2all-deepep-ll-20260912-05` each passed the ten drained 100-request
warmup gate, 100 formal requests, 1100 client rows, eight-worker identity
validation, and the exact first-formal 4096-prefill predicates. The PPLX
capability path remains unresolved and has no formal timing.

Reduced batch-only DP0 TP0--TP3 first-formal medians are native/naive
`83.551776886 ms`, DeepEP HT `86.486446381 ms`, and DeepEP LL
`719.293090820 ms`. DeepEP HT is approximately `+3.51%` versus native/naive.
DeepEP LL is capability-valid but is a decode-oriented path applied to the
4096-token prefill: vLLM's default `VLLM_MOE_DP_CHUNK_SIZE=256` drives roughly
16 chunks through 48 MoE layers, with synchronous low-latency dispatch/combine
calls in every chunk. The formal LL row is therefore a backend/workload
semantic variant, not a normal calibration reference. Adjacent LL rows show
66--68 ms decode-only work before the formal batch and 1.87--2.58 s mixed
backlog rows afterward.

Reports are persisted at
`test_report_2026-09-12_h200_all2all_deepep_ll.md` and
`analysis/backend-comparison-20260912.md`. The separate accepted clean
comparison remains Frontier `59.190354384 ms` versus vLLM
`78.118782043--79.307357788 ms`; after correcting the sign of the approximate
`9.740 ms` Frontier compute overestimate, the implied non-compute remainder is
approximately `28.668--29.857 ms`, without proving that all of it is
communication. No Frontier production correction, predictor change, CPU add-on,
fitted residual, or clean/diagnostic reconciliation is authorized by these
results. CUDA and operator gates remain open.

## Active checkpoint — 2026-09-13 clean PPLX boundary and naive Kineto profiling

The user authorized two parallel H200 investigations under the unchanged
`step_main + h200 + num_gpu_blocks_override=310809` recipe: (a) resolve the
PPLX timing path and record a clean first-formal batch CUDA boundary, and (b)
profile the native `naive` vLLM path with Kineto while all Frontier per-op and
diagnostic loggers are disabled. Both runs use ten complete drained 100-request
warmup replays followed by 100 formal requests (1100 client rows).

The PPLX capability and instrumented replay are already complete and must not be
rerun. The minimal boundary-only source patch is committed in the external
checkout `/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt` as
`cf1ef5de9c0c45aedec4cf9d22c8eb56caf8bf0b`, based on the existing PPLX
compatibility source `150fa4a1`. It records one CUDA event envelope only when
the first formal `cmpl-pf4096_dc1024:0-0` batch satisfies batch size 1,
4096-prefill and zero-decode predicates. Non-selected forwards do not record
events or synchronize. The replay wrapper/analyzer are committed as
`33c8e375` plus worker pin update `ce5b959a`. The boundary-only RJob `yc26-h200-pplx-boundary-20260913-01` has
been submitted after a successful H200 `predict-only` check and is queued/
starting behind the active Kineto allocation; it writes to
`analysis/pplx-boundary-20260913-01/`.

The naive Kineto retry `yc26-h200-naive-kineto-20260913-02` is Running on H200
and has completed two of ten warmup rounds at the latest checkpoint. Its trace
and client artifacts are not yet complete. The parser currently produces a
diagnostic first dense GPU segment; before accepting any breakdown, verify the
trace categories, formal-window alignment, and whether memcpy/memset events are
represented as kernel activities. A profiled segment that leaves the 70--110 ms
clean scale is a profiler/harness issue to investigate, not a latency result.

## Active checkpoint — 2026-09-13 clean PPLX boundary and naive Kineto profiling

The user authorized two parallel H200 investigations under the unchanged
`step_main + h200 + num_gpu_blocks_override=310809` recipe: (a) resolve the
PPLX timing path and record a clean first-formal batch CUDA boundary, and (b)
profile the native `naive` vLLM path with Kineto while all Frontier per-op and
diagnostic loggers are disabled. Both runs use ten complete drained 100-request
warmup replays followed by 100 formal requests (1100 client rows).

The PPLX capability and instrumented replay are already complete and must not be
rerun. The minimal boundary-only source patch is committed in the external
checkout `/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt` as
`f025cc30a9b8e59fe0304e7ce8b03c76a2fb5c8e`, based on the existing PPLX
compatibility source `150fa4a1`. It records one CUDA event envelope only when
the first formal `cmpl-pf4096_dc1024:0-0` batch satisfies batch size 1,
4096-prefill and zero-decode predicates. Non-selected forwards do not record
events or synchronize. The replay wrapper/analyzer are committed as
`33c8e375`; GPU execution is pending until the active Kineto RJob releases the
H200 allocation.

The naive Kineto retry `yc26-h200-naive-kineto-20260913-02` is Running on H200
and has completed two of ten warmup rounds at the latest checkpoint. Its trace
and client artifacts are not yet complete. The parser currently produces a
diagnostic first dense GPU segment; before accepting any breakdown, verify the
trace categories, formal-window alignment, and whether memcpy/memset events are
represented as kernel activities. A profiled segment that leaves the 70--110 ms
clean scale is a profiler/harness issue to investigate, not a latency result.

## Latest resume checkpoint — 2026-09-11 H200 ABBA completed

RJob `yc26-h200-paired-normal-skip-20260911-02` completed with `Succeeded` and `worker_exit_code=0` on one H200 node using `step_main + h200 + 310809`. The four standard arms (`normal_1 -> skip_1 -> skip_2 -> normal_2`) each passed 3×100 drained warmups, 100 formal requests, 400 clean rows, 400 batch rows, identity validation, and the exact first-formal predicates (`cmpl-pf4096_dc1024:0-0`, batch size 1, 4096 prefill/0 decode, DP0 TP0–TP3, `batch_dp_token_counts=[4096,1]`).

First-formal DP0 TP0–TP3 statistics (ms): normal_1 `81.084751129/81.096882629/81.099166870/0.101020813` (median/P90/max/spread); skip_1 `80.058879852/80.318404388/80.426269531/0.510169983`; skip_2 `80.260639191/80.642729950/80.786048889/0.630340576`; normal_2 `84.223361969/84.401792145/84.403327942/0.473953247`. Pair deltas are skip_1−normal_1 `-1.025871277 ms (-1.2652%)` and skip_2−normal_2 `-3.962722778 ms (-4.7050%)`; normal_2 is `3.138610840 ms` slower than normal_1. The direction rejects minimal post-MoE AR bypass as the source of the previous 5× slowdown, but the magnitude is not stable and does not close completion/queue attribution. Do not perform reconciliation or production profiling correction from this result.

Detailed report: `test_report_2026-09-11_h200_normal_skip_abba.md`. Primary artifacts: `analysis/h200-paired-normal-skip-02/{normal_1,skip_1,skip_2,normal_2}/{identity_validation.json,standard_result.json}`.

## Latest resume checkpoint — 2026-09-10 15:28 UTC, paired replay queued

This checkpoint supersedes prior live-job state. Frontier HEAD a27b80f5. RJob `yc26-h800-paired-normal-skip-20260910-01` was submitted at 15:24:33 UTC and is Pending: codesign-default queue position 4, remaining H800 quota 0. Keep this job; launch process PID979533 remains active. No new formal timing exists. Entry `/data/ycfeng/tmp/issue26-h800-paired-20260910/launch.sh` calls `tests/e2e/issue26_h800_paired_replay_worker.sh`, output `analysis/h800-paired-normal-skip-01/`. The worker runs normal_1 -> skip_1 -> skip_2 -> normal_2 on one allocation, each full standard clean+batch replay, 3x100 drained warmups and 100 formal. It records node, arm markers, identity validation, standalone standard result and worker_exit.txt. Canonical sources remain 0d633a946/e29a8f925, profiler and boundary capture OFF. Worker syntax checked; GPU execution pending, worker file uncommitted.

Prior profiler attempts h800-nsys-first-formal-01/-02 and h800-torch-profiler-first-formal-01 ended before vLLM: missing nsys in fixed image, then diagnostic source dubious ownership. They produced no formal CUDA span or trace. Existing diagnostic source at f827a4ecc has wrong prefix env binding and crossed CUDA/Torch selection counters; leave this source out of pairing. The untracked issue26_h800_nsys_worker.sh is unvalidated and deferred. DP1 dummy forwards bypass execute_model capture; any future four-rank local trace cannot establish eight-rank shared-forward causality.

Standalone analyzer now accepts explicit source and --no-historical-reference (commit a27b80f5). Direct checks against completed normal and skip artifacts reproduce 77.483665466 and 77.935920715 ms exactly; these are old evidence, not new runs. Finish queued ABBA and compare explicit new arms before causal completion profiling. Monitoring uses `timeout 60s systemd-run --user --scope -p MemoryMax=2G brainctl ...`; system scope needs unavailable interactive auth, user scope verified. This submission omitted predict-only; record as execution deviation and restore handbook preflight on the next allocation. User already authorizes queue submission when predict-only reports no resources.

## Latest resume checkpoint — 2026-09-10 minimal bypass complete

This checkpoint supersedes older job/source state below. Frontier HEAD `aa489b9a`. H800 minimal bypass RJob `yc26-h800-post-moe-bypass-20260910-03` completed on gpu-h800-0600, worker exit0. No active work remains on this RJob. Standard clean and batch each400complete rows (3x100warmup +100formal), full drain and eight-worker identity PASS. DP0 first formal request cmpl-pf4096_dc1024:0-0, batch4718, TP0-3 spans77.940704346/77.919746399/77.931137085/78.142173767ms; median77.935920715,P90 78.081732941,max78.142173767,spread0.222427368ms. Normal reproduced median77.483665466ms on gpu-h800-0496; bypass difference+0.583678%. Minimal bypass does not cause5x slowdown in this run. Different-node single-run evidence does not close stable AR/queue attribution.

Source `/data/ycfeng/tmp/issue26-vllm-post-moe-bypass-20260910`, commit e29a8f925216d517bcbf7ffad47b93970d72d1f5; parent canonical0d633a946. Only post-MoE AR invocation commented, DP combine intact, no added clone/scalar/logging. Eight ignored FlashAttention runtime assets copied byte-equal from canonical source after -02 import failure. Preserve these assets when copying source; git clone alone omits them. Two standard H800 workers now accept ISSUE26_DIAGNOSTIC_VLLM_SOURCE and pinned ISSUE26_DIAGNOSTIC_VLLM_COMMIT; normal defaults preserved. Launch explicitly clears boundary and old AR-mode variables.

Read `test_report_2026-09-10_minimal_post_moe_bypass.md` and `analysis/h800-standard-post-moe-bypass-03/{standard_result.json,identity_validation.json,execution_status.json,runtime_assets.json,launch.sh}`. -01 stopped from short launch timeout; -02 platform Succeeded but worker exit1 (missing ignored FlashAttention assets). -03 fully passed. Do not restart completed experiments. Only two pre-existing D005 scripts remain untracked. Task docs remain Git-ignored.

Next: repeated same-node normal/skip pairing within H800 -> CUDA completion-based collective/queue attribution -> discuss reconciliation only when repeat evidence supports it. Old522ms result was normal with confounded boundary I/O, workload and uniform routing. Preserve H800 codesign/h800/176000 and H200 step_main/h200/310809 as separate settings/evidence. General profiling shared contract and CPU corrections remain downstream of CUDA closure.

# Cross-session handoff — D019 / D020

Final snapshot: 2026-09-08 14:23 UTC (22:23 Asia/Hong_Kong). This document supersedes historical pending-approval/run-status passages elsewhere. Recheck live processes and artifacts on arrival; do not treat this snapshot as a permanent process status. The task is NOT complete.

## Workspace and first reads

- Active repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
- Branch: `task/issue26-ttft-h200-20260907`; snapshot HEAD `fb3ed797`. It was created from main `d71ad80...`; do not create another worktree or restart from the older September4 worktree.
- Task: `task_memory/task_2026-09-07_issue26_ttft_h200/` inside that repository. All paths below are relative to this task unless explicitly prefixed with `tests/` or `frontier/` (repository-relative).
- Read in order: this file -> `requirements.md` (especially D019 approval and D020 decision) -> active section of `plan.md` -> latest `progress.md` entries -> `summary.md` active checkpoint -> `future.md`.
- Carry forward the user's AGENTS instructions: start user-facing messages with `boss YC`; use Chinese with English technical terms; English code/comments. Keep code concise. Use grill-me for unresolved material design choices. Preserve existing approvals and other edits; do not manufacture a new approval gate for ordinary authorized work.

## Frozen decisions and environment

One case only: Qwen3-30B-A3B-Instruct-2507 (`qwen3-a3b-30b-moe`),4096 prefill/1024 output, TP4/DP2/PP1/EP8, one8GPU replica, BF16, eager, FLASHINFER, prefix caching OFF on both sides, chunked prefill OFF, uniform routing. Model48layers,hidden2048,128experts,topk8,expertwidth768. vLLM max_model_len/max_batch_tokens16384,max_num_seqs1024,310809KV blocks of16. Original workload100formal requests at Poisson QPS2, three drained100-request warmup rounds per vLLM mode. Do not introduce another E2E case.

- H200 allocation: `step_main`; no H800. Topology was verified8H200/NV18/NVSwitch.
- Keep `collective_sim`/htsim and intra-node `nvlink_analytic`. Backend submodule/build already initialized and verified; retain the active checkout/build.
- D020: keep ideal EP dispatch/combine and straggler max synchronization. Naive protocol modeling, selector, native broadcast and physical-DP schema are explicitly deferred optional work in `future.md`. Do not reintroduce them as a blocking requirement or fit their discrepancy into a constant.
- D019 explicitly approved parallel A/B/C measurements, analysis and demonstrated gated SiLU/reduction repair. CPU/workflow integration comes AFTER CUDA attribution/validation and a fresh clean E2E checkpoint. Do not add CPU overhead now.
- No prior-version measurement/model cache reuse. Valid current-task, unchanged-operator data may be retained with provenance. All obsolete incomplete GG targets are disabled in the current sparse training input.
- CPU Python: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, conda `dev-vidur-v03-hopper-e2e`, Python3.13.13. Use `PYTHONPATH=$PWD`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`; disable W&B.
- GPU image: `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`.
- GPU Python: `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, Python3.10.16, Torch2.8.0+cu128, FlashInfer0.3.0, CUDA12.8.93. Use the existing worker's verified library/runtime setup.
- Clean vLLM: `/data/ycfeng/tmp/vLLM-BS`, HEAD `46f7b179fd3bf42b9616dc4670cba419afdb2085`.
- Diagnostic vLLM: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`, branch `feature/frontier-comparison-instrumentation`, HEAD `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`. Clean/diagnostic sources remain distinct. Do not modify a checkout during its run.
- Before GPU/Docker commands read `/data/ycfeng/stepfun-env-handbook/guidence.md` and `docker.md`; bound heavy platform inspection with timeout and MemoryMax=2G.
- For external/company network operations source `/data/ycfeng/tmp/issue26-h200-network/company-proxy.sh`; synchronize uppercase HTTP_PROXY/HTTPS_PROXY/ALL_PROXY to lowercase values and both NO_PROXY/no_proxy, retaining localhost exclusions. GPU worker proxy recipe is in the handbook. Never print proxy credentials.
- Temps/raw scratch/cache: `/data/ycfeng/tmp`; persistent selected evidence: this task directory. Do not use memory-backed `/tmp` for large outputs.

## Verified progress and authoritative evidence

| Work | Verified result | Read |
| --- | --- | --- |
| Shared forward/workflow baseline | Shared mixed-phase synchronization implemented; previous full100-request replay exists. Route/enqueue inversion and selector equivalence diagnosed, full feedback attribution still incomplete. | `analysis/parallel_rca_summary.md`; `analysis/dp-workflow-rca/next_step_proposal.md` |
| MoE computation repair | Commit5dd5ee39 adds gated SiLU and routed reduction;4GPU bitwise tests and27exact-layout checks PASS. | `analysis/d019-moe-repair.md`; `analysis/d019-moe-review.md` |
| Corrected GG input and actual P | New9M4096 anchors; all774old GG targets excluded. P=.461944884724ms/layer, measured-exact; other10compute P unchanged. M4095 holdout errors+3.72%,+1.06%,−2.49%. | `analysis/d019-moe-integration/report.md`, `first_forward_validation.json`, `holdout_query.json` |
| vLLM compute coverage | Five modes each400complete records, total2000=1500warmup+500formal.3080selected scopes,19comp/mem mapping rows, separate comm table. | `analysis/d019-compute-results/report.md`, `baseline_compute_mapping.csv`, `p_s_v_linear_attention.csv`, `full-suite/summary.json` |
| Existing collective calibration |8ranks/80rows/560event samples/40kernel correlations PASS; fit8/12/24/32MiB,16MiB heldout−.644124%. | `analysis/d019-communication-sweep.md`, `.json`; `analysis/d019-communication-fit-review.md` |
| Integrated actual first forward | CorrectedGG + calibrated existingAR ->59.190354384ms, zeroRFfits, allcompute queries unchanged. CUDA gate FAIL. | `analysis/d019-first-forward-integrated/report.md`, `config.json`, `query_receipt.json`, `validation.json` |
| Linear context diagnostic |72event rows/608launch correlations/36target scopes PASS. Same kernels, large context effect; Event construction is a small effect. | `analysis/d019-linear-timing-context.md`, `d019-linear-timing-context-results.json` |
| Context implementation choices | Task-local candidate input is supported without production changes; general context selection is a new shared contract requiring design review. | `analysis/d019-linear-context-proposal.md` |

Key numerical checkpoints (DES forward boundary versus batch-only CUDA reference, NOT official TTFT):

- Original current-task Frontier65.790076904ms.
- MoE-only corrected Frontier72.327535217ms.
- Corrected MoE + reviewed AR parameter59.190354384ms.
- vLLM before79.307357788ms / after78.118782043ms.
- Integrated errors−25.365872682% /−24.230315891%, absolute20.117003404/18.928427660ms. Near-parity before AR correction was error cancellation.
- Existing case field `collective_sim_cc_backend_config_nvlink_allreduce_launch_overhead_us=4.384788772964477` replaces50 ONLY in the bounded integrated/candidate configurations. Bandwidth450GB/s,efficiency.8,latency.5us unchanged. AttentionAR48layers17.899443200→4.762262367ms; idealEP8 A2A remains2.125341867ms per phase. Global default/canonical fullcase configs were not automatically changed.
- AR primitive calibration does not cover small custom-AR decode, other group sizes, or in-context waits; the old in-context AR10%gate still fails.
- Original-timer synthetic/hot ms per layer: QKV.133432/.078192,RoPE.031784/.019216,outproj.046752/.029824. Hot/newV differences approximately−3.02%,−1.84%,−1.60%. Kernel identities/order unchanged. Profiler perturbs hot timing (.149856ms QKV envelope vs.078192 event-only); do not subtract its gaps as clean CPU time.
- Last completed clean fullcase baseline remains official meanTTFT131.268637180ms vs Frontier105.443668418ms (19.673373105% error). This is historical within THIS task and not a corrected-result claim.

## Final interruption point: no active jobs

All GPU jobs have ended: compute01 and profiles02/04 completed; profiles01/03 are preserved partial failures with independently valid completed phases. The last CPU query also completed naturally during handoff. Former PID616810 is no longer active; do not wait for it or relaunch that finished experiment.

Completed candidate evidence:

- `analysis/d019-linear-context-candidate/{report.md,config.json,linear_op_candidate.csv,export_receipt.json,query_receipt.json,validation.json,query.log}`.
- Raw output `/data/ycfeng/tmp/issue26-d019-linear-context-candidate-query-01/`, same-prefix `.log`; exact export/query commands are in the report.
- Exporter `tests/e2e/issue26_linear_context_candidate.py`, committed `fb3ed797` (84lines).
- Source82rows has exactly oneM4096/TP4 row. All40 event-only/original/prefill_hot samples per target contribute to pooled statistics; only3operators'18statcells change. All otherCSV strings preserved.
- Actual three candidate medians: QKV.0781439989805221,RoPE.0192160001024603,outproj.0298560000956058ms. These pooled medians differ slightly from means of the two round medians; they are a fixed aggregation, not selected minima.
- Actual DES boundary **54.91489841730945ms**;17successful RF.fit calls,13unique query entries/2592returns. Three changedops match their unique measured-exact rows; other8independentcompute values/features/branches/counts remain unchanged. Arithmetic reconciliation error1.42e−14ms.
- Candidate ingestion PASS; CUDAgate FAIL:−30.756868027%/−29.703335125% versus the two batch-only references. This is a sensitivity input, not canonical profile replacement, general context support or officialTTFT validation.
- Cache `/data/ycfeng/tmp/issue26-d019-moe-integration-01/predictor-cache`; existing config-sensitive keys retrained some unchanged targets. Do not claim only3fits occurred. No prior-version cache was introduced.
- InitialNFS read waits resolved without cleanup. Finalscratch checker initially mishandled a legitimate nullfeature_key in a runtime_cache branch; the corrected checker passed. Failure and commit/verification sequence are recorded in `test_report_2026-09-08_d019_linear_context_candidate.md`; no production failure or history rewrite.

The tracked working tree is clean. Only the two original D005 scripts remain untracked. The next session starts with interpretation/adoption and remaining RCA, not collecting a missing candidate result.

## Remaining plan, in execution order

1. **Recover the completed candidate evidence.** Read its report/actual receipt/validation and distinguish isolated data-ingestion PASS from CUDAgateFAIL. The export/query/verification/commit sub-step is complete; do not repeat it without a new question. Use its unchanged-op proof and actual54.914898417ms result when planning the next correction.
2. **Resolve profiling-context adoption at the correct scope.** Use the measured evidence and source proposal. Task-local isolated input validation is already authorized. Formal general linear context support is not present; it requires explicit eligible-op, prefill/mixed/decode and missing-context semantics across producer and trainers. Consult YC with a concrete minimal design before a new shared contract. Do not silently insert a hot prefix globally or mix synthetic/hot same-shape rows.
3. **Continue first-forward CUDA RCA/correction.** Complete qualified P/S/V and communication accounting, inspect remaining attention/shuffle, missing model-level work, rank waits and inside-forward host gaps. Reuse existing probes; add only measurements that can resolve a specific causal uncertainty. Retain ideal/naive and F4096/V4097 limitations. Do not assume the remaining18.9–20.1ms is all CPU, all protocol, or a scalar ML bias. If the retained abstraction prevents the acceptance target, report that evidence explicitly; do not silently reverse D020.
4. **Validate scoped fixes with fresh first-forward references.** Report per-op signed/absolute errors and reference spread; no total pass by cancellation, cross-run subtraction, parallel-rank summation or inclusive-parent double counting. Keep instrumentation perturbation explicit.
5. **After CUDA attribution/validation, obtain a fresh clean E2E checkpoint.** Before a full100-request Frontier run, supplement correctedGG reachable-shape coverage and small custom-AR decode as needed; the current9anchors have oneunique feature key and cannot qualify the whole case. B prepared a32-shape train/holdout plan in `analysis/d019-moe-integration/`; do not launch188exact shapes blindly. Use fresh clean arrivals, not an old trace relabeled as a new paired run.
6. **Only then address outside-forward CPU/workflow.** Reuse current profiling interfaces and historical test components, collect new actual-shape evidence, separate GPU logits/sample and marker-internal gaps from outside-marker CPU, and revisit route/enqueue/admission timing. Finish fresh officialTTFT/TPOT/E2E/throughput validation, consolidate evidence/commits on this branch, and update task summary. Do not mark calibration complete merely because an isolated gate passes.

Dependencies: completed_candidate_review -> scoped_context_decision_and_remaining_CUDA_RCA -> reviewed_corrections -> fresh_CUDA_validation -> fresh_clean_E2E -> conditional_CPU/workflow -> full_case_acceptance.

Parallelism is authorized: read-only compute/context analysis and communication/residual analysis can proceed independently, with disjoint file ownership. One coordinator owns shared docs/configs and GPU scheduling; serialize measurements per allocation. Old session agents/mailboxes are not required to continue.

## Relevant modules and reusable scripts

- Predictor/actual bounded replay: `tests/e2e/issue26_predictor_query_audit.py`; it deliberately stops at first stage. The sequential simulator's ordinary time_limit did not reliably stop the old diagnostic; do not replace the explicit hard boundary with that assumption.
- Candidate export: `tests/e2e/issue26_linear_context_candidate.py`; corrected GG export: `tests/performance/issue26_moe_profile_dataset.py`.
- Operator analysis: `tests/e2e/issue26_first_batch_op_rca_compute.py`, `issue26_first_batch_op_rca_frontier.py`, `issue26_first_batch_op_rca_vllm.py`, `issue26_diagnostic_identity_analysis.py`.
- GPU harness: `tests/e2e/issue26_h200_compute_suite.sh`, `issue26_h200_diagnostics_worker.sh`, `issue26_h200_environment_probe.sh`; `tests/performance/issue26_h200_d019_profiles_worker.sh` selects independent/resume phases.
- GPU profiles: `tests/performance/issue26_moe_exact_profile.py`, `issue26_attention_exact_profile.py`, `issue26_h200_collective_microbenchmark.py`, `issue26_linear_timing_context.py`.
- Production profiling: `frontier/profiling/moe/{moe_vllm_kernel.py,moe_wrapper.py}`, `frontier/profiling/linear_op/{linear_op_wrapper.py,linear_op_impl.py,main.py}`, `frontier/profiling/common/cuda_timer.py`, existing `frontier/moe_gating_runtime.py`.
- Prediction/training: `frontier/execution_time_predictor/{sklearn_execution_time_predictor.py,sklearn_moe_execution_time_predictor.py,shared_prediction_model_manager.py}` and public trainers. Check project cleanup/split requirements before extending large critical modules; prefer bounded existing seams.
- Communication: `frontier/cc_backend/cc_backend_config.py`, collective_sim adapter and its submodule `python/collective_sim_core/intra_server_model.py`; protocol source mapping in `analysis/d019-communication.md`.
- Historical reference clone: `/data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908`, branchv0.3-hopper-testbed, HEAD9fd7fea..., task `task_memory/task_2026-04-19_frontier_vllm_v1_accuracy_matrix`. Reuse components/source, NEVER historical numerical data. Read `analysis/cpu_overhead_reuse_d019.md` before selecting CPU tests.

## Operational safeguards and closure

D005 untracked `tests/e2e/issue26_endpoint_ab_analysis.py` and `issue26_h200_endpoint_ab_worker.sh` predate this step: preserve them. Do not clean caches, remove/move files, delete branches/worktrees, rewrite history or expand shared contracts without applicable authorization. The earlier24-cache cleanup was limited to its original approved list, not blanket future cleanup permission.

Task docs are Git-ignored and persisted on NFS; do not force-add them. Commit each verified code sub-step only with its own files. Read `requirements.md` again before final reporting. The handoff request is a transfer of execution, not task completion.

- 2026-09-11: Switched ABBA pairing to H200 using step_main + h200 + 310809. RJob yc26-h200-paired-normal-skip-20260911-01 submitted; initial phase Starting/Scheduled with no machine available messages, then still starting. Output analysis/h200-paired-normal-skip-01. H200 worker script committed faedd1aa; no timing rows yet. H800 queued job remains separate and untouched.

- 2026-09-11: H200 paired job -01 completed clean normal_1 (400 rows, 3 warmups + formal drain PASS) then failed at diagnostic source/commit assertion because H200 diagnostics worker hardcoded old checkout. Root cause fixed in commit 7b462dfa using explicit SOURCE/COMMIT override for both H200 official and diagnostic workers. New job yc26-h200-paired-normal-skip-20260911-02 is Running; no timing yet.

## Latest resume checkpoint — 2026-09-11 H200 completion capture queued

The H200 completion-capture wrapper is committed as `035da750`. External diagnostic source commits `b12bbcf48`, `ab6cb0f97`, and `f35033e05a5fa5632ac4a4952981cb8c14e9f346` provide the CUDA event capture and corrected DP-local metadata. The first launch `yc26-h200-completion-normal-20260911-01` failed before vLLM because a short commit was passed to an exact full-hash check; it produced only environment-probe artifacts and must not be used. The corrected launch `yc26-h200-completion-normal-20260911-02` uses `step_main + h200 + 8 GPU + 310809`, full source hash `f35033e05a5fa5632ac4a4952981cb8c14e9f346`, and output `analysis/h200-completion-normal-02/`; it is currently queued after the platform reported no machine available. Continue monitoring this same RJob.

Completion rows remain diagnostic-only because the runner synchronizes once and writes rank-separated files after each forward. Select DP0 by `dp_rank=0` and TP0-TP3 after verifying batch identity; retain `global_rank` for physical process alignment. Do not claim queue/collective RCA until the exact first formal batch contains complete 48-layer combine and tp_ar rows per selected rank.

## Latest verified checkpoint — 2026-09-11 H200 local-stage repeat

RJob `yc26-h200-completion-local-stage-20260911-03` completed on H200 `step_main + h200 + 310809` with worker success. The standard chain passed three drained 100-request warmup replays, 100 formal requests, 400 client rows, formal identity validation, and the exact first-formal DP0 predicates. The selected request is `cmpl-pf4096_dc1024:0-0`, `batch_id=3882`, batch size 1, 4096 prefill, zero decode, and `batch_dp_token_counts=[4096,1]`.

DP0 TP0–TP3 outer spans are `91.194595337/91.198524475/91.387741089/91.428733826 ms` (rank max `91.428733826 ms`, spread `0.234138489 ms`). Completion rows selected by persisted `batch_id=3882` contain 48 layers × 48 `local_moe_apply`, `combine`, and `tp_ar` rows per rank. Local CUDA sums are TP0/TP1/TP2/TP3 `35.018848/32.306528/47.142784/25.494016 ms`; TP2 is latest at local-stage end and combine start on 37/48 layers. TP AR sums are `21.707904/25.163552/6.539872/32.188768 ms`, showing inclusive rendezvous wait on earlier ranks.

The late rank changed from TP1 in the prior `-02` run to TP2 here. The repeat therefore supports per-run local-MoE device execution or queued-device-work variation before combine, rather than a fixed TP-rank cause. Local-to-combine CUDA gaps remain `0.002976–0.003104 ms` median and host gaps `0.030914–0.031907 ms` median. The diagnostic envelope still does not separate kernels, expert token populations, host enqueue, stream idle, or NCCL internals, and its synchronize/file-I/O path is not clean evidence. Do not close the CUDA gate, claim op-gap closure, reconcile clean/diagnostic spans, or apply Frontier production corrections from this capture.

Primary artifacts:

- `analysis/h200-completion-local-stage-03/identity_validation.json`
- `analysis/h200-completion-local-stage-03/completion_local_stage_first_formal_summary.json`
- `test_report_2026-09-11_h200_local_stage_repeat.md`
