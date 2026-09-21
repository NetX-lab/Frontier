## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Implemented a pinned PPLX first-formal CUDA boundary-only source and H200 worker/analyzer; static gates pass, GPU replay deferred to coordinator. |
| 2026-09-11 | Recorded H200 local-stage repeat completion, formal identity, and rank-varying RCA. |
| 2026-09-10 | Recorded profiler failures, standalone validation and queued same-node normal/skip replay. |
| 2026-09-10 | Recorded approved single-call AR bypass and standard-scale gate. |
| 2026-09-10 | Recorded successful standard H800 replay reproduction and corrected baseline provenance. |
| 2026-09-10 | Recorded the ten-warmup H800 bounded formal-span result and its validation limits. |
| 2026-09-09 | Recovered and independently revalidated the pre-H800 H200 batch-only baseline; exact source-pin rerun remains queued. |
| 2026-09-08 | Updated cross-session handoff and live candidate continuation state. |
| 2026-09-08 | YC deferred naive protocol modeling as optional; resumed D019 with ideal communication. |
| 2026-09-08 | Authorized and started parallel workflow and first-batch operator RCA with bounded supplements. |
| 2026-09-08 | Completed D017 shared-forward design after tracing admission, prediction, ownership, and request completion. |
| 2026-09-08 | Investigated actual DP load balancing and recorded incomplete replay/platform errors. |
| 2026-09-08 | Recorded confirmed communication/proxy decisions and completed H200 runtime/backend checks. |
| 2026-09-07 | Initialized the fresh H200 single-case calibration task. |
| 2026-09-09 | Added synchronization-anchored post-MoE AR RCA plan and isolated diagnostic A/B source. |
| 2026-09-13 | Recorded completed-workload NVTX run03 capture failure and retained the no-report limitation. |

## 2026-09-13 — PPLX boundary-only harness implementation

- **Motivation:** The PPLX capability probe passed after correcting its metadata process group, but the existing batch diagnostics path synchronizes and logs every active forward. That path would contaminate a clean first-formal CUDA span.
- **Expectation:** A PPLX run must keep the standard ten drained warmups plus 100 formal requests and identity predicates while adding only one selected model-forward CUDA event envelope per DP0 TP rank.
- **Method:** Created the independent vLLM worktree `/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt` from `150fa4a1cf46500c22d5fa585ebc9801a43e5c2d`, retained its ignored FlashAttention runtime assets, and committed boundary capture as `f025cc30a9b8e59fe0304e7ce8b03c76a2fb5c8e`. Added `tests/e2e/issue26_h200_pplx_boundary_worker.sh` and `tests/e2e/issue26_pplx_boundary_analysis.py`. The worker clears per-op, scheduler, routing, PP, completion, and batch log variables; sets `VLLM_ALL2ALL_BACKEND=pplx`; runs the standard token-ID client with `ISSUE26_WARMUPS>=10`; and analyzes exactly `cmpl-pf4096_dc1024:0-0` with batch size 1, 4096 prefill tokens, and zero decode tokens.
- **Result:** Source AST, worker contract, analyzer synthetic artifact, shell syntax, Python compilation, pinned commit, and clean-source checks PASS. No GPU job was launched. The event value will be a per-rank model-forward CUDA stream envelope and may include queued device work; it excludes host wall-clock gaps and is not a pure-kernel sum or full diagnostic outer span.

## 2026-09-09 — Mandatory warmup/formal harness gate

- **Motivation:** Recent H200 diagnostic attempts produced startup-contaminated values in the 10,000+ ms range. YC required every subsequent test and analysis to use the established warmup-complete standard test suite so that the documented 70--110 ms batch-only scale remains reproducible.
- **Expectation:** A run is admissible only after three fully drained client warmup replays, 100 unique formal requests, 400 total client rows, and a passing identity validator. The first formal batch must be selected by request identity and explicit 4096-prefill predicates on DP0 TP0--TP3.
- **Method:** Added `task_memory/task_2026-09-07_issue26_ttft_h200/harness.md` with the fixed worker/client/validator chain, warmup drain requirements, formal boundary predicates, scope separation, and reporting obligations. No production code or GPU job was changed.
- **Result:** The task now has an explicit gate that rejects minimum-`batch_id` selection, warmup/startup rows, mixed batches for the single-request boundary, and instrumentation-only spans as formal evidence. The exact-pin H200 RJob remains queued and is still monitored read-only.

## 2026-09-09 — Exact-pin H200 retry after no-resource prediction

- **Motivation:** YC requested a direct RJob submission when `predict-only` reports no available H200 resources, so the platform queue can retain the request.
- **Expectation:** Submit the same pinned H200 recipe without changing source, image, topology, or warmup contract, then monitor the queued worker until it starts or reaches a terminal failure.
- **Method:** `predict-only` returned `no machine available`. Submitted `yc26-h200-batch-repro-historical-361-20260909-03` with `step_main`, `positive-tags=h200`, 8 GPUs, 64 CPU, 400 GiB, the pinned image digest, source clone `361d941c97fcec52e544f74b7ab91c54192de9c9`, and the standard historical batch worker. Captured status and event output under `/data/ycfeng/tmp/`.
- **Result:** The RJob was created successfully and is `Starting` with one pending worker. Queue event: `Insufficient GPU quota`, H200 queue remaining `4`; scheduling event: `0/4373 nodes are available` due to memory/GPU/CPU capacity and H200 selector mismatch on other nodes. No runtime timing artifact exists yet. The job remains queued and was not stopped or deleted.

## 2026-09-09 — H800 standard-suite replay

- **Motivation:** YC requested the same warmup/formal replay and batch trace suite on H800, with the cluster-specific settings separated from H200.
- **Expectation:** Reuse the three-drained-warmup plus 100-formal-request client and identity-analysis semantics while using the H800 probe and H800-safe KV-cache block override.
- **Method:** Added H800 counterparts for the replay, uniform-router preflight, official worker, and batch diagnostic worker under `tests/e2e/`. H800 uses `codesign + positive-tags=h800`, the dedicated `/data/ycfeng/tmp/issue26_h800_environment_probe.sh`, and `--num-gpu-blocks-override 4096`; H200 remains `step_main + positive-tags=h200`, the H200 probe, and override `310809`. Submitted `yc26-h800-standard-replay-20260909-01`, which completed before vLLM because of a corrected-path issue, then fixed the absolute probe reference and submitted `yc26-h800-standard-replay-20260909-02`.
- **Result:** H800 `-01` passed `H800_RUNTIME_PROBE_PASS` and `UNIFORM_ROUTING_RUNTIME_PASS 24` but produced no timing rows because the official worker referenced the probe path incorrectly. H800 `-02` is currently `Starting`; it has been scheduled on `gpu-h800-0350` and is pulling the pinned image. No H800 formal timing result is available yet.

## 2026-09-09 — H800 standard replay retry provenance correction

- **Motivation:** The first corrected H800 replay still failed before request submission because the H800 probe exported `PYTHONPATH=/data/ycfeng/tmp/vLLM-BS`, while the official worker checked the diagnostic clone at `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab`.
- **Expectation:** The worker must import and validate the same diagnostic clone whose commit is recorded in the run manifest, then proceed to the standard warmup/formal client.
- **Method:** Added an explicit `export PYTHONPATH=/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab` in `issue26_h800_official_ttft_worker.sh` and submitted the fresh RJob `yc26-h800-standard-replay-20260909-03` with a new output directory.
- **Result:** H800 `-02` is classified as an execution/provenance failure with no client, batch, or timing rows. H800 `-03` is `Starting` after scheduling on `gpu-h800-0157` and image pull; formal validation remains pending. H200 artifacts and jobs remain separate.

## 2026-09-09 — H800 source-pin follow-up

- **Motivation:** Monitoring showed H800 `-03` also produced only probe artifacts. The actual clean diagnostic clone HEAD is `0d633a946e6600c77a251bef8b5553ec7f43f7e7`, so the H800 worker's hard-coded `8453dd342...` expectation was stale.
- **Method:** Updated the H800 official and batch diagnostic workers to validate the observed clean H800 clone commit `0d633a946e6600c77a251bef8b5553ec7f43f7e7`, preserving explicit H800 `PYTHONPATH` and all H800 resource settings. Submitted `yc26-h800-standard-replay-20260909-04` with a fresh output directory.
- **Result:** H800 `-03` is retained as no-timing provenance evidence. H800 `-04` is currently `Starting`, scheduled on `gpu-h800-0592` and pulling the pinned image. No formal H800 rows are available yet.

## 2026-09-10 — H800 standard replay completed; formal-shape gate remains open

- **Motivation:** YC requested completion of the H800 standard chain and the warmup/formal identity and first-batch statistics.
- **Method:** Monitored `yc26-h800-standard-replay-20260909-04`, confirmed both clean and batch workers, counted client rows, and ran the standard `issue26_diagnostic_identity_analysis.py --mode batch`. Added only the minimal batch-ID normalization needed to distinguish `cmpl-warmup:` logger IDs from client `warmup:` IDs.
- **Result:** H800 batch client completed 400 rows: 300 warmup and 100 formal; all four replay barriers drained and formal IDs are unique. The first formal request `cmpl-pf4096_dc1024:0-0` is aligned on DP0 TP0--TP3 at `batch_id=12530`, with spans `81.8163833618`, `81.9474563599`, `81.5399017334`, and `81.6683502197` ms; median `81.7423667908` ms, P90 `81.9081344604` ms, rank max `81.9474563599` ms, spread `0.4075546265` ms. The strict validator does not pass for the complete run because a later formal row is chunked/mixed under H800's 4096-block capacity (`cmpl-pf4096_dc1024:24-0` has `request_num_tokens=4959`, with another prefill token vector in the same batch). The first formal boundary itself satisfies the requested predicates, but the full H800 run is not an admissible all-formal-shape PASS.

## 2026-09-10 — H800 formal block capacity assigned from H200 reference

- **Motivation:** YC clarified that `4096` is probe-only and requested the H800 formal `num-gpu-blocks-override` derived from the H200 reference.
- **Method:** Used measured runtime capacities from the standard probes: H200 `143771 MiB`, H800 `81559 MiB`. Scaled the H200 `310809` blocks by the measured memory ratio: `310809 * 81559 / 143771 = 176316.9988`; selected the conservative integer `176000` blocks. Updated both H800 official and batch diagnostic workers; H200 remains `310809`.
- **Result:** Submitted `yc26-h800-standard-replay-20260910-01` with H800 `176000` blocks and a fresh output directory. The worker is scheduled on `gpu-h800-0604` and pulling the pinned image. The prior H800 `4096` run remains diagnostic evidence only and is not reused as formal capacity-parity data.

## 2026-09-10 — H800 reference-scaled capacity replay PASS

- **Motivation:** Validate that the H800 reference-scaled `176000` block setting supports the complete warmup/formal standard suite without the `4096`-block formal chunking observed previously.
- **Method:** Completed `yc26-h800-standard-replay-20260910-01`, counted the batch client rows and replay barriers, ran `issue26_diagnostic_identity_analysis.py --mode batch`, and computed DP0 TP0--TP3 statistics from the validator's first formal records.
- **Result:** RJob `Succeeded`; batch client has 400 rows with 3 drained warmup replays and 100 unique formal requests. Validator status is `PASS` across all 8 workers. First formal `cmpl-pf4096_dc1024:0-0` is `batch_id=4696`, `[4096]`, prefill `4096`, decode `0` on DP0 TP0--TP3. Spans are `80.6409912109`, `80.6874542236`, `80.6687698364`, and `80.5964126587` ms; median `80.6548805237` ms, P90 `80.6818489075` ms, rank max `80.6874542236` ms, spread `0.0910415649` ms. This is H800-only evidence under `codesign + h800 + 176000` and is not mixed with H200 `310809` results.

## 2026-09-10 — H800 post-MoE AR comparison resumed

- **Motivation:** YC requested the historical post-MoE AR comparison to resume on H800, with the newly established formal H800 capacity value and explicit cluster separation.
- **Method:** Updated `harness.md` with the H200/H800 parameter matrix and mode contract. Added `tests/e2e/issue26_h800_ar_worker.sh`, retaining the existing bounded AR instrumentation and `normal`/`scalar_sync`/`skip` mode semantics, but using H800 probe, source commit `0d633a946`, and `num-gpu-blocks-override=176000`. Submitted the first serial mode `yc26-h800-ar-normal-20260910-01` with 3 warmups and 1 bounded formal request.
- **Result:** Normal H800 AR RJob is scheduled on `gpu-h800-0067` and pulling the pinned image. Scalar and skip remain pending until the normal run reaches a terminal result; no cross-cluster values are combined.

## Current user redirect — H200 batch baseline recovery — 2026-09-09

- **Motivation:** YC requested a return to the pre-H800 documentation and the original H200 first-batch CUDA span scale after later diagnostic runs produced 10,000+ ms values.
- **Expectation:** identify the exact H200 recipe and test suite, select the first formal 4096-prefill boundary by request identity after drained warmups, and reproduce the documented 70--110 ms batch-only scale.
- **Method:** re-read the historical launcher, replay worker, uniform-router preflight, batch diagnostic worker, token-ID client, identity validator, and the completed `h200-historical-replay-02` artifacts. Reran the identity validator and recomputed DP0 TP0--TP3 from the raw per-worker JSONL.
- **Result:** validator `PASS`; 400 client rows (`300` warmup + `100` formal), all eight worker files, and three drained warmup replays. First formal request `cmpl-pf4096_dc1024:0-0` is `batch_id=4769`, `[4096]` tokens, prefill `4096`, decode `0`. TP0--TP3 spans are `77.1250228882`, `77.1454391479`, `76.9347839355`, and `77.1880340576` ms; median `77.1352310181` ms, rank max `77.1880340576` ms, spread `0.2532501221` ms.
- **Root cause of the old outlier:** selecting `batch_id=0` or another startup/warmup row includes initialization/JIT/queue time. Full operator and record-function probes also perturb the queue and are not the batch-only metric. The canonical baseline is the identity-filtered batch logger span above.
- **Exact-pin follow-up:** `yc26-h200-batch-repro-historical-361-20260909-02` uses the isolated vLLM clone at commit `361d941c97fcec52e544f74b7ab91c54192de9c9`; it remains `Starting` with no node capacity and no timing rows. The task-local worker now explicitly exports `VLLM_MOE_UNIFORM_ROUTING=1` before writing its manifest. The pending job is retained; no H800 or downstream diagnostic run is started.
- **Evidence:** `test_report_2026-09-09_h200_baseline_recovery.md`, `runs/h200-historical-replay-02/`, and `/data/ycfeng/tmp/issue26-historical-replay-02-validation-rerun.json`.

## Final cross-session handoff checkpoint — 2026-09-08 14:23 UTC

- completed: authoritative resume state is `handoff.md`; copyable English prompt is `handoff_prompt.md`. User requested transfer, not calibration completion. All CPU/GPU jobs ended. Source HEADfb3ed797; tracked tree clean, only2originalD005untrackedfiles preserved.
- completed: lastcandidate actual54.91489841730945ms,17fits,13uniqueentries/2592returns;3changedops measured-exact,8othercompute values/features/branches/counts unchanged; independent boundary reconciliation1.42e−14ms. Source/export/actualquery/validation/log/report persisted under `analysis/d019-linear-context-candidate/`; exporter committedfb3ed797. No launch or wait is pending.
- current facts: GG+AR integrated59.190354384ms vsV79.307357788/78.118782043ms; candidate54.914898417ms has−30.756868027%/−29.703335125%errors. BothCUDAgatesFAIL. D020idealretained; naiveoptional; CPUaccountingOFF; no generalcontextselection implemented.
- pending: candidate interpretation/adoption at the proper scope; remaining CUDA RCA and fresh gate; freshcleanE2E; conditionalCPU/workflow and fullcaseacceptance. FullcaseGG/smallcustomARcoverage remainsunqualified. Exact dependencies andreadorder inhandoff.
- resolved handoff-check failure: scratch querymatcher tried tuple(null) forruntime_cache; corrected legitimate-null handling thenallchecksPASS. Sourcecommit preceded thislastchecker correction; recordedtruthfully inthecandidate testreport, nohistoryrewrite. No newproductionfailure.


# Progress

## D019 execution resumed — 2026-09-08T12:50:44.050128+00:00

- completed: recovered HEAD fc205071, tracked tree clean; preserved two D005 untracked scripts. Recorded YC approval for A/B/C and demonstrated gated SiLU/reduction repair.
- in-progress: vLLM missing scope patch, Frontier MoE implementation/routing-input repair, and communication protocol/microbenchmark work assigned in parallel with disjoint ownership. Root serializes GPU timing and owns integration.
- pending: focused checks, fresh H200 shape/operator measurements, P/S/V review, CUDA-span acceptance and new clean E2E; CPU/workflow phase waits for CUDA gate. No new timing result yet.

## D017 final-source full replay — 2026-09-08T09:09:13.950637+00:00

- completed: full fresh-cache numerical run exit0,100/100unique completions, per-request4096once+1times1023 invariantPASS;3662stage batches,98prefill-containingbatches. Evidence and exact commands: test_report_2026-09-08_shared_forward_full_case.md.
- completed: TTFT105.44366841784294ms versus131.26863718032837ms,25.824968762485426ms/19.673373105114784percent raw error. TPOT56.84539231594382vs79.24111335051272ms;E2E58.25828000762837vs81.19485246896744s. Calibration numeric gate remains open; no CPU attribution.
- completed:78/100sameDP,5/100samecompleteprefillmembership; first5clientIDs0–4match. Client6is the first divergence in clean engine-queue order. Client5/6 mapping order is correctly preserved, not confused with Frontier request IDs.
- new evidence gap: vLLM route selection precedes engine QUEUED, while Frontier currently reroutes at replayed QUEUED offsets. Clean frontend5-before6 by2.606391907ms and engine6-before5 by0.200871844ms are observed. Actual route order/snapshot is not recorded; contribution to the22placement mismatches is unresolved. D018 minimal diagnostic proposal prepared, no implementation or GPU submission.
- no active jobs remain. Pending: new diagnostic scope decision and remaining semantic/operator/official-TTFT closure. Production8ba22b49 and analyzers ff8c229a/15a35242 committed; only prior D005untracked scripts remain.

## D017 review and implementation — 2026-09-08

- Code committed `8ba22b496de121713f4fec8068ee962c29c012ea` after the focused and trained reproduction checks. Fresh100request CPU worker started at /data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime with scratch /data/ycfeng/tmp/issue26-cpu-frontier-sj80zdzj.
- ACCEPT after second-pass review: retained real-handler counterexample and raw group5/step240-241 evidence; reduced event/state machinery by reusing existing event adapters and direct source completion helpers. YC's conditional authorization and code-simplicity follow-up are recorded in requirements.md.
- Implemented shared MONOLITHIC room and phase-independent step identity, per-source wave metadata/continuation, one cohort ownership restoration, mixed-source decode-layer updates, and explicit decode model-component timing. Existing Request callbacks, predictor schema, routing settings, and CC backend remain intact.
- Focused checks:56PASS initially; source-specific additions23PASS; latest combined unit/sequential-PDD suite73PASS. Initial regression log has8actual cross-phase no-wave failures plus10test-only direct-field lookup failures; corrected the assertion to the public step getter. No production workaround was used for the test error.
- Trained original4096/1024 three-request replay PASS at /data/ycfeng/tmp/issue26-shared-forward-03req-01: 3 unique completions, 2050 stage batches, each request executes prefill4096 once and decode1 exactly1023 times. Selected evidence: analysis/shared-forward-three-request. It reuses only the current-task diagnostic cache to test control flow; final100request replay uses fresh caches. No full-case numerical acceptance claim.

## D017 design completed — 2026-09-08

- Motivation: YC requested the shared synchronization design, explicitly including local mixed batches and per-request shape/KV/progression.
- Method: read codebase-design skill and inspect ForwardSyncState, sync_entry, StageExecutionContext admission, EPWaveInputs, EP planning/tickets, both collective completion helpers, predictor mixed/KV features, and terminal request callbacks. No production edits or runtime test executions in this step.
- Result: design_shared_forward_sync.md specifies a common group/layer identity, sealed membership, neutral MONOLITHIC events, source-local attention/final timing, one shared EP ticket lifecycle, and the existing request completion chain. Full phase-combination, ownership, timing, mixed-feature, token/TTFT, and trained-replay checks are defined for implementation.
- Source findings: current collective helpers reuse sample_batch timing across sources; source-specific prediction is required even for same-phase unequal batches. The existing predictor already requires attn_decode_in_mixed for local mixed decode. Request.on_batch_end already credits the first generated token at final MONOLITHIC prefill; preserve that credit exactly once.
- Scope refinement: approximately10–13production files including small event/adaptation edits, replacing the preliminary5–7estimate. Request and predictor base behavior/schema changes are not required by this design. Existing PDD/PD-AF behavior remains outside the MONOLITHIC migration.
- Validation status: source-grounded design review complete; all runtime regression and numerical acceptance gates remain pending. Existing three untracked E2E scripts were preserved. No new calibration result or CPU-overhead attribution.
- Next: review the concrete design, implement the bounded shared lifecycle, run focused and trained3request checks, commit verified code, then fresh100request replay and batch/metric comparisons.

## 2026-09-07 — Initialization

- completed: Created worktree /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907, branch task/issue26-ttft-h200-20260907, from main d71ad80b using git worktree add. The main worktree contains unrelated modifications; none were changed or copied.
- completed: Read prior requirements, plan, manifest, progress, notes, and incomplete summary. Identified pf4096_dc1024 and the old missing attn_decode_in_mixed profile failure. Prior clean rerun7 corrected FLASHINFER backend alignment; its numeric results are historical only.
- in-progress: Recover exact image/runtime and H200 step_main access. Read guidence.md and docker.md before platform operations.
- pending: Formal case freeze, fresh profiles and measurements, comparison, RCA, repair, and rerun. No new measurement exists yet.
- inspection issues: A broad handbook search hit permission-denied files; constrain follow-up reads to Markdown documents. Default ~/.ssh/config is absent. These observations do not establish a platform access failure.

## 2026-09-07 — Configuration and platform audit

- completed: Recorded exact runtime/image provenance and source anchors in context_recovery.md. Verified the new branch base and clean vLLM-BS c169f48fa source overlay.
- completed: Identified communication backend conflict, missing explicit device, independent arrival seeds, canonical-vs-raw TTFT mismatch, and old warmup scheduling overlap. These are observed configuration/source discrepancies, not measured causes of a new H200 error.
- in-progress: D001 communication model question is pending user input. H200 step_main access still needs verification; no allocation submitted.
- platform observations: inherited proxy causes API EOF. A process-local proxy-free read reaches Forbidden for quotagroup GET; this is not a job submission failure. No persistent network or account configuration changed.
- pending: case manifest freeze; fresh H200 profiling and caches; fresh clean/diagnostic groundtruth; fresh Frontier; TTFT comparison; RCA; scoped repair and rerun.
- code changes: none. Documentation is local-only under the repository task_memory ignore rule, as intended by main d71ad80b.

## D001 resolved — Backend preparation

- completed: YC confirmed collective_sim/htsim and authorized initialization, build, and H200 topology inspection.
- in-progress: Initialize the submodule at the main-pinned gitlink b8518afcc310f0fe0e3ce52ba6b4f0bf57a3be04 and build a fresh binary. No historical prediction cache will be used.

## Backend build and direct verification completed

- completed: Fresh submodule build with make -j8, exit 0. Binary dependencies resolve on the local CPU host. Full log retained as backend_build.log.
- completed: 9 targeted topology tests passed in 20.25s using dev-vidur-v03-hopper-e2e Python 3.13.13.
- completed: Four direct Frontier->collective_sim->htsim invocations passed; 16 MiB TP4 allreduce / EP8 alltoall yielded 0.0524851 / 0 ms with legacy_fabric, and 0.3729050666666667 / 0.044277955555555554 ms with nvlink_analytic defaults. Structural diagnostics only, not H200 measurements.
- completed: Prepared and syntax-checked tests/e2e/issue26_h200_environment_probe.sh; committed the probe and its verified CUDA library setup. No simulator or vLLM source changed.
- pending information: Asked YC for the step_main access/submission recipe or existing worker. No answer received yet. Formal GPU job submission and physical H200 topology remain pending.
- newly discovered issue: The default intra-server path returns zero EP8 time. Final intra-server parameters require actual H200 topology/communication evidence; no arbitrary correction factor or unapproved H100 profile has been adopted.
- next steps: Resolve step_main entry -> run H200 runtime/topology probe -> freeze remaining semantics -> regenerate all profiling and timing evidence -> compare TTFT -> RCA and scoped repair -> rerun.

## step_main prediction and probe submission

- completed: A proxy-free rlaunch --predict-only --charged-group=step_main --positive-tags=h200 --gpu=8 --cpu=64 --memory=409600 --private-machine=group --backoff-limit=1 --predict-node-num=3 returned gpu-h200-0019 and gpu-h200-0742, each with eight GPUs. The step_main-as-quota route is now supported by live prediction evidence; the earlier access question is superseded by this observation.
- in-progress: Submitted yc26-h200-probe-20260907-1534 with the pinned frontier-env image digest, /data:/data mount, 2h platform wait, detached execution, and tests/e2e/issue26_h200_environment_probe.sh. Platform acknowledged RJob creation. Output directory: runs/h200-environment-probe-01. No formal timing workload has started.

## Probe launcher failure and correction

- observed failure: yc26-h200-probe-20260907-1534 stopped at 2026-09-07T15:33:32Z while pulling the image. Launcher output: Stopping rjob reason=interrupt; RJob stop request sent. The short outer timeout and subsequent attempt to remove that supervisor interrupted the launcher. No worker command ran and no runtime evidence was produced.
- root-cause correction: Resubmitted the same probe as yc26-h200-probe-20260907-1536 without a shell timeout. The platform retains its 2h wait. Job is scheduled to gpu-h200-0742 and pulling the pinned image. Keep the stopped job for audit.
- pending decision D002: Use collective_sim's existing nvlink_analytic intra-server path and validate parameters against fresh H200 evidence, or pause for user redirection. Interactive question sent; no answer yet. The default legacy_fabric returns 0 ms for the exact EP8 intra-server structural call and is not a viable formal baseline. The default 50 us per-step allreduce overhead needs evidence.

## H200 image starts; Git provenance probe corrected

- observed: Probe-02 reached worker execution at 2026-09-07T15:41:34Z, then exited on fatal: detected dubious ownership in repository. The GPU checks had not executed. Pinned image pull/start is established; GPU/runtime success is not.
- correction: tests/e2e/issue26_h200_environment_probe.sh now captures GPU inventory/topology before Git provenance, uses GIT_CONFIG_GLOBAL in the task temporary area, and explicitly adds only the trusted worktree to that independent config. Syntax and git diff checks passed; committed as 07bdc8d4. The exact reason the image rejected command-line safe.directory remains to be checked against its Git version; the observed cause is ownership admission.
- in-progress: Submitted yc26-h200-probe-20260907-1544 at 2026-09-07T15:43:33Z, same eight H200 GPUs/step_main/image digest and command; fresh output runs/h200-environment-probe-03.

## Current checkpoint

- Local backend preparation: completed. Fresh build, four actual backend calls, and nine topology tests are recorded.
- H200 access: step_main confirmed through prediction and accepted submissions; no further access clarification is required.
- Active job: yc26-h200-probe-20260907-1544, gpu-h200-0128, image pull/Starting as of 15:44:39Z. Fresh probe output will be runs/h200-environment-probe-03. The launcher session is 48462; use the platform job identity for durable resumption.
- Pending user decision: D002, adopting the existing nvlink_analytic intra-server model with fresh H200 parameter evidence. Question remains unanswered.
- Pending task work: physical topology/runtime verification, case semantic freeze, fresh profiling and clean/diagnostic measurements, TTFT comparison, RCA, scoped repair, and fresh rerun. No formal metric value exists.
- Newly discovered unresolved issues: default EP8 intra-server prediction is zero; canonical clean TTFT producer and workload warmup/arrival contract need final resolution. Resolved local issues: short launcher timeout and Git ownership probe setup have concrete corrections, with GPU verification of the latter pending.

## D002 resolved and network instruction updated

- completed: YC approved the existing nvlink_analytic path while retaining collective_sim/htsim. Earlier awaiting-user entries are historical and superseded.
- in-progress: Set company HTTP proxy for subsequent external/company network operations and inspect the existing probe-03 job before any new submission. No duplicate probe is planned.
- pending: Actual topology/runtime verification and all formal measurement/RCA work remain open.

## 2026-09-08 — Company proxy and worker execution verified

- completed: Retrieved the documented company proxy recipe from deploy.i.shaipower.com/httpproxy. The inherited localhost proxy returned HTTP 502; direct access was used only to bootstrap this configuration. All subsequent platform commands source the company recipe and normalize upper/lowercase proxy variables. The verified endpoint is proxy.i.shaipower.com:3128; company no_proxy is retained with localhost additions.
- observed: Probe-03 stopped at 2026-09-07T15:49:56Z before worker execution; no physical evidence. Its launcher process was no longer present on resumption.
- completed: Probe-04 launched from independent user systemd service yc26-h200-probe-launcher-04; service exit 0. Worker on gpu-h200-0019 started at 16:00:07Z and reported H200_RUNTIME_PROBE_PASS. Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0; all eight H200 GPUs passed a 32x32 matrix product.
- unresolved: nvidia-smi emitted no output for inventory, topology, or NVLink. Runtime success does not establish topology.
- in-progress: Probe-05 adds executable-path inspection and independent NVML active-link, remote-device-type, and peer connectivity records. Bash/embedded-Python syntax and git diff checks passed; committed 3319bc00. Job yc26-h200-probe-20260908-0005 uses the same image, step_main H200 allocation, and independent service.
- pending: Physical topology confirmation, canonical clean TTFT endpoint decision, full case freeze, and all new formal measurements/RCA.

## H200 topology and in-image backend checks completed

- completed: Probe-05 identified the zero-byte /usr/bin/nvidia-smi shadowing the injected binary. NVML verified eight H200 GPUs, 18 active NVLinks each, four NVSwitch PCI addresses, and all 56 directed peer checks OK.
- completed: Backend worker yc26-h200-backend-20260908-0010 explicitly used /usr/local/nvidia/bin/nvidia-smi and produced NV18 topology. It rebuilt htsim with make -B -j8 inside the pinned image and completed actual Frontier backend calls with nvlink_analytic. H200_BACKEND_RUNTIME_PASS and service exit 0 observed. Code committed as 591c5f05 after the live check.
- evidence: runs/h200-h200-backend-runtime-01; see test_report_2026-09-08_h200_runtime_backend.md. The initial report command used a singly prefixed path and failed; corrected to the actual preserved output path.
- pending decision D003: Preserve queue-visible arrival to prefill forward completion and extend existing vLLM E2E endpoint recording with minimal validated probes. Existing clean metrics end at frontend first-token receipt; existing batch logger forces GPU synchronization. User has been asked; vLLM measurement code is unchanged.
- pending: Full case freeze, fresh profiling/measurement/caches, comparison, RCA, repair, rerun. No formal TTFT result exists.

## D003 approved — implementation begins

- completed: YC approved the minimal E2E endpoint extension and validation. D003 awaiting-user entries are superseded.
- code-change marker: Extend the clean vLLM measurement chain only: worker prefill-forward CUDA completion capture, queue-visible monotonic arrival export, and focused validation. No raw first-token endpoint substitution; no new per-batch global CUDA synchronization. Record clock alignment uncertainty and measure overhead.
- selected vLLM checkout: /data/ycfeng/tmp/vLLM-BS, feature/frontier-comparison-instrumentation, clean at c169f48fa0ce16455f64101bc4366b23b1652a43 before editing. Existing code inspection found a natural non-async sampled-token synchronization in _bookkeeping_sync.

## Minimal endpoint recorder: first GPU check PASS

- implemented, not yet integration-validated: vLLM worker captures a CUDA event immediately after model forward for requests completing prefill, then reads it after existing _bookkeeping_sync. New queue_arrival_monotonic_s exports stats.queued_ts. Recorder logs initial host/GPU clock bracket, per-rank request IDs, completion timestamps, and observed natural-sync return time. Eager/synchronous/non-speculative/PP1 guards are explicit.
- verified: eight-H200 recorder probe completed, 512 records, anchor widths 14.85–19.82 us, zero separation from independent short-run clock brackets, mean added cost 17.98–22.04 us. See test_report_2026-09-08_prefill_endpoint_clock.md. Full-vLLM integration and longer-duration overhead verification remain pending; vLLM edits have not been committed as complete.
- D004 pending: source confirms old vLLM prefix caching default enabled versus Frontier default disabled. Asked YC whether both sides should explicitly disable caching (recommended) or align enabled cache state.

- prepared: Reused the original token-ID client implementation with freshly generated Poisson offsets, one common trace across replays, and an await/drain boundary after every complete warmup. No historical numeric file was copied. New worker tests/e2e/issue26_h200_endpoint_ab_worker.sh is prepared for the exact 100-request 4096/1024 case, 3 full warmups per mode, baseline versus endpoint capture, and a mandatory user-approved prefix policy. It has passed syntax checks only and has not been launched while D004 is unresolved.

## Current handoff checkpoint

- completed: D003 approved; actual eight-H200 minimal recorder clock/overhead check PASS. vLLM helper committed as db3fde591; Frontier probe scripts committed as 910229d2.
- in-progress: vLLM integration edits remain uncommitted in vllm/v1/worker/gpu_model_runner.py, vllm/v1/metrics/stats.py, and vllm/v1/engine/output_processor.py pending full-workload validation.
- prepared, not executed: Frontier tests/e2e/issue26_h200_endpoint_ab_worker.sh and tests/e2e/issue26_token_id_client.py remain untracked intentionally until required live validation. The A/B worker requires ISSUE26_PREFIX_CACHING=disabled or enabled, with no default.
- pending user input: D004 prefix caching policy. The original vLLM default and Frontier default disagree. Existing D001/D002/D003 approvals persist.
- GPU jobs: environment/backend/clock probes have completed; no A/B server or formal calibration job is running.
- next steps: resolve D004 -> run exact-case endpoint A/B and validate joins/clock/overhead -> commit integration and freeze remaining semantics -> fresh profiles and Frontier/vLLM measurements -> TTFT comparison/RCA/repair/rerun. No old profiling rows, predictor caches, or TTFT values have been consumed.

## D004 approved — exact-case endpoint integration validation

- completed: YC confirmed prefix caching disabled on both sides. Earlier pending D004 entries are superseded.
- in-progress: Launch exact 4096/1024 vLLM A/B recorder verification using the prepared worker, three complete warmup replays per mode, 100 formal-namespace validation requests, QPS2, seed20260908, TP4/DP2/EP8, BF16 dummy weights, eager execution. This is endpoint integration/overhead validation with a preserved uncommitted patch, not final clean Frontier-vLLM parity evidence.

## A/B attempt 01: client connection failure

- FAIL: baseline completed all 100 requests in warmup replay 0, then the next replay failed while writing a POST to a closing pooled transport (`aiohttp.ClientConnectionResetError`, followed by `ClientOSError`). The client exception triggered the worker EXIT cleanup and server shutdown at 16:30:00Z. No formal or endpoint-mode result exists.
- Cause addressed: the reused client pool retained streaming connections across a long replay/drain boundary. Use TCPConnector(force_close=True, limit=concurrency) so each request owns its connection; no retries or hidden dropped requests. Exact source patch is scoped to the calibration client.
- Validation: rerun the same complete A/B workload into fresh runs/h200-endpoint-ab-02. This run must cross every replay boundary and deliver all 800 request completions.

## Prefix policy and admission settings recorded

- D004 is explicit in config/frontier_settings.json and the executed vLLM no-enable-prefix-caching flag. Frontier settings remain draft pending fresh profiling and endpoint validation; no simulator result has been generated.
- Source audit: build_replica_scheduler_maps creates separate per-attention-DP child schedulers with the same scheduler config; each owns its KV allocation state. Observed 4,972,944 tokens translates to 310,809 blocks of 16 tokens per DP lane. Max sequences=1024 and max batched tokens=16384 per owner.
- Current live job yc26-h200-endpoint-ab-20260908-0035: baseline warmup replays 0 and 1 completed (200/200 requests), no repeated connection error, replay 2 in progress. The first failed attempt is retained separately.

- completed: Attempt-02 baseline client PASS: 400/400 unique full-length requests and four drained phases, no transport failure. Commit the now-verified token-ID client sub-step. Endpoint-mode server is starting on the same allocated worker; vLLM integration remains pending.

## A/B attempt 02: long-run endpoint mapping failure

- baseline client PASS: all 400 requests and all replay barriers, committed client 29345cb5.
- endpoint mode FAIL: first warmup completed 100 requests; at the start of warmup replay 1, rank 4 threw `CUDA/host clock mapping exceeds observed completion time` in frontier_prefill.py. CUDA elapsed from its initial anchor was approximately 130 seconds. Natural-sync event completion did not fail; the host mapping order check failed. No formal endpoint requests completed.
- causal status: single startup anchor is not sufficient evidence for a whole-run clock mapping. Long-run GPU/host drift is the leading hypothesis; CUDA-event precision and host-bracket validity are alternatives. Keep the failing check intact.
- in-progress: isolated all-eight-GPU clock-drift probe at elapsed 0/30/60/90/120/150 seconds, job yc26-h200-clock-drift-20260908-0050, fresh runs/h200-clock-drift-01. This directly compares CUDA elapsed time with independent host brackets without model execution or client transport.
- prepared but not applied: /data/ycfeng/tmp/issue26-h200-network/clock-finalization.patch adds a shutdown reference; it supplies evidence but does not itself fix a drifting mapping. No source patch has been applied after the failed A/B.

- Probe launch-01 failed before measurements because the sourced environment probe derived the repository from caller $0. Corrected its own location to BASH_SOURCE[0], syntax PASS. Probe launch-02 (yc26-h200-clock-drift-20260908-0052) uses fresh runs/h200-clock-drift-02. GPU/host mapping source remains unchanged; no numerical workaround applied.

- observed: independent clock probe reproduced cumulative drift by 60 seconds; rank4 midpoint offset +265.57 us and non-overlap 249.75 us. This falsifies the fixed startup-anchor assumption beyond its exported initial bracket.
- pending decision D005: asked YC about per-prefill local anchors and the added local event synchronization. No measurement-path correction is applied while waiting. Complete the isolated probe and document its evidence independently.

## Current checkpoint after clock RCA

- completed: D001–D004 applied to preparation; H200 topology/backend verified; explicit prefix caching false recorded on both sides; client lifecycle repair verified 400/400 and committed 29345cb5; sourced environment path fix verified and committed 70cc5e7f.
- completed: isolated long-run clock probe 48/48 rows, launcher exit 0; fixed-anchor clock invariance FAIL, rank4 final midpoint +641.458 us and interval separation 625.640 us. Report test_report_2026-09-08_endpoint_clock_rca.md.
- pending decision: D005 per-prefill local anchor plus measured local event synchronization. No answer received as of this checkpoint.
- pending work: implement approved clock correction and full endpoint A/B validation; freeze remaining semantics; regenerate all H200 profiles/caches; run fresh Frontier/vLLM comparison; perform case-level RCA/repair/rerun.
- active GPU jobs: none; all current probes and A/B attempts have ended. No canonical formal comparison or calibration repair has completed.

## Metric-boundary source audit for YC discussion

- Inspected current Frontier Request.ttft = prefill_completed_at - arrived_at. ExecutionTime.total_time adds active CPU overhead to model_time; stage construction consumes total_time. The current case draft explicitly skips CPU overhead modeling. A metric interval can include CPU semantically even when its current predictor omits that work.
- Verified official vLLM v0.10.2 stats.py and output_processor.py from vllm-project GitHub using the company HTTP proxy. Official server first_token_latency uses IterationStats.iteration_timestamp - req_stats.arrival_time, distinct from QUEUED and engine-core NEW_TOKEN timestamps. Output statistics update precedes detokenization and response creation in output_processor. Current fork retains this calculation.
- Inspected local serving_completion -> AsyncLLM.add_request -> Processor.process_inputs: arrival_time defaults inside process_inputs; earlier API preparation is not necessarily included in the server metric. Official benchmark client measures from before POST to first streamed completion chunk, a broader endpoint.
- Recommended reasoning: preserve queue-to-forward as an internal diagnostic interval; compare official server TTFT with a corresponding extended Frontier prediction. Add only missing critical-path stages, including before-queue, within-engine omitted host work, and post-forward work. Do not label the whole gap CPU or double-count existing overhead; scheduling changes may require DES replay rather than a constant post-hoc addition. No code or acceptance-boundary change made in this discussion.

## D006 confirmed: continue on official server TTFT

- D005 deferred; the failed GPU/host endpoint mapping is no longer a prerequisite for the primary metric. Preserve the experiment and failure evidence.
- Next bounded step: retain the verified queue-arrival metadata extension, stash only the unverified worker endpoint hooks, and use the clean instrumentation branch for fresh official E2E and separate diagnostic runs. No global Frontier metric contract is being changed.

- D006 clean preflight PASS: selected vLLM 46f7b179f is clean on the required branch, with remote tip ea95f571 checked through the company proxy. Verified queue metadata is committed; only unverified endpoint worker hooks are stashed. New run h200-official-clean-01 fixes the prior observed KV budget (310809 blocks per DP, block16), explicit seed0 and naive all2all. Full command/provenance recorded before launch. Numeric parity remains pending fresh routing/operator evidence.
- Recovered preparation failure: root-owned runs directory denied local mkdir; created only the new run parent with local-user ownership through sudo install -d. Initial launcher service exited before any GPU submission because the preparation had not produced its script.

- Submitted clean RJob yc26-h200-official-clean-20260908-01 through the company proxy; it reached image pull on gpu-h200-0043. Submitted independent profiling RJob yc26-h200-fresh-profiles-20260908-01 (still queued). Launchers are persistent user systemd services. Fresh profile grid includes 4096-prefill true mixed batches and 4096–5119 decode contexts. These shapes are initial coverage, not a routing-alignment claim. No old numeric data consumed.

- FAIL fresh profiling generation01: CLI rejected token point4112 because the runner omitted max_tokens (default4096). No timing rows were emitted. Corrected the explicit linear cap16384 and MoE cap32768; preserve failed run, rerun into generation02.
- D007 approved: implemented identity/per-worker log paths in an independent local vLLM clone (exact branch retained), 2 source files, 44 added/5 replaced lines. Existing clean/profiling checkout remains unmodified. Syntax/diff checks PASS; actual logger and worker validation pending.

- PASS fresh official clean generation01: 400/400 full4096/1024 completions;3 drained warmups;100 unique formal joins. Official server TTFT mean=115.982880592ms;client mean=120.504552970ms. Observed queue trace and ID map generated from these fresh records. No Frontier prediction/error available yet.
- PASS D007 utility tests:9 passed in5.56s in pinned H200 runtime; vLLM source commit6d0f7bbc4 on independent exact instrumentation branch. Actual eight-worker integration remains pending diagnostic execution.

- Diagnostic generation01 FAIL before model load: missing vllm.vllm_flash_attn.layers in the new Git clone. Git tracks only .gitkeep; the working reference checkout additionally contains ignored vllm_flash_attn2.7.2.post1 installed runtime files. Installed the same8 runtime files into the independent checkout (no numeric data/cache copied); source tree remains clean. Exact dependency receipt: analysis/d007_runtime_dependency.json. Generation02 prepared with fresh output paths.
- Fresh profiling generation02: linear phase completed82/82 profile tasks and emitted fresh linear_op.csv; attention currently executing.

## D007 integration recovery and fresh profile completion

- Diagnostic generation02 passed model import after restoring the same ignored FlashAttention runtime dependency, but failed on the first client warmup: `Missing frontier positions meta fields for attn_rope`. The public metadata predicate checked timing scope membership without checking `meta_enabled`; GPUModelRunner correctly omitted metadata production when disabled. A direct execution of the original predicate reproduced expected=False/actual=True. The scoped correction adds a logger metadata predicate and two regression cases; H200 runtime unit validation is running. Timing scopes, inference and official metrics are unchanged. This is a necessary diagnostic logger fix under D007, not a Frontier numerical repair.
- Fresh profile generation02 completed:82 linear rows,2026 attention rows (272 target4096 prefill rows),387 MoE rows. CUDA_EVENT/BF16 metadata and finite/nonnegative populated timing means PASS. Initial shape coverage is verified; runtime operator/routing comparability remains pending. No historical numeric rows consumed.
- Submitted fresh Frontier baseline generation01 with fresh profile inputs, official-clean01 observed queue trace and new predictor/collective caches. This is diagnostic-only until routing/operator semantic gates close.

- Training preflight found missing standalone_legacy MoE gating rows; actual context filter reproduced ValueError. The queued Frontier worker will first collect387 fresh standalone rows, losslessly merge with387 current-task prefill_hot rows via the existing verified merge tool, and revalidate both contexts. No simulation had started when its worker snapshot and manifest were updated.

- Queue checkpoint02:00HK: yc26-h200-fresh-frontier-20260908-01 (submitted01:48:56) and yc26-h200-d007-unit-20260908-02 (submitted01:49) remain Pending. No new runtime artifacts exist. Preserve both jobs through the documented20-minute startup window. Metadata patch preserved in analysis/d007_metadata_pending.patch; do not mark it committed/validated.

- Read-only source checks: model JSONs agree on48layers/hidden2048/head_dim128/128experts/topk8; linear and attention profiling resolve explicit head_dim128. Current shared model manager trains communication CSV models only for VIDUR; collective_sim runtime and skip_cpu_overhead_modeling=true exclude old network/CPU training rows. All new predictor model caches point at the fresh run directory.

- D008 approved and executed: deleted exactly24 inventoried historical predictor cache directories; parent run directories preserved. Receipt: analysis/storage_cleanup_receipt.json. No other path was deleted.

- D007 metadata fix validation PASS: 11 logger tests in5.05s on H200; committed 361d941c97fcec52e544f74b7ab91c54192de9c9. Fresh diagnostic generation03 prepared with clean source and separated operator/routing modes.
- Frontier generation01 FAIL at simulator import: ModuleNotFoundError plotly in profiling environment; training never started. Fresh standalone MoE supplement completed387 rows, merged774 with both required contexts verified. Investigating existing image simulator runtime before retry.
- D008 read-only post-check PASS: exactly24 recorded paths absent and every parent retained; cleanup report added.

- Prepared Frontier generation02: retain completed fresh MoE supplement; audit full simulator imports and native sklearn fit/predict in both pinned-image environments, choose only a passing interpreter with its own library path. New model/collective caches and metrics remain isolated. Diagnostic generation03 submitted at02:18:47HK.

- Fresh MoE merged-artifact verification PASS:774 rows,387 per context, all387 base rows retained, no drops, COMMITTED payload integrity valid. Initial ordinary receipt read failed PermissionError (root0600); read-only sudo verification succeeded without changing permissions/data.

- D007 operators generation03 completed400/400 requests,100 formal. Formal diagnostic client mean215.853714580ms vs clean client120.504552970ms; formal replay171.672841225s vs clean118.973802078s. Instrumentation perturbation is observed and excluded from the primary metric. Streaming identity validation is running over18.03GiB; routing collection follows separately.
- Frontier generation02 remains Pending; predict-only reports3 capable8-H200 nodes, but actual RJob status remains queuing. Capacity prediction is not launch admission. Preserve existing job.

- Frontier generation02 reached20-minute startup window withoutRunning (submitted02:20:56HK, stillPending02:40:56); recorded analysis/frontier_02_queue_checkpoint.json. Retain original RJob, no cancellation/deletion/resubmission. Continue independent routing and operator identity checks. This is queue delay, not numeric execution or a repaired runtime PASS.

- Frontier generation02 scheduled02:43:48HK after22m52s. Full simulator runtime auditPASS only in vidur_te (profiling Python missing plotly/ddsketch). Actual100-request run initialized collective_sim/h2008 and started fresh training02:44:23HK. Environment-recovery substep committedbdd261abafa235cda48c852fe2a05b8eced4ff12; numerical result pending.

- D007 live validationPASS for both operators androuting, each400 requests/100formal/8workers; counts={"operators": {"workers": 8, "formal_requests": 100, "batch_rows": 42048, "detail_rows": 36389760}, "routing": {"workers": 8, "formal_requests": 100, "batch_rows": 48848, "detail_rows": 2344704}}. Exact request-prefillTPcoverage and perworker metadata checked. Worker/helper committede87b0b541a478eb9d2e422f9d92dfc689cff4949; global collective alignment and numeric RCA remain separate.

## Frontier generation02 execution failure after fresh training

- Observed: all fresh compute predictors completed training at18:50:36Z. Simulation initialized18:50:44Z; first prefill finished, then the first decode EP wave failed18:50:47Z. `htsim_runner` returned2 with `missing required fields: [tensor_bytes]`; Frontier exit_code.txt=1. The run does not contain complete request metrics and cannot supply a mean TTFT comparison.
- Preserve all run artifacts and new caches as failed-generation evidence. Bounded read-only RCA is tracing the actual scenario and required-field boundary; no fallback backend or numerical patch has been applied.
- Independent work: extract the first3 formal operator batches per worker and investigate an existing routing-logger extension that could observe the missing global expert-count vector without joining unrelated DP-local batch IDs.

- Formal operator extractionPASS:24 worker-local batches spanningprefill/mixed/decode,21120 raw rows retained(8.92MB). Nested dispatch/combine and fused add duplication explicitly separated. Helper committed42496e98. Source and validation artifacts:analysis/formal-operators-03/. No cross-side comparable-batch claim.
- Backend RCA source finding: first decode wave routes8 assignments toEP0 andzero toEP1–7. Scenario serialization retainszero tensor_bytes; runner merging/required validation treatszero asmissing. Existing EP ZeroPayloadPolicy=PREDICT and nvlink_analytic preserve collective step latency atzero bytes, so an exact-no-op substitute would alter current semantics. Deterministic boundary reproduction pending.

- Backend execution repairPASS:26 tests in27.14s, real zero/nonzero runner and PREDICT policy. Frontier69764e50 pins collective-sime564935. Zero-byte prediction remains0.0035ms; no model formula or topology changed. Prepared fresh H200 generation03, new predictor/collective caches, same current-task profiles and queue trace.

- Submitted fresh Frontier03 at03:06:35HK via persistent service yc26-h200-fresh-frontier-20260908-03; same H200step_main/pinnedimage/exactcase, no old caches.
- Asked D009 global-routing-counts decision through interactive input. Existing topk IDs on CPU allow minimal additional count recording; shared-record edit and routing rerun remain pending explicit answer.

- Mean-comparison helper negative validationPASS: generation02 incomplete request events are rejected before any output/mean creation. Receipt:analysis/incomplete_frontier_rejection.json. Full-success validation and helper commit await generation03 completion.
- Clarified pending D009 population: global counts represent dispatched4097tokens for the first formal prefill, not the4096real-only input. Source-DP slice separation would require a distinct larger extension and is outside the pending proposal.

- Source-grounded clean decompositionPASS: official servermean115.982880592ms = firstSCHEDULED-to-firstEngineCoreOutput80.554247736ms +remaining35.428632856ms. Derived from current clean model_execution/tpot fields; remaining includesqueuewait andoutside-enginework, notCPU-only. analysis/clean_engine_prefill_decomposition.json preserves100formalrows.

- Independent runner reviewPASS for active single-node H200 case. Found a separate multi-server zero-payload chunk-counting division-byzero, not reachable here; deferred in future.md, no scope expansion.
- First-prefill operator qualification completed: exactfreshlinear TP4 row51 andattention rows68/82; pre_proj/rope/cache-save/prefill logical boundaries eligible, post_proj requiresTP4ARgrouping. Replicatednormprofile isfused andcannotstandinfor layer0standaloneRMSNorm. analysis/prefill_operator_qualification.md.

## Frontier03 20-minute startup checkpoint

- 2026-09-07T19:27:23.664456+00:00: yc26-h200-fresh-frontier-20260908-03 remainsPending / RJobisqueuing after1249seconds. Capacity prediction listed5candidate8-H200nodes, butactualadmissionnotgranted. Persistentlauncher/RJobretained; no cancellation,deletion,orresubmission. Receipt:analysis/frontier_03_queue_checkpoint.json.
- Pendingwork:complete100-request Frontierexecution andmeancomparison; obtainD009decision andvalidatefullroutingvector;finishcaseRCA/scopednumericalrepair/freshverification. D009sharedrecordmutationisnotapplied. No FrontiernumericTTFTgatePASS.

## Fresh Frontier03 complete baseline

- LauncherFRESH_FRONTIER_EXECUTION_COMPLETE; simulator metrics completed20:07:35Z. Actual100-request analyzerPASS. Frontiermean98.783926557ms vs officialserver115.982880592ms; absolute17.198954035ms, relative14.828872974%. This is an unadjusted baseline, not a CPU estimate or D006formalPASS. Analyzer committed85240ab5. Report:test_report_2026-09-08_frontier_baseline.md.
- UserDPquestion verified against actual request/batchrecords and pinnedvLLMsource; same-version official source docs fetched via companyproxy after documentationwebsite403. DP2routes independentrequests, permits emptylocalwork; MoEglobalEPrequiresdummyforward participation. Corrected certainty:extra1tokendummy is source-backed inference, notdirectremote-roundlog. D009stillpending.


## Routing import capability audit — D010

- Motivation: YC requires equal Frontier/vLLM routing distributions and asks to reuse an existing import interface if present.
- Completed source audit: moe_routing_trace_path is a deferred configuration field with no file reader. PD-AF rejects it; co-location routing generation does not consult it. ROUTING-SNAPSHOT is output only. Runtime routing maps are static replica/layer/expert ratios; scheduler lookup has no batch key.
- Direct check PASS in dev-vidur-v03-hopper-e2e Python3.13.13: the exact extracted generator returns48 balanced layers with a nonexistent trace path; the real pure materializer reconstructs a synthetic128-count vector exactly at4096tokens/32768assignments; using4097tokens yields32776assignments and different counts. This establishes a reusable materialization boundary, not a working trace importer or measured equality.
- D010 objective recorded; no runtime routing setting changed or new numerical result produced. Proposed D009 revision records both source-DP and dispatch full counts; existing local16 counts are preserved. Shared context/record design awaits YC. Missing batch-specific import and incomplete global source data are newly established issues. Baseline98.783926557ms vs115.982880592ms remains unaligned.


## D011 execution start

User redirected to vLLM existing deterministic uniform routing and CPU-master Frontier. Source VLLM_MOE_UNIFORM_ROUTING dispatches to uniform_topk; round-robin assignments match balanced Hamilton counts for the same population. Added a bounded H200 router preflight wrapper and isolated compilation-cache paths by run parent; clean modes remain free of op/routing/CPU diagnostics. Static shell and diff checks PASS; live validation pending. CPU and profiling preparations delegated independently under explicit parallel request. D012 shared runtime override awaits YC; no core source change applied.

- D011 launch: yc26-h200-uniform-clean-20260908-01 submitted08:43:16HK, scheduledgpu-h200-0661 at08:43:21; GPU runtime probePASS, direct uniform router check executing. CPU runner20c8adf1 is verified and committed. Parallel uniform MoE profiling job submitted by dedicated worker; current task profiles remain separate from historical inputs. D012 concrete proposal saved inanalysis/uniform_runtime_selection_proposal.md.

- Uniform runtime router PASS24checks on8H200; direct CPU Frontier materializer comparison PASS24/24 with zero128-axis count distortion at identical population. Evidence analysis/uniform_routing_same_population.json and test_report_2026-09-08_uniform_routing.md. Source/dispatchpopulation differences remain separate. Clean service initialization continues.

- Uniform MoE profiling COMPLETE/PASS:774freshrows,387eachcontext,43tokenpoints; accepted merged artifact supplements/moe-uniform-01/moe.csv. Initial merge namespace and new-directory permissions issues recovered with existing tool; original worker exit1 is retained, not relabeled as full workerPASS. Corrected future worker committedae9ab866 after source dataset and final merge validation.

- Corrected a launch-manifest template defect before clean result validation: inherited oldstatusPASS/completed_utc/validationreference/Frontiercommit/worker_snapshot replaced with currentRUNNING, actual launchbase20c8adf1 and exact worker snapshots. No old numeric values entered the new measurement. Before/after receipt analysis/uniform_clean_manifest_correction.json.

- Uniform clean COMPLETE/PASS at validation2026-09-08T00:57:09.493519+00:00:400/400full4096/1024requests,300warmups excluded,100formaluniquejoins. Officialservermean127.380511761ms,client133.159564410ms, observedqueue span47.140908699s. Worker completion marker observed; analyzerPASS. Newarrivalmap underanalysis/uniform-clean-01. CPUcandidateconfig prepared but blocked on D012 unresolved sharedruntimecontract; no uniform Frontier result or relative error exists.

- D012 approved by YC; code-change marker recorded2026-09-08T01:09:41.452064+00:00 before first core edit. Parent owns config/resolver/predictor and their tests; CPUworker owns sharedmanager and separate tests. Existing untracked D005 endpoint artifacts preserved.

- D012 implementation verified:54 config/resolver/predictor/CLI checks +20 shared-manager/cache checks PASS. Initial CLI global-state contamination recovered through subprocess test isolation; production guard preserved. Exact commands/failures in test_report_2026-09-08_d012_routing_runtime.md. Ready to commit the six scoped files and execute fresh CPU uniform case.

- D012 committed 9e3d1874fca51491d7f14b4788394811d3e48ffa. Launched CPU uniform generation01 at 2026-09-08T01:21:57.648626+00:00 through persistent systemd user service; memory cap18GiB, fresh caches and fresh uniform queue trace. Run manifest records exact command/config/provenance. Numerical result pending.

- Fresh CPU run passed full predictor training and entered simulation09:29:27HK. Effectiveconfig confirms balanced/uniform_topk/prefixOFF/nvlink_analytic. First-prefill internal48-layer accountingPASS65.790074082ms, waveendpoint65.790076904ms; actual completedrequestendpoint stillpending. D013activationprofiling proposal sent via grill-me surface; no D013 source mutation before explicit answer and active run completion.

- CPU generation01 produced first9complete requests. Request0 formalTTFT65.79007690374297ms matches its EPwave endpoint within2.5704e-10ms;48-layer components close within2.822e-6ms. Actual request0 E2E59096.12969894329ms. Full100-request means remain pending.

## D012 fresh uniform rerun completed

- Completed2026-09-08T02:02:25.622414Z; CPU worker exit0/completionmarker observed;2427.974s wall elapsed including freshtraining.100/100requests complete, each4096/1024;200events and100CSVrecords joined to currentuniformgroundtruth exactly. Artifactand27effectiveconfigchecksPASS.
- Frontiermean103.51040122462338ms vs officialserver127.38051176071167ms; abs23.87011053608829ms, relative18.739217016908402%. Newbaseline replaces priorcomparisons forcurrentcalibration. D006formalPASS notclaimed; noCPUconstantapplied.
- Samecase TPOT57.583941406ms vs78.336091547ms andE2E59.011882459s vs80.264999897s aresecondarydiagnostics. Firstprefillcomponent/endpointclosed. Reports:test_report_2026-09-08_uniform_frontier.md andanalysis/uniform_error_analysis_summary.md.
- D012implementationandreruncompleted at9e3d1874; noactiveCPU/GPUjobfromthisturn remains. D013activationprofiling scope awaitsYC; sourcechangesnotstarted. Newlyobservedexecutionerrors:none. DeferredD005untrackedfilespreserved.


## D014 / D015 continuation — 2026-09-08

- YC deferred gated-SiLU repair and requested first-TTFT batch membership comparison. Frontier current ledger directly records first batch 0: request IDs [0], token vector [4096], no decode tokens, stage interval 0 to 65.790076904 ms. Current uniform vLLM clean run has no batch log because its worker explicitly disables instrumentation. Warmup replay logs show all 300 warmup requests completed before the formal phase. The next formal engine queue arrival is 81.169424579 ms after request 0; this does not alone establish vLLM batch membership or remote dummy participation. No batch equality claim has been made.
- D015 authorizes cloning/reviewing the historical accuracy matrix and attempting replay using its vLLM controls with current Frontier. Clone launched with company HTTP proxy configured. Documentation/tests and concrete semantic differences are under investigation; no new GPU run or activation edit has started.

- 2026-09-08T02:30:06.882881+00:00: Reference clone completed at /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908, branch v0.3-hopper-testbed, HEAD9fd7fea5, tree clean. Reviewed task docs, MoE manifests, runner/extractor/comparator and existing regression suites.22component checks PASS. Historical TTFT implementation uses reconstructed client planned arrival to synchronized batch timestamp; historical clean enables batch instrumentation. Review and exact commands recorded in analysis/historical_accuracy_matrix_review.md and test_report_2026-09-08_historical_harness_components.md. Current uniform baseline percentiles recomputed with historical percentile helper; no new E2E execution claimed. Case scope question is pending YC; independent review and validation complete.

- 2026-09-08T03:11:14.352362+00:00: YC confirmed recommended current-case replay. Bounded code substep: add batch-only selection to existing diagnostic worker and three dispatch timing fields to existing client; no vLLM core/Frontier predictor edit. Planned checks: shell syntax, localhost streamed-token client check, then real clean/batch H200 execution with per-worker identity.

- 2026-09-08T03:18:33.083230+00:00: Replay harness verified and committed897d2482. Shell syntax/diff checksPASS; localhost SSE client checkPASS with two observed tokens and exact dispatch-lag equation. Initial CPU client check failed because aiohttp absent; used isolated /data/ycfeng/tmp/issue26-client-validation-20260908 venv (Python3.13.13,aiohttp3.14.3) after company-proxy installation. No shared simulator environment modified. Launched yc26-h200-historical-replay-20260908-01 on H200 step_main; scheduled gpu-h200-0844, image pull active. Exact launch and mode manifests preserved under runs/h200-historical-replay-01.
- Independently reproduced round-robin incremental DP reset; analysis/round_robin_incremental_rca.md and JSON saved. D016 fix review requested; Frontier core unchanged pending answer. vLLM run continues independently.

- Batch identity analyzer now accepts batch-only mode and preserves full formal prefill membership per worker, committed91f7147a. Synthetic eight-worker identity/prefill fixturePASS; malformed request/token vector rejected. Fixture /data/ycfeng/tmp/issue26-batch-identity-check-fck1by8w is helper validation only, never numerical evidence. GPU launcher source was897d2482; later91f7147a changes only offline analyzer. Record effective worker checkout from environment probe when startup completes.

- 2026-09-08T03:23:46.102376+00:00: GPU worker passed H200 eight-device compute/NVLink probe and24uniform-router checks. Clean server initialization started; no request metrics yet. Effective worker Frontier checkout 91f7147a5ff8b7ea2fccaf48b0227d10df747b4f; launch revision897d2482 is retained separately, subsequent commit changes only offline analyzer. D016 review remains pending; no Frontier core repair applied.


## D016 DP routing research — 2026-09-08

- YC asked whether vLLM uses the same round-robin policy and how Frontier should change. Production code remains unchanged. Research skill delegates exact-version source review; local work verifies Frontier state and runs actual routing methods.
- PASS source-method probe: current RR singleton [0,0,0,0,0,0] versus burst [0,1,0,1,0,1]; vLLM burst without refresh alternates, but fresh empty snapshots before each arrival all choose DP0. At DP0=(waiting0,running10), DP1=(waiting0,running0), vLLM chooses DP1 whereas current Frontier LOR chooses DP0. The vllm_v1 pending property excludes running requests. Exact commands/environment/results: test_report_2026-09-08_dp_routing_source_probe.md and analysis/dp_routing_source_probe.json.
- Read failure: presumed frontier/config/cluster_scheduler_config.py does not exist; rg resolved actual config definitions to frontier/config/config.py:2101. No code behavior was changed for this inspection error.
- Proposed repair is separated into RR continuity and vLLM weighted-load parity. A dedicated registry policy should consume separate scheduler waiting/running state; do not redefine shared num_pending_requests. Snapshot publication and visibility are part of parity, not a hardcoded per-request 100ms refresh. Exact placement is still sensitive to workload/completion timing.
- New execution issue: replay clean/client.jsonl remains232 rows, last updated03:31:12 UTC; server.log last updated03:31:13 UTC. No completed formal100 result is claimed. Company-proxied, timeout60s/MemoryMax2G platform queries returned Bad Gateway and EOF. Worker status is unknown; no restart, duplicate job, cleanup or speculative root-cause attribution.

- Upstream verification: company-proxied curl fetched official v0.10.2 core_client.py successfully; AST routing method equals active clean and diagnostic methods. Concrete proposal saved in analysis/dp_routing_repair_proposal.md. No production code edit or new numerical acceptance result.

- D016 confirmed by YC: independent RR repair plus vLLM snapshot-aware DP policy. Approval recorded in repairs/d016_human_review.json before code changes. Change markers D016_RR_CONTINUITY and D016_VLLM_DP_SNAPSHOT. Earlier source-method reproduction closes diagnosing phases1–4; proceed to focused regression and repair. Skill per-change reapproval does not override explicit authorization of both substeps.

- RR continuity committed ab752f97 after20passingchecks. New snapshot-policy focused suite now73PASS, covering its load model plus existing config/DP/KV-related checks. Initial manual registry probe used a string where get_class requires an enum; corrected probe and all six enum/config registrationsPASS without changing the registry contract.
- Actual pinned vLLM coordinator replay versus new DES model:6001millisecond statesPASS, including previous-step publication and heartbeat; artifact analysis/dp_coordinator_reference_check.json. Initial script invocation lacked PYTHONPATH and failed ModuleNotFoundError; rerun with PYTHONPATH=$PWDPASS.
- Initial real DES diagnostic completed3/3requests, each4096/1024,159907events. Route owners [DP0,DP0,DP1], both lanes in stage ledger; no numerical TTFT claim because operator timing is dummy. The suppressed-report/deadline edge was refined during that run, so a final callback-observed run on the updated code is active before committing the policy.
- Platform visibility remains unavailable: bounded brainctl get returns EOF; explicit company-proxy CONNECT to platform.shaipower.com returns502. Existing handbook contains no verified remedy for this condition. brainctl config --minify is unsupported; its help/options can expose default authentication fields, so subsequent diagnostics must select only non-sensitive fields. No external configuration change, restart or duplicate allocation.

- Final D016 snapshot verificationPASS:73focusedchecks,6001source-coordinator time-state matches,3/3full4096/1024DES requests,2049post-step callbacks (DP0:1025,DP1:1024), final waiting/running zero on both lanes. Selected evidence preserved under analysis/dp_snapshot_final_des_evidence. Numerical accuracy not claimed because operator timings are dummy.
- Committed snapshot policy a4a0c496d370c84d83a5b098d3fdd8561af1c74e after verification. Previous RR commit ab752f97 retained independently. Only the two pre-existing D005 files remain untracked; unrelated work preserved.
- Full numerical candidate prepared at config/frontier_dp_snapshot_candidate.json, selecting vllm_load_balancing and the expected NEW clean arrival trace. Input trace unavailable; do not launch or substitute the historical/current-task baseline trace for this formal replay. Current blocker: platform EOF/502 and incomplete clean artifacts. No active CPU validation job remains.

### Retry recovery 2026-09-08T04:55:49.839205+00:00

- Observed: lowercase company proxy variables conflicted with inherited uppercase localhost proxy and NO_PROXY. Synchronizing both cases restored consecutive platform queries.
- Old replay-01 replica terminated at 03:31:16Z, exitCode=0, reason=Completed; only 232 warmup client rows and no formal rows or replay completion marker exist. Platform success is not experiment success; premature termination cause remains unresolved.
- Prepared fresh replay-02 manifests at Frontier a4a0c496d370c84d83a5b098d3fdd8561af1c74e, pinned clean/diagnostic vLLM commits unchanged, clean trees and remote tip verified. No old measurements copied. Launch will use an independent user systemd service.

- 2026-09-08T04:58:57.365526+00:00: H200 predict-only PASS (three eligible eight-GPU nodes). Submitted replay-02 through user service yc26-h200-historical-replay-20260908-02.service; scheduled gpu-h200-1091, image pull in progress. Company proxy cases synchronized in launcher. CPU candidate now targets analysis/historical-replay-02-clean/frontier_queue_arrivals.csv; no numerical CPU run started before trace validation.

### Clean replay-02 completed; fresh CPU launch 2026-09-08T05:17:20.383819+00:00

- PASS:400client and400server rows, no missing/duplicate/extra IDs,300warmups excluded,100formal4096/1024requests validated; three fully drained warmup phases and one formal phase recorded. Official server TTFT mean131.26863718032837ms; client mean135.40103293ms; first request officialTTFT121.48427963256836ms. New queue-arrival trace spans47.156283407937735s.
- Recovered validation race: automatic followup observed client formal completion before all server metric rows were visible and failed len(server)==400. No output trace or CPU run was created by that attempt. Later complete server artifact has400unique matching rows and passed the unchanged full analyzer. Preserve failed analysis traceback in /data/ycfeng/tmp/issue26-h200-network/cpu-dp-snapshot-numerical-02.log and analysis/replay02_cpu_followup_receipt.json. Client marker alone is insufficient as artifact readiness.
- Observed shutdown TCPStore Broken pipe warnings occur after formal phase completion; finalized per-request metrics and empty-engine shutdown records are intact. No timing impact is inferred from shutdown warnings.
- Launched fresh numerical CPU Frontier via yc26-cpu-dp-snapshot-numerical-20260908-02-r1.service, exact /data/ycfeng/tmp/issue26-h200-network/launch-cpu-dp-snapshot-numerical-02.sh. Raw output /data/ycfeng/tmp/issue26-dp-snapshot-numerical-02/runtime; loglevelERROR; fresh predictor/collective caches; newpolicy. GPU batch diagnostics running concurrently.

### Numerical replay exposed forward progress failure 2026-09-08T05:28:52.987594+00:00

- GPU batch PASS:400clients,100formal4096/1024requests,8workeridentities, eachformalprefill appears once on each TP rank of exactly one DP lane; HISTORICAL_CONTROL_REPLAY_EXECUTION_COMPLETE observed. FirstformalDP0batch4769=request0/4096, DP1batch4804=request2/4096.
- CPU FAIL after fresh predictor training: exit1, Sequential simulation ended with non-empty scheduler state, event_queue_length0 at47.156283407937735s. DP0 running requests0/1 and46queued; DP1 running request2 and51queued. No formal request completion established. Dummy-timing3requestPASS did not exercise this predictor-timed interleaving. No TTFT parity claim.
- Started exact-command diagnostic replay with only fresh output/json event tracing changed, retaining this same-commit run's newly trained caches to avoid retraining during causal diagnosis; final repaired numerical acceptance will use fresh caches. Diagnostic source: /data/ycfeng/tmp/issue26-h200-network/observe_deadlock.py; output /data/ycfeng/tmp/issue26-dp-deadlock-repro-01.

### Shared mixed-phase protocol decision pending 2026-09-08T05:38:20.115048+00:00

- Controlled entry-handler probe:prefill/prefill emits1wave;decode/decode emits1wave;decode/prefill emits0waves and leaves two open rooms. This test bypasses load-balancer selection, distinguishing the synchronization defect from snapshot weights.
- Minimal3request4096/8replay repeats the same0.29563780122719624s stall;2request4096/8negative control completes2/2. These are diagnostic reductions, not new calibration examples. Full formal scope stays4096/1024.
- Prepared analysis/mixed_phase_forward_stall_rca.md and preserved selected raw traces/commands/state in analysis/mixed-phase-stall-evidence/. Proposed shared group synchronization with per-lane phase/shape/completion;5–7production module scope. Asked YC whether to extend into this shared protocol; asynchronous question accepted by the tool, actual decision still pending. No protocol change or substitute has been applied.

Additional control completed: the first2requests with the original4096/1024shape also finish2/2with exit0. Thus the three-request reproduction does not depend on shortening decode length; the mixed-phase third request is required for this observed stall. Evidence: analysis/mixed-phase-stall-evidence/two_request_decode1024_metrics.jsonl.

- Metric workflow adaptation: e2e-metrics-gap references were read. Its legacy normalizer only accepts request_arrival_to_prefill_completion (normalize_request_metrics.py:115–122), which conflicts with YC D006 official-server target. Retain the already verified issue26 official/request analyzers and separate boundary labels; reuse independent count, arithmetic-mean and throughput-window rules. No fabricated canonical timestamps or change to the official target. A guessed historical_clean_analysis.py filename failed read; rg resolved the existing helper to tests/e2e/issue26_official_ttft_analysis.py.

- Run-manifest creation under existing runs/ failed PermissionError13. Runtime output was unaffected. Recorded this generation under writable analysis/cpu-shared-forward-01; preserved existing permissions and worker directories.

- Batch analyzer verified on first3realrequest artifacts:3/3sameDP and3/3sameMembers; committed ff8c229a. It derives the formal namespace/count from the mapping. The active numerical process remains production8ba22b49.
- Independent TPOT equation check FAIL for exact reconstruction from E2E minus TTFT. Source root cause: pinned vLLM output_processor.py:268–280 records completion_time with time.time(), but TPOT uses stats.last_token_ts minus stats.first_token_ts. Mean reconstruction delta0.000073436742334ms/token; maximum absolute0.002321042939812ms/token. E2E wall endpoint equations pass. Preserve native TPOT, do not widen tolerance or substitute reconstructed values. This is a metric-boundary distinction, not evidence of a simulator synchronization defect.
- Full numerical run completed predictor training and began arrivals after about7.5minutes; no failure reported at this checkpoint.

- Three-request prefill endpoint check PASS: each TTFT equals arrival-to-stage-start plus original-source stage interval. Request2=53.445856443ms before stage +65.867846586ms stage =119.313703029ms TTFT. Evidence: analysis/shared-forward-three-request/prefill_boundaries.json. These intervals do not independently identify CPU work; request2 pre-stage admission time is retained by the endpoint.

- First-batch independent timing comparison completed: current full-run Frontier65.790076904ms vs isolated vLLM CUDA batch span77.125022888ms,11.334945984ms/14.696846186percent. Clean official firstTTFT121.484279633ms. Later DP placement mismatches cannot alone explain the first-request gap, and cross-run CUDA/official subtraction is not a CPU measurement. D018 qualifies full-case workflow; operator/critical-path RCA still remains separately pending.

## D018 parallel RCA started — 2026-09-08

- In progress: root coordinates exact-image H200 supplemental runs and diagnostic integration; dp_workflow_rca owns source/control-flow counterfactuals; first_batch_op_rca owns timing boundary and per-op analysis; route_instrumentation owns three existing vLLM route/enqueue logger hooks and validation. All workers have disjoint file ownership and preserve unrelated changes.
- Observed source boundary: route precedes engine QUEUED, while the current Frontier arrival trace uses QUEUED to reroute. This establishes distinct decision instants, not the numerical cause of all placement differences. Missing evidence: actual route snapshots and same-run enqueue identity.
- H200 capacity predict-only passed with two eligible eight-GPU nodes. Company proxy uppercase/lowercase variables synchronized. No new GPU execution started yet. An inspection used an incorrect historical worker filename; rg/actual launcher resolved issue26_h200_replay_worker.sh, with no state changes.

- H200 RCA generation01 launched on gpu-h200-0844, eight H200 runtime compute/NVLink probe PASS. Diagnostic vLLM8453dd342 clean includes route records and bounded first3formal operator selection. Worker a7673346 launch revision updated to26a57780 before server-mode loop to leave unsupported runtime metadata disabled; record effective flags per mode. Source review found metadata emitted outside an active scope; no such exception has been observed in this run.
- Recovered local check invocation error: guessed selection_check.py did not exist; actual issue26_first_batch_op_rca_selection.py passes7cases and unchanged execution-call checks. Worker bash syntax and diff checks PASS; exact reports below. No production Frontier timing/scheduling changed in D018.

- Fresh route/batch mode completed400clients,300warmups excluded,100formal exact4096/1024; existing identity analyzer PASS across8DP/TP workers. FirstDP0TP0 forward80.335617065ms, measured on currentnewH200run; this does not replace cleanTTFT with diagnostic timing. ActualCUDA-event mode manifest confirms runtime_meta=0, prefix filter and limit3, route/scheduler logging off.
- New same-run route records directly confirm client5 routed toDP1 beforeclient6 routedDP0 by2.023212ms; engine QUEUED order reverses (client6 before5 by0.048529ms). Both decisions use precisely observed before-increment snapshots. Frontier new-route/new-enqueue prefix controls are running to separate boundary and state effects.
- FreshFrontier first-prefill48layers/384EP-lane rows collected with fresh caches; sequential time_limit did not stop the diagnostic as configured, so agent explicitly interrupted after complete firstbatch evidence (exit130). This is bounded operator evidence, not completedE2E. Unrelatedtime-limit defect recorded/deferred, no productionchange.

- Workflow first-divergence RCA complete with independent ACCEPT: actual-state selector equivalence400/400; same-run enqueue-input6/8 owners vsroute-input8/8 with identical predictors. Prefill membership5/7→7/7 but request0 prior scheduled tokens4097→4098, versusvLLM4096; single-arrival substitution is not a complete repair. Full100timing-feedback attribution remains explicitly unquantified. No active workflow CPU tasks.
- Full CUDA-event operator supplement completed400requests with24selected rank-batches. FirstDP0TP0 instrumented span116.210144ms vsfresh batch-only80.335617ms; collective rank skew shows profiling perturbation. Analysis corrected an initially wrong49AR-scope expectation: actual default log has embedding AR only, attention AR uses an excluded scope name inside inclusive projection. Raw failure retained; actual scope table used.
- To resolve this observed perturbation, started an independent H200 step_main communication-only diagnostic via the existing scope allowlist (97pairs perfirstprefill:48attentionTP+48MoETP+1embedding). Same4096/1024,3warmups+100formal,uniform/prefixOFF settings. Worker e108696a; no vLLM or Frontierproductionchange. Outputanalysis/h200-rca-comm-01. Capacity preflight found3nodes; briefqueue then scheduledgpu-h200-0761. Currentkerneltrace run continues independently on0844.

- RCA01 record_function mode FAILED (launcher exit1): third selected DP0TP0 batch4736 has5kernel launches without device activity in the raw Chrome trace, including embedding AR correlation96038. Firsttwo selected DP0TP0 traces have0missing launch correlations. Existing collector correctly rejects absent timing; no zero-value fallback or frozen-source edit. Specific CUPTI startup/buffer mechanism remains unproven. Completed batch/route and CUDA-event modes remain separately valid diagnostics.
- Communication supplement passed eight-H200 compute/NVLink probe on0761; effective mode manifest confirms the three intended AR scopes, CUDAevent/default/per_scope, runtime_meta0, uniform routing and bounded3formal selection. vLLM warmup is active.

- D018 communication supplement completed400requests with3drainedwarmups+100formal, worker completion marker, launcher exit0/inactive. Independent batch identity PASS100formal/8workers; scoped analyzer PASS24rank-batches/2328finitepositive unique97-scope rows. Firstforward86.412033–86.443520ms; attentionTP5.537248–5.708768ms vsFrontier17.899440; extraMoETP4.792544–28.413280ms remains rank-wait sensitive. Numerical additive attribution is incomplete; no fitted correction.
- Integrated RCA and next workflow design proposal recorded in analysis/parallel_rca_summary.md and analysis/dp-workflow-rca/next_step_proposal.md. Diagnostics/parser code substeps committed; git diff --checkPASS and onlytwo pre-existing D005 files remain untracked. No active CPU/GPU work remains. Full100feedback decomposition, threealignedopgate, deeperCUPTIcoverage cause and production correction review remain open; originalcleanTTFTgap19.673373percent unchanged.

## D019 review and replanning in progress

- YC challenges the operator-completeness claim and requires all compute/memory mappings, separate communication gaps, a concrete route-time example, and a parallel plan for review. Acknowledge operator mapping/correction and forward CUDA-span repair remain incomplete. Prioritize CUDA closure before CPU workflow integration; no new GPU runs or production changes in this planning turn.
- Parallel read-only reviews assigned: existing operator author audits full comp/mem/comm coverage; independent reviewer inspects historical CPU test suites; workflow author inspects current profiling coverage and the minimal predictor-vs-implementation separation. Root owns clarified timeline, requirements and canonical plan.
- Inspected actual trace builder: route input sorts recorded route times, normalizes to first route, writes arrived_at and an ID mapping; enqueue control does the equivalent with engine QUEUED. No DP assignment is injected into either trace. Both retain existing scheduler/predictor inputs.
- A skill read used an incomplete path and failed; corrected to skills/frontier-calibration/op-gap-analysis/SKILL.md. A guessed notes.md was absent; existing historical_accuracy_matrix_review.md supplies the reference clone path. No runtime state changed.

- D019 read-only audits completed: all15Frontier comp labels and38device families (32comp/mem,6comm/staging,3221activities) inventoried with mapping, timing family and missing rows. Root independently verified label coverage, numeric event-gap arithmetic and common-origin route5/6 timeline. Initial verification misread a descriptive missing-scope string as numeric; corrected the selector, preserved evidence, thenPASS. Exact command/failure in test_report_2026-09-08_d019_readonly_audit.md.
- Current predictor exact-feature lookup precedes RF. Multiple first4096P values equal measured CSV anchors; no blanket sparse-RF cause established. Actual MoE CSV local counts4081/4058/4073 with nonzeroCV do not match perfect-balanced runtime, despite routing metadata; current source separates gating and shuffle/GEMM input construction. Historical immutable producer-source receipt remains unverified, so retain that limit.
- Historical CPU component source/test-body audit complete. Existing validatedCSV/predictor interfaces are reusable, but fixed256/3B JSON adapter and gap-to-Ray/clipping/copied-shape helpers are not physical4096CPU measurements. No historical numerical values or testPASS imported.
- Canonical D019 parallel plan prepared and independently reviewed; incorporated centralized A/B/C GPU scheduling, evidence-driven minimal reruns, and inventory-versus-numerical-closure distinction. Execution is pending YC review per explicit current request. Per-op correction and CUDA-span repair remainINCOMPLETE; CPU/workflow integration comes later. No active newGPU/CPU jobs or production edits; existingtwoD005untrackedfiles preserved.


## D019 fresh H200 submission — 2026-09-08 13:00 UTC

- completed: A diagnostic scopes/DP CPU metadata committed through4bc1bc026c91dff78bd7cf5ba6411f14d15e043d; source computation AST unchanged. Root workers committed7b912c3a and388ae7a3. Independent MoE source review ACCEPT; GPU numerical validation pending.
- in-progress: yc26-h200-d019-compute-20260908-01 on gpu-h200-0928 pulls approved fixed image; batch-before -> attention -> MoE -> detail -> batch-after, original400records each. yc26-h200-d019-profiles-20260908-01 submitted separately for four numerical cases -> exact MoE/profile repeats ->8rank comm microbench. All modes serial within each dedicated8H200 allocation.
- CPU P audit verifies exact gating and RF shuffle/GG paths; final receipt pending precision-safe serialization. No new numerical correction/acceptance claim.


## D019 verified correction and current interruption point — 2026-09-08T13:06:52.679507+00:00

- completed: MoE coverage production substep committed5dd5ee39 after independent source ACCEPT and GPU4/4 bitwise cases PASS (14.64s);27 exact-routing profile numerical comparisons and5400 component samples checked. Source paths include activation/reduction and same shared workspace as vLLM. Full op/CUDA/E2E closure remains pending.
- completed: actual P audit9 measured-exact models and2 RF models. New linear repeats differ from current anchors by approximately+1.1 to+6.5percent, insufficient to explain the old QKV/RoPE E discrepancies.
- recovered failure: H200profiles01 completedMoE/linear then attention output serialization raised TypeError: AttentionBackend is not JSON serializable; communication never started. Preserve originalraw evidence. Helper converts the known enum field through value; actual conversion statement JSON testPASS. H200profiles02 resumes attention/communication only with fresh output. GPUrepeat validationpending.
- in-progress: H200compute01 finishedenvironmentprobe and starts batch_before; originalfive400requestmodes continue on dedicated8H200. H200profiles02 separate8H200 imagepull. Both fixedimage/proxy/step_main; noH800.
- pending key decision: D020 bounded communication runtime/config/physicalDPdescriptor proposal independently reviewed. Currentideal andvLLMnaive differ in ownership/order, not just scalarcost. Newshared implementation paused untilYC; existingA/Bmeasurementscontinue.
- pending: newcompleteVscopegroups; exactCSV ingestion preserving oldtiming provenance andcorrectedGG scope; samerunP/S/V+C tables; firstforwardCUDAgate; freshcleanE2E; onlythenCPU/workflow.


## D020 resolved by explicit deferral — 2026-09-08

- completed: YC retained ideal communication, deferred vllm_naive optionalmodeling; recorded inrequirements/plan/future. No newprotocolimplementation remainsawaitingapproval.
- completed: recoveredprofiles02 attentionexport and8rankcommmicrobench execution, formalnumericalverification ongoing.
- in-progress: A newVscopeanalysis; B correctedGGdataset/predictorfirstforwardintegration; C actualcommmeasurementanalysis withinideal. Root retainsGPUmonitoring andintegration.
- pending: fullP/S/Vcoverage andqualifiedCUDAgate, freshE2E, thenCPU/workflow. No numericalacceptance implied byprotocoldeferral.

- in-progress: profiles03 submitted after fresh8H200 capacity PASS (1083/0710/1070), serial multi-size TP sweep with16MiB holdout -> linear event/context controls. Frozen diagnostic source snapshots and execution_manifest.json saved before submit. Existing successful MoE/linear/attention runs are not repeated.

- observed failure: fresh Pnew MoE admission rejected mixed CUDA_EVENT/cuda_event labels. Dataset producer now uses existing enum normalization, preserving failed v1 and measured values; no production loader bypass. Fresh linear/attention caches remain valid within the same task, MoE retry pending.

- completed: Pnew first-stage diagnostic72.327535217ms (old65.790076904), correctedGG measured-exact9rows .461944884724ms/layer, other10computeP unchanged; actualruntime dropped774obsoleteGG targets. This is not CUDA closure because current50us AR term still overpredicts.
- completed: C sweep8ranks/80ops/560events and40kernel-size correlations PASS; fixedNVLink parameters with four training sizes and16MiBheldout supports existingAR-floor candidate4.384788772964477us/step; fullcontext residual remains distinct.
- recovered diagnostic error: synthetic GPTModel executes2embeddings/iteration, so40pairs/20iterations were valid. Updatedcontextdiagnostic validates actualwarmup multiplicity and retainsallpairs; profiles04 submitted context-only, source snapshots preserved.

- completed: scoped GG+AR integration actualfirstforward59.190354384ms, allcomputequeries unchanged, zeroRFfits, freshCCcache; independent AR-delta reconciliation PASS. CUDAgate FAIL versus bracket79.307/78.119ms (−25.366/−24.230percent), showing priorMoE-only near-parity was error cancellation. NoCPU add-on or naive implementation.
- completed: profiles04 contextdiagnostic72eventrows/608launches/0missingcorrelations PASS, QKV/RoPE/outproj identities unchanged; hotcontext effect material while eventprecreation small. Productioncontextintegration proposal remainsunderreview. Workerphasecommitd480a7d0 afteractualGPUverification.

## Resume checkpoint — 2026-09-08

- completed: Re-read handoff, requirements, active plan, candidate validation, integrated first-forward report, compute mapping, and context diagnostic. Git state is clean except the two pre-existing D005 untracked scripts.
- completed: Independent residual review confirms the largest descriptive gaps are attention projection/context and `moe_shuffling`; the isolated linear candidate is diagnostic only because its observed 54.914898417 ms boundary fails the CUDA gate and worsens the gap.
- in-progress: Verify whether model-level `embedding`, `attn_output_init`, and `final_layernorm` terms have a real Frontier insertion path. Preserve `moe_sum` as already included in repaired grouped-GEMM and avoid scalar residual fitting.
- pending: YC decision on any general shared profiling-context contract; fresh scoped CUDA validation; clean E2E; conditional CPU/workflow and official metric validation.

## YC rethink request — CUDA span RCA and gate status — 2026-09-08

- confirmed: Formal per-operator correction gate is not met. Synthetic Frontier versus diagnostic vLLM gaps are +70.13% (`attn_pre_proj`), +65.18% (`attn_rope`), +54.36% (`attn_post_proj`), +90.57% (`moe_shuffling`), and +12.45% for repaired grouped-GEMM plus sum. Only the bounded communication primitive holdout is within 10% (0.644%); in-context AR and the full CUDA span remain outside the gate.
- confirmed: Operator RCA is not closed. MoE activation/reduction repair and scoped AR fit are verified substeps; no clean same-semantics per-op convergence or end-to-end CUDA closure exists.
- confirmed: Instrumented vLLM probes are materially above clean batch-only references: attention +12.956772 ms, MoE +13.568832 ms, detail +17.377251 ms versus the 79.307358 ms clean-before TP0 span. Therefore the current diagnostic V operator values cannot be treated as clean ground truth for fitting.
- clarified: Context/queue attribution means a controlled same-wrapper comparison of synthetic, synchronized-drain, and existing prefill-hot execution contexts, examining CUDA-event elapsed time together with kernel identity and profiler launch gaps. It was justified by identical kernels with large timing differences; it is diagnostic evidence, not an added CPU term or a production correction. Existing results show a material context effect and profiler perturbation, so broad additional context work is deferred until clean span comparability is established.
- pending: YC review of the prioritized CUDA-span plan: clean low-perturbation span reconciliation -> attention projection context/root-cause closure -> same-run MoE shuffle/grouped-GEMM attribution -> scoped communication recheck. Model-level missing terms and shape coverage remain lower priority.

## CUDA sign-discrepancy RCA — 2026-09-08

- Motivation: YC questions why mostly larger Frontier compute predictions yield a smaller whole CUDA span and requests investigation of large communication scopes before discussing further instrumentation.
- Expectation: Reconstruct actual Frontier accounting and existing same-run vLLM scopes, then distinguish physical communication from participant waiting without cross-run additions.
- Method: Parallel read-only communication source/raw-row review by `/root/comm_large_spans`; root independently recomputed every retained finding and inspected same-operation host/device timelines. The `analyze` skill was applied to the code/evidence investigation. Only required task documentation is updated.
- Result: Frontier `50.177408284 + 4.762262367 + 2.125341867 + 2.125341867 = 59.190354384 ms`, matching the actual bounded receipt. Existing communication-only raw records already contain post-MoE TP4 totals `4.792544/25.384416/28.413280/27.393568 ms` across TP0-3, with maxima `0.102400/0.777888/0.770144/0.705856 ms` per call. All four sets contain exactly 48 calls; outer spans are `86.412033/86.437950/86.414658/86.443520 ms`.
- Result: Full-scope batch4706 TP0 ledger is `41.336256 compute + 17.231616 projection/attention-AR + 6.981152 dispatch + 4.196768 DP combine + 37.563168 post-TP + 0.222336 embedding-AR + 8.678848 uncovered = 116.210144 ms`. Outer-only EP scopes and excluded nested adds prevent duplicate accounting.
- Result: Same-run profiler batch4734 ordinal28 post-MoE AR shows TP2 entering at relative0 ms for1.764004 ms; TP1 submits at1.589029 ms, starts at1.667748 ms, runs0.096800 ms; all four end within approximately1.4 us. Common trace base and3077/3077 launch correlations per TP rank were checked. This supports participant-arrival waiting in the captured trace, not a clean-run20ms attribution.
- Decision: The previous clean-reconciliation-first recommendation is superseded. Existing data already identifies major communication/wait terms. Prioritize their physical protocol and critical-path ownership in further RCA. D020 retains ideal EP; no protocol change or additive correction is performed. Small model terms, broad GG coverage, custom decode AR, and outside-forward CPU remain downstream.
- Pending: Quantify how the identified physical protocol/participant effects contribute to the latest78-79ms batch-only reference; discuss a targeted evidence gap only after this RCA. Source/report details and limitations are appended in place to `analysis/d019-communication.md`.

## Span identity clarification — 2026-09-09

- Motivation: YC identified an apparent contradiction between the reported `116.210 ms` and approximately `86 ms` batch CUDA spans.
- Expectation: classify each value by batch identity and instrumentation, then select one canonical calibration reference without combining incomparable runs.
- Method: cross-checked the full-scope RCA manifest/ledger, the reduced communication manifest/rows, and the batch-only before/after records.
- Result: `116.210144043 ms` is the observed full-scope diagnostic span for batch 4706; `86.412033081–86.443519592 ms` are the observed reduced-communication diagnostic spans for batch 4250 across TP ranks. They are different runs with different probe scopes and cannot be added, subtracted, or treated as one forward. The calibration references remain batch-only `79.307357788 ms` before and `78.118782043 ms` after. Neither diagnostic span closes the operator or CUDA gate.

## TP0 post-MoE AR rank-stability RCA — 2026-09-09

- Motivation: YC asked whether TP0's smaller post-MoE TP AR scope means rank0 performs extra CPU/model work and becomes the critical path.
- Expectation: compare multiple aligned batches across independent diagnostic runs, verify operator counts and source rank layout, and distinguish rank-local work from collective participant waiting.
- Method: independently rechecked reduced communication batches 4250–4252, full RCA batches 4706–4708, formal batches 3851–3853, DP1 rank changes, source layout and Qwen3 MoE branches. A fresh communication-only H200 repeat was submitted as `yc26-h200-rank-stability-20260909-01`, but remains Pending because the queue reports H200=0 available.
- Result: reduced DP0 batches consistently made TP0 the shortest AR scope (4.792544/4.809696/2.390848 ms), while full RCA batches made TP2 shortest (5.652832/5.638912 ms) and formal batch 3852 made TP1 shortest (8.274688 ms). All aligned ranks retained 48 post-MoE AR calls. Outer spans stayed nearly equal while per-rank scope sums differed by tens of milliseconds. Source inspection found no active rank0-only Qwen3 operation or extra bias path. The evidence closes the rank0-extra-work hypothesis as unsupported and attributes the observed short scope to batch-dependent participant arrival/queue/wait skew; the exact clean-forward split between routing load and host scheduling remains open. No production code changed. Full details: `test_report_2026-09-09_rank0_post_moe_rca.md`.

## Post-MoE wait attribution RCA — 2026-09-09

- Motivation: YC asked why rank-local post-MoE AR sums differ by roughly 20 ms while outer batch spans are nearly equal, and whether the late participant actually slows the batch.
- Expectation: align each rank's preceding MoE scopes with its post-MoE AR scope and test whether the observed wait is overlapped by the late rank's preceding work.
- Method: independently summed `moe_gating`, dispatch, shuffle, grouped GEMM and combine scopes per rank/batch, then correlated them with the 48-layer post-MoE AR sum. Recomputed Pearson values on 24 rank/batch samples per validated run.
- Result: pre-MoE versus post-MoE AR correlation is `-0.7603954471` for `h200-rca-01` and `-0.4868734905` for `formal-operators-03`. In batch 4706, TP2 has 75.664 ms pre-MoE versus roughly 48.5–49.0 ms on peers and only 5.653 ms post-AR versus roughly 37.6–37.9 ms on peers. In batch 3851, TP0 has the largest pre-MoE scope (66.231 ms) and the shortest post-AR (5.250 ms). This supports: late rank pre-work delays the shared collective completion, while early ranks record overlapping wait. The late rank's preceding delay can drag the outer span, but the 20–30 ms rank-local scope difference is not an additional serial term and must not be summed across ranks. Detailed evidence: `test_report_2026-09-09_post_moe_wait_attribution.md`.

## Synchronization-anchored post-MoE AR RCA — 2026-09-09

- Motivation: YC requested a causal explanation for recurring late TP participant submission, starting from the nearest completed synchronization point, with an explicit AR bypass/payload isolation experiment.
- Expectation: distinguish DP-pair combine completion skew, host submission delay, queued device work and TP payload/backend cost; do not infer cause from rank-local AR scope alone.
- Method: four read-only lanes ran in parallel. Source review verified DP groups `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]` and TP groups `[0..3]`, `[4..7]`; trace review compared combine and post-AR absolute starts across batches 4734–4736; an isolated vLLM diagnostic copy added env-gated `normal`, `scalar_sync`, and `skip` branches. The frozen diagnostic checkout remains at `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`; patched copy is `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab` at `e60f4dfd5e8f362a4f0625dd4de3fcfee1516efd`.
- Result: combine is a DP-pair all-reduce, not a TP-wide barrier, and is enqueued asynchronously on each rank's current CUDA stream. Existing traces show latest combine arrival matching shortest post-MoE AR scope in 48/48 layers for batch 4734, 45/48 for 4735 and 30/48 for 4736. First-layer batch 4734 combine-start spread is approximately 1.31 ms median and 1.665 ms maximum, while combine duration is about 0.08 ms. This closes the prior premise that combine equalizes TP arrivals; it does not yet split local compute from host/stream queue delay.
- Execution status: the previously authorized baseline job `yc26-h200-rank-stability-20260909-01` is running against the restored clean diagnostic checkout. Scalar and skip runs are prepared in the patched copy and will be submitted serially after the baseline completes, using independent persistent output directories. No Frontier production code changed.
- Lane D result: fresh routing artifacts show actual local-load differences after uniform routing metadata: for one DP0 batch, local routed counts across TP0–TP3 are `4176/4083/4296/4081` (5.2% max spread), while the DP global token buffers are `4097` and `12289` because of dummy-token padding and cross-DP count exchange. The current route logger does not persist post-combine `states.shape`; a fresh boundary probe must record it before making any payload-size claim.
- Scalar A/B result: fresh normal batch `4193` and scalar-sync batch `4179` each recorded 48 post-MoE collectives per TP rank. Normal sums were `30.775808/19.624416/6.887840/26.885280 ms`; scalar sums were `26.297856/0.884320/25.805088/18.144064 ms`. Outer spans remained about `89.205–89.239 ms` for scalar and `90.057–90.091 ms` for normal. Replacing the hidden-state payload with one element did not remove the 20–25 ms rank-local spread, so payload size is not the primary cause. The split between local routed work, host delay and queued device work remains open; bounded skip is next.
- Skip status: `yc26-h200-ar-skip-20260909-01` failed before request submission because the client rejects `warmups=0`; all operator/batch artifacts are zero bytes. The diagnostic source now has boundary logging at commit `0d633a946` (combine return, TP AR call/return, shape/rank/host monotonic timestamp). Since H200 skip was invalid, matching bounded `normal`, `scalar_sync`, and corrected `skip` jobs were submitted on H800 with `warmups=3`, `requests=1`: `yc26-h800-ar-normal-20260909-01`, `yc26-h800-ar-scalar-20260909-01`, `yc26-h800-ar-skip-20260909-01`. All three are currently starting/queued; no H800 timing result yet.

- 2026-09-09 H800 diagnostic execution checkpoint: `yc26-h800-ar-normal-20260909-01` failed before worker start with platform `UnexpectedAdmissionError` (`no healthy devices present ... mellanox.com/mlnx_rdma`); no timing artifacts. Initial `yc26-h800-ar-scalar-20260909-01` and `yc26-h800-ar-skip-20260909-01` reached worker but stopped in the H200-specific environment probe: H800 NVML reports `NVMLError_NotSupported`, and the probe's `CUDA_HOME`/`PATH` combined assignment triggered `CUDA_HOME: unbound variable`. These runs produced environment logs only and are invalid timing runs. A dedicated H800 probe was created at `/data/ycfeng/tmp/issue26_h800_environment_probe.sh` with explicit H800 checks, capability logging, and corrected sequential environment exports; the three mode workers now source it. Reruns `yc26-h800-ar-normal-20260909-02`, `yc26-h800-ar-skip-20260909-02`, and `yc26-h800-ar-scalar-20260909-03` were submitted with the same diagnostic commit `0d633a946`, `warmups=3`, `requests=1`; as of this checkpoint they are queued/starting, with no timing evidence yet. Existing failed/empty runs are preserved.
- 2026-09-09 H800 rerun `-02/-03` reached workers after probe fix but all three server processes failed during EngineCore KV-cache initialization: H800 has ~79.19 GiB total, only 477 MiB free, and the inherited `--num-gpu-blocks-override 310809` requested another 2.37 GiB. Each rank emitted 144 boundary rows during model memory profiling, but no client request, batch rows, or operator timing; these rows are invalid for first-forward RCA. H800 workers were adjusted to `--num-gpu-blocks-override 4096` and reruns `yc26-h800-ar-{normal,skip}-20260909-04` and `yc26-h800-ar-scalar-20260909-04` were submitted.
- 2026-09-09 H200/H800 separation: independent H200 diagnostic workers were prepared with H200 probe and original 310809 block override, then submitted as `yc26-h200-ar-{normal,scalar,skip}-20260909-05` using `step_main + h200`; H800 jobs remain `codesign + h800` with the reduced 4096 override. No cross-cluster timing comparison is permitted.
- 2026-09-09 H800 memory-safe `-04` reruns passed H800 runtime probe and entered real vLLM request execution. All three jobs (`yc26-h800-ar-normal/scalar/skip-20260909-04`) are `Running`; server logs show `Running: 1 reqs` and batch/boundary files are non-empty and growing. The 4096 block override avoided the prior KV-cache OOM. Complete client phase is still in progress, so no first-forward reconciliation has been performed. H200 `yc26-h200-ar-{normal,scalar,skip}-20260909-05` remain separately queued under `step_main + h200`.
- 2026-09-09 continued polling: H800 `-04` normal/scalar/skip remain `Running` with active request (`Running: 1 reqs`, generation throughput ~0.3–1.1 tok/s); boundary and batch artifacts are non-empty and growing (rank0 boundary files ~19–23 MB; batch rows ~0.28–0.33 MB at checkpoint). Complete client logs remain zero bytes because the 4096+1024 token warmup/request sequence is still in progress. H200 `-05` normal/skip are `Starting`, scalar is `Running`; no H200 timing interpretation yet.
- 2026-09-09 continued: all six separated reruns are now active (`h800 -04` normal/scalar/skip and `h200 -05` normal/scalar/skip). H800 server logs show one formal request and generation throughput ~0.2–1.0 tok/s; H200 server logs also show one request and generation throughput ~0.4–0.6 tok/s. Client logs remain open while the 1024-token decode completes; batch and boundary files are non-empty. No reconciliation or RCA claim has been made yet.
- 2026-09-10 continued: H800 `-04` completed the bounded client request (`client.jsonl` now present; 1024 batch rows on rank0 for each mode) while RJobs remain in worker cleanup. Observed client metrics: normal TTFT `83622.728559 ms`, E2E `1313552.909898 ms`; scalar_sync TTFT `85808.417517 ms`, E2E `1269712.959884 ms`; skip TTFT `72944.941725 ms`, E2E `1257150.520207 ms`. These are diagnostic run metrics only; boundary files contain many warmup/decode rows and still require request-window alignment. H200 `-05` jobs also remain Running with active request logs.
- 2026-09-10 boundary semantics clarification: `combine_return` is recorded immediately after `get_ep_group().combine(states)` returns; `tp_ar_call` immediately before `maybe_all_reduce_tensor_model_parallel(states)`; `tp_ar_return` immediately after that function returns. `combine_return→tp_ar_call` is a host timestamp gap after the DP-pair combine call returns and before TP AR invocation; it is not DP CUDA completion time. `tp_ar_call→tp_ar_return` is the host elapsed scope around the TP AR function, inclusive of Python/NCCL enqueue and any blocking/stream interaction observed by the caller; it is not guaranteed physical NCCL kernel duration. H200 first request windows are now extracted for normal/scalar/skip (48 layers × 3 phases × 4 TP ranks) with median gaps ~3.9 ms and layer-level cross-rank spreads 19–74 ms. H800 first request windows remain summarized in `analysis/h800-ar-04-boundary-summary.md`. Final causality still needs CUDA event/NCCL completion evidence.

## Diagnostic mode intent clarification — 2026-09-10

- YC clarified that the primary experiment is a same-cluster comparison of the normal path against a skip-post-MoE-AR path, using the same first 4096-prefill batch and the DP0 TP0–TP3 outer batch span. `scalar_sync` is retained only as a secondary payload-versus-rendezvous probe; it is not a required third production-like condition.
- The valid artifacts align `batch_id=0` to the same warmup request on TP0–TP3 within each run. H200 outer-span candidates are normal `18087.072266 ms` and skip `19656.929688 ms` (`+1569.857422 ms`, `+8.679%`); H800 candidates are normal `24219.837891 ms` and skip `18703.576172 ms` (`-5516.261719 ms`, `-22.776%`). These independent diagnostic runs show why the skip delta cannot yet be called pure AR cost: skip changes the hidden-state reduction semantics and instrumentation remains active. Cross-cluster values are kept separate.

## H800 ten-warmup formal-span checkpoint — 2026-09-10

- Motivation: YC requested ten warmups and a fresh formal first-forward CUDA span.
- Expectation: inspect completed artifacts, verify drained phases/identity, report actual scale.
- Method: bounded analyzer over 11 client rows and all 8 worker logs; inspected CUDA event measurement source and completion marker.
- Result: bounded identity/drain PASS; DP0 batch5120 TP0-3 spans 519.799133301/522.410522461/522.551147461/522.017639160 ms; median522.214080811, P90 522.508959961, max522.551147461, spread2.752014160 ms. Normal-scale reproduction FAIL. Previous 3-warmup DP1 max684.848632812ms; difference-162.297485352ms. Different node/DP lane prevents causal warmup attribution.
- Modification: corrected bounded worker formal_requests metadata100->1 after confirming CLI and completion records; raw manifests preserved. Shell syntax and both actual 3/10-warmup analyses PASS. New analyzer verifies bounded cases without weakening the standard400-row validator.
- Limitations: ten one-request warmups, not the full standard suite; batch worker routing flag unqualified; CUDA event spans can contain host-induced gaps. Boundary per-call file I/O is present but its contribution is unmeasured.
- Evidence: analysis/h800-ar-176000-normal-10warm-01/report.md and formal_span_validation.json. Ten-warmup request completed; normal/skip/scalar and CUDA closure remain pending.

- Verified code substep committed as `d9fdf61a`: bounded H800 worker and formal-span analyzer. Existing identity-validator edits and other workers remain untouched.

## H800 standard replay reproduction complete — 2026-09-10

- Motivation: YC requested reproduction of the existing H80080.655ms reference.
- Expectation: same176000blocks, standard3x100warmups/100formal, uniform routing and first-formal identity; approximately80ms forward scale.
- Method: continued existing RJob yc26-h800-standard-replay-176000-repro-20260910-01 on gpu-h800-0496; compared source/env/server args against h800-standard-replay-05; standard identity validator and independent drain/shape/statistics checks.
- Result: Succeeded; clean400/batch400rows, warmup300/formal100each,24router checks PASS,8worker identity PASS. DP0 batch4711 TP0-3 spans77.486846924/77.460639954/77.482528687/77.484802246ms; median77.483665466,P90 77.486233521,max77.486846924,spread0.026206970ms. Median versus historical80.654880524ms is-3.931833%; scale reproduction PASS.
- Issues: TCPStore Broken pipe during shutdown and shared-memory cleanup warnings retained, after complete requests; platform terminal success confirmed. Prior claim that no valid H80070-110ms record existed is corrected by both original raw batch4696 and fresh batch4711.
- Evidence: analysis/h800-standard-replay-176000-repro-01/{report.md,identity_validation.json,reproduction_summary.json}. Standard reproduction complete; paired normal/skip and CUDA causal attribution remain pending.

Verified standard replay scripts and warmup identity handling committed as `07dbbdeb`. Four shell syntax checks and the real400-row batch validator PASS. Only the two pre-existing D005 scripts remain untracked.

## Minimal bypass preparation — 2026-09-10

- Motivation: isolate AR removal from the old diagnostic harness changes.
- Expectation: standard warmup/identity and a comparable outer CUDA span.
- Method: source /data/ycfeng/tmp/issue26-vllm-post-moe-bypass-20260910 at e29a8f925216d517bcbf7ffad47b93970d72d1f5, one call commented; two existing H800 workers accept source path with default unchanged. Independent audit completed in parallel; GPU scheduling remains central.
- Result: exact one-call diff and AST/shell syntax PASS; actual GPU validation pending. Preflight and launch under analysis/h800-standard-post-moe-bypass-01/.
- RCA correction: 522.214 ms and 675.207 ms medians came from normal AR with boundary logging, one-request warmups and absent uniform flag. They do not establish skip-induced slowdown. Boundary timestamp gaps include synchronous per-row logging and scheduling; complete attribution of the excess remains open.

- Launch failure: bypass-01 stopped before worker execution because local timeout interrupted rlaunch and rlaunch sent stop. Corrected by running the same launch without a short lifetime limit; independent status queries retain timeout/MemoryMax. Replacement bypass-02 uses fresh outputs. No GPU timing result from -01.

## Bypass runtime dependency repair — 2026-09-10

- Motivation: bypass-02 exited before requests with missing vllm.vllm_flash_attn.layers, despite platform Succeeded. Worker exit code was 1.
- Expectation: identical runtime dependencies to verified normal source, preserving single-call bypass.
- Method: copied eight ignored FlashAttention Python/binary assets from canonical source into missing isolated paths; tracked diff remains only one AR call.
- Result: asset presence and clean tracked tree PASS; fresh bypass-03 GPU validation pending. No timing data in -02.

- 2026-09-10 09:09 UTC: bypass-03 scheduled to gpu-h800-0600; Starting/ContainerCreating, image pull ongoing, restartCount=0. Eight copied runtime assets byte-equal to normal source. No client or batch evidence yet. Continue same job, launch session82636 and /data/ycfeng/tmp/issue26-h800-bypass-03-launch.log. Worker script changes remain uncommitted until actual GPU verification.

- 2026-09-10 09:25 UTC: bypass-03 on gpu-h800-0600 completed clean mode: 400 rows, 300 warmup/100 formal, all 4096 prompt/1024 observed output, all four rounds 100 complete and drained PASS. Server args equal normal baseline; runtime source e29a8f925 and uniform flag1 verified. Clean shutdown emits TCPStore warnings after all clients complete, as also seen in normal reference. Batch mode pending.

- 2026-09-10 09:33 UTC: batch three 100-request warmup rounds drained. First formal cmpl-pf4096_dc1024:0-0 selected on DP0 batch4718, TP0-3 77.940704346/77.919746399/77.931137085/78.142173767ms; median77.935920715,max78.142173767,spread0.222427368ms. Same [4096,1] DP counts as normal. Formal100 drain and complete identity validation pending; preliminary normal-scale result only.

## Minimal post-MoE bypass completed — 2026-09-10

- Motivation: test the user's simplest bypass under the reproduced standard workload.
- Expectation: qualified full warmup/identity and direct normal-vs-bypass first-forward comparison.
- Method: same suite, one source call commented, same eight runtime assets, boundary logging OFF; actual run followed by standard identity validator and direct request-specific drain/span analyzer.
- Result: worker exit0, two modes each400complete rows, 3x100warmups,100formal,24router checks,8worker identity PASS. DP0 batch4718 median77.935920715,P90 78.081732941,max78.142173767,spread0.222427368ms. Normal median77.483665466/max77.486846924; delta+0.452255249ms(+0.583678%)/+0.655326843ms(+0.845727%). Minimal bypass did not reproduce5x slowdown.
- Analysis fix: membership plus batch-prefill predicate mistakenly included47mixed rows where target was decoding. Selecting target's own4096scheduled tokens fixes the cause; direct revalidation PASS. Standard identity validator was independently PASS.
- Limits: different nodes, one run per arm; no pure AR duration or stable queue attribution established. Old522ms normal run remains confounded by boundary I/O, different warmup workload and absent uniform flag; complete excess attribution pending. CUDA/operator gate and official full calibration remain incomplete.

- Verified substep committed aa489b9a (two H800 source-selection workers and direct standard-result analyzer). Runtime bypass source separately committed e29a8f925; durable patch/assets/results recorded. Only original D005 files remain untracked.

## 2026-09-10 continuation: paired standard replay, Pending

- Motivation: remove different-node comparison and return to approved normal/skip priority before profiler attribution.
- Expectation: two opposite-order pairs on one H800 allocation, qualified first-formal DP0 TP0-3 boundary in every arm.
- Method: reviewed canonical normal/skip sources (clean, exact one-call diff); standard chain normal_1 -> skip_1 -> skip_2 -> normal_2, 176000 blocks, profiler/boundary OFF. New wrapper preserves actual worker exit and runs identity and drain/span validation after each arm. Root owns GPU scheduling; /root/pairing_audit independently reviewed chain and source propagation.
- Result: RJob yc26-h800-paired-normal-skip-20260910-01 submitted 15:24:33 UTC, still Pending at 15:28 UTC, codesign-default rank4/H800 quota0. No runtime rows. Shell syntax PASS; actual GPU PASS pending.
- Analyzer correction: source and historical reference were fixed to old bypass/normal, inappropriate for paired runs. Added explicit source and standalone mode, reused existing drain/request predicates; direct existing-artifact checks reproduce normal77.483665466 and skip77.935920715ms, both clean/batch400rows and drain PASS. Committed a27b80f5. No new timing measurement is implied.
- Failures recovered: nsys jobs01/02 never reached vLLM (image lacks nsys); Torch profiler01 failed dubious ownership. Source f827a4ecc has crossed counters and prefix-name mismatch. Profiler work deferred; source excluded from normal/skip. DP1 dummy capture remains unimplemented. See failure report.
- Operational deviation: allocation submitted without predict-only. Preserve queued authorized job; future allocations restore documented predict preflight. System-scope MemoryMax query failed interactive authentication; --user scope succeeded with same2G cap.

- 2026-09-10 15:35 UTC: same paired RJob remains Pending, queue position4; no output directory or worker/node receipts. No duplicate GPU job submitted. ABBA review notes: caches follow existing standard paths and are reused; preflight router runs baseline vLLM-BS. Compare batch_dp_token_counts across all four arms before causal delta; expected historical target is [4096,1]. Client-only duration estimate ~71min for four arms, derived from completed bypass timings.

- 2026-09-11: Switched ABBA pairing to H200 using step_main + h200 + 310809. RJob yc26-h200-paired-normal-skip-20260911-01 submitted; initial phase Starting/Scheduled with no machine available messages, then still starting. Output analysis/h200-paired-normal-skip-01. H200 worker script committed faedd1aa; no timing rows yet. H800 queued job remains separate and untouched.

- 2026-09-11 02:12 UTC: H200 RJob yc26-h200-paired-normal-skip-20260911-01 advanced Pending -> Starting; task worker creation reported, but replica listing still has no ready row and no output artifacts. Continue monitoring same job; no timing conclusion.
- 2026-09-11 02:20 UTC: H200 replica cfcd2084 is 1/1 Running; runtime probe PASS on all 8 H200 GPUs. normal_1 has started standard client setup and FlashInfer JIT compilation; no client/batch timing rows yet.

- 2026-09-11: H200 paired job -01 completed clean normal_1 (400 rows, 3 warmups + formal drain PASS) then failed at diagnostic source/commit assertion because H200 diagnostics worker hardcoded old checkout. Root cause fixed in commit 7b462dfa using explicit SOURCE/COMMIT override for both H200 official and diagnostic workers. New job yc26-h200-paired-normal-skip-20260911-02 is Running; no timing yet.
- 2026-09-11 02:05 UTC: replacement H200 RJob -02 Running; new analysis/h200-paired-normal-skip-02 has node/preflight/runtime receipts and normal_1 clean server startup. No formal timing rows yet.
- 2026-09-11 02:40 UTC: replacement -02 passed source/commit preflight and started normal_1 clean. First warmup replay completed (100 rows, phase 157.12s); no batch timing yet. Initial JIT startup is outside formal span.
- 2026-09-11 03:20 UTC: H200 -02 normal_1 clean completed 400 rows and full drain; batch diagnostic source override passed and 8 rank batch logs exist. Batch warmup replay0 and replay1 completed (200 rows total), each 100 requests with contiguous phase boundaries; replay2/formal pending.

- 2026-09-11: H200 `step_main + h200 + 310809` ABBA paired replay `yc26-h200-paired-normal-skip-20260911-02` completed with worker exit 0 and RJob `Succeeded`. All four arms (`normal_1 -> skip_1 -> skip_2 -> normal_2`) passed clean and batch identity/drain validation: each arm has 3×100 drained warmups, 100 formal requests, 400 clean rows and 400 batch rows; first formal predicates are request `cmpl-pf4096_dc1024:0-0`, batch size 1, 4096 prefill/0 decode, DP0 TP0–TP3, `batch_dp_token_counts=[4096,1]`.
- 2026-09-11: H200 first-formal batch statistics: `normal_1` median/P90/max/spread = `81.084751129/81.096882629/81.099166870/0.101020813 ms`; `skip_1` = `80.058879852/80.318404388/80.426269531/0.510169983 ms`; `skip_2` = `80.260639191/80.642729950/80.786048889/0.630340576 ms`; `normal_2` = `84.223361969/84.401792145/84.403327942/0.473953247 ms`.
- 2026-09-11: Paired deltas are `skip_1 - normal_1 = -1.025871277 ms (-1.2652%)` and `skip_2 - normal_2 = -3.962722778 ms (-4.7050%)`. Direction is consistent (minimal post-MoE AR bypass does not increase span), while magnitude is not stable because the two normal arms differ by `+3.138610840 ms`; this rejects a 5× bypass slowdown but does not close completion/queue attribution or op-gap correction.

- 2026-09-11 completion-capture preparation: added and committed `tests/e2e/issue26_h200_completion_capture_worker.sh` (035da750). The external diagnostic vLLM source passed static checks in commits `b12bbcf48` and `ab6cb0f97`; review found that `get_dp_group().rank` was global rank, so `dp_rank` was corrected to `rank_in_group` in `f35033e05a5fa5632ac4a4952981cb8c14e9f346`.
- 2026-09-11 completion-capture attempt `yc26-h200-completion-normal-20260911-01` reached H200 environment probe and then failed before vLLM because the worker compares the full 40-character diagnostic commit while the launch passed short `ab6cb0f97`. It produced no timing rows and is excluded from RCA. The failure is preserved under `analysis/h200-completion-normal-01/`.
- 2026-09-11 corrected H200 `step_main` completion capture `yc26-h200-completion-normal-20260911-02` was submitted with full source commit `f35033e05a5fa5632ac4a4952981cb8c14e9f346`, standard 3x100 drained warmups plus 100 formal batch, and completion output under `analysis/h200-completion-normal-02/`. Platform currently reports no machine available and the RJob is retained in queue.
## 2026-09-11 — H200 local MoE stage diagnostic queued

Completed the formal completion-capture report for `yc26-h200-completion-normal-20260911-03`. The diagnostic run passed the standard three drained 100-request warmups, 100 formal requests, 400 client rows, identity validation, and the exact first formal 4096-prefill predicates. Its DP0 TP0–TP3 outer spans were 88.206016541, 88.172286987, 88.200736999, and 88.078498840 ms. The run establishes collective arrival skew but cannot identify the local source of TP3's late arrival.

To separate local MoE work from the post-local-work gap, diagnostic-only vLLM commit `eb4c9a139` records CUDA events around the non-chunked `quant_method.apply()` path (`phase=local_moe_apply`) and attaches its metadata to the following `combine` row. The patch is isolated to `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab`; it does not modify Frontier production code or add per-collective synchronization. Python compilation and `git diff --check` pass.

RJob `yc26-h200-completion-local-stage-20260911-01` was submitted on H200 `step_main` with `num_gpu_blocks_override=310809`, the standard worker chain, and the full diagnostic source commit. Predict-only initially listed no immediate slot, then the RJob was scheduled on `gpu-h200-0379` and is currently `Starting` while the worker pod initializes. No timing result is claimed until the full warmup, drain, identity, and first-formal gates pass.

The first submission was stopped with reason `config_or_code_error` before user workload execution because the worker contract compares the full `git rev-parse HEAD` and the launch had passed the short hash `eb4c9a139`. Corrected RJob `yc26-h200-completion-local-stage-20260911-02` uses the full commit `eb4c9a1394ef136f134b5a36538847ec01679c68` and remains the authoritative rerun; the stopped job is retained as an execution failure record only.

Corrected RJob `yc26-h200-completion-local-stage-20260911-02` completed with `Succeeded`. It produced 400 client rows after three completed 100-request warmup replays, 8 rank boundary logs, and 8 completion logs. Identity validation passed; the first formal DP0 TP0–TP3 batch is `batch_id=3880`, request `cmpl-pf4096_dc1024:0-0`, one request, 4096 prefill, zero decode, and DP token counts `[4096,1]`, with outer span 90.690338135 ms on TP0.

The new `local_moe_apply` rows identify a different critical-path pattern from completion run `-03`: TP1 local CUDA sum is `47.901280 ms` (median `0.993168 ms/layer`), versus TP0 `27.315488 ms`, TP2 `33.951136 ms`, and TP3 `29.063264 ms`. TP1 is latest local-stage end on 47/48 layers and latest combine start on 47/48 layers. Local-to-combine gaps are only `0.002944–0.003008 ms` CUDA median and `0.030232–0.032006 ms` host median across ranks. All four local inputs are identical `[4097,2048]` BF16 tensors. This local stage is therefore the measured source of the late arrival in this run; its internal kernel/workload cause is still under repeat validation.

Repeat RJob `yc26-h200-completion-local-stage-20260911-03` was submitted with the same H200 `step_main + h200 + 310809` configuration and full diagnostic commit. Predict-only listed capacity, but the job is currently queued/scheduled awaiting a machine. No repeat timing claim is made yet.

## 2026-09-11 — H200 local-MoE completion repeat completed

- **Motivation:** Repeat the local-stage completion capture on the approved H200 `step_main + h200 + 310809` recipe to determine whether the late participant is a fixed TP rank or per-run local execution variation.
- **Expectation:** Require the complete standard warmup/formal gates and compare the same first formal DP0 request against the prior `-02` capture.
- **Method:** Completed RJob `yc26-h200-completion-local-stage-20260911-03` using diagnostic commit `eb4c9a1394ef136f134b5a36538847ec01679c68`; verified 3×100 drained warmups, 100 formal requests, 400 rows, identity PASS, and target `batch_id=3882`. Extracted only persisted completion rows with that batch ID and joined 48 layers × `local_moe_apply`/`combine`/`tp_ar` for DP0 TP0–TP3.
- **Result:** Outer spans are TP0/TP1/TP2/TP3 `91.194595337/91.198524475/91.387741089/91.428733826 ms`; rank max `91.428733826 ms`, spread `0.234138489 ms`. TP2 local CUDA sum is `47.142784 ms` and is latest on 37/48 layers; TP0/TP1/TP3 local sums are `35.018848/32.306528/25.494016 ms`. TP2's TP AR sum is shortest at `6.539872 ms`, while TP3 records `32.188768 ms`; local→combine CUDA medians remain `0.002976–0.003104 ms` and host medians `0.030914–0.031907 ms`. The late rank changed from TP1 in `-02` to TP2 in `-03`, so the supported explanation is per-run local-MoE device/queued-work variation, not a fixed TP-rank cause.
- **Artifacts:** `analysis/h200-completion-local-stage-03/{identity_validation.json,completion_local_stage_first_formal_summary.json,completion_local_stage_summary.log}` and `test_report_2026-09-11_h200_local_stage_repeat.md`.
- **Limits:** The capture still synchronizes and writes diagnostic JSONL per forward. `local_moe_apply` remains an envelope and does not identify expert token populations, individual kernels, host enqueue delay, stream idle, or NCCL internals. Clean CUDA gate, production op correction, and clean/diagnostic reconciliation remain open.

## 2026-09-11 — H200 step_main MoE subphase suite submitted

- Motivation: The two successful `local-stage` captures proved that the late participant moves between TP ranks, but `local_moe_apply` remains a single envelope. The next RCA step must distinguish existing MoE child scopes (`moe_shuffling`, W1 grouped GEMM, activation, W2 grouped GEMM, `moe_sum`) while preserving the same warmed first-formal identity.
- Expectation: A fresh standard H200 `step_main + h200 + 310809` compute suite should produce five independently warmed runs (`batch_before`, attention, MoE, detail, `batch_after`), each with three drained 100-request warmups and 100 formal requests. The MoE child scopes should identify whether rank skew tracks routing/shuffling, expert compute, or routed reduction; nested parent `moe_grouped_gemm` will remain an envelope only.
- Method: Ran predict-only first; it listed available H200 capacity. Submitted RJob `yc26-h200-compute-subphase-20260911-01` using `/kubebrain/rlaunch`, the fixed standard compute suite, diagnostic vLLM `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab` at full commit `eb4c9a1394ef136f134b5a36538847ec01679c68`, and H200 `num_gpu_blocks_override=310809`. The job entered `Starting` and is retained for platform queue monitoring.
- Result: No timing result is claimed yet. The queue phase is an execution status only; formal evidence remains pending until all five suite arms, identity validation, full drain, and first-formal predicates pass.
- Limits: CUDA-event diagnostic scopes include queued device work and the suite remains diagnostic. Parent/child MoE scopes cannot be summed, rank-local TP AR sums cannot be added, and no Frontier correction or clean/diagnostic reconciliation is authorized from this run alone.

## 2026-09-11 — H200 warmed MoE subphase suite completed

- Motivation: Split the existing `local_moe_apply` envelope into available gating, shuffling, grouped-GEMM parent, and routed-reduction scopes on the same H200 recipe.
- Expectation: Require five complete arms with 3×100 drained warmups and 100 formal requests; use request identity `cmpl-pf4096_dc1024:0-0`, DP0 TP0–TP3, and exact 4096-prefill/0-decode predicates.
- Method: RJob `yc26-h200-compute-subphase-20260911-01` ran `batch_before -> compute_attention -> compute_moe -> compute_detail -> batch_after` using H200 `step_main + h200 + 310809`, diagnostic source commit `eb4c9a1394ef136f134b5a36538847ec01679c68`, CUDA-event default scopes, and no per-scope synchronization. Parsed persisted first-formal rows into `analysis/h200-compute-subphase-20260911-01/subphase_summary.json`.
- Result: All five arms completed 3×100 warmups and 100 formal unique requests. First-formal batch IDs were 4048, 4525, 4211, 4626, and 4529 respectively. Clean-control medians were 82.269920 ms (`batch_before`) and 137.100624 ms (`batch_after`), proving these separate controls cannot form a clean subtraction. Diagnostic arm medians were 102.303600 ms (attention), 134.537086 ms (MoE), and 103.391888 ms (detail). MoE parent scope totals were TP0/1/2/3 `16.865632/24.452032/16.778560/16.966240 ms`; `moe_sum` was stable at `2.802240/2.835392/2.811328/2.812800 ms`; the detail arm showed W1 `11.624992/7.977568/6.514912/10.733152 ms`, activation `6.556608/6.549664/6.546880/6.568224 ms`, and W2 `4.054656/4.003712/4.037920/4.073280 ms`.
- RCA: Rank-local spread moves between stages and runs. The compute-MoE TP1 gating/shuffling inflation and detail-arm TP0 W1 inflation are not a fixed-rank signature; default CUDA-event scopes include queued device work/host submission effects. Stable `moe_sum` rules out routed reduction as the dominant skew source. The run does not identify a hidden 20 ms kernel or close the clean Frontier-vLLM gap.
- Limits: Operator arms use separate executions and cannot be summed across arms. Existing `ops` identity validator rejects these artifacts because they do not carry the legacy `op_profile_selected` contract; direct request-id/predicate checks confirmed one first formal batch on every DP0 TP0–TP3 arm. No production correction, clean/diagnostic reconciliation, or operator gate closure follows.
- Report: `test_report_2026-09-11_h200_moe_subphase.md`.
- 2026-09-12: Rebuilt DeepEP low-latency capability probe `yc26-h200-deepep-ll-capability-20260912-04` completed on H200 `step_main + h200`; all 8 ranks passed `dist_initialized`, `hint=545260672` with `max_tokens_per_dp_rank=256`, DeepEP buffer initialization, and barrier (`torchrun.exit_code=0`). This supersedes the invalid `16384` probe and the earlier host/device NVSHMEM mismatch from the unreconstructed wheel. Formal timing was authorized by the capability result but not yet measured.
- 2026-09-12: Submitted formal H200 `deepep_low_latency` replay `yc26-h200-all2all-deepep-ll-20260912-05` after `predict-only` listed available H200 capacity. It uses `step_main + h200 + 310809`, the frozen 4096/1024 workload, diagnostic source commit `150fa4a1cf46500c22d5fa585ebc9801a43e5c2d`, rebuilt DeepEP package first on `PYTHONPATH`, 10 drained 100-request warmups, and 100 formal requests. The RJob is `Running` on `gpu-h200-0379`; model startup and JIT are active, with no formal timing rows yet. No backend comparison or reconciliation claim is made until full client drain, identity validation, and first-formal predicates pass.

## 2026-09-12 — H200 alternative all2all backend experiment completed

- The H200 `step_main + h200 + num_gpu_blocks_override=310809` backend experiment is complete for the capability-positive `naive`, `deepep_high_throughput`, and `deepep_low_latency` arms. Each arm used the standard replay/diagnostic chain, ten fully drained 100-request warmup replays, 100 formal requests, 1100 client rows, eight rank batch logs, and the first-formal request/batch predicates. The PPLX capability path remains unresolved and has no formal timing.
- Native/naive RJob `yc26-h200-all2all-naive-20260912-01` first-formal DP0 TP0--TP3 spans were `83.554786682/83.607101440/83.546211243/83.548767090 ms`; median/P90/rank-max/spread were `83.551776886/83.591407013/83.607101440/0.060890198 ms`.
- DeepEP high-throughput RJob `yc26-h200-all2all-deepep-ht-20260912-02` first-formal spans were `86.499839783/86.509506226/86.207138062/86.473052979 ms`; median/P90/rank-max/spread were `86.486446381/86.506606293/86.509506226/0.302368164 ms`. It is approximately `+2.934669495 ms` or `+3.51%` versus the native/naive arm.
- DeepEP low-latency capability probe `yc26-h200-deepep-ll-capability-20260912-04` passed on all eight ranks after rebuilding DeepEP against the active NVSHMEM/NCCL overlay. Formal RJob `yc26-h200-all2all-deepep-ll-20260912-05` completed with identity validator PASS and first-formal spans `719.277648926/719.308532715/719.308898926/719.197387695 ms`; median/P90/rank-max/spread were `719.293090820/719.308789063/719.308898926/0.111511230 ms`.
- LL RCA is source-backed: `VLLM_MOE_DP_CHUNK_SIZE=256` (`vllm/envs.py:130`, `fused_moe/config.py:320`) feeds `max_tokens_per_rank` into DeepEP LL (`fused_moe/layer.py:173-195`); the 4096-token formal prefill therefore enters approximately 16 chunk iterations, repeated through 48 MoE layers (`layer.py:1696-1806,1820-1828`). Each chunk calls `low_latency_dispatch(..., async_finish=False)` and `low_latency_combine(..., async_finish=False, zero_copy=False)` (`deepep_ll_prepare_finalize.py:151-165,205-235`). vLLM documentation explicitly positions high-throughput for prefill and low-latency for decode (`docs/serving/expert_parallel_deployment.md:17-23,182-190`).
- LL raw DP0/TP0 rows reinforce the queue effect: decode-only batches immediately before formal were about `66.62--67.51 ms`, formal batch `10495` was `719.277648926 ms`, and following mixed batches were `1870.592041016 ms` and `2583.209716797 ms`. These values are not added to the first-formal span and do not identify a pure kernel duration.
- New reports: `test_report_2026-09-12_h200_all2all_deepep_ll.md` and `analysis/backend-comparison-20260912.md`. No Frontier communication code, predictor, CPU add-on, fitted residual, or clean/diagnostic reconciliation was changed. CUDA and operator gates remain open. The separate accepted clean comparison remains Frontier `59.190354384 ms` versus vLLM `78.118782043--79.307357788 ms`; correcting the sign of the `~9.740 ms` compute excess implies an approximately `28.668--29.857 ms` non-compute remainder, without attributing all of it to communication.
## 2026-09-13 — clean PPLX boundary and naive Kineto profiling in progress

- User direction: run two parallel H200 investigations. The first must resolve PPLX capability/timing and measure a clean first-formal batch CUDA span. The second must profile the native `naive` vLLM path with Kineto, with Frontier per-op instrumentation disabled, and classify compute, communication, memory activity and idle time.
- Fixed configuration: H200 `step_main + h200`, eight GPUs, `num_gpu_blocks_override=310809`, Qwen3-30B-A3B dummy model, TP4/DP2/EP8, 4096-prefill/1024-output, eager, BF16, FLASHINFER, uniform routing, prefix caching/chunked prefill OFF.
- Warmup gate: ten complete drained 100-request replays, then 100 formal requests; admissible client artifact has 1100 rows and identity validator/formal predicates must pass. Task harness was corrected to state 1100 rows explicitly.
- PPLX capability and instrumented replay are already complete and are not rerun. A minimal boundary-only source patch based on compatibility source `150fa4a1` is committed in `/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt` as `f025cc30`; it suppresses the existing diagnostic path on non-selected forwards and records one CUDA event envelope plus one post-end synchronization only for the selected DP0 TP0–TP3 first-formal batch. Replay worker/analyzer are committed as `33c8e375`; no PPLX GPU timing exists yet.
- Naive Kineto retry `yc26-h200-naive-kineto-20260913-02` is Running. At this checkpoint it has produced 200 client rows (two completed warmup rounds) and no profiler traces/formal rows yet. The prior `-01` failure was only an unbound `NO_PROXY` shell expansion before vLLM startup; fix `40ba1423` was verified and the retry uses that source.
- Pending: allow the naive run to complete, validate trace count and formal identity, inspect real Kineto event categories before accepting the parser's breakdown, then schedule the PPLX boundary-only RJob on the released H200 slot. No clean/diagnostic reconciliation or Frontier production correction is authorized from either pending run.

- 2026-09-13 03:11 +0800: H200 `predict-only` returned ten available 8-GPU candidates. Submitted `yc26-h200-pplx-boundary-20260913-01` with `step_main + h200 + 310809`, fixed image, and output `analysis/pplx-boundary-20260913-01/`. The RJob entered `Starting` with no machine currently available while the naive Kineto RJob consumes the active allocation; it is intentionally retained for FIFO scheduling. No PPLX timing rows exist yet.

## 2026-09-13 — naive Kineto first-formal window and launch correlation review

- observed: `yc26-h200-naive-kineto-20260913-02` produced ten complete drained warmups, 100 formal completions, 1100 client rows, eight GPU traces and one `async_llm` trace. The RJob later terminated during distributed teardown with TCPStore broken-pipe warnings after the artifacts had been written; the timing artifacts are retained, while the terminal RJob status is not treated as a clean process-pass.
- completed: Corrected parser selection requires a 48-layer group with prefill-scale attention markers. DP0 TP0--TP3 select the target first-formal prefill; DP1 group 48 is concurrent decode and is excluded as `NO_FORMAL_PREFILL_WINDOW`. The resulting report is `test_report_2026-09-13_naive_kineto.md` and the parser output is `analysis/naive-profiler-20260913/kineto_breakdown_run02_prefill.json`.
- result: DP0 marker-window spans are `111.735588/108.395569/107.340562/106.682713 ms` (TP0--TP3), with median `107.868065 ms`, P90 `110.733582 ms`, max `111.735588 ms`, spread `5.052875 ms`. Category unions are compute `27.45--28.88 ms`, communication `21.55--69.63 ms`, memory `3.63--3.89 ms`; TP0 carries `57.415055 ms` of marker-window idle/non-kernel remainder.
- result: Launch correlation audit finds exactly `3064` CPU CUDA launches and `3064` matched GPU kernels on every DP0 TP rank, with `144` matched NCCL all-reduce kernels per rank. The all-reduce launch-to-start median is `0.181 ms` (TP0), `1.611 ms` (TP1), `2.255 ms` (TP2), and `2.055 ms` (TP3), while launch-owned GPU spans converge to `111.937--112.138 ms`. The rank-local communication union gap is therefore queue/collective placement and marker-boundary alignment, not evidence of a missing TP0 model operation.
- limitation: Kineto windows remain `~27--43 ms` above the accepted clean native/naive `78.118782--79.307358 ms` reference and cannot be used for clean span reconciliation, predictor fitting, or CUDA gate closure. A no-profiler CUPTI/nsys run is still needed to separate pure kernel execution from stream wait, host launch delay, and profiler overhead. No Frontier production code or communication model was changed.
## 2026-09-13 — PPLX boundary source pin finalized

- completed: Updated `tests/e2e/issue26_h200_pplx_boundary_worker.sh` to default to vLLM boundary source commit `cf1ef5de9c0c45aedec4cf9d22c8eb56caf8bf0b`, which includes the explicit instrumentation-required validation. The earlier `f025cc30` commit remains recorded as the initial implementation.
- completed: Updated `test_report_2026-09-13_pplx_boundary_static.md` to use the final source commit. The source worktree remains clean and the four pre-existing unrelated untracked scripts remain untouched.
- completed: Re-ran `python -m py_compile tests/e2e/issue26_pplx_boundary_analysis.py` and `bash -n tests/e2e/issue26_h200_pplx_boundary_worker.sh`; both passed.
- pending: Coordinator must submit the H200 `step_main + h200` RJob with `ISSUE26_WARMUPS=10`, then verify the 1100 client rows and clean PPLX CUDA-event span before any performance claim.
## 2026-09-13 — PPLX boundary RJob startup failure RCA

- observed: `yc26-h200-pplx-boundary-20260913-01` reached `Running` and terminated `Failed` before health readiness. The environment probe completed H200/NVLink checks, but no client rows or timing artifacts were produced.
- root cause: `runtime/server.log:251` and repeated worker traces show `AssertionError: pplx_kernels not found` in `PPLXAll2AllManager` during model-parallel initialization. The RJob launch annotation omitted `ISSUE26_OPTIONAL_PYTHONPATH` and the PPLX NCCL/NVSHMEM library path, so the pinned source was present but the verified PPLX overlay was not discoverable.
- evidence: the corrected capability probe `yc26-h200-pplx-capability-20260913-02` passes all eight H200 ranks using `/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03`, confirming this is a launch-environment omission rather than unsupported PPLX runtime.
- report: `test_report_2026-09-13_pplx_boundary_failure.md` records exact RJob status, error lines, and the bounded retry environment. No capability or instrumented replay rerun is authorized; one fresh boundary-only retry remains pending with explicit overlay/library variables and a new output directory.


## 2026-09-13 — PPLX repair and naive Kineto run-03 completion

- completed: Rechecked PPLX lane after the analyzer correction commit `ad0f6f0d`. The first formal request `pf4096_dc1024:0` was served on the single complete DP1 lane; DP0 boundary files were empty. The repaired analyzer requires one validated non-empty DP lane and all TP0--TP3 rows, and reports `PASS` with median `542.338012695 ms`, P90 `542.361901855 ms`, rank max `542.371643066 ms`, spread `0.390197754 ms`. This is a boundary-only CUDA-event envelope for a fused PPLX protocol, not the canonical native clean span and not a Frontier correction input.
- completed: Monitored `yc26-h200-naive-kineto-20260913-03`. The client completed ten drained 100-request warmups plus 100 formal requests (1100 rows), captured the first formal request `pf4096_dc1024:0`, and wrote eight GPU Kineto traces plus one async CPU trace.
- observed failure: the RJob terminal phase is `Failed` because `tests/e2e/issue26_h200_naive_profiler_worker.sh:109` used an invalid Bash arithmetic expression in the post-client row-count check. The client rows, phase records, and traces were already complete before this failure. The worker expression was corrected to an explicit `EXPECTED_ROWS` arithmetic assignment and committed as `f1f5ba74`; `bash -n`, arithmetic execution, and `git diff --check` pass. The run remains recorded as artifact-complete with a post-client harness failure; no claim of a successful platform job is made.
- completed: Independently regenerated the trace manifest and ran `parse_kineto_trace.py` against run-03. Parser status is `PASS`; all DP0 TP0--TP3 traces contain complete 48-layer attention/MoE marker groups and H200/NCCL distributed metadata. The formal marker spans are TP0 `113.293890625 ms`, TP1 `110.268503906 ms`, TP2 `114.339026367 ms`, TP3 `112.168838867 ms`; median `112.731364746 ms`, P90 `114.025485645 ms`, max `114.339026367 ms`, spread `4.070522461 ms`.
- completed: Run-03 category unions are recorded in `analysis/naive-profiler-20260913/run-03_kineto_breakdown_formal.json`: compute `27.821732--28.975674 ms`, communication `22.931490--72.972239 ms`, memory `3.674188--3.908914 ms`, all-kernel union `55.816078--105.084194 ms`, and idle/non-kernel `7.084645--58.522948 ms`. The independent audit is `analysis/naive-profiler-20260913/run-03/run-03_gate_audit.json`; the report is `test_report_2026-09-13_naive_kineto_run03.md`.
- interpretation: the `112.731 ms` Kineto median is `33.424--34.613 ms` above the accepted clean `78.118782043--79.307357788 ms` reference (about 42--44%); it is profiler-perturbed diagnostic evidence. Rank-local communication/idle values are complementary (TP2 has `22.931490 ms` communication and `58.522948 ms` idle), matching the run-02 rank-rotation pattern. The evidence supports marker/stream-queue placement effects and does not support summing rank-local communication or inventing a missing 20 ms operation.
- pending: no clean no-profiler pure-kernel decomposition exists yet. PPLX chunk-size/JIT/backend attribution remains optional follow-up after this audit; no new H200 RJob is active. Frontier predictor, communication model, operator accounting, CPU add-on, and clean/diagnostic reconciliation remain unchanged.

## 2026-09-13 — Nsight run-01 importer failure and run-02 repair

- **Observed:** `yc26-h200-nsys-profiler-20260913-01` completed the required 10 drained warmup replays and 100 formal requests (1100 client rows); the first formal callback created both Nsight control markers. The RJob then failed during Nsight report conversion. No `.nsys-rep` or stats CSV was produced.
- **Root cause:** `runtime/server.log` reports `Importer error status: The importer binary and its dependencies were not found`, followed by `No reports were generated`. The target-only Nsight runtime tarball did not contain the same-version `host-linux-x64/QdstrmImporter` required for QDSTRM import. This is a profiler packaging failure after vLLM execution, not a workload or CUDA model failure.
- **Repair:** packaged `/opt/nvidia/nsight-systems/2025.6.3/host-linux-x64` as `/data/ycfeng/tmp/issue26-nsys-host-runtime-20260913-01.tgz`, SHA256 `b858b6448f5aef90a1ed32ee51cb2896ea19ef5716fee7c32b527b3fa7fcfe92`; updated `tests/e2e/issue26_h200_nsys_profiler_worker.sh` to verify/extract the host importer and corrected `nsys stats --force-overwrite=true`. Worker repair commits are `2933d699` and `57cbd961`; syntax and diff checks PASS.
- **Queued retry:** `yc26-h200-nsys-profiler-20260913-02` was submitted after H200 `predict-only` returned six candidate nodes. It uses the same H200 `step_main + h200 + num_gpu_blocks_override=310809`, source `vLLM-BS@46f7b179fd3bf42b9616dc4670cba419afdb2085`, 10 warmups, and the host runtime SHA above. The RJob is currently `Starting`; no timing or trace result is claimed until `.nsys-rep`, stats exports, and formal identity are all verified.
- **Parser:** `tests/e2e/issue26_nsys_breakdown.py` is committed as `11462105`; its synthetic Nsight-header clipping/union checks PASS. It reports per-device compute/communication/memory/unknown unions and idle remainder without inventing rank identity.

## 2026-09-13 — Nsight PPLX overlay propagation repair

- **Observed:** The first PPLX Nsight run reached the worker and initialized Nsight, but its `server.log` ended with `No reports were generated`; the launch had supplied no optional PPLX overlay. A corrected `-02` launch supplied the verified `pplx_kernels` Python path and NCCL/NVSHMEM library paths, but the worker still overwrote `PYTHONPATH` after source selection.
- **Root cause:** `tests/e2e/issue26_h200_nsys_profiler_worker.sh` did not preserve `ISSUE26_OPTIONAL_PYTHONPATH`, `ISSUE26_EXTRA_LD_LIBRARY_PATH`, or `ISSUE26_LD_PRELOAD`. PPLX imports and its dynamic libraries were therefore unavailable to the target process and Nsight child.
- **Repair:** Added optional path/preload propagation before capturing `TARGET_LD_LIBRARY_PATH`; native/naive defaults remain unchanged. `bash -n` and `git diff --check` pass. Committed as `cd592796`.
- **Execution:** H200 `yc26-h200-pplx-nsys-profiler-20260913-02` remains queued/Starting and uses the shared updated worker plus the verified runtime overlays. No PPLX clean timing is claimed until it produces 1100 rows, control markers, `.nsys-rep`, and parseable stats.
- **Report:** `test_report_2026-09-13_nsys_pplx_path_fix.md`.

## 2026-09-13 — Nsight client URL shell-argument hardening

- **Observed:** Native Nsight run-03 completed 10 warmups, 100 formal requests, both control markers, and a `.nsys-rep`, but the worker exited in the client invocation with `line 152: ttp://127.0.0.1:8000: No such file or directory`; no automatic stats exports were written.
- **Repair:** Bound the URL to `BASE_URL` and pass it as a single quoted `--base-url=$BASE_URL` argument. `bash -n` and `git diff --check` pass; committed as `54d4538b`.
- **Independent evidence:** The run-03 `.nsys-rep` was copied to `/data/ycfeng/tmp/issue26-nsys-run03-manual-stats/`; Nsight 2025.6.3 host tools successfully exported `cuda_gpu_trace`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, and `cuda_api_sum`. These exports are diagnostic artifacts because the platform job had a post-client harness failure.

## 2026-09-13 — native Nsight run-03 completed data gate; RJob post-client failure

- **Execution:** H200 `step_main + h200 + 310809`, vLLM native/naive source `0f34fb271fd66d7dd84201ebdd4722781f829390`, `VLLM_FRONTIER_INSTRUMENTATION=0`, Nsight Systems CUDA/NCCL capture, `ISSUE26_WARMUPS=10`. RJob: `yc26-h200-nsys-profiler-20260913-03`.
- **Gate:** ten complete drained warmups, 100 formal requests, 1100 client rows, formal first request `pf4096_dc1024:0`, start/stop markers, and `.nsys-rep` all present. `phase_records.json` records every phase and formal completion.
- **Platform status:** RJob is `Failed` after artifact completion. `runtime/client.log` contains `line 152: ttp://127.0.0.1:8000: No such file or directory`; this is a worker shell status defect after client/capture completion. Preserve as execution FAIL; do not call the RJob `Succeeded`.
- **Postprocess:** worker-created Nsight directory is root-owned. Report CSVs were generated in user-owned `analysis/nsys-profiler-20260913-run03-postprocess/` from a byte-copied `.nsys-rep`; `cuda_gpu_trace` has 24,052 rows across four H200 devices.
- **Observed diagnostic decomposition:** trace-domain API window (100.157909–200.984879 ms) gives per-device compute unions `23.017–23.566 ms` (device 0–2) and `37.433 ms` (device 3), communication unions `50.414–80.553 ms`, memory unions `2.027–2.606 ms`, and activity envelopes `71.217–99.514 ms` with `0.798–3.255 ms` internal idle. Un-clipped envelopes are `104.266–142.650 ms` because `cudaProfilerStop` drains queued work.
- **Communication population:** `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` has 756 instances (294.974 ms inclusive total, median 97.232 us, max 28.344 ms); `cross_device_reduce_1stage` has 354 instances (155.411 ms total, median 5.568 us, max 28.460 ms); broadcast has 1,470 instances (126.579 ms total). These are cross-device aggregate totals and are not to be summed as a TP latency.
- **Interpretation:** Nsight confirms a large native collective population and 28 ms outliers, supporting communication/queued-work as a major residual candidate. It does not yield a valid 78–79 ms clean decomposition because profiling and stop-drain alter the activity window; no clean/diagnostic reconciliation or Frontier correction is authorized.
- **Artifacts:** `test_report_2026-09-13_nsys_run03.md`, `analysis/nsys-profiler-20260913-run03-postprocess/nsys_breakdown_api_window.json`, `stats_cuda_gpu_trace.csv`, `stats_cuda_gpu_kern_sum.csv`, `stats_cuda_gpu_mem_time_sum.csv`, `stats_cuda_api_sum.csv`, and `stats_cuda_api_trace.csv`.

## 2026-09-13 — PPLX Nsight run-02 postprocess persisted

- **Motivation:** The coordinator requested durable evidence for the completed H200 PPLX Nsight run before comparing communication implementations. The run's platform phase is terminal `Failed`, so artifact validity and platform success must remain separate.
- **Method:** Audited `analysis/pplx-nsys-profiler-20260913-run02/` and regenerated the Nsight CUDA GPU/API CSVs from `first_formal.nsys-rep` with the version-matched host importer. Persisted `pplx_per_process_classified.json`, `run02_gate_audit.json`, `analyze_pplx_per_process.py`, and `artifact_sha256sums.txt` under `analysis/pplx-nsys-profiler-20260913-run02-postprocess/`.
- **Gate result:** Ten drained 100-request warmups, 100 formal requests, 1100 client rows, all formal prompt/completion identities 4096/1024, first formal `pf4096_dc1024:0`, start/stop markers, `.nsys-rep`, and five parseable stats CSVs are present. The machine-readable audit reports `ARTIFACT_COMPLETE_PLATFORM_FAILED`; no narrower platform failure cause is claimed because the persistent run does not retain the final platform stderr.
- **Observed diagnostic result:** Three process-local stop windows are 694.920, 682.208, and 675.986 ms. Their compute unions are 63.620, 61.872, and 61.082 ms; communication/protocol unions are 215.267, 551.502, and 555.737 ms; memory unions are 21.094, 20.733, and 20.494 ms; idle/non-kernel gaps are 394.014, 47.200, and 37.889 ms. PID 1172 has no observed stop and is excluded from bounded rows. PPLX protocol aggregates include 6128 main dispatch kernels (1918.894 ms inclusive), 12258 batched-triton kernels (1262.891 ms), 6129 combine kernels (909.533 ms), and 196 NCCL all-reduces (180.649 ms).
- **Limit:** The Nsight windows are diagnostic and profiler/queue/stop-drain perturbed; process IDs do not provide persisted DP/TP identity; inclusive kernel totals are cross-process/device aggregates. Keep the accepted native clean 78.118782043--79.307357788 ms reference and the PPLX boundary-only 542.338012695 ms result separate. No Frontier predictor, communication backend, CPU accounting, or clean/diagnostic reconciliation changed.
- **Report:** `test_report_2026-09-13_pplx_nsys_run02.md`.

## 2026-09-13 — PPLX clean retry and native CUDA-only Nsight run-04 submitted

- **Motivation:** Execute the two user-authorized H200 lanes: resolve PPLX clean boundary timing and obtain a lower-perturbation native/naive CUDA activity breakdown without Frontier per-op instrumentation.
- **Configuration:** H200 `step_main + h200`, 8 GPU, 64 CPU, 409600 MiB, `num_gpu_blocks_override=310809`, Qwen3-30B-A3B dummy model, TP4/DP2/EP8/PP1, BF16, eager, FLASHINFER, uniform routing, prefix/chunked caching OFF, 4096-prefill/1024-output, ten drained 100-request warmups and 100 formal requests (1100 rows).
- **PPLX lane:** Existing RJob `yc26-h200-pplx-clean-20260913-01` retained; it uses source `448f2b65e7679ae7490114ad382b6ba79becb3c3`, verified PPLX runtime overlay, `VLLM_ALL2ALL_BACKEND=pplx`, and controlled `VLLM_MOE_DP_CHUNK_SIZE=4096`. It remains `Starting` while waiting for a worker; no timing artifact exists yet. No duplicate was submitted.
- **Native lane:** Added and committed `tests/e2e/issue26_h200_nsys_cuda_only_worker.sh` (`e7d01533`). This worker keeps `VLLM_FRONTIER_INSTRUMENTATION=0` and uses Nsight `--trace=cuda` only, removing NCCL API/GPU CUPTI hooks while retaining CUDA kernel names for comp/comm/mem interval classification. Static checks (`bash -n`, `git diff --check`) pass.
- **Execution:** Predict-only listed H200 capacity. Submitted `yc26-h200-nsys-cuda-only-20260913-04`; platform reports `Running` and image pull completed. Its first artifact is `analysis/nsys-cuda-only-20260913-run04/preflight/environment.log`; formal rows/traces are pending.
- **Interpretation boundary:** Existing native run-03 (`~100 ms` Nsight window) and Kineto (`~110--114 ms`) remain profiler-perturbed diagnostics. Run-04 is intended to test whether CUDA-only capture approaches the accepted clean `78.118782043--79.307357788 ms` scale; it cannot replace that clean reference until formal identity, trace window, and category audit pass.

## 2026-09-13 — PPLX clean retries hit platform scheduling failures

- `yc26-h200-pplx-clean-20260913-01` reached `Failed` before worker start. `brainctl describe` records `FailedScheduling`: 0/4372 nodes available and nominated `gpu-h200-0222` had a terminating pod; the output directory contains only `launch.sh` and no preflight/runtime artifact.
- A second RJob `yc26-h200-pplx-clean-20260913-02` was already present from the concurrent scheduling line and used the explicit PPLX overlay/library paths. It also reached `Failed` before worker execution with the same resource/preemption condition; no client or boundary artifact exists. These are platform scheduling failures, not PPLX timing results.
- The failed jobs are retained. A new PPLX retry will be considered only after the active native capture releases its H200 worker, to avoid duplicate allocations and preserve FIFO scheduling evidence.

## 2026-09-13 — Native Nsight category classifier corrected

- **Observed issue:** The original communication regex matched the word `collective`, which misclassified FlashInfer `CollectiveEpilogue` attention kernels as network communication.
- **Repair:** Removed the generic `collective` token from `tests/e2e/issue26_nsys_breakdown.py`; explicit NCCL/all-reduce/broadcast/all-gather/all-to-all/cross-device/P2P names remain. Committed as `108c4c48`. Python compile, classifier smoke checks, and `git diff --check` pass.
- **Reanalysis:** Corrected run-03 artifact is `analysis/nsys-profiler-20260913-run03-postprocess/nsys_breakdown_api_window_reclassified.json`. Median per-device unions are compute 26.903 ms, communication 74.880 ms, memory 2.142 ms, all-activity 95.217 ms, and control-window idle 5.610 ms. Communication remains dominated by NCCL/cross-device kernels with approximately 28 ms maxima.
- **Interpretation:** This improves semantic classification but does not change the diagnostic boundary: Nsight window remains above the accepted clean 78.118782043--79.307357788 ms span and has no reliable DP/TP identity. No Frontier correction or reconciliation follows.

- After the two pre-start scheduling failures, a third PPLX clean retry `yc26-h200-pplx-clean-20260913-03` was submitted with `--backoff-limit=5` and explicit PPLX overlay/library variables. The platform scheduled it on `gpu-h200-0246` and pulled the image; RJob is `Starting` while the worker initializes. This is the only active PPLX retry. No timing claim exists until its full artifact gate passes.

## 2026-09-13 — PPLX clean boundary run05 completed

- completed: Monitored the H200 PPLX retries `yc26-h200-pplx-clean-20260913-03`, `-04`, and `-05`. Run05 reached terminal `Succeeded` and contains the complete runtime artifact; runs03 and04 were still `Running` at this checkpoint and are retained without using their partial rows.
- gate: `analysis/pplx-clean-boundary-20260913-run05/runtime/client.jsonl` has 1100 rows (10 drained warmups × 100 plus 100 formal). The client log records replay indices 0–9 with `completed_requests=100`, followed by `formal_requests=100` and `formal_unique_ids=100`. The boundary analyzer reports `status=PASS`, first formal `pf4096_dc1024:0` / `cmpl-pf4096_dc1024:0-0`, one selected DP0 lane, and all TP0–TP3 rows with batch size 1, 4096 prefill tokens, and zero decode tokens.
- result: clean boundary event durations TP0–TP3 are `183.506790161`, `183.467132568`, `183.122406006`, and `183.405410767 ms`; median `183.436271667 ms`, P90 `183.494892883 ms`, rank max `183.506790161 ms`, rank spread `0.384384155 ms`.
- provenance: H200 `step_main + h200 + num_gpu_blocks_override=310809`, vLLM source commit `448f2b65e7679ae7490114ad382b6ba79becb3c3`, `VLLM_ALL2ALL_BACKEND=pplx`, `VLLM_MOE_DP_CHUNK_SIZE=4096`, Frontier instrumentation/per-op/scheduler/routing/full diagnostic loggers disabled. Preflight confirms eight H200 GPUs and NV18 links.
- interpretation: PPLX is capability-valid and now has a complete clean first-formal boundary artifact, but its 183.436 ms median is 131.30–134.82% slower than the accepted native/naive 78.118782043–79.307357788 ms reference. This is an independent backend result and a semantic/performance finding; it is not a pure-kernel decomposition, Frontier correction input, or clean/diagnostic reconciliation. Per-rank CUDA-event envelopes can include queued forward-stream device work and must not be summed across ranks.
- report: `test_report_2026-09-13_pplx_clean_boundary_run05.md`. The run05 worker/analyzer provenance and source patch status are preserved in `analysis/pplx-clean-boundary-20260913-run05/{preflight,runtime}`.
- completed: The earlier retry `yc26-h200-pplx-clean-20260913-03` also reached `Succeeded` after the run05 checkpoint. Its analyzer PASS artifact is `analysis/pplx-clean-20260913-run03/runtime/boundary_analysis.json` with TP0–TP3 `182.934814453/182.368316650/183.000671387/182.785339355 ms`, median `182.860076904 ms`, P90 `182.980914307 ms`, max `183.000671387 ms`, spread `0.632354736 ms`. The run has the same 10 warmup + 100 formal + 1100-row and first-formal DP0 predicates as run05.
- repeatability: Run03 and run05 medians differ by `0.576194763 ms` (`0.31%`), indicating a stable PPLX boundary near 183 ms under `VLLM_MOE_DP_CHUNK_SIZE=4096`; both remain independent backend evidence and do not replace native/naive clean 78–79 ms or justify any Frontier correction.
- still monitoring: `yc26-h200-pplx-clean-20260913-04` remains `Running` with partial rows (300 at the latest check) and no boundary analysis. Preserve it as a separate retry; do not mix its incomplete data with completed runs.
- verification: Re-ran `tests/e2e/issue26_pplx_boundary_analysis.py --warmups 10` independently on both completed runtime directories (writing fresh outputs under `/data/ycfeng/tmp/pplx-audit.3DXbfV/` to preserve persisted artifacts). Both returned `status=PASS`, `client_rows=1100`, `formal_requests=100`, and DP0 TP0–TP3 complete.


## 2026-09-13 — Nsight SQLite communication classification correction

- **Motivation:** The rank-aware SQLite parser matched the generic token `collective`, which misclassified FlashInfer attention's `CollectiveEpilogue` kernel as communication.
- **Change:** Removed the generic `collective` regex branch from `tests/e2e/issue26_nsys_sqlite_breakdown.py`; explicit NCCL/all-reduce/broadcast/cross-device/point-to-point names remain communication candidates. Committed as `8d5dfc5f`.
- **Verification:** `py_compile`, `git diff --check`, and reprocessing of run04 SQLite passed. Corrected output is `analysis/nsys-cuda-only-20260913-run04-postprocess/nsys_sqlite_rank_breakdown_reclassified.json`; report is `test_report_2026-09-13_nsys_sqlite_reclassification.md`.
- **Observed:** Per-device compute unions are 35.519/33.800/32.840/32.727 ms; communication unions 17.580/60.053/58.672/61.518 ms; memory 2.273/2.173/2.246/2.114 ms; activity envelopes 105.741/101.189/98.432/100.695 ms; idle 50.370/5.163/4.674/4.337 ms. The envelopes remain above the accepted clean 78--79 ms batch boundary, so this is diagnostic only.

## 2026-09-13 — independent final gate audit and report update

- **Motivation:** Verify the two parallel lanes independently after both platform jobs reached terminal states, and preserve the distinction between a complete artifact gate and a clean latency result.
- **Method:** Re-read the H200 manifests, client JSONL, phase records, PPLX boundary analyzers, Nsight API/SQLite postprocess, and platform phases. Re-ran Python compilation, Bash syntax checks, `git diff --check`, and a direct 1100-row/10-warmup/formal-identity audit.
- **Result:** PPLX runs `yc26-h200-pplx-clean-20260913-03` and `-05` are `Succeeded` and analyzer `PASS`; each has 10 drained 100-request warmups, 100 unique formal requests, 1100 rows, first formal `pf4096_dc1024:0`, server `cmpl-pf4096_dc1024:0-0`, and complete TP0–TP3 rows. Native run `yc26-h200-nsys-cuda-only-20260913-04` is `Succeeded` with the same 10/100/1100 client gate and complete `.nsys-rep`, SQLite, CUDA reports, and corrected classifier output.
- **Measured:** PPLX medians are `182.860077` and `183.436272 ms` with `VLLM_MOE_DP_CHUNK_SIZE=4096`. Corrected native Nsight device-window median/P90/max/spread are `101.822669/105.378236/106.824398/7.628297 ms`; category medians are compute `33.320378 ms`, communication `59.362114 ms`, memory `2.209333 ms`, all-activity `94.891825 ms`, and idle `4.918915 ms`. The external formal-start to first-token marker span is `11,925.042 ms`, while the API-trace capture windows are about `99.196--106.824 ms`, proving the host marker and Nsight trace clocks must not be mixed.
- **Limits:** Nsight and PPLX values are diagnostic/backend-specific CUDA-event evidence. Nsight did not produce a clean 79 ms decomposition; one process lacks a stop API row and uses earliest-stop fallback, and the persisted device rows do not carry durable TP labels. No predictor, communication backend, CPU accounting, fitted residual, or clean/diagnostic reconciliation was changed. Reports: `test_report_2026-09-13_nsys_cuda_only_run04.md` and `test_report_2026-09-13_pplx_clean_boundary_runs03_05.md`.
- completed: Final retry `yc26-h200-pplx-clean-20260913-04` reached `Succeeded`. Its analyzer PASS artifact is `analysis/pplx-clean-boundary-20260913-run04/runtime/boundary_analysis.json`, with TP0–TP3 `183.881500244/183.869415283/183.598724365/183.461471558 ms`, median `183.734069824 ms`, P90 `183.877874756 ms`, max `183.881500244 ms`, spread `0.420028687 ms`; the full 10-warmup/100-formal/1100-row and DP0 first-formal predicates pass.
- PPLX repeat gate: complete run medians run03/run05/run04 are `182.860076904/183.436271667/183.734069824 ms` (mean `183.343472799 ms`, range `0.873992920 ms`, relative range `0.48%`). This provides stable backend evidence near 183 ms with chunk size 4096. All three clean boundary reports are persisted (`test_report_2026-09-13_pplx_clean_boundary_run03.md`, `run04.md`, `run05.md`). No PPLX job remains active.
## 2026-09-13 — Native NVTX run03 completed-workload capture failure

- **Execution:** Continued the existing H200 RJob `yc26-h200-nsys-nvtx-20260913-03` without resubmitting it. It used `tests/e2e/issue26_h200_nsys_nvtx_worker.sh`, H200 `step_main + h200 + num_gpu_blocks_override=310809`, native/naive vLLM, and Frontier instrumentation disabled.
- **Workload gate:** Ten fully drained 100-request warmups and 100 formal requests completed. `runtime/client.jsonl` has 1100 rows; the first formal request is `pf4096_dc1024:0`; `nsys-control/start`, `nsys-control/stop`, and `nvtx-watcher-status.json` are present and the watcher reports `PASS`.
- **Failure:** The RJob ended `Failed` during Nsight shutdown. `runtime/server.log` reports `The target application terminated. One or more process it created re-parented`, followed by `Generated: No reports were generated`. The `nsys/` directory contains only `version.txt`; no `.nsys-rep`, SQLite, or stats CSV exists.
- **RCA:** This is the second completed-workload failure of the same NVTX process-tree trigger (run02 and run03). The watcher status only proves that the watcher emitted an NVTX range; Nsight emitted no capture-start/end messages, so trigger delivery or process-scope capture remains the unresolved failure. It is not a vLLM latency result.
- **Limits/decision:** The run cannot provide compute/communication/memory/idle decomposition and has no value to compare with the accepted clean native `78.118782043--79.307357788 ms`. Do not repeat the identical NVTX mechanism, modify Frontier, or perform clean/diagnostic reconciliation. The existing CUDA-only Nsight run04 and Kineto run03 remain diagnostic windows above clean scale.
- **Report:** `test_report_2026-09-13_nsys_nvtx_run03_failure.md`.
