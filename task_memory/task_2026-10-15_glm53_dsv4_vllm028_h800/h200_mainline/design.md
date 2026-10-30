# H200 temporary mainline design: DSV4 on H200-32 and H200-64

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Section 9 lead review: the composed harness is `7ac25c74` plus the observer changes; the source-only CPU check replaces the removed vLLM 0.28 environment; the observation cost is accepted against the size of the measured wait. |
| 2026-10-30 | Added section 9 for D-P2-64-WAVE-DIAGNOSTIC: all-engine and TP0 wave records, timeline analysis, the one-repetition package_02 builder, CPU checks and their unavailable native-environment prerequisite, record volume, cost and acceptance. |
| 2026-10-25 | Recorded the P2-32 lifecycle result (section 8): Succeeded, R4 FAIL on the shutdown order only, observer acceptance PASS, parts a-g. |
| 2026-10-25 | Created: purpose, cases, resources and waves, package changes, checks, acceptance and limitations. |
| 2026-10-25 | Recorded the package names actually built: P2-32p uses `package_02` after an incomplete `package_01` build; `launch.py` is a changed retained file for profile packages. |
| 2026-10-25 | Recorded the failed first lifecycle launch from `package_01` (no job created) and the submission from `package_02` with the platform environment on the launch command. |
| 2026-10-25 | Added the P2-32 lifecycle case (D-H200-H4B-LIFECYCLE): H4b-(a) on H200, its builder source, changes, checks and acceptance (section 8). |
| 2026-10-25 | Recorded where the two H200 step-cost tables live, the same-table substitution for P2-64 (D-H200-STEP-COST-SUBSTITUTION), and the driver, gate and attribution cases. |

**Authority.** D-H200-TEMP-MAINLINE in [requirements.md](../requirements.md) (window 15:00 UTC 2026-10-25 to 03:00 UTC 2026-10-26; at most 64 H200 for this session, running plus queued, `step_main`, non-preemptible). The other rules it keeps are named there: DIAGNOSTIC label, no cross-hardware qualification, D-GPU-SHORTAGE-REPORT, D-INPUT-PERSISTENCE, D-MAINLINE-LOOP-CAP-DSV4 and the D-H4A-CHECKPOINT code bounds.

## 1. Purpose

Repeat the H800 P0 path on H200, reusing the H800 packages and tools:

1. a clean native reference at P16+D16, accepted by the R4 checker;
2. a native serving profile at the same layout, which gives the H200 step-cost table;
3. a Frontier DIAGNOSTIC comparison with the H4a/H4c driver and the canonical client gate;
4. the same at P32+D32 (H200-64), as far as the window allows.

H200 results never qualify an H800 cell, and an H800 table is never used for an H200 prediction.

## 2. Cases

| ID | Source package | Layout | Hosts | Run tag | Done when |
| --- | --- | --- | --- | --- | --- |
| P2-32r | native06 (`candidate_h800_p0_06`) | P16+D16, TP8/DP2/EP16/PP1 per role | 4 × 8 H200 | `dsv4-h200-p2-canonical-20261025-01` | R4 PASS on a local copy |
| P2-32p | profile02 (`candidate_h800_p0_profile_02`) | same | 4 × 8 H200 | `dsv4-h200-p2-profile-20261025-01` | job Succeeded, two decode TP0 traces present, step-cost table built |
| P2-32d | — (Frontier, CPU) | same | — | — | four errors against the mean of P2-32r, by the canonical client gate |
| P2-64r | native06 | P32+D32, TP8/DP4/EP32/PP1 per role | 8 × 8 H200 | `dsv4-h200-p2-64-canonical-20261025-01` | R4 PASS; also covers the P1b-style basic validation items at DP4 |
| P2-64p | profile02 | same | 8 × 8 H200 | `dsv4-h200-p2-64-profile-20261025-01` | as P2-32p, four decode traces |
| P2-64d | — | same | — | — | as P2-32d against P2-64r |

Every case keeps the 68f891b workload (three repetitions × 16 warmups + 128 formal requests of 6144/1024 tokens at QPS 1), the vLLM 0.28.0 engine at NATIVE `2cf0a69`, the C26 overlay, dummy weights, `balanced_topk` routing and the locked image `vllm-openai@sha256:2286e853…`. The image ran DSV4 on 4 × 8 H200 in `step_main` before (`h200_dsv4_g0b_20261021`, exit 0), with the same `cuda_compat_path` and cloud mount.

Cloud outputs: `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/<run tag with _>/case/runs/groundtruth_clean/<run tag>`.

## 3. Resources and waves

| Wave | Jobs | H200 in use |
| --- | --- | --- |
| 1 | P2-32r and P2-32p together | 64 |
| 2 | P2-64r, after wave 1 ends | 64 |
| 3 | P2-64p, after P2-64r ends, if the window allows | 64 |

Each wave is submitted only after a `step_main` quota read shows enough free H200. The jobs are non-preemptible (`ResourceConfig.preemptible` stays at its default `False`). Each job's `timeout_seconds` is 3600, as in the H800 packages. native06 took 22.5 min of job time and profile02 a similar time, so a 32-GPU job costs about 12-16 GPU-hours and a 64-GPU job about 27-37 GPU-hours.

Soft gates: the only in-run collectors are the packages' own (the `FullModelInventory` or `NativeServeProfile` worker extension and the serving inventory). They isolate their own errors and do not stop serving. All checkers run after the job ends.

Loop cap: at most 3 GPU jobs per case without its done condition. After that the case stops and goes to YC with the evidence.

## 4. Package changes

`build_package.py <case> [--package NAME]` copies the source package into `<case>/<NAME>/` (default `package_01`) and applies only the changes below. Launch outputs and `case/runs/` are not copied, and every path that points into the source package directory is moved to the new package. It then compares every file with the source and fails on any other difference.

| File | Change |
| --- | --- |
| `config.json`, `case/inputs/native_config.json` | moved paths; `gpu_type` H800 → H200; `charged_group` codesign → step_main; `run_tag`, `job_tag`, `output_dir`; the hashes in `retained_files` and `tool_hashes` of the changed files. DP4 only: `replica` 4 → 8, `engine.data_parallel_size` 2 → 4. Profile only: `engine.profiler_config.torch_profiler_dir` → `<output_dir>/torch_traces` |
| `engine.json`, `case/inputs/engine.json` | DP4 only: `data_parallel_size`. Profile only: `profiler_config.torch_profiler_dir` |
| `case/inputs/frontier.json` | `cluster_config_{prefill,decode}_replica_config_device` h800 → h200. DP4 only: `replica_config_attn_dp` 4 and both roles' `moe_expert_parallel_size` 32 |
| `case/manifest.yaml` | `case_id`, `run_generation`, `purpose`; `policy.priority` P2; `policy.gpu_type` H200; `policy.charged_group` step_main; native06 only: `policy.H200_nonpreemptible_running_plus_queued_limit` 32 → 64; `runtime_source_overlay.sha256`. DP4 only: `topology.vllm.{prefill,decode}.dp` 4 and `.ep` 32 |
| `launch.py` | the resource assertion expects `('H200', 'step_main', 8, 2 × DP)`; DP4 only: the DP assertion expects 4 |
| `source_lock.json` | moved paths; each changed retained file's entry (profile02 also lists `launch.py` among its retained files, so its entry changes too) gets its new hash, the source package file as `source`, the old hash as `source_sha256`, and `changed: true` |
| `overlay.json` | moved paths only |
| `evidence_readiness.json` | `purpose`, `target`, and the two GPU-hour cost fields |

Everything else stays byte-identical: the staged harness, worker, probes, model metadata, trace and seal script. `profile_plan.json` stays profile02's plan: its `run_tag` and `cloud_root` still name profile02, and its hash stays bound in the manifest and the source lock. The run's actual profiler settings are `engine.profiler_config`, which equals the plan apart from the trace directory.

`replica_config_network_device` stays `h800_dgx`. The repository has no `h200_dgx` network tables, and R3 does not check this field. In the P2-32d/64d comparisons the step-cost table prices the steps, so the network tables are not used for them. The comparison reports state this.

The harness is the source package's own: native06 uses worktree `v4-balanced-native-model` at `b62c155a`; profile02 uses worktree `v4-native-profile-run` at `a90c9dbb` (branch `feat/v4-native-profile-run`, recreated for this work; the commit profile02's run manifest records). `write_run_manifest` checks each staged harness file against that commit and requires a clean harness tree.

## 5. Checks before submission (CPU)

The builder runs these steps and records them in `<case>/<package>_build.json`:

1. `compare` (section 4);
2. `seal.py`;
3. the run-manifest preflight: `write_run_manifest` on a scratch copy of the case, which records the harness commit, a clean harness tree, NATIVE `2cf0a69`, the run tag, and resources H200, `step_main` and 8 × replicas GPUs;
4. the Frontier reconstruction with R3's `verify_balanced_frontier`, run at the harness commit. Both roles must show device h200, TP8, `attn_dp` DP, EP 8 × DP, PP1 and world size 8 × DP;
5. the offline launch path for 2 × DP placeholder hosts (`node_plan`, `server_command`, `kv_transfer_config`), as in P1b;
6. `launch.py --check-only` prints `INPUT_HASHES_PASS_NO_SUBMISSION`.

Built packages: P2-32r `p2_32_reference/package_01`; P2-32p `p2_32_profile/package_02`. The first P2-32p build, `p2_32_profile/package_01`, stopped at step 3 because the builder had not put `tests/comparison/dp_placement_pp` on `sys.path` (profile02's worker imports `vllm_replay` at module level). It is incomplete and unused; the builder refuses to overwrite a directory, so the fixed build went to `package_02`.

## 6. Acceptance

- **Platform (every job).** Creator `i-fengyicheng`, H200, `step_main`, preemptible `no`, the locked image, the local NFS mount of the package, 2 × DP replicas of 8 GPUs, worker `nvidia-smi` showing H200, all replicas Succeeded.
- **P2-32r and P2-64r.** R4 (`task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py`, as accepted for native06) PASSes all 8 predicates on a local copy of the case. R4 reads the GPU type, replicas and DP from the run's own config, so it applies to H200 without change. A predicate that fails only because of the H200 or DP4 layout, with the run itself correct, goes to YC; the checker is not changed without approval.
- **P2-32p and P2-64p.** Job Succeeded; one TP0 decode trace per decode DP lane under `torch_traces/decode/dp{d}/`; `prepare_dsv4_step_cost.py --device h200` builds `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json` from them. The builder currently names profile02's two trace files; it is changed to find the TP0 trace of each decode lane in the run (one small change in `tests/`, no `frontier/` change).
- **P2-32d and P2-64d.** The Frontier run completes every request without dummy mode; the canonical client gate gives the four errors against each round and the mean of the H200 reference. The result is DIAGNOSTIC.

**Step-cost tables.** The loader reads `dsv4_step_cost.json` beside the resolved CUDA_EVENT attention input and requires the table's topology to equal the replica's, so one directory holds one topology.

| Case | Table | Built from | How the driver selects it |
| --- | --- | --- | --- |
| P2-32d | `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json` (`attn_dp` 2, EP16) | P2-32p | the default path for device h200 |
| P2-64d | `h200_mainline/diagnostic/step_cost_attn_dp4/dsv4_step_cost.json` (`attn_dp` 4, EP32) | P2-64p | `random_forrest_execution_time_predictor_config_atten_input_file` points into that directory; no `attention.csv` is read on the step-cost path |

P2-64p's profiler window (64 iterations from the start of repeat 1) holds steps with k = 1 and k = 9 newly arrived requests only with 0 decode requests. Under D-H200-STEP-COST-SUBSTITUTION the builder prices those compositions with 1 to 128 decode requests from the same measured step, and the table lists each such entry (`samples` 0, `substituted_from_num_decode_requests`). The H800 and P2-32 tables have no such gap and rebuild byte-identical.

**Runs.** `h4a_p0_diagnostic/h4a4_driver.py`, `h4a4_gate.py` and `h4a5_attribution.py` take the case name (`native06`, `p2_32` or `p2_64`). Each case pins its retained Frontier input, its table, the reference's `num_blocks` and logged maximum concurrency, and its persisted R4 result. The native06 case reproduces the H4a results byte-identically.

## 7. Limitations

- The R4 checker and harness were accepted on H800 at DP2. They read hardware and DP from the config, but they have not run on H200 or at DP4.
- The profile run's own client metrics include profiler overhead; as on H800, only its non-profiled repeats price the prefill step.
- H200 node capacity can change between the quota read and scheduling; a queue wait longer than 30 minutes is reported to YC under D-GPU-SHORTAGE-REPORT.

## 8. P2-32 lifecycle: H4b-(a) on H200

**Authority.** D-H200-H4B-LIFECYCLE. The run is DIAGNOSTIC; it feeds the H4b-(b) design and never qualifies an H800 cell. The H800 H4b-(a) job stays queued as the H800 evidence.

**Case.** `p2_32_lifecycle`, run tag `dsv4-h200-p2-lifecycle-20261025-01`, case id `dsv4-h200-p2-lifecycle-fixed128-qps1`. Layout, workload, engine, image and timeouts are those of P2-32r. The only addition is the engine-core scheduler lifecycle observer of [H4b-(a)](../h4b_lifecycle_measurement/design.md) (sections 2 and 3 there).

**Builder.** `build_package.py p2_32_lifecycle` uses the source `lifecycle`: the H4b-(a) `package_01` (`../h4b_lifecycle_measurement/package_01`, sealed `manifest.json` `67bb7e6e…`, harness `h4b-lifecycle-harness` at `7ac25c74`). Each source now names its package path relative to the task directory and the worktree it was built in, because the H4b package was built in this worktree and its configs carry this worktree's path. The section 4 changes apply unchanged: device h200, `gpu_type` H200, `charged_group` step_main, case identity, run tag, output path and the `launch.py` resource assertion. Each case records the session's H200 limit when it was built (`h200_limit`): 64 for the four P2 cases (D-H200-TEMP-MAINLINE) and 32 for this case, which is a single 32-GPU job under the 32-GPU global `step_main` limit. The source manifest already says 32, so it is not changed. The offline launch path now also checks that every server command passes the engine's `scheduler_cls` (none for the four P2 cases, the observer for this one).

**Build result** (2026-10-25, log `/data/ycfeng/tmp/glm53_dsv4/h200/lifecycle/build_package_01.log`, record `p2_32_lifecycle/package_01_build.json`): PASS. 44 sealed files; `manifest.json` SHA-256 `abbda7bfa0f190c5d6838f405b13c9058939298f24ebe5e1538ad09cc3fd2f0c`. The changes are only those listed above. The preflight run manifest binds `7ac25c74`, NATIVE `2cf0a69`, H200, `step_main` and 32 GPUs. R3 rebuilds both Frontier roles as device h200, TP8, `attn_dp` 2, EP16, PP1. All 4 server commands carry `--scheduler-cls native_scheduler_lifecycle.LifecycleObservedScheduler`. `launch.py --check-only` printed `INPUT_HASHES_PASS_NO_SUBMISSION`. A cross-check against `p2_32_reference/package_01` finds, apart from tags and paths, only the observer items: `scheduler_cls`, the observer module and its hashes, the 3 added NATIVE pins, the restaged `vllm_replay.py`, and the harness commit. `frontier.json` and `launch.py` are byte-identical.

**Submission.** One job, 32 H200, `step_main`, non-preemptible, through the Pi executor (`/ds4-exec`): quota read, `launch.py --check-only`, start of the wrapper `/data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_32_lifecycle.sh` under `setsid nohup`, and the platform identity check. The launcher stays alive until the job is terminal.

The first start, from `package_01`, failed before any job existed: the Pi HOST environment has neither `KUBEBRAIN_CLUSTER_ENTRY` nor `KUBEBRAIN_NAMESPACE`, and `brainpp` `RJobClient` accepts only the kubebrain type, which needs both (`brainpp/rjob/client.py:654-662`, raise at `:78`). The launcher had already written the run manifest, and `write_run_manifest` creates the run directory with `exist_ok=False` (`native_pd_worker.py:93`), so `package_01` cannot start again. It stays as the record of that attempt. `package_02` is the same build into a new directory (record `p2_32_lifecycle/package_02_build.json`, `manifest.json` `a572066c…af50`, the same changes as `package_01`). Its wrapper `launch_p2_32_lifecycle_02.sh` differs only in the package path, and the start command sets `KUBEBRAIN_CLUSTER_ENTRY=https://platform.shaipower.com KUBEBRAIN_NAMESPACE=shai-core KUBEBRAIN_WORKSPACE_NAME=kun-workspace-vgen2`, as the quota and status tools already do on that host.

**Acceptance.** The H4b-(a) acceptance items 1-5 (H4b design section 8), with P2-32r in the place of native06: R4 PASS on a local copy; observer files for all 4 engine cores with no `error` record and a full timeline for every formal D request; client metrics compared with P2-32r r1-r3 and reported, not gated; parts a-g from `analyze_lifecycle.py <run copy> <P2-32r copy> <output>`; results copied into the task directory.

**Result** (2026-10-25, [test report](../test_report_2026-10-25_h200_h4b_lifecycle.md)). `exp-1026-031327-540424` Succeeded on `gpu-h200-0043/0628/0705/1191`; the copy is hash-identical (70 files).
1. R4: FAIL on `native_workers` and `native_lifecycle`, both on decode headless 1191. Every serving-window check passes. The decode head 0705 began its shutdown 120 s after the drain, while the four earlier runs took 2-19 s. The headless node waits only 90 s (`stop_timeout_s`), so it stopped first and recorded `Head shutdown marker missing`. The delay lies in the head's once-per-second poll of the drained markers on the cloud volume; its cause is not determined.
2. Observer: PASS. 4 engine cores, 0 `error` records, 384 of 384 complete timelines, every READ in WAITING_FOR_REMOTE_KVS, every promotion k=1.
3. Client metrics, run mean against P2-32r: TTFT +9.1%, TPOT +2.7%, E2E +2.8%, THPT −0.3%, proxy handoff +8.6%. All are outside the P2-32r range, on different hosts; reported as a possible observer effect.
4. Parts on decoding lanes (mean ms): a 0.96, b 13.61, c 0.06, d 77.94, e 0.19, f 42.05, g 3.67. Engine c..f is 120.24 and the remainder a+b+g 18.23, for a proxy handoff of 138.47.
   - d is about 2 decode steps (39.6 ms) plus the maximum READ transfer time over the 8 handles. That maximum is quantized at the step size: 1.9 ms when every handle finishes within the first step, and about 20 ms more per further step.
   - f is one iteration of about 2 steps.
5. Persisted: inputs `diagnostic/inputs/p2_32l/` (21 files; the 4 lifecycle files stay out of Git with their hashes in `manifest.json`), results `diagnostic/results/p2_32l/`.

The run is DIAGNOSTIC evidence for H4b-(b) only. It is not an R4-accepted reference.


## 9. P2-64 wave restart diagnostic

**Authority and case.** D-P2-64-WAVE-DIAGNOSTIC, requirements item 42. `p2_64_wave/package_02`, case id `dsv4-h200-p2-64-wave-fixed128-qps1`, run tag `dsv4-h200-p2-64-wave-20261030-01`. This is one DIAGNOSTIC run, outside the H3o operator-measurement caps, with no E2E reference admission and no H800 qualification. It keeps P32+D32, TP8/DP4/EP32/PP1 per role, 8 hosts of 8 H200, `step_main`, non-preemptible, the lifecycle package's image, NATIVE `2cf0a691`, C26 overlay, engine settings and timeouts. It runs one repetition: 16 warm-up and 128 formal requests, 6144/1024 tokens at QPS 1. Submission, monitoring and download remain with the lead through `/ds4-exec`.

**Question.** [A10 section 3](../test_report_2026-10-30_a10_operator_path_replay.md) places the P2-64 TTFT difference mostly inside the native P-side prefill step. Native ungated compute is about 300 ms; a request arriving after the previous wave stopped cannot complete until about 450-540 ms after that wave end. The source wake-up path has no explicit delay. This run separates delayed start-wave messages or input handling, delayed idle-engine dummy dispatch, delayed worker entry into the CPU DP all-reduce, the reduction after all lanes joined, and work after the reduction. It supplies measured mechanism evidence before any Frontier restart-cost change.

**Engine installation.** `native_dp_wave.WaveObservedScheduler` subclasses `LifecycleObservedScheduler`, preserving its request and connector records. Scheduler resolution imports the harness module in every engine-core process before `run_busy_loop`. With the evidence-directory environment variable set, the class wraps `DPEngineCoreProc.run_busy_loop`; on entry it wraps that engine's queue and methods. The original native loop remains the executable loop. `_handle_shutdown` marks iteration boundaries, including the native idle `continue`; every completed iteration has one `busy_iteration` record. A `ParallelConfig.sync_dp_state` wrapper reports invocation of the existing 32-step sync. Input queue `put_nowait` is wrapped on the actual queue object used by the native input thread, and output queue `put_nowait` records successful wave emissions. Queue timestamps are taken before the native put and written only after it succeeds. Wrapping the input queue's existing Condition.wait records actual blocking without an empty-queue probe race. Both roles and all four DP ranks are covered.

**Worker installation.** The existing `worker_extension_cls` stays `native_full_structure.FullModelInventory`. Its import path in every worker process calls the evidence-gated bootstrap in `native_dp_wave_worker`. The bootstrap wraps `Worker.init_device`, activates only for the wave scheduler, and installs instance hooks after native device/process-group initialization on TP rank 0. It wraps the DP utility module's `sync_cudagraph_and_dp_padding` and the existing `dist.all_reduce` while that sync is active. The V2 runner calls `dispatch_cg_and_sync_dp`, whose module-global sync lookup reaches this wrapper. No device work is introduced by the hooks. All 8 TP0 workers are covered; TP1-7 remain unobserved. Retaining the extension name is necessary because the existing `NATIVE_WORKER_EXTENSIONS` allowlist controls evidence-environment propagation; a new extension name would require changing the retained replay driver.

**Records and clocks.** The schema is `native_dp_wave.v1`. Sparse and outer-call records have CPU entry `t`/`mono` and return `end_t`/`end_mono`. Nested spans store `mono` and `end_mono`; the analysis derives their wall times from the containing record's clock pair. Engine spans share the existing `<node>/scheduler_lifecycle/dp{rank}_pid{pid}.jsonl`. Worker spans go to `<node>/worker_dp_wave/dp{rank}_tp0_pid{pid}.jsonl`.

| Event | Fields beyond timestamps |
| --- | --- |
| `engine_wave_start` | schema, engine_index, dp_rank, iteration, current_wave, engines_running, step_counter |
| `busy_iteration`, `busy_loop_return` | iteration, current_wave and engines_running after the iteration, step_counter; `calls` contains the spans actually executed |
| `calls.process_input_queue` | blocked, block_requested |
| `calls.process_engine_step` | model_executed, the unchanged native boolean result |
| `calls.engine_dummy_batch` | entry and return monotonic timestamps |
| `calls.global_unfinished` | local_unfinished, step_counter before and after, all_reduce_ran, result |
| `start_dp_wave_enqueue` | wave, excluded_engine_index |
| `add_enqueue` | request_id, request_wave |
| `handle_client_request` for START_DP_WAVE | request_type, wave, excluded_engine_index, current_wave_before/after, engines_running_before/after |
| other `handle_client_request` | request_type, iteration, current_wave, engines_running, step_counter before; current_wave_after, engines_running_after |
| `engine_add` | request_id, request_wave, current_wave_before/after, engines_running_before/after, start_wave_sent |
| `wave_complete`, `start_wave_sent` | emitted wave and engine iteration/wave/running/counter state |
| `worker_wave_start` | schema, host, pid, role, dp_rank, tp_rank |
| `worker_execute_model` | tokens, request_ids, nested `calls` |
| `worker_dummy_batch` | nested `calls` |
| `worker_sample_tokens` | entry and return timestamps |
| `calls.dp_padding_sync` | forward, stage, local num_tokens/num_reqs, dp_rank/dp_size, desired_graph_mode, uniform_token_count, optional max_query_len, num_active_loras; reduced_num_tokens, graph_mode, padded_num_tokens |
| `calls.dp_cpu_all_reduce` | forward, input CPU matrix values and reduced_values (tokens, graph modes, uniform counts, max query lengths) |
| direct `dp_padding_sync`, `dp_cpu_all_reduce` | same fields and wall/monotonic spans when no outer worker call encloses the sync, including initialization |
| `worker_warmup_complete` | marks native warmup return; the forward counter resets and stage becomes serving |
| `error`, `stop` | unchanged LifecycleLog error/stop contract; native-exception spans also carry native_exception=true |

**Observer contract.** Arguments and native return values are retained; native exceptions propagate unchanged. Observation failures are isolated, buffered as `error` records, and marked with `NATIVE_SCHEDULER_LIFECYCLE_UNVERIFIED` on stderr, with the existing error cap. Without `FRONTIER_NATIVE_SERVING_EVIDENCE_DIR` neither bootstrap installs hooks. There is no device synchronization, new collective, INFO logging or native/overlay-source edit. `LifecycleLog` now uses a reentrant lock because the input thread and engine loop share its buffer. Engine records flush with lifecycle writes, once a wave ends and before the next idle wait, and at loop return/drain/shutdown. Worker records flush after each outer call on 32-forward boundaries, on the existing 2-second cadence, at warmup completion and at shutdown. Packing inner sync spans into the outer worker record reduces per-forward serialization.

**Coordinator coverage.** The native coordinator is started by `DPCoordinatorProc.run_coordinator`, and no existing harness bootstrap executes in that process. No coordinator hook is installed. The source-to-engine-enqueue interval therefore combines front-end and coordinator transport/processing. Engine enqueue versus handling distinguishes its input queue; handling versus first dummy step distinguishes the idle engine; the worker and CPU-reduction spans distinguish late joining from the reduction after the last join. This coverage separates the stated candidates without patching native sources.

**Analysis.** `tests/analysis/native_dp_wave_timeline.py <run copy> <new output directory>` reads `nodes/<host>/scheduler_lifecycle`, `worker_dp_wave`, and the proxy request records. It writes `prefill_timelines.jsonl` and `summary.json`. For every formal P prefill it records arrival, add enqueue, engine add, scheduler add/schedule, previous rank-0 wave_complete, each other P lane's matching START_DP_WAVE enqueue/handling and first subsequent dummy, that forward's DP sync/all-reduce, and the first-token update. Request IDs identify the real forward; stage and the serving forward counter join peers, and identical reduced matrices on all four lanes verify that join. Complete timelines require all four engine/TP0 lanes, every requested timestamp, matching reductions, a first-token update and valid interval ordering.

The script retains `p_blackout.py`'s previous-prefill grouping (300 ms compute, 260/760 ms arrival gaps, 100 ms served slack). It follows the last-joining lane and reports message/input queue, idle engine, worker dispatch/join, reduction after the last join, and excess work after the reduction. The largest interval identifies the candidate; all component durations remain in the row. Missing records are unavailable; inconsistent reduction joins or clock ordering are unverified and do not count as complete. Cross-host wall clocks are not measured for synchronization. Post-reduction time includes device work and CPU output/update dispatch and cannot by itself prove a pure device delay. Historical lifecycle records reproduce 43 served at once, 36 held, 302 in the 260-760 ms group, and 3 ungrouped (384 formal), with all new intervals unavailable.

**Package changes.** `build_package.py p2_64_wave` defaults to `package_02`. The source is `p2_64_lifecycle/package_01`; every other retained file must equal that source. A wave-only `--dry-check <scratch directory>` copies and compares package bytes and checks the offline 8-host launch plan without requiring a clean commit or writing into the real package directory. `--harness <clean composed harness>` selects the actual commit for the real build. File/leaf comparison, seal, preflight run manifest, R3 Frontier reconstruction, offline launch path and `launch.py --check-only` remain in the real build.

| Package file | Difference from P2-64 lifecycle package_01 |
| --- | --- |
| `config.json`, `case/inputs/native_config.json` | relocated paths; case run/job tags and output; harness_worktree; wave scheduler binding; two added harness_files and four observer tool hashes; relocated overlay hash |
| `engine.json`, `case/inputs/engine.json` | scheduler_cls becomes native_dp_wave.WaveObservedScheduler; FullModelInventory extension retained |
| `tests/comparison/calibration/native_dp_wave.py`, `native_dp_wave_worker.py` | new observers |
| `tests/comparison/calibration/native_scheduler_lifecycle.py`, `native_full_structure.py` | thread-safe lifecycle buffer and worker bootstrap; inventory API retained |
| `trace/manifest.json`, `case/inputs/trace/manifest.json` | only repetitions metadata: repeat_indices=[1], clean_repetitions=1, matching request-ID/sequence descriptions |
| `trace/SHA256SUMS`, `case/inputs/trace/SHA256SUMS` | manifest hash only |
| `case/manifest.yaml` | identity, generation, DIAGNOSTIC purpose; workload manifest/SHA256SUMS and relocated overlay hash |
| `source_lock.json` | moved paths; actual harness commit, committed-file bindings and hashes for the observers; qualification purpose |
| `overlay.json` | moved paths only |
| `evidence_readiness.json` | purpose, harness commit, observer bindings/hashes, one-repetition request count and scheduled-arrival cost |

The 16 changed/added files above leave the trace CSVs/IDs, model, Frontier input, launch.py, seal.py, native_pd_worker.py, pd_replay.py and vllm_replay.py intact. `pd_replay.py:415` already reads repeat_indices; no second repetition control is added. The scheduled-arrival cost is derived from the unchanged 127-second formal and 15-second warmup spans (142 seconds total), and both trace copies are checked with the staged canonical_workload consumer.

**Build prerequisite.** Starting harness HEAD `34cc34be` already differs from the source package in retained native_pd_worker.py, pd_replay.py and vllm_replay.py. The builder reports those paths in dry mode and refuses a real build with them; it does not restage those driver changes. The lead must compose and commit a clean harness retaining the source drivers and the approved observer changes, then pass it with --harness. native_full_structure.py also has pre-existing DP-sequence record/reset changes relative to the source package; they are outside this assignment's worker-bootstrap diff. The builder removes only the bootstrap AST for comparison and requires the inventory's remaining executable code to equal the source package. This adds native_full_structure.py as a fourth composition prerequisite and preserves the inventory contract in the real build. No real package has been built or submitted by this CPU assignment.

**CPU checks and overhead.** `check_native_dp_wave.py` drives the pinned native busy loop/methods through idle, blocking queue wake, START_DP_WAVE, 32 dummy steps, wave end, stale-wave add/restart and a second wave end. It verifies unchanged native results, event fields, error isolation, all 8 engine bootstraps, TP0-only worker bootstraps for both roles and all ranks, CPU DP utility results, native exception identity and buffered records. Five rounds of 4,000 active iterations measure engine cost; a separate TP0 dummy-forward measurement records worker cost. The serial sum is a conservative CPU estimate, not observed native wall time, and the 50 µs target is reported separately from result/schema checks.

The required `/data/ycfeng/envs/oneshot-opd-py312/bin/python` is absent on this host (both native checks fail to start with exit 127). The source-only fallback compiles unmodified pinned native method bodies with scheduler/executor and CPU tensor/collective doubles under Frontier Python 3.10.6; it explicitly does not fulfill installed-vLLM acceptance. Its 29 behavioral checks pass; the engine-only cost passes, while the serial-overhead target fails (exit 1). The packed source-only measurement (cpu_source_08) adds 48.27 µs per engine iteration and 68.11 µs per TP0 forward; the serial estimate is 116.38 µs, above the target. Earlier engine timing attempts exceeded 50 µs, leading to packed iteration spans; worker packing reduced its measured cost from 92.56 to 68.01 µs. Do not infer less than 1% total native overhead from the engine-only measurement. The 19 timeline tests, 63 existing observer-bootstrap tests and 1 GLM observer test pass (83 total).

**Record volume.** The existing lifecycle logs omit dummy-only iterations. Their active D-lane cadence is 40.8983, 40.8997, 40.9043 and 40.9116 iterations/s (median 40.9020). For 150 seconds, four P lanes at the estimated 5 ms dummy cadence bound P iterations at 120,000; four D lanes give about 24,541 iterations. With one engine record plus one packed worker record per iteration and one extra sample_tokens record per D forward, this is about 313,624 new records. CPU-check encoding sizes (about 682 bytes/engine and 675 bytes/worker dummy record) give about 199.5 MB including estimated D sample records. Existing lifecycle records, initialization and sparse messages add to this; actual P dummy volume will be lower when waves sleep. This is an estimate from cadence evidence, not a measurement of the proposed run.

**Cost.** The Oct 26 P2-64 lifecycle platform receipt has earliest container start 05:01:19 UTC and latest finish 05:26:51 UTC: 1,532 seconds of envelope time, 26.302 aggregate container GPU-hours. Its formal r1 last token is mono 2921515.874058352 and r3 last token 2921910.244711168. Removing the two later repetitions saves about 394.37 seconds while keeping startup, the first warmup, final drain and shutdown. The conservative envelope estimate is (1532 - 394.37) × 64 / 3600 = 20.23 GPU-hours, about 19 minutes wall time. Subtracting from staggered-container accounting instead gives about 19.29 GPU-hours; 20-21 GPU-hours remains the planning value. Queue time and host variation are not predicted, and the unchanged 3600-second timeout remains the guard.

**Lead review decisions (2026-10-30).**

- *Composed harness.* The run must differ from the P2-64 lifecycle run only by the observers. The harness is therefore a new branch from the lifecycle harness commit `7ac25c74` that adds `native_dp_wave.py`, `native_dp_wave_worker.py`, the CPU check scripts, the locked `LifecycleLog` and the worker bootstrap on the `7ac25c74` copy of `native_full_structure.py`. The later driver and inventory changes on main stay out of this run.
- *CPU check environment.* `/data/ycfeng/envs/oneshot-opd-py312` was removed from this host at 05:22 UTC on 2026-10-30, outside this task. The source-only check replaces the installed-wheel check: it runs the unmodified NATIVE `2cf0a691` method bodies, and the H4b check found the installed 0.28.0 wheel byte-identical to NATIVE for the engine-core, scheduler and output files. The lead also checked in the NATIVE source every attribute the hooks use: `DPEngineCoreProc` loop and methods, the two queues, `EngineCoreRequestType.START_DP_WAVE`, `EngineCoreOutputs.wave_complete` and `start_wave`, the static `ParallelConfig.sync_dp_state`, the `Worker` methods, `get_tp_group`, `SchedulerOutput` fields, and the V2 runner (the default on CUDA for `DeepseekV4ForCausalLM`) reaching `sync_cudagraph_and_dp_padding` through `dispatch_cg_and_sync_dp`. The run's soft checks (records from all 8 engines and 8 TP0 workers, no error records) are the installed-path check.
- *Observation cost.* The serial CPU estimate (116 µs per iteration in Codex's run, 143 µs in the lead's rerun on a loaded host) exceeds the 50 µs target. The lead accepts it for this run. The measured mean P wait is 264.2 ms. Even 32 observed dummy iterations at 143 µs add about 4.6 ms, under 2% of that wait. A D step of about 24.4 ms gains under 0.6%. The client TTFT, TPOT and throughput of this run are compared with the lifecycle run (`exp-1026-125836-702505`), so any perturbation is visible.

**Acceptance and handoff.** Before submission, the source-only CPU check must pass every behavioral check. Its timing checks are reported, not gated (lead decision above). The real builder must pass all six section 5 checks at the composed clean harness commit. After the single native run: observer files for all 8 engine cores and 8 TP-rank-0 workers, no error records, at least 95% complete formal P timelines (122 of 128), and client TTFT/TPOT/E2E/THPT reported without a gate. Historical reproduction and synthetic tests establish analysis behavior, not that future-run acceptance. Remaining CPU prerequisites and the overhead limit make B25 a partial handoff; acceptance, commits, package build and GPU actions remain with the lead.
