# H200 temporary mainline design: DSV4 on H200-32 and H200-64

## Modification History

| Date | Change |
| --- | --- |
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
