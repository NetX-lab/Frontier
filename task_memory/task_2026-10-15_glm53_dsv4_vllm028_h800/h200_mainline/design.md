# H200 temporary mainline design: DSV4 on H200-32 and H200-64

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: purpose, cases, resources and waves, package changes, checks, acceptance and limitations. |
| 2026-10-25 | Recorded the package names actually built: P2-32p uses `package_02` after an incomplete `package_01` build; `launch.py` is a changed retained file for profile packages. |

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

## 7. Limitations

- The R4 checker and harness were accepted on H800 at DP2. They read hardware and DP from the config, but they have not run on H200 or at DP4.
- The profile run's own client metrics include profiler overhead; as on H800, only its non-profiled repeats price the prefill step.
- H200 node capacity can change between the quota read and scheduling; a queue wait longer than 30 minutes is reported to YC under D-GPU-SHORTAGE-REPORT.
