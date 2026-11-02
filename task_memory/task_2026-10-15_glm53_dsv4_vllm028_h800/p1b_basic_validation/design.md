# P1b design: H800-64 basic native validation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: purpose, resources, priority, inputs, package changes, pre-submission checks, acceptance and limitations. |
| 2026-10-25 | Corrected the port check to what `pd_replay.kv_transfer_config` actually checks; the launch-path check now also builds the KV transfer config; recorded the package_01 build. |

**Authority.** D-P1-BASIC-PARALLEL (purpose, resources and acceptance recorded before submission), D-H4A-CHECKPOINT (P1b runs in parallel at lower priority), D-V4-H800-RJOB-AUTHORITY, D-GPU-SHORTAGE-REPORT, D-INPUT-PERSISTENCE and D-MAINLINE-LOOP-CAP-DSV4 in [requirements.md](../requirements.md). Ticket: [10](../tickets/10-h800-predictive-e2e.md) `p1_h800_64.basic_validation`; binding row P1-H800-64 in [ticket 17](../tickets/17-dsv4-canonical-case.md).

## 1. Purpose

Show that the native DSV4 P/D serving setup that produced native06 on 32 H800 also runs correctly on 64 H800 with the P1 layout. Ticket 10 lists what must be verified: all 64 ranks and both roles, the real device, backend and runtime, initialization, representative forwards and their output, and terminal evidence.

The run is labelled `basic_validation` and DIAGNOSTIC:
- It is not the P1 formal reference. The P1 formal campaign still needs the P0 30% promotion receipt (ticket 10 `p1_h800_64.formal_calibration`).
- Its client metrics are recorded for information only. They are not compared with Frontier or with any threshold, and they make no calibration claim.

## 2. Case

| Item | Value |
| --- | --- |
| Hardware | H800, charged group `codesign`, non-preemptible |
| Layout | P32 + D32. Each role is one Replica with TP8, attention DP4, EP32, PP1; `data_parallel_size_local` 1, so one host per DP lane |
| Hosts | 8 replicas × 8 GPUs = 64 GPUs; per host 64 CPUs and 512 GB, as native06 |
| Workload | 68f891b unchanged (ticket 17 requires it for every V4 binding): per repetition 16 warmups and 128 formal requests of 6144/1024 tokens at QPS 1; three repetitions |
| Engine, image, NATIVE source, overlay, dummy weights, routing | native06 unchanged: vLLM 0.28.0 at `2cf0a69`, accepted C26 model overlay, `load_format` dummy, `balanced_topk`, async scheduling, FULL_DECODE_ONLY graphs, chunked prefill and prefix caching off |
| Harness | native06's: worktree `v4-balanced-native-model` at `b62c155a`, staged files unchanged. No lifecycle observer |
| Case identity | `case_id` `dsv4-h800-p1-basic-fixed128-qps1`, `run_generation` `v4-native-v028-h800-p1-basic-20261025-01`, run tag `dsv4-h800-p1-basic-20261025-01` |
| Cloud output | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p1_basic_20261025/case/runs/groundtruth_clean/dsv4-h800-p1-basic-20261025-01` |

**Why the harness needs no change.** These all read the DP size and host count from the config instead of assuming DP2 or 4 hosts:
- `native_pd_worker.node_plan` requires `2 × (dp / local)` peers. It assigns roles by sorted host position, gives lane `position % (dp / local)`, and makes peers 0 and `dp / local` the heads.
- `pd_replay.kv_transfer_config` checks that the two NIXL side-channel windows `[port, port + dp)` are disjoint and contain neither the role's HTTP port nor its DP RPC port. With DP4 the windows are 5600-5603 (P) and 5700-5703 (D); HTTP 8100/8200, DP RPC 24471/24571 and proxy 8000 lie outside them.
- The idle and lane checks iterate over `range(dp_size)`.
- The R2 checker requires `replica == 2 × dp / local`, `gpu == tp × local`, `len(nodes) == 2 × dp`, and rank groups `world` and `ep` = `range(tp × dp)` and `dp` = `range(rank, tp × dp, tp)`. R3's Frontier check requires `attn_dp = dp` and `moe_expert_parallel_size = tp × dp`.

## 3. Priority and resources

- **Priority.** H4b-(a) (`exp-1025-121318-827142`, 32 H800) is the higher-priority job and is still Pending in `codesign-default`. P1b is submitted only after that job is Running or finished, and only when a quota read shows at least 64 free H800. P1b never displaces a ready P0-path job (requirements.md, "Highest-priority ready work receives shared staff/quota first").
- **Shortage.** At the last read (13:14 UTC and again later) the queue had 23 free H800. That is not enough for H4b-(a), let alone P1b. YC was told under D-GPU-SHORTAGE-REPORT.
- **Cost.** native06 took 12.0 GPU-hours on 32 GPUs, which is 22.5 min of job time. P1b is expected to take 25-35 min on 64 GPUs, about 27-37 GPU-hours. `timeout_seconds` 3600 caps it at 64 GPU-hours.
- **Soft gates.** The run uses 64 GPUs. The only in-run collectors are native06's: the `FullModelInventory` worker extension and the serving inventory. They isolate their own errors (`native_full_structure.observe` and `error`) and do not stop serving. The run checker runs after the job ends.
- **Loop cap.** At most 3 GPU jobs for P1b without a PASS. After that P1b stops and goes to YC with the evidence.

## 4. Package changes

`build_package.py` copies native06's package (`v4_p0_native_20261021/candidate_h800_p0_06`) into `package_01/`, applies only the changes below, and then compares every file with native06. Launch outputs and `case/runs/` are not copied. Every path field pointing at native06's directory is moved to `package_01/`.

| File | Change |
| --- | --- |
| `config.json`, `case/inputs/native_config.json` | moved paths; `replica` 4 → 8; `engine.data_parallel_size` 2 → 4; `run_tag`, `job_tag`, `output_dir`; the hashes of `case/inputs/frontier.json` and `overlay.json` in `retained_files` and `tool_hashes` |
| `engine.json`, `case/inputs/engine.json` | `data_parallel_size` 2 → 4 |
| `case/inputs/frontier.json` | `replica_config_attn_dp` 2 → 4; `cluster_config_prefill_replica_config_moe_expert_parallel_size` and `cluster_config_decode_replica_config_moe_expert_parallel_size` 16 → 32 |
| `case/manifest.yaml` | `case_id`, `run_generation`, `purpose`, new `diagnostic_only: true` and `e2e_reference_admission: false`; `topology.vllm.{prefill,decode}.dp` 2 → 4 and `.ep` 16 → 32; `policy.priority` P0 → P1; `runtime_source_overlay.sha256` (the moved `overlay.json`) |
| `launch.py` | the resource assertion expects 8 replicas; the DP assertion expects 4 |
| `overlay.json`, `source_lock.json` | moved paths only |
| `evidence_readiness.json` | `purpose`, `target`, `cost` |

All other files are byte-identical to native06: the staged harness, worker, probes, model metadata, trace and seal script. `case/manifest.yaml` keeps native06's `groundtruth_checker_path`, as H4b-(a) does; the run checker is named in section 6.

`frontier.json` is changed only so that its P/D topology matches the native engine; R3's Frontier check requires this. It is not a P1 prediction input: any Frontier P1 comparison belongs to P1f.

## 5. Checks before submission

1. `compare`: the package differs from native06 only by section 4.
2. `seal.py`, then the run-manifest preflight (`native_pd_worker.write_run_manifest` on a scratch copy of the case). It records harness `b62c155a`, a clean harness tree and NATIVE `2cf0a69`.
3. Frontier reconstruction: R3's `verify_balanced_frontier` on the package config with harness commit `b62c155a`. Expected: both roles with `attn_tensor_parallel_size` 8, `attn_dp` 4, `moe_expert_parallel_size` 32, `num_pipeline_stages` 1 and world size 32.
4. Offline launch path: `node_plan`, `vllm_replay.server_command` and `pd_replay.kv_transfer_config` for 8 placeholder hosts, building the same command as the worker. Expected: hosts 0-3 are P lanes 0-3, hosts 4-7 are D lanes 0-3; hosts 0 and 4 are heads; the others are headless with `data_parallel_start_rank` equal to their lane; every command has `--data-parallel-size 4`.
5. `launch.py --check-only` prints `INPUT_HASHES_PASS_NO_SUBMISSION`.
6. Quota read (section 3) just before submission.

## 6. Acceptance

1. **Platform.** Creator `i-fengyicheng`, H800, `codesign`, preemptible `no`, the locked image, the local NFS mount of `package_01/`, 8 replicas of 8 GPUs, all Succeeded.
2. **Run checker.** The checker that accepted native06 (R4, `task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py`, SHA-256 `67f31923…`, freeze `FINAL_FREEZE_R4.json`) is run on a local copy of the case, as for native06. It must PASS all 8 predicates. Expected values for this layout:
   - `native_workers`: 8 distinct native nodes, both roles and all 4 DP lanes per role; source, model and GPU verified. This covers the 64 ranks and roles, and the real device and runtime.
   - `rank_runtime_graph`: 64 physical ranks, 6 phases per rank, every rank's world, EP and DP group membership, device UUID and graph configuration, and successful serving forwards in every repetition and phase. This covers initialization, backend and representative forwards.
   - `harness_pinned_and_client_rows`: 128 formal client rows in each repetition, with full outputs. This covers the output.
   - `requests_and_idle_lanes`: all boundaries, both roles and all 4 DP lanes.
   - `native_lifecycle` and `terminal_status`: ordered shutdown, return codes 0, Succeeded with no restarts. This covers the terminal evidence.
   - `checkout` and `inputs_unchanged`.
3. **Record.** The test report lists the 8 predicates with their evidence. It also gives the client TTFT, TPOT, E2E and THPT from the existing client reader, labelled DIAGNOSTIC basic_validation, not compared, and not a reference. The four ticket 10 basic-validation boxes are ticked only after items 1-2 pass.

If a predicate fails, its cause is diagnosed from the run's own evidence before any rerun. A checker predicate that fails only because of the DP4 layout, with the run itself correct, is reported to YC. The checker is not changed without YC's approval.

## 7. Limitations

- The R4 checker and the harness were accepted at DP2 on 4 hosts. They are written for any DP size, but they have not run at DP4.
- Reusing native06's workload makes the run shaped like a formal campaign. Its labels (`purpose`, `diagnostic_only`, `e2e_reference_admission: false`, its own `case_id`) keep it from being admitted as the P1 formal reference.
- The cost estimate is scaled from native06's job time; startup on 8 hosts may take longer.
