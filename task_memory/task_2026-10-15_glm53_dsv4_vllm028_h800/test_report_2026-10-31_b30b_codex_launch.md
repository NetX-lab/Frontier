# B30b Codex integration and hardware execution

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Verified the UUID identity repair with 8 original node/phase replays and 166 affected tests; prepared fresh links/KV02 and analyzed all 84 F2 trials. |
| 2026-10-31 | Recorded integrated source, merged checks, quota/readiness and approved hardware plan before direct Codex submission. |

B30b scoped commit `dbd3ff6f7e2cfdf30050945039df7b66cced7d14` is integrated as `5d1754d0eb11d6cd1fe7a164f7f3161aa2f43f07`. Decision46 and D-CODEX-DIRECT-RJOB authorize direct personal Python RJobBackend submission/monitoring. Measured CC remains formal; F1b/F2/F3 are diagnostics.

Expected: 163 required CPU tests without failures/errors/skips, both locks on committed mounted sources, exact old-plan preservation, two authorized H200 jobs, complete phase artifacts plus actual platform/container terminal evidence. Actual CPU checks: **163 PASS in 5.42 s**, 0 failures/errors/skips. Each stage lock covers15 committed paths, including the main-only K5/runtime helpers. Existing8 successful historical listings are byte-identical and5 older failures retain their errors in the sidecar report. No simulator or formal-data file changed.

Exact merged command:

```bash
ulimit -v 31457280
export PYTHONPATH="$PWD:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout 240s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_b30b_ex_situ_probes.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_vllm_dsv4_forward_collectives.py tests/unit/test_vllm_forward_collectives.py tests/unit/test_collective_kernel_evidence.py task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/tests -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/b30b_codex_merged_20261031_pytest --junitxml=/data/ycfeng/tmp/b30b_codex_merged_20261031_pytest.xml
```

Working directory: current main worktree. CPU Python3.10.6; exact import environment and command are in `h3o_operator_costs/b30b/integration_20261031/readiness.json`. Retained JUnit XML and two source-lock/worker-command previews are beside it. The original sidecar report documents its Python/torch dependency paths and synthetic-fit results.

| Job | Cases/trials | GPUs | Estimated wall s | Estimated GPUh | Payload timeout s / GPUh exposure |
| --- | ---: | ---: | ---: | ---: | --- |
| b30b_links_kv |628 instances/1884 pass rows |16 |1976.8120210185284 |8.785831204526794 |4500 /20 |
| b30b_launch_depth |84 trials |1 |807.6709306640623 |0.22435303629557288 |3600 /1 |

Combined estimate9.010184240822367 GPUh. Worker timeout exposure21 GPUh excludes platform initialization/billing overhead. The current actual case arrays and arithmetic are the retained `h3o_operator_costs/b30b/fresh_dry_run_20261031_01/` and74-check `validation_20261031/generated_plan_audit.json`. The older README/dry-run budgets remain historical.

Personal read-only quota at2026-10-31T15:28:51–15:28:53Z:0 non-preemptible H200 reservations; probes make17/32 PASS. H200 non-preemptible quota393/512;15 nodes have>=8 usable GPUs. Read-only SSH check confirms both intended personal cloud output directories are absent; no cloud write was performed by that check. Quota and capacity are snapshots, not placement proof.

Submission entry: `/data/ycfeng/tmp/stepmind-env/bin/python -u h3o_operator_costs/g5h/launch/submit.py --config h3o_operator_costs/b30b/stage_b30b_links_kv.json` and the matching `stage_b30b_launch_depth.json`, after handbook personal auth/proxy initialization. Exact absolute argv, host, source identity, launch log paths and job names are recorded in execution receipts as they become available. Launchers remain alive throughout the workloads. Stage image is the locked company digest, source mount is the current local worktree, cloud output is under `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/`.

Hardware status at 2026-10-31T15:39Z: **links/KV FAILED before measurement; F2 execution PASS, analysis pending**.

| Attempt | Exact RJob | Platform / replica | Business evidence |
| --- | --- | --- | --- |
| links/KV01 | exp-1031-233253-628432 | Failed / both Failed; both container exit1, restart0; launcher exit1 | All four phases skipped at physical UUID verification. Two distinct H200 nodes0082/1112, 8 CUDA UUIDs each, 21/21 runtime checks. No collective/NIXL rows. |
| F2 depth01 | exp-1031-233305-902815 | Succeeded / Succeeded; container exit0, restart0; launcher exit0 | F2 phase returncode0, 84 trial records, 13 graph traces, NVIDIA H200/torch2.13.0+cu130/CUDA13.0. |

Both jobs used creator i-fengyicheng, step_main, preemptible=no, the locked company digest, RDMA resource requests and current local NFS100.96.128.195. Exact filtered platform/container JSON, node placement/runtime/phase records and redacted replica logs are retained under `h3o_operator_costs/b30b/hardware_20261031/`. Raw F2 trial/graph artifacts stay at `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/b30b_launch_depth_01/`.

Cause observation: PyTorch reports bare UUIDs, while nvidia-smi prefixes the same UUIDs with GPU-. The selection code compares the full strings and rejects every requested GPU; CPU replay of both original node records returns the same ValueError. Missing ip in the platform background p2p_ping script is separately observed and is not substituted for the business failure. Correct UUID identity comparison, re-run the original reproduction and affected tests, and use a fresh cloud generation. The successful F2 job is not rerun. F2 numerical analysis and real links/NIXL measurements remain unfinished; formal qualification and A10 accuracy are not established.

## UUID repair and fresh links/KV02 preparation

The code change normalizes CUDA allocation UUIDs to the NVML `GPU-` representation before the existing allocation census and membership checks. The gain is correct admission of the actual allocated physical GPUs; it does not change simulator timing. It adds one production line in `frontier/profiling/collectives/rank_placement.py` and expands the existing order test to both observed UUID representations. No allocation or physical-index check is bypassed. Original stage01 and cloud evidence remain unchanged.

Expected: the six representation/order regressions and each of the original two-node/four-phase placements pass; missing, duplicate or unallocated devices remain rejected. Actual: focused RED **3 FAIL / 3 PASS**, focused GREEN **6 PASS / 25 deselected in 0.35 s**, original raw node records **8/8 PASS**, canonical duplicate identities rejected, required affected suite **166 PASS in 5.39 s**, zero failures/errors/skips. Exact CPU commands, Python3.10.6, declared environment, source head, UTC bounds and JUnit evidence are in `h3o_operator_costs/b30b/uuid_repair_20261031/commands.json` and adjacent files. This profiling-only change does not reach simulator scheduling/pricing or public-example paths; the affected profiling/launch suite settles its regression scope.

Original reproduction command is the direct `select_physical_devices` call for every phase's actual indices and the unchanged `placement.json` values under `hardware_20261031/links_kv_01/nodes/`; before/after results retain input paths and hashes. No new GPU measurement is claimed by the CPU PASS.

Fresh config: `h3o_operator_costs/b30b/stage_b30b_links_kv_02.json`, job tag `dsv4-b30b-links-kv-02`, cloud root `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/b30b_links_kv_02`. Only job tag and output root differ from stage01; phase cases and budget estimates are identical. The existing planner CLI generated `fresh_dry_run_20261031_02/`: **628 cases, 16 H200, 1,976.812021 estimated wall seconds, 8.785831 GPUh; 4,500-second payload timeout gives 20 GPUh exposure**, excluding platform overhead. Fresh personal quota/output checks, committed source lock and actual submission/terminal evidence follow below. F2 is not submitted again.

## F2 initial numerical analysis

The complete 3,258,535-byte `launches.jsonl` was downloaded read-only to `/data/ycfeng/tmp/glm53_dsv4/b30b_f2_analysis_20261031/` (SHA256 `5dfd37990d9f48679f4b28dff55a725f3f8316a9b79872042c4713aa16ac2b0f`). The existing `launch_analysis.py` CLI analyzed all **84/84 uncensored trials** with the native **2,240 kernels/forward** count. Output: `h3o_operator_costs/b30b/f2_analysis_20261031/analysis.json`; exact download/analysis commands and logs are adjacent.

Observed graph depth is 1,020 for 1–32 kernels/graph, then 610/303/153/77/40/20/10 for 64/128/256/512/1,024/2,048/4,096 kernels/graph. All six trials at each graph size agree, including both 500/1,000-ms sleep groups; plain launch median is 1,020. The existing two single-capacity hypotheses return **AMBIGUOUS**, with log RMSE1.627085 for graph launches and1.255410 for kernel nodes. `predicted_K` and `predicted_minus_19` are null. These poor fits do not authorize selecting either model's candidate K (255.34 or7.30) for replay. The curve needs a cause-specific interpretation before a predictive F2 input; the actual successful GPU measurements are retained and are not repeated.
