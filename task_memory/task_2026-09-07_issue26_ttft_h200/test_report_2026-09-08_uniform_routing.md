## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded the actual H200 uniform-router preflight and CPU materializer comparison. |

# Uniform routing verification

## Execution

GPU worker: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_h200_uniform_groundtruth_worker.sh. Launched command:

```bash
bash tests/e2e/issue26_h200_uniform_groundtruth_worker.sh /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-uniform-clean-01
```

Exact platform command: /data/ycfeng/tmp/issue26-h200-network/launch-uniform-clean-01.sh. RJobyc26-h200-uniform-clean-20260908-01; H200step_main; pinned image recorded in run_manifest.json. Source vLLM46f7b179f, clean exact instrumentation branch. GPU Python /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python3.10.16; torch2.8.0+cu128. CPU comparison uses /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python3.13.13 and the actual pure frontier/moe_ep_workload.py materializer.

## Criteria and evidence

- GPU runtime and8H200 visibility PASS.
- Existing VLLM_MOE_UNIFORM_ROUTING=1 branch calls real fused_topk -> uniform_topk;24 checks spanning8GPUs times token counts1/4096/4097 allPASS.
- tokens1:8assignments, counts0or1; tokens4096:32768assignments, every expert256; tokens4097:32776assignments, expert0..7count257, others256.
- Weights are exactly1/8; all128expert IDs covered by the count vector.
- CPU comparison calls materialize_layer_ep_workload with routing_ratios={i:1/128 for i in range(128)}, router_topk8, EP8, the same token count, contiguous ownership, then compares all128counts. All24vectors yield absolute count error0 and relative distortion0.

Selected evidence: runs/h200-uniform-clean-01/uniform-preflight/routing_check.json; analysis/uniform_routing_same_population.json; runtime/routing_mode.txt confirms the clean child uses the enabled setting. This is real router plus real materializer evidence, not a label-only comparison.

## Limits and current status

This check does not establish equal token populations or batch scheduling in a distributed run. It does not measure clean TTFT. Clean vLLM collection is running; Frontier requires uniform routing profile selection (D012pending) and complete fresh inputs. No new numerical parity claim is made. Standard-fused-topk profiles cannot substitute for uniform_topk timing rows.

## Profile-selection preflight

Executed the real resolve_moe_gating_routing_runtime_path('balanced') and filter_moe_gating_routing_topk_rows on supplements/moe-uniform-01/moe.csv in the CPU environment. Observed current requested pathstandard_fused_topk, available pathuniform_topk, expectedValueError: `No moe_gating_routing_topk profiling rows match the requested routing_runtime_path='standard_fused_topk'.` The negative checkPASS proves the explicit selection boundary remains unresolved; it does not permit relabeling or fallback. Receipt:analysis/uniform_profile_selection_preflight.json.

## Completed clean groundtruth validation

Observed both OFFICIAL_TTFT_EXECUTION_COMPLETE and UNIFORM_GROUNDTRUTH_EXECUTION_COMPLETE. Actual validation command from active worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_official_ttft_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-uniform-clean-01/runtime/clean --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/uniform-clean-01
bash -n tests/e2e/issue26_h200_uniform_groundtruth_worker.sh tests/e2e/issue26_h200_official_ttft_worker.sh
git diff --check
```

PASS:400unique client/server completions with complete4096/1024tokens; three drained100-requestwarmups and100uniqueformal joins; finite positive metrics. Officialservermean127.38051176071167ms; clientmean133.15956441ms. Queue-arrivalspan47.14090869948268s. New trace and ID mapping created. No uniform Frontier prediction exists yet, so paired absolute/relative TTFT errors are NOT_EVALUATED.

Runmanifest template mistakenly inherited prior completion/provenance fields; corrected before result validation with before/after evidence analysis/uniform_clean_manifest_correction.json. Final PASS refers only to this run's actual analyzer and worker completion. No prior numeric results were used.
