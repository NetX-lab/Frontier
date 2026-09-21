## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Closed full five-mode completion: 2000requests,1500warmup+500formal; all identity checks PASS. |
| 2026-09-08 | Validated first-formal scopes from all three fresh H200 compute groups and completed the baseline-P/new-V mapping; first-forward before/after references validated. |

# D019 compute collection verification

PASS_COLLECTION_DIAGNOSTIC_ONLY for attention, MoE and detail first-forward records. CUDA-span calibration and clean official-TTFT acceptance remain INCOMPLETE. Large probe/run sensitivity is observed and retained, not fitted away.

## Execution

Analysis host: CPU master. Working directory `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`. Python `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python` version3.13.13, conda `dev-vidur-v03-hopper-e2e`, `PYTHONDONTWRITEBYTECODE=1`.

Analyzer absolute path: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_first_batch_op_rca_compute.py`. Exact three executed commands are retained in `analysis/d019-compute-results/report.md`, under Execution and Detail parser command. They select each actual `analysis/h200-d019-compute-01/runtime/compute_{attention,moe,detail}/operators` directory and write one corresponding `analysis/d019-compute-results/{attention,moe,detail}/` output. All three returned `PASS_COLLECTION_DIAGNOSTIC_ONLY` with four first-request rank groups.

GPU mode manifests record Python3.10.16, vLLM commit `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`, H200 step_main, TP4/DP2/EP8, eager FLASHINFER, uniform routing, prefix caching disabled and the explicit scope allowlists. GPU collection is owned and launched by root. The original five-mode bracketed suite is complete; the final addendum validates all2000request records in addition to the first-forward scope checks.

## Criteria and evidence

| Criterion | Expected | Actual |
| --- | --- | --- |
| Formal identity | `cmpl-pf4096_dc1024:0-0`, one4096prefill request, no local decode | PASS on all12rank/group records |
| Physical global input | `[4096,1]` for first realDP0, preserved in runtime metadata | PASS all12records |
| Rank provenance | Actual realDP lane with TP0–3, PP0 | PASS each group |
| Attention coverage | 48each QKV/RoPE/KVsave/prefill/pure output GEMM | PASS240scopes perTP |
| MoE coverage | 96gating,48shuffle,48GGparent,48sum | PASS240scopes perTP |
| Detail coverage | 48each input norm/post norm/output init/W1/activation/W2; one embedding/final norm | PASS290scopes perTP |
| Timing family | `cuda_event/default/per_scope`; unique contiguous scope sequences; positive finite values | PASS all3080selected first-forward scopes |
| Pairing | GG and sum are siblings in same run/rank/layer | PASS48pairs perTP |
| No artificial aggregation | No sum across ranks, separate runs or nested parent/child | PASS parser output structure |
| Full mapping register | 11nonzeroFparents,4missing model terms,3fused zero names and1routing alias | PASS19rows in baseline_compute_mapping.csv |
| Refreshed profile comparison | Seven P/S/newV rows with explicit shape/context limits | PASS p_s_v_linear_attention.csv |

The 3080 scoped events are `(240+240+290)*4`; this count covers the selected real DP participants, not independent dummy-lane timings. A peer token count of1 alone does not establish dummy status; that conclusion still uses scheduler/request evidence.

## Numerical evidence and limits

First before-reference event spans across TP0/1/2/3:79.307357788/79.550849915/79.559837341/79.498626709ms. First TP0 attention/MoE/detail probe spans:92.264129639/92.876190186/96.684608459ms. Relative increases to TP0 before-reference:16.337415%,17.109172%,21.911272%. After-reference first-forward was subsequently validated; see addendum below. These observed increases prevent treating the complete instrumented forward as the clean target or assigning its excess directly to production CPU cost.

Pure output projection is now measured independently: TP0 total1.454815997ms versus baseline Frontier2.245632ms (+54.358%). Gated activation totals6.548864ms onTP0, with6.548640–6.560480ms acrossranks. Expert sum from the other group totals2.790784ms onTP0,2.790784–2.859296ms acrossranks. They are separate known work; the net repair cannot be inferred by adding these two omissions to an old profile that also included substitute copy work and different routing.

All first-forward compute rows, exact mappings, signed/absolute errors, rank ranges and P/S/V distinctions are preserved in `analysis/d019-compute-results/report.md` and its CSVs. There is no fresh full-forward Frontier acceptance result in this report.

No check failed during this collection analysis. New unresolved design issues: none. Pending: root's communication-calibrated fresh Frontier result and causal closure of remaining profile/runtime-context gaps. Per root's recorded YC steering, optional naive-protocol modeling is deferred and the current ideal collective abstraction is retained.

## First-forward bracket and repaired-P addendum

Both before and after references now pass actual first-request checks on all fourTP workers: correct4096/0 local shape,physical[4096,1],no selected op probes,positive event durations and unique first-prefill records. `analysis/d019-compute-results/reference_bracket.json` preserves complete source rows and12per-rank/probe comparisons. After-reference TP0/1/2/3 spans78.118782043/78.112770081/78.021697998/78.157279968ms; before→after TP0 drift−1.188575745ms (−1.499%). Probe inflation remains material against both references. Full after-mode formal completion was still in progress when its first-forward record was extracted.

Root's bounded repaired-MoE predictor receipt `/data/ycfeng/tmp/issue26-d019-moe-integration-query-02/query_receipt.json` yields72.327535217ms endpoint and22.173354467ms complete expert computation across48layers. The mapping CSV now includes explicit after-MoE P and revised matching V scope: grouped-GEMM plus sibling sum19.718015995ms. Sum is covered inside the repaired P and is not added twice. The receipt's global4096simulation shape is preserved against vLLM4097; communication overcount remains to be corrected, so the apparent total error below10% does not establish closure.

## Full suite completion closeout

PASS_FIVE_MODE_COMPLETED_COLLECTION. All five modes have400records, each split into100requests in each warmup roundr0/r1/r2 and100formal requests. All400response IDs per mode are unique. Every request records4096prompt tokens and1024observed completion tokens with ordered first-token/completion timestamps. The existing identity analyzer validates all eightworker logs and proves each formal request receives exactly one4096prefill on the fourTP ranks of exactly oneDP lane. The two reference modes have no selected op batches; each compute mode has exactly eightselected real batches (one perworker).

Total2000completed requests =1500warmup+500formal. This closes the earlier full-mode-completion pending item. The report does not claim clean officialTTFT or first-forward numerical calibration acceptance. No new measurement was launched. Parent/root's observed unit inactive/ExecMainStatus0 is corroborated by all completed request artifacts.

Output: `analysis/d019-compute-results/full-suite/summary.json` plus one per-mode identity report. Existing reusable analyzer absolute path: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_diagnostic_identity_analysis.py`. It was run with `--mode batch` for all modes because op collection is intentionally bounded to one formal batch; its unrestricted ops gate does not apply here.

Exact executed commands, with the same working directory/Python environment stated above:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-compute-01/runtime/batch_before/batch --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-compute-results/full-suite/batch_before.json --mode batch > /data/ycfeng/tmp/issue26-d019-suite-validation/batch_before.stdout.log
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-compute-01/runtime/compute_attention/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-compute-results/full-suite/compute_attention.json --mode batch > /data/ycfeng/tmp/issue26-d019-suite-validation/compute_attention.stdout.log
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-compute-01/runtime/compute_moe/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-compute-results/full-suite/compute_moe.json --mode batch > /data/ycfeng/tmp/issue26-d019-suite-validation/compute_moe.stdout.log
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-compute-01/runtime/compute_detail/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-compute-results/full-suite/compute_detail.json --mode batch > /data/ycfeng/tmp/issue26-d019-suite-validation/compute_detail.stdout.log
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-compute-01/runtime/batch_after/batch --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-compute-results/full-suite/batch_after.json --mode batch > /data/ycfeng/tmp/issue26-d019-suite-validation/batch_after.stdout.log
```

All five processes exited0; all five JSON reports have statusPASS,8workers,100formal requests. The outputs use the existing analyzer's exclusive creation mode, so a new output path is required for any later independent rerun. No rerun is currently necessary.
