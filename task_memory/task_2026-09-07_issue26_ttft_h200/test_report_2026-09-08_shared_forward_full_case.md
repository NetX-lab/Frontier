## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded final-source fresh100request replay, token conservation, batch membership, and independent runtime metrics. |

# Shared-forward full-case validation

## Execution

Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`. Branch task/issue26-ttft-h200-20260907. Simulator source8ba22b49; later ff8c229a and15a35242 modify offline analyzers only. CPU master conda `dev-vidur-v03-hopper-e2e`, Python3.13.13. Exact settings/argv/runtime audit and selected outputs are under analysis/cpu-shared-forward-01. Fresh predictor/collective/htsim scratch `/data/ycfeng/tmp/issue26-cpu-frontier-sj80zdzj`; no old-task prediction data or cache reused. Runtime elapsed about28minutes, including fresh model training.

Frozen H200 profiles were collected in this current-main task. Backend collective_sim/htsim with nvlink_analytic; Qwen3 TP4/DP2/PP1/EP8;4096/1024; uniform routing; eager; prefixOFF; chunked-prefillOFF; CPUoverheadOFF. Groundtruth is fresh h200-historical-replay-02, with300drainedwarmups and100formalrequests. This step required no new GPU allocation.

Exact commands (completed output directories are exclusive; use fresh paths when repeating):

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=ERROR /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_cpu_frontier_worker.py --config task_memory/task_2026-09-07_issue26_ttft_h200/config/frontier_dp_snapshot_candidate.json --output /data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_frontier_baseline_analysis.py --metrics /data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime/metrics/qwen3_a3b_30b_moe/online_serving/runtime --mapping task_memory/task_2026-09-07_issue26_ttft_h200/analysis/historical-replay-02-clean/request_id_map.csv --groundtruth task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-historical-replay-02/clean/runtime/clean/server.request_metrics.jsonl --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/shared-forward-numerical-01
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_prefill_batch_comparison.py --ledger /data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime/metrics/qwen3_a3b_30b_moe/online_serving/runtime/frontier_stage_batch_ledger.jsonl --mapping task_memory/task_2026-09-07_issue26_ttft_h200/analysis/historical-replay-02-clean/request_id_map.csv --diagnostic task_memory/task_2026-09-07_issue26_ttft_h200/analysis/historical-replay-02-batch-validation.json --clients task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-historical-replay-02/batch/runtime/batch/client.jsonl --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/cpu-shared-forward-01/batch_comparison.json
```

## Criteria and functional evidence

Detect the original stalled event queue, missing/duplicate requests, changed lengths, double first-token credit, and malformed timing/identity joins. PASS requires process exit0,100unique request completions,100matching arrivals, exactly4096/1024lengths, endpoint/CSV consistency, and per-request model calls of4096once plus1exactly1023times. The existing final-prefill first-token credit yields1024generated tokens, not1025.

**PASS**:100arrivals+100unique completions; token call invariant holds for every request;3662stage batches (DP0:1823,DP1:1839);98prefill-containing batches covering100prefill requests; final completion104.30949378135755s. No simulation error. Selected evidence: analysis/cpu-shared-forward-01/token_validation.json, metrics_ground_truth.jsonl, request_metrics.csv, prefill_stage_ledger.jsonl. Original full ledger remains at `/data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime/metrics/qwen3_a3b_30b_moe/online_serving/runtime/frontier_stage_batch_ledger.jsonl`.

Token invariant can be reproduced with the same Python:

```python
import collections, json
from pathlib import Path
counts = collections.defaultdict(collections.Counter)
for row in map(json.loads, Path("/data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime/metrics/qwen3_a3b_30b_moe/online_serving/runtime/frontier_stage_batch_ledger.jsonl").open()):
    for rid, tokens in zip(row["request_ids"], row["request_num_tokens"]):
        counts[rid][tokens] += 1
assert dict(counts) == {str(i): {4096: 1, 1: 1023} for i in range(100)}
```

Focused regression and original trained3request evidence are in test_report_2026-09-08_shared_forward_sync.md:73PASS plus3/3full4096/1024completion. Full run now verifies final production source, including the final duplicate-ownership assertion.

## Numerical comparison

| Metric | Frontier | vLLM | Absolute error | Relative error | Numerical threshold |
| --- | ---: | ---: | ---: | ---: | --- |
| ttft (ms) | 105.443668418 | 131.268637180 | 25.824968762 | 19.673373% | FAIL |
| tpot (ms) | 56.845392316 | 79.241113351 | 22.395721035 | 28.262754% | FAIL |
| request_e2e (ms) | 58258.280007628 | 81194.852468967 | 22936.572461339 | 28.248801% | FAIL |
| request_throughput (requests/s) | 0.958685508 | 0.782723735 | 0.175961773 | 22.480700% | FAIL |
| token_throughput (tokens/s) | 4908.469799242 | 4007.545522014 | 900.924277228 | 22.480700% | FAIL |
| decode_throughput (tokens/s) | 981.693959848 | 801.509104403 | 180.184855446 | 22.480700% | FAIL |

All latency statistics are arithmetic means over100eligible formal rows. Warmup300excluded. Each rate uses its own last-formal-completion minus first-formal-arrival window: Frontier104.30949378135755s, vLLM127.7589979171753s. Numerators are100requests,512000total tokens,102400decode tokens. No missing or duplicate IDs; counts, denominators and exact paths are in analysis/shared-forward-numerical-01/runtime_comparison.json and e2e_metrics_table.csv. Each table threshold is10percent.

**D006 is not closed**: TTFT compares the current unadjusted Frontier queue-to-prefill endpoint with official vLLM server first-token latency. The25.824968762ms difference is not an independently measured CPU correction. Secondary metrics also remain outside10percent. Native vLLM TPOT is retained because its token timestamp boundary differs slightly from E2E's output-record timestamp; see the focused report for the failed reconstruction assertion and source-supported explanation.

## Batch evidence and limits

**COMPLETE comparison; equality fails globally**:78/100sameDP,5/100samecomplete member/token vectors. Client requests0–4 have matching member vectors. First formal request is DP0, sole request0/4096 on both sides. First divergence in clean engine-queue order is client request6: Frontier DP1 with decode requests2/3 and prefill6; vLLM diagnostic DP0 with decode requests0/1/4 and prefill6. Clean request mapping correctly distinguishes client6fromFrontier5 and client5fromFrontier6.

The diagnostic runs separately from clean. Its batch IDs are local toDP; missing dummy/global-round records prevent exact global collective joins. Even matching members do not prove matching KV progress: request1's mixed batch contains request0 after one prior decode model step in Frontier, versus zero in the diagnostic. This does not invalidate the shared source-preservation fix; it limits cross-run operator comparability.

## Newly observed next-stage evidence gap

Pinned vLLM core_client.py choosesDP before sending to EngineCore; scheduler.py records QUEUED only after receipt. Frontier currently routes requests using the supplied queue-arrival trace. Clean request5arrives at the server2.606391907ms before request6, but request6enters the engine queue0.200871844ms before request5. Actual routing order and routing-time load snapshots are missing. This establishes distinct observation boundaries, not a quantified cause for all22placement mismatches or the25.825msTTFT gap. Proposed minimal diagnostic extension is analysis/dp_routing_endpoint_proposal.md; simulator timing changes await that evidence and the applicable scope decision.

## Execution issues and disposition

No full-run execution failures. Creating the new run-manifest directory under existing runs/ failed PermissionError13; evidence was saved under writable analysis/cpu-shared-forward-01 without changing permissions. Guessed filenames during source inspection were corrected with rg; no runtime code was changed to conceal read failures. The original mixed-phase failure and initial regression errors are retained in earlier reports.

First-batch limit: fresh Frontier first-stage65.790076904ms versus isolated vLLM CUDA batch span77.125022888ms (14.696846percent diagnostic gap); official clean firstTTFT121.484279633ms. Later DP placement divergence cannot alone explain this first-request difference. The route diagnostic is proposed to qualify full-case workflow, not claimed to close all latency residual. Evidence: analysis/cpu-shared-forward-01/first_batch_timing_comparison.json.
