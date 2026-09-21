## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Completed H800 standard replay reproduction, full client gates and first-formal comparison. |

# H800 standard replay reproduction

## Test script information

Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
Job `yc26-h800-standard-replay-176000-repro-20260910-01`, node `gpu-h800-0496.host.platform.shaipower.com`, namespace shai-core. Platform terminal phase **Succeeded**.

Exact submitted launch: `bash /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-replay-176000-repro-01/launch.sh` (already completed; preserve job/output identity).
Script chain (all absolute paths rooted at `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/`):
`issue26_h800_replay_worker.sh` -> `issue26_h800_uniform_groundtruth_worker.sh` -> `issue26_h800_official_ttft_worker.sh`, followed by `issue26_h800_diagnostics_worker.sh batch`; both server modes use `issue26_token_id_client.py`.

Worker environment: conda vllm-bs-0.10.2, Python3.10.16, Torch2.8.0+cu128, FlashInfer0.3.0. CPU validator /usr/bin/python3.12.3. Source commit0d633a946e6600c77a251bef8b5553ec7f43f7e7, fixed image b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc.

H800 codesign/h800,8GPU/64CPU/400GiB,176000KV blocks of16,TP4/DP2/EP8/PP1,BF16,eager,FLASHINFER,uniform routing,prefix caching OFF,chunked prefill OFF. Prompt4096/output1024,QPS2,seed20260908. Each mode runs3x100 drained warmup requests then100formal requests.

Exact executed standard validator:
```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
python tests/e2e/issue26_diagnostic_identity_analysis.py --run /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-replay-176000-repro-01/batch/runtime/batch --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-replay-176000-repro-01/identity_validation.json --mode batch > /data/ycfeng/tmp/issue26-h800-standard-repro-identity.log
```
Validator output uses exclusive creation; repeat with a fresh output filename. Reproduce the rank statistics from its complete selected records:
```python
import json, statistics
from pathlib import Path
run = Path('/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-replay-176000-repro-01')
v = json.loads((run / 'identity_validation.json').read_text())
rows = [r for w in v['workers'] if w['identity'][0] == 0
        for r in w['formal_prefill_batches']
        if r['request_ids'] == ['cmpl-pf4096_dc1024:0-0']]
assert len(rows) == 4
assert all(r['batch_size'] == 1 and r['request_num_tokens'] == [4096]
           and r['batch_num_prefill_tokens'] == 4096
           and r['batch_num_decode_tokens'] == 0 for r in rows)
a = sorted(r['batch_execution_time_ms'] for r in rows)
print(statistics.median(a), .3*a[2]+.7*a[3], max(a), max(a)-min(a))
```

## Validation criteria and results

PASS: standard identity validator across8workers;100formal requests each have one complete4096-token prefill on four same-DP TP ranks. PASS: both clean and batch client artifacts have400unique expected IDs,300warmup/100formal,4096prompt/1024completion tokens each. PASS: four client phases have100completions each; next phase starts after preceding phase ends and previous client completion precedes next phase arrival. PASS:24direct uniform-router checks. PASS: runtime server args equal the historical H800 run; recorded VLLM environment equal apart from output paths; same source commit. Both batch manifests omit boundary-log and AR-diagnostic overrides.

Completion marker `HISTORICAL_CONTROL_REPLAY_EXECUTION_COMPLETE` in `/data/ycfeng/tmp/issue26-h800-standard-repro-launch.log`. Logs include `DIAGNOSTIC_EXECUTION_COMPLETE selection=batch`.

First formal identity `cmpl-pf4096_dc1024:0-0`,DP0,batch4711,batch_size1,request_num_tokens[4096],prefill4096,decode0,DPtoken counts[4096,1],op_profile_selected=false. Historical H800 comparison batch4696, same predicates.

## Numerical evidence

| TP rank | Historical H800 ms | Fresh H800 ms |
| --- | ---: | ---: |
| TP0 | 80.640991211 | 77.486846924 |
| TP1 | 80.687454224 | 77.460639954 |
| TP2 | 80.668769836 | 77.482528687 |
| TP3 | 80.596412659 | 77.484802246 |

| Metric | Historical H800 ms | Fresh H800 ms | Delta ms |
| --- | ---: | ---: | ---: |
| Rank median | 80.654880524 | 77.483665466 | -3.171215057 |
| Rank P90 | 80.681848907 | 77.486233521 | -3.195615387 |
| Rank max | 80.687454224 | 77.486846924 | -3.200607300 |
| Rank spread | 0.091041565 | 0.026206970 | -0.064834595 |

Verdict: **PASS for reproduction of the approximately80ms H800 standard first-forward scale**. Median differs by-3.931833%,rank max by-3.966673%. Rank spread remains sub0.1ms. These are an independent replay's observed values, not an exact numerical replay or repeated-run confidence interval. Rank P90 is calculated over four TP ranks; rank max is the maximum rank-local outer event span.

## Failures, warnings and limits

Batch shutdown contains TCPStore Broken pipe / NCCL should-dump-flag warnings after all formal client completions. Cleanup terminates the API server and its workers; the warning reports that the store is unavailable during teardown. Worker completion markers and platform Succeeded confirm successful job completion. Clean shutdown also reports three leaked shared-memory objects. These warnings are retained; request completion/identity validation passes.

The event span can include host-induced device gaps. Batch instrumentation and scheduler logs remain enabled; this is the standard batch-only forward boundary, not clean client/server TTFT. Prior statement that no valid H80070-110ms record existed is corrected: historical raw H800 standard batch4696 contains80.596-80.687ms, and this fresh run reproduces77.461-77.487ms.

The500ms AR diagnostic used different logging and workload conditions. This reproduction does not assign its excess to a single cause. No H200 timing is included. CUDA attribution, normal/skip paired causal analysis and final Frontier accuracy remain separate unfinished work.

Verified standard replay scripts and warmup identity handling committed as `07dbbdeb`. Four shell syntax checks and the real400-row batch validator PASS. Only the two pre-existing D005 scripts remain untracked.
