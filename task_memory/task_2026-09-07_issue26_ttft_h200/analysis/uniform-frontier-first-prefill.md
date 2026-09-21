## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Validated bounded first-prefill accounting for uniform Frontier01. |

# First prefill: Frontier internal accounting

PASS_FRONTIER_FIRST_PREFILL_ACCOUNTING. This is request0, iteration0, replica0, stage0, schedule_epoch1, all48layers,4096real prefill tokens. Each layer has128experts with256assignments/expert,32768assignments in total. It is solely new Frontier accounting; no vLLM operator durations or CPU overhead estimates enter this result.

| Component | 48-layer total (ms) | Counting rule |
| --- | ---: | --- |
| Attention compute | 19.300608000 | One ATTENTION/TOTAL per layer |
| Attention TP communication | 17.899440000 | One attn_tp_allreduce trace per layer |
| EP pre-dispatch | 8.703446196 | Maximum EP-lane pre-dispatch per wave |
| EP dispatch | 2.125341867 | Maximum collective duration per wave |
| EP routed compute | 15.635896153 | Maximum EP-lane routed compute per wave |
| EP combine | 2.125341867 | Maximum collective duration per wave |
| EP post-combine | 0 | Maximum post-combine per wave |
| Component sum | **65.790074082** | Attention plus one critical-path EP wave per layer |

The last EP-WAVE-END timestamp is **65.790076904ms**. Component sum differs by **-0.000002822ms**, consistent with attention/TP traces rounded to6decimal places. Barrier and wave traces retain12decimal places. Each layer's gap from the preceding wave endpoint equals its attention compute+TP interval within0.0000011ms. The EP wave total is28.590026082ms.

## Identity and duplicate control

Read only the first **1,990,858bytes /10,266lines** of `runs/cpu-uniform-frontier-01/runtime/frontier.log`, with an8MiB hard cap, and stopped at `[TOKEN-ROLLOUT] req=0 ... decode_progress=1/1024 ... (first decode token after prefill)`. No growing-log full scan was performed. Attention batch_id0 has two identical ATTENTION/TOTAL and two identical attention-TP trace occurrences per layer; each pair is counted once. Child attention OP-TRACE records are not added to TOTAL. EP-WORKLOAD ephemeral lane prediction batch IDs are not mistaken for the attention batch.

All384EP-WORKLOAD rows bind request_ids=[0],iteration_ids=[0],request_runtime_epochs=[0],replica0/stage0/schedule_epoch1. Each wave has one operation_id and all8EP lanes, with exact layers0–47. The96barriers verify expected/arrived EP IDs and max_lane+collective conservation;48wave ends verify dispatch+combine+post conservation. Full selected lane/barrier/wave records and source line numbers are retained in `uniform-frontier-first-prefill.json`.

Source definitions checked: `sklearn_execution_time_predictor.py:7517-7531` sums attention compute only; `sklearn_moe_execution_time_predictor.py:3314-3321` emits attention TP communication; `entities/execution_time.py:1006-1015` adds them. `scheduler/utils/expert_parallel.py:197-237` uses a separate maximum for pre-dispatch, dispatch, routed compute, combine and post-combine. Thus EP8 times must not be summed as eight sequential participants. `prefill_collective.py:109-143` and `ep_wave_schedule.py:148-170` append attention/wave components to the actual prefill ledger.

## Endpoint limitation

The bounded metrics inspection found request0 arrival_at=0, but **no request_completion record yet**. The final request TTFT endpoint comparison remains pending;65.790076904ms is the observed last EP wave endpoint, not a substituted completion metric. `collective_timing.py:60-94` allows pipeline/CPU time after the wave, so completion must be checked from its eventual request artifact even though this case hasPP1 and CPU modeling disabled.

Execution environment: conda `dev-vidur-v03-hopper-e2e`, `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, Python3.13.13. Inline parsing used only Python standard-library ast/re/json/math and hard-bounded file iteration. A256KiB cap was applied separately to metrics_ground_truth.jsonl. Validation criteria: exact identities/48layers/EP8load conservation, unique-equal attention traces, phase maxima once per wave, finite durations, barrier arithmetic and timeline consistency. All passed; full-request endpoint availability is separately pending.

Recompute retained component totals and endpoint difference from the audit artifact:

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'CHECK'
import json, math
from pathlib import Path
p=Path('task_memory/task_2026-09-07_issue26_ttft_h200/analysis/uniform-frontier-first-prefill.json')
a=json.loads(p.read_text())
assert [r['layer_id'] for r in a['rows']]==list(range(48))
assert all(r['request_ids']==[0] and r['iteration_ids']==[0] for r in a['rows'])
for name in a['rows'][0]['components_ms']:
    print(name, math.fsum(row['components_ms'][name] for row in a['rows']))
total=math.fsum(row['component_sum_ms'] for row in a['rows'])
print('sum_ms',total,'last_wave_ms',a['rows'][-1]['wave_end_time_s']*1000)
CHECK
```

## Completed request endpoint verification

The current run subsequently emitted request0 completion with all1024decode tokens. Its TTFT is65.79007690374297ms, equal to the last EP wave65.790076904ms within2.5704e-10ms. The48-layer component sum is lower by only2.822e-6ms from logged decimal rounding. The earlier endpoint-availability limitation is now closed for request0; whole-run validation remains separate. Receipt: uniform_frontier_first_prefill_endpoint.json.
