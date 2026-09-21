## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded full and minimized predictor-timed failure, negative controls, and sync-entry isolation. |

# Predictor-timed forward progress diagnosis

## Execution

CPU conda dev-vidur-v03-hopper-e2e, Python3.13.13. Root: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`, source a4a0c496. Exact original simulation command is in analysis/dp_snapshot_numerical_02_run_manifest.json. Diagnostic commands are in analysis/mixed-phase-stall-evidence/*_command.json. Observer: `/data/ycfeng/tmp/issue26-h200-network/observe_deadlock.py`, preserved as analysis/mixed-phase-stall-evidence/observer_source.py.txt.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
export FRONTIER_LOG_LEVEL=ERROR PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH="$PWD"
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python /data/ycfeng/tmp/issue26-h200-network/observe_deadlock.py /data/ycfeng/tmp/issue26-dp-deadlock-repro-03req
```

The observer captures state at the existing simulator failure-report boundary and changes no scheduling. It uses this same source revision's newly trained predictor caches only for diagnosis. Diagnostic artifacts cannot replace a fresh repaired numerical run.

## Criteria and actual results

| Check | Expected correct behavior | Actual | Result |
| --- | --- | --- | --- |
| Full100requests,4096/1024 |100complete, scheduler empty |exit1,event queue empty with unfinished requests |FAIL |
| Same-command diagnostic |Same verdict as original |Same stall;1157recorded events |REPRODUCED |
| First3requests,4096/1024 |3complete |Same stall at0.29563780122719624s |FAIL |
| First2requests,4096/1024 |2complete |2complete,exit0 |PASS negative control |
| First3requests,4096/8 |3complete |Same stall |FAIL minimized control |
| First2requests,4096/8 |2complete |2complete,exit0 |PASS minimized control |
| Same-group prefill/prefill entry |1shared wave |1wave,0open rooms |PASS |
| Same-group decode/decode entry |1shared wave |1wave,0open rooms |PASS |
| Same-group decode/prefill entry |1shared wave |0waves,2open rooms |FAIL |

Precise failure: `Sequential simulation ended with non-empty scheduler state`. Both real lane batches have provisional cohort5; decode room240 owns lane0 requests0/1, prefill room241 owns lane1 request2. The last collective-entry events occur at0.2954594669165676s and0.29563780122719624s. No combined EP wave follows. Full-run final47.156283407937735s is merely the last external arrival.

Raw control outcomes: analysis/mixed_phase_sync_control.json. Persistent negative-control request metrics and failed-state traces are under analysis/mixed-phase-stall-evidence/.

## Limits and next action

No complete Frontier mean TTFT, absolute error, or relative error exists for the repaired policy run. Clean vLLM reference is independently valid at131.26863718032837ms mean official TTFT. First-prefill membership matches, but whole-run membership comparison awaits a completing simulator. The existing new tests/e2e/issue26_prefill_batch_comparison.py is an uncommitted analysis draft pending that real run; it has not been claimed validated.

Repair proposal and source-level causality: analysis/mixed_phase_forward_stall_rca.md. Shared-protocol implementation remains pending YC's D017 decision. GPU replay and CPU checks have exited; no ongoing allocation or numerical job is needed for this pending decision.
