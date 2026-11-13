# D29 main08 clock alignment verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Parent verifies launch/scheduler origins and corrects the unsupported slowdown attribution. |

## Expectation and method

Use the retained offline command, execution receipt and52scheduler records.
`execution.wall_s` includes initialization; scheduler timestamps are absolute
epoch seconds. Compare matching boundaries before inferring a phase duration.
No full trace parse, new simulator run or source change is required.

The following command ran in the primary worktree under frontier-py310,
Python3.10.6 (GCC11.3.0):

```bash
/data/ycfeng/envs/frontier-py310/bin/python - <<'PY'
import json,datetime
from pathlib import Path
p=Path('/data/ycfeng/tmp/d29-parent-position-main08/offline')
r=[json.loads(s) for s in (p/'scheduler.jsonl').read_text().splitlines()]
e=json.loads((p/'execution.json').read_text());end=(p/'execution.json').stat().st_mtime;start=end-e['wall_s']
f=r[0]['timestamp'];i=next(x['timestamp'] for x in r if x['event']=='iteration_start' and x['cluster_type']=='DECODE' and x['iteration_id']==4 and x['num_running_reqs']==1)
print(json.dumps({'command_bash_mtime':(p/'command.bash').stat().st_mtime,'inferred_start_from_execution_mtime':start,'execution_mtime':end,'first_scheduler_epoch':f,'pre_scheduler_s_approx':f-start,'iteration4_epoch':i,'iteration4_since_first_scheduler_s':i-f,'iteration4_remaining_to_execution_mtime_s':end-i,'apparent_s_per_written_wave_58':(end-i)/58,'first_scheduler_local':datetime.datetime.fromtimestamp(f).isoformat(),'end_local':datetime.datetime.fromtimestamp(end).isoformat()},indent=2))
PY
```

## Observed result

PASS: command mtime1792751317.662415 and inferred launch1792751317.666937
agree within approximately4.5ms. First scheduler event1792752430.011742 is
1112.344805s later. Active iteration4 starts1792753515.053171; execution receipt
mtime1792753717.842189 leaves202.789017s, not1315s. The old subtraction mixed
process-relative and scheduler-relative intervals and is invalid.

The58persisted MoE layers give approximately3.496s per persisted wave over the
remaining interval; up to24waves may still be buffered. No anomalous slowdown
or I/O stall is established. File mtimes approximate launch/receipt boundaries;
they are not additional native telemetry or a precise model-manager timer.

Also corrected: byte-size division yields representative row equivalents, not
an exact event census; distinct lane/operator event identities do not rule out
equal immutable payload leaves. Source-bound90200waves and representative
262.1GB/mode remain projections/count derivations, not a completed workflow.
