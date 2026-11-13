# Legacy measured replay cancellation audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Primary inspected saved results after the legacy agent cancellation. |
| 2026-10-22 | Parent exit observer completed; archive both terminal modes and verify retained online artifacts. |

## Scope and result

The cancelled agent is `ses_ea1aa93d6ffeCP9B6Fa1DHGcYM`, description
`Finish full measured replay`. Cancellation is an agent lifecycle event;
it is not an observed application failure or successful completion of both modes.

Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d20-integration`, HEAD
`acc3a88`; saved integration changes remain uncommitted there. The three-file
diff and hashes are archived without modifying that worktree. These are audit-time
source snapshots, not reconstructed process-start provenance.

Existing run: `/data/ycfeng/tmp/d20_actual_main_request_free_h80032_09`.
Offline execution exits0 after21040.243170 seconds. Primary checked its JSONs:

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Request |6144 input,1024 output|6144/1024|PASS|
| Initial D work |16 local,6128 cached|16/6128|PASS|
| Useful D forwards |1024|1024|PASS|
| READ completion |ready=release=end|67.17311383299996s for all three|PASS|
| Final metrics |finite and positive|all recorded final metric values satisfy this|PASS|

The reported synthetic READ cost is3.5ms. The fixture has the superseded
metadata-free dummy identity and8192 max-model length. The original
`native_admitted=false` remains unchanged. This is historical CPU execution
evidence, not acceptance of the current MRv2 fixture or native accuracy.

Online had no execution/validation receipt at audit time. One process inspection
(`ps -p 94827,849385 -o pid,ppid,stat,etime,time,rss,pcpu`) observed parent94827
and child849385 still running; the child used one CPU and about1.12GiB RSS.
No process was stopped or relaunched. Both-mode completion was unverified at
that audit time; the terminal audit below closes this historical result.

## Reproducibility

`receipt.json` records exact source/artifact hashes and the assertions checked by
primary Python3 JSON inspection. `offline_validation.json` and
`offline_execution.json` preserve original bytes; `saved_integration_changes.diff`
preserves the current three integration-file changes. Inspect the original run
for raw logs and larger metrics; no large trace was loaded for this audit.

Continuation at the first audit: truthful MRv2 D23 actual-main verification in
`/data/ycfeng/Frontier/.worktrees/step5-mrv2-main`, using the already-fitted
`/data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json`. The newer owner
session is `ses_ea08efc2cffe6Z1Rc7srkVdebh`.

## Terminal audit after the parent-exit notification

The existing pidfd observer reported that parent94827 exited. Both original
execution receipts report exit0 and both original validators report PASS.
Offline wall time is21040.243170s (5.8445h); online is20634.534379s (5.7318h).
The parent checked the retained summary against both validation JSONs, the raw
request and decode-batch CSVs, system metrics and the transfer ledger.

| Check | Offline | Online |
| --- | --- | --- |
| Completed requests / request shape |1 /6144+1024|1 /6144+1024|
| Useful D forwards |1024|1024|
| First D batch / following widths |16 /1023 one-token batches|16 /1023 one-token batches|
| Cached prefix |6128|6128|
| Reported READ end = readiness = source release |67.17311383299996s|67.42311383299996s|
| READ completed/bound; duration |PASS;3.5ms|PASS;3.5ms|
| Client TTFT, TPOT and E2E |finite, positive|finite, positive|

These are historical **superseded metadata-free CPU fixture** results. The
8192 capacity and `native_admitted=false` are retained. They do not establish
D23 MRv2 execution, current performance acceptance or native accuracy.

Command environment: standard-library `python3`, run with `timeout 60s` from
`/data/ycfeng/Frontier/.worktrees/step5-preview`. The core checks can be reproduced
without reading the large trace:

```python
import csv, json, math
from pathlib import Path
root = Path('/data/ycfeng/tmp/d20_actual_main_request_free_h80032_09')
for mode in ('offline', 'online'):
    case = root / mode
    value = json.loads((case / 'validation.json').read_text())
    execution = json.loads((case / 'execution.json').read_text())
    assert execution['returncode'] == 0 and value['status'] == 'PASS'
    run = Path(value['run_dir'])
    with (run / 'request_metrics.csv').open() as stream:
        [request] = list(csv.DictReader(stream))
    assert request == value['request']
    assert (int(request['request_num_prefill_tokens']), int(request['request_num_decode_tokens'])) == (6144, 1024)
    with (run / 'decode_batch_metrics.csv').open() as stream:
        batches = [row for row in csv.DictReader(stream) if int(row['batch_size']) > 0]
    assert len(batches) == value['decode_batch_count'] == 1024
    assert int(batches[0]['batch_num_tokens']) == 16
    assert all(int(row['batch_num_tokens']) == 1 for row in batches[1:])
    assert value['cached_prefix_tokens'] == 6128
    [transfer] = [json.loads(line) for line in (run / 'frontier_transfer_ledger.jsonl').read_text().splitlines()]
    assert transfer == value['transfer'] and transfer['status'] == 'completed' and transfer['target_bound']
    assert abs(transfer['duration_ms'] - 3.5) < 1e-6
    assert value['source_release_time'] == value['target_ready_time'] == value['completed_read_event']['time'] == transfer['end_ts_s']
    assert all(math.isfinite(float(request[key])) and float(request[key]) > 0 for key in ('client_ttft', 'tpot', 'request_e2e_time'))
    system = json.loads((run / 'system_metrics.json').read_text())
    assert system == value['system_metrics']
    assert system['simulation_metadata']['completed_requests'] == system['simulation_metadata']['total_requests'] == 1
```

`terminal_receipt.json` stores selected observations, timestamps and SHA256 of
the inspected artifacts. `online_execution.json` and `online_validation.json`
are byte-preserving copies. The original audit receipt and offline copies are
preserved. The large event trace was not reloaded: readiness/release values were
cross-checked between retained validation and the transfer ledger. Original
process-start source provenance was not reconstructed.

Current acceptance remains actual06, using fixture02 and immutable7e8bea3 in
`fidelity-candidate-7e8bea3`, with its original2400s bound. See
`../../../task_2026-10-22_step5_mrv2_main/progress.md` for the active commands.
