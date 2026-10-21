# Legacy measured replay cancellation audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Primary inspected saved results after the legacy agent cancellation. |

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
No process was stopped or relaunched. Both-mode completion remains unverified.

## Reproducibility

`receipt.json` records exact source/artifact hashes and the assertions checked by
primary Python3 JSON inspection. `offline_validation.json` and
`offline_execution.json` preserve original bytes; `saved_integration_changes.diff`
preserves the current three integration-file changes. Inspect the original run
for raw logs and larger metrics; no large trace was loaded for this audit.

Current continuation: truthful MRv2 D23 actual-main verification in
`/data/ycfeng/Frontier/.worktrees/step5-mrv2-main`, using the already-fitted
`/data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json`. The newer owner
session is `ses_ea08efc2cffe6Z1Rc7srkVdebh`.
