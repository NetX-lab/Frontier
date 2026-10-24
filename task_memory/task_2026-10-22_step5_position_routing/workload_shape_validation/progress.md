# D29 Required Wave Shape Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Record3/3 failed attempts, completed static finding, corrected checker-only5/5 controls and no authorized retry. |
| 2026-10-24 | Recover cancelled-session commands; primary owns the unchanged bounded shape scope. |

## Status

- completed/cancellation census: complete OpenCode v2 assistant-message cursor
  traversal recovered35/28/23 messages for shape/copy/init respectively. The
  histories contain38/32/21 shell calls, exclusively inspections, and no patch
  or production-wave invocation. All three end with `Step interrupted`. This
  is not evidence of three failed shape runs. Source:
  `/data/ycfeng/tmp/d29_cancelled_probe_receipts/execution_history.json` and its
  page manifests. Optional `executed=false` is not used to infer non-execution;
  command inputs and result contents settle the invocation census.
- completed/artifact inspection: none of the three assigned records directories
  existed in retained worktrees; no new shape/copy/init scratch output was found.
  The cancellation census confirms no new artifact-write command.
- completed/metadata and syntax preflight: `preflight_01.json` in
  `/data/ycfeng/tmp/d29_required_wave_shapes_20261024` verifies exact clean
  dbe19f53,23 source hashes and262 unchanged fixture size/mtime bindings without
  running a production method. Driver/parameter hashes are recorded there.
  Cached adapter bffe2503 remains unchanged. Cases use actual Batch constructors,
  forward binding, aggregate preparation and routing/materialization functions.
- failed/three shape attempts: P6128,D16,BF16 each exit1 after pricing when
  primary's checker treats unselected add_attn_residual=0.0 as missing evidence.
  P6128/D16 complete288numerical/144proof/64collective/16point calls;BF16layer88
  completes256/128/64/16. No complete emissions are verified;89/90 are not reached.
- completed/static finding: D29-SHAPE-DRIVER-PREFLIGHT,session
  `ses_e9217fb19ffe995qZCog0dwXN5`, independently identifies the same checker
  boundary error. Its result arrived after the three runs;it did not prevent them.
- completed/checker correction: source code is unchanged;task-local guard skips
  only unselected0.0 slots and still rejects selected zero,negative andNaN prices.
  Corrected driverSHA256e099270ef4c8bfdf36336d4c4d494d8ddecdafbc57003efabb07762f416a1d9e.
  Five AST-extracted checker controlsPASS,with0 production invocations.
- completed/evidence archive:16 exact artifacts in evidence/raw_probe_failures/,
  plus the failed driver in evidence/failed_run_case.py. Post-failure23source/
  262fixture bindingsPASS. See test_report_2026-10-24_required_shapes.md.
- blocked/further shape qualification:3/3 used,0remaining. No fourth invocation
  is authorized. P6128/D16/BF16 remainUNVERIFIED pending an explicit quota decision.

## Invocation Ledger

| Category | Previously used | New used | Limit | State |
| --- | ---: | ---: | ---: | --- |
| All production-path shape invocations | 0 | 3 | 3 | Exhausted;0remaining,no retry |

## Inspection Error and Resolution

Capturing a large `opencode-v2 api get .../message` response through subprocess
stdoutPIPE returned exit0 but raised `JSONDecodeError: Unterminated string` at
column191094. Redirecting the same read-only API request to a private regular
file returned valid346499-byte JSON, SHA256
`3594bceaf55ef2b2c40446f021dd22113ec1be03a779627b94115a994bb755d8`.
The failed tool output is retained. Full cursor traversal then succeeded using
regular-file stdout. No service/config/credential/database change was made;
this observation does not establish the CLI's internal truncation mechanism.

## Independent Work

- D29-SNAPSHOT-COPY-FINISH: session `ses_e92340744ffe58Yx5C7xx4qApF`, records
  worktree `/data/ycfeng/Frontier/.worktrees/d29-snapshot-copy-finish`; source
  inspection and at most one minimal CPU contract control, no full-wave rerun.
- D29-INIT-PREPARATION-FINISH: session `ses_e92338e41ffe081mlmDNjpmxP5`, records
  worktree `/data/ycfeng/Frontier/.worktrees/d29-init-preparation-finish`; source
  and retained small evidence only, no new initialization or benchmark.
