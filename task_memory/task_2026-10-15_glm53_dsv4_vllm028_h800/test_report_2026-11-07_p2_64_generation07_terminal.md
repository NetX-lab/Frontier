# Test report: P2-64 generation07 terminal cause (2026-11-07)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-07 | Created: generation07 mid-run reads, terminal evidence, cause classification and GPU-hour accounting; D-P2-64-FIFTH exhausted. |

## Scope

The last step under D-P2-64-FIFTH (requirements.md), continuing
[the generation06/07 report](test_report_2026-11-07_p2_64_generation06_terminal_generation07_submission.md):
establish why generation07 (`exp-1107-150132-396145`, platform-cause resubmission 2 of 2) ended, and count its GPU
hours. It covers no native result and no metric comparison: the job produced no native reference.

## Environment

- Pi on HOST, model `custom/deepseek-v4.1-flash-sys`, thinking `high`, through
  `bash /data/ycfeng/codex-home/skills/ds4-exec/scripts/run_ds4.sh <label> /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 <prompt> read-only`.
- Platform access: personal i-fengyicheng StepMind Python RJobBackend, `/data/ycfeng/tmp/stepmind-env/bin/python`
  (Python 3.10), environment from `h200_mainline/stepmind_env.sh`.
- Cloud reads: `ssh -o BatchMode=yes shai-ycfeng 'sudo -n ...' < <lead script>`, read-only, under the one run root
  `/mnt/ws-jfs/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h200_p2_64_formal_20261107_07/case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261107-07`
  (the same directory as `/mnt/codesign-exp/ycfeng/...` on the workers).
- Lead audit and waits: system `python3` 3.10.6.

| Label | Turns | Purpose | Prompt |
| --- | --- | --- | --- |
| `p64-formal7-inspect-1107o` | 2 | mid-run reads at 07:46Z and 07:48Z: events and audit, live receipt, node records | `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal7-inspect-1107o.md`, `.2.md` |
| `p64-formal7-terminal-1107p` | 1 | terminal receipt, events and audit, replica logs, summary of the native run tree | `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal7-terminal-1107p.md` |

Each turn used one tool call on a lead-written script, so I-010 did not interrupt them. The prepared post-run label
`p64-formal7-post-1107n` (copy on success) was not launched, because the job did not succeed.

## Observed sequence

- 07:01:32Z create (the only user operation); queue wait for H200 quota; 07:17:41Z all 8 replicas created.
- 07:23Z image pulls; 07:25:35-07:26:35Z containers start; all 8 nodes write `ready.json`, so the rendezvous of 8 passes.
- gpu-h200-0726 (replica cfcd2084) writes its rendezvous record at 07:26:35Z and nothing after: no plan, no
  `server.log`, no `failed.json`. The other 7 nodes planned it as decode lane 2.
- Monitor reads: all Running at 07:26:50Z; cfcd2084 `Stopped` at 07:31:52Z. Its container state stays at the last
  reported `running`; it has no stop operation in the audit and no event after the image pull.
- Prefill head gpu-h200-0023 is healthy at 07:36:37Z (`server_ready.json`); the 4 prefill nodes wait for replay.
- The decode DP head gpu-h200-0300 never becomes healthy: its server waits for the missing decode lane 2. No file in
  the run tree changes after 07:36:40Z (read at 07:48Z).
- 07:56:45Z the decode head raises its 1800 s startup timeout
  (`TimeoutError: Native P/D rendezvous or workload deadline exceeded`); decode lane 1 sees its headless server exit 0;
  decode lane 3 and prefill lane 2 see the peer failure. Two containers exit 1 (1679091c on 1007, eccbc87e on 0300).
- 07:56:47Z job Failed, `Task fail policy [any] caused undesired running replicas.`; rjob-controller stops the rest;
  the launcher exits at 07:56:50Z (RuntimeError, exit 1). No replay started.

## Lead audit

Command: `PYTHONDONTWRITEBYTECODE=1 timeout 120 python3 h200_mainline/p2_64_formal_retry07/terminal_20261107/lead_terminal_audit.py`.
It copies 35 evidence files with sha256 (`archive_manifest.json`) and writes `lead_terminal_acceptance.json` =
`PASS_TERMINAL_PLATFORM_CAUSE_NO_REFERENCE`, 17/17 checks.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Job identity and stop reason | i-fengyicheng; Failed by the any-replica fail policy | as expected, 07:56:47Z | PASS |
| User operations in the audit | only the create | 1 row: create at 07:01:32Z | PASS |
| Stopped replica | cfcd2084 on gpu-h200-0726, phase Stopped, container state still `running` | as expected | PASS |
| Stop operation on cfcd2084 | none | audit: create only | PASS |
| Events of cfcd2084 | nothing after the image pull | FailedScheduling (co-scheduling wait), Pulling | PASS |
| Node record of gpu-h200-0726 | ends at rendezvous, no native error | stage `rendezvous`; files ready.json, result.json, worker.log only | PASS |
| Missing lane | the node the peers planned as decode lane 2 | decode lanes present 0, 1, 3 | PASS |
| Stop time | between two monitor reads | after 07:26:50Z, by 07:31:52Z | PASS |
| Decode head failure | startup timeout | `TimeoutError` at the healthy wait | PASS |
| Prefill head | reached server_ready | `server_ready.json` present | PASS |
| Replay | not started | no `replay_finished.json`, no `replay.log` | PASS |
| Failed containers | 1679091c and eccbc87e, exit 1 | as expected | PASS |
| Teardown stops | by rjob-controller at or after the job end | as expected | PASS |
| Platform `p2p_ping` helper error (`'ip'` not found) | on every replica, so not specific | 8/8 | PASS |
| Launcher exit after job end | later than 07:56:47Z | 07:56:50Z | PASS |
| Event and audit queries | all pass | all PASS (query_log.json) | PASS |

Classification: platform cause (inferred). The replica stopped from outside within minutes of starting, before any
native server ran on it, with no stop operation, no native error and no event naming preemption or node loss. The
handbook rule that preemption stops a whole job does not match a single-replica stop, so the exact platform
mechanism is unknown. The 30-minute wait that followed is the native reaction to a lost peer: the harness has no
check that a peer which passed the rendezvous is still alive, so the decode head waits for its full 1800 s startup
timeout (issues.md I-013).

## GPU-hour accounting

- Known container GPU hours: 8.264 (the two Failed containers, 07:25:35/56Z-07:56:45Z); container clocks are incomplete.
- Counted upper bound: 64 GPUs x 2352 s, from the first replica create (07:17:41Z) to the last replica event
  (07:56:53Z, a container kill during teardown) = 41.81 GPUh.
- Cumulative under D-P2-64-FIFTH: 18.52 + 8.57 + 41.81 = 68.90 of 128.

## Not run

Cloud copy (no native reference; the read-only node-record summary `native_tree.json` is archived and the full run
tree stays at the cloud path), pair coverage, R5 and the metric readings.

## Result and limits

- D-P2-64-FIFTH is exhausted: generation05 and both platform-cause resubmissions ended without a native reference.
  Any further GPU job needs a new user decision.
- Observation vs inference: the stop window, the missing lane and the cascade are observed. The cause of the stop
  (node loss or platform reclaim) is an inference; the platform recorded no reason.
- The platform logs hold only the 58-line worker-init prefix of each replica; the native logs come from the cloud run
  tree.
