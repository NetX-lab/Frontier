# Issues: failures and resolutions

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-11 | I-017 narrowed: the divergence is two parallel evolutions, not a missing feature (pair warmup implemented twice, only heartbeat 57caf2f7 absent from mainline), and a leaked heartbeat daemon thread is ruled out as the timeout cause. |
| 2026-11-08 | I-017 opened: the P2-64 startup-harness fix that made generation08 succeed lives only on branch `p2-64-startup-harness` (57caf2f7), which is not an ancestor of mainline HEAD, and its recorded CPU unit evidence does not reproduce today (12 failed / 6 passed). |
| 2026-11-08 | I-016 opened and resolved: `tests/unit/test_native_pd_worker.py` passed the Python 3.11+ `-P` flag to a child interpreter, so 4 of 41 mainline unit tests failed on the declared 3.10 floor; replaced with a portable `sys.path` prelude and verified 41 passed. |
| 2026-11-08 | Cleaned up the generation08-era artifacts: 313 files and 36 empty directories, 2,570,104,832 allocated bytes: 178 in-repo files over two codex-implement rounds and the lead's 135-file out-of-repo duplicate removal; the published P2-64 metrics were also recomputed from the row CSVs and reproduce the report exactly. |
| 2026-11-08 | I-015 opened and resolved: a launcher from the ended G0-D track had polled a deleted job for 18 days and grown its log to 4.33 GB; stopped on YC's approval after identity and job-state evidence. |
| 2026-11-08 | I-014 resolved: the copy job later succeeded on its own and both copy routes hold identical job output (`PAYLOAD_IDENTICAL`), so the deviation has no numerical consequence. |
| 2026-11-08 | I-014 opened: the generation08 CPU copy job stayed in Starting after an image pull failure; the copy came from the read-only SSH route instead. Also recorded that compare.py needs `PYTHONPATH` set to the repository root. |
| 2026-11-07 | I-013 harness gap fixed under D-P2-64-SIXTH: peer heartbeat check in the native worker (harness 57caf2f7), packaged as generation08. |
| 2026-11-07 | I-013 opened and classified: one generation07 replica stopped from outside after the rendezvous; the peers waited 30 minutes for the decode startup timeout. Platform cause; D-P2-64-FIFTH is exhausted. |
| 2026-11-07 | I-012 opened and classified: generation06 failed at spot-GPU device admission before any container started; platform cause, generation07 (last resubmission) submitted. I-010 did not recur on two single-call turns. |
| 2026-11-07 | I-011 opened and classified: external stops of three generation05 replicas cascaded into the job failure; platform cause, generation06 submitted. |
| 2026-11-07 | I-010 recurred on four new labels after their first tool call; each resumed with a continuation prompt, and one combined call per turn completed the monitor start. |
| 2026-11-07 | I-010 worked around: a lead-written single-run submission script driven by a Pi turn of at most three single tool calls submitted generation05 (`exp-1107-135759-122907`); the earlier process checks are corrected. |
| 2026-11-07 | I-010 opened: the DS4 Pi provider duplicates tool calls inside one message, so every generation05 submission turn ends before the launch step; nothing submitted. |
| 2026-11-01 | I-009 resolved at 06:36 UTC: name resolution works again; quota check passed and the supplement job `exp-1101-143721-849401` was submitted once. |
| 2026-11-01 | I-009 rechecked at 06:28 UTC: name resolution still fails; nothing submitted. |
| 2026-11-01 | I-009 opened: this session's runtime has no DNS resolution, so the approved A6 supplement passes (D-A6-THREE-RUN) cannot be submitted from it; the package is verified and waits for a networked launch. |
| 2026-10-31 | I-008 resolved: A10 replays of B28 finished; the EP prices of the 6,144-token prefill now match the native critical paths, and both cases fail on TTFT for reasons outside I-008. |
| 2026-10-31 | I-008: B28 merged (`ee29d548`); check of the uniform eager EP rows that B28 also replaced, against the nine uniform forwards of the native P2-64 profile. |
| 2026-10-30 | I-008 measured: the isolated G5h-32 run (`exp-1030-234535-245357`, 8.01 GPU-hours) and the native critical paths show that a single aligned call reproduces the eager EP calls on EP32 and EP16 (within 1-18 µs), while back-to-back timing fits the decode-graph and TP calls. Repair assigned as B28. |
| 2026-10-30 | I-008 opened: the back-to-back collective timing in `data/profiling/network/h200_dgx/summary.csv` gives the period of pipelined calls, not the time of one call started on aligned ranks. At 6,144 rows the EP combine is priced 164.7 µs per layer low on EP32 and 16.6-45.6 µs high on EP16. |
| 2026-10-30 | I-007 opened: the scheduler-lifecycle runs of 2026-10-25/26 are slower on the prefill side than the canonical runs in every repeat (13.8-34.4 ms); the 300 ms native prefill step taken from them is overstated. |
| 2026-10-30 | I-006 resolved by B26b (`414fdd03`, merge `49e7e5fa`): a test-fixture timing problem caused by `b0bf0c78`; the fixture's third arrival moved from 8 ms to 10 ms; no simulator change. |
| 2026-10-30 | I-002 resolved by B27 (`f00859be`, merge `3834bab6`): 128 passed. I-006 assigned with B26b (request `a92368f9`). |
| 2026-10-30 | I-006 opened: `test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` fails on main `88fbef52`; found during the B26 review. |
| 2026-10-30 | I-002 assigned to Codex as B27 (request `7653c18e`); reproduced on main `591d7c54`: 126 passed, 2 failed. |
| 2026-10-30 | I-005 opened: the B25 wave observer wrote the engine identity as bytes into JSON, so the START_DP_WAVE records of the P2-64 wave run were lost (896 error records); the forward-level records were complete and carried the diagnosis. |
| 2026-10-30 | I-004 opened and worked around: the vLLM 0.28 environment `oneshot-opd-py312` was removed at 05:22 UTC outside this task; the B25 installed-wheel check became a source-only check. |
| 2026-10-30 | I-003 opened and resolved: `/data/ycfeng` filled at 20:58 on 2026-10-29 and every running simulation failed with ENOSPC; space was recovered by YC-approved deletions and the runs were restarted. |
| 2026-10-29 | I-002 opened: two `test_balanced_routing.py` cases fail on main since B16b; the module is skipped without torch, so earlier suites did not run it. |
| 2026-10-28 | Created: unit-test fixtures wrote into this worktree's Git directory through an exported `GIT_DIR`. |
| 2026-10-28 | I-001 resolved: repair applied after YC's approval and verified. |

## I-017: the P2-64 startup-harness fix is not merged into mainline and its CPU evidence no longer reproduces (2026-11-08)

- Observation: `git merge-base --is-ancestor 57caf2f7 HEAD` is false, and `git branch -a --contains 57caf2f7` lists only
  `p2-64-startup-harness`. The mainline copy of `native_pd_worker.py` contains no occurrence of `heartbeat`; the copy in
  `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness` contains 28. The peer heartbeat check recorded as the I-013
  resolution under D-P2-64-SIXTH, which the successful generation08 run used, therefore exists only on that branch.
- Second observation: re-running the harness worktree's own unit evidence today gives `12 failed, 6 passed in 85.73s`
  (`h200_mainline/p2_64_formal_retry08/heartbeat_tests_20261108/harness_worker_unit_tests.log`), against the recorded
  `18 passed in 5.74s`. Four of the twelve failures are I-016's `-P` defect. The other eight all end in
  `TimeoutError: Native P/D rendezvous or workload deadline exceeded`, meaning the test's `urlopen` -> `Health` stub was
  never entered and the case waited out a real deadline.
- Inference: the timeouts are order-dependent, not a defect in the sealed worker. The isolation probe
  (`heartbeat_tests_20261108/isolation_probe.log`, 2026-11-08T11:32:01Z) shows two distinct patterns.
  `test_worker_forwards_canonical_clean_mode_and_idle_configuration` passes alone in 0.19 s and fails inside the full
  run, so it depends on state left by earlier cases. The parameterized group reproduces its own failure in isolation:
  run by itself it still gives `3 failed, 1 passed in 30.36s`, with `[control]` passing and the next three parameters
  timing out. Both patterns indicate a patch or module-state leak between cases in the harness branch's test module,
  and the group's leak is self-contained in its first parameter. It is not root-caused yet.
- Scope limit: generation08 ran natively on the platform and produced the accepted P2-64 reference, so no published
  metric depends on this CPU evidence. The defect is in the harness branch's test isolation, not in the measurement.
- Open work, recommended rather than performed: merge the `p2-64-startup-harness` worker and test changes back into
  mainline, per the standing rule that worktree code and documentation return to the main branch on completion, and
  root-cause the eight timeouts before accepting the merged tests. Both copies of `native_pd_worker.py` and of the test
  module have diverged in opposite directions, so the merge touches the native measurement harness and needs its own
  verification pass rather than a fast-forward.
- Narrowed 2026-11-11, while explaining the item: the divergence is not one-sided. The harness branch carries four
  commits mainline lacks (`4861e680`, `f7dc9416`, `f33306d4`, `57caf2f7`), and mainline carries six commits touching
  the same two files, including `e2289283b` "Integrate P2-64 pair warmup into the calibration harness". The pair-warmup
  capability was therefore implemented twice, once per branch, which is why the test module differs by 53 insertions
  against 161 deletions and collects 41 cases on mainline against 18 on the harness branch. Only the heartbeat commit
  `57caf2f7` is genuinely absent from mainline, so the merge is a reconciliation of two parallel evolutions, not the
  transplant of a missing feature.
- Also ruled out 2026-11-11: a leaked heartbeat daemon thread is not the cause of the eight timeouts.
  `tests/comparison/calibration/native_pd_worker.py:438-442` sets `stop_heartbeat` and joins the thread inside a
  `finally` block, so the thread does not survive a failing case. The leak is elsewhere in the test module and is still
  not root-caused.

## I-016: a Python 3.11+ interpreter flag broke 4 mainline unit tests on the declared 3.10 floor (2026-11-08, resolved)

- Observation: `python -m pytest tests/unit/test_native_pd_worker.py -q -p no:cacheprovider` on mainline HEAD gave
  `4 failed, 37 passed in 1.19s` under `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6). Each failure asserted on
  the child's stderr: `Unknown option: -P`, child return code 2. Failing cases:
  `test_worker_launch_exports_staged_extension_path_to_real_child` and the three
  `test_worker_child_ready_timeout_matches_configured_startup_budget` parameters.
  Log: `h200_mainline/p2_64_formal_retry08/heartbeat_tests_20261108/worker_unit_tests.log`.
- Cause: `tests/unit/test_native_pd_worker.py:122` launched the real probe child as
  `[sys.executable, '-P', '-c', probe]`. `-P` (do not prepend the script directory or cwd to `sys.path`) exists only
  from Python 3.11, while `environment.yml` pins `python=3.10` and `pyproject.toml` declares `requires-python >=3.10`.
  Introduced on 2026-10-21 by `9c54f663` ("Export staged native extension path to serving child processes"), which added
  these cases; the defect was latent because the module needs no GPU and had not been re-run on the floor interpreter
  since.
- Resolution: replaced the flag with a portable prelude prepended to the probe source, which strips `''` and the cwd
  from `sys.path` inside the child before the import. This preserves the actual assertion, that the probe resolves
  `native_full_structure` through the exported staged path rather than through the working directory. Verified
  `41 passed in 1.14s`, exit 0 (`mainline_worker_tests_after_fix.log`).
- Note on discovery: the codex-implement review of the H200 mainline did not find this, because that executor had no
  pytest available and audited records and committed diffs only. It was found by running the suite directly.

## I-015: a launcher from the ended G0-D track kept polling a deleted job for 18 days (2026-11-08, resolved)

- Observation: pid 1878671, started 2026-10-20T12:35:25Z as
  `/data/ycfeng/tmp/stepmind-env/bin/python -u native_track_20261020/g0d_launch/launch.py`, was still running on
  2026-11-08. Every poll of its job `exp-1020-203529-472675` failed with a `ProxyError` on `127.0.0.1:51015`, the local
  proxy port it was started with, and it wrote the full traceback each time. `native_track_20261020/g0d_launch/launcher.log`
  had reached 4,332,828,936 bytes and grew 54,320 bytes per 20 s (about 230 MB/day). `launcher_exit.json` was never
  written, so its `finally` block never ran.
- Impact: the growing log blocked the `cleanup_20261107_gen05_07` run, which could not take a stable hash of the
  protected inputs and therefore deleted nothing (its `implementation.md` records the same log growing during both
  hash passes).
- Inference: the launcher never reached a terminal job state because the proxy it captured at start no longer listens,
  so its monitor loop could not observe the job and never exited. The job itself is long gone: a read on 2026-11-08
  returned `NotFoundException`, 18 days after submission, so no live workload depended on the process. It was an H800
  `codesign` job, unrelated to the current H200 step_main work.
- Resolution: identified and stopped through Pi on HOST (label `g0d-stale-launcher-1108a`, two single-call turns), on
  YC's approval of 2026-11-08 conditioned on identifying the writer and confirming its job was no longer running. The
  stop was gated on the live argv matching the recorded argv exactly; one SIGTERM ended it in 0.5 s, with no children
  and `ppid` 1. Evidence: `/data/ycfeng/tmp/ds4_runs/g0d-stale-launcher-1108a/scratch/{log_writers.json,stop_report.json}`,
  status `STOPPED_LOG_QUIET`. Independent lead check: the log held 4,332,828,936 bytes with no growth over 25 s and its
  mtime frozen at the SIGTERM instant.
- Observed limit: in that read-only ds4 turn, `/proc/<pid>/{fd,cwd,exe}` returned `PermissionError` for this process
  even at the same uid, while `/proc/<pid>/cmdline` and `stat` read normally and the signal was delivered. The scan's
  open-file test therefore found no holder and the identification rests on argv. The kernel has no
  `yama/ptrace_scope`, and the turn ran under bubblewrap, which creates a user namespace; a child user namespace cannot
  ptrace-read a process outside it while signals still follow uid credentials. That explanation fits the evidence but
  was not tested directly. Practical rule: identify host processes by argv in a `read-only` turn, and use a `write`
  turn when open-file evidence is required.

## I-014: the generation08 CPU copy job starts late after an image pull failure (2026-11-07, resolved)

- Observation: the CPU-only copy job `exp-1108-010820-461640` (created 17:08:20Z by `cloud_copy.py submit`) was
  placed on gpu-h200-0577. The image pull started at 17:15:15Z and failed at 17:16:25Z with `failed to read expected
  number of bytes: unexpected EOF`; the replica then stayed in ImagePullBackOff (37 back-offs by 17:29:56Z) and the job
  in phase Starting. Earlier copies took about 30 s from creation to `FINAL_STATUS`. Evidence (read-only):
  `/data/ycfeng/tmp/ds4_runs/p64-formal8-copyread-1107t/scratch/copy_job_1.json` and `events_1/`.
- Inference: a registry transfer failure on a node without the image cached; not caused by the run or the script.
- Resolution: the copy came from the read-only SSH route of `h200-ssh-copy-1025c` (label `p64-formal8-sshcopy-1107u`):
  130 files, `HASH_IDENTICAL_DOWNLOAD`, independently rehashed by the lead. The job held no GPU throughout.
- Closing evidence (2026-11-08): a later pull succeeded, so the job copied and ended Succeeded at 17:32:43Z and its own
  launcher stopped it and exited (`p64-formal8-post-1107r/scratch/copy_launcher.log`, `FINAL_STATUS succeeded`). No stop
  operation was needed. Its output went to its own directory
  `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08`, so the two routes give two local copies of the
  same cloud run. `h200_mainline/p2_64_formal_retry08/copy_route_crosscheck.py` compared them: status
  `PAYLOAD_IDENTICAL`, the same 130 job-output files, 75,585,300 bytes and joint sha256 `5ef547e0a1206076...` on both
  sides, nothing present in one copy only and nothing differing. Only the two files each copy tool writes about its own
  route differ by design (`download_receipt.json` route and timings; `cloud_inventory.json`, whose root records the same
  cloud directory under the two mount views). The deviation therefore has no numerical consequence for the P2-64
  verdict. The platform-route copy is now a duplicate for cleanup; the comparison consumed the SSH copy.
- Related: the first compare.py run for generation08 lacked `PYTHONPATH` and stopped in `run_gate.py` with
  `ModuleNotFoundError: No module named 'tests'` after R5 PASS. The retained command record sets
  `PYTHONPATH="$REPO"`; `post_run_command.json` now records that environment, and the rerun finished.
  [Test report](test_report_2026-11-07_p2_64_generation08_formal.md).

## I-013: a replica lost after the rendezvous leaves its peers waiting for the full startup timeout (2026-11-07)

- Observation: in generation07 (`exp-1107-150132-396145`) gpu-h200-0726 (replica cfcd2084) wrote `ready.json` and its
  rendezvous record at 07:26:35Z and nothing after. The replica reached phase Stopped between the monitor reads at
  07:26:50Z and 07:31:52Z, with no stop operation in the audit, no event after the image pull, no native error and a
  container state left at `running`. The other 7 nodes had planned it as decode lane 2, so the decode DP head never
  became healthy and raised its 1800 s startup timeout at 07:56:45Z; the any-replica fail policy then ended the job.
  41.81 GPUh counted (upper bound); no file in the run tree changed after 07:36:40Z, so the last 20 minutes
  were spent waiting on the lost peer.
- Inference: a platform-side stop of one replica (node loss or reclaim; the platform recorded no reason). A stopped
  replica does not trigger the fail policy; only the later Failed replicas do.
- Harness gap (observed): `native_pd_worker.py` checks only that 8 `ready.json` files exist and that no peer wrote
  `failed.json`; it has no liveness check for a peer that passed the rendezvous and then disappeared, so the wait
  lasts until `startup_timeout_s`. In generation05 the lost replica was the DP head, so the cascade was immediate.
- Resolution: classified as a platform cause. D-P2-64-FIFTH is exhausted, so nothing is resubmitted; a peer liveness
  check is part of the proposal for the next decision. Evidence:
  `h200_mainline/p2_64_formal_retry07/terminal_20261107/`.
- Fix (D-P2-64-SIXTH): harness commit 57caf2f7 in `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness`. Each node
  refreshes `nodes/<host>/heartbeat.json` every 10 s from the rendezvous on; a waiting node fails once an undrained
  peer's heartbeat is unchanged for more than 180 s. CPU check: the stopped-peer case now fails on the heartbeat
  instead of the startup timeout. Packaged as generation08 (manifest 1a2bbc13...). It shortens the failure; it does
  not prevent platform stops. [Test report](test_report_2026-11-07_p2_64_heartbeat_generation08_package.md).

## I-012: spot-GPU device-admission failure ends a job before any container starts (2026-11-07)

- Observation: in generation06 (`exp-1107-144321-305543`) the 8 replicas started at 06:43:55-56Z and waited on their
  nodes for spot GPUs to be released by preemption. At 06:48:30Z the kubelet on gpu-h200-0757 rejected replica
  8f14e45f: `UnexpectedAdmissionError: Allocate failed due to requested number of devices unavailable for
  nvidia.com/spot-gpu. Requested: 8, Available: 7, which is unexpected`. No container started. The platform recorded
  `Some critical tasks failed.` and rjob-controller closed the other replicas at 06:48:31-47Z.
- Inference: the scheduler placed the pod before the preempted job had released all 8 GPUs of the node. This is a
  platform-side race, not a package, native or code defect.
- Resolution: classified as a platform cause under D-P2-64-FIFTH; 8.57 GPUh counted as the upper bound; generation07
  (`exp-1107-150132-396145`) submitted as resubmission 2 of 2, the last. Evidence:
  `h200_mainline/p2_64_formal_retry06/terminal_20261107/`.

## I-011: external replica stops of a preemptible job cascade into a job failure (2026-11-07)

- Observation: in generation05 (`exp-1107-135759-122907`) three replicas entered phase Stopped from outside while
  vLLM was starting (8f14e45f by 06:09:59Z, a87ff679 at 06:10:34Z, c81e728d at 06:15:56Z). The audit has no stop
  operation on them and their logs show no native error. One of them was the decode DP head. Decode lanes DP1 and DP2
  then lost its TCPStore, vLLM exited 0, the worker raised `Native server exited early: 0`, and both containers exited 1.
  The platform's any-replica fail policy stopped the job at 06:16:02Z.
- Inference: platform-side reclaim of spot GPUs of the preemptible job (no event names preemption). The cascade is the
  expected reaction of the native worker to a lost DP peer, not a separate defect.
- Resolution: classified as a platform cause under D-P2-64-FIFTH; generation06 (`exp-1107-144321-305543`) submitted
  as resubmission 1 of 2. Evidence: `h200_mainline/p2_64_formal_retry05/terminal_20261107/`.

## I-010: provider stream error blocks the generation05 submission turns (2026-11-07)

- Observation: Pi labels `p64-formal5-submit-1107a` (turns 1-3) and `p64-formal5-submit-1107d` (turns 1-16) all end with
  `OpenAI Responses stream completed with an unfinished tool call`. In each failed assistant message the provider sends a
  complete tool call, then an empty text item, then the same tool call id again without its end event
  (`events.jsonl` of 1107d, line 89). Pi rejects the stream and executes no tool from that message.
- A fresh short read-only probe `provider-probe-1107c` completed four messages normally at 05:42 UTC, so the provider is
  reachable; the failure is intermittent per message and became 16/16 on the resumed 1107d session.
- Inference: provider-side Responses stream defect, not a task, package or platform failure. Prompt rules (one tool call
  per message) did not prevent it.
- State before the workaround: no generation05 submission. Lead checks before every resume found no
  `launcher_started.json`, no `launcher_exit.json` and no `case/runs`. Correction: the lead's "no retry05 process"
  check ran inside the lead's isolated PID namespace (its `/proc` shows only its own processes), so it could not see
  HOST processes. The submission script's HOST-side guard at 05:57:50Z later confirmed no live retry05 launcher or
  wrapper and no launch markers (`p2_64_formal_retry05/generation05_submission_acceptance.json`,
  `pre_launch_evidence`). Auto-resume driver and log: `/data/ycfeng/tmp/glm53_dsv4/h200/ds4_resume/`.
- Workaround (2026-11-07, worked around, provider defect not fixed): the lead wrote
  `h200_mainline/submit_package.py` (approval and seal checks, launch-evidence guard, check-only, quota snapshot, one
  detached launch, job-name capture, one exact-job read, 18 checks). Pi label `p64-formal5-submit-1107e` ran it with
  three single tool calls and exited normally. Result `SUBMITTED_ALL_CHECKS_PASS`, job `exp-1107-135759-122907`,
  launcher PID 2753825. Monitoring uses the same pattern: Pi label `p64-formal5-monitor-1107f` started the detached
  read-only `h200_mainline/monitor_job.py` (PID 2774280). Later Pi turns keep to at most three single tool calls,
  with the work in lead-written scripts.
- Recurrence (2026-11-07, 06:20-06:47Z): labels `p64-formal5-terminal-1107i`, `p64-formal6-submit-1107j` and
  `p64-formal6-monitor-1107k` ended with the same `unfinished tool call` error, mostly right after the first tool call
  of a turn. Resuming the same label with a prompt that continues from the last completed step worked; when the
  second call also failed, folding the dependent steps into one combined command (the monitor start) completed the
  step in one call. The lead checks the state before each resume, so no step runs twice.

## I-001: unit-test fixtures wrote into the worktree's Git directory (2026-10-28)

Status: resolved 2026-10-28. YC approved the repair ("批准，继续"); the lead applied it with exit 0, and every check below passed. Owner: lead.

**Failure.** During the B8 review, the lead ran the whole unit suite twice in parallel: before (on a `git archive dd4c4317` snapshot) and after (on the working tree). The runner was `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_review/checks/run_unit_pair.sh`. The before half exported `GIT_DIR=/data/ycfeng/Frontier/.git/worktrees/glm53-dsv4-vllm028` to give the snapshot a Git identity. Fixture tests that build temporary repositories then sent `git init`, `git config` and `git commit` to that directory instead of their temporary directories:
- `tests/unit/test_moe_ep_baseline_replay.py:167`;
- `tests/unit/test_moe_ep_non_dummy_matrix.py:5615`;
- `tests/unit/test_native_pd_manifest.py:35`;
- `tests/unit/test_refactor_fidelity_gate.py:464`.

**Observed state** (writes between 10:43:04 and 10:45:31 UTC):
- `refs/heads/feat/glm53-dsv4-vllm028`: `dd4c4317` (B7) → `80aacd8e` ("baseline", Frontier Tests, parent `dd4c4317`, deletes 4994 files and adds `tracked.txt`).
- This worktree's HEAD is detached at `4a107a47` ("initial", author `t`), after 29 fixture commits. Its index holds the 9 fixture files.
- The shared `/data/ycfeng/Frontier/.git/config`:
  - `core.bare = true`, written by the reinitializing `git init`; the main checkout is a normal checkout, so the value must be `false`;
  - an appended `[user]` section, `name = t` and `email = t@example.com`, which would give every repository commit without its own identity the author `t`.
- Not affected:
  - the working-tree files (the fixtures' work tree was their temporary directory), so the B8 edits are on disk;
  - the main checkout's `HEAD` and `index`;
  - every other branch and every remote-tracking ref;
  - other worktrees' HEADs.

**Backups.** In `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_review/git_repair/`: `config.polluted`, `worktree_HEAD.polluted`, `worktree_index.polluted`, `branch_ref.polluted`.

**Repair.** The permission check first blocked it as a destructive Git action; it was applied after YC's approval. It restores the earlier state exactly, and the reflog keeps the old values:
1. `git config --file /data/ycfeng/Frontier/.git/config core.bare false`
2. `git config --file /data/ycfeng/Frontier/.git/config --remove-section user`
3. `git update-ref refs/heads/feat/glm53-dsv4-vllm028 dd4c4317606789e5cafcba148f6c1955f3b96f7f 80aacd8e282bd1ae2d877caeb794e5718dbf7686`
4. In `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`: `git symbolic-ref HEAD refs/heads/feat/glm53-dsv4-vllm028`, then `git read-tree dd4c4317606789e5cafcba148f6c1955f3b96f7f`. This touches the index only; the working tree is not changed.

**Checks after the repair:**
- `git -C /data/ycfeng/Frontier rev-parse --is-bare-repository` prints `false`;
- `git config user.name` resolves to the global identity;
- in this worktree, `git branch --show-current` prints `feat/glm53-dsv4-vllm028` and `git log -1` shows `dd4c4317`;
- `git status --short` lists only the B8 changes, the untracked task records and the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`.

Observed after the repair:
- every check above as expected; the main checkout's `git status` works on branch `main`;
- `git status --short` in this worktree lists exactly the B8 changes and the known untracked paths;
- the whole unit suite rerun on the working tree without `GIT_DIR`: 133 failed, 6650 passed, 133 skipped, 109 errors. It differs from the first after-run only by the R4 test (failure → passed), so the 91 s overlap changed no result.

B8 was then committed.

**Prevention.** Never export `GIT_DIR` or `GIT_WORK_TREE` around a test suite or any command that runs `git init`. A before-snapshot that needs a Git identity gets its own worktree; otherwise its Git-reading tests are accepted as failing in the comparison.

## I-003: `/data/ycfeng` full; all stage 4 simulations lost (2026-10-29)

Status: resolved for now; the external writer is unidentified. Owner: lead.

Observed: at 20:58 on 2026-10-29, `df` showed `/dev/rbd1` (XFS with project quota, mounted at `/data`) with 1 MB free. Every simulation that was still running failed with `OSError: [Errno 28] No space left on device`:

- official A10 runs on the promoted table: `stage4/a10/p2_32_cold_g1hs`, `p2_32_warm_g1hs`, `p2_64_cold_g1hs` and `p2_64_warm_g1hs` (all mid-run; the observer stream close and the receipt write failed);
- old-table A10 runs `stage4/a10/p2_64_cold` and `p2_64_warm`;
- the mid-run profile `stage4/probe/p2_64_mid` (it had reached 61.3 simulated s after 8,460 wall s; no `profile.pstats` was written);
- the census runs under `stage4/census/`.

Probe `stage4/probe/p2_64_c` (old table, no observer) had already ended with `EXIT 0` at 20:37; its gate output is kept.

Cause (inference): the simulations were not the writer. Each run root is about 300 MB including its 1,758-file cache. `find -mmin`/`-cmin` over the readable trees showed no matching growth. Free space also moved both ways while no lead process was writing (14,256 MB, then 14,972 MB two minutes later on 2026-10-30). The likely writers are processes outside this sandbox, such as GPU workers writing as uid 65534 through NFS into directories this user cannot read.

Resolution: YC approved deleting three groups (2026-10-30):

- the G1h supplement traces `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_s01/download/G1H_SUPPLEMENT/bridge/FULL_AND_PIECEWISE/traces/` (9.6 GB, 450 files; the raw records and the promoted table do not depend on them);
- the temporary directories `stage4/promote_real/scratch` (830 MB) and `stage4/g1h_s01_post_dryrun` (615 MB);
- the old-table caches under `stage4/a10/{p2_32,p2_64}_{cold,warm}/cache` and `stage4/probe/p2_32_{a,b,c,perf,perf2,prof}/cache` (1,758 files each).

The G1h-02 traces were not approved and were kept. Free space after the deletions: 14,257 MB. A background `df` logger now writes `stage4/disk_watch.log` every 300 s so a repeat can be timed.

## I-002: `test_balanced_routing.py` cache test fails on main since B16b (2026-10-29)

Status: resolved on 2026-10-30 by B27 (commit `f00859be`, merged as `3834bab6`): the test now trains and reads the V4 models under `KERNEL_ONLY` and asserts the empty `eager` view; lead rerun with CPU torch: 128 passed (log `/data/ycfeng/tmp/b27_lead.log`). Assigned as B27 (request `7653c18e`, [assignment](h3o_operator_costs/codex_assignment_B27_balanced_routing_v4_test.md), worktree `.worktrees/b27-balanced-routing-v4-test` from main `b1cf3447`). Reproduced on main `591d7c54` with the command of the assignment: 126 passed, 2 failed (both parametrizations). Owner: lead.

Observed during the B23 review. With CPU torch on the path (`PYTHONPATH=...:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages`, `VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0`), `tests/unit/test_balanced_routing.py` gives 126 passed and 2 failed, the same on main `2da2c186` and on the B23 tree:

```text
tests/unit/test_balanced_routing.py:316: KeyError: 'moe_gating_routing_topk'
    assert manager.get_models_for_cluster(role)["eager"][op] is cold[role]
```

Cause (from the code): B16b (`61c2f9d3`) makes `_get_measurement_types_for_cluster` return only the V4 measurement type, `KERNEL_ONLY`, for every DeepSeek-V4 role, so the `eager` view of `get_models_for_cluster` is empty for V4. The test still trains V4 gating models from `CUDA_EVENT` rows and reads them from the `eager` view. Without torch the module is skipped by `pytest.importorskip("torch")`, so suites run without torch do not report it.

Next action: update the test to the B16b contract (V4 roles read `KERNEL_ONLY`), in a later assignment; no simulator behavior is involved.

## I-004: vLLM 0.28 environment removed during the B25 implementation (2026-10-30)

Status: worked around; the environment is not restored. Owner: lead.

**Observed.** Codex's B25 CPU check exited 127: `/data/ycfeng/envs/oneshot-opd-py312/bin/python` no longer existed. The `df` logger (`stage4/disk_watch.log`) shows used space on `/data/ycfeng` falling from 986,494 MB at 05:18 UTC to 934,741 MB at 05:23 and 851,306 MB at 05:28, which matches the removal time. No command from this task removed the environment (inference from the session history; the deleting process is not known).

**Impact.** B25 planned to check the observer hooks against the installed vLLM wheel. Without the environment, the attributes could be checked only against the NATIVE source tree (`/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`).

**Resolution.**
- `check_native_dp_wave_source.py` checks every hooked attribute against the NATIVE source; it passed on the B25 branch and on the composed harness branch `p2-64-wave-composed` (`6488ea15`).
- The H4b check had already found the installed wheel byte-identical to NATIVE for the engine-core, scheduler and output files, and the lead verified each hooked attribute in the NATIVE source.
- The soft checks of the instrumented run itself serve as the installed-path check: an attribute that differs in the image appears as an error record, not as a silent gap.
- Recorded in `h3o_operator_costs/` B25 design section 9.

Next action: none for B25. A later task that needs a local vLLM 0.28 Python must rebuild the environment first.

## I-005: B25 wave observer lost the START_DP_WAVE records (2026-10-30)

Status: open; fix only before a future run (no rerun is approved). Owner: lead.

**Observed.** The instrumented native P2-64 run `exp-1030-144848-164721` (requirements item 42) succeeded, but its observer files hold 896 error records instead of the wave-start messages. `tests/comparison/calibration/native_dp_wave.py` (`enqueue_fields`, line 105, and `handle_begin`, line 136) copies `excluded_engine_index` into the JSON record as received. In vLLM-BS `2cf0a691` that value is the engine identity, a `bytes` value: the front end sends `("FIRST_REQ", chosen_engine)` (`vllm/v1/engine/core_client.py:1422`), and the coordinator forwards it as `exclude_engine_index` in START_DP_WAVE (`vllm/v1/engine/coordinator.py:354` and `466`). `json` cannot encode `bytes`, so every such record failed.

**Impact.** The wave-start times of the coordinator and the engines are missing. The forward-level records (worker `dp_cpu_all_reduce` and `dp_padding_sync`, scheduler `schedule`/`update`/`add`) are complete on all eight hosts, and they carry the diagnosis (A10 report section 3.5). The run is not repeated for this.

**Resolution (pending).** Encode the identity before writing it (for example as hex) in the observer, with a unit test that feeds a `bytes` identity. Apply it only when a further wave run is approved.

## I-006: placement test of the late-joining DP lane fails on main (2026-10-30)

Status: resolved 2026-10-30 by B26b (request `a92368f9`, [assignment](h3o_operator_costs/codex_assignment_B26b_placement_key_and_i006.md), [report](h3o_operator_costs/b26b/report.md); commit `414fdd03`, merged into main as `49e7e5fa`). Owner: lead.

**Observed.** `tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` fails on main `88fbef52` (18 passed, 1 failed; log `/data/ycfeng/tmp/main_placement.log`) and in the same way on the B26 branch. The assertion at line 481 expects lane 2 to start forward group 0 with a dummy forward, `{2: (0, True)}`, and receives a real batch, `{2: (0, False)}`. The test file last changed in `5f907f66` (H4b-(b) item 1); 143 commits follow it on main, so the commit that broke the test is not known.

**Impact.** The test checks the four-lane MoE shape of issue W9-04 under both placement policies: a lane that is idle when a peer reaches the first MoE layer joins that forward with a dummy forward, and a request that arrives during it starts the next forward. Either the simulator now places the request's batch into the dummy forward's group, or the test's setup no longer reaches that shape. A10 is not known to be affected, because its prefill requests arrive 1 s apart, but this is not verified.

**Cause.** The test setup, not the simulator. Commit `b0bf0c78` (2026-10-08) added the forward preparation time (schedule time plus input preparation time) before the first Prefill/Decode sync event of a forward. With the fixture's 1 ms operator durations, the first lanes now reach the first MoE layer at 9 ms and 11 ms instead of 7 ms and 9 ms. The fixture's third request arrives at 8 ms, before the idle lanes are woken, so lane 2 correctly forms a real forward 0. Codex checked exported trees: the parent `2c9d625d` passes the test and `b0bf0c78` fails it. Main first received this test in merge `9a650fba` (2026-10-14), and that tree already fails. The earlier premise that the test passed at `5f907f66` was wrong: it already failed there.

**Native check.** vLLM's busy loop (`core.py:2153`, `2169`, `2178-2182`) runs the dummy forward before the next input poll, so a request that arrives during the dummy forward starts the next forward. The test's intended assertion still matches native behavior.

**Resolution.** `LATE_JOIN_TRACE`'s third arrival moved from 8 ms to 10 ms, and the test now asserts that the arrival falls inside lane 2's dummy forward (dummy start 9 ms, arrival 10 ms, dummy end 65 ms). Both placement policies keep placement `[0, 1, 2]`. No production source changed. The lead's run of the 13 affected test files passed (269 passed; log `/data/ycfeng/tmp/b26b_lead/verify.log`). A10 uses `round_robin` placement and does not run this path.

## I-007: the scheduler-lifecycle runs slow down the native prefill step (2026-10-30)

Status: open (cause not identified; the affected readings are corrected). Owner: lead.

**Observed.** On the proxy clock, without first-contact requests, every repeat of `dsv4-h200-p2-lifecycle-20261025-01` (P2-32: 260.7 / 255.4 / 256.4 ms) and of `dsv4-h200-p2-64-lifecycle-20261026-01` (P2-64: 570.3 / 581.6 / 574.0 ms) has a longer prefill side than every repeat of the canonical runs (P2-32r: 234.4 / 236.0 / 236.1 ms; P2-64r: 556.5 / 552.4 / 547.2 ms). The P2-64 wave run of 2026-10-30 (551.5 ms) falls inside the canonical range. Evidence: `h3o_operator_costs/a10/k19/p_side_per_repeat_stdout.txt`; A10 report section 5.5.

**Impact.** The added time sits inside the engine step: the P2-32 lifecycle step averages 244.6 ms against about 222 ms left by the canonical prefill side, and the lifecycle run's ingress is not larger. Readings that took the native prefill step from the lifecycle runs overstate it: the 300 ms compute of A10 report sections 3.2 and 3.5 and the "about 25 ms of compute" difference of section 3.2. On the wave run, a prefill served at once takes 289.6 ms (n 6). The H4b handoff measurements are not affected: the lifecycle runs' handoff is not longer than the canonical one (A10 report section 3.6).

**Next.** Native step values come from the wave run or from canonical-timed runs. The lifecycle observer's code is not inspected yet; the cause matters only if another lifecycle run is planned.

## I-008: back-to-back timing misprices the 6,144-row EP combine (2026-10-30)

Status: resolved on 2026-10-31 (repair B28 merged as `ee29d548`; A10 replays read on 2026-10-31, below). Decode-width EP calls keep the back-to-back rows, which match native (G5h report, "Revised decision"). Owner: lead.

**Observed.** The P2-64 native lane-0 prefill takes 2,176.5 µs per layer in the EP combine `ncclDevKernel_Reduce_Sum_bf16_RING_LL` on TP rank 0. Frontier prices that call at 2,011.6 µs, the back-to-back median of G5h-32 `_01` (`eager_t6144_1_1_1`), which is 7.08 ms per forward low. Evidence: A10 report section 5.6; `h3o_operator_costs/a10/k19/prefill_compute/`.

**Cause.** In the forward, all ranks start the call together; lanes and TP ranks arrive within 3.5 µs and 21.4 µs (medians), and no other stream overlaps the Reduce. The root rank 0 ends last. In the benchmark's 10 back-to-back calls, ranks 1-7 start each call 450-720 µs before rank 0, because they finished the previous call earlier. Rank 0's kernel is therefore shorter, about 2,000 µs. On EP16 (G5h-16 `_02`) that head start is only 60-260 µs, and the benchmark is above native: 764.4 µs priced against 718.8 µs (lane 0) and 747.8 µs (lane 1). No single factor fits both topologies.

**Impact.** P2-64 prefill step −7.08 ms; through the FIFO queue, about −10.6 ms of TTFT. P2-32 step +0.7 to +2.0 ms. Decode-width calls use the same timing mode. Whether they are also affected is not checked; the G5h report recorded decode biases of −12 to +15 µs per call.

**Next (done 2026-10-30, below).** Measure every EP collective from an aligned start (a barrier on the group's stream before each timed call; per-rank end times from the kernel trace; price = the latest end in the timed lane) on EP32 and EP16. Then compare with the native kernels of every group before the consumer changes.

**Measured (2026-10-30).** G5h-32 isolated `_01` (RJob `exp-1030-234535-245357`, 32 H200, 8.01 GPU-hours; G5h 25.32 of 36) repeats the G5h-32 `_01` grid with one aligned call per repeat; its kernels equal the native P2-64 kernels in 9 of 9 groups. The native side is now the critical path of each layer over all EP ranks (`ep_critical.py`), not one rank's kernel. Per layer, at 6,144 rows:

| Call | Native critical path | Isolated | Back-to-back |
| --- | --- | --- | --- |
| EP32 combine (P2-64) | 2,171.2 µs | 2,172.0 µs | 2,011.1 µs |
| EP32 dispatch (P2-64) | about 1,048 µs | 1,043.0 µs | 1,031.9 µs |
| EP16 combine (P2-32) | 717.0 µs | 734.6 µs | 763.9 µs |
| EP16 dispatch (P2-32) | 541.8 µs | 560.2 µs | 801.8 µs |

The isolated mode is 43-93 µs above back-to-back for the decode-graph EP calls and 4-36 µs above for the TP calls, where back-to-back matches native. Details and the reading: [G5h report](test_report_2026-10-28_g5h_collective_kernel_comparison.md), section "I-008" and "Revised decision".

**Repair (B28).** (1) `data/profiling/network/h200_dgx/summary.csv` takes the EP rows of every eager scenario with a 6,144-token lane from the isolated runs (G5h-16 isolated `_01`, G5h-32 isolated `_01`); all other rows stay back-to-back. (2) `MeasuredCCBackend` prices an uneven EP dispatch from its exact measured `AllGatherV` row, as it prices the uneven combine. Expected change per 6,144-token prefill forward: P2-64 +5.50 ms, P2-32 −11.64 ms (`evidence/collective_kernels/expected_prices.txt`). With this repair alone both cases are expected to fail the TTFT gate (about −10.5% and −11.1%) until the ingress and the host part of the step are modeled.

**Uniform rows (2026-10-31).** B28 (merged as `ee29d548`) also replaced the rows where every lane carries 6,144 tokens. On EP16 the uniform (6,144, 6,144) prices fell: combine 768.5 → 454.3 µs, dispatch 801.7 → 567.5 µs. On EP32 the uniform (6,144 × 4) prices rose by 3.5% and 5.8%: combine 865.2 → 895.8 µs, dispatch 1,076.0 → 1,138.8 µs. The native P2-32 profile cannot test the EP16 rows: each of its three prefill forwards pairs one prefill lane with a dummy lane. The native P2-64 profile has nine uniform forwards (98-106): all four DP lanes are in prefill, and the EP kernels are the uniform `AllGather_RING_LL` and `ReduceScatter_Sum_bf16_RING_LL`, not the uneven `AllGatherV` and `Reduce`. Readings on TP rank 0 of dp1 and dp3 (the dp0 trace ends at forward 95):

- Dispatch: the 43 EP calls are exactly the 43 AllGather kernels above 700 µs; the next one is 545-751 µs. Their shortest per forward is 1,052-1,386 µs, close to both prices (1,076.0 and 1,138.8 µs). Their medians (1,132-4,040 µs) include the wait for later ranks.
- Combine: duration does not separate the 43 EP calls from the TP ReduceScatter calls (44-47 kernels above 700 µs per forward). The 43 longest have minima 809-849 µs and medians 834-852 µs (one forward 1,211 µs): 5-10% below the new price 895.8 µs and 1.5-6.5% below the old 865.2 µs.

One rank's kernel durations are not critical paths, so these readings check consistency only. For EP16 the support is an inference from scaling: the EP32 uniform call moves twice the data of the EP16 one, and the isolated rows scale by 1.97 (combine) and 2.01 (dispatch), while the old back-to-back rows scale by 1.13 and 1.34. That fits the old EP16 uniform rows carrying the back-to-back queueing that I-008 describes. Evidence: `h3o_operator_costs/b28/lead/uniform/` (`uniform_ep.py` → `uniform_ep_p2_64.txt`, `uniform_ep_ranks.py` → `uniform_ep_ranks.txt`, forward lists `forwards_p2_{32,64}.txt`).

**A10 replays (2026-10-31).** The warm K=19 replays on `ee29d548` (A10 report section 5.7) change each 6,144-token prefill forward by a fixed amount for each DP lane of the prefill. On dp0 the change equals the price check: P2-32 −11.642 ms, P2-64 +5.498 ms. The means over all waves are −12.359 and +4.644 ms. On dp0, the EP prices are now 1.55 ms (P2-32) and within 0.2 ms (P2-64) per forward from the native critical paths. TTFT: P2-32 −11.22% (B24 −7.94%), P2-64 −10.60% (B24 −11.62%); both FAIL. The other metrics stay within 2.45%. The remaining gap is outside I-008: the unmodeled path outside the engine (about 22 ms per request) and the V4 host part of the step. Evidence: `h3o_operator_costs/a10/k19/*_k19b28*`, `h3o_operator_costs/a10/k19/b28/`.

## I-009: no DNS in the session runtime blocks the A6 supplement submission (2026-11-01)

Status: resolved 2026-11-01 06:36 UTC (name resolution works again in this runtime; launch record at the end of this section). Owner: lead.

**Observed.** `socket.getaddrinfo` fails with `gaierror [Errno -3] Temporary failure in name resolution` for `deploy.i.shaipower.com`, `oss.i.shaipower.com`, `artifactory.stepfun-inc.com` and `github.com` (checked 2026-11-01 05:44 UTC, after an earlier failure of the read-only quota inspection `h3o_operator_costs/a6_three_run_20261101/inspect_quota.py`). The proxy recipe, the StepMind RJob API and `ssh shai-ycfeng` all need name resolution. No platform operation was attempted or performed; `quota.json` does not exist. A recheck at 06:28 UTC still failed for `oss.i.shaipower.com` and `deploy.i.shaipower.com`.

**Impact.** D-A6-THREE-RUN is approved but cannot start: the package `/data/ycfeng/Frontier/.worktrees/g1h-supplement-build/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_three_run_20261101/package_01` passes its geometry/source check (`verification.json`, 432 workloads × 2 passes = 864 cases, estimated 3,332 s = 0.9255 GPUh) and its ResourceConfig dry run (`payload_check.json`: 1 H200, step_main, non-preemptible, 5,400 s payload timeout). CPU-only work (C and B of the same decision set) continued.

**Launch once the network works** (bash, from the build worktree; the launcher process must stay alive for the whole job):

1. Read-only quota and personal-reservation check: run `h3o_operator_costs/a6_three_run_20261101/inspect_quota.py` from the main worktree after the handbook proxy recipe, with the StepMind environment Python; it writes `quota.json` next to itself. Submit only if personal non-preemptible H200 reservations stay at most 32 with this job.
2. `bash /data/ycfeng/Frontier/.worktrees/g1h-supplement-build/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_three_run_20261101/package_01/launch/launch_glm_producers.sh /data/ycfeng/Frontier/.worktrees/g1h-supplement-build/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_three_run_20261101/package_01/jobs/A6_SUPPLEMENT_PASSES.json`

**Resolution (2026-11-01).** At 06:36 UTC `getent hosts deploy.i.shaipower.com` returned `10.70.201.188`. Step 1 ran at 06:36:46 UTC (`quota.json` written): five live personal jobs, all preemptible (`quotagroup.brainpp.cn/preemptible=yes`, `step_main`), so this job makes one personal non-preemptible H200; the quota-group object read returned 403 and is not used. `ssh shai-ycfeng` (`id -un` = `i-fengyicheng`) confirmed the output directory `/mnt/ws-jfs/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/a6_supplement_passes_20261101_01` did not exist. Step 2 ran once under `setsid nohup` at 06:37:17 UTC (launcher pid 3087410, log `/data/ycfeng/tmp/a6_supplement_passes_20261101_01/launcher.log`): `Job Create` `exp-1101-143721-849401`. Read back at 06:38 UTC: creator `i-fengyicheng`, preemptible `no`, `step_main`, queue `step-main-default`, 1 GPU / 16 CPU / 125 Gi, the locked image digest `2286e853…`, NFS source `100.96.128.195:/data/ycfeng/Frontier/.worktrees/g1h-supplement-build` (current host address), phase Running.
