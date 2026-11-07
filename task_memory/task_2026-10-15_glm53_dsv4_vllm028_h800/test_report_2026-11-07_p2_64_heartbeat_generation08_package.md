# Test report: native peer heartbeat check and the generation08 package (2026-11-07)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-07 | Created: heartbeat change in the native worker, CPU tests, generation08 build, lead package audit and submission preparation under D-P2-64-SIXTH. |

## Scope

D-P2-64-SIXTH (requirements.md), step 1: before any further P2-64 job, add a peer liveness check to the native worker
so that a replica lost after the rendezvous (generation07, issues.md I-013) fails the run within minutes instead of
after the 1800 s startup timeout. Then rebuild the package as generation08 and verify it on CPU. No GPU job ran in this
step; native code, workload, promotion receipt, R5 and the numerical criteria are unchanged.

## Change

Harness worktree `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness` (branch `p2-64-startup-harness`), the same
harness generation07 used (`f33306d4`). Commit `57caf2f7` "Fail native P/D waits when a peer heartbeat stops", two
files:

- `tests/comparison/calibration/native_pd_worker.py`
  - Each node writes `nodes/<host>/heartbeat.json` (a counter and UTC time) before `ready.json`, then a daemon thread
    rewrites it every `heartbeat_interval_s`. The thread stops at the start of the `finally` block, when the node has
    drained or failed.
  - `await_condition` now also fails when any peer that has not drained keeps the same heartbeat content for more than
    `peer_heartbeat_timeout_s`. The age is measured on the waiting node's own monotonic clock, so clock differences
    between nodes do not matter.
  - Both values come from the run config; there is no default.
- `tests/unit/test_native_pd_worker.py`: the two config fields in the fixture, plus two cases:
  - a peer whose heartbeat stops while the head waits for health: the worker fails with the heartbeat error;
  - a peer whose heartbeat keeps changing for longer than the limit: the worker completes normally.

Builder `h200_mainline/build_package.py`: the CASES entry `p2_64_formal_retry08` (run tag
`dsv4-h200-p2-64-formal-20261107-08`) and `PEER_HEARTBEAT = dict(heartbeat_interval_s=10, peer_heartbeat_timeout_s=180)`,
added to every formal config. The limit is 18 refresh periods, so a slow cloud write does not trip it; a lost replica
is detected about 3 minutes after its last heartbeat.

## Environment

- Harness tests: `/data/ycfeng/envs/openmopd-py312/bin/python` (Python 3.12), from the harness worktree root.
- Build: the same interpreter, from `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
- Lead audit: system `python3` 3.10.6.

## Checks

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Worker unit tests | `python -m pytest tests/unit/test_native_pd_worker.py -q -p no:cacheprovider` | all pass | 18 passed in 5.74 s | PASS |
| Related harness tests | `python -m pytest tests/unit/test_native_pd_worker.py tests/unit/test_native_pd_manifest.py tests/unit/test_calibration_pd_harness.py tests/unit/test_native_pd_canonical_http.py -q -p no:cacheprovider` | all pass | 101 passed in 19.50 s | PASS |
| Old vs new worker, stopped-peer case (startup timeout 10 s, limit 1 s) | one-off script loading `git show HEAD:...native_pd_worker.py` before the commit | old waits for the timeout; new fails on the heartbeat | old: `TimeoutError` after 10.01 s; new: `RuntimeError: Native P/D peer heartbeat unchanged for more than 1 s: ['host3']` after 1.01 s | PASS |
| Build | `build_package.py p2_64_formal_retry08 --package package_01 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness` | exit 0, check-only pass | exit 0, `INPUT_HASHES_PASS_NO_SUBMISSION`, harness commit 57caf2f7 | PASS |
| Lead package audit | `python3 h200_mainline/p2_64_formal_retry08/lead_package_audit.py` | see below | `PASS_PACKAGE_IDENTITY_HEARTBEAT_AND_CHECK_ONLY` | PASS |

The lead package audit compares generation08 with generation07:

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Sealed files | 47, hashes match in both packages | 47 | PASS |
| Byte-identical files | 40 | 40 | PASS |
| Changed files | the 6 identity files plus `native_pd_worker.py` | as expected | PASS |
| Identity files | exact replacement of package root, run tag, cloud output, promotion build time, purpose, harness commit and the derived overlay and worker digests | as expected | PASS |
| Config | only `heartbeat_interval_s` = 10 and `peer_heartbeat_timeout_s` = 180 added | as expected | PASS |
| Staged worker | equals the committed worker at 57caf2f7; generation07's equals f33306d4 | as expected | PASS |
| Harness diff f33306d4..57caf2f7 | the worker and its unit test only | as expected | PASS |
| Pins | harness 57caf2f7 and native a06a72a4, both clean | as expected | PASS |
| Resources | 8 x 8 H200, step_main, preemptible, 3600 s | as expected | PASS |
| check-only | `INPUT_HASHES_PASS_NO_SUBMISSION`, rc 0 | as expected | PASS |
| Generation07 manifest | unchanged 8fd5d9dc... | unchanged | PASS |

Generation08 manifest: `1a2bbc13339b7c741cb8151611864c7e7815d8bb180588ffff2700ae888d3508`.

The build left a bytecode cache `package_01/__pycache__/native_pd_worker.cpython-312.pyc` (written at 08:29:54Z
during the build's in-process check). It is outside the seal and was removed; the audit's check-only runs with `-B`.

## Submission preparation (not yet run)

- Approval receipt `h200_mainline/p2_64_formal_retry08/approval_receipt.json`: D-P2-64-SIXTH, first job, chain
  ceiling 128 container GPUh, 0 counted.
- Launcher wrapper `/data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry08.sh`: the generation07 wrapper with the
  new authorization text and package path only.
- Post-run command `h200_mainline/p2_64_formal_retry08/post_run_command.json`: the generation07 command with the
  generation08 package, download and output paths.
- Pi label `p64-formal8-submit-1107q`: prompt and single-call script
  `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal8-submit-1107q.{md,sh}` (the generation07 script with the
  generation07 reader smoke, the generation08 package and wrapper).
- Planned time: about 16:30Z (00:30 Beijing), a low-demand night hour; the platform guide says large preemptible jobs
  submitted in the daytime are preempted more often.

## Limits

- The CPU tests use stand-ins for the platform, the native server and the cloud file system. They do not show how
  quickly the cloud file system makes one node's heartbeat visible to another; the 180 s limit leaves wide room for
  that.
- The check only shortens the time to failure after a lost replica. It does not prevent platform stops.
