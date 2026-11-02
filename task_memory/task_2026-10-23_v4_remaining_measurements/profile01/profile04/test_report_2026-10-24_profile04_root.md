# Profile04 primary verification and submission

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Primary accepted run-data and corrected metadata lanes after independent review and byte-identical P2 replay; full trace/profile/cost gates remain open. |
| 2026-10-24 | Recorded independent193-check/all32metadata CPU replay, semantic reviewP2 and24complete/8timeout trace boundary without final data admission. |
| 2026-10-24 | RecordedSucceeded4/4/exit0/restart0, immutable202-file download/45-host-input/40-runtime-manifest bindings and opened local data audits. |
| 2026-10-24 | Archived primary CPU/image/source checks, the corrected launch environment, and a queued platform snapshot. |

## Scope and expected result

Validate the existing `torch_profiler_dump_cuda_time_total=false` configuration without changing runtime1437ace6, native/C26/workload bytes, P512/D64 collection limits or the native300-second RPC budget. Submit at most one4×8H800/codesign non-preemptible validation under D-V4-H800-RJOB-AUTHORITY. Platform submission and CPU checks do not establish runtime, serving or cost acceptance.

## Commands and observed validation

Commands ran from the exact worktrees recorded in the linked receipts. No conda environment was activated. Local pytest used `/data/ycfeng/envs/asyncopd-py312/bin/python`; sandboxed image checks used Python3.12.3, and personal platform reads used `/data/ycfeng/tmp/stepmind-env/bin/python`3.10.6.

| Check | Expected | Actual | Verdict and receipt |
| --- | --- | --- | --- |
| Required worker/manifest regressions | No failures/errors/skips | 55tests,0failures,0errors,0skips | PASS: `/data/ycfeng/tmp/glm53_dsv4/profile04_root_tests_g2vtrw1t/receipt.json` |
| Locked-image entry and four node plans | Five explicit PASS witnesses; unchanged inputs | All five PASS; P512/D64, dump=false,600000/268435456 limits | PASS: `/data/ycfeng/tmp/glm53_dsv4/profile04_root_image_egwpqxbz/receipt.json` |
| Actual SDK resource constructor | Four8GPU/64CPU/512mem_gb non-preemptible replicas, existing RDMA/mounts | Requested expression preserved; no platform call in this check | PASS: `/data/ycfeng/tmp/glm53_dsv4/profile04_root_sdk_09unc5v7/receipt.json` |
| Fixed source/Git/package binding | 45inputs and13runtime identities unchanged | 153checks PASS | PASS: `/data/ycfeng/tmp/glm53_dsv4/profile04_root_final_0esz0v_h/receipt.json` |
| Final offline metadata and native-statistics controls | Reproduce their limited accepted outputs | Exact equality; original failed-run limitations retained | PASS: [failure acceptance](../profile03/failure/primary_acceptance.json) |
| Post-submission seal binding | Original seal,45inputs and all11bound receipts unchanged | All hashes match | PASS: primary inline hash check, recorded in this session and [submission archive verification](root_archive_verification.json) |
| Personal platform identity/requested shape | One job, personal creator,codesign,no preemption | exp-1024-172225-512616; four8GPU/64CPU/500Gi replicas; RDMA limits1/8; exact local NFS | PASS for submission only: [root_run_check.json](root_run_check.json) |
| Actual runtime/data | Four hosts/32H800 identities;432successful requests; complete formal metadata, coverage, DP/stream/post-stop evidence | Starting; four replicas, zero assigned nodes at09:49:36Z | PENDING, not PASS |

The exact pytest argv, sanitized environment, source hashes and XML are in the tests receipt. Its executable command is `sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s runuser -u brainpp -- env -i ... /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_pd_worker.py tests/unit/test_native_pd_manifest.py -q -p no:cacheprovider --basetemp <receipt-directory>/pytest --junitxml <receipt-directory>/tests.xml`.

The five Docker argv lists are in the image receipt. They retain `--pull=never --network=none --cap-drop=ALL --cap-add=DAC_OVERRIDE --memory=6g --cpus=2 --pids-limit=128`, a read-only package bind mount and `--entrypoint python3 <locked-image> -I -B -`. Only the CPU verification sandbox sets `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`; no such package/runtime correction was made. Reused stdin scripts and their hashes are retained beside that receipt.

Source/SDK/native-statistics and metadata controls retain exact argv or stdin bindings in their own receipts. They are not GPU performance measurements. The unchanged77-case matrix is a previously captured source/domain binding, not a new matrix execution. Both package reviews returned No findings; [review.md](review.md) records their static-only limits.

## Preserved failures and corrected launch

1. The first pytest environment imported an editable `verl.tests` instead of Frontier's namespace. Preserved receipt `/data/ycfeng/tmp/glm53_dsv4/profile04_root_tests_1gqiv68z/receipt.json`; the owner-verified asyncopd environment passed without any source/fixture change.
2. The first four image position checks exhausted128PIDs while importing CPU libraries. Original receipt `/data/ycfeng/tmp/glm53_dsv4/profile04_root_image_58u02yq7/receipt.json`; direct cgroup/thread evidence `/data/ycfeng/tmp/glm53_dsv4/profile04_thread_probe_gd64onn_/receipt.json`. Sandbox-only thread limits resolved it with all45package hashes unchanged.
3. Local launch1 failed at `launch.py:32`, `os.environ['AUTHORIZATION']`, before `write_run_manifest` or any platform submission. Its original receipt remains byte-identical in [root_launch_01_presubmission_failure.json](root_launch_01_presubmission_failure.json); traceback `/data/ycfeng/tmp/glm53_dsv4/profile04_launch_20261024.log`.
4. Local launch2 exported the existing `AUTHORIZATION=D-V4-H800-RJOB-AUTHORITY`, refreshed capacity/destination reads, reverified the original seal, then executed `/data/ycfeng/tmp/stepmind-env/bin/python -u -B <sealed-package>/launch.py` with `STEPMIND_BACKEND=rjob` and `PYTHONPATH=/data/ycfeng/steptron` after the handbook proxy bootstrap. Exact command/environment and prior failure hash are in [root_submission.json](root_submission.json). Fresh prelaunch receipt `/data/ycfeng/tmp/glm53_dsv4/profile04_prelaunch_env_omcv47c3/receipt.json`; actual launcher log `/data/ycfeng/tmp/glm53_dsv4/profile04_launch_env_20261024.log` records exactly one Job Create at17:22:25.676+08:00.

Platform reads used the personal Python client, exact job `exp-1024-172225-512616`, `get_namespaced_custom_object(...,_request_timeout=25)`, `get_rjob_infos(name)` and each known replica's `_get_replica_kapis(name)`. The09:49:36Z snapshot performed zero submissions/control operations. The submitted template still reports preferred anti-affinity rather than the configured required form; preserve this discrepancy and verify actual placement after scheduling.

## Remaining acceptance

Keep launcherPID3950111 alive and inspect only this job. After scheduling verify four distinct hosts,32H800 UUIDs and actual RDMA transport. On completion inspect all432client rows, raw traces, published formal metadata, at least three useful kernel-bearing steps per role/lane, raw DP counters without offset correction, stream correlation and post-stop evidence. The optional text tables are intentionally absent. Synchronous raw export can still exceed300seconds; a failure ends this attempt without automatic resubmission. Captured per-operator ownership and all serving/operator-cost admission remain independent and unverified. Profile03 remains FAILED with its original evidence unchanged.

## Terminal/input checks (2026-10-24)

The preceding pending statements are the09:49:36Z submission snapshot. The launcher now reports `FINAL_STATUS succeeded`; [terminal_platform.json](results/terminal_platform.json) records platformSucceeded/fourCompleted/exit0/restart0. Actual placement is read fromstatus.nodeName (0030/0158/0187/0574), not absent spec.nodeName. Installed-clientlabel_quotagroup resolves `quotagroup.brainpp.cn/quotagroup`; [platform_group_binding.json](results/platform_group_binding.json) confirms codesign, personal creator andpreemptible=no. Both inspections are read-only; first receipt's wrong-key null is preserved. Expected/actual terminal and identity checks PASS; actual NET/IB, GPUUUIDs and data checks remain pending.

Download command: `sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- /usr/bin/python3 -B /data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/download.py`. The retained script adapts only the prior inventory root, reads via personalSSH/sudo, streams tar into a fresh local directory and rehashes every file. Expected202files/allsize-and-hash equality; actual202files/6024578382bytes andboth archive exits0: PASS. [Receipt](results/download_receipt.json) pins exact inventory/archive/extract argv, scriptc6887e78...9f76 andinventory85b0ef23...7c8c. No raw artifact is overwritten or uploaded.

Primary inline Python rehashed the45sealed host inputs againstmanifest25bc61be...d58f and the40run_manifest input entries against either downloadedcase-relative files or exact absolute package files: allPASS ([host_package_check.json](results/host_package_check.json), [download_input_check.json](results/download_input_check.json)). The four generation/role/lane joins match, diagnostic_only=true/reference_admission=false, and32trace/32profile-stop files exist. Content checks are separate. A first harness attempt incorrectly read result.json.status and raisedKeyError; [preserved failure](results/first_input_check_failure.json) and inspectedqualification_verified/observation_errors replace that mistaken lookup, not the native behavior. Local run, metadata and trace owners are now auditing disjoint files; independent review/reproduction and limited data acceptance remain unfinished.

## Independent data replay and current limits (2026-10-24)

**Current limited decision:** [run/metadata acceptance](results/run_metadata_acceptance.json) and [review](results/review_2026-10-24_run_metadata.md) accept R3 retained run-data and R4 metadata-lane evidence after the independent P2 re-review and corrected primary replay. The pre-fix replay below remains history, not the current metadata verdict. R5 is still incomplete24/32; actual kernel streams, captured ownership, whole DP alignment, natural child exit and cost/reference admission are not accepted.

Primary ran `sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- /usr/bin/python3 -B /data/ycfeng/tmp/glm53_dsv4/profile04_root_data_o2_qhawu/verify.py`. The script, per-case exact commands/stdout/stderr and [receipt](results/root_cpu_replay_01.json) retain Python3.10.6/pandas2.0.1,193checks/99bindings PASS, expected token-count negative exit1,32metadata outputs byte-equal24b45c44..., and unchanged selected sources/results. Metadata workers retain60seconds/30G/concurrency2; coordinator has no aggregate timeout. A byte-identical coordinator copy redirects only fresh harness-log paths, not logic. R3owner's original bare/user-systemd permission failures remain preserved; primary's actualsudo-systemd commands both executed and matched expected results.

The independent review found one semanticP2: old `pair()` populated `unpublished_tmp_*` partitions from formal snapshots although this run has no temporary files. The mechanical byte match does not close that defect. R4owner archives/corrects only derived metadata and adds semantic controls; formal DP observations must remain unchanged. R3has no actionable finding, and its inspected432successful clients/fourhosts/32H800UUIDs support only limited diagnostic evidence. Thirteen childSIGKILLs remain observed native escalation, not all-natural shutdown; actualNET/IB/Socket and workerbinary rehash remainUNKNOWN.

R5completed24ranks, with at least3useful kernel-bearing steps on P1/D0/D1; all8P0ranks hit60seconds/exit124. Full32coverage isINCOMPLETE. See [trace report](results/test_report_2026-10-24_trace.md). Sparse CPU stage diagnosis is separately allocated before any audit-algorithm correction; no increased limit or GPUretry. Streams/dependencies/captured per-op ownership and any reference/cost admission remain separate and unverified.
