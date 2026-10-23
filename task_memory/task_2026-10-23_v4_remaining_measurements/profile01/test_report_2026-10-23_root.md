# Root native-profile package verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded initial profile01 image/hook checks, fresh-entry RED controls and preserved verification-harness errors. |
| 2026-10-23 | Rechecked repaired profile02 after both reviews and merge cc0227e9; final entry/image/four-position checks and 45-file seal passed. |
| 2026-10-23 | Recorded actual submission, platform resource/mount checks, queued follow-up and separate watch permission failure. |

## Scope and environment

Initial package source `69f92c55`, records `62092da3`, accepted runtime `05952396`. Package path is recorded in `initial_delivery.json` once archived by its owner; its basename is `candidate_h800_p0_profile_01` under the `v4-native-profile-run` worktree's `v4_p0_native_20261021/` directory. It remains unsealed and unsubmitted because its actual entry import fails. This report grants no GPU, native-reference, operator-cost or E2E qualification.

Host: `kun-workspace-vgen2`, local NFS IP `100.96.128.195`. Host verification interpreter: `/data/ycfeng/envs/frontier-py310/bin/python` (Python3.10.6); locked-image interpreter Python3.12.3. Image digest: `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`. No conda activation. Raw evidence below is under `/data/ycfeng/tmp/glm53_dsv4/`.

## Observed results

| Check | Expected condition | Observed result | Verdict |
| --- | --- | --- | --- |
| Initial image invocation | Explicit source/binary verifier JSON | Exit0 with empty stdout: Docker stdin was not enabled | FAIL, harness invocation; no image verdict |
| Corrected locked-image check | Explicit PASS with87source/23tool/20binary pins; unchanged45files | Exact expected counts, no torch/vLLM import or CUDA initialization, package unchanged | PASS for pins only |
| Four isolated actual-wrapper probes | Inject selected extension and install hooks, correct limits/paths, no CUDA | Four separate PASS witnesses;8profile hooks and7inventory hooks, empty error lists, four role/DP paths | PASS for isolated hooks, not actual entry |
| Fresh direct worker import | Import from package root without preloading dependencies | `ModuleNotFoundError: No module named 'vllm_replay'` | RED, package defect reproduced |
| Actual copied launcher | Reach manifest-call boundary without any run write | Same error at launcher30/worker19 | RED, package defect reproduced |
| First bootstrap callback adapter | Execute fresh child through actual bootstrap environment | `TypeError: inspect_launch() takes 2 positional arguments but 3 were given` | FAIL, root adapter; no entry verdict |
| Corrected bootstrap callback | Fresh child imports worker using generated environment | Same `vllm_replay` import error | RED, package defect reproduced |

The isolated probe imports `pd_replay` before the worker, which inserts the needed dependency path. Its successful hook results therefore do not settle launcher/bootstrap entry correctness. The fresh-process controls remove that masking dependency.

## Exact commands and retained evidence

Image verification adapts only the two expected count literals in the existing `/data/ycfeng/tmp/glm53_dsv4/verify_v4_prefill_image.py` in memory, preserving its original SHA-256 `55236cfa09d3463fc26a81120260903d5542c32565904a4a342860fa9068038d`. The exact Docker argv and executed `stdin.py` are retained in each directory:

- `profile01_image_2r56_agv/{receipt.json,verdict.json,stdin.py,stdout.log,stderr.log}`: rejected empty-output attempt.
- `profile01_image02_yrvh04pj/{receipt.json,stdin.py,stdout.log,stderr.log}`: corrected `docker run -i --rm --pull=never --network=none --cap-drop=ALL --cap-add=DAC_OVERRIDE --memory=4g --cpus=2 --pids-limit=128 -v <package>:<package>:ro --entrypoint python3 <locked-image> -I -B - <package>`; stdout contains the explicit PASS witness.
- `profile01_startup_41qkb0xr/{receipt.json,stdin.py,0.receipt.json,1.receipt.json,2.receipt.json,3.receipt.json}` plus per-position logs: four Docker commands with6GiB memory and the executed CPU stand-in probe. Actual native worker constructors, model hooks, forwards and graph replays were not executed.

The direct-import command and original traceback are in `profile01_entry_red_acbrv5ib/receipt.json` and `stderr.log`:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -I -B -c \
  'import sys; sys.path.insert(0,sys.argv[1]); import native_pd_worker' <profile01-package>
```

The real launcher control was:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B \
  /data/ycfeng/tmp/glm53_dsv4/profile_package_entry_probe.py <profile01-package>
```

`profile_entry_naaqrp6l/` retains the exact copied launcher argv, test-only manifest, `fixture/sitecustomize.py`, stdout/stderr and receipt. The only added import path contains the trace hook, not a dependency module. No dependency is preloaded. Authorization is absent as an additional stop before any real manifest write if the trace hook fails; no platform call or original-package seal occurs.

The real bootstrap-child control reuses the image-verifier callback with `/data/ycfeng/tmp/glm53_dsv4/profile_bootstrap_entry.py`; exact composed `stdin.py` and Docker argv are in `profile01_bootstrap_entry_red_4wcvq03i/` (adapter error) and `profile01_bootstrap_entry_red02_7mv1c4cq/` (actual entry RED). The latter uses the generated child environment and only models the script's own package-directory entry in `sys.path`; it does not add the missing dependency directory. Both attempts preserved all45package files.

## Platform and storage preparation

Read-only personal Python census: `profile01_quota_7hgcy67x/{quota.json,census.json}` at2026-10-23T09:06:17Z, Python3.10.6/brainpp2.8.6. The existing `read_quota.py` was executed with only its output destination changed in memory; source SHA-256 `19746066439f539ff889dd154cac6f32bffd5467a63f892e8dd87f3ca784f275`. Codesign non-preemptible use124/128GPUs; zero active personal jobs. This does not reserve capacity or prohibit a single queued submission. Fresh quota is required at submission.

The personal cloud parent `/mnt/ws-jfs/ycfeng/frontier/glm53_dsv4_vllm028/t18` was readable/traversable with mode0755, and child `dsv4_h800_p0_profile_20261023` did not exist. Read-only `ResourceConfig()` inspection confirmed `preemptible=False` and RDMA defaults `mellanox.com/mlnx_rdma:1`, `rdma/mlnx_shared:8`.

## Open acceptance

Source repair `d87f4c74` was accepted by both delta reviewers and merged at `cc0227e9ea0f7e173e1f37d7e302f943ff86707a`. Root then wrote `profile02_source_binding.json` and ran final checks against those immutable package bytes. Submission and actual retained-artifact verification remain pending; all runtime unknowns in the proposal remain open.

## Final profile02 verification

- Before the final run, both maintained builder/plan files matched the reviewed source commit; all 13 staged runtime modules matched `git show 05952396:<source-path>`, retained source/staged hashes matched, and all 45 package hashes matched delivery. The original profile01's 45 files were independently rehashed unchanged.
- Final receipt: `/data/ycfeng/tmp/glm53_dsv4/profile02_final_cc0227e9_8hf9_hy7/receipt.json`. Exact commands, timestamps, per-command stdout/stderr and executed image/hook scripts are alongside it. Root ran the actual StepMind Python interpreter, not a substitute environment, for the launcher control (`profile_entry_dv74yxw3/receipt.json`).
- Results are separate: actual launcher entry **PASS**; real locked-image bootstrap/fresh child **PASS** with exactly 87 source, 23 tool and 20 binary pins; positions 0, 1, 2 and 3 each **PASS** for isolated wrapper/profile hooks and configuration. No actual GPU constructor/model forward/replay was invoked. All 45 package files were unchanged after the checks.
- The intermediate `profile02_image_entry_hvboefyj` run proved the fresh child import but failed the old verifier's one-directory PYTHONPATH assertion. The template expectation was corrected in memory to require exactly package root plus the staged DP directory, preserving cwd and every source/binary check. Original failure and executed scripts are retained; the accepted final result uses this corrected expectation.
- Root invoked the unchanged staged `seal.py`, then `launch.py --check-only`, with `/data/ycfeng/tmp/stepmind-env/bin/python -B`. Seal contains exactly the 45 verified inputs; manifest SHA-256 `51e39a42ec863e496f20be8a165230e0dee17c34906ad6bd4fade14bf898ee91`. Check-only printed `INPUT_HASHES_PASS_NO_SUBMISSION`. `profile02_root_readiness.json` records **SEALED_READY_DIAGNOSTIC**, not GPU/reference admission.
- Fresh personal quota census: `/data/ycfeng/tmp/glm53_dsv4/profile02_quota_4fw5fy06/census.json`, 2026-10-23T10:02:36Z, zero active personal jobs; codesign non-preemptible use 126/128 GPUs. Fresh read-only destination proof: `profile02_destination_rotlq0zx/receipt.json`, personal profile root absent and parent mode 0755. These are snapshots, not capacity reservations.

## Submission and platform observations

The unchanged staged launcher was executed as `/data/ycfeng/tmp/stepmind-env/bin/python -u -B <profile02-package>/launch.py`, after the documented company-proxy/personal-auth initialization, `STEPMIND_BACKEND=rjob` and `PYTHONPATH=/data/ycfeng/steptron`. Authorization text cites D-V4-H800-RJOB-AUTHORITY; credential values are not retained. Exact command and prelaunch predicates are in `profile02_root_submission.json`; log is `/data/ycfeng/tmp/glm53_dsv4/profile02_launch_20261023.log`. The script uses existing `spawn_tasks()`/Python RJobBackend and mounts the local package directory.

Created job **exp-1023-180821-373637** at **2026-10-23T10:08:21Z**. Root's exact-job personal Python query at10:10:16Z verified:

- Creator `i-fengyicheng`, charged group `codesign`, preemptible label `no`.
- Four replicas, each requesting 8 GPUs and 64 CPUs; actual platform memory limit **500Gi**, corresponding to the unchanged SDK input `mem_gb=512`.
- Locked image; RDMA limits `mellanox.com/mlnx_rdma=1`, `rdma/mlnx_shared=8`.
- NFS source/target `100.96.128.195:<profile02-package>:<profile02-package>`.
- Status **Pending**, reason/message `RJob is queuing`, no replicas yet; launcher PID3236764 alive.

Receipt: `profile02_platform_launch.json`. Follow-up at10:22:53Z remained Pending with no replicas and the same launcher alive (`/data/ycfeng/tmp/glm53_dsv4/profile02_platform_72do4e1w/receipt.json`). Worker GPU identity, native startup, business completion, profile coverage and terminal state are therefore **not yet observed**.

A separate Kubernetes watch request received403 Forbidden at10:11:53Z. Exact retained error and classification are `profile02_watch_error.{log,json}`. Snapshot get/list access worked before and after this failure. The watcher performed no job/launcher control, and no duplicate job was submitted. It is an auxiliary monitoring limitation, not a workload failure.
