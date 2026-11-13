# D25 native workload executions

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Both jobs Succeeded; all12 containers exit0/restarts0 and all96 H800 UUIDs verified. |
| 2026-10-22 | Sealed and submitted both H800 workload packages; verified actual platform identities and live launchers. |

## Source and preflight

- Source `e998210d1ba75a30c88f66a1c6c9f278c4c8853e`, clean build checkout `/data/ycfeng/Frontier/.worktrees/step5-d25-native-e998210`.
- Package root `/data/ycfeng/tmp/step5_d25_workload_20261022/`. Exact builder commands/results: `stage_receipts.json`; real StepMind Python dry-run/ResourceConfig construction/native-pin checks: `parent_preflight.json`.
- Each package verifies37 staged file pins and50 exported native source pins; all Python sources compile and submit wrappers pass `bash -n`.
- Both use `workload`, clean native execution, one API frontend per role, three repetitions of16 warmups and1286144/1024 requests atQPS1, unchanged frozen workload bytes and graph/scheduler configuration. API/DP/KV/proxy ports are disjoint between jobs; all12 observed worker IPs are distinct.
- Image: `a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`.

| Scale | Package | Config SHA256 | Job |
| --- | --- | --- | --- |
| H80032 | `h800_32_workload01/stage` | `d9424e11d3da6e2a5189af35848001a8071d2a25b644073208e95682fe0d8b00` | `exp-1022-024130-721461` |
| H80064 | `h800_64_workload01/stage` | `ac80f263f803bd5fbf43610d9232e1c99757bb71c38ac87a13945fd5a1ae046d` | `exp-1022-024130-735731` |

## Submission and platform evidence

Each was submitted once locally through its sealed `submit.sh`, personal StepMind Python `spawn_tasks`/`RJobBackend`, using `umask077` and `setsid`. Exact receipts are in each stage's `launcher/{submission_attempt,submitted,payload}.json`; `launcher.log` is beside the stage.

```sh
umask 077
exec setsid bash /data/ycfeng/tmp/step5_d25_workload_20261022/h800_32_workload01/stage/submit.sh > /data/ycfeng/tmp/step5_d25_workload_20261022/h800_32_workload01/launcher.log 2>&1
```

The64-GPU command is identical with `h800_64_workload01`. Both actual submissions occurred at2026-10-21T18:41:30Z (local2026-10-22 02:41:30).

`parent_identity01.json` beside each stage records platform reads at18:44:47Z:

- creator `i-fengyicheng`, group `codesign`, preemptible=`no`;
-4/8 replicas ×8 GPUs,32CPU and125GiB per node, both native RDMA limits retained, expected image exact;
- current local NFS server `100.96.128.195`, source/target equal to the corresponding package workspace;
- both overall Starting with individual workers Starting/Running; worker runtime and workload completion remain pending;
- actual persistent launcher PIDs839408 (32) and839406 (64), PPID1. `setsid` returned in the tool shell after forking; that return was not job completion. Separate `tail --pid=<PID> -f /dev/null` notifications wait for the real launchers to exit.

Execution caps:90minutes each, at most48/96GPU-hours. Durable cloud output roots:

- `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d25_h800_32_workload01`
- `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d25_h800_64_workload01`

## Terminal acceptance

Both jobs Succeeded:32 at19:10:05Z,64 at19:11:19Z. Primary `parent_terminal01.json` reads verify all12 containers exit0/restarts0 and exact image digest. `native_receipt.json` verifies all96 H800 UUIDs distinct, complete P/D lane inventory, clean worker observation receipts and workload returncode0. Container runtime totals12.966667/26.564444GPU-hours. Actual source-hash checks precede worker rendezvous and all staged tool hashes still match.

Both request populations pass primary acceptance: each432/432 complete, six strict boundaries, complete metrics and no measured TTFT above5s. The64 independent audit is complete and agrees with primary recomputation. Prompt regeneration and vocabulary bounds pass for both scales. Native readiness/profile/accuracy admission remains UNVERIFIED.
