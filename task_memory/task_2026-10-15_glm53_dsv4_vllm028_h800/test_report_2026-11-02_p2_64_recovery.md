# P2-64 generation 02: startup repair and package verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | Recorded the approved native startup repair, source/CPU checks, frozen harness and sealed generation-02 package. |

## Scope

D-P2-64-STARTUP-RECOVERY approves the startup-only native correction and a second formal run. Native commit `a06a72a4326cba02234df6cfbdc21b78d3a15d26` is based on `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Four startup files retain a coordinator-owned TCPStore, propagate its port to remote engines via HELLO, publish distinct engine group rendezvous keys and reuse the store for the formal `nnodes=1` worker world. The socket-backed store explicitly disables libuv. Model, kernel, routing, request and timing code are unchanged.

Frozen harness: `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness`, `f33306d40443cfc0c44842d6ee044f6703c736e7`. The approved exact source SHA is added to the existing v0.28 source allowlist; arbitrary SHAs remain rejected. The older harness stays at `4861e680`.

## Verification and evidence

Commands run from `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` unless stated. Direct Python is `/data/ycfeng/envs/openmopd-py312/bin/python` (3.12.13, torch2.8.0+cu128), without conda activation. `python` is absent from the shell; `python3` is available.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| CPU native startup | DP2/DP4 with occupied/free legacy port; distinct engine groups; real worker world sums; independent P/D groups; listener released | Six cases, 16 engine ranks, 32 engine initializations and 32 worker ranks; all expected sums and listener checks pass | PASS |
| Independent native source review | No actionable defect on the formal nonelastic mp/nnodes1 path | Pi p64-native-review-1102h settled, exit0; lead accepted with precise local-engine/HELLO clarification | PASS, source scope |
| Frozen harness tests | All applicable tests pass | 120 passed,21 skipped in52.92s; exact source allowlist control passes | PASS within CPU scope |
| Formal recovery package01 | Source identity agrees before run | FAIL: builder retained old engine native_reference SHA; no launch | FAIL, preserved |
| Formal recovery package02 | Seal, source identity, preflight, topology, launch plan and check-only pass | 47 sealed files; TP8/DP4/EP32/PP1 per role; INPUT_HASHES_PASS_NO_SUBMISSION | PASS |
| Independent lead check-only | Same sealed manifest passes | exit0; manifest 9f9a1ab954ef3b7d61ff67f2f62c891e9da8547590a21b19db020c8557e8ed7d | PASS |
| Locked image CPU check | Actual wheel base hashes and torch runtime communication pass | exp-1102-232618-248194 Succeeded; actual six base hashes and all6 CPU cases PASS with Python3.12.3/torch2.13.0+cu130 | PASS |
| Independent final package review | Workload, identities, overlays and resources agree | All source/package/workload/resource checks PASS; old R4 rejects the new approved source identity | Package PASS; reader repair pending |
| Formal GPU reference | Complete three-repeat native output and post-run admission | Not submitted yet | NOT RUN |

CPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 300s /data/ycfeng/envs/openmopd-py312/bin/python tests/comparison/calibration/check_native_dp_startup.py --native-root /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/p2-dp-startup-coordination --output /data/ycfeng/tmp/glm53_dsv4/h200/p2_dp_fix_cpu_02
```

Build command (PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=$PWD):

```bash
/data/ycfeng/envs/openmopd-py312/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/build_package.py p2_64_formal_recovery --package package_02 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness
```

See `test_report_2026-11-02_startup_harness.md` for the exact harness test command. The independent check-only argv and result are retained in `h200_mainline/p2_64_formal/recovery_02_20261102/lead_check_only.json`.

All selected evidence is under `h200_mainline/p2_64_formal/recovery_02_20261102/`: native.patch, native_commit.txt, cpu_result.json, native_review.txt, review_receipt.json, harness_tests.log, build_01_failure.log, build_02.log and lead_check_only.json. Failed package01 remains immutable. The source change and its native repository task records are preserved by the native commit.

## Limits and next actions

CPU source extraction uses fixture configuration/model/NUMA wrappers and real sockets/TCPStore/spawn/Gloo/all-reduce; it does not validate NCCL, remote HELLO or full-model serving. The formal GPU run provides that evidence. The locked image check uses zero GPUs and has its own platform/terminal evidence. Resource limits remain64H200/8x8/all preemptible/step_main/timeout3600/ceiling64GPUh. The new run uses dsv4-h200-p2-64-formal-20261102-02; the issued promotion receipt is retained. Complete terminal/copy evidence precedes pair coverage, R4, all-request gates and warm readings; the completed Frontier replay is reused.

Generation01 has no request results and remains failed. Its seven missing finish clocks keep exact GPU-hours unknown; the historical35-job total remains158.27166666666665GPUh. Observed-runtime estimates are not actual completed accounting.

## Independent review disposition

Pi `p64-package-review-1102j` confirmed a real post-run reader incompatibility: frozen R4 inherits the base2cf/singleC26-only source predicate. Its overall review remains FAILED in `package_review.json`; the lead accepts its package/source/workload/resource checks and assigns the exact-source R5 successor plus comparison reader selection to `p64-reader-1102m`. D-P2-64-STARTUP-RECOVERY already authorizes this ordinary debug correction. R4 bytes and all numerical criteria remain unchanged. The post-run reader correction proceeds alongside the authorized GPU run; it does not cancel or block healthy native work.

The pinned-image job completed at2026-11-02T15:30:32Z. Full pod stdout contains all expected markers; the six base hashes equal packaged expectations. Lead independently checked all16engine ranks/32engine groups/32worker ranks and source hashes against package02. The executor final prose incorrectly called the regression source base2cf; the hashes bind corrected sourcea06a72a4. Job/replica status are Succeeded/SUCCEED; personal pod-object reads were403, so container exitcode is unavailable. This was0GPU. Evidence: image_cpu_result.json, image_cpu.log, image_cpu_terminal_receipt.json, image_cpu_lead_verification.json and review_disposition.json in the recovery evidence directory.

One fresh exact-job/eight-replica clock inquiry returned all objects without API errors, but the seven stopped replicas still expose running-state records without finishedAt. Exact generation01 total remainsnull; only the failed head's1824s×8/3600=4.053333333333334GPUh is exact. The complete34.01883068666667 observation estimate is preserved separately. See generation01_clock_followup.json.
