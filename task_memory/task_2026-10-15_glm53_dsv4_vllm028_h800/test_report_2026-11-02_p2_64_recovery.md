# P2-64 generation 02: startup repair and package verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | Accepted R5 source binding and legacy control; recorded generation02 on eight distinct Running nodes. |
| 2026-11-02 | Recorded generation02 submission and the lead's reproducible R5 call-chain failure; native execution and reader correction proceed in parallel. |
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
| Independent final package review | Workload, identities, overlays and resources agree | All source/package/workload/resource checks PASS; old R4 rejects the new approved source identity | Package PASS; R5 code/control subsequently accepted below |
| Formal GPU reference | Complete three-repeat native output and post-run admission | Generation02 exp-1102-233941-576872 submitted15:39:41Z;8replicas Running at15:58:41Z, eight distinct nodes verified15:58:59Z | IN PROGRESS |

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

CPU source extraction uses fixture configuration/model/NUMA wrappers and real sockets/TCPStore/spawn/Gloo/all-reduce; it does not validate NCCL, remote HELLO or full-model serving. The formal GPU run provides that evidence. The locked image check uses zero GPUs and has its own platform/terminal evidence. Resource limits remain64H200/8x8/all preemptible/step_main/timeout3600/ceiling64GPUh. The new run uses dsv4-h200-p2-64-formal-20261102-02; the issued promotion receipt is retained. Complete terminal/copy evidence precedes pair coverage, R5, all-request gates and warm readings; the completed Frontier replay is reused.

Generation01 has no request results and remains failed. Its seven missing finish clocks keep exact GPU-hours unknown; the historical35-job total remains158.27166666666665GPUh. Observed-runtime estimates are not actual completed accounting.

## Independent review disposition

Pi `p64-package-review-1102j` confirmed a real post-run reader incompatibility: frozen R4 inherits the base2cf/singleC26-only source predicate. Its overall review remains FAILED in `package_review.json`; the lead accepts its package/source/workload/resource checks and assigns the exact-source R5 successor plus comparison reader selection to `p64-reader-1102m`. D-P2-64-STARTUP-RECOVERY already authorizes this ordinary debug correction. R4 bytes and all numerical criteria remain unchanged. The post-run reader correction proceeds alongside the authorized GPU run; it does not cancel or block healthy native work.

The pinned-image job completed at2026-11-02T15:30:32Z. Full pod stdout contains all expected markers; the six base hashes equal packaged expectations. Lead independently checked all16engine ranks/32engine groups/32worker ranks and source hashes against package02. The executor final prose incorrectly called the regression source base2cf; the hashes bind corrected sourcea06a72a4. Job/replica status are Succeeded/SUCCEED; personal pod-object reads were403, so container exitcode is unavailable. This was0GPU. Evidence: image_cpu_result.json, image_cpu.log, image_cpu_terminal_receipt.json, image_cpu_lead_verification.json and review_disposition.json in the recovery evidence directory.

One fresh exact-job/eight-replica clock inquiry returned all objects without API errors, but the seven stopped replicas still expose running-state records without finishedAt. Exact generation01 total remainsnull; only the failed head's1824s×8/3600=4.053333333333334GPUh is exact. The complete34.01883068666667 observation estimate is preserved separately. See generation01_clock_followup.json.

## Terminal clock completeness correction

The existing `h200_mainline/terminal_receipt.py` summed missing durations as zero. The observed seven missing generation01 clocks make that state reachable and would understate total usage. It now emits `container_gpu_hours: null` until every expected replica/container has complete timestamps; `known_container_gpu_hours` retains only measured pairs and `container_clocks_complete` makes the distinction explicit. A terminated container with a missing timestamp is also kept unpriced. Platform/native acceptance rules are unchanged.

Verification: `PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/glm53_dsv4/h200/check_terminal_clock_completeness.py` (Python3.10.6, fake client and fake auth, no platform requests). Complete two-replica clocks yield16GPUh; missing replica, running container and missing end timestamp each yield exactnull/known8GPUh. **4/4 PASS**, receipt `terminal_clock_checks.json` in the recovery evidence directory.

The package commit's whitespace check reports three inherited trailing-space lines in verbatim `native_config_parallel.py` (228,271,274). These source bytes must match native commit and remain unchanged; authored files pass the scoped check. Pi's first submission executor was stopped before any GPU launch after a quota-query environment error; the wrapper now explicitly sets `KUBEBRAIN_CLUSTER_ENTRY`, `KUBEBRAIN_NAMESPACE` and workspace identity. `bash -n` passes. Replacement executor `p64-formal2-1102n` retains the single-submission contract.

## R5 first review: reachable call chain fails

Lead review imported the new reader with Python3.12.13 and called `native_rank_graph_evidence` on generation02's actual config, engine and manifest with the required eight-node geometry. Expected: invoke the approved source predicate and proceed to later evidence checks. Observed: `(False, "UNVERIFIED: NameError: name 'native_v4_source_binding' is not defined")`. All inherited `main`, `check` and graph functions retain the separate exec namespace; the installer removed its binding and replaced only the exported module binding. The direct source-binding tests did not reach this defect. This first reader review is **FAIL**, distinct from package/source admission. Evidence: `recovery_02_20261102/r5_initial_call_chain_failure.json`. Correction prompt: `/data/ycfeng/tmp/ds4_runs/prompts/p64-reader-1102m.turn2.md`; acceptance requires the actual inherited call chain and self-contained source fixtures. No native job was stopped or resubmitted.

## R5 correction accepted; full legacy control preserved

R5 now installs the exact approved source predicate into the exec namespace used by inherited `main`, `check`, `check_native_canonical` and `native_rank_graph_evidence`. The original base2cf/C26 predicate is retained. The new source binding requires exactly the four approved startup sources plus C26, with base/new source hashes, sealed/retained/prelaunch inputs, committed-source mapping and config checks. R2/R3/R4 bytes and every metric rule stay unchanged. `compare.py --reference-checker r5` selects the reader and records its digest; the default remains R4.

Pi `p64-reader-1102m` turn2 passed32new tests. Lead ended the implementation turn (exit130), removed the unnecessary recursive bytecode-inspection test code, limited fixture copies to sealed inputs, and independently ran the required checks below. Fixtures derive their prelaunch manifest from sealed case/config/source-lock inputs and materialize native bytes from the pinned Git objects, so they do not need launcher output or a native worktree's mutable file contents. Their declared prerequisite is a repository containing2cf0a691/a06a72a4; `FRONTIER_NATIVE_SOURCE_REPO` selects an alternate location, and missing objects fail clearly rather than skip.

Environment: `TMPDIR=/data/ycfeng/tmp`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=$PWD`, repository cwd; Python3.12.13 at `/data/ycfeng/envs/openmopd-py312/bin/python` for reader/tests and Python3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python` for comparison/control. No conda activation or GPU work in these checks.

```bash
/data/ycfeng/envs/openmopd-py312/bin/python -m pytest tests/unit/test_native_startup_source_binding.py tests/unit/test_native_reader_binding.py -q -p no:cacheprovider -k 'not test_balanced_strategy_actual_import_and_missing_identity'
# In h3o_operator_costs/formal_p2_64_native_20261102:
/data/ycfeng/envs/openmopd-py312/bin/python -m pytest test_tools.py -q -p no:cacheprovider
# From the repository root:
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_64_native_20261102/compare.py --control --reference-checker r5 --r4-python /data/ycfeng/envs/openmopd-py312/bin/python --output /data/ycfeng/tmp/glm53_dsv4/h200/r5_lead_control_20261102
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_64_native_20261102/verify_control.py --control /data/ycfeng/tmp/glm53_dsv4/h200/r5_lead_control_20261102 --output /data/ycfeng/tmp/glm53_dsv4/h200/r5_lead_control_verification.json
```

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Applicable reader regression | New binding/call-chain tests and existing unaffected reader tests pass | 153 passed,14 deselected in61.40s | PASS |
| Coverage/comparison tools | Existing25tests pass | 25 passed in3.79s | PASS |
| Actual generation02 graph-entry call | Approved source reaches later evidence check; forged commit rejected | Approved source reaches missing-client-phase rejection; forged identity rejected by R5 | PASS, source binding only |
| Complete old reference CLI via R5 | All old admission checks and numerical results preserved | TOOLS_FINISHED, readerPASS, errors[]; all-requestFAIL and expected NOT_COVERED retained | PASS |
| Independent legacy metric verification | Warm/empty readings and all12per-repeat metric rows equal original control | PASS_EXACT; all-requestFAIL/warmPASS preserved | PASS |

The14deselected tests already fail before these edits: both their test file and collector bytes equal HEAD, but their fixed expected collector hash e6a9025c differs from HEAD19f462f1. Evidence `r5_unchanged_reader_failure_scope.json` records both current/HEAD hashes; this work does not repair or waive that separate test debt. Retained evidence is in `recovery_02_20261102/r5_lead_*`; `r5_lead_acceptance.json` binds the accepted files. These checks accept the reader correction and preserve the old comparison. They do not admit or numerically qualify the new running reference.

## Continuation verification — 2026-11-03 Asia/Hong_Kong

Lead recomputed SHA-256 for all three accepted R5 files and matched `r5_lead_acceptance.json` exactly; retained test/control logs agree with the reported counts and PASS_EXACT result. No code changed after those checks. `generation02_running_nodes.json` records eight distinct nodes with all containers Running and restartCount0, captured2026-11-02T15:58:59Z. This is placement evidence; native full-model initialization and request completion remain pending.
