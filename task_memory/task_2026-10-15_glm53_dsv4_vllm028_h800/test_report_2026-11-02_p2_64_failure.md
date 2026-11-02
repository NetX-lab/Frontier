# Formal P2-64 first-generation startup failure

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | Accepted exact-job RCA and independent CPU reproduction; retained missing finish clocks, mixed coordination-control results and pending recovery approval. |
| 2026-11-02 | Recorded confirmed failure, independently verified failed-output copy, and pending RCA/accounting verification. |

## Scope and result

Ticket 18/19, D-P2-64-FORMAL-B. `exp-1102-164536-160317` became **Failed** at 2026-11-02T12:50:51Z, before client request replay. The frozen first-generation package, harness and native source are preserved. No second GPU job, native patch, timeout increase, Frontier replay or metric comparison was performed during this investigation.

The first failing replica was `exp-1102-164536-160317-a87ff679`, UID `f189604c-43df-4984-bda3-ed133c8912b2`, on `gpu-h200-0760.lgcm.sh.istep.fun` (decode DP0/API head). Its container ran from 12:20:25Z to 12:50:49Z and exited 1. The job controller subsequently stopped the other seven replicas. The launcher cleanup at 12:50:53.024Z follows the recorded failure and is not its initiating cause.

## Identities and environment

- Worktree: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, code HEAD `e2289283`.
- Package: `h200_mainline/p2_64_formal/package_02`; manifest SHA256 `c22085987055d018efb8abc104081cfe56d001b22026bb3aa367564d46754cd7`.
- Frozen harness: `/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup`, commit `4861e680548a5666f3107945ec4881b7022c23fb`.
- Frozen native source: `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`, commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
- Runtime versions recorded by the decode head: vLLM 0.28.0, torch 2.13.0+cu130, aiohttp 3.14.3; its `nvidia_smi` records eight NVIDIA H200 GPUs.
- Platform operations: Pi `/ds4-exec`, personal StepMind Python client; RCA label `p64-formal-rca-1102d`, CPU-copy label `p64-formal-copy-1102e`, local CPU reproduction label `p64-dp-port-repro-1102f`.

## Failed-output recovery — PASS

The unchanged `h200_mainline/cloud_copy.py submit` ran once using `package_02/config.json`, copy directory `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261102_01`, and the cloud root:

```text
/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h200_p2_64_formal_20261102_01/case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261102-01
```

The 0-GPU copy job `exp-1102-222827-251818` reported `FINAL_STATUS succeeded`; launcher exit was 0 at 2026-11-02T14:28:47Z. It used the existing CPU16/64GB copy configuration. The independent lead verification rehashed every file with Python 3, checked sizes, the complete relative-path set, and the SHA256 of `cloud_inventory.json` against `download_receipt.json`.

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Download status | HASH_IDENTICAL_DOWNLOAD | HASH_IDENTICAL_DOWNLOAD | PASS |
| File hashes and sizes | Every inventory row matches | 74/74, 29,702,405 bytes | PASS |
| Missing or additional files | Zero | 0 / 0 | PASS |
| Inventory SHA256 | Match receipt | `00cd2f97bcc5ebd1288a520dd4e86e51ab75b9416a8cf480f4594bf3bd24411a` | PASS |
| Run identity | Formal generation 01, frozen source/harness | Exact match | PASS |
| Complete formal request data | Three repetitions | No request output; only startup artifacts | FAIL |

Local full artifacts are in the copy directory under `dsv4-h200-p2-64-formal-20261102-01/artifacts`. The root contains only `run_manifest.json`; node records contain startup/serving instrumentation but no client warmup/formal results. All eight node results retain `qualification_verified=false`. These are failed-run artifacts, not an admitted reference.

The initial detached shell command returned an error while reading a relative log path, but the detached launcher had submitted and completed successfully. No duplicate copy job was submitted. The copy executor then stalled for over six minutes in an unbounded recursive source search. The lead stopped that Pi turn (exit 130) after independently verifying the complete copy and the launcher's terminal success; no GPU job was stopped.

## Startup evidence

The retained decode-head `server.log` records `Address already in use. Retrying with a new port.` from `parallel.py:655` at 12:22:14Z. Peer logs show TCPStore errors to `10.204.22.14:40273`, including `Failed to recv, got 0 bytes` and an invalid ping response. The head's worker then raises `TimeoutError: Native P/D rendezvous or workload deadline exceeded` at `native_pd_worker.py:345`, while waiting for native readiness under the existing 1800 s startup deadline.

This establishes the startup failure and a preceding rendezvous fault. The original owner of port 40273 and the head's replacement port are not captured in the worker logs. The CPU reproduction below verifies that the frozen nonelastic DP path can diverge after a rank-0 port conflict. This is a reachable source defect consistent with the observed startup failure. The missing original port occupant and replacement-port observation prevent claiming a complete reconstruction of the GPU incident or a confirmed infrastructure/preemption cause.

## Accounting and acceptance limits

The accepted RCA has actual container start clocks for 8/8 replicas and finish clocks for only 1/8. The exact job total remains `null` / `INCOMPLETE_CLOCKS`; the seven missing ends are not zero. The terminal receipt's `4.053333333333334` GPUh is only the failed head container's exact subtotal, not the job total.

For an explicitly labeled observation estimate, use each actual start; use the failed head's exact finish and each other replica's last worker-log timestamp. Summing duration × 8 GPUs / 3600 gives **34.01883068666667 observed GPUh**. This is neither complete billing nor a rigorous upper/lower bound. `rca_verification.json` retains every row and the lead's independent recomputation.

The historical 35 complete jobs remain **158.27166666666665 exact container GPUh**. Adding the new estimate gives **192.2904973533333 estimated cumulative GPUh**; it must not replace the exact ledger total. The successful copy job consumed zero GPUh. Ticket 19's complete-accounting item remains unchecked.

RCA review rejected the first report's log-to-log total, premature infrastructure classification and unsupported fixed-port/shorter-deadline/new-GPU-probe suggestions. Pi turn 2 corrected these. Its retained JSON is executor evidence: references to a TCPStore at the contested address do not establish who actually owned it. For estimation endpoints, the row-wise lead verification is authoritative; the failed head uses `finishedAt`, not its last log.

Pair coverage, R4, all-request gate and warm reading are **NOT RUN** for this failed reference because no client request output exists. No predicted/actual latency or relative error is available. Previously accepted code tests and the unchanged formal P2-64 Frontier replay were not reopened.

## Durable evidence

- `h200_mainline/p2_64_formal/failure_01_20261102/download_verification.json`
- `h200_mainline/p2_64_formal/failure_01_20261102/download_receipt.json`
- `h200_mainline/p2_64_formal/failure_01_20261102/cloud_inventory.json`
- `h200_mainline/p2_64_formal/failure_01_20261102/startup_log_evidence.json`
- `h200_mainline/p2_64_formal/failure_01_20261102/selected_artifacts/`
- `h200_mainline/p2_64_formal/failure_01_20261102/launcher_observation.json`
- `h200_mainline/p2_64_formal/failure_01_20261102/rca/` and `rca_verification.json`
- `h200_mainline/p2_64_formal/failure_01_20261102/cpu_reproduction/`

## CPU mechanism reproduction — PASS; production repair — NOT RUN

Executor `p64-dp-port-repro-1102f` used real torch TCPStore/Gloo on loopback with source-extracted methods from frozen native commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`. The lead inspected the extraction and call wrappers, then copied the two scripts unchanged to `/data/ycfeng/tmp/glm53_dsv4/h200/p2_64_failure_lead_20261102/` and ran the following commands there:

```bash
timeout 60s /data/ycfeng/envs/openmopd-py312/bin/python /data/ycfeng/tmp/glm53_dsv4/h200/p2_64_failure_lead_20261102/repro_dp_port_collision.py --out /data/ycfeng/tmp/glm53_dsv4/h200/p2_64_failure_lead_20261102/baseline --ranks 4 --timeout 8 --no-squatter
timeout 60s /data/ycfeng/envs/openmopd-py312/bin/python /data/ycfeng/tmp/glm53_dsv4/h200/p2_64_failure_lead_20261102/repro_dp_port_collision.py --out /data/ycfeng/tmp/glm53_dsv4/h200/p2_64_failure_lead_20261102/collision --ranks 4 --timeout 8 --squatter
```

Both commands exited 0. In the collision command, zero means the diagnostic captured the expected failure; native startup itself failed. Command clocks, stdout, per-rank JSON, source hashes and script hashes are in `failure_01_20261102/cpu_reproduction/`. Fresh invocations need fresh `--out` directories. The script sets `NATIVE_ROOT`, `VLLM_PORT` and `REPRO_DP_TIMEOUT_S` for its child processes; these do not change the formal package. CPU environment: Python **3.12.13**, torch **2.8.0+cu128**, direct interpreter above, no conda activation or package installation. Evidence-copy/recomputation command: `python3 /data/ycfeng/tmp/glm53_dsv4/h200/p2_64_failure_lead_20261102/retain_evidence.py`.

| Case | Expected condition | Observed | Verdict |
| --- | --- | --- | --- |
| Lead baseline, four ranks | One shared port; every group initialization returns | All ranks use 41544 and return `ok` | PASS |
| Lead occupied-port case, four ranks | Rank 0 retries; peers retain old port; every rank fails | Rank 0 uses 43900 → 43899; peers use 43900; rank 0 `DistStoreError`, peers `DistNetworkError` | Reproduction PASS / native startup FAIL |
| Executor minimal two-rank pair | Healthy baseline; divergent collision | Both expected outcomes in retained JSON | Reviewed PASS |
| Port reservation probe | Ports can be rebound after `get_open_ports_list` returns | 5/5 rebound in the same process; child rebind succeeds | Reviewed PASS |
| Existing coordination-store control | Consistent port and successful initialization | Both successful and failed trials; failed trial uses one shared port but rank 0/2 report Gloo errors | NOT QUALIFIED |

The core extracted methods are `get_next_dp_init_port`, `_pick_stateless_dp_port`, and `stateless_init_dp_group` (`vllm/config/parallel.py:569–662`). With `_coord_store_port=0`, each rank independently pops the next preselected port. An `EADDRINUSE` retry is local and does not notify peers. The nonelastic port-list branch is at `parallel.py:878–884`; the production HELLO handshake (`vllm/v1/engine/utils.py:1330–1348`) sends the list and master address/port but not a coordination-store port. The extractor supplies config/import/environment shims, preserves the core method bodies, and calls real distributed utilities. It does not execute full GPU package imports, the production HELLO transport, NCCL, or model serving. Baseline success checks group creation, not an all-reduce or full serving workload.

All eight GPU workers captured the same `parallel.py` hash as the frozen source. Corresponding worker hashes for `distributed/utils.py` and `network_utils.py` are unavailable; their local frozen hashes are recorded. The CPU torch version differs from the GPU torch **2.13.0+cu130**. These limits prevent equating CPU mechanism reproduction with a full GPU incident reproduction.

The optional coordination-store experiment injects that store into a shim with `enable_elastic_ep=True`. Its mixed outcomes remain visible in `executor_coord_store_{failed,passed}.json`; the aggregate rerun's PASS does not erase the earlier failure. Neither switching production to elastic/Ray nor the proposed multiprocess integration is validated. Further generic harness work is stopped; a production correction requires its own targeted CPU checks.

## Executor review and termination

The CPU executor wrote `findings.md`, then continued optional assertions and aggregate reruns after the main symptom was already isolated. The lead stopped Pi after capturing sufficient results and independently reproducing the baseline/collision pair. Exit **130** is an explicit stop, not a settled successful executor turn. `cpu_reproduction/executor_receipt.json` records the session and `custom/deepseek-v4.1-flash-sys` / `high`; `executor_findings.txt` preserves the submitted findings. The selected results are accepted by lead review, not by executor exit status.

The lead rejects three overstatements in that report. First, peers agreeing on a port does not imply a shared `VLLM_PORT`: the inspected production HELLO metadata propagates the selected list. Second, the original log warning does not directly record which bind failed, the replacement port, or the contested listener owner. Third, the failed coordination-store trial is not established as unrelated torch/network flakiness. The accepted conclusion is the source-extracted CPU mechanism and its consistency with the observed incident; the full incident reconstruction and a corrected production startup remain unverified. The report's optional fail-fast recommendation is not adopted as a substitute for successful startup.

## Recovery and outstanding work

[The recovery proposal](h200_mainline/p2_64_formal/failure_01_20261102/recovery_proposal.md) recommends a startup-only native coordination correction, followed by a fresh frozen generation under the same 64-H200/3600-second/64-GPUh per-job limit. It preserves the original source/package and uses a distinct run tag and cloud path. This proposal is prepared for YC; no native implementation or second submission has occurred.

Final record checks: the authored-document diff check passed; Python3 parsed all 26 retained JSON files. The full `git diff --cached --check` flags 10 trailing-whitespace lines in two preserved raw logs (`lead_baseline.log` and decode-head `server.log`, including its CRLF/banner lines). Those evidence bytes are intentionally unchanged. `git diff --cached --check -- '*.md' '*.json' '*.txt'` passes. `git -C /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 rev-parse HEAD` returned the frozen native commit, and `git status --short --untracked-files=no` in that checkout was empty. No production code changes enter this records-only commit.

The approval boundary is D-P2-64-FORMAL-B's explicit three package changes and conditional infrastructure/preemption retry, not a new skill-imposed gate. The current failure does not establish that retry condition. Full P2-64 admission, three-repeat comparison, exact new-job usage, and tickets 18/19 remain unfinished. The completed Frontier replay and earlier passing checks are reused.
