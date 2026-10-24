# Profile03 immutable-artifact failure timeline

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded exact local observations, input hashes and deterministic audit; no runtime changes. |

Task: `W128-PROFILE03-FAILURE-TIMELINE`. Audit **PASS** is artifact consistency, not GPU-hang reproduction, successful run qualification or a root-cause finding.
Evidence root `B=/data/ycfeng/tmp/glm53_dsv4/profile03_failed_ksdoj1z0`; node paths below are `B/artifacts/nodes/gpu-h800-NNNN.host.platform.shaipower.com/`. Machine evidence: [timeline.json](timeline.json); verification: [command.md](command.md).

## Exact request counts
| Repeat | Phase | JSONL rows | Dispatched | Completed | Failed | Not submitted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 01 | warmup | 16 | 16 | 16 | 0 | 0 |
| 01 | formal | 128 | 128 | 121 | 7 | 0 |
| 02 | warmup / formal | 16 / 128 | 0 / 0 | 0 / 0 | 0 / 0 | 16 / 128 |
| 03 | warmup / formal | 16 / 128 | 0 / 0 | 0 / 0 | 0 / 0 | 16 / 128 |
`run/client_requests.jsonl`: 432 unique IDs, 144 dispatched, 137 completed, 7 failed, 288 not submitted. Repeat01 occupies lines 1–144; repeat02/03 occupy 145–288/289–432. All completed requests report 6144 prompt / 1024 output tokens, stream DONE and finish_reason=length.
`run/canonical_repetitions.json:2–30` independently records only repeat01: warmup16/16, formal128/128 and the same seven failed formal IDs `r121`–`r127`; client_workload_verified=false, reference/profiling UNVERIFIED. JSONL `recorded_requests` includes failed requests, not just successes.
`run/proxy_requests.jsonl`: 144 unique submitted IDs; 137 prefill200/decode200, seven prefill500 without decode, surfaced to clients as502. Warmup has8 successful prefill handoffs through each P lane; formal successes have P0=61/P1=60. No repeated-dispatch inference from the three planned repeats.

## First observed fatal cluster and recorded cascade
1. P1/0329 `server.log:505–551`: EngineCore_DP1 pid1461 fatal traceback at **10-24 07:06:27**, shared-memory response read TimeoutError chained to `TimeoutError: RPC call to execute_model timed out.`
2. P0/0108 `server.log:809–824`: both API output handlers report **10-24 07:06:27** EngineDeadError; seven completion500 responses follow at825–831. `failed.json` and `worker.log:7–11` explicitly observe **native server exit1**, without a marker timestamp.
3. P1 `failed.json`/`worker.log:10` names P0's failed marker; D0/0422 `failed.json`/`worker.log:10` names P0 and P1 failed markers. This establishes marker-dependent propagation, not cross-host ordering from rounded clocks. D0 later logs EngineDeadError at10-24 07:06:43 (`server.log:851`); D1 has no selected failed marker or recorded terminal native result.
4. `B/receipt.json`: P1 container terminated exit1 at **2026-10-24T07:06:43Z**; RJobFailed finish **07:06:45Z**, failed1/stopped3, with `Task fail policy [any] caused undesired running replicas.` P1's retained native-stop return0 is a separate subprocess observation, not container success.
5. `/data/ycfeng/tmp/glm53_dsv4/profile03_launch_20261024.log:20–29`: SDK `workers.wait()` raises RuntimeError for job failed; final Job Stop log is **2026-10-24 15:06:47.402**, timezone unspecified. No logged SDK process exit code is available.
**Limits:** P1 timeout and P0 EngineDeadError share a second-resolution label; P1→P0 temporal causality and GPU-hang root cause are **UNKNOWN**. Three Stopped replicas retain old `container_state.running`; their terminal exit codes are **UNKNOWN**, not inferred from wrapper logs.

## Per-lane observed profiler lifecycle
All times below are wall-log labels on10-24, **not asserted UTC**. Every lane has eight USDT starts and eight USDT stops. “Report complete” is the timestamped wrapper success message; the summary table itself has no timestamp.
| Lane / host | Config cap | USDT start | Autostop request | USDT stop | Report complete | server.log evidence lines |
| --- | ---: | --- | --- | --- | --- | --- |
| P0 /0108 | 512 | 06:55:23 | 06:55:30 | 06:55:30 | 06:56:43 | 540–551;559–567;673–676 |
| P1 /0329 | 512 | 06:55:23 | 06:57:25 | 06:57:27–28 | 07:06:27 | 363–374;376–384;498–501 |
| D0 /0422 | 64 | 06:55:23 | 06:55:27 | 06:55:27 | 06:55:46 | 568–577;582–590;698–701 |
| D1 /0618 | 64 | 06:55:23–24 | 06:55:48 | 06:55:48 | 06:56:23 | 391–400;401–409;514–517 |
Canonical start_profile calls both return200; attempt times remain client-host monotonic23427986.323611613–23427988.086751312(P) and23427986.324121874–23427988.365730893(D). Recorded client origin_wall epochs convert to warmup06:53:53.986469Z/formal06:55:25.465482Z; no clock-offset mapping to other hosts is applied.
P1's report completion and fatal traceback share07:06:27; this does not identify the cause or duration of a GPU hang. Shared-memory waits also appear during successful startup, so those warnings alone are not a root cause.

## Actual identities, later success and transport limits
Each `result.json` independently records host/IP and eight NVIDIA H800 UUIDs: P0/0108=10.191.5.54, P1/0329=10.191.0.34, D0/0422=10.191.4.31, D1/0618=10.191.2.4. Four distinct hosts/IPs and32 distinct UUIDs are observed, rather than inferred from anti-affinity configuration.
All four results agree on87 observed native-source hashes, four model-file hashes, vLLM0.28.0/Torch2.13.0+cu130/aiohttp3.14.3 and NativeServeProfile SHA256 `c8f712177fcd3e8ef6fbbfe2ceb62a30e06d298d432da48bd7790ed214669f58`. Native source_sha is `2cf0a6915ce544dc493a0990f2ea38d81601128a`; source-map digest `1737349f185dea21aea168d128bb2aa51b5a914cc7dda0d845a284dfcf11659e`. Complete hashes, mount command paths and manifest bindings are retained in timeline.json; local source bytes were not revalidated here.
Four platform logs show background `p2p_ping` failing because executable `ip` is missing at06:41–42. Later head `server_ready.json` explicitly records health200 at P06:53:51.287543Z/D06:52:32.193512Z, matching fresh-service canonical bindings; all16 warmups and121 formal requests complete. Thus the missing-ip check is not sufficient evidence for the later fatal failure.
Platform logs configure RoCE, NCCL_SOCKET_IFNAME/NCCL_IB_HCA/GID_INDEX and NVSHMEM mappings; results list UCX devices. Server logs observe PYNCCL DP/EP backend registration and profiler summaries contain NCCL kernels. **Zero selected log lines establish NET/IB, NET/Socket or “Using network” transport**; actual collective and NIXL transport choice remains unverified.
Full trace/step/DP pairing/stream admission and a real source-backed reproduction remain root-owned. No GPU, StepMind, platform, SSH, network, Docker, Git writes, subagents or parent-document changes were performed.
