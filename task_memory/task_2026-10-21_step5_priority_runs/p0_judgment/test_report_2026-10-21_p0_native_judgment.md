# D21-P0-NATIVE-JUDGMENT: retained H80032 native evidence

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Read-only judgment of P0 qualification01; original outcomes preserved. |

## Verdict and scope

**24-request fixed-length/idle qualification: PASS. Basic runtime qualification: PARTIAL. Ready for formal: NO.** The retained evidence establishes working H800 P16/D16 TP8/DP2/EP16/PP1 serving, MRv2, exact lengths, both request-owner lanes, NIXL reads and drain. It does **not** establish actual FULL replay per lane, G3 content/lifetime qualification, G4 clean reference, admitted measured profiles, or any of the four accuracy errors.

Task ID: `D21-P0-NATIVE-JUDGMENT`. Job: `exp-1021-195639-166258`. Source: `2033376877ac713670cdd5574b5e95fc00b0cdb4`. Evidence root **R**: `/data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification01`; **E**: `R/stage/evidence`; node abbreviations below expand to `E/nodes/gpu-h800-NNNN.host.platform.shaipower.com/`. Line references count LF-delimited physical lines; progress-bar carriage returns are not additional lines.

Authority: primary task `spec.md:85–136`, `requirements.md:1108–1128`, tickets `09-native-h20064-qualification.md:21–65` and `23-canonical-case-freeze.md:51–73`. Parent task records hold the implementation plan; this child owns judgment only. No GPU/auth/network/SSH, simulator investigation, production edit or new auxiliary harness was performed.

## Preserved outcomes and identity

- `R/terminal01.json:2–14,18–38`: platform **Succeeded**, four replicas Succeeded, finish `2026-10-21T12:10:22Z`.
- `R/identity01.json:3–21`: creator `i-fengyicheng`, charged `codesign`, non-preemptible, four replicas × eight GPUs, local source NFS mount and immutable image `sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`.
- `E/replay/canonical_summary.json:584–608`: `CLIENT_WORKLOAD_VERIFIED`, client true, no observation errors; **runtime_qualification=UNVERIFIED**, **reference_admission=UNVERIFIED**, `repeat_metric_means=null`. All four `result.json:3` retain `qualification_verified=false`; none was rewritten or interpreted as a native failure.
- All node results finish with workload return code 0 and requested SIGINT/server return code 0: 0083 `result.json:191–196`, 0114 `:171–176`, 0216 `:167–172`, 0434 `:171–176`. All four `drained.json:1–4` and `native_stopped.json:1–4` exist.
- Local rehash: **28/28 staged tool/config/trace pins**, **3/3 original metadata pins**, zero mismatches. Config retains **47 native source pins**. Staged worker `native_pd_worker.py:132–138` checks installed source and metadata before rendezvous; all nodes reached finished. This is execution-path evidence, not a separately retained per-file worker hash census. Workload manifest `5c3acb69…c55d3` and SHA256SUMS `594c3cb6…13b9` match the common D18 package; this qualification does not replay its entire formal population.

## Topology, runtime and capacity

| Role/lane | Host | MRv2 line | KV capacity / concurrency at 7168 | Server log lines |
| --- | --- | --- | --- | --- |
| P / 0 | 0083 | 79 | 332,758 tokens / 46.42x | 27–28,65–79,96–105,178 |
| P / 1 | 0114 | 30 | 329,899 tokens / 46.02x | 18–30,47–54,125 |
| D / 0 | 0216 | 117 | 329,899 tokens / 46.02x | 29–30,75–117,134–143,184 |
| D / 1 | 0434 | 64 | 332,758 tokens / 46.42x | 20–21,26–64,81–88,127 |

Each `result.json:8–9` binds role/lane. `nvidia_smi` at result lines 105/106/105/106 proves **32 distinct H800 UUIDs**, eight per host. Logs initialize role world ranks 0–15, show all eight TP/EP-labelled model-load workers per node, each at **40.74 GiB**, and identify DP0/DP1 with EP ranks 0–7/8–15. Thus EP16 spans each role's TP×DP, not extra GPUs. This establishes startup rank coverage, not complete per-request useful/dummy-forward accounting.

Actual EngineCore initialization records `max_seq_len=7168`, TP8/DP2/PP1, `speculative_config=None`, BF16 default, FP8, prefix/chunked prefill off, eager false. Native non-default args in head logs `:8` and staged/result commands retain budget8192, max_num_seqs128, graph ceiling256 and utilization0.92. The concurrency estimate is a native capacity report, **not proof of 128 simultaneous requests or sustained saturation**. Observed maximum client in-flight is16 during warmup and8 in the concurrent wave; each lane completes8 then4 requests. No in-flight native queue/occupancy series or NIC/NUMA affinity map is retained. UCX port lists are in all results, and native connector configuration is CUDA-buffer NixlConnector/UCX. DEEPGEMM FP8 MoE selection appears at server lines86/37/124/71; all nodes detect STEP5_FLASH_ATTN and STEP5_DSA. No full precision/finite-state or layer-execution census is claimed.

### Graph evidence: an explicit remaining gap

The requested graph configuration is `FULL_AND_PIECEWISE`, and head logs contain **capture** progress (0083 `server.log:148–152,314–316`; 0216 `:154–158,321–323`). However the four complete retained server logs contain **zero `Engine NNN` graph-statistics headers and zero non-capture FULL runtime rows**; `canonical_summary.json` contains no graph metric series. There is therefore no observed FULL replay size/count that can be assigned to P0/P1/D0/D1. **Actual decode replay remains UNVERIFIED**, rather than inferred from graph capture, flags or successful generation. This does not create a requirement for 32 independent replay counters.

## Requests and required idle boundaries

- `E/replay/client_requests.jsonl:1–16` and `:17–24`: 16 same-shape warmups and8 simultaneous qualification requests, one repetition, all HTTP200/completed/stream-completed, no validation errors. Every request accepts exactly6144 IDs and returns exactly1024 token IDs; final usage agrees. Accepted prompt SHA equals produced prompt SHA for all24. Totals: **147,456 input / 24,576 output tokens**. Count token IDs, not SSE chunks:10 requests have1023 recorded deltas because one delta contains multiple IDs, yet each sums to1024.
- `proxy_requests.jsonl:1–24`: one-to-one joins, both P/D HTTP200, NIXL pull, remote_num_tokens6128 throughout. Both P DP owners contribute12 requests overall (8 warm +4 concurrent); native P/D completion counters independently report the same split.
- Warmup dispatch span **15.000262 s**, maximum dispatch lateness **2.023 ms**. Concurrent dispatch span **1.817 ms**, maximum lateness **2.156 ms**; all8 overlap. The concurrent first-submission-to-last-completion window is39.131584 s. These are qualification diagnostics, not the128-request formal schedule.
- `canonical_summary.json:470–525`: after warmup at monotonic31813855.428717945, both roles' lanes0/1 expose **running=waiting=KV=0**, completed16 per role /8 per lane. `:527–581`: after concurrent at31813895.798040047, same gauges zero, completed24 per role /12 per lane. Both required idle boundaries PASS.
- `canonical_summary.json:588–605` preserves absent first-start gauges and `diagnostic_only=true`, verified false. **Initial absence is diagnostic only**, never substituted with zero and not a hard failure. Later idle evidence is present and remains mandatory.

## Native prompt split and transfer measurements

Concurrent delta comes from `canonical_summary.json:19–465`; cumulative totals are recomputed with the unchanged staged `serving_metrics.parse_prometheus/transfer_delta` from its retained pages at lines520–521/576–577 against first-start pages600–601. Present counter baselines are used; absent startup gauges are not used as zeros.

| Observed metric | Warm16 cumulative | Concurrent8 delta | All24 cumulative |
| --- | ---: | ---: | ---: |
| P local prompt compute / prompt-total counter | 98,048 | 49,024 | 147,072 |
| D external-KV prompt tokens | 98,048 | 49,024 | 147,072 |
| D local prompt compute | 256 | 128 | 384 |
| D accepted prompt-total counter | 98,304 | 49,152 | 147,456 |
| D NIXL read count | 128 | 64 | 192 |
| D NIXL read bytes | 17,905,549,312 | 8,952,774,656 | 26,858,323,968 |
| D descriptor count sum | 267,168 | 1,268,864 | 1,536,032 |
| D transfer-time sum (s) | 11.467107 | 2.507299 | 13.974406 |
| D post-time sum (s) | 1.532306 | 0.393696 | 1.926002 |

Both roles' local-cache-hit counters are0; all four NIXL failure/notification/expiry counters stay0 on both lanes, both roles. P read counters are0, as expected for pull. Concurrent D splits equally across owners:32 reads and4,476,387,328 bytes per lane. Arithmetic normalization gives139,887,104 bytes/read,1,119,096,832 aggregate rank-read bytes/request,39.1765 ms/read transfer-time mean and6.1515 ms/read post-time mean for the concurrent wave. These are **aggregated rank transfer statistics**, not critical-path wall latency or proof of KV/summary content equality.

Observed prompt counters give **P6128 local tokens/request; D6128 external +16 local tokens/request**, with proxy remote_num_tokens6128. Do not label P as computing6144. Separately, the P `request_prefill_kv_computed_tokens` histogram sums49,152 across8 concurrent requests (`summary:218–234`); its full-prompt semantic count is not a reason to overwrite source counters or infer physical compute. Content per layer/group/rank, region/page boundaries and per-request P-retain→D-read-complete→release ordering are not captured here. Final zero KV usage proves drain, not those G3 lifetime/content conditions.

## Timing diagnostics and limits

| Population | Client TTFT mean (ms) | TPOT mean (ms) | Request E2E mean (ms) | P-response→first-client-token mean (ms) |
| --- | ---: | ---: | ---: | ---: |
| Warm16 | 5133.753 | 43.366 | 49496.997 | 1150.962 |
| Concurrent8 | 3192.382 | 34.466 | 38451.560 | 586.089 |

Client/proxy use the same prefill-head Linux monotonic clock (`summary:587`). For concurrent8, the last column ranges259.152–1034.310 ms, or12.293–27.325% of each request's client TTFT (mean individual fraction19.710%). Warm16 range567.256–2830.831 ms /11.285–33.721%. The post-P interval is plainly not negligible against a1-percentage-point diagnostic budget, but includes transport/queue/read/D-first-step/client delivery; it is **not transfer time alone**, and is not a measured simulator error. T2's tail-compute price remains unmeasured here. No simulator path was investigated or changed.

Preserve observed warnings: P coordinator out-of-order statistics (`0083/server.log:528,531`); warmup JIT during inference (`0083:520,524–526`, `0216:564–565`, `0434:338–339`); native forced shutdown after successful drain (`0083:555–587`, `0216:587–614`) and follower TCPStore disconnect/broken-pipe warnings. No `ERROR` or `Traceback` occurs in these server logs. Absence of a nonfinite warning is not finite-state proof. Basic qualification timings are not promoted to clean reference timings.

## Remaining admission prerequisites

1. Case-aligned ticket08 G3 content/lifetime and boundary/concurrent coverage; retain first-start diagnostic separately.
2. Actual lane-labelled FULL decode replay evidence plus required useful/dummy-forward accounting, remaining native correctness/capacity/placement observations and complete frozen track readiness.
3. Hardware/scale-compatible admitted measured profiles and predictors, followed by **three clean128-request repetitions**, each16 warmups and verified idle boundaries. This run supplies neither G4/reference admission nor the four Frontier-vLLM errors. It cannot release H80064 under the30% rule; final four independent10% gates remain unchanged.

No automatic GPU retry or tooling repair is requested by this read-only judgment. Parent decides the smallest authorized action for the real missing prerequisites, not a retry solely for startup gauges.

## Verification record

Environment: local Linux; `/usr/bin/python3`, Python3.10.6, no conda activation; worker receipts separately report Torch2.13.0+cu130/vLLM0.1.dev22912+g8224bb82f/aiohttp3.14.5. Read-only inline Python checks parsed all24 client/proxy rows, exact token-ID sums and hashes, all four node lifecycle results,32 distinct GPU UUIDs, source/metadata pins, native graph-row absence and both idle boundaries. Expected and actual values above match: **retained request/identity/idle checks PASS; actual replay and formal prerequisites UNVERIFIED**. Exact concise recheck command is in `progress.md` in this directory.

Original inspection errors retained: first `python - <<'PY'` failed127 (`python: command not found`); switched to existing `python3`, no environment edit. A later inline inspection exited1 on a nonexistent `stage/trace/manifest.json`; actual package is `stage/trace/workload/manifest.json`, subsequently verified with the expected hash. These are local inspection errors, not native failures. No auxiliary script, validator or GPU run was added. `receipt.json` hashes the used evidence; raw outcomes remain unchanged.
