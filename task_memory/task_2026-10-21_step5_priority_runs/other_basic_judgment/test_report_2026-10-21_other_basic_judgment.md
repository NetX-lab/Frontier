# D21-OTHER-BASIC-JUDGMENT: H20032 and H80064 native evidence

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Judged the two completed qualification02 runs from retained native outputs; no original verdict was changed. |
| 2026-10-21 | Parent revalidates79 hashes and supplies the post-terminal H200 identity observation without rewriting the original receipt. |

## Verdict

| Track / job | Fixed-length requests and required idle boundaries | Full basic runtime | Ready for formal |
| --- | --- | --- | --- |
| H20032 / `exp-1021-201544-965435` | **PASS** | **PARTIAL** | **NO** |
| H80064 / `exp-1021-202325-579469` | **PASS** | **PARTIAL** | **NO** |

Both runs establish startup, native KV capacity reports, all request-owner lanes, exact 6144/1024 completion, NIXL reads, and drain. Neither retained evidence set establishes actual FULL decode replay or complete useful/DP-dummy forward accounting. G3 content/lifetime, G4 clean reference, admitted measured inputs/predictors, and four-metric accuracy remain **UNVERIFIED / not measured**, not silently accepted from platform success.

Task ID: `D21-OTHER-BASIC-JUDGMENT`. Read-only scope: local evidence only; writes are confined to this directory. No source edit, Git operation, subagent, network, GPU, SSH, authentication, platform operation, new test suite, or graph-logging root-cause investigation. Another owner retains graph-logging diagnosis and old-H20064 reuse.

Parent follow-up: the original H200 identity gap in `receipt.json` is resolved by `/data/ycfeng/tmp/step5_d21_priority_20261021/h200_32_qualification02/parent_identity02.json`, observed through the personal API at13:09:29UTC after terminal completion. It verifies i-fengyicheng,step_main,32 non-preemptible GPUs,the pinned image and the exact local NFS source. This is not a reconstructed preflight. Parent rechecked all79 original evidence hashes; both basic/formal verdicts and the original receipt remain unchanged.

Authority: `task_memory/task_2026-10-15_step5_preview_h800/requirements.md:1108–1128`, `spec.md:86–137`, `plan.md:110–153`, `tickets/09-native-h20064-qualification.md:22–65`, and `tickets/23-canonical-case-freeze.md:51–73`; parent `task_2026-10-21_step5_priority_runs/requirements.md:9–19`. This is a basic qualification judgment, not formal collection or comparison.

## Exact evidence roots and original outcomes

All paths below are local retained evidence. Let **B** = `/data/ycfeng/tmp/step5_d21_priority_20261021`, **H** = `B/h200_32_qualification02`, **V** = `B/h800_64_qualification02`, and **E** = `<case>/stage/evidence`. Node names in the tables expand to `E/nodes/<full-hostname>/`; `server.log` references count LF-delimited physical lines, not carriage-return progress updates.

| Record | H20032 | H80064 |
| --- | --- | --- |
| `parent_status02.json:3–15` | Succeeded, 4 replicas, `2026-10-21T12:27:23Z` | Succeeded, 8 replicas, `2026-10-21T12:38:18Z` |
| `E/replay/canonical_summary.json` | `CLIENT_WORKLOAD_VERIFIED`; runtime/reference UNVERIFIED (`:585–608`) | Same (`:733–756`) |
| Node `result.json` | All 4: finished, workload return code 0, SIGINT/server return code 0, qualification_verified=false | All 8: same |
| `drained.json`, `native_stopped.json` | Present on all 4 nodes | Present on all 8 nodes |

Original `qualification_verified=false`, runtime/reference UNVERIFIED, empty observation-error lists and `repeat_metric_means=null` are preserved. These auxiliary qualification fields are not native workload failures. The launcher argument `--mode clean` does not turn this 24-request qualification plan into a formal clean reference.

### Source and runtime identity

- Both `stage/config.json:4–7` identify source `42dda4cc2fca89c836b5593ce4c5f45d2abb28ea` from `/data/ycfeng/Frontier/.worktrees/step5-priority-runs`. Independent local rehash: **37/37 staged tool/input pins and 3/3 original metadata pins per case, zero mismatches**. The common manifest SHA-256 is `5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3`; common SHA256SUMS-file SHA-256 is `594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9`.
- Each config retains **47 installed native source pins**. `stage/native_pd_worker.py:132–138` checks model metadata and installed native sources before rendezvous. Finished nodes establish passage through those checks; no separate per-file native-worker hash census is retained. This judgment does not compare the live repository against the source commit and does not perform Git operations.
- All node results report Torch `2.13.0+cu130`, vLLM `0.1.dev22912+g8224bb82f`, aiohttp `3.14.5`. The staged/recorded immutable image is `a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`.
- H80064 `parent_identity01.json:3–14` independently records creator `i-fengyicheng`, `codesign`, non-preemptible, 64 GPUs, the local source NFS mount, and that image digest. H20032 has **no retained parent platform-identity receipt in its case root**: `stage/config.json`, `run_definition.json`, the dry run and launcher record request `step_main`, 4×8 H200, non-preemptible, and the local workspace. H200 creator, actual charged group/preemptibility/source mount and global cross-session quota compliance are **UNVERIFIED by the supplied platform receipts**, not inferred from the intended config.

## Physical topology, capacity and resolved execution

| Track | Role / lane | Full hostname | MRv2 line | KV capacity tokens / native concurrency at7168 | KV log line |
| --- | --- | --- | ---: | --- | ---: |
| H20032 | P/0 | `gpu-h200-0645.lgcm.sh.istep.fun` | 79 | 1,029,540 /143.63x | 178 |
| H20032 | P/1 | `gpu-h200-0675.lgcm.sh.istep.fun` | 30 | 1,029,540 /143.63x | 125 |
| H20032 | D/0 | `gpu-h200-0819.lgcm.sh.istep.fun` | 117 | 1,029,540 /143.63x | 184 |
| H20032 | D/1 | `gpu-h200-1080.lgcm.sh.istep.fun` | 64 | 1,029,540 /143.63x | 127 |
| H80064 | P/0 | `gpu-h800-0091.host.platform.shaipower.com` | 115 | 530,140 /73.96x | 214 |
| H80064 | P/1 | `gpu-h800-0174.host.platform.shaipower.com` | 30 | 530,631 /74.03x | 125 |
| H80064 | P/2 | `gpu-h800-0175.host.platform.shaipower.com` | 30 | 530,631 /74.03x | 125 |
| H80064 | P/3 | `gpu-h800-0233.host.platform.shaipower.com` | 30 | 530,140 /73.96x | 125 |
| H80064 | D/0 | `gpu-h800-0302.host.platform.shaipower.com` | 157 | 530,140 /73.96x | 224 |
| H80064 | D/1 | `gpu-h800-0310.host.platform.shaipower.com` | 64 | 530,631 /74.03x | 127 |
| H80064 | D/2 | `gpu-h800-0346.host.platform.shaipower.com` | 64 | 530,631 /74.03x | 127 |
| H80064 | D/3 | `gpu-h800-0575.host.platform.shaipower.com` | 64 | 530,140 /73.96x | 127 |

All twelve node results bind role/lane and show eight physical GPUs each. UUID census: **32/32 distinct NVIDIA H200 GPUs**, **64/64 distinct NVIDIA H800 GPUs**. Eight TP/EP-labelled model-load workers per node report40.74GiB (H20032) or23.14GiB (H80064). Native world/rank assignments and worker labels establish per-role TP8/DP2/EP16 or TP8/DP4/EP32, PP1; EP spans role TP×DP ranks and adds no GPUs. This is startup coverage, not complete per-forward/rank execution accounting.

Every EngineCore initialization records `max_seq_len=7168`, TP8, the expected DP size, PP1, `speculative_config=None`, BF16 default, FP8 quantization, `load_format=dummy`, eager=false, prefix caching=false and chunked prefill=false. Native commands/head non-default arguments retain8192 batched-token budget,128 sequences,0.92 utilization, graph ceiling256 and `FULL_AND_PIECEWISE`. All nodes explicitly report `Using V2 Model Runner`, DEEPGEMM FP8 MoE selection, and detected `STEP5_FLASH_ATTN` / `STEP5_DSA`. CUDA-buffer `NixlConnector`, UCX, pull and failure-policy=fail are recorded. UCX device lists exist, but no complete physical GPU↔NIC/NUMA affinity map is retained. Native capacity estimates do not prove128 simultaneous7168-token requests, all graph shapes or saturation.

### Graph, reached batches and DP dummy forwards

Both complete evidence sets contain **zero `Engine NNN` graph-statistics headers, zero non-capture FULL runtime rows, and zero directly recorded per-forward useful/dummy shapes**. Captures exist: H200 P-head `server.log:148–151,314–316`, D-head `:153–157,321–323`; H800 P-head `:183–188,350–352`, D-head `:193–198,361–363`. Capture is not actual serving replay. Retained Prometheus pages expose `iteration_tokens_total` histogram buckets/sum/count, but no graph metric series or exact request/rank/forward shape association. Therefore:

- Actual lane-labelled FULL decode replay: **UNVERIFIED** on every lane.
- Actual reached native batch sizes, padded decode shapes, and DP dummy-forward shapes/counts: **UNVERIFIED**; no estimates derived from flags, capture counts, average tokens, cumulative requests, or client concurrency.
- Maximum client in-flight16 during warmup and8 during the concurrent wave are directly observed client overlap only, **not native execution batch sizes**. Per-layer finite state, precision/exclusion consumption and a complete backbone/expert physical census are not established by these logs.

No diagnosis of why graph logging is absent was attempted; no new observation tooling or GPU retry is proposed here.

## Requests, all-lane participation and mandatory idle boundaries

`E/replay/client_requests.jsonl:1–16` contains16 warmups; `:17–24` contains8 simultaneous qualification requests, one repetition. Both cases have24 unique client/proxy IDs, HTTP200 on client/P/D, streaming complete, finish_reason=length and no validation errors. Every request has6144 accepted prompt token IDs,1024 generated token IDs and final usage6144+1024=7168. Accepted prompt hashes equal produced prompt hashes in all48 requests. Each case totals **147,456 accepted input /24,576 output tokens**. H800 has one1022-delta and one1023-delta response, but token-ID lengths sum to1024; SSE-delta count is not used as output-token count.

The raw Prometheus pages, not just `verified=true`, were independently parsed. At both boundaries **each expected lane** exposes running=waiting=KV usage=0, with no missing lane or gauge. H200 summary anchors: `:470–525` after warmup and `:527–581` after concurrent. H800: `:570–649` and `:651–729`.

| Track / boundary | P completed by lane | D completed by lane | Idle outcome |
| --- | --- | --- | --- |
| H200 warm16 | [7,9] | [8,8] | All4 role/lanes zero |
| H200 all24 | [11,13] | [12,12] | All4 role/lanes zero |
| H800 warm16 | [4,2,5,5] | [4,4,4,4] | All8 role/lanes zero |
| H800 all24 | [7,3,6,8] | [6,5,7,6] | All8 role/lanes zero |

Concurrent deltas are P[4,4]/D[4,4] for H200; P[3,1,1,3]/D[2,1,3,2] for H800. Proxy remote-engine/host joins independently establish P-owner participation; D participation is from native lane-labelled completion/transfer counters, not an invented per-request D-lane join.

| Track / segment | Actual dispatch span | Maximum dispatch lateness | Peak client in-flight | First submission→last response |
| --- | ---: | ---: | ---: | ---: |
| H200 warm16 | 14.999767s | 2.020ms | 16 | 51.706709s |
| H200 concurrent8 | 1.124ms | 1.300ms | 8 | 34.259796s |
| H800 warm16 | 15.000405s | 2.011ms | 16 | 71.704443s |
| H800 concurrent8 | 1.319ms | 1.522ms | 8 | 51.922314s |

First-start gauges are absent: both summaries preserve `startup_observation.verified=false`, `diagnostic_only=true`, the original missing-series errors and raw pages. Missing gauges were **not replaced by zero**, and do not invalidate the observed mandatory post-warmup/post-concurrent zero gauges. These are not three formal repetitions or the complete128-request formal schedule.

## Native prompt split and transfers

Use unchanged staged `serving_metrics.parse_prometheus/transfer_delta` (`:41–55,80–106`) on retained startup/post-warmup/post-concurrent pages. The observed source/transfer counters are present; absent startup running/waiting/KV gauges are not inputs to counter arithmetic.

| Metric (both cases unless stated) | Warm16 | Concurrent8 | All24 |
| --- | ---: | ---: | ---: |
| P local prompt compute / P prompt-total counter | 98,048 | 49,024 | 147,072 |
| D external-KV prompt tokens | 98,048 | 49,024 | 147,072 |
| D local prompt compute | 256 | 128 | 384 |
| D accepted prompt-total counter | 98,304 | 49,152 | 147,456 |
| D NIXL read count | 128 | 64 | 192 |
| D NIXL read bytes | 17,905,549,312 | 8,952,774,656 | 26,858,323,968 |
| H200 descriptor sum | 602,416 | 1,268,864 | 1,871,280 |
| H800 descriptor sum | 545,560 | 1,268,496 | 1,814,056 |
| H200 transfer-time sum(s) | 9.922761 | 1.324295 | 11.247056 |
| H800 transfer-time sum(s) | 17.003509 | 5.583566 | 22.587075 |
| H200 post-time sum(s) | 2.027982 | 0.387538 | 2.415520 |
| H800 post-time sum(s) | 2.915659 | 1.083388 | 3.999047 |

Observed per-request prompt split: **P6128 local; D6128 external+16 local**; all24 proxies per case report `remote_num_tokens=6128`, `transfer_mode=pull`, TP8/PP1. Do not claim P physically computes6144 from accepted prompt length or the separate full-prompt histogram. Both roles' local-cache-hit counters stay0, P read counters stay0, and all four transfer/notification/expiry failure counters stay0 on all lanes.

Concurrent D-lane read counts/bytes: H200 [32,32] /[4,476,387,328,4,476,387,328]; H800 [16,8,24,16] /[2,238,193,664,1,119,096,832,3,357,290,496,2,238,193,664]. These are aggregated rank-read statistics, not unique logical bytes or critical-path wall latency. Equal aggregate bytes across tracks do not establish equal timing or state content. No layer/group/rank KV or side-summary content comparison, page/region-boundary qualification or per-request retain→read-complete→release ordering is captured. Final zero KV usage establishes drain, not G3 content/lifetime.

## Diagnostic timing and preserved warnings

| Track / population | Mean client TTFT(ms) | Mean TPOT(ms) | Mean request E2E(ms) |
| --- | ---: | ---: | ---: |
| H200 warm16 | 4181.462 | 37.150 | 42186.276 |
| H200 concurrent8 | 2669.375 | 30.304 | 33670.074 |
| H800 warm16 | 11663.620 | 48.063 | 60832.411 |
| H800 concurrent8 | 6418.956 | 42.826 | 50230.099 |

These numbers describe only the retained qualification populations. Client/proxy share the prefill-head Linux monotonic clock. No Frontier result, comparative error, clean stability, transfer-only latency attribution or timing scaling is claimed.

Preserve native warnings: JIT during inference on all nodes, H200 D coordinator out-of-order statistics (`0819/server.log:572`), forced native shutdown and follower TCPStore disconnect warnings. H200 logs have no ERROR/Traceback matches. H800 P-head records two `AsyncLLM output_handler failed` / `EngineDeadError` traces at `0091/server.log:758–778`, **after** native shutdown starts (`:729–734`) and after the final completed/idle page; result workload/server return codes remain0. Observed order supports classifying these as shutdown-associated errors, not evidence of failed recorded requests; their causal root was not diagnosed. Do not summarize H800 as error-free or infer finite-state validity from successful completion.

## Remaining formal prerequisites

1. Both tracks: actual lane-labelled decode FULL replay and required useful/DP-dummy forward accounting; remaining native correctness, physical placement/capacity evidence, case-aligned G3 KV/summary content, boundary/concurrent/reuse/cancellation and lifetime/release qualification. Preserve absent evidence as UNVERIFIED. The existing graph owner handles logging diagnosis; this report neither repeats nor prescribes it.
2. Track-bound runtime/window/source freeze; hardware/scale-compatible admitted compute/communication profiles and predictors, independent measured inputs, bundle and existing acceptance dependencies. Profile reach must cover actual6144 prefill,7168 context, D tail and genuinely observed native shapes, not client concurrency estimates.
3. Three clean repetitions, each16 same-shape warmups, verified all-lane idle, all128 formal requests at QPS1, exact counts and unchanged complete-population metrics. H20032 has **no H800 accuracy dependency**. H80064 additionally waits for all four H80032 errors<=30% **in each of three clean complete-population repetitions** plus its own readiness. Neither present run satisfies a release gate. Final four independent10% gates remain unchanged.

No blanket test suite, automatic GPU repeat or source repair was added. Parent owns formal readiness and status updates.

## Verification record and limits

Local Linux, `/usr/bin/python3 -B`, Python3.10.6; no conda activation or native-package import. Focused inline read-only checks parsed all48 client/proxy records, exact token-ID sums/hashes, both cases' raw all-lane idle gauges, transfer counters,12 node lifecycle results,96 unique per-case GPU UUIDs,37+3 pins per case and complete retained logs. Expected versus actual:24=16+8 per case;6144/1024 each;32/64 unique GPUs;all expected post-workload gauges present and0;zero local pin mismatches — **PASS**. Actual FULL rows and exact forward shapes absent — **UNVERIFIED**, not FAIL or inferred PASS. Source hash passage is not a complete runtime/finite-state census.

Exact reproducible check command is in `progress.md`. `h200_32_evidence.sha256` records31 small source evidence files /2,538,561bytes; `h800_64_evidence.sha256` records48 /3,728,125bytes, with paths relative to H or V respectively. SHA manifests bind existing retained files without copying raw logs or touching source evidence. No correctness conclusion is claimed for unrelated files or for evidence absent from these roots.

Inspection-only errors preserved: an early structural print assumed `preflight.json` was a dict, raising AttributeError (it is a list); an initial token check used `len(delta)` on a dict, raising AssertionError (the actual tokens are `delta['token_ids']`). Subsequent direct inspections and corrected checks passed. These were local inspection mistakes, not native failures, and no original file was changed.
