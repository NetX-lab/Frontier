# D25 H80064 native workload artifact audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Completed read-only local population, token, timing, phase, throughput and native-log audit. |

## Outcome and scope

- Task ID: `D25-H80064-METRICS`; parent-provided job ID: `exp-1022-024130-735731`.
- Source pinned by the sealed config: `e998210d1ba75a30c88f66a1c6c9f278c4c8853e`.
- Local evidence: `/data/ycfeng/tmp/step5_d25_workload_20261022/h800_64_workload01/stage/evidence`.
- Inspected only that evidence and the sealed stage config, trace and harness. No authentication, GPU/platform/cloud operations, source edits, delegation, or publishing.
- **PASS: complete client workload collection.** All 432 unique requests are present: 48 warmups and 384 formal requests, with three separate 128-request formal windows. Formal p90 TTFT is **1.911773s**, with **0/384 above 5s** and maximum **2.473084s**.
- **Not a stable-latency or native-qualification verdict.** TPOT/E2E improve materially between repetitions, and a first-formal-window decode JIT is recorded. Native runtime qualification, reference admission and Frontier accuracy remain **UNVERIFIED**. Platform terminal/creator/mount acceptance belongs to the primary session and was not independently inspected here.

## Validation

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Frozen canonical package | Four checked members plus pinned checksum file | All five hashes match; 37 sealed tool/trace hashes also match config | PASS |
| Existing canonical reader and native client reader | Three complete formal DataFrames | `canonical_workload` and `native_client_rows(..., source_format='step5')`: 128/128/128 | PASS |
| Client/proxy population | 432 unique IDs each, exact bijection | 432 each; three times 16 warmup plus 128 formal; no extras/duplicates/omissions | PASS |
| Native completion | HTTP 200, completed stream, length finish, no validation/transport errors | All 432 client records and both proxy role responses pass; phase failure lists empty | PASS |
| Token counts and usage | Per request 6144 prompt + 1024 output = 7168 | All 432 exact; 442,368 native output IDs in 442,368 single-token deltas | PASS |
| Prompt identity | Stable template controls content, sent/accepted hashes match | 144 template prompt hashes, identical across all three repeats; accepted count/hash match sent count/hash | PASS, captured-hash scope |
| Output delta timestamps | Finite, ordered, within client dispatch/response endpoints | Every delta checked; first/last exactly equal endpoint fields; nonempty integer nonnegative IDs | PASS |
| Endpoint metric provenance | TTFT=first-dispatch; TPOT=(last-first)/1023; E2E=last-dispatch | Maximum discrepancy against saved client metrics: **0ms** | PASS |
| Arrival identity | Frozen exact phase-relative offsets 0..15 or 0..127s | All planned offsets and template/transport ID rules exact; actual dispatch lateness bounded below | PASS |
| Phase separation and idle | Six complete post-phase boundaries with all four DP lanes idle in both roles | All six verified; running/waiting/KV-usage gauges all zero; completion counters exact | PASS |
| Native counter provenance | Delta recomputed from captured Prometheus pages | All six phase deltas agree; floating sums compared at abs 1e-9, rel 1e-12 | PASS |
| Worker-local lifecycle | Eight role/lane receipts complete | All eight `stage=finished`, `workload_returncode=0`, `server_stop={SIGINT,0}`, observation errors empty | PASS, local receipt only |

The trace CSV has no request-ID column; its row order, lengths and arrivals match `request_ids.json`. Identity is bound through the separate frozen request-ID artifact. Actual dispatch offsets were recomputed against each recorded phase origin; lateness is not substituted into the frozen arrival schedule.

| Repeat | Warmup maximum lateness (ms) | Formal maximum lateness (ms) | Formal actual dispatch span (s) |
| --- | ---: | ---: | ---: |
| 1 | 1.603124 | 4.365894 | 126.999960 |
| 2 | 1.840914 | 2.028765 | 126.999948 |
| 3 | 1.647174 | 1.980578 | 127.000064 |

Boundary cumulative completions in **each** role are 16, 144, 160, 288, 304, 432; every DP lane has 4, 36, 40, 72, 76, 108 respectively. All six final idle samples start after the last client response of their own phase, by 0.058268, 0.235024, 0.249504, 0.418789, 0.453490, 0.657105s. Startup was also observed idle but explicitly diagnostic; it is not a calibration-readiness proof.

## Formal metrics, warmups excluded

Nearest-rank p90 is sorted value `ceil(0.90*N)-1` in zero-based indexing: rank 116 for N=128 and rank 346 for N=384. Latencies are per request, not phase makespan.

| Population | Metric | Mean | p90 | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| Repeat 1, N=128 | TTFT (s) | 1.759821 | 1.952135 | 1.147480 | 2.473084 |
| Repeat 1, N=128 | TPOT (ms/token) | 82.384560 | 99.983773 | 55.118197 | 100.590789 |
| Repeat 1, N=128 | E2E (s) | 86.039226 | 103.978045 | 58.444335 | 105.311527 |
| Repeat 2, N=128 | TTFT (s) | 1.655267 | 1.903968 | 1.140727 | 2.041075 |
| Repeat 2, N=128 | TPOT (ms/token) | 73.926230 | 89.337139 | 52.813963 | 90.979000 |
| Repeat 2, N=128 | E2E (s) | 77.281800 | 93.053702 | 55.469286 | 94.867842 |
| Repeat 3, N=128 | TTFT (s) | 1.587757 | 1.885374 | 1.172660 | 2.123943 |
| Repeat 3, N=128 | TPOT (ms/token) | 67.568029 | 78.902957 | 53.125689 | 80.049567 |
| Repeat 3, N=128 | E2E (s) | 70.709852 | 82.290655 | 55.722559 | 83.492183 |
| Pooled, N=384 | TTFT (s) | 1.667615 | 1.911773 | 1.140727 | 2.473084 |
| Pooled, N=384 | TPOT (ms/token) | 74.626273 | 93.557086 | 52.813963 | 100.590789 |
| Pooled, N=384 | E2E (s) | 78.010292 | 97.515587 | 55.469286 | 105.311527 |

Because all repeats have the same formal population, the arithmetic mean of the three latency means equals the pooled mean. The pooled p90 is calculated over all 384 formal rows, **not** averaged from the three p90s.

### Throughput

Each window is **first formal client dispatch to last formal client output token**, excluding warmups and later drain/idle observations. Numerator per window: 786,432 input + 131,072 output = **917,504 total tokens**.

| Repeat | Window (s) | Total tokens/s | Output tokens/s | Formal start -> last output, UTC on 2026-10-21 |
| --- | ---: | ---: | ---: | --- |
| 1 | 185.444295 | 4947.598965 | 706.799852 | 18:59:36.434535 -> 19:02:41.878829 |
| 2 | 182.469233 | 5028.266863 | 718.323838 | 19:03:45.268202 -> 19:06:47.737435 |
| 3 | 182.722624 | 5021.293927 | 717.327704 | 19:07:51.537682 -> 19:10:54.260306 |
| Arithmetic mean of 3 repeats | — | **4999.053252** | **714.150465** | Each repeat weighted equally |
| Pooled window-weighted | 550.636152 summed | **4998.785481** | **714.112212** | 2,752,512 total / 550.636152s; 393,216 output / same |

Do not divide pooled tokens by first-repeat start to third-repeat end: that would include warmups and inter-phase drainage. These client rates are not the 10-second instantaneous server log rates.

## TTFT above 5 seconds: warmup only

There are **12/48 warmup requests above 5s**, all repeat01 warmup r000-r011. Repeat01 warmup mean/p90/max TTFT: **6.445976 / 9.353529 / 11.506277s**. Repeat02 and repeat03 warmup maxima: **1.892877s** and **1.842284s**. These rows are not included in any formal statistic above.

Measured decomposition below uses a bijective request-ID join between client and proxy. P interval is proxy arrival through P response; D interval is decode response headers through the client's first token. It does not identify individual GPU compute, queue or transfer completion events.

| Repeat01 warmup suffix | TTFT (s) | Proxy arrival -> P response (s) | D headers -> first client token (s) |
| --- | ---: | ---: | ---: |
| r000 | 6.247235 | 4.247361 | 1.992903 |
| r001 | 11.506277 | 7.580721 | 3.904503 |
| r002 | 8.968069 | 6.583276 | 2.365735 |
| r003 | 7.486197 | 5.578410 | 1.885656 |
| r004 | 7.366006 | 4.581997 | 2.763930 |
| r005 | 9.353529 | 5.541266 | 3.793775 |
| r006 | 9.102510 | 4.544461 | 4.541935 |
| r007 | 6.909255 | 3.542269 | 3.349705 |
| r008 | 5.210183 | 2.545304 | 2.649558 |
| r009 | 6.100648 | 3.020559 | 3.069691 |
| r010 | 5.586413 | 2.018332 | 3.556572 |
| r011 | 5.000677 | 1.885747 | 3.109299 |

For the maximum r001: client-to-proxy **0.000952s**, proxy-through-P **7.580721s**, P-response-to-D-submit **0.001345s**, D-submit-to-headers **0.018757s**, D-headers-to-first **3.904503s**, sum **11.506277s**. P/D connection establishment was **0.000277/0.004223s**. Across these 12 requests, new-connection durations are at most **5.799ms**; some P connections are reused. Fresh TCP setup is therefore not the observed multi-second component.

### Concrete native evidence and interpretation

- First warmup origin is **2026-10-21 18:58:31.782151 UTC**. P's four lanes log first-inference JIT for `_count_expert_num_tokens`, `fused_moe_kernel`, and `_paged_scores_sm90` at 18:58:31-38. D logs `_paged_scores_sm90` and `_gumbel_sample_kernel` at 18:58:37-44. Native logs therefore directly establish JIT during these high-latency warmups. They do **not** provide per-request JIT start/end attribution or enough timing to equate each full delay to compilation.
- D's 18:58:44 log shows running requests, one deferred/waiting request on several lanes, and KV fetch in progress or completed-waiting. This is concrete native pending-work evidence, not a claim that the whole D interval equals transfer time.
- Native warmup P mean TTFT/E2E is **3.612002s**, D mean TTFT **2.795151s**. Warmup D NIXL transfer count/sum is **128 / 24.829997s**, versus repeat02 **128 / 5.697508s** and repeat03 **128 / 4.123691s**. Those counts are rank-level observations (16 requests × 8 TP ranks), and summed overlapping rank times must not be treated as request elapsed time.
- No NIXL failure/expiry/late-notification counter increases occur in any phase. D's observed prompt split is **6128 transferred + 16 locally computed tokens per request**, matching the sealed block-size16 pull contract. P's scheduling prompt counter is 6128 per request, while request-level computed/accepted input evidence is 6144; do not interpret P's lower scheduling counter as a truncated client prompt.
- Formal native P TTFT means are **1.023953 / 1.024093 / 1.030205s**. Formal native D TTFT means are **0.706865 / 0.601037 / 0.520914s**. Client proxy decomposition means are respectively P interval **1.035840 / 1.036015 / 1.047276s**, post-P-to-first **0.722901 / 0.618222 / 0.539477s**, and client-to-proxy about **1.0ms**. This independently supports the role-side timing interpretation without replacing client metrics.

## TPOT/E2E instability and limits

The latency population is complete but not stationary: TPOT mean **82.384560 -> 73.926230 -> 67.568029ms**, and E2E mean **86.039226 -> 77.281800 -> 70.709852s**. Repeat01-to-repeat03 improvement is about **18.0% TPOT** and **17.8% E2E**; sample CV across repeat means is **9.96% / 9.86%**. Total-token throughput varies less: sample CV **0.894%**, repeat maximum/minimum ratio about **1.0163**.

- Native D ITL mean independently reproduces the decline: **82.389709 / 73.929556 / 67.572279ms** (130,944 intervals each). Across all four D lanes, repeat ITL means are closely aligned, so this is not isolated to one DP lane.
- Repeat01 has **108 inter-token gaps above 1s**, affecting 105 requests: 8 gaps end at 18:59:46 and 100 at 19:01:23. Maximum **1.143440s**. Repeat02/03 have no gaps above 1s, maxima **0.587116/0.595071s**. The first burst at 18:59:46 is not attributed by the retained logs.
- D's four lanes log `_direct_scores` first-inference JIT at **19:01:23-24**, inside repeat01 formal, coinciding at one-second log resolution with the second widespread >1s output-gap burst. This proves that the 16-request warmup did not remove every later JIT case; causally assigning all of the repeat drift to that JIT would exceed this evidence.
- Native throughput/running logs show load building during each finite arrival window and later draining; per-request TPOT/E2E is also nonuniform within each repeat. Quartile TPOT means for arrivals 0-31 / 32-63 / 64-95 / 96-127 are **98.89/90.14/77.30/63.22ms**, **88.18/80.37/68.15/59.01ms**, **77.76/70.90/63.12/58.49ms**. Do not describe these finite-window statistics as steady-state decode speed.
- P's graph logs show runtime `NONE` for 6128-token prefill work; D logs `FULL` and `PIECEWISE`. Configured/captured graph support is not proof of graph replay on every required rank, and this audit does not award that qualification.
- Configured load format is **dummy**, not trained checkpoint weights. Output counts/IDs are native transport evidence, not model quality or numerical accuracy.
- The retained client artifact contains accepted prompt count/hash, not the raw echoed accepted prompt ID list. All 432 accepted/sent hashes match, and all 144 template hashes are repeat-stable. The vocabulary-bearing model metadata is outside this audit's allowed paths; independently regenerating the deterministic prompt IDs or proving output IDs below that vocabulary bound was not performed.
- Proxy response completion can occur before the client's last buffered token is read. In 183/432 requests, client last-token time is after proxy decode-response time; this is not an invalid client timestamp order. All requests retain ordered dispatch->first->last->client-response endpoints, ordered proxy/HTTP events through first-token reception, and proxy completion before client response.
- P head records an `EngineDeadError` at **19:11:11**, in explicit SIGINT shutdown after the final formal output (**19:10:54.260306**) and final all-lane idle boundary (~19:10:54.919 UTC). All eight local stop receipts return0 and all business requests completed. Reported as a shutdown issue; not omitted or reclassified as a workload request failure.

### Evidence line references

Paths below are under `stage/evidence/nodes/`; line numbers count literal LF separators (`text.split('\n')`), preserving embedded carriage-return graph progress lines.

- `gpu-h800-0064.host.platform.shaipower.com/server.log`: 525-527 first P JIT; 3220-3249 shutdown region, including 3235-3242 `EngineDeadError`.
- `gpu-h800-0071.host.platform.shaipower.com/server.log`: 318-324 first P lane1 JIT; equivalent lane2/3 entries in `gpu-h800-0114...` and `gpu-h800-0180...`.
- `gpu-h800-0390.host.platform.shaipower.com/server.log`: 541-542 first D JIT; 569/587/603/616 pending KV observations; 1877 `_direct_scores`; 1880/1899/1921/1940 D running-load observations.
- `gpu-h800-0399.host.platform.shaipower.com/server.log`: 338-339 warmup JIT and372 formal `_direct_scores`; `gpu-h800-0566...`:354-355/372; `gpu-h800-0575...`:338-339/372.

## Reproduction and environment

Audit Python: `/usr/bin/python3`, **3.10.6**, GCC11.3; no conda used. Worker-recorded versions: vLLM **0.1.dev22912+g8224bb82f**, torch **2.13.0+cu130**, aiohttp **3.14.5**. Worker paths indicate Python3.12; exact worker Python patch version was not present in the allowed result receipt.

Every heavy inspection was run under `timeout 60s` and Python `RLIMIT_AS=30GiB`, with bytecode writes disabled. Peak successful extended audit RSS: **325,740KiB**, well below30GB. `systemd-run --user --scope -p MemoryMax=30G` failed with `Job failed`; global scope failed with `Interactive authentication required`. No auth was attempted. The bounded Python address-space limit was used instead.

Exact successful final metric recomputation command (working directory `/data/ycfeng/Frontier/.worktrees/step5-preview`):

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import resource
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
from pathlib import Path
import sys, json, math, statistics
stage = Path('/data/ycfeng/tmp/step5_d25_workload_20261022/h800_64_workload01/stage')
sys.path.insert(0, str(stage))
from tests.comparison.calibration.client_metrics import native_client_rows
from tests.comparison.calibration.pd_replay import canonical_workload
workload = canonical_workload(stage / 'trace/workload')
identities = json.loads((stage / 'trace/workload/request_ids.json').read_text())
frames = native_client_rows(stage / 'evidence', workload, identities, source_format='step5')
rows = [json.loads(line) for line in (stage / 'evidence/replay/client_requests.jsonl').open()]
assert len(rows) == len({row['request_id'] for row in rows}) == 432
windows = []
for repeat in (1, 2, 3, 'pooled'):
    formal = [r for r in rows if r['segment'] == 'formal' and (repeat == 'pooled' or r['repeat'] == repeat)]
    result = {'repeat': repeat, 'n': len(formal)}
    for name, values in {
        'ttft_s': [r['first_token_monotonic'] - r['dispatch_monotonic'] for r in formal],
        'tpot_ms': [1000 * (r['last_token_monotonic'] - r['first_token_monotonic']) / 1023 for r in formal],
        'e2e_s': [r['last_token_monotonic'] - r['dispatch_monotonic'] for r in formal],
    }.items():
        v = sorted(values)
        result[name] = dict(mean=statistics.mean(v), p90=v[math.ceil(.9 * len(v)) - 1], min=v[0], max=v[-1])
    if repeat != 'pooled':
        window = max(r['last_token_monotonic'] for r in formal) - min(r['dispatch_monotonic'] for r in formal)
        windows.append(window)
        result.update(window_s=window, total_tokens_s=917504/window, output_tokens_s=131072/window)
    print(json.dumps(result))
print(json.dumps(dict(total_tokens_s_arithmetic=statistics.mean(917504/w for w in windows), output_tokens_s_arithmetic=statistics.mean(131072/w for w in windows), total_tokens_s_window_weighted=2752512/sum(windows), output_tokens_s_window_weighted=393216/sum(windows))))
print('PASS', {name: len(frame) for name, frame in frames.items()}, 'maximum_rss_KiB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
PY
```

Actual final command: PASS, three128-row frames, all numerical values above reproduced; peak RSS272,876KiB. Extended direct checks additionally looped through every token delta, prompt hash, proxy event, phase gauge, checksum and captured page delta as recorded in the validation table. Audit-only false assertions were corrected after inspecting the actual schema/semantics: CSV has no ID column; native floating sums differ at rounding precision; proxy completion need not follow client buffered-token reception. No evidence or source files were changed to make checks pass.

## Hashes

| Artifact relative to stage | SHA-256 |
| --- | --- |
| `config.json` | `ac80f263f803bd5fbf43610d9232e1c99757bb71c38ac87a13945fd5a1ae046d` |
| `trace/workload/SHA256SUMS` | `594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9` |
| `trace/workload/manifest.json` | `5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3` |
| `trace/workload/trace.csv` | `2d557d7d923b71fbf62bc7e552d1fb547a8ec88c1e54660c0c76cc2de3c49ec2` |
| `trace/workload/request_ids.json` | `318cd1ed1ab0bf98c54be354daa4817982733dc954831819fd3a2c2f8b2511f6` |
| `trace/workload/warmup_pool.csv` | `55a970c57ed6819fb3e50991aa159af0113e5b0fdbd6e47fc90d7a2f5038b99e` |
| `evidence/replay/client_requests.jsonl` | `dcf2b459b1d10905e609320a56f2b2b6f8611e1f29da62c259269fe4046589e4` |
| `evidence/replay/proxy_requests.jsonl` | `97c74de3bd5f4a55ffc8375c05d45aa9f50aaaf32bd01936cd5ec22576b5b680` |
| `evidence/replay/canonical_summary.json` | `88df7253f353b90b8a18953df304224abfb132254c1b0efc085a32342a516078` |
| P head `gpu-h800-0064.../server.log` | `b75f3ca34b398afc15fe6f59bc6024931deb02f868e3524fbd248367c7e92a7f` |
| D head `gpu-h800-0390.../server.log` | `9bc9ceaad17fc3636c1ce920b9fc855a59f689dac6b86096a96b8bf94ff97fb3` |

## Phase checkpoint

### 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 0｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| H80064 完整请求、token、arrival 与阶段审计 | PASS | 432 条记录、6 个 idle boundary 和已有 reader 全部通过 |
| H80064 p90、均值、范围及两种吞吐汇总 | PASS | 原始 endpoint 独立复算，warmup 未计入 |
| >5s TTFT 与 native 日志解释 | PASS | 12 条均为首轮 warmup；正式 JIT/延迟差异已保留 |
| 主任务双规模汇总及平台最终验收 | TODO | 属于 primary；本审计未访问平台或 H80032 |

### 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 本审计开始至报告落盘｜**当前前沿：** 返回 H80064 证据给 primary

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 仅已有 reader 和直接计算，未建设额外工具 |
| D2 细节反复重试 | CLEAR | schema/浮点/缓冲次序错误均经具体证据定位后修正；无重复参数试探 |
| D3 支线问题修补 | CLEAR | shutdown/JIT 仅记录，未越权修改 |
| D4 可并行任务停滞 | CLEAR | primary 明确负责32与平台；本子任务禁止再委派 |
| D5 超范围镀金 | CLEAR | 只有所需审计报告和小型JSON receipt |
| D6 分析空转 | CLEAR | 分析直接形成已请求的数字、异常解释与限度 |
| D7 路线违背 | CLEAR | 无 auth/GPU/platform/cloud/source-edit/delegation/push |

**文档检查：** S1-S6 均未发现明显缺陷；主任务 progress 尚需 primary 结合双规模结果更新。
**归因：** 无。

### 3. 纠正建议

**分支：** A：spec/plan 合理，无需改动。
**需停止的偏离行为：** 无。

**接下去的路线：**
1. 返回本报告与JSON receipt → 完成标志：primary 可直接使用数字及限制。
2. primary 合并32/64结论并完成平台验收 → 完成标志：双规模请求的最终报告完整。

### 4. 最值得关注的 3 点

1. 正式 TTFT 无 >5s，但首轮 warmup 的12条高延迟有实测分段证据。
2. 首个正式窗口仍有 `_direct_scores` JIT 和同步输出间隔，不能宣称延迟稳定或 native readiness。
3. 本次仅完成64卡本地证据审计；平台终态、32卡和准确性 verdict 不在本子任务范围。
