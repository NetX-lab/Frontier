# DS-V4 native06 client acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded independent existing-helper client acceptance, exact formal metrics, cold/drain boundaries, provenance, retained exception and limits. |

**Task:** `W128-V4-NATIVE06-CLIENT-ACCEPTANCE`  
**Result:** **PASS — client business and three-round client stability only.** Full native reference admission and calibration are **not claimed**.

Job `exp-1023-043123-830898`; clean run `dsv4-h800-p0-canonical-20261021-06`; harness `b62c155abd981a71e12e62aa0c0fe3990741d2f8`; declared routing `balanced_topk`.

## Inputs and acceptance machinery

- Case: `/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023`.
- Run: the case's `runs/groundtruth_clean/dsv4-h800-p0-canonical-20261021-06`.
- Frozen helpers: `/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_06`.
- Reused complete `canonical_workload()` and `native_client_rows()`, `dedicated_services()`, `formal_metrics()` and `metric_values()`. For the seven boundaries, directly called the existing `native_idle_evidence()` from unchanged frozen R2, the dependency whose body R3 retains. Import did not execute checker `main()` or write case analysis.
- All accessed input bytes, sizes and nanosecond mtimes were identical before/after. Helpers match the run's recorded hashes. No new validator, source/config/checker/rule edits, run-artifact writes, GPU/platform/SSH/cloud/Docker/native operation, delegation or Git stage/commit.

## Business checks

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Population | 432 = 48 warmup + 384 formal | 432 completed; each of three rounds has 16 warmup + 128 formal | PASS |
| Identity / arrivals | Exact pinned template IDs, unique repeat-qualified transport IDs; phase seconds 0..15 / 0..127 | Zero missing, duplicate or unclassified IDs; exact scheduled arrivals | PASS |
| Status / stream | HTTP200, completed, no error, stream done, finish reason length | All 432; zero status/validation errors | PASS |
| Lengths | Every request 6144 input / 1024 output | All 432 exact, usage total7168; no omitted or shortened output | PASS |
| Formal totals | 786432 input / 131072 output / 917504 combined per round | Exact in every round | PASS |
| Proxy | Exactly one corresponding row per client ID; upstream200 | 432 unique rows, 864 upstream200; NIXL-pull metadata6144/TP8 | PASS, transport evidence only |
| Repetitions / boundaries | Rounds1,2,3; seven cold/drain boundaries | Complete, ordered, verified; no observation errors | PASS |
| Complete client helper provenance | Clean source; client/summary written after manifest | Unmodified full `native_client_rows()` succeeds | PASS |
| Native admission | Root frozen R3 and independent all-rank acceptance | Stored `reference_admission=UNVERIFIED`, `qualification_verified=false` | NOT CLAIMED |

The comprehensive invocation records **10 PASS categories / 0 failed categories**; these are not a count of internal per-row assertions. A separate decisive replay passed. Maximum formal physical send lateness was **2.063195ms**; QPS1 describes the exact schedule, not perfectly timed dispatch.

## Seven boundaries and drain gauges

The initial named proof is `dedicated_services_ready_client_zero_native_counters_zero`. Existing `dedicated_services()` rebinds manifest hash, generation, PIDs, start/readiness receipts and the P/D lane ownership. Client dispatch count is0; all five finish-reason counters are explicit zero on lanes0,1 of both roles. Initial gauges are **unpublished**, with `idle=false`; missing gauges were not filled with zeros.

Every later boundary has exactly `num_requests_running`, `num_requests_waiting` and `kv_cache_usage_perc`, each0 on both lanes of both roles. Every boundary has one ordered, error-free observation after its phase's last output token. Completion totals are exact and fresh; only length counters increase.

| Boundary | Completed per role, expected = observed | P lanes0/1 | D lanes0/1 | Result |
| --- | ---: | --- | --- | --- |
| after_repeat01_warmup | 16 | 8 / 8 | 8 / 8 | PASS |
| after_repeat01_formal | 144 | 72 / 72 | 72 / 72 | PASS |
| after_repeat02_warmup | 160 | 80 / 80 | 80 / 80 | PASS |
| after_repeat02_formal | 288 | 144 / 144 | 145 / 143 | PASS |
| after_repeat03_warmup | 304 | 152 / 152 | 153 / 151 | PASS |
| after_repeat03_formal | 432 | 216 / 216 | 217 / 215 | PASS |

Warmup-last-token to formal-first-submission gaps: **1.405193 / 1.372366 / 1.394145s**. This establishes gauge-level request-owned KV release, not native READ/device completion. Balanced token-to-expert routing is not a requirement for exactly equal DP request placement.

## Exact three-round formal metrics

TTFT is client issue to first output token; request TPOT is `(last-first)*1000/1023`; E2E is issue to last output token. Latencies are arithmetic means of all128 formal requests per independent round. THPT is `917504 / (last formal last-token − first formal submission)`, not the127s arrival span or later idle timestamp. Warmup is excluded only by its predeclared phase; no spikes or requests were removed.

| Round | Mean TTFT ms | Mean TPOT ms | Mean E2E ms | Total-token THPT tok/s | Formal window s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 403.30500297932304 | 19.03109878276599 | 19872.119057748932 | 6273.332979026521 | 146.25463100196794 |
| 2 | 393.21618992471485 | 19.016502763090106 | 19847.098516565893 | 6273.129500109898 | 146.2593750031665 |
| 3 | 399.9459093065525 | 18.98073687169182 | 19817.239729047287 | 6269.771959563796 | 146.33769871015102 |

`formal_metrics()` reproduces every stored aggregate exactly. `metric_values(native_client_rows(...))` agrees within1e-9; round2 TPOT's normalized sum differs by approximately3.6e-15ms. Existing p50/p95 diagnostics and full endpoint precision are retained in `client_receipt.json`.

**Existing three-round client stability: PASS.** All three complete independently warmed populations, valid endpoints, non-overlapping phase windows, seven proven boundaries and error-free replay/observations satisfy the existing client rule. The inspected canonical helpers and ticket18 define no numerical cross-round CV/spread threshold; none was added. This is not the full native stability/admission verdict: all-rank graph/runtime and ordered shutdown still require root acceptance. No pooled three-round mean is substituted.

## Retained spikes and exception

Repeat1 warmup's first two requests have TTFT **32873.955312 / 63383.023298ms** and E2E **51470.365559 / 82298.166314ms**. Both completedHTTP200 with1024 output tokens. Warmup windows are **83.684736 / 34.401140 / 34.256125s**. These rows remain intact; their specific JIT/server-side cause is not established here.

Formal TTFT ranges are **356.775–883.807 / 352.478–781.474 / 350.068–879.231ms**. All formal spikes remain in the means. Startup warnings were not equated with timed request failures, and no CPU-overhead analysis was performed.

One supplementary in-memory assertion incorrectly copied native05's observed50/50 lane counts as a requirement and failed at `after_repeat02_formal`:

```text
AssertionError: boundary_04_explicit_finish_counters_and_exact_totals
```

Actual D145/143 sums to288. The existing frozen boundary helper had already passed unchanged; it requires fresh role totals and all-lane zero gauges, not equal request allocation. Removed only the unsupported assertion from the in-memory check and reran existing helper acceptance. Full raw traceback is preserved in `client_receipt.json` and `/data/ycfeng/tmp/glm53_dsv4/native06_client_existing_helpers_20261023.stderr.log`, SHA256 `2138952e30eab6626d1d1bf908f4a122267e3d700829407a8568c74b9735cc97`. This failed supplementary attempt is not hidden or a native-run failure.

## Provenance, evidence and remaining limits

Original manifest SHA256: `2f42d7ef89387d776603b2f40469d463eeebfd5e28f47837bf45f686acbac41d`. Its preserved mtime is **2026-10-22T20:31:20.258972Z**; client and summary mtimes are **20:54:01.748749Z / 20:54:01.761858Z**. Launcher SHA256 `b8cf2968967a626f266ff3ce9b3fa8be55617391b327fede9596ccc0cf12c6a9` binds the same manifest/generation/job and execution-config hash. Unlike native05's recovered local-mtime limitation, native06's complete helper passes directly without copying, reconstruction or timestamp repair.

Environment `/usr/bin/python3`3.10.6, pandas2.0.1, no conda, `PYTHONDONTWRITEBYTECODE=1`, timeout60000ms. Comprehensive helper acceptance took0.143139s; decisive replay0.387867s. Exact command, PASS conditions and raw-output hashes are in `test_report_2026-10-23_client.md`; all28 accessed input/helper hashes, sizes and nanosecond mtimes are in `client_receipt.json`.

Local file census matches68 run files and32 rank records. This task did not inspect rank graph/runtime correctness. Root's retained platform proof reports Succeeded4/4, exit0/restarts0 and12.002222GPUh; no platform query was performed here. Actual native routing strategy/runtime proof also belongs to root R3, not the declared input label alone.

Native05 remains historically **UNVERIFIED**. No historical performance comparison or retrospective admission is made. Root must combine this client PASS with the frozen R3 full checker and independent native acceptance before admitting18. Ticket19 still needs an admitted reference and a qualified complete non-dummy predictor; no calibration, promotion or coverage gate is closed by this report.
