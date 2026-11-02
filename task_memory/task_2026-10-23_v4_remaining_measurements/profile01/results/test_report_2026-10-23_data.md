# Profile02 retained-data CPU audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Root accepted the diagnostic audit after exact full32-trace reproduction; large raw audit/index files are archived in the personal cloud and referenced by archive_receipt.json. |
| 2026-10-23 | W128-PROFILE02-DATA: completed all 32 traces, source/inventory/client/window checks and raw-index verification; retained all admission refusals. |

## Outcome

**CPU evidence audit complete; native-reference, serving-total and operator-cost admissions remain UNVERIFIED.** No platform operation or source/admission change occurred. Primary acceptance belongs to root.

| Check | Expected / passing condition | Observed | Result |
| --- | --- | --- | --- |
| Download integrity | Every downloaded path has the inventory's exact bytes and SHA256 | 226/226 files, 4,549,518,256 bytes; zero mismatch | PASS |
| Frozen package bytes | All source-binding files match | 45/45 SHA256 matches, runtime `05952396`, packaging `d87f4c74` | PASS |
| Native source / inputs | Installed-source receipts match pinned local source, approved overlay and manifest inputs | 87/87 native source checks, 40/40 manifest entries; all four nodes agree | PASS byte identity only |
| Platform completion | Four successful replicas, exit 0, restart 0 and locked image | Supplied terminal receipt matches all four | PASS receipt audit only |
| Client completion | 432 unique complete requests with valid outputs | 432/432 HTTP200/completed/stream_done, 1,024 output tokens each; zero validation errors | PASS |
| Profiling RPC / drain | Both roles start/stop successfully, first formal drains | Four HTTP200 attempts, drain verified; original profiling status still UNVERIFIED | PASS fields only |
| Collector window | 32 closed windows; actual retained steps reported separately from span | 32/32 closed, 64 steps each; zero drops/suppressed errors | PASS |
| Collector snapshots | Preserve startup, stopped and cumulative records | 96/96 snapshots: 32 each of before_profile/profile_stopped/pre_shutdown; zero drops | PASS |
| Annotation join | Complete exact-name join without duplicate IDs/names | 326,328/326,328 matched; no missing or duplicate names | PASS |
| Useful-step membership | Native IDs normalize to first-formal repeat01 client IDs; tokens and correlated kernels present | Every admitted useful row passes; per-lane counts below | PASS membership |
| TP agreement | Eight TP ranks agree on ordinal, request set, descriptor and mode for useful steps | All four lanes pass; zero observed-field mismatch | PASS useful rows only |
| At least three useful steps per lane | Each lane retains at least three distinct `(role,lane,step)` values | Prefill DP1 has only 2 | FAIL evidence sufficiency |
| Kernel preservation / partition | Retain every kernel once in the raw index and partition ownership status | 7,600,008/7,600,008 unique original-event indices; 32 SHA256/status histograms match | PASS |
| Event helper generations | Wait has a preceding matching object/generation/source record | 44,800/44,800 waits paired; no unpaired wait or duplicate record generation | PASS host-helper census only |
| CUDA stream mapping | Actual collector trace stream IDs available | 0/192 mapped; all 192 errors are `libcupti.so` not found | FAIL mapping |
| Captured per-op ownership | Trace identifies each captured segment's Python operation | 6,802,600 graph kernels only have replay/envelope evidence | UNKNOWN; not admitted |

## Per-lane first-formal coverage

All eight TP ranks in each lane agree on these counts. Rows are counted by step ordinal, not request set. No cross-DP merged or aligned count is claimed.

| Role / lane | Window start → stop | Span, not coverage | Retained steps | Zero-token steps | Distinct useful steps | ≥3 |
| --- | --- | --- | --- | --- | --- | --- |
| prefill / DP0 | 30 → 139 | 109 | 64 | 41 | 23 | Yes |
| prefill / DP1 | 28 → 433 | 405 | 64 | 62 | 2 | **No** |
| decode / DP0 | 2169 → 4740 | 2571 | 64 | 3 | 61 | Yes |
| decode / DP1 | 1102 → 3757 | 2655 | 64 | 3 | 61 | Yes |

`data_audit.json → lanes[].observed_useful_tuples` retains every actual request/token/descriptor/mode/has_prefill tuple and normalized request IDs; `traces[].steps` retains all 2,048 rank-step rows, including zero-token rows. A total of 109 distinct zero-token lane steps (872 rank rows) omit synchronized descriptor/mode. These missing fields are preserved, not filled in; every useful row has complete required fields and agrees across its TP peers. Cross-DP iteration correspondence remains UNKNOWN.

## Kernel ownership and dependency limits

The kernel partition is 732,792 unique non-graph innermost owners, 6,802,600 captured-segment UNKNOWN kernels and 64,616 unowned kernels. Uncorrelated, ambiguous-correlation and ambiguous-innermost counts are each zero in this data; those states remain represented. Graph ID never establishes captured per-op ownership. Unique non-graph ownership includes retained dummy/capture/eager execution contexts and is not itself an operator-cost admission.

Hierarchy metrics count all enclosing-range memberships and are non-additive. Leaf diagnostic metrics are disjoint, exclude graph replay and retain execution context. Each trace/group reports `count`, raw-duration `sum_ms` and interval `union_ms` separately; they must not be summed across nested scopes, streams or ranks into serving latency. All 7,600,008 kernel raw-index rows remain available, with original-event indices resolving every original name, time, duration, correlation, stream and graph field.

Observed same-exported-stream prior-interval overlaps: **926,301**. Adjacent unique-correlation launch-start reversals: **0**. These are observations, not completion order, PDL proof, an additive cost model or a critical path. Graph stream IDs are not guessed to be native collector streams.

Helper synchronization retains 38,160 records and 44,800 waits, all on prefill. Decode retains zero helper synchronization rows. Generation and source matching therefore proves only observed host-helper ordering; helper coverage is partial by construction. Trace event-record/wait calls lack event/stream-handle arguments, and every collector `trace_stream_id` is null. GPU event-to-trace mapping and complete graph dependency coverage remain UNKNOWN.

Prefill retains 10,320 `_fused_q_kv_rmsnorm_kernel` samples with `launch_pdl=True` and 10,320 `_fused_inv_rope_fp8_quant_per_head` samples with `launch_pdl=True, USE_GDC=True`. It also retains compression-call `pdl_kwargs.launch_pdl=False` samples (14,880 save_partial_states; 5,040 Triton and 9,840 CuteDSL compress calls). These are Python argument observations, not complete per-kernel launch attributes. Decode retains no such argument samples. Driver/runtime trace key censuses remain in each trace record.

## Identity and runtime fields

Collector and filename rank are local 0–7 on every lane; `distributedInfo.rank` and filename EP rank are 0–15 per role. Every header host, local rank, lane and EP/distributed identity is cross-checked and retained. Each trace exports `world_size=16`, `baseTimeNanoseconds`, vLLM0.28.0 and H800 device properties. A directory's eight traces are not a complete 16-rank role world; the existing reader CLI was not used.

The observed image digest matches `2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`; node versions report torch `2.13.0+cu130`, vLLM `0.28.0`, aiohttp `3.14.3`. All original node `qualification_verified=false`, diagnostic-only/reference-admission refusals, inventory errors and weight-audit fields are retained. Local auditing uses Python3.10.6, not the worker's Python runtime.

Downloaded shutdown evidence differs from the earlier two-headless-node summary: prefill DP1 retains `observation_errors=[]` and `head_stopped_before_headless_stop=true`; decode DP1 retains `Head shutdown marker missing; native stop order unverified` and that field is false. Both result and native_stopped documents agree; the log search found no additional matching message. This audit does not resolve or alter the separate root-owned shutdown RCA. `launcher_receipt.json` is absent from the downloaded run root and stays explicitly missing.

## Commands, timings and artifacts

Exact per-trace argument arrays/log paths/source hashes are in `data_audit.json → execution.commands.commands[]`; the executable pattern, supplemental inspections and original CPU inspector failures are in [command.md](command.md). All memory-heavy reads used `timeout 60s` inside a systemd scope with `MemoryMax=30G`. At most two traces were open concurrently. No GPU, Docker, network, cloud read, StepMind, Git write, repository source/test/input edit or further delegation occurred.

- Temporary task adapter: `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py`, **197 lines**, SHA256 `7ed0ea45a61d5150e2f45bbc1cdf245f2331e40a351c235228b209b064ede0c8`.
- 32/32 per-trace commands exited0; maximum analysis time26.024345s, maximum command wall time27.291087s; maximum per-process RSS2,631,736KiB, below the30-GiB scope.
- Independent raw-index verifier: **PASS**,32 indices /7,600,008 rows,33.370841s. Receipt: `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/raw_index_verification.json`.
- Full audit: `data_audit.json`,53,441,663bytes, SHA256 `5dbdd16b882f285da51e31f18ed1200b44c22829b596c5a405792a342687da9e`; cloud location and per-file hashes are in [archive_receipt.json](archive_receipt.json). The large raw file is retained locally for analysis and in that archive rather than duplicated in Git.
- Original remote inventory SHA256 `33ed03ea23e094deb89039b1dde2bd4bd80cbff288bd017867bbfe85411ed566`; all226 original downloaded files were independently rehashed.

The first final-document checker required the literal `UNVERIFIED` in both Markdown files; command.md described separate qualification but omitted that literal. Its AssertionError is retained in the execution transcript. The command record now states the unchanged admission boundary explicitly; no source, data, trace or measurement was rerun for this document correction.

## Progress checkpoint and remaining work

The I4 audit deliverable is complete, not a calibration or admission milestone. The required checks stopped at concrete missing evidence: prefill DP1 has two useful steps; CUDA stream/event mapping is unavailable; captured per-op ownership and cross-DP correspondence are absent. No parser generalization, repeated workload run, dependency guess, timing scale or CPU-gap classification was added.

The required `check-progress` self-check found D1–D7 CLEAR and two S5 fact corrections for the original document owners. Its full four-section record is retained in `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/checkpoint.md`. The three-file final check passed as `PASS_FINAL_LOCAL_ARTIFACT_CHECK`; receipt `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/final_verification.json` retains file hashes and the raw-index verification hash. All local audit subtasks are complete; root independent acceptance remains pending.

Root should independently accept these three files, retain the actual one-node shutdown-field discrepancy for its RCA, and decide any authorized follow-up evidence collection. The stale global-rank wording in `../analysis_plan.md` and two-headless-marker summary in task progress should be corrected by their owner; this agent does not own those files. Native reference, serving/op cost, workflow/gap calibration and CPU-overhead-last boundaries remain unchanged.

Root completion: [primary_acceptance.json](primary_acceptance.json) accepts the diagnostic audit with gaps; [root_verification.json](root_verification.json) reproduces all32 non-clock trace receipts and every decompressed kernel index. [review.md](review.md) records the independent review and upstream document corrections. [shutdown_rca.md](shutdown_rca.md) limits the shutdown conclusion to the failed receipt and explicitly retains cross-host timing uncertainty.
