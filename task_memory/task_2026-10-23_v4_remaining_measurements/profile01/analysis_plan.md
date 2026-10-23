# Native V4 diagnostic-profile analysis plan (read-only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | W128-V4-PROFILE01-ANALYSIS-PLAN: mapped expected artifacts to collector IDs, chose the smallest existing parser path, separated directly verifiable checks from UNKNOWN gaps. No run, no source edit. |
| 2026-10-23 | Root factual correction applied: unsuffixed `post_warmup`/`pre_shutdown` snapshots; per-step batch identity instead of equal request sets (DP alignment UNKNOWN); `(role,lane,rank)` binding; hierarchy vs disjoint leaf accounting; `graph id` does not prove captured ownership. |
| 2026-10-23 | Removed the deliberate incompatible-CLI step 8 in favour of source evidence plus `trace_header`/`load_trace` reuse; marked `…` items as analysis steps; useful-batch candidate wording admits no zero-token step. |

Task `W128-V4-PROFILE01-ANALYSIS-PLAN`. Source `05952396`, planned run first-formal 64 worker steps, P16D16, TP8/DP2/EP16/PP1, 32×H800. Package/launch owned elsewhere; nothing here is a runtime result.

## 1. Expected artifacts (paths, from driver code not yet sealed)

Per node `<host>`, under the run root `R` (package-owned absolute `torch_profiler_dir`, UNKNOWN):
`R/nodes/<host>/` `result.json`, `ready.json`, `server_ready.json`, `drained.json`, `native_stopped.json`, `server.log`, `replay.log`, `worker.log`,
`serving/rank<N>.json` (inventory, `native_full_structure.py:620-639`),
`serving_profile/rank<N>.json` — the cumulative inventory/profile document, rewritten by `emit('post_warmup')` and `emit('pre_shutdown')`;
named snapshots exist only as `serving_profile/rank<N>.before_profile.json` and `serving_profile/rank<N>.profile.json`
(`native_serve_profile.py:409-431`; `suffix` is keyword-only and `:114,123` are its only callers, `:443,471` pass the stage positionally).
`R/run/` `client_requests.jsonl`, `canonical_repetitions.json`, `replay_summary.json`, `proxy_requests.jsonl` (`pd_replay.py:747-750,778-780`);
`R/replay_finished.json`, `R/run_manifest.json`, `R/launcher_receipt.json`, `R/platform_terminal_receipt.json`.

Traces: `torch_profiler_dir` is rewritten per node to `<root>/<role>/dp<lane>` (`native_pd_worker.py:159-165`), so 4 directories
(`prefill|decode/dp0|dp1`), 8 TP traces each (`local=1`, `dp=2`, `nodes_per_role=2`, `native_pd_worker.py:135-179`).
Filename `dp{dp}_pp0_tp{tp}_dcp0_ep{ep}_rank{global_rank}.{time_ns}.pt.trace.json.gz` — `get_worker_rank_suffix` (`vllm/distributed/utils.py`)
with `profile_prefix=None` (route `profile/api_router.py:21-33` passes no prefix) and gzip on by default (`config/profiler.py:torch_profiler_use_gzip`).
Also `profiler_out_<local_rank>.txt` per worker in the same directory (`profiler/wrapper.py:267-275`), overwritten per window — one window only.

Mapping: a trace `rank{N}` is **not** globally unique — prefill and decode are separate instances that both number from 0, so every join carries
`(role, lane, rank)` taken from `nodes/<host>/result.json → plan{role,lane,position}` (`native_pd_worker.py:135-179`) plus the trace's own directory
`<role>/dp<lane>`:  trace `(role,lane,rank)` ↔ collector `rank` on that same node ↔ `serving/rank{N}.json` ↔ `serving_profile/rank{N}.json|profile.json`.
Joins are identity-based, never timestamp-based across hosts.
Collector annotation name `frontier_native|<id>|<scope>` (`native_serve_profile.py:180-182`) ↔ trace `user_annotation` event by the embedded `<id>`;
per-step context (`step,kind,repeat,request_phase,has_prefill,token_phase,execution,graph_mode,startup_warmup,layer,category,scope`) from the row with that `<id>` (`:160-188,478-488`);
step request IDs are the native `cmpl-…` scheduler IDs, matched to client rows by stripping `cmpl-` and the trailing `-0[-8hex]` (`native_full_structure.py:400-405`).

## 2. Reusable parser and the bounded corrections it needs

`tests/comparison/calibration/forward_device_timeline.py` is the only existing **gzip** native-torch reader: `trace_header` `:55-63`, `load_trace` `:79-81`,
`device_timeline` `:84-154`, correlation join `:90-97`, graph detection `:110`. Reuse `trace_header`/`load_trace` **by import** (module import is side-effect free).
Do **not** drive its CLI here: `rank_traces` `:66-76` requires exactly one trace per world rank in one directory, so it cannot accept a `role/dp<lane>` directory holding 8 of 16,
`add_all_reduce_minimum` `:169-182` needs the full world, and its `"Forward"`/`"Forward (graph)"` annotation name does not match the native annotations.
That source evidence is sufficient to skip those entry points; the native-annotation join is then added in the calling analysis, sized from the actual trace it reads.

`tests/analysis/v4_kernel_coverage.py` and `frontier/profiling/attention/dsa_kernel_trace.py` are **not** usable as-is:
`dsa_kernel_trace.load_trace_events:136-138` uses plain `open`; `parse_annotation_name:49-56` requires prefix `frontier_dsa` and 4 fields;
`aggregate_kernel_trace:88-92` raises on same-thread nested ranges and `:99-100` on repeats — native annotations nest and repeat across steps.
`v4_kernel_coverage.py:45-47,54-63` requires a sealed package, an acceptance receipt and `dsv4_raw_e0.jsonl`; none exist for this run.

Bounded correction (inline, not a new module): gzip-open; parse `frontier_native|<id>|<scope>`; assign a launch to **every**
enclosing annotation interval on its own `(pid,tid)` for coverage, while per-op sums take only the innermost owner; keep per-stream grouping.
Everything else — `cat in {cuda_runtime,cuda_driver}` → `args.correlation` → `cat=="kernel"`, `args.stream`, `args["External id"]` — is schema-compatible.

## 3. Minimum checks

1. Integrity: `result.json` model/kernel/source hashes and `serving_instrumentation.module_sha256`, `server_env_set` incl. `FRONTIER_NATIVE_PROFILE_LIMITS`
   (`native_pd_worker.py:233-241,279-303`), image digest from `run_manifest.json`/`manifest.yaml`, versions and `nvidia-smi -L`.
2. Window: every expected `serving_profile/rank{N}.profile.json` exists with `window.start_step < window.stop_step`, `dropped` explicit, `errors` listed;
   collector `limits` are capacity, never coverage.
3. Membership: `steps[]` rows with `context.repeat==1 and context.phase=='formal'`, every `request_ids` entry matching the inventory regex with repeat01 and index<128,
   and a subset of `canonical_repetitions.json` → `profiling.request_ids` (the 128 planned IDs). Anything else is not first-formal.
4. Useful batch: `local_num_tokens>0`, ≥1 annotation row for that step, and ≥1 kernel in **that rank's** trace correlated to a launch inside one of those annotations.
5. Distinctness: one first-formal worker step that passes check 4 is one logical useful batch; a zero-token worker step is not one, and `frozenset(request_ids)` is not the key —
   consecutive decode steps of one request share that set, and the two DP lanes legitimately serve different sets. Count distinct `(role, lane, step)` values among the candidates
   admitted by check 4; never sum `serving/rank*.json → counters.rows`.
6. Alignment: within a lane, the 8 TP ranks must agree on step ordinal, local request set, `synchronized_descriptor` (`num_tokens,num_reqs`) and `synchronized_mode`.
   Across the 2 DP lanes the collector keeps only per-lane step counters plus the all-reduced `dp_post_all_reduce` snapshot — no shared iteration ordinal — so cross-DP iteration
   correspondence is **UNKNOWN**; report each lane separately and claim no offset, merged count or ≥3 cross-DP aligned batches until a shared identity is actually observed.
7. Phase claim: take phase from the lane's own classifier input `has_prefill` (`native_serve_profile.py:153,485`) and state the DP-synchronized mode separately; DP forces a lane
   eager when another lane wants eager (`vllm/v1/worker/gpu/dp_utils.py:52-61`), so a lane's phase never implies the group's mode.
8. Report the actually captured `(num_requests, num_tokens, descriptor, mode)` tuples per lane; a 64-step cap can leave <3 useful decode steps (`wrapper.py` stop at `max_iters+1`).
9. Attribution: every enclosing annotation interval is valid for **coverage hierarchy**, but per-op cost uses the disjoint **leaf-most** owning scope only — never add parent and child
   kernels. Keep count, `sum_ms` and interval `union_ms` separate (`v4_kernel_coverage.py:32-40`); uncorrelated kernels are counted as unowned, not dropped.

## 4. Events, streams, PDL, capture — what the data can and cannot show

`synchronization` rows carry `event_object=id(event)`, `event_handle=int(event.cuda_event)`, `generation`, `source_stream`, `destination_stream`,
`recording_stream`, `scope`, `parallel_call` (`native_serve_profile.py:249-266`); a wait is usable only if a record of the same `event_object` precedes it with
`generation` equal to the wait's (re-record after the wait makes the pair ambiguous), and `source_stream` matches. Records exist **only inside**
`maybe_execute_in_parallel`/`execute_in_parallel` (`:275-276`), so edge coverage is partial by construction.
Stream join: `streams[key].trace_stream_id` from `cuptiGetStreamIdEx` (`:224-247`) against kernel `args.stream`.
Retained V4 trace evidence shows `cudaEventRecordWithFlags`/`cudaStreamWaitEvent` args are only `{cbid, correlation}` and kernels carry `graph id`/`graph node id`
(`audit/kernel_coverage_receipt_r2.json → trace_fields`) — so GPU event→trace-event mapping is UNKNOWN. Kernel `graph id`/`graph node id` only group kernels of one graph replay;
they identify neither the Python scope nor the captured segment that produced them, so captured-segment ownership stays UNKNOWN.
Same-stream overlap and PDL: reuse the reversal/overlap census (`v4_kernel_coverage.py:168-181`) as an observation only; enqueue order is not completion order and no critical path or additivity follows.
PDL launch attributes are retained only at the two wrapped Triton launchers (`native_serve_profile.py:184-186,49-52`); other operators stay UNKNOWN.

## 5. Smallest next actual-data steps (read-only, after artifacts exist)

`R=<run root>`. Items 1-4 are complete runnable commands; items 5-7 are analysis steps/pseudocode ("Analysis step"), not validated commands.

1. `find $R/nodes -maxdepth 2 -name 'serving_profile' -o -maxdepth 2 -name 'serving'` ; `ls $R/nodes/*/serving_profile/`
2. `find $R -name '*.pt.trace.json.gz' | sort` — expect 4 dirs × 8 files, then `find $R -name '*.pt.trace.json.gz' | wc -l`
3. `python3 -c "import gzip,json,sys;p=sys.argv[1];d=json.load(gzip.open(p,'rt'));print(sorted(d));print(sorted({e.get('cat') for e in d['traceEvents']}))" <one trace>` — confirm gzip, header keys, `distributedInfo`, cats.
4. Same one-liner over one trace: print `sorted({tuple(sorted(e['args'])) for e in d['traceEvents'] if e.get('cat')=='kernel'})` and the same for `cudaEventRecordWithFlags`/`cudaStreamWaitEvent` — this settles the event-handle and stream-argument question empirically.
5. Analysis step: import `trace_header`/`load_trace` from `forward_device_timeline.py`, count `user_annotation` names starting `frontier_native|` per loaded trace against annotation `id`s in `serving_profile/rank<N>.profile.json` — the join rate is the first real gate.
6. Analysis step: over `serving_profile/rank*.json` + `.profile.json`, bind `(role,lane,rank)` from `result.json → plan`, then per lane report first-formal step count, candidate step count admitted by check 4, and descriptor/mode/`has_prefill` per step (checks 3-9).
7. Analysis step: over `run/canonical_repetitions.json`, report `profiling.status`, per-role HTTP attempts, `drain.verified`, `receipt_errors`, `client_workload_verified`.

## 6. Directly verifiable vs UNKNOWN

Verifiable from artifacts: artifact presence and naming (`rank<N>.json`, `.before_profile.json`, `.profile.json` only), `(role,lane,rank)` binding, window bounds and drops,
sources/images/inputs, step membership, per-step descriptor/mode/`has_prefill`, client completion and drain, annotation↔collector id join, kernel→launch→annotation id via correlation,
coverage hierarchy plus leaf-only per-op sums, stream-id join when `trace_stream_id` is non-null, event record/wait call order and generations within the parallel helpers, terminal and shutdown ordering.
UNKNOWN until data: exact `torch_profiler_dir` and engine JSON (package not sealed); whether `distributedInfo` and `baseTimeNanoseconds` are exported;
presence of event handles/stream args in kineto sync events; success of `cuptiGetStreamIdEx`; any shared cross-DP iteration identity, so no cross-lane alignment or merged count;
captured-segment Python ownership (`graph id` groups a replay, it does not name a scope); kernel→stream semantics for graph replay; cross-host clock alignment;
PDL attributes outside the two launchers; whether ≥3 distinct useful steps per lane survive the 64-step cap.
Soft-gate rule: every check above is offline and non-blocking; a failed check marks evidence UNVERIFIED and never aborts native work.
