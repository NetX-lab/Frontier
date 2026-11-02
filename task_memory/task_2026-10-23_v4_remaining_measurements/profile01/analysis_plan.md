# Native V4 diagnostic-profile analysis plan (read-only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Corrected the observed filename/collector local-rank versus distributed/EP-rank distinction after the full32-trace audit and independent review. Actual outcomes are in results/primary_acceptance.json. |
| 2026-10-23 | W128-V4-PROFILE01-ANALYSIS-PLAN: mapped expected artifacts to collector IDs, chose the smallest existing parser path, separated directly verifiable checks from UNKNOWN gaps. No run, no source edit. |
| 2026-10-23 | Root factual correction applied: unsuffixed `post_warmup`/`pre_shutdown` snapshots; per-step batch identity instead of equal request sets (DP alignment UNKNOWN); `(role,lane,rank)` binding; hierarchy vs disjoint leaf accounting; `graph id` does not prove captured ownership. |
| 2026-10-23 | Removed the deliberate incompatible-CLI step 8 in favour of source evidence plus `trace_header`/`load_trace` reuse; marked `…` items as analysis steps; useful-batch candidate wording admits no zero-token step. |
| 2026-10-23 | W128-PROFILE02-ANALYSIS-BINDING: bound the sealed profile02 run root and trace directories, corrected the annotation-name and request-ID joins from source, replaced the stale "package unsealed" unknowns with the actual bindings, and separated trace-presence UNKNOWN from known source schema. No run, no cloud read, no source edit. |

Task `W128-V4-PROFILE01-ANALYSIS-PLAN`. Sealed package `candidate_h800_p0_profile_02` (packaging `d87f4c74`, merged `cc0227e9`, runtime `05952396`), RJob `exp-1023-180821-373637`, first-formal 64 worker steps, P16D16, TP8/DP2/EP16/PP1, 32×H800, `run_tag` `dsv4-h800-p0-profile-20261023-02`. This is the retained collection/analysis plan; actual I4 outcomes and unresolved gaps are recorded in [primary acceptance](results/primary_acceptance.json).

## 1. Expected artifact paths (bound to sealed config)

Run root `R` = `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261023/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261023-02`
(sealed `config.json → output_dir`; `native_pd_worker.py:190-195` requires it under `/mnt/codesign-exp/ycfeng` and creates `R/nodes/<host>/`).

Per node `<host>`, under `R`:
`R/nodes/<host>/` `result.json`, `ready.json`, `server_ready.json`, `drained.json`, `native_stopped.json`, `server.log`, `replay.log`, `worker.log`,
`serving/rank<N>.json` (inventory, `native_full_structure.py:620-639`),
`serving_profile/rank<N>.json` — the cumulative inventory/profile document, rewritten by `emit('post_warmup')` and `emit('pre_shutdown')`;
named snapshots exist only as `serving_profile/rank<N>.before_profile.json` and `serving_profile/rank<N>.profile.json`
(`native_serve_profile.py:409-431`; `suffix` is keyword-only and `:114,123` are its only callers, `:443,471` pass the stage positionally).
`R/run/` `client_requests.jsonl`, `canonical_repetitions.json`, `replay_summary.json`, `proxy_requests.jsonl` (`pd_replay.py:686,748-750,778-780`);
`R/replay_finished.json` (`native_pd_worker.py:363`), `R/run_manifest.json` and `R/launcher_receipt.json` (staged copies, `:117-132`).
Process status is outside `R`: `launcher_started.json`/`launcher_exit.json` beside `launch.py` (`:37-50`), which only record this package's launcher; no source writes a terminal receipt.
`profile01/profile02_platform_launch.json` is only the queued submission snapshot, not completion evidence — root must retain a new personal-client terminal snapshot incl. replica/container results at I3 completion.

Traces: base `torch_profiler_dir` = `R/torch_traces` (sealed `config.json`), rewritten per node to `<base>/<role>/dp<lane>` (`native_pd_worker.py:159-165`, `lane = position % nodes_per_role * local`), so exactly 4 directories `prefill|decode/dp0|dp1`, 8 traces each (`tensor_parallel_size=8`, `data_parallel_size_local=1`, `replica=4`).
Observed filename `dp{dp}_pp0_tp{tp}_dcp0_ep{ep}_rank{local_rank}.{time_ns}.pt.trace.json.gz`: filename and collector rank are0–7 in every lane; `ep` and `distributedInfo.rank` are0–15 per role. Although the source calls `get_worker_rank_suffix(global_rank=worker.rank)`, that argument is the observed local worker rank here (`vllm/v1/worker/gpu_worker.py:1156-1174`, `vllm/distributed/utils.py:696-740`). Gzip is enabled.
Also `profiler_out_<local_rank>.txt` per worker in the same directory (`profiler/wrapper.py:271-275`; `torch_profiler_dump_cuda_time_total` defaults true) — one window only.

Mapping: a trace `rank{N}` is **local to its lane**, so every join carries
`(role, lane, rank)` taken from `nodes/<host>/result.json → plan{role,lane,position}` (`native_pd_worker.py:135-179`) plus the trace's own directory
`<role>/dp<lane>`: trace `(role,lane,local_rank)` ↔ collector `rank` on that same node ↔ `serving/rank{N}.json` ↔ `serving_profile/rank{N}.json|profile.json`. For example decodeDP1 has `ep8_rank0` through `ep15_rank7`, never collector ranks8–15.
Joins are identity-based, never timestamp-based across hosts.
Collector annotation name `frontier_native_<category>|<id>|<scope>` (`native_serve_profile.py:180-182`) ↔ trace `user_annotation` event by the whole name;
per-step context (`step,kind,repeat,request_phase,has_prefill,token_phase,execution,graph_mode,startup_warmup,layer,category,scope`) from the row with that `<id>` (`:160-167,180-188,478-488`);
step request IDs are the native `cmpl-…` scheduler IDs, matched to client rows by stripping `cmpl-` and the trailing `-0[-8hex]`; the native ID contract is
`cmpl-fixed-6144-1024-req128-qps1-(warmup|formal)-r(\d{3})-repeat(\d{2})-0(?:-[0-9a-f]{8})?` (`native_full_structure.py:400-409`), client IDs being the same string without `cmpl-`/`-0[-8hex]`.

## 2. Reusable parser and the bounded corrections it needs

`tests/comparison/calibration/forward_device_timeline.py` is the only existing **gzip** native-torch reader: `trace_header` `:55-63`, `load_trace` `:79-81`,
`device_timeline` `:84-154`, correlation join `:90-93`, graph detection `:110`. Reuse `trace_header`/`load_trace` **by import** (module import is side-effect free).
Do **not** drive its CLI here: it hard-codes a `"Forward"` filter (`:88`) that does not match the collector's `frontier_native_<category>|<id>|<scope>` schema (whether those annotations reach the trace is itself UNKNOWN, check 4/step 5), and `add_all_reduce_minimum` `:169-182` needs one matching trace per peer
of the full world in the same directory. `rank_traces` `:66-76` itself only asks for one trace per rank of the trace's own `distributedInfo.world_size`, so whether a
`role/dp<lane>` directory of 8 satisfies it depends on that exported value — UNKNOWN until data, and nothing here should depend on it.
That source evidence is sufficient to skip those entry points; the native-annotation join is then added in the calling analysis, sized from the actual trace it reads.

`tests/analysis/v4_kernel_coverage.py` and `frontier/profiling/attention/dsa_kernel_trace.py` are **not** usable as-is:
`dsa_kernel_trace.load_trace_events:136-138` uses plain `open`; `parse_annotation_name:49-56` requires prefix `frontier_dsa` and 4 fields;
`aggregate_kernel_trace:88-92` raises on same-thread nested ranges and `:99-100` on repeats — native annotations nest and repeat across steps.
`v4_kernel_coverage.py:45-47,54-63` requires the isolated-prefill package/admission and `dsv4_raw_e0.jsonl`; the sealed native profile02 package does not supply that input contract.

Bounded correction (inline, not a new module): gzip-open; parse `frontier_native_<category>|<id>|<scope>` — three pipe-delimited fields with the category inside the prefix (`native_serve_profile.py:182`); assign a launch to **every**
enclosing annotation interval on its own `(pid,tid)` for coverage, while per-op sums take only the innermost owner; keep per-stream grouping.
Everything else — `cat in {cuda_runtime,cuda_driver}` → `args.correlation` → `cat=="kernel"`, `args.stream`, `args["External id"]` — is schema-compatible.

## 3. Minimum checks

1. Integrity: `result.json` model/kernel/source hashes and `serving_instrumentation.module_sha256`, `server_env_set` incl. `FRONTIER_NATIVE_PROFILE_LIMITS`
   (`native_pd_worker.py:233-241,279-303`), image digest from `run_manifest.json`/`manifest.yaml`, versions and `nvidia-smi -L`.
2. Window: every expected `serving_profile/rank{N}.profile.json` exists with `window.open==false` and `start_step < stop_step`, `dropped` explicit, `errors` listed;
   `max_iterations=64` stops the profiler *inside* the window (`profiler/wrapper.py:118-125`, stepped once per `Worker.execute_model`, `vllm/v1/worker/gpu_worker.py:931-946,1112`; CPU control `/data/ycfeng/tmp/glm53_dsv4/g2_native_iteration_control.json` = limit 3 → `states_before_execution [true,true,true,false,false]`) without closing it, so it closes only at the `/stop_profile` RPC (`pd_replay.py:489`)
   and `stop_step - start_step` is not retained coverage — report the actual `steps[]` count separately from the window span. Collector `limits` are capacity, never coverage.
3. Membership: `steps[]` rows with `context.repeat==1 and context.phase=='formal'` (`native_full_structure.py:394,414-415`), every `request_ids` entry matching the native
   regex with repeat01 and index<128, and every ID — after dropping `cmpl-` and the trailing `-0[-8hex]` — present in `canonical_repetitions.json → profiling.request_ids`
   (`pd_replay.py:461-462`). Anything else is not first-formal.
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

`R` is the run root bound in §1. Items 1-4 are complete runnable commands; items 5-7 are analysis steps/pseudocode ("Analysis step"), not validated commands.

1. `find $R/nodes -maxdepth 2 -name 'serving_profile' -o -maxdepth 2 -name 'serving'` ; `ls $R/nodes/*/serving_profile/`
2. `find $R -name '*.pt.trace.json.gz' | sort` — expect 4 dirs × 8 files, then `find $R -name '*.pt.trace.json.gz' | wc -l`
3. `python3 -c "import gzip,json,sys;p=sys.argv[1];d=json.load(gzip.open(p,'rt'));print(sorted(d));print(sorted({e.get('cat') for e in d['traceEvents']}))" <one trace>` — confirm gzip, header keys, `distributedInfo`, cats.
4. Same one-liner over one trace: print `sorted({tuple(sorted(e['args'])) for e in d['traceEvents'] if e.get('cat')=='kernel'})` and the same for `cudaEventRecordWithFlags`/`cudaStreamWaitEvent` — this settles the event-handle and stream-argument question empirically.
5. Analysis step: import `trace_header`/`load_trace` from `forward_device_timeline.py`, count `user_annotation` names starting `frontier_native_` per loaded trace against the `annotation` strings in `serving_profile/rank<N>.profile.json` — the join rate is the first real gate.
6. Analysis step: over `serving_profile/rank*.json` + `.profile.json`, bind `(role,lane,rank)` from `result.json → plan`, then per lane report first-formal step count, candidate step count admitted by check 4, and descriptor/mode/`has_prefill` per step (checks 3-9).
7. Analysis step: over `run/canonical_repetitions.json`, report `profiling.status`, per-role HTTP attempts, `drain.verified`, `receipt_errors`, `client_workload_verified`.

## 6. Planning-time verifiable vs UNKNOWN

The following were the pre-run evidence questions. Their actual outcomes are in `results/primary_acceptance.json`; they are not a current missing-artifact census.

Verifiable from artifacts: artifact presence and naming (`rank<N>.json`, `.before_profile.json`, `.profile.json` only), `(role,lane,rank)` binding, window bounds and drops,
sources/images/inputs, step membership, per-step descriptor/mode/`has_prefill`, client completion and drain, annotation↔collector id join, kernel→launch→annotation id via correlation,
coverage hierarchy plus leaf-only per-op sums, stream-id join when `trace_stream_id` is non-null, event record/wait call order and generations within the parallel helpers, terminal and shutdown ordering.
UNKNOWN until data: whether `R` and its 4 trace directories / 32 traces / per-node `serving*` files actually exist (package is sealed, cloud not yet read);
whether `distributedInfo` and `baseTimeNanoseconds` are exported (`forward_device_timeline.py:85,193` require both);
presence of event handles/stream args in kineto sync events; success of `cuptiGetStreamIdEx`; any shared cross-DP iteration identity, so no cross-lane alignment or merged count;
captured-segment Python ownership (`graph id` groups a replay, it does not name a scope); kernel→stream semantics for graph replay; cross-host clock alignment;
PDL attributes outside the two launchers; whether ≥3 distinct useful steps per lane survive the 64-step cap.
Soft-gate rule: every check above is offline and non-blocking; a failed check marks evidence UNVERIFIED and never aborts native work.
