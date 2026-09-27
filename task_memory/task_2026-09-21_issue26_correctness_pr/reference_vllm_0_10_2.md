# Reference Behavior: pinned vLLM checkout

Authoritative reference table for validating a Frontier simulator change.
All citations are `path:line` relative to the checkout root
`/data/ycfeng/Frontier/.real-engine/vLLM-BS` unless written absolute.
No server was started; this is pure source reading at a fixed revision.

---

## Part 1 — Upstream relationship

### 1.1 Identity

| Item | Value | How established |
|---|---|---|
| Checkout | `/data/ycfeng/Frontier/.real-engine/vLLM-BS` | — |
| HEAD (verified) | `ea95f571e20937c7c908c6d59ddd1cd6bf9268f1` | `git rev-parse HEAD` |
| HEAD subject | `Instrument vLLM attention ops for Frontier calibration` | `git log --oneline -1` |
| Working tree | clean (no modified/untracked files) | `git status --short` → empty |
| `origin` | `https://github.com/fwyc0573/vLLM-BS.git` | `git remote -v` |
| `upstream` (added read-only this session) | `https://github.com/vllm-project/vllm.git` | `git remote add upstream …` |
| Tag fetch | **SUCCEEDED** — `refs/tags/v0.10.2` → local `refs/tags/upstream-v0.10.2` | `git fetch --no-tags upstream refs/tags/v0.10.2:refs/tags/upstream-v0.10.2` |
| **Resolved commit of upstream v0.10.2** | **`01efc7ef781391e744ed08c3292817a773d654e6`** (lightweight tag; `git cat-file -t` → `commit`) | `git rev-parse refs/tags/upstream-v0.10.2^{commit}` |
| **merge-base(HEAD, v0.10.2)** | **`01efc7ef781391e744ed08c3292817a773d654e6`** — i.e. the tag commit itself | `git merge-base HEAD refs/tags/upstream-v0.10.2^{commit}` |
| **Is HEAD a descendant of the tag?** | **YES.** `git merge-base --is-ancestor` returns 0 | — |
| Distance | HEAD is **40 commits ahead, 0 behind** | `git rev-list --left-right --count v0.10.2...HEAD` → `0  40` |

Because the merge-base equals the tag commit and HEAD is behind by zero, the
fork is a **clean linear/merge-free descendant of v0.10.2**: nothing from
upstream after v0.10.2 is present, and no upstream commit was dropped. The tag
comparison below is therefore exact, not approximate.

`vllm/version.py:5` reads the version from a generated `vllm/_version.py`
(setuptools-scm) and falls back to `"dev"` at `vllm/version.py:13`; it carries
no hard-coded version string, so the git tag comparison — not the source tree —
is the authority here.

### 1.2 Files differing from v0.10.2, inside the requested paths

Requested paths: `vllm/v1/engine/`, `vllm/v1/core/sched/`, `vllm/forward_context.py`,
`vllm/v1/worker/gpu_model_runner.py`, `vllm/model_executor/layers/fused_moe/`,
`vllm/model_executor/models/qwen3_moe.py`.

**Unchanged (byte-identical to v0.10.2) — important for Part 2 A/B/E:**

| File | Status |
|---|---|
| `vllm/v1/engine/core_client.py` | **UNCHANGED** — every answer in section A is pristine upstream v0.10.2 behavior |
| `vllm/v1/engine/coordinator.py` | **UNCHANGED** — every answer in section B is pristine upstream v0.10.2 behavior |
| `vllm/forward_context.py` | **UNCHANGED** |
| `vllm/v1/core/sched/{async_scheduler,interface,output,request_queue,utils}.py` | **UNCHANGED** |
| `vllm/v1/engine/{__init__,async_llm,llm_engine,utils,detokenizer,logprobs,exceptions,parallel_sampling}.py` | **UNCHANGED** |
| `vllm/distributed/device_communicators/all2all.py` | **UNCHANGED** |

Verified per-file with `git diff --numstat <tag> HEAD -- <file>` returning no rows.

**Changed, with characterization** (`+add/-del` from `git diff --numstat`):

| File | +/- | Characterization |
|---|---|---|
| `vllm/v1/engine/core.py` | +43/-0 | **BEHAVIOR CHANGE, env-gated (default OFF).** Adds a "wait for all initial requests before the first scheduling step" barrier in `EngineCoreProc._process_input_queue`, gated on `VLLM_FRONTIER_WAIT_INITIAL_REQUESTS=="1"` **and** `VLLM_FRONTIER_EXPECTED_NUM_REQUESTS>0` (`core.py:61-67`, guard at `core.py:763-766`). With defaults (`"0"` / `0`) the branch is dead and upstream control flow is bit-identical. |
| `vllm/v1/engine/output_processor.py` | +71/-2 | **Instrumentation.** Populates the previously-unused `RequestOutput.metrics` field with a new `FrontierRequestMetrics` record, built only for finished leaf requests (`output_processor.py:258-262`). It reads existing `RequestStateStats` timestamps; it does not alter token generation, stopping, or detokenization. |
| `vllm/v1/engine/processor.py` | +8/-1 | **BEHAVIOR CHANGE, env-gated (default OFF).** `VLLM_FRONTIER_FORCE_ARRIVAL_TIME_ZERO=="1"` pins `arrival_time = 0.0` instead of `time.time()` (`processor.py:33-35`, `processor.py:342-345`). Default `"0"` preserves upstream. |
| `vllm/v1/core/sched/scheduler.py` | +471/-8 | **Mostly instrumentation + 2 real behavior changes.** See 1.3. |
| `vllm/v1/worker/gpu_model_runner.py` | +514/-65 | **Mixed: env-gated instrumentation + 6 UNGATED behavior changes.** See 1.4. The behavior changes are inert on a non-spec-decode, PP=1 run, but they are inert because the spec-decode predicate is false — **not** because they are flag-gated. |
| `vllm/model_executor/models/qwen3_moe.py` | +206/-21 | See 1.3 / section F. |
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | +230/-146 | See 1.3 / section F. |
| `vllm/model_executor/layers/fused_moe/layer.py` | +125/-89 | See 1.3 / section F. |
| `vllm/model_executor/layers/fused_moe/deepep_ht_prepare_finalize.py` | +32/-27 | See 1.3 / section F. |
| `vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py` | +23/-17 | See 1.3 / section F. |
| `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py` | +34/-27 | See 1.3 / section F. |
| `vllm/model_executor/layers/fused_moe/configs/README → configs/specific-README` | 0/0 | **Pure rename**, 100% similarity (`git diff --find-renames --summary`). No content change. |

### 1.3 `vllm/v1/core/sched/scheduler.py` — the only 3 non-instrumentation edits

The other ~460 added lines are logging scaffolding: a `VLLM_FLOW_VALIDATION`
logger (`scheduler.py:41-75`), a `VLLM_FRONTIER_SCHED_LOG_PATH` /
`VLLM_FRONTIER_SCHED_DECISION_LOG_PATH` JSONL logger (`scheduler.py:87-96`), and
`_emit_frontier_schedule_decision` (`scheduler.py:283-334`). All are no-ops when
the env vars are unset (`_log_flow` early-returns; `_emit_frontier_schedule_decision`
returns at `scheduler.py:299-300` when the logger is `None`).

| # | Change | HEAD line | Upstream | Verdict |
|---|---|---|---|---|
| 1 | `skipped_waiting_requests.prepend_request(request)` → `.add_request(request)` at **6 call sites** | `scheduler.py:645, 656, 666, 690, 727, 800` | `prepend_request` | **BEHAVIOR CHANGE, always on.** Commit `dcb9709e0` "Fix skipped-waiting FCFS order in scheduler". The temporary `skipped_waiting_requests` queue is re-prepended wholesale to `self.waiting` at `scheduler.py:882-883`; building it with `prepend_request` reversed the relative order of skipped requests, `add_request` preserves FCFS order. Changes the order requests are retried, not which requests are admitted. |
| 2 | `preempted_req = self.running.pop()` → `preempted_req = self.running[-1]` … `self.running.pop()` | peek at `scheduler.py:493`, pop at `scheduler.py:507` | single `pop()` | **Instrumentation only / behaviorally identical.** Split into peek + pop so the victim's pre-removal state can be logged (`scheduler.py:494-505`); the element is still removed at `:507`. |
| 3 | `num_accepted = len(generated_token_ids) - 1` → `min(_get_num_accepted_spec_tokens(...), num_draft_tokens)` | `scheduler.py:1298-1301`, helper at `scheduler.py:78-84` | bare `len(...) - 1` | **BEHAVIOR CHANGE, always on.** Clamps below at 0 (`return max(len(generated_token_ids) - 1, 0)`, `scheduler.py:84`) and above at `num_draft_tokens`. Feeds `num_rejected` at `scheduler.py:1302`, which decrements `request.num_computed_tokens` at `:1308`. Only reachable with speculative decoding enabled; on the non-spec path `scheduled_spec_token_ids` is falsy and the block is skipped. |

### 1.4 `vllm/v1/worker/gpu_model_runner.py` — instrumentation vs ungated behavior

Abbreviated `GMR` below. Instrumentation clusters (all default-off):

| Cluster | HEAD lines | Gate / default |
|---|---|---|
| Frontier env flags, `frontier_trace` / CUDA-event / MoE-routing logger imports | `GMR:6-7, 13, 17-42, 52, 117-120` | `VLLM_FRONTIER_INSTRUMENTATION`, default `"0"` (`GMR:18-19`) |
| Instrumentation config validation + logger construction in `__init__` | `GMR:537-620` | same; `VLLM_FRONTIER_RUNTIME_META_ENABLED` default `0`, `VLLM_FRONTIER_OP_TIMING_MODE` default `"record_function"` |
| Per-batch trace/metric capture in `execute_model` (CUDA events, JSONL batch log, PP-boundary timestamps, scopes around `set_forward_context`) | `GMR:2240-2245, 2302-2391, 2394-2396, 2407-2459` | `FRONTIER_INSTRUMENTATION_ENABLED and frontier_trace.is_active()` (`GMR:2302-2303`). **When ON it forces `torch.cuda.synchronize()` at `GMR:2428`** — perturbs timing, not values |
| Per-request positions metadata in `_prepare_inputs` | `GMR:1131-1145` | `FRONTIER_RUNTIME_META_ENABLED` (default off) |
| New `get_frontier_batch_metrics()` / `clear_frontier_batch_metrics()` | `GMR:2571-2589` | additive; returns `[]` when off (`GMR:2581-2582`) |
| Attention-backend admission check in `initialize_attn_backend` | `GMR:3638-3671` | `FRONTIER_INSTRUMENTATION_ENABLED` |

**Ungated behavior changes** — no env var, no config flag:

| # | Change | HEAD lines | Inert when |
|---|---|---|---|
| 1 | Runner buffers resized by `_get_max_num_input_tokens = max_num_batched_tokens + max_num_reqs * num_speculative_tokens`; applied to `InputBatch`, `input_ids`, `positions`, `inputs_embeds`, `mrope_positions`, `arange_np`, `kv_sharing_fast_prefill_logits_indices`, `make_empty_intermediate_tensors` | def `GMR:191-197`; uses `GMR:319-328, 413, 443, 445, 453, 475, 492, 505, 3196, 3788` | `speculative_config is None` (term becomes `max_num_reqs * 0`) |
| 2 | `_prepare_inputs` folds `num_draft_tokens` into per-request input token counts, changing `req_indices`, `positions`, `input_ids`, slot mapping, `query_start_loc`, `seq_lens`, `num_actual_tokens`, `max_query_len` | `GMR:1102-1114, 1119-1127, 1164-1169, 1178-1190, 1199, 1211-1213, 1255-1281, 1326-1330`; `_calc_spec_decode_metadata` `GMR:1476-1501`; helper `GMR:176-188` | `num_draft_tokens is None` → helper returns input unchanged (`GMR:186-188`) |
| 3 | Cudagraph/DP padding keyed on a draft-inclusive count | `GMR:1977-1998` (`total_num_input_tokens = num_scheduled_tokens + sum(len(token_ids) for token_ids in scheduler_output.scheduled_spec_decode_tokens.values())`), `:2016, :2021` | no scheduled spec tokens → sum is 0 |
| 4 | EAGLE draft request reordering: `drafter.prepare_inputs` returns a 3-tuple incl. `request_order`; `_apply_request_order` / `_restore_request_order` | def `GMR:200-227`; uses `GMR:2648, 2670-2671, 2682-2688, 2699-2700`; paired with forked `vllm/v1/spec_decode/eagle.py:624-629` (upstream returned a 2-tuple) | EAGLE not in use |
| 5 | Mixed dummy-batch layout clamp: upstream `num_decode_tokens = num_tokens // 2` → `min(num_tokens // 2, max(max_num_reqs - 1, 0))` | `GMR:164-173` | `num_tokens // 2 <= max_num_reqs - 1`; affects **warmup shape only** |
| 6 | `hasattr(self, "drafter")` guards around `drafter.dummy_run` and `validate_same_kv_cache_group` | `GMR:3235-3236, 4028-4029` | PP=1; `self.drafter` only exists under `if self.speculative_config and get_pp_group().is_last_rank:` (`GMR:379-390`) — this is a crash fix for PP>1 + spec decode |

Also outside the requested paths but relevant: `vllm/distributed/parallel_state.py` (+11/-0) wraps
`GroupCoordinator.dispatch`/`combine` in `record_function_or_nullcontext("expert_parallel_alltoall_dispatch" / "..._combine")` when `self.unique_name.startswith("ep:")` (`parallel_state.py:843-848, 856-860`). Instrumentation only — `record_function_or_nullcontext` is a `nullcontext` unless `VLLM_CUSTOM_SCOPES_FOR_PROFILING` or the Frontier contextvar logger is live (`vllm/v1/utils.py:1179-1191`).

---

## Part 2 A — `vllm/v1/engine/core_client.py` (file is byte-identical to v0.10.2)

Class picked at runtime: `CoreEngineProcManager.make_async_mp_client` returns
`DPLBAsyncMPClient` for `data_parallel_size > 1` **without** external LB
(`core_client.py:96-101`), `DPAsyncMPClient` for external LB (`:99`), and plain
`AsyncMPClient` for DP=1 (`:102`). Only `DPLBAsyncMPClient` scores engines.

| # | Question | Answer | file:line | Verbatim |
|---|---|---|---|---|
| A1 | How are DP engines **ordered**? | Engines are held in **rank order** in `self.core_engines`, built from `engine_ranks_managed = range(dp_rank, dp_rank + num_ranks)`. The scan is a full linear pass over all of them, but **rotated** to start at `self.eng_start_index` so that different frontends break ties on different engines. Ties keep the **first** engine visited in rotated order (strict `<`). | order: `core_client.py:480-489`; comment "Engines are in rank order." `core_client.py:1134`; rotation `core_client.py:1141-1144`; start index `core_client.py:1129-1130` | `idx = (self.eng_start_index + i) % num_engines`  /  `self.eng_start_index = (len(self.core_engines) * self.client_index) // client_count` |
| A1 | How are they **scored**? | Each engine carries a `[waiting, running]` pair; the score is a **weighted sum with waiting weighted 4x**, and the minimum score wins. | `core_client.py:1145-1149` | `waiting, running = current_counts[idx]`<br>`score = waiting * 4 + running`<br>`if score < min_score:`<br>`    min_score = score`<br>`    eng_index = idx` |
| A2 | What does a single frontend **reserve locally** after picking? | It increments **only the `waiting` element** of its own local `lb_engines` row for the chosen engine, **by `self.client_count`** (not by 1) — a deliberate over-reservation so that N frontends racing between coordinator updates spread out rather than all piling onto the same engine. `running` is not touched. | `core_client.py:1150-1152` | `# Increment local waiting count for better balancing between stats`<br>`# updates from the coordinator (which happen every 100ms).`<br>`current_counts[eng_index][0] += self.client_count` |
| A2 | When is that reservation **reset/replaced**? | It is **never decremented**. It is **wholesale replaced** — not merged — the next time the coordinator publishes counts: `self.lb_engines` is rebound to a freshly decoded slice, discarding every local increment accumulated since the last update. Note `current_counts` at `:1136` is a *reference* to `self.lb_engines`, so the `+=` mutates the live list until that rebind. | replace: `core_client.py:1074-1078`; alias: `core_client.py:1136`; init: `core_client.py:979` | `counts, wave, running = msgspec.msgpack.decode(buf)`<br>`…`<br>`sliced_counts = counts[count_slice]`<br>`self.lb_engines = sliced_counts` |
| A3 | Are coordinator updates applied as **replace** or **incremental adjust**? | **Full replace (absolute snapshot).** The coordinator sends absolute `[waiting, running]` counts for *all global* engines; the client slices out the engines it manages and rebinds `self.lb_engines` to that slice. There is no delta arithmetic anywhere. Additionally, the receive loop **drains the socket and keeps only the latest** message before applying it, so intermediate snapshots are dropped. | rebind `core_client.py:1078`; slice `core_client.py:1005-1006`; drain `core_client.py:1065-1073` | `count_slice = slice(self.engine_ranks_managed[0],`<br>`                    self.engine_ranks_managed[-1] + 1)`<br>and `# Drain all stats events (we only care about latest).` |
| A4 | What changes with **multiple frontends**? | Three things, all driven by `client_count`/`client_index`: (a) the scan **start offset** differs per frontend, so empty engines are not all picked in the same order; (b) the local reservation step is **`client_count`** instead of 1, modelling the other frontends' likely concurrent picks; (c) each frontend applies only its own **`count_slice`** of the global count vector. The frontends never talk to each other — the only shared state is the coordinator's periodic broadcast. | (a) `core_client.py:1129-1130`, `:1144`; (b) `core_client.py:1152`; (c) `core_client.py:1005-1006`, `:1077` | `self.eng_start_index = (len(self.core_engines) * self.client_index) // client_count` |
| A4 | What changes with an **explicit DP rank**? | The walrus at `:1135` short-circuits: if `request.data_parallel_rank is not None`, `eng_index` is taken directly from it and **the entire scoring loop AND the local `waiting` reservation are skipped** — a pinned request neither consults nor perturbs the load-balancer state. The rank is range-validated earlier in the frontend. | skip `core_client.py:1135`; chosen `core_client.py:1154`; validation `vllm/v1/engine/processor.py:336-340` | `if (eng_index := request.data_parallel_rank) is None:`<br>validation: `if data_parallel_rank is not None and not (0 <= data_parallel_rank < data_parallel_size): raise ValueError(...)` |
| A4 | External-LB mode | With `data_parallel_external_lb`, the plain `DPAsyncMPClient` is used and selection is a constant — no scoring at all, and (per `coordinator.py:53-55`) the engines publish no stats. | `core_client.py:97-99`; `core_client.py:1103-1104` | `def get_core_engine_for_request(self, request: EngineCoreRequest):`<br>`    return self.core_engine` |

Also recorded for abort routing: `self.reqs_in_flight[request.request_id] = chosen_engine`
(`core_client.py:1156`), cleared on finish at `core_client.py:1167-1171`.

---

## Part 2 B — `vllm/v1/engine/coordinator.py` (file is byte-identical to v0.10.2)

The whole publish loop is `DPCoordinatorProc.process_input_socket`,
`coordinator.py:144-339`.

| # | Question | Answer | file:line | Verbatim |
|---|---|---|---|---|
| B1 | When are **changed counts** published? | Counts are published **only on a poller timeout** — never directly in response to an engine's stats message. The timeout length itself is what encodes "changed": `wait_for` is the short interval when `stats_changed`, else the long heartbeat. So the publish condition is *"the poller expired, and the expiry deadline was computed from `stats_changed`"*. | condition `coordinator.py:204-219`; deadline `coordinator.py:194-205` | `elapsed = int(time.time() * 1000) - last_publish_time`<br>`# Send at stats_update_interval_ms interval if the stats have`<br>`# changed, or otherwise every 5 seconds.`<br>`wait_for = (self.stats_update_interval_ms`<br>`            if stats_changed else 5000)`<br>`…`<br>`events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))`<br>`if not events:`  ← the publish branch |
| B1 | What exactly goes out | A 3-tuple `(counts, wave, engines_running)`. On the timeout path `counts` is a real list. On a **wave/state change** a second, separate message goes out immediately with `counts = None`, which the client at `core_client.py:1076` ignores for LB purposes while still updating `current_wave`/`engines_running`. | timeout publish `coordinator.py:215-218`; state-change publish `coordinator.py:337-339` | `to_publish = (engine_req_counts_list, current_wave, engines_running)`<br>`publish_front.send(msgspec.msgpack.encode(to_publish))`<br>and `message = (None, current_wave, engines_running)` |
| B2 | Role of the **previous-step snapshot** (`last_step_counts`) | Prevents publishing a *torn* count vector. Engines report independently; when a stats message arrives that belongs to a **newer step** while `stats_changed` is still set from the previous step, the coordinator **deep-copies the counts as they stood for the older, now-complete step** and publishes that snapshot on the next timeout instead of the live half-updated vector. It is consumed once and cleared. | capture `coordinator.py:296-298`; consume+clear `coordinator.py:208-210`; deep copy `coordinator.py:353-357` | `if stats_changed:`<br>`    last_step_counts = self._get_engine_counts(do_copy=True)`<br>consume: `if last_step_counts is not None:`<br>`    engine_req_counts_list = last_step_counts`<br>`    last_step_counts = None` |
| B2 | Role of the **minimum collection wait** | Forces the poller to block at least 50 ms **when no snapshot is pending**, giving all DP engines time to report for the current step before the coordinator publishes. Once a snapshot is pending the wait drops to 0 because the data is already known-consistent, so it can go out immediately. | `coordinator.py:200-205` | `# Wait at least 50ms to ensure we've received all stats for`<br>`# the current step.`<br>`min_timeout = 50 if last_step_counts is None else 0` |
| B2 | Role of the **unchanged-state heartbeat** | When nothing changed, still republish every 5 s so late-joining / resubscribing frontends (XPUB/XSUB, no history replay) converge, and so a frontend's stale local reservations (A2) are eventually flushed. On the heartbeat path `stats_changed` is reset to `False`. | `coordinator.py:197-198`, `:211-213` | `wait_for = (self.stats_update_interval_ms if stats_changed else 5000)`<br>and `else:`<br>`    engine_req_counts_list = self._get_engine_counts()`<br>`    stats_changed = False` |
| B2 | Role of **wave/step ordering** | `(wave, step)` is the logical clock used to decide whether an incoming engine report belongs to a *newer* step, which is what triggers the snapshot. It is compared **lexicographically, wave-major**. Out-of-order reports are logged and **do not advance the clock**, but — note — the counts from them are still written (`:308-310` runs unconditionally after the ordering block). | compare `coordinator.py:293-295`; advance `:299-300`; warn `:301-307`; unconditional write `:308-310` | `if (stats_wave > last_stats_wave`<br>`        or stats_wave == last_stats_wave`<br>`        and stats_step > last_stats_step):` |
| B2 | Where wave state itself moves | `wave_complete` from rank 0 → `current_wave = wave + 1`, `engines_running = False` (`:312-322`). `start_wave` from a stale-wave engine → adopt wave, resume, rebroadcast `START_DP_WAVE` (`:323-335`). A frontend `FIRST_REQ` while paused → resume + broadcast (`:264-274`). | `coordinator.py:264-335`; broadcast helper `coordinator.py:341-351` | `engines_running = True`<br>`wave_state_changed = True`<br>`self._send_start_wave(publish_back, current_wave, engine_to_exclude)` |

### B3 — Numeric constants, checked against the Frontier spec expectation

| Expected by spec | Confirmed? | Actual | Symbol name | file:line | Verbatim |
|---|---|---|---|---|---|
| waiting-count weight **4** | **CONFIRMED** | `4` | *no symbol* — inline literal in the scoring expression. **And it does not live in `coordinator.py` at all**; it is in the frontend client. | `vllm/v1/engine/core_client.py:1146` | `score = waiting * 4 + running` |
| **50 ms** min collection wait | **CONFIRMED** | `50` | *no symbol* — bare literal assigned to the local `min_timeout` | `vllm/v1/engine/coordinator.py:202` | `min_timeout = 50 if last_step_counts is None else 0` |
| **100 ms** changed-stats interval | **CONFIRMED** | `100` | `min_stats_update_interval_ms` (parameter, **default** 100) → stored as `self.stats_update_interval_ms` | default `coordinator.py:116` and `coordinator.py:130`; stored `coordinator.py:122`; used `coordinator.py:197` | `def __init__(self, engine_count: int, min_stats_update_interval_ms: int = 100):`<br>`self.stats_update_interval_ms = min_stats_update_interval_ms` |
| **5000 ms** unchanged heartbeat | **CONFIRMED** | `5000` | *no symbol* — bare literal in the ternary | `vllm/v1/engine/coordinator.py:198` | `wait_for = (self.stats_update_interval_ms`<br>`            if stats_changed else 5000)` |

Four caveats a simulator must not gloss over:

1. **The weight 4 is not a coordinator constant.** It is applied per-frontend at
   request-admission time (`core_client.py:1146`). The coordinator transports raw
   `[waiting, running]` pairs and never weights them.
2. **Only `100` is a named/overridable symbol.** `50` and `5000` are hard-coded
   literals with no name and no env/config override; `DPCoordinator.__init__`
   (`coordinator.py:58-93`) does not even forward `min_stats_update_interval_ms`
   into the `Process` kwargs (`coordinator.py:81-86`), so in practice the default
   `100` at `coordinator.py:130` is always what is used.
3. **100 ms is a floor, not a period.** `poller.poll` returns early on any socket
   event, and the publish only happens on the *timeout* branch; a busy coordinator
   re-enters the loop with a recomputed `elapsed`, so the effective publish cadence
   is ">= 100 ms", not "every 100 ms".
4. The client-side comment at `core_client.py:1151` calls it
   "every 100ms", matching constant 3.

### B4 — Initialization epoch and equal-timestamp ordering

| Question | Answer | file:line | Verbatim |
|---|---|---|---|
| What is the **initialization epoch**? | `last_publish_time = 0`, i.e. **Unix epoch 0**, set once before the loop. Since `elapsed = int(time.time()*1000) - 0` is ~1.7e12 ms, `wait_for - elapsed` is hugely negative on the very first iteration, so `max(min_timeout, …)` collapses to `min_timeout`. Net effect: **the first poll always waits exactly the 50 ms minimum and then publishes an all-zero count vector**, rather than waiting 5 s. | set `coordinator.py:192`; used `coordinator.py:194`, `:204-205` | `last_publish_time = 0`<br>`elapsed = int(time.time() * 1000) - last_publish_time` |
| Initial **logical clock** | `current_wave = 0`, `engines_running = False`, `stats_changed = False`, `last_stats_step = -1`, `last_stats_wave = -1`, `last_step_counts = None`. The `-1` sentinels guarantee the very first engine report (wave 0, step >= 0) satisfies the ordering test and is accepted. | `coordinator.py:151-158` | `last_stats_step = -1`<br>`last_stats_wave = -1`<br>`last_step_counts: Optional[list[list[int]]] = None` |
| Initial **per-engine counts** | Every engine starts `[waiting=0, running=0]`; the frontend mirrors this with `[[0, 0] for _ in self.core_engines]`, so before any real data every engine scores 0 and the rotated scan order (A1) is the sole tie-break. | `coordinator.py:108-109`, `:120`; client side `core_client.py:979` | `self.request_counts = [0, 0]  # [waiting, running]` |
| How are **equal-timestamp events ordered**? | There is no wall-clock timestamp in the ordering at all — ordering is on the **`(wave, step)` logical clock**, compared **wave-major then step**, both **strictly greater**. Consequences: a report with *exactly equal* `(wave, step)` is **not** newer, so it does **not** trigger a snapshot and does **not** log a warning (the `elif` at `:301-302` is false when both are equal) — it silently overwrites that engine's counts and sets `stats_changed`. This is the normal case for the 2nd..Nth engine reporting the same step. `step_counter` resets to 0 on each wave boundary (`core.py:1128-1129`), which is exactly why `wave` must dominate the comparison. | `coordinator.py:293-295`; equal-case fallthrough `:301-302`; unconditional write `:308-310`; reset `vllm/v1/engine/core.py:1127-1129` | `if (stats_wave > last_stats_wave`<br>`        or stats_wave == last_stats_wave`<br>`        and stats_step > last_stats_step):`<br>reset: `self.current_wave += 1`<br>`self.step_counter = 0` |
| Ordering of the two publish paths within one loop iteration | A wave-state change publishes at the **end of the same iteration** (`:337-339`) and always carries `counts = None`; count publishes only ever happen at the **top** of an iteration on the timeout branch. They cannot be emitted in the same iteration. | `coordinator.py:206-219` vs `:337-339` | `if wave_state_changed:`<br>`    message = (None, current_wave, engines_running)` |

---

## Part 2 C — `vllm/v1/engine/core.py` and the scheduler stats producers

There are **two independent stats producers** with different triggers,
different payloads and different suppression rules. Conflating them is the main
modelling hazard.

| # | Question | Answer | file:line | Verbatim |
|---|---|---|---|---|
| C1 | **Producer 1** — the per-step metrics report | `Scheduler.make_stats()` is called at the **very end of `update_from_output()`**, i.e. **after** all per-request state updates for the step have been applied. It is attached to the `EngineCoreOutputs` of **exactly one** frontend, and an empty `EngineCoreOutputs` is synthesized if no client had outputs this step. | call `vllm/v1/core/sched/scheduler.py:1413-1419` | `if (stats := self.make_stats(spec_decoding_stats)) is not None:`<br>`    # Return stats to only one of the front-ends.`<br>`    if (eco := next(iter(engine_core_outputs.values()), None)) is None:`<br>`        # We must return the stats even if there are no request`<br>`        # outputs this step.`<br>`        engine_core_outputs[0] = eco = EngineCoreOutputs()`<br>`    eco.scheduler_stats = stats` |
| C1 | Producer-1 trigger chain | `run_busy_loop` → `_process_engine_step` → `step_fn()` = `EngineCore.step()` → `schedule()` → `execute_model` → `update_from_output()` (which emits). So **one report per executed engine step**, and `step()` returns early with `{}, False` — producing **no** report — when `self.scheduler.has_requests()` is false. | loop `core.py:1091-1099`; `_process_engine_step` `core.py:803-813`; `step` `core.py:290-309`; early return `core.py:299-300` | `if not self.scheduler.has_requests():`<br>`    return {}, False`<br>`scheduler_output = self.scheduler.schedule()`<br>`model_output = self.execute_model_with_error_logging(...)`<br>`engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)` |
| C1 | Producer-1, extra trigger | A stats object is **also** produced on the model-execution **exception** path, purely to enrich the crash dump — it is passed to `dump_engine_exception` and is not published to any frontend. | `core.py:278-288` | `dump_engine_exception(self.vllm_config, scheduler_output,`<br>`                      self.scheduler.make_stats())` |
| C1 | **Producer 2** — the DP load-balance report | `DPEngineCoreProc._maybe_publish_request_counts()` is called from `run_busy_loop` **immediately after `_process_engine_step()` returns**, i.e. after `update_from_output()` has already removed finished requests. It builds a *minimal* `SchedulerStats` carrying only `(num_running_reqs, num_waiting_reqs, step_counter, current_wave)` and pushes it with `client_index = -1` (broadcast to the coordinator, not to a frontend). | call site `core.py:1098-1099`; body `core.py:1075-1087` | `executed = self._process_engine_step()`<br>`self._maybe_publish_request_counts()`<br>…<br>`stats = SchedulerStats(*counts,`<br>`                       step_counter=self.step_counter,`<br>`                       current_wave=self.current_wave)`<br>`self.output_queue.put_nowait(`<br>`    (-1, EngineCoreOutputs(scheduler_stats=stats)))` |
| C1 | Note on positional construction | `SchedulerStats(*counts, …)` relies on `get_request_counts()` returning `(running, waiting)` in exactly the field order of the dataclass — `num_running_reqs` then `num_waiting_reqs`. | dataclass `vllm/v1/metrics/stats.py:49-57`; getter `vllm/v1/core/sched/scheduler.py:1502-1504` | `num_running_reqs: int = 0`<br>`num_waiting_reqs: int = 0`<br>`# These are used for internal DP load-balancing.`<br>`step_counter: int = 0`<br>`current_wave: int = 0`<br>getter: `return len(self.running), len(self.waiting)` |
| C1 | Which one the coordinator consumes | The coordinator reads `outputs.scheduler_stats.{num_waiting_reqs, num_running_reqs, step_counter, current_wave}` and asserts the message carries **no** request outputs — i.e. it is fed by **Producer 2 only**. | `coordinator.py:282-309` | `assert not outputs.outputs`<br>`assert outputs.utility_output is None`<br>…<br>`stats[0] = scheduler_stats.num_waiting_reqs`<br>`stats[1] = scheduler_stats.num_running_reqs` |
| C2 | Producer-1 suppression | Suppressed entirely when `log_stats` is false — `make_stats` returns `None` and the `if` at `:1413` never fires. | `vllm/v1/core/sched/scheduler.py:1609-1610` | `if not self.log_stats:`<br>`    return None` |
| C2 | Producer-2 suppression (mode) | Suppressed entirely unless a coordinator exists **and** LB is not external — i.e. only in "internal" and "hybrid" DP LB modes. Also structurally absent for DP=1 (the method lives on `DPEngineCoreProc`). | gate `core.py:1075-1077`; definition `core.py:503-510` | `if not self.publish_dp_lb_stats:`<br>`    return`<br>set by: `# Only publish request queue stats to coordinator for "internal"`<br>`# and "hybrid" LB modes .`<br>`self.publish_dp_lb_stats = (self.has_coordinator and not vllm_config.parallel_config.data_parallel_external_lb)` |
| C2 | Producer-2 suppression (dedup) | **Suppressed whenever the `(running, waiting)` pair is unchanged since the last publish** — a pure equality check against `self.last_counts`, seeded `(0, 0)`. This is why the coordinator needs its 5 s heartbeat: a steady-state engine emits nothing at all. | `core.py:1079-1082`; seed `core.py:1014` | `# Publish our request counts (if they've changed).`<br>`counts = self.scheduler.get_request_counts()`<br>`if counts != self.last_counts:`<br>`    self.last_counts = counts`<br>seed: `self.last_counts = (0, 0)` |
| C2 | Consequence for `step_counter` monotonicity | Because of the dedup, **consecutive published reports can skip step numbers arbitrarily**. `step_counter` is incremented once per call to `_has_global_unfinished_reqs` (`core.py:1133-1135`), not once per publish. A simulator must treat `step_counter` as a sparse, strictly-increasing-within-a-wave tag, not a dense counter. | `core.py:1133-1135`; reset `core.py:1127-1129` | `# Optimization - only perform finish-sync all-reduce every 32 steps.`<br>`self.step_counter += 1`<br>`if self.step_counter % 32 != 0:`<br>`    return True` |
| C2 | Batch-queue (pipelined) variant | With `step_with_batch_queue`, a scheduled-but-not-yet-completed batch returns `None, True` early, so **no** `EngineCoreOutputs` and therefore no Producer-1 report is emitted for that iteration. | `core.py:318-353`, early return at `core.py:349-353` | `# Don't block on next worker response unless the queue is full`<br>`# or there are no more requests to schedule.`<br>`return None, True` |

---

## Part 2 D — `vllm/v1/core/sched/scheduler.py`: what counts as waiting vs running

Both numbers are plain `len()` of two live containers — `self.running: list[Request]`
and `self.waiting: RequestQueue` — with **no filtering by status**. Everything
below is therefore a question of *which container an object is in at the moment
of the `len()`*.

| # | Case | Counted as | Why / file:line | Verbatim |
|---|---|---|---|---|
| D1 | Definition | `num_running_reqs = len(self.running)`, `num_waiting_reqs = len(self.waiting)`. Identical in both producers. | `scheduler.py:1614-1615`; `scheduler.py:1502-1504` | `num_running_reqs=len(self.running),`<br>`num_waiting_reqs=len(self.waiting),` |
| D1 | Newly arrived request | **waiting** | `add_request` pushes straight onto the waiting queue; nothing else. | `scheduler.py:1506-1508` | `def add_request(self, request: Request) -> None:`<br>`    self.waiting.add_request(request)`<br>`    self.requests[request.request_id] = request` |
| D1 | **Admitted-but-not-yet-scheduled** (admitted this step, model not yet run) | **running** | Admission appends to `self.running` *inside* `schedule()`, before `execute_model` is ever called. Status is set to `RUNNING` a few lines later at `:838`. So from the instant `schedule()` returns, a first-chunk-prefill request already counts as running. | append `scheduler.py:812-813`; status `scheduler.py:838` | `req_index += 1`<br>`self.running.append(request)`<br>…<br>`request.status = RequestStatus.RUNNING` |
| D1 | Running request that `schedule()` chose **not** to schedule this step (budget exhausted) | **running** | It is never removed from `self.running`; the code explicitly asserts the scheduled set may be a strict subset. | `scheduler.py:890-894` | `# Since some requests in the RUNNING queue may not be scheduled in`<br>`# this step, the total number of scheduled requests can be smaller than`<br>`# len(self.running).`<br>`assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running))` |
| D1 | **Preempted** request | **waiting** (moves running→waiting **within** `schedule()`) | Victim is popped/removed from `self.running` (`:489` priority, `:507` FCFS), marked `PREEMPTED` with `num_computed_tokens = 0`, then **prepended to the head of the waiting queue**. So a preemption is `running -= 1` and `waiting += 1` atomically, visible in the same step's report. | remove `scheduler.py:489`, `:507`; status `scheduler.py:513-514`; requeue `scheduler.py:537-538` | `preempted_req.status = RequestStatus.PREEMPTED`<br>`preempted_req.num_computed_tokens = 0`<br>…<br>`self.waiting.prepend_request(preempted_req)` |
| D1 | Preempted request **resumed** later | **running** | On re-admission it takes the same `self.running.append` path and is classified `scheduled_resumed_reqs`. | `scheduler.py:813`, `:819-820` | `elif request.status == RequestStatus.PREEMPTED:`<br>`    scheduled_resumed_reqs.append(request)` |
| D1 | Request awaiting **remote KV** (`WAITING_FOR_REMOTE_KVS`) | **waiting** | It is popped from `self.waiting`, parked in the temporary `skipped_waiting_requests` queue, and that whole queue is **prepended back onto `self.waiting`** before `schedule()` returns. Net: still in `self.waiting` when counted. Same for `WAITING_FOR_FSM`, `max_loras` blocking, unknown external-token count, and oversized non-chunked prefill. | park `scheduler.py:644-645`, `:655-656`, `:665-666`, `:689-690`, `:726-727`, `:799-801`; restore `scheduler.py:882-883` | `# Put back any skipped requests at the head of the waiting queue`<br>`if skipped_waiting_requests:`<br>`    self.waiting.prepend_requests(skipped_waiting_requests)` |
| D1 | **Requests finishing this step** | **neither** — removed before the report | In `update_from_output`, a stopped request is freed, bucketed by its pre-stop status, and then removed from the container it was in — `self.running` for `RUNNING`, `self.waiting` for anything else (a preempted request that hit a stop condition). This happens at `:1381-1385`, i.e. **before** `make_stats()` at `:1413`. | bucket `scheduler.py:1332-1337`; remove `scheduler.py:1381-1385` | `if stopped:`<br>`    kv_transfer_params = self._free_request(request)`<br>`    if status_before_stop == RequestStatus.RUNNING:`<br>`        stopped_running_reqs.add(request)`<br>`    else:`<br>`        stopped_preempted_reqs.add(request)`<br>…<br>`if stopped_running_reqs:`<br>`    self.running = remove_all(self.running, stopped_running_reqs)`<br>`if stopped_preempted_reqs:`<br>`    self.waiting.remove_requests(stopped_preempted_reqs)` |
| D1 | Externally **aborted** requests | removed from whichever queue they were in, same step | `finish_requests` path rebuilds `self.running` via `remove_all`. | `scheduler.py:1556` | `self.running = remove_all(self.running, running_requests_to_remove)` |
| D1 | Finished request still pending an outbound `finished_req_ids` notification | **neither** | `_free_request` only adds the id to `self.finished_req_ids` / `finished_req_ids_dict`; those sets are **not** part of either count. But they *do* keep `has_requests()` true (`interface.py:126-129`), so the engine can still take a step whose report shows `0, 0`. | `scheduler.py:1581-1584`; `interface.py:126-129` | `self.finished_req_ids.add(request_id)`<br>…<br>`return self.has_unfinished_requests() or self.has_finished_requests()` |
| D2 | **At what point are the counts observable?** (Producer 1) | At the **end of `update_from_output()`**, after: token append & stop detection, `_free_request`, stopped-request removal from both queues, KV-connector finish handling, and `EngineCoreOutputs` assembly. So the report is a **post-step, post-retirement** snapshot: admissions made by *this* step's `schedule()` are already counted as running, and completions from *this* step are already gone. | `scheduler.py:1413` reached after `:1381-1391` | `if (stats := self.make_stats(spec_decoding_stats)) is not None:` |
| D2 | **At what point are the counts observable?** (Producer 2) | One further step later in the call chain, but on the **same state**: `_maybe_publish_request_counts()` runs after `_process_engine_step()` has fully returned (which includes `update_from_output` **and** `post_step`). Between the two producers' reads, nothing mutates `self.running`/`self.waiting` — `post_step` only forwards draft token ids. So the two reports agree on the counts for a given step. | `core.py:1098-1099`; `post_step` `core.py:311-317` | `executed = self._process_engine_step()`<br>`self._maybe_publish_request_counts()` |
| D2 | What is **not** observable | There is no hook between `schedule()` and `execute_model`. A simulator cannot observe the mid-step state in which a preemption victim has left `self.running` but the model has not yet run. Preemption is only ever visible as an already-settled `(running, waiting)` delta. | `core.py:301-306` | `scheduler_output = self.scheduler.schedule()`<br>`model_output = self.execute_model_with_error_logging(…)`<br>`engine_core_outputs = self.scheduler.update_from_output(…)` |
| D2 | `AsyncScheduler` caveat | `AsyncScheduler` overrides only `_update_after_schedule` and `_update_request_with_output` (`async_scheduler.py:16-47`); it does **not** override `make_stats`, `get_request_counts`, or any queue mutation. All of section D applies unchanged. | `vllm/v1/core/sched/async_scheduler.py:14-47` | — |


---

## Part 2 E — `vllm/v1/worker/gpu_model_runner.py` (`GMR`) and `vllm/forward_context.py` (`FC`, UNCHANGED from v0.10.2)

### E1 — Why DP source batches can be in different local phases yet share one expert forward

| # | Question | Answer | file:line | Verbatim |
|---|---|---|---|---|
| E1a | The one per-step DP agreement collective | A single **all-reduce over a zero-filled `dp_size` vector** — a sum-as-allgather idiom, on the **CPU** process group. Each rank writes only its own slot; the sum reconstructs the full vector. | `FC:78-85` | `num_tokens_across_dp = [0] * dp_size`<br>`num_tokens_across_dp[dp_rank] = num_tokens`<br>`num_tokens_tensor = torch.tensor(num_tokens_across_dp,`<br>`                                 device="cpu",`<br>`                                 dtype=torch.int32)`<br>`from vllm.distributed.parallel_state import get_dp_group`<br>`dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)`<br>`return num_tokens_tensor` |
| E1a | What does **not** exist at this revision | **There is no `should_ubatch_across_dp` and no `vllm/v1/worker/dp_utils.py`** in v0.10.2 — repo-wide grep finds nothing. DP micro-batch coordination is a later-vLLM feature. The only other DP-wide collective is the every-32-step liveness all-reduce. | absence verified by grep; liveness `vllm/config/parallel.py:239-250` (`ReduceOp.MAX`), driven from `vllm/v1/engine/core.py:1131-1139` | — |
| E1b | What is **forced to agree** | The **padded row count**. `get_dp_padding` all-reduces, takes the max, and returns a vector in which *every* entry is that max; the caller then inflates its own batch to it. So all ranks execute the same number of rows and the collective buffers line up. | `GMR:1920-1927`; applied `GMR:2001-2002` | `max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()`<br>`num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] * dp_size,`<br>`                                        device="cpu", dtype=torch.int32)`<br>`return max_tokens_across_dp_cpu - num_tokens, num_tokens_after_padding`<br>caller: `num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)`<br>`num_input_tokens += num_pad` |
| E1b | What stays **ragged** | Under `enforce_eager` (or DP=1) `get_dp_padding` early-exits with `None`, and `DPMetadata.make` then recomputes the vector itself and leaves it **unequalized**; the ragged prefix sums in `cu_tokens_across_dp_cpu` are what the naive all2all dispatch slices by. So equal row counts are a *cudagraph* requirement, not an MoE requirement. | early exit `GMR:1916-1918`; recompute `FC:111-113`; prefix sums `FC:115` | `if dp_size == 1 or self.vllm_config.model_config.enforce_eager:`<br>`    # Early exit.`<br>`    return 0, None`<br>`…`<br>`cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)` |
| E1b | The cudagraph **mode** is NOT forced to agree | `uniform_decode` is computed from this rank's own batch, *after* DP padding, and fed to a rank-local dispatcher. A prefilling rank gets `False`, a decoding rank `True`, so `dispatch` may legitimately return FULL on one rank and PIECEWISE/NONE on another. **Only the row count must agree, not the replay mechanism.** | `GMR:2291-2298`; dispatcher `vllm/v1/cudagraph_dispatcher.py:92-121` | `uniform_decode = (max_query_len == self.uniform_decode_query_len) and (`<br>`    num_scheduled_tokens == self.input_batch.num_reqs * max_query_len)`<br>`batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,`<br>`                                   uniform_decode=uniform_decode)`<br>`cudagraph_runtime_mode, batch_descriptor = \`<br>`    self.cudagraph_dispatcher.dispatch(batch_descriptor)` |
| E1b | The "all ranks must call the collective" invariant | Structural, not a flag: the naive all2all manager issues `dp_group.broadcast` / `all_reduce` **unconditionally** inside dispatch/combine, reached whenever `dp_size > 1`. A rank that skipped the forward would hang every other rank — which is precisely why the idle path (E1d) must exist. | collectives `vllm/distributed/device_communicators/all2all.py:39-44, :64`; reached from `vllm/model_executor/layers/fused_moe/layer.py:1804-1808`; condition `layer.py:1790-1793` | `do_naive_dispatch_combine` is true iff `dp_size > 1` |
| E1c | Where local phase is decided, and why it need not agree | **Each DP rank runs its own `Scheduler` in its own `EngineCore` process** — there is no cross-rank coordination in `schedule()` at all. The prefill-chunk-vs-decode split is a purely rank-local token-budget decision. Downstream, **attention metadata is built per rank from rank-local arrays and never crosses the DP boundary**, so phase disagreement is invisible to attention. Only the MoE/expert path is collective, and it consumes **token counts, not phases** — which is the whole answer to E1. | scheduler per rank `vllm/v1/engine/core.py:133`, class chain `core.py:72, 470, 997`; decode side `vllm/v1/core/sched/scheduler.py:408-424`; prefill chunking `:712-731`; rank-local attn metadata `GMR:1178-1290` | `self.scheduler: SchedulerInterface = Scheduler(...)` |
| E1d | "Even with no work, still participate" | `GPUWorker.execute_dummy_batch` runs a 1-token forward. It is invoked from the DP busy loop whenever the engine is in a running wave but scheduled nothing. | `vllm/v1/worker/gpu_worker.py:556-557`; driver `vllm/v1/engine/core.py:1102-1109`; chain `core.py:403-404` → `vllm/v1/executor/abstract.py:97-98` → `multiproc_executor.py:197-198` | `def execute_dummy_batch(self) -> None:`<br>`    self.model_runner._dummy_run(1)`<br>driver: `if not executed:`<br>`    if not local_unfinished_reqs and not self.engines_running:`<br>`        # All engines are idle.`<br>`        continue`<br>`    # We are in a running state and so must execute a dummy pass`<br>`    # if the model didn't execute any ready requests.`<br>`    self.execute_dummy_batch()` |
| E1d | Same invariant restated for EPLB | `GMR:3240-3248` comments that the dummy run is needed "to avoid blocking DP… some DP ranks do not have any requests… we still have to trigger EPLB… in synchronization". | `GMR:3240-3248` | — |

### E2 — Token populations and dummy/idle participants

| # | Question | Answer | file:line | Verbatim |
|---|---|---|---|---|
| E2a | The DP dataclass carried into the forward pass — **all** its fields | `DPMetadata` has exactly **three** fields. `local_sizes` is transient: set only inside the `chunked_sizes` context manager and cleared in its `finally`. It hangs off `ForwardContext.dp_metadata`, populated only when `data_parallel_size > 1`. | dataclass `FC:65-69`; built `FC:114-116`; `chunked_sizes` `FC:118-155` (clear at `:155`); reader `FC:157-158`; field on context `FC:175`; population `FC:215-220` | `@dataclass`<br>`class DPMetadata:`<br>`    max_tokens_across_dp_cpu: torch.Tensor`   *(FC:67)*<br>`    cu_tokens_across_dp_cpu: torch.Tensor`    *(FC:68)*<br>`    local_sizes: Optional[list[int]] = None`  *(FC:69)* |
| E2a | Guard that keeps DP metadata alive for an attention-free dummy run | The `num_tokens is not None` disjunct. | `FC:216-217` | `if vllm_config.parallel_config.data_parallel_size > 1 and (`<br>`        attn_metadata is not None or num_tokens is not None):` |
| E2b | `_dummy_run` signature | 8 parameters; the idle DP call uses defaults for all but `num_tokens=1`, so `cudagraph_runtime_mode=NONE` and `skip_eplb=False`. | `GMR:3035-3045`; idle invocation `vllm/v1/worker/gpu_worker.py:557` | `def _dummy_run(self, num_tokens: int, cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE, force_attention: bool = False, uniform_decode: bool = False, skip_eplb: bool = False, is_profile: bool = False, create_mixed_batch: bool = False, remove_lora: bool = True) -> tuple[torch.Tensor, torch.Tensor]:` |
| E2b | What `_dummy_run` sets | DP padding **first** — the same `get_dp_padding` call as the real path, so the idle rank is inflated to `max_tokens_across_dp_cpu`; then a synthetic token/req layout; attention metadata **only** if `force_attention or cudagraph_runtime_mode == FULL`, so for the plain idle case `attn_metadata` stays **`None`**; then slices of the same persistent buffers; then the real model call; then `eplb_step(is_dummy=True, …)`. | padding `GMR:3070-3072`; layout `GMR:3096-3123`; attn gate `GMR:3129-3168`, default `GMR:3125`; buffers `GMR:3175-3188`; model call `GMR:3215-3228`; eplb `GMR:3247-3248` | `# Padding for DP`<br>`num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)`<br>`num_tokens += num_pad` |
| E2b | **What distinguishes a dummy batch from a real one at the forward-context level** | **Nothing.** `ForwardContext` has **no `is_dummy` / `is_profile` field** (`FC:161-179`: `no_compile_layers`, `attn_metadata`, `virtual_engine`, `dp_metadata`, `cudagraph_runtime_mode`, `batch_descriptor`), and `set_forward_context` is called with the **identical argument set** in both paths. Differences are only indirect: `attn_metadata=None` for a plain dummy run, and optionally randomized `input_ids`. **A simulator cannot distinguish real from dummy participation from inside the MoE layer — and neither can vLLM.** | dataclass `FC:161-179`; dummy call `GMR:3215-3221` vs real call `GMR:2387-2393`; randomization `GMR:2978-3005` | dummy: `with self.maybe_randomize_inputs(input_ids), set_forward_context(`<br>`        attn_metadata, self.vllm_config, num_tokens=num_tokens,`<br>`        num_tokens_across_dp=num_tokens_across_dp,`<br>`        cudagraph_runtime_mode=cudagraph_runtime_mode,`<br>`        batch_descriptor=batch_descriptor):` |
| E2b | Input randomization | Active **only** when `VLLM_RANDOMIZE_DP_DUMMY_INPUTS` is set and `dp_size > 1`; its docstring names the purpose as balancing expert selection during DP dummy runs. | `GMR:2978-3005` (docstring `:2980-2984`) | — |
| E2c | Is cudagraph padding real rows or metadata? | **Real tensor rows.** Two padding stages (local cudagraph bucket, then DP max) both raise `num_input_tokens`, and that padded count then slices the real device buffers. Only `[:total_num_input_tokens]` was ever written, so pad rows carry **stale buffer contents**, are computed on through the GEMMs **and the MoE collective**, and are discarded. They are neutralized only at the metadata level. | bucket `GMR:1982-1998`; DP `GMR:2001-2002`; slices `GMR:2025, 2035, 2041`; writes `GMR:1190, 1199`; neutralization: seq-len fill `GMR:~1184`, slot-mapping fill `GMR:1266-1268` | `num_input_tokens = self.vllm_config.pad_for_cudagraph(total_num_input_tokens)`<br>`…`<br>`input_ids = self.input_ids.gpu[:num_input_tokens]`<br>`positions = self.positions.gpu[:num_input_tokens]`<br>`# Fill unused with -1. Needed for reshape_and_cache in full cuda graph mode`<br>`blk_table.slot_mapping[total_num_input_tokens:].fill_(-1)` |
| E2c | Simulator implication | Padding rows are **real MoE work and real collective payload**. A cost model that charges only scheduled tokens will under-count whenever DP ranks are imbalanced or a cudagraph bucket rounds up. | — | — |
| E2d | Who consumes `num_tokens_across_dp` | Passed as a kwarg into `set_forward_context` (both paths), converted to `DPMetadata` with an assertion that this rank's slot matches its batch size. Then: (1) `cu_tokens_across_dp_cpu` → naive all2all buffer sizing and per-rank slice bounds; (2) `max_tokens_across_dp_cpu` → the chunked MoE loop bound; (3) `local_sizes` → `get_local_sizes()` for the flashinfer-cutlass prepare/finalize; (4) CPU runner override. | pass `GMR:2392` (real), `GMR:3219` (dummy); convert+assert `FC:109-110, :218-220`; (1) `vllm/distributed/device_communicators/all2all.py:28-44, :46-55, :57-66`; (2) `vllm/model_executor/layers/fused_moe/layer.py:1743`, loop `:1753-1766`, chunk size `fused_moe/config.py:320` = `envs.VLLM_MOE_DP_CHUNK_SIZE`; (3) `layer.py:1762-1763` → `flashinfer_cutlass_prepare_finalize.py:16-17`; (4) `vllm/v1/worker/cpu_model_runner.py:128` | assert: `assert (num_tokens_across_dp is None`<br>`        or num_tokens_across_dp[dp_rank] == batchsize)`<br>(2): `max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu`<br>buffer: `torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)))` |
| E2d | The lockstep guarantee, verbatim | In the chunked path, a rank that has run out of real tokens is given a **synthetic token** so that it still enters every chunk iteration and every collective. | `FC:50-62` | `if local_size[i] <= 0:`<br>`    local_size[i] = 1  # ensure lockstep even if done` |

---

## Part 2 F — Qwen3-MoE expert path, reductions, and LOCAL vs DISTRIBUTED

Path selection is fixed by construction: `Qwen3MoeSparseMoeBlock.__init__` builds
`FusedMoE(... reduce_results=True, renormalize=config.norm_topk_prob ...)`
(`vllm/model_executor/models/qwen3_moe.py:146-155`). It passes neither
`apply_router_weight_on_input` nor `activation`, so the `FusedMoE.__init__`
defaults apply: `apply_router_weight_on_input: bool = False`, `activation: str = "silu"`
(`vllm/model_executor/layers/fused_moe/layer.py:787-788`), with
`use_grouped_topk=False` and `custom_routing_function=None`.

### F1 — Exact call order, gate → returned value

Abbreviations: `qwen3_moe.py` = `vllm/model_executor/models/qwen3_moe.py`;
`layer.py` / `fused_moe.py` = `vllm/model_executor/layers/fused_moe/{layer,fused_moe}.py`.

| # | Step | file:line | Verbatim |
|---|---|---|---|
| 1 | Gate linear — `ReplicatedLinear`, **no collective** | `qwen3_moe.py:185`; class `vllm/model_executor/layers/linear.py:352-360` | `router_logits, _ = self.gate(hidden_states)` |
| 2 | FusedMoE call | `qwen3_moe.py:186` | `final_hidden_states = self.experts(hidden_states=hidden_states,` |
| 3 | `CustomOp.forward` → `_forward_method` from `dispatch_forward()` | `vllm/model_executor/custom_op.py:47-48`, dispatch `:85-110` | `return self._forward_method(*args, **kwargs)` |
| 4 | `FusedMoE.forward_cuda` → `forward_native` | `layer.py:1654` | `return self.forward_native(hidden_states, router_logits)` |
| 5 | `forward_native` → registered custom op (`shared_experts is None` for Qwen3) | `layer.py:1634-1635` | `fused_output = torch.ops.vllm.moe_forward(`<br>`    hidden_states, router_logits, self.layer_name)` |
| 6 | `moe_forward` op body | `layer.py:1934` | `return self.forward_impl(hidden_states, router_logits)` |
| 7 | `forward_impl` → quant method | `layer.py:1821` | `final_hidden_states = self.quant_method.apply(` |
| 8 | `UnquantizedFusedMoEMethod.apply` → `self.forward(...)` → CustomOp dispatch → `forward_cuda` | `layer.py:394`, `:423`, `:446` | `return self.forward(` |
| 9 | Routing | `layer.py:470` | `topk_weights, topk_ids = FusedMoE.select_experts(` |
| 9a | → `fused_topk` | `layer.py:1506-1507` (guard `elif custom_routing_function is None:` at `:1506`) | `topk_weights, topk_ids, token_expert_indices = fused_topk(` |
| 9b | → `vllm_topk_softmax` via `dispatch_topk_func()` | `fused_moe.py:1001-1003`; fn `fused_moe.py:884-898` | `topk_weights, topk_ids = topk_func(topk_weights, topk_ids,`<br>`                                   token_expert_indices,`<br>`                                   gating_output_float, renormalize)` |
| 10 | `fused_experts` | `layer.py:518` | `return fused_experts(` |
| 11 | → `dispatch_fused_experts_func(inplace)` (`inplace=True` at `layer.py:525`) → `torch_vllm_inplace_fused_experts` → `fused_experts_impl` | `fused_moe.py:1466-1469`, `:1542` | `return dispatch_fused_experts_func(inplace)(` |
| 12 | Input quant — no-op for bf16 | `fused_moe.py:1710-1715` | `qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(` |
| 13 | **Expert sort + `expert_map` applied here** | `fused_moe.py:1718-1720` | `sorted_token_ids, expert_ids, num_tokens_post_padded = (`<br>`    moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],`<br>`                         global_num_experts, expert_map))` |
| 14 | **w1 grouped GEMM = gate **and** up fused, `N = 2 * intermediate_size`** | `fused_moe.py:1723-1725`; kernel launch `fused_moe.py:621` | `invoke_fused_moe_kernel(qcurr_hidden_states,`<br>`                        w1,`<br>`                        intermediate_cache1,` |
| 15 | **SiLU*mul — a SEPARATE CUDA op, NOT inside the triton kernel** | `fused_moe.py:1746-1748` | `if activation == "silu" and is_act_and_mul:`<br>`    torch.ops._C.silu_and_mul(intermediate_cache2,`<br>`                              intermediate_cache1.view(-1, N))` |
| 16 | **w2 grouped GEMM (down)** | `fused_moe.py:1775-1777` | `invoke_fused_moe_kernel(qintermediate_cache2,`<br>`                        w2,`<br>`                        intermediate_cache3,` |
| 17 | **Routing weights applied — inside the w2 kernel** | `fused_moe.py:469-473` | `if MUL_ROUTED_WEIGHT:`<br>`    moe_weight = tl.load(topk_weights_ptr + offs_token,`<br>`                         mask=token_mask,`<br>`                         other=0)`<br>`    accumulator = accumulator * moe_weight[:, None]` |
| 18 | **Local top-k sum reduction** | `fused_moe.py:1797-1798` | `ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.size()),`<br>`            out_hidden_states[begin_chunk_idx:end_chunk_idx])` |
| 19 | Distributed reduce — conditional | `layer.py:1859-1860` | `if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):`<br>`    states = self.maybe_all_reduce_tensor_model_parallel(states)` |
| 20 | Value returned to the model | `layer.py:1866`, then `qwen3_moe.py:190-191` | `return reduce_output(final_hidden_states)` |

**Gated/SwiGLU precision.** `w1` is the fused `w13_weight` of shape `[E, N, K]` with
`N = 2 * intermediate_size` (`E, N, _ = w1.size()` at `fused_moe.py:1689`;
`intermediate_cache2` is `(M * top_k_num, N // 2)` at `fused_moe.py:1660`). The triton
kernel writes the **raw `[.., 2*inter]` accumulator and contains no SiLU**. The SwiGLU
nonlinearity is a separate `torch.ops._C.silu_and_mul` launch at `fused_moe.py:1747-1748`,
selected by `if activation == "silu" and is_act_and_mul:` (`fused_moe.py:1746`, with
`is_act_and_mul: bool = True` default at `fused_moe.py:1481`). It is a **bare op call, not
the `SiluAndMul` `CustomOp` module** (`vllm/model_executor/layers/activation.py:59-89`),
so `CustomOp` enable/disable and `forward_native` have **no effect** on this path.

**Kernel sequence per MoE layer:** `topk_softmax` → `moe_align_block_size` →
grouped-GEMM(w1) → `silu_and_mul` → grouped-GEMM(w2, routing weights folded in) → `moe_sum`.

### F2 — Routing weights, workspace, expert map, local reduction

| # | Question | Answer | file:line | Verbatim |
|---|---|---|---|---|
| F2a | Where are routing weights applied? | **Inside `invoke_fused_moe_kernel` via the `MUL_ROUTED_WEIGHT` constexpr, on the w2 GEMM only** — not after, and not on w1. Parameter is positional #11, `mul_routed_weight: bool`. The two call sites pass exactly the expected `False` / `True`. | param `fused_moe.py:503`; **w1 → False** `fused_moe.py:1733`; **w2 → True** `fused_moe.py:1785`; forwarded `fused_moe.py:657` (dense) and `:607` (gptq/awq); applied `fused_moe.py:469-473` (dense) and `:257-261` (gptq/awq); guard `fused_moe.py:514` | w1 site: `apply_router_weight_on_input,` (→ `False`, default `layer.py:787`)<br>w2 site: `not apply_router_weight_on_input,` (→ `True`)<br>forward: `MUL_ROUTED_WEIGHT=mul_routed_weight,`<br>guard: `assert topk_weights is not None or not mul_routed_weight` |
| F2a | Where in the kernel | After the K-loop **and after the bias add** (`fused_moe.py:467-468`), **before** the dtype cast (`fused_moe.py:474-482`). Same pattern in the modular `TritonExperts`: `False,  # mul_routed_weights` for w1 (`fused_moe.py:2095`), `not apply_router_weight_on_input,` for w2 (`:2129`). | `fused_moe.py:467-482` | — |
| F2b | `CHUNK_SIZE` value | **32768.** Note a real mismatch in `envs.py`: the type annotation says `64 * 1024` but the runtime lambda — which is what is actually read — says `"32768"`. The annotation is cosmetic; upstream has the same mismatch. | use `fused_moe.py:1626-1627`; annotation `vllm/envs.py:55`; **runtime** `vllm/envs.py:652-653` | `CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE`<br>`M = min(num_tokens, CHUNK_SIZE)`<br>annotation: `VLLM_FUSED_MOE_CHUNK_SIZE: int = 64 * 1024`<br>runtime: `lambda: int(os.getenv("VLLM_FUSED_MOE_CHUNK_SIZE", "32768")),` |
| F2b | Workspace layout (`E, N, _ = w1.size()`, `K = w2.size(1)`, `top_k_num = topk_ids.size(1)`) | `cache13` is one flat `(M * top_k_num * max(N, K),)` allocation; **`intermediate_cache1` and `intermediate_cache3` are aliasing views into it**, while `intermediate_cache2` is a separate allocation because it is live at the same time as cache1. | alloc `fused_moe.py:1653-1662`; rationale comment `fused_moe.py:1651-1652` | `cache13 = torch.empty(M * top_k_num * max(N, K), …)`<br>`intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)`<br>`intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)`<br>`# This needs separate memory since it's used concurrently with cache1`<br>`intermediate_cache2 = torch.empty((M * top_k_num, N // 2), …)` |
| F2b | Chunking loop | Iterates `(num_tokens // CHUNK_SIZE) + 1` times over `hidden_states[begin:end]`. Only a **short trailing chunk** triggers re-slicing of the three caches and a recomputed `config`. Output goes straight into `out_hidden_states`, which **is** `hidden_states` because `inplace=True`. | loop `fused_moe.py:1685`, slice `:1689`; trailing-chunk branch `:1695`, re-slice `:1700-1703`, config `:1704`; inplace `:1674-1675`, caller `layer.py:525` | `for chunk in range((num_tokens // CHUNK_SIZE) + 1):`<br>`if tokens_in_chunk < CHUNK_SIZE and chunk > 0:` |
| F2c | How is `expert_map` built, and what marks "not on this rank"? | `determine_expert_map(ep_size, ep_rank, global_num_experts)` returns `(global_num_experts, None)` when `ep_size == 1`; otherwise a length-`global_num_experts` int32 tensor prefilled with **`-1`**, with this rank's contiguous slice overwritten by `0..local_num_experts-1`. **Sentinel = `-1`.** | fn `layer.py:680-721`; EP=1 short-circuit `layer.py:703-704`; build `layer.py:714-719` | `# Create a tensor of size num_experts filled with -1`<br>`expert_map = torch.full((global_num_experts, ), -1, dtype=torch.int32)`<br>`start_idx = ep_rank * base_experts + min(ep_rank, remainder)`<br>`expert_map[start_idx:start_idx + local_num_experts] = torch.arange(`<br>`    0, local_num_experts, dtype=torch.int32)` |
| F2c | Where is it applied? | **Not to `topk_ids`** — those stay global. It is applied to the **per-block `expert_ids`** produced by the sort, right after `ops.moe_align_block_size`. | `vllm/model_executor/layers/fused_moe/moe_align_block_size.py:84-85` | `if expert_map is not None:`<br>`    expert_ids = expert_map[expert_ids]` |
| F2c | What the kernel does with `-1` | Short-circuits the block and **writes zeros**, so the downstream unweighted sum stays correct without any masking. | `fused_moe.py:379-387`; same guard `:163-171` (gptq/awq) | `off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)`<br>`if off_experts == -1:`<br>`    # Write back zeros to the output when the expert is not`<br>`    # in the current expert parallel rank.`<br>`    write_zeros_to_output(...)`<br>`    return` |
| F2c | **Simulator consequence** | Under EP, each rank still allocates the **full `[M, top_k, ...]` workspace** and still runs a **full `moe_sum` over all `top_k` slots**; off-rank slots are zero-filled locally, not dropped. Only the GEMM blocks are skipped. Cost must be modelled as "full workspace + full reduction, reduced GEMM occupancy", not "1/ep_size of everything". | `fused_moe.py:379-387` + `:1653-1657` + `:1797-1798` | — |
| F2d | Where is the local top-k output reduction? | `ops.moe_sum`, reducing `[num_tokens, topk, hidden] → [num_tokens, hidden]`. Implementation is a plain unrolled sum over the topk axis, dispatched by `topk`. Modular `TritonExperts` equivalent at `fused_moe.py:2142`. | call `fused_moe.py:1797-1798`; impl `csrc/moe/moe_align_sum_kernels.cu:110-123` (accumulate `:119`), dispatch `:277-294` | `ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.size()),`<br>`            out_hidden_states[begin_chunk_idx:end_chunk_idx])`<br>kernel: `x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);` |
| F2d | Were topk weights already folded in? | **Yes** — by `MUL_ROUTED_WEIGHT=True` on the w2 GEMM (F2a). **`moe_sum` is an unweighted sum. A simulator must not apply routing weights again at the reduction step.** | `fused_moe.py:1785` + `:469-473` vs `:1797-1798` | — |

### F3 — LOCAL expert aggregation vs DISTRIBUTED communication

| # | Reduction / op | file:line | Class |
|---|---|---|---|
| R1 | Softmax normalize over experts in the topk kernel, plus the Python renormalize | `fused_moe.py:895-896` | **LOCAL** |
| R2 | w1 GEMM K-loop accumulate | `fused_moe.py:461/463`, launch `:621` | **LOCAL** (intra-GEMM) |
| R3 | w2 GEMM K-loop accumulate | same kernel, launch `fused_moe.py:1775` | **LOCAL** (intra-GEMM) |
| R4 | `MUL_ROUTED_WEIGHT` scaling | `fused_moe.py:469-473` | **LOCAL** (a scale, not a reduction) |
| R5 | `ops.moe_sum` — sum the k expert rows into 1 | `fused_moe.py:1797-1798` | **LOCAL — this *is* the expert aggregation, and it involves no network** |
| R6 | EP dispatch | `layer.py:1807-1808` | **DISTRIBUTED**, conditional |
| R7 | EP combine | `layer.py:1857` | **DISTRIBUTED**, conditional |
| R8 | TP all-reduce | `layer.py:1860` | **DISTRIBUTED**, conditional |

**R8 — group identity, the key anti-double-count fact.** The group is **`_TP`
(`get_tp_group()`), not the EP group**, even when the profiling scope is *named*
`expert_parallel_allreduce`. `layer.py:1599-1613`:

```
        if (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels):
            return final_hidden_states
        else:
            op_name = ("expert_parallel_allreduce"
                       if self.ep_size > 1 else "moe_tensor_parallel_allreduce")
            return tensor_model_parallel_all_reduce(
                final_hidden_states,
                record_scope_name=op_name,
            )
```

It bottoms out in `tp_group.all_reduce(input_)` with `tp_group = get_tp_group()`
(`vllm/distributed/communication_op.py:16, :30, :32`; `get_tp_group` → `_TP` at
`vllm/distributed/parallel_state.py:907-909`). **The fork's `record_scope_name`
only names the scope; it does not change the group.** *Do not infer the group from
the scope name.*

**The `reduce_results` guard is NOT in `Qwen3MoeSparseMoeBlock.forward`.** In this
version `forward` (`qwen3_moe.py:176-191`) contains no reduce logic at all; the model
only sets `reduce_results=True` at construction (`qwen3_moe.py:150`). The guard lives
in `FusedMoE.forward_impl.reduce_output`, `layer.py:1859-1860`:

```
            if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
                states = self.maybe_all_reduce_tensor_model_parallel(states)
```

`self.tp_size` / `self.ep_size` here are **MoE-local** values from
`FusedMoEParallelConfig.make` (`vllm/model_executor/layers/fused_moe/config.py:266-301`):
with `enable_expert_parallel=True` it returns `tp_size=1, ep_size = dp_size*tp_size`
(`config.py:293-301`), so the `or self.ep_size > 1` arm is what keeps the all-reduce
alive in EP mode — and the tensor still travels over `_TP`, which is correct because
`_EP` is constructed as the `dp × tp` flatten (`vllm/distributed/parallel_state.py:1168-1169`),
so with `DP=1` the EP and TP rank sets coincide.

**R6/R7 condition** — `layer.py:1790-1793`:

```
        do_naive_dispatch_combine: bool = (
            self.dp_size > 1
            and not self.moe_parallel_config.use_deepep_ht_kernels
            and not self.moe_config.use_flashinfer_cutlass_kernels)
```

Dispatch/combine are **skipped entirely when `dp_size == 1`**. When they do run,
`get_ep_group().dispatch/.combine` (`vllm/distributed/parallel_state.py:838-860`)
forward to the CUDA communicator (`cuda_communicator.py:286-293`) and then to
`NaiveAll2AllManager`, whose traffic is on the **DP group**:

- `dispatch` → `naive_multicast` ×2 (hidden_states, router_logits), each a loop of
  per-rank broadcasts — `vllm/distributed/device_communicators/all2all.py:39-43`,
  called at `:50-53`.
- `combine` → `all_hidden_states = self.dp_group.all_reduce(hidden_states)` —
  `all2all.py:64`.
- `self.dp_group = get_dp_group()` —
  `vllm/distributed/device_communicators/base_device_communicator.py:41`.

**Cross-check — is any all2all used on the DEFAULT path? NO.** Two independent
source-level reasons:

1. The default backend is `"naive"`, which **is not an all-to-all at all** — it is DP
   broadcasts plus a DP all-reduce. Docstring `all2all.py:17-23` says *"It uses
   all-reduce under the hood"*; code at `:39-43`, `:64`. Config:
   `vllm/envs.py:159` `VLLM_ALL2ALL_BACKEND: str = "naive"`, runtime
   `vllm/envs.py:1109-1110` `lambda: os.getenv("VLLM_ALL2ALL_BACKEND", "naive"),`;
   selection at `vllm/distributed/device_communicators/cuda_communicator.py:85-88`.
2. The all2all *kernel* paths are gated off regardless:
   `vllm/model_executor/layers/fused_moe/config.py:175-176`
   `def use_all2all_kernels(self): return self.dp_size > 1 and self.use_ep`, and
   `use_pplx_kernels` / `use_deepep_ht_kernels` / `use_deepep_ll_kernels` each
   additionally require `envs.VLLM_ALL2ALL_BACKEND == "pplx" | "deepep_high_throughput" |
   "deepep_low_latency"` (`config.py:178-191`). With the default all three are `False`,
   so `forward_impl_chunked` is never taken (`layer.py:1785-1788`) and the deepep/pplx
   `prepare_finalize` modules are never constructed.

### F3 — final count: network collectives per MoE layer forward

| Topology (default `VLLM_ALL2ALL_BACKEND=naive`) | Collectives per MoE layer | Which |
|---|---|---|
| TP=N>1, DP=1, EP **off** | **1** | 1 × all-reduce on `_TP` (size N), scope `moe_tensor_parallel_allreduce` — `layer.py:1860` → `communication_op.py:30` |
| TP=N>1, DP=1, EP **on** | **1** | 1 × all-reduce on `_TP` (size N), scope `expert_parallel_allreduce`. MoE-local `tp_size` is 1, but `ep_size=N` satisfies the guard at `layer.py:1859`. No dispatch/combine, since `dp_size == 1` (`layer.py:1791`) |
| TP=1, DP=1 (single GPU) | **0** | guard at `layer.py:1859` is false |
| TP=N, DP=M>1, EP on | **2M + 2** | dispatch = 2 × (M DP broadcasts) (`all2all.py:39-43`, ×2 for hidden_states + router_logits); combine = 1 × DP all-reduce (`all2all.py:64`); plus 1 × `_TP` all-reduce (`layer.py:1860`) |

**Simulator guidance.** On the standard TP-and/or-EP, DP=1 deployment there is
**exactly one network collective per MoE layer**: a single hidden-size all-reduce over
the model TP group, emitted **after** `moe_sum`. Everything upstream of it — the
topk-softmax normalize, both GEMM K-loops, `MUL_ROUTED_WEIGHT`, and `moe_sum` — is
intra-GPU and must **not** be charged as communication. The gate (`ReplicatedLinear`,
`vllm/model_executor/layers/linear.py:352-360`) has no collective. Counted separately:
the attention block's `o_proj` (`RowParallelLinear`, `qwen3_moe.py:381`) contributes its
own TP all-reduce per decoder layer — that is outside the MoE block and must not be
double-counted against R8.

### F — fork-vs-upstream for these files

Method: `git diff -w --ignore-blank-lines`. Raw diff over the six files is 977 changed
lines; the whitespace-ignoring diff is 327 — **roughly two thirds of the diff is pure
yapf/line-wrap reformatting.**

| File | raw / `-w` lines | Verdict |
|---|---|---|
| `qwen3_moe.py` | 227 / 185 | **Instrumentation-only.** All adds are `record_function_or_nullcontext(...)` scopes, `record_frontier_op_meta(...)` blocks, and cached attributes used solely by them (`self.rope_scaling` `:233`, `self.layer_idx` `:235`, `self.qkv_bias` `:236`, `self.rms_norm_eps` `:237`, `:444`). No math, no collective, no routing changed. One edge: the `attn_rope` metadata block can `raise RuntimeError` on missing positions meta (`:325-330`) — reachable only while recording is active. |
| `fused_moe.py` | 376 / 86 | **BEHAVIOR CHANGE** — see below. |
| `layer.py` | 214 / 40 | **Instrumentation-only.** `frontier_moe_routing_context(...)` wrappers around `quant_method.apply` (`:1694-1702`, `:1812-1820`), `record_function_or_nullcontext` scopes (`:1477`, `:1802`, `:1855`), `log_frontier_moe_routing_from_context(topk_ids)` (`:1487`, `:1578`). The only non-scope edit is `maybe_all_reduce_tensor_model_parallel` passing `record_scope_name=op_name` (`:1608-1613`) — the underlying call is still `get_tp_group().all_reduce`. |
| `deepep_ht_prepare_finalize.py` | 59 / 5 | **Instrumentation-only.** 1 import + 2 scopes around `buffer.dispatch` (`:88`) and `buffer.combine` (`:274`). |
| `deepep_ll_prepare_finalize.py` | 40 / 8 | **Instrumentation-only.** 1 import + 2 scopes around `low_latency_dispatch` (`:152`) / `low_latency_combine` (`:225`); the remaining `-`/`+` pair is an argument re-wrap. |
| `pplx_prepare_finalize.py` | 61 / 7 | **Instrumentation-only.** 1 import + 3 scopes around `a2a.dispatch` (`:206`, `:243`) and `a2a.combine` (`:322`). |

Byte-identical to upstream and therefore safe to treat as reference:
`modular_kernel.py`, `utils.py`, `all2all.py`, `activation.py`. Instrumentation-only
and always falling through to the same underlying call: `communication_op.py` (+25),
`parallel_state.py` (+11).

Instrumentation is inert by default: `record_function_or_nullcontext` returns
`contextlib.nullcontext()` unless `envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING`
(`vllm/envs.py:188`, default `False`) **or** a Frontier op-logger contextvar is installed
and `frontier_trace.is_active()` (`vllm/v1/utils.py:1179-1200`);
`should_record_frontier_op_meta` returns `False` under the same conditions
(`vllm/v1/utils.py:1212-1216`).

#### `fused_moe.py` — the real changes

| # | Change | HEAD line | Gated? | Default |
|---|---|---|---|---|
| 1 | `uniform_topk()` added — deterministic round-robin routing, all weights `1.0/topk` | `fused_moe.py:908` | `VLLM_MOE_UNIFORM_ROUTING` | **OFF** (`vllm/envs.py:150` `VLLM_MOE_UNIFORM_ROUTING: bool = False`; `vllm/envs.py:1044` `lambda: bool(int(os.getenv("VLLM_MOE_UNIFORM_ROUTING", "0")))`) |
| 2 | `fused_topk` early-returns to `uniform_topk` | `fused_moe.py:980-981` | same env | OFF |
| 3 | `grouped_topk` early-returns to `uniform_topk`, bypassing all grouped logic including `e_score_correction_bias` | `fused_moe.py:1023-1026` | same env | OFF — and not on the Qwen3 path anyway (`use_grouped_topk=False`) |
| 4 | **`vllm_topk_softmax` gains a `renormalize` parameter and passes it as a 5th argument to the C++ op** | `fused_moe.py:884-898`, esp. `:893` | **NOT gated — always on** | always |

**Item 4 is an ABI change and the one thing to flag before any run.** The Python side now
calls the op with **5** arguments (`fused_moe.py:893`; wrapper
`vllm/_custom_ops.py:1506-1510`), but the **in-tree C++ side was not changed** —
`git diff --numstat <tag> HEAD -- csrc/` returns **zero rows**, and the registered schema
is still **4** arguments (`csrc/moe/torch_bindings.cpp:6-9`:
`"topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! token_expert_indices, Tensor gating_output) -> ()"`;
`csrc/moe/moe_ops.h:5-7` declares four `torch::Tensor&`).

The fork does not build its own csrc: `vllm/_moe_C.py:14-21` is a new shim loading
`_moe_C.abi3.so` from `$VLLM_FRONTIER_COMPILED_PACKAGE` (`vllm/_C.py` is the identical
sibling for `_C`). So this call path **requires an out-of-tree prebuilt extension whose
`topk_softmax` accepts `renormalize`**. That prebuilt package could not be located on this
host, so the 5-arg binary was **not verified**. Numerically the change is a no-op either
way: `fused_moe.py:895-896` still performs
`topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)` when
`renormalize`, so an in-kernel renormalize would be idempotent with it.

---

## Could not determine / open items

| # | Item | What is missing |
|---|---|---|
| 1 | Whether the 5-arg `topk_softmax` binary actually exists | `vllm/_moe_C.py:14-21` loads `_moe_C.abi3.so` from `$VLLM_FRONTIER_COMPILED_PACKAGE`, which was not set and whose canonical path (`/local/ycfeng/anaconda3/envs/frontier/lib/python3.10/site-packages/vllm`) does not exist on this host. The only `_moe_C.abi3.so` files present on this machine belong to other environments and register the **4-arg** schema, which would raise on this call path. **Resolving this needs the actual profiling environment**, not more source reading. Note the task forbade installing packages and starting servers, so this was not pursued further. |
| 2 | Empirical confirmation of the coordinator timing constants | The values `4`, `50`, `100`, `5000` are confirmed **statically**. Their *effective* cadence depends on poller wake-ups and the dedup in `core.py:1081`, which only a running DP deployment would show. No server was started, per the task constraints. |
| 3 | `min_stats_update_interval_ms` override reachability | `DPCoordinator.__init__` does not forward it into the `Process` kwargs (`coordinator.py:81-86`), so `100` appears to be unconditional in practice. I found no caller that passes a non-default value, but I did not exhaustively search external/test callers of `DPCoordinatorProc.run_coordinator`. |
| 4 | Behavior of `stats_changed` when an out-of-order report arrives | `coordinator.py:308-310` writes the counts and sets `stats_changed = True` **unconditionally**, including after the out-of-order warning at `:303-307`. Whether that is intended (counts are still the engine's latest known state) or a latent bug is a design question the source does not settle. Flagging it because a simulator that mirrors the ordering check but *not* the unconditional write would diverge. |
| 5 | Exact HEAD line for one `gpu_model_runner.py` seq-len fill | The `self.seq_lens.np[num_reqs:].fill(0)` "Fill unused with 0 for full cuda graph mode" line is cited approximately (`GMR:~1184`). The neighbouring slot-mapping fill at `GMR:1266-1268` was read directly and is exact. |
| 6 | Runtime effect of the fork's forced `torch.cuda.synchronize()` | `GMR:2428` adds a sync when Frontier tracing is active. It changes measured timing but not values. The magnitude was not measured — that requires a GPU run. |
