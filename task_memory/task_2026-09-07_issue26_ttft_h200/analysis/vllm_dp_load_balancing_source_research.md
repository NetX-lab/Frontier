## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-08 | Established the exact vLLM V1 internal-DP routing semantics used by the H200 case and compared them with persistent round-robin and Frontier's current LOR signal. |

# vLLM V1 Internal DP Load-Balancing Source Research

## Outcome

The active vLLM case does **not** use persistent round-robin to assign requests to DP lanes. It uses `DPLBAsyncMPClient`, a load-aware picker. For every request without an explicit `data_parallel_rank`, the picker minimizes:

```text
score(dp) = 4 * waiting_requests(dp) + running_requests(dp)
```

Ties are resolved by a fixed scan order for each API-server process. In this run there is one API-server process, so equal scores select DP0. After each local choice, the front end immediately increments the selected DP lane's local waiting count. This local increment makes a burst arriving between coordinator snapshots look round-robin while the scores remain symmetric, but there is no persistent request counter and no rotating tie-break in v0.10.2.

Therefore, changing Frontier's DP assignment from a per-`schedule()` `local_idx` to a persistent round-robin counter would repair Frontier's own round-robin stream-partition bug, but it would **not** reproduce the active vLLM routing policy. Exact parity needs a distinct vLLM-compatible DP policy with separate waiting/running counts, the `4:1` score, fixed tie order, an immediate local waiting increment, and—if routing decisions must match request by request—the coordinator's delayed snapshot behavior.

## Request Type and Version Context

- Request type: implementation-reference lookup plus version-aware source comparison.
- Active clean ground-truth checkout: `/data/ycfeng/tmp/vLLM-BS` at `46f7b179fd3bf42b9616dc4670cba419afdb2085`.
- Active diagnostic checkout: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908` at `361d941c97fcec52e544f74b7ab91c54192de9c9`.
- Official comparison target: `vllm-project/vllm` tag `v0.10.2`, commit `01efc7ef781391e744ed08c3292817a773d654e6`.
- Network verification used the required company proxy. The active fork's `vllm/v1/engine/core_client.py` and `vllm/v1/engine/coordinator.py` are byte-for-byte identical to official v0.10.2 (`sha256` respectively `bf815c...2189` and `c7ad09...70d1`). The diagnostic checkout has the same two hashes. Thus the DP picker and coordinator semantics below are upstream v0.10.2 semantics, not local instrumentation behavior.

## Why This Run Selects `DPLBAsyncMPClient`

Observed run evidence:

- `issue26_h200_official_ttft_worker.sh:38-45` launches one `vllm serve` endpoint with `--data-parallel-size 2`, `--tensor-parallel-size 4`, and no `--data-parallel-rank`, `--data-parallel-hybrid-lb`, or `--api-server-count` override.
- `vllm/entrypoints/openai/cli_args.py:248-252` gives `--api-server-count` a default of one.
- The current clean `server.log:8,13,15,34-40` observes the same DP2/TP4 arguments, V1 execution, one DP coordinator, and two engine-core processes (`DP0`, `DP1`). This is runtime evidence; it is not inferred only from CLI defaults.

The source path is:

1. `EngineArgs.create_engine_config()` sets `data_parallel_external_lb` only when an explicit data-parallel rank was supplied (`vllm/engine/arg_utils.py:1225-1257,1312-1323`). It remains false in this case.
2. V1 `AsyncLLM` always calls `EngineCoreClient.make_async_mp_client()` (`vllm/v1/engine/async_llm.py:135-143`).
3. With `data_parallel_size > 1` and external LB disabled, the factory returns `DPLBAsyncMPClient` (`vllm/v1/engine/core_client.py:84-102`).

Official documentation independently identifies this topology as self-contained internal load balancing: a single endpoint configured with `--data-parallel-size`, with balancing in the API-server process based on each engine's running and waiting queues. See [vLLM v0.10.2 Data Parallel Deployment](https://docs.vllm.ai/en/v0.10.2/serving/data_parallel_deployment.html#internal-load-balancing).

## Exact Selection Algorithm

`DPLBAsyncMPClient.get_core_engine_for_request()` implements the following behavior (`vllm/v1/engine/core_client.py:1129-1157`; upstream: [`vllm-project/vllm@01efc7e:core_client.py:L1129-L1157`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/core_client.py#L1129-L1157)):

1. If `request.data_parallel_rank` is explicitly set, it uses that rank directly.
2. Otherwise it scans every managed DP engine, starting at `eng_start_index`.
3. It computes `waiting * 4 + running` and retains a candidate only on strict improvement (`score < min_score`). Therefore equal scores keep the first engine in scan order.
4. It increments the selected engine's local waiting count by `client_count` immediately.
5. It records request-to-engine identity only so later aborts reach the correct engine. Finished-request cleanup of that identity map does not decrement the LB counts (`core_client.py:1166-1171`); coordinator snapshots replace those counts.

`eng_start_index` is calculated once as:

```text
floor(number_of_managed_engines * client_index / client_count)
```

It is not advanced after a decision. For this run, `client_count=1`, `client_index=0`, and two engines, so the scan order is always DP0 then DP1.

Concrete consequences for DP=2:

| Front-end view before request | Choice | Reason |
| ----------------------------- | ------ | ------ |
| DP0=`[0,0]`, DP1=`[0,0]` | DP0 | Equal score; fixed order chooses DP0. |
| Immediately following the first choice, before a snapshot: DP0=`[1,0]`, DP1=`[0,0]` | DP1 | Scores are 4 versus 0 because of the local waiting increment. |
| A burst from an all-zero view | DP0, DP1, DP0, DP1, ... | This resembles round-robin only as an emergent result of local load accounting. |
| Equal idle snapshots arrive before each isolated request | DP0 repeatedly | There is no persistent rotation state. |
| DP0=`[1,0]`, DP1=`[0,3]` | DP1 | Scores are 4 versus 3. Current Frontier LOR also chooses DP1 here; this example distinguishes the formula from a hypothetical waiting+running policy, not from current LOR. |
| DP0=`[0,4]`, DP1=`[0,0]` | DP1 | Running load changes the route independently of request ordinal. |

These are direct consequences of the inspected implementation. The exact lane sequence in an online run also depends on when coordinator snapshots reach the API server, so a source-level sequence example is not a claim about the still-running replay's observed request placement.

## What `waiting` and `running` Mean

The vLLM scheduler returns `(len(self.running), len(self.waiting))` from `get_request_counts()` (`vllm/v1/core/sched/scheduler.py:1502-1504`; upstream: [`vllm-project/vllm@01efc7e:scheduler.py:L1090-L1092`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/sched/scheduler.py#L1090-L1092)). The fork's scheduler line numbers differ because diagnostic logging was inserted, but this method is behaviorally unchanged.

- A new request enters `self.waiting` when added (`scheduler.py:1506-1508`).
- Admission pops it from `self.waiting` and appends it to `self.running` (`scheduler.py:794-819`).
- `running` includes admitted requests that remain active, including a request that may not be scheduled in every engine step; vLLM explicitly states that the number scheduled in a step may be smaller than `len(self.running)` (`scheduler.py:890-894`).
- A preempted request is returned to waiting and is counted there until resumed.

The engine publishes these counts only when the pair changes, after an engine step (`vllm/v1/engine/core.py:1075-1099`; upstream: [`vllm-project/vllm@01efc7e:core.py:L1032-L1056`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/core.py#L1032-L1056)). `SchedulerStats` stores the fields as running then waiting; the coordinator deliberately converts them into its `[waiting, running]` representation (`vllm/v1/engine/coordinator.py:285-310`).

## Coordinator Snapshot Cadence and Staleness

The source comment “every 100ms” is a useful shorthand, but the exact contract is more nuanced:

- `DPCoordinatorProc` defaults `min_stats_update_interval_ms` to 100 ms (`coordinator.py:112-134`; upstream: [`vllm-project/vllm@01efc7e:coordinator.py:L112-L134`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/coordinator.py#L112-L134)).
- When counts have changed, the coordinator waits for the 100 ms publication interval; when unchanged, its heartbeat interval is 5 seconds (`coordinator.py:192-218`; upstream: [`vllm-project/vllm@01efc7e:coordinator.py:L192-L218`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/coordinator.py#L192-L218)).
- It waits at least 50 ms in the ordinary collection path to gather engine statistics for a step. When a newer `(wave, step)` arrives while changes are pending, it copies the prior step's counts so it can publish a coherent previous-step view rather than a partially updated mixed-step view (`coordinator.py:200-210,285-310`).
- The API-server task drains queued publications and keeps only the latest, then replaces its entire local `lb_engines` list (`core_client.py:1061-1080`). This replacement reconciles completed/admitted requests and overwrites speculative local increments.

Thus “100 ms snapshots” are delayed, event-driven snapshots with IPC and step-collection ordering, not synchronous reads of live scheduler state and not a guaranteed exactly-periodic timer. Between snapshots, the front end accounts only for requests it has just routed through its immediate waiting increment. It does not locally model admission, running progression, or completion.

## Comparison with Frontier

### Frontier `round_robin`

The known Frontier behavior resets DP selection to `local_idx=0` on each `schedule()` invocation. That causes a series of singleton calls to choose DP0 every time, while a multi-request call alternates lanes. This is a stream-partition bug for a component named round-robin.

A persistent ordinal such as:

```text
replica_id = ordinal % num_replicas
dp_id = (ordinal // num_replicas) % dp_size
```

would make Frontier round-robin invariant to how arrivals are partitioned across `schedule()` calls. It remains valuable as a correction to that policy. It must not be presented as vLLM semantic parity because vLLM has no persistent ordinal and changes choices according to waiting/running load.

### Frontier `lor`

The existing Frontier LOR scheduler is closer in category because it is load-aware, but it is still not a compatible approximation:

- Frontier's `num_pending_requests` for the active V1 replica scheduler is `len(_request_queue) + len(_preempted_requests)` and excludes `_running_requests`. vLLM uses both waiting and running.
- Frontier minimizes an unweighted pending count. vLLM uses `4 * waiting + running`, so it can choose a lane with several running requests over one with a single waiting request.
- Frontier reads simulator state directly. vLLM's front end chooses from a delayed coordinator snapshot plus local increments.
- vLLM has a fixed per-client tie order; the active single client always starts at DP0. A generic minimum selection can match this only if its deterministic tie order is explicitly the same.

## Concrete Change Guidance

Use two separate policies rather than changing what `round_robin` means:

1. Repair Frontier `round_robin` with a persistent global request ordinal and add a regression proving that singleton and burst partitioning produce the same lane sequence. This fixes the independent round-robin defect.
2. Add or extend a vLLM-compatible internal-DP policy through Frontier's existing cluster-scheduler registry. The smallest semantically meaningful state contract is a per-`(replica, dp_lane)` tuple `(waiting, running)`, not the current scalar pending count.
3. At each request route, minimize `4 * waiting + running`, scan in stable DP-rank order, and increment the chosen lane's front-end waiting estimate immediately before routing the next request in the same call.
4. Update the engine-side counts when Frontier's simulated scheduler admits, preempts, or finishes a request. Reading these counts directly at routing time would be a live-state approximation, not equivalent frontend visibility.
5. Model the inspected coordinator publication and frontend snapshot replacement semantics when claiming vLLM alignment. Do not replace them with an exactly-periodic 100 ms sampler. Do not assume an unchanged-count heartbeat is harmless: replacing frontend counts can also overwrite speculative local increments. A routing-decision trace containing request ID, chosen DP rank, visible snapshot and scores would help verification, but extending that diagnostic contract requires separate scope approval.

The live-state version in step 4 matches the scoring semantics but can differ at snapshot boundaries. For TTFT calibration at QPS=2 with long active requests, that difference is material enough that it should be measured rather than assumed negligible. The exact-snapshot model is therefore the recommended target for a claim of vLLM alignment; the live-state version should be labeled an approximation.

Even matching the policy and feedback model does not guarantee identical request placement across two executions: differences in service times, arrivals, and publication ordering can change the visible state. Request-by-request equality still requires observed evidence.

## Official and Upstream Evidence

- [vLLM v0.10.2 Data Parallel Deployment](https://docs.vllm.ai/en/v0.10.2/serving/data_parallel_deployment.html#internal-load-balancing) — defines self-contained internal DP and states that API-server balancing uses engine running/waiting queues.
- [`vllm-project/vllm@01efc7e:vllm/v1/engine/core_client.py:L84-L102`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/core_client.py#L84-L102) — selects `DPLBAsyncMPClient` for internal DP.
- [`vllm-project/vllm@01efc7e:vllm/v1/engine/core_client.py:L1129-L1157`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/core_client.py#L1129-L1157) — exact score, fixed scan order, strict tie behavior, and local increment.
- [`vllm-project/vllm@01efc7e:vllm/v1/engine/coordinator.py:L192-L218`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/coordinator.py#L192-L218) and [`L285-L310`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/coordinator.py#L285-L310) — publication cadence and count assembly.
- [`vllm-project/vllm@01efc7e:vllm/v1/engine/core.py:L1032-L1056`](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/core.py#L1032-L1056) — engine-side count publication after scheduler steps.

## Caveats

- The v0.10.2 documentation describes the signal class but does not document the `4:1` weighting, fixed tie order, or publication algorithm. Those details are established by byte-verified upstream source.
- This report establishes the routing algorithm and active runtime selection. It does not claim an observed DP assignment sequence for the incomplete clean replay because the clean instrumentation does not yet emit per-request DP decisions and the replay had not completed at research time.
- Prefix caching is disabled in the frozen case, so KV-cache-aware affinity is outside this comparison. Official v0.10.2 also says KV-aware internal routing is only a possible future extension.
