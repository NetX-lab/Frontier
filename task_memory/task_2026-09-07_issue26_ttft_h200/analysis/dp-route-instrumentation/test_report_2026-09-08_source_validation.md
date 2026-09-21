## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Implemented and checked the authorized routing, snapshot, and enqueue diagnostic extension. |

# DP routing diagnostic implementation and source validation

## Scope and implementation

The user requested parallel RCA and supplemental measurements for DP placement/admission divergence and first-batch operator timing. This is the minimal routing evidence extension described in D018. It changes diagnostic observation only, without modifying Frontier routing or the clean vLLM checkout.

- Diagnostic checkout: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`.
- Branch: `feature/frontier-comparison-instrumentation`.
- Original commit: `361d941c97fcec52e544f74b7ab91c54192de9c9`.
- New commit: `8453dd342c6aa2721aaf4b410998aab2f38bc2ec`.
- Source changes: 60 added lines across `vllm/v1/frontier_trace.py`, `vllm/v1/engine/core_client.py`, and `vllm/v1/core/sched/scheduler.py`.
- Verification script: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/tests/v1/test_frontier_dp_route_source.py`.
- Both pre-change and post-commit diagnostic working trees were clean.

Set `VLLM_FRONTIER_DP_ROUTE_LOG_PATH=/absolute/output/server.route` before process startup. Each process writes `/absolute/output/server.route.pid<PID>.jsonl`. The helper uses a lazy process-specific FileHandler. It does not depend on model trace activation, so frontend records and drained warmup requests remain visible. All hooks test the cached path before copying counts or sampling a timestamp. No GPU synchronization or tensor copy is introduced.

## Record schema

All rows include `pid`, `event`, and `timestamp_monotonic_s`.

| Event | Additional fields | Timestamp and state boundary |
| --- | --- | --- |
| `route` | `request_id`, `client_index`, `client_count`, `eng_start_index`, `explicit_dp_rank`, `selected_dp_rank`, `counts`, `wave` | Monotonic time and nested integer count copy are captured immediately before the existing choice algorithm. Counts precede the route-local waiting-count increment. `selected_dp_rank` maps the selected engine index through the actual managed-rank list. |
| `snapshot_receive` | `client_index`, `counts`, `counts_updated`, `wave`, `engines_running` | Emitted after the existing stats task drains received messages and applies the latest snapshot. This records the frontend-visible applied snapshot, not coordinator production time or every discarded intermediate update. `counts_updated=false` means a wave/running-only update reused existing counts. |
| `enqueue` | `request_id`, `client_index`, `dp_rank` | Reads `request.events[-1].timestamp` immediately after the unchanged `record_event(QUEUED)` call. It does not sample a replacement timestamp. This hook requires existing `log_stats=true`. |

Each `counts` vector contains `[waiting, running]` pairs in the existing managed-engine order. The current case manages global DP ranks `[0, 1]`. Record comparison must keep PID/client identity, filter formal request IDs explicitly, and join request IDs to the engine enqueue rows. The same-host monotonic ordering assumption must be verified against the actual run environment; the source check does not establish cross-host clock equivalence.

## Execution and acceptance criteria

CPU environment: conda `dev-vidur-v03-hopper-e2e`, Python `3.13.13`. No CUDA/vLLM package import is needed: the test loads the real lightweight trace helper and compiles the actual modified methods from their AST. This is a source-method diagnostic check, not a GPU integration result.

Exact commands, from `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`:

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/v1/test_frontier_dp_route_source.py
git diff --check
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python --version
```

Acceptance criteria:

1. Enabled and disabled calls produce identical selected engines, post-route counts, and request-to-engine bookkeeping. The selected engine also matches an independent weighted-score/tie-order reference.
2. Recorded count vectors remain equal to the original pre-route state after the source counts are subsequently mutated.
3. Enqueue records contain the exact existing QUEUED event timestamp and actual configured DP rank.
4. Logging works while the model trace is deactivated and uses a PID-specific output path.
5. No whitespace errors are introduced.

## Observed results and limits

PASS: all three source tests, 0.028 seconds. The routing test covers 486 combinations (81 count states, two tie-start indices, and three explicit-rank choices), each with logging disabled and enabled. The timestamp check observes one event and its exact original timestamp. The logger check succeeds while model tracing is inactive. `git diff --check` passed; Python reported `3.13.13`.

```text
test_enabled_disabled_routing_and_immutable_counts ... ok
test_enqueue_reuses_original_event_timestamp ... ok
test_logger_ignores_model_activation_and_uses_pid ... ok
Ran 3 tests in 0.028s
OK
```

Pending: H200 integration must establish complete formal route/enqueue pairs, route/enqueue rank agreement, route-before-queue ordering, and replay of the recorded routing scores. This source check does not claim that logging is timing-neutral or that the load-snapshot root cause is established. Per-request host FileHandler I/O can perturb diagnostic timings; clean E2E evidence remains separate.
