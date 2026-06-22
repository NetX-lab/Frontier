# Architecture Examples

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-22 | Removed legacy split-decode terminology from the public PDD surface. |
| 2026-06-14 | Added PDD pd-disaggregation script list, configuration contract, and validation criteria for local PR preparation. |

This directory contains one-click architecture entrypoints for Frontier's release-supported runtime layouts.

## Release Scope

`pre-release-v0.2` foregrounds **PDD / `pd-disaggregation`** examples. Prefill runs in the `PREFILL` cluster, decode runs in the unified `DECODE` cluster, and KV cache is transferred between them. The public PDD example path uses the sequential simulator mode through `--no-enable_parallel_clusters`.

`co-location` examples remain available as baseline comparison recipes and v0.1-compatible architecture references. Additional disaggregated research prototypes outside the PDD path are not exposed as release examples.

## Scripts

| Path | Scenario | Notes |
|------|----------|-------|
| `co-location/run_all.sh` | Full co-location suite | Runs all five offline cases and all five online cases; pass extra Frontier CLI flags after `--` |
| `co-location/offline/dense_model_basic.sh` | Offline dense co-location baseline | Analytical backend by default, dummy execution time, `decode_cuda_graph_mode=full_decode_only`, Chunked Prefill, CSV/JSON metrics |
| `co-location/offline/moe_model_basic.sh` | Offline MoE co-location baseline | Analytical backend by default, dummy execution time, shared-domain MoE invariant, Chunked Prefill, CSV/JSON metrics |
| `co-location/offline/thinking_mode_basic.sh` | Offline Thinking Mode v1 co-location | Analytical backend; one hidden round plus one final round; CSV/JSON metrics |
| `co-location/offline/moe_spec_dec.sh` | Offline MoE Speculative Decoding / MTP | Speculative Decoding / MTP enabled; uses `decode_cuda_graph_mode=none` to avoid the current conflict |
| `co-location/offline/moe_prefix_caching.sh` | Offline MoE Prefix Caching | Prefix Caching enabled with `examples/fixtures/prefix_cache_shared_session_trace.csv` |
| `co-location/online/dense_model_basic_online.sh` | Online dense co-location baseline | Mirrors dense offline settings with analytical backend and `--simulation_mode online` |
| `co-location/online/moe_model_basic_online.sh` | Online MoE co-location baseline | Mirrors MoE offline settings with analytical backend and `--simulation_mode online` |
| `co-location/online/thinking_mode_basic_online.sh` | Online Thinking Mode v1 co-location | Mirrors Thinking Mode offline settings with `--simulation_mode online` |
| `co-location/online/moe_spec_dec_online.sh` | Online MoE Speculative Decoding / MTP | Mirrors Speculative Decoding offline settings with `--simulation_mode online` |
| `co-location/online/moe_prefix_caching_online.sh` | Online MoE Prefix Caching | Replays the same prefix-cache fixture with `--simulation_mode online` |
| `pdd/run_all.sh` | Full PDD suite | Runs all five offline PDD cases and all five online PDD cases; pass extra Frontier CLI flags after `--` |
| `pdd/offline/dense_model_basic.sh` | Offline dense PDD baseline | Sequential `pd-disaggregation`, analytical backend, dummy execution time, Chunked Prefill, CSV/JSON metrics |
| `pdd/offline/moe_model_basic.sh` | Offline MoE PDD baseline | Sequential `pd-disaggregation`, reference-runnable shared-domain MoE topology, Chunked Prefill, CSV/JSON metrics |
| `pdd/offline/thinking_mode_basic.sh` | Offline Thinking Mode v1 PDD | Thinking Mode with two KV transfer handoffs for the one-request smoke configuration |
| `pdd/offline/moe_spec_dec.sh` | Offline MoE PDD Speculative Decoding / MTP | Speculative Decoding enabled; Prefix Caching intentionally disabled; `DECODE_CUDA_GRAPH_MODE=none` |
| `pdd/offline/moe_prefix_caching.sh` | Offline MoE PDD Prefix Caching | Sticky scheduler with `examples/fixtures/prefix_cache_shared_session_trace.csv` |
| `pdd/online/dense_model_basic_online.sh` | Online dense PDD baseline | Mirrors dense offline settings with `--simulation_mode online` |
| `pdd/online/moe_model_basic_online.sh` | Online MoE PDD baseline | Mirrors MoE offline settings with `--simulation_mode online` |
| `pdd/online/thinking_mode_basic_online.sh` | Online Thinking Mode v1 PDD | Mirrors Thinking Mode offline settings with `--simulation_mode online` |
| `pdd/online/moe_spec_dec_online.sh` | Online MoE PDD Speculative Decoding / MTP | Mirrors Speculative Decoding offline settings with `--simulation_mode online` |
| `pdd/online/moe_prefix_caching_online.sh` | Online MoE PDD Prefix Caching | Replays the same prefix-cache fixture with `--simulation_mode online` |

## PDD Configuration Contract

All PDD scripts use these release-supported defaults unless overridden from the shell:

- `--sys_arch pd-disaggregation`
- `--no-enable_parallel_clusters`
- explicit `PREFILL` and unified `DECODE` cluster settings
- `--cc_backend_config_type analytical`
- dummy execution-time prediction enabled by default
- CSV/JSON metrics enabled by default through `--metrics_config_write_metrics` and `--metrics_config_store_request_metrics`
- plots, Chrome trace, and JSON event trace disabled for lightweight one-click artifacts

MoE PDD scripts also enforce that each role's attention and MoE parallel domains match before launching Frontier. This fail-fast check prevents known non-runnable MoE topology combinations from entering the simulator.

## Thinking Mode v1

The Thinking Mode examples use:

- `--enable_thinking_mode`
- `--thinking_depth 2`
- one explicit hidden round via `--thinking_round_prefill_tokens` and `--thinking_round_decode_tokens`
- `--tool_call_latency 0.001`
- explicit `vllm_v1` scheduler settings
- `--cc_backend_config_type analytical` so the one-click smoke run works on a minimal single-replica layout
- CSV/JSON metrics enabled by default, with plots, Chrome trace, and JSON event trace disabled for lightweight artifacts

Under PDD, one user request can produce multiple prefill-to-decode KV handoffs. The default Thinking Mode smoke case completes one request and records two KV transfers.

## Recommended Start Order

```bash
# Full PDD suite for pre-release-v0.2.
bash examples/architecture/pdd/run_all.sh

# PDD offline cases.
bash examples/architecture/pdd/offline/dense_model_basic.sh
bash examples/architecture/pdd/offline/moe_model_basic.sh
bash examples/architecture/pdd/offline/thinking_mode_basic.sh
bash examples/architecture/pdd/offline/moe_spec_dec.sh
bash examples/architecture/pdd/offline/moe_prefix_caching.sh

# PDD online cases.
bash examples/architecture/pdd/online/dense_model_basic_online.sh
bash examples/architecture/pdd/online/moe_model_basic_online.sh
bash examples/architecture/pdd/online/thinking_mode_basic_online.sh
bash examples/architecture/pdd/online/moe_spec_dec_online.sh
bash examples/architecture/pdd/online/moe_prefix_caching_online.sh

# Full co-location comparison suite.
bash examples/architecture/co-location/run_all.sh

# Offline cases.
bash examples/architecture/co-location/offline/dense_model_basic.sh
bash examples/architecture/co-location/offline/moe_model_basic.sh
bash examples/architecture/co-location/offline/thinking_mode_basic.sh
bash examples/architecture/co-location/offline/moe_spec_dec.sh
bash examples/architecture/co-location/offline/moe_prefix_caching.sh

# Online cases.
bash examples/architecture/co-location/online/dense_model_basic_online.sh
bash examples/architecture/co-location/online/moe_model_basic_online.sh
bash examples/architecture/co-location/online/thinking_mode_basic_online.sh
bash examples/architecture/co-location/online/moe_spec_dec_online.sh
bash examples/architecture/co-location/online/moe_prefix_caching_online.sh
```

Use the dense baseline scripts first, then use the Thinking Mode, Speculative Decoding / MTP, and Prefix Caching recipes as advanced cases.

## Cross-validation Criteria

For each offline/online pair:

1. Confirm the script exits with code `0`.
2. Confirm `request_metrics.csv` and `system_metrics.json` exist in the metrics output directory.
3. Record expected request count, actual request rows, completed request rows, total input tokens, total output tokens, mean TTFT, mean latency, and request throughput when present.
4. Confirm offline outputs include the `offline_batch` taxonomy segment and online outputs include `online_serving`.
5. Treat latency differences as expected when online mode preserves request arrival times; investigate only if counts, token totals, output files, or finite numeric metrics diverge unexpectedly.

For every PDD script, the release gate should additionally record:

1. The script exits with code `0`.
2. `request_metrics.csv` and `system_metrics.json` exist in the metrics output directory.
3. Request row count, `total_requests`, and `completed_requests` match the expected case size.
4. KV transfer count, total KV bytes, and KV transfer time are present and positive.
5. Request-level `ttft`, `tpot`, `request_e2e_time`, and `transfer_kv_cache` are finite and positive.
