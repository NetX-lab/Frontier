# Architecture Examples

This directory contains one-click architecture entrypoints for Frontier's release-supported runtime layout.

## Release Scope

`pre-release-v0.1` supports only `co-location`. Disaggregated architecture examples are intentionally absent from this branch because the runtime guard rejects `pd-disaggregation` and `pd-af-disaggregation`.

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

## Thinking Mode v1

The Thinking Mode examples use:

- `--enable_thinking_mode`
- `--thinking_depth 2`
- one explicit hidden round via `--thinking_round_prefill_tokens` and `--thinking_round_decode_tokens`
- `--tool_call_latency 0.001`
- explicit `vllm_v1` scheduler settings
- `--cc_backend_config_type analytical` so the one-click smoke run works on a minimal single-replica layout
- CSV/JSON metrics enabled by default, with plots, Chrome trace, and JSON event trace disabled for lightweight artifacts

## Recommended Start Order

```bash
# Full suite.
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

Use the baseline scripts first, then use the Speculative Decoding / MTP and Prefix Caching recipes as advanced cases.

## Cross-validation Criteria

For each offline/online pair:

1. Confirm the script exits with code `0`.
2. Confirm `request_metrics.csv` and `system_metrics.json` exist in the metrics output directory.
3. Record expected request count, actual request rows, completed request rows, total input tokens, total output tokens, mean TTFT, mean latency, and request throughput when present.
4. Confirm offline outputs include the `offline_batch` taxonomy segment and online outputs include `online_serving`.
5. Treat latency differences as expected when online mode preserves request arrival times; investigate only if counts, token totals, output files, or finite numeric metrics diverge unexpectedly.
