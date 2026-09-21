## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-11 | Recorded H200 warmed MoE subphase diagnostic suite and critical-path RCA limits. |

# H200 warmed MoE subphase diagnostic suite

## 1. Test Script Information

- RJob: `yc26-h200-compute-subphase-20260911-01` (`Succeeded`).
- Cluster: H200 `step_main`, selector `h200`, 8 GPUs, 64 CPU, 400 GiB.
- KV-cache setting: `--num-gpu-blocks-override 310809`.
- Worker: `tests/e2e/issue26_h200_compute_suite.sh`.
- Chain per arm: `issue26_h200_diagnostics_worker.sh` -> `issue26_token_id_client.py` -> identity/batch artifacts.
- Diagnostic source: `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab`, commit `eb4c9a1394ef136f134b5a36538847ec01679c68`.
- Runtime: H200, Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0.
- Formal request identity: `cmpl-pf4096_dc1024:0-0`; lane DP0 TP0--TP3.
- Persistent artifacts: `analysis/h200-compute-subphase-20260911-01/runtime/` and `subphase_summary.json`.

The suite arms were `batch_before`, `compute_attention`, `compute_moe`, `compute_detail`, and `batch_after`. Each arm used three drained 100-request warmup replays followed by 100 formal requests. The operator arms used CUDA-event `default` scopes; no per-scope synchronization was added. The parent `moe_grouped_gemm` scope is treated as an envelope and is never added to its child scopes.

## 2. Validation Criteria

A run was admissible only when each arm had three completed warmup replays, 100 unique formal requests, all DP0 TP0--TP3 boundary rows, and the exact first-formal predicates:

```text
request_ids = [cmpl-pf4096_dc1024:0-0]
batch_size = 1
batch_num_tokens = 4096
batch_num_prefill_tokens = 4096
batch_num_decode_tokens = 0
batch_dp_token_counts = [4096, 1]
```

The MoE analysis monitored the existing scopes:

```text
moe_gating (two calls/layer)
moe_shuffling
moe_grouped_gemm (parent envelope)
moe_grouped_gemm_w1
moe_activation
moe_grouped_gemm_w2
moe_sum
```

The non-overlapping local MoE compute view is `moe_shuffling + W1 + activation + W2 + moe_sum`; the parent grouped-GEMM envelope is used only for overlap consistency. The comparison reports outer-span median, P90, rank maximum, and rank spread, then reports per-rank scope sums and counts.

## 3. Test Results and Evidence

### Execution gates

| Arm | Warmup replay 0/1/2 | Formal | Identity/boundary artifacts |
| --- | --- | --- | --- |
| `batch_before` | 100 / 100 / 100 | 100 unique | all DP0/TP0--3 rows |
| `compute_attention` | 100 / 100 / 100 | 100 unique | direct first-formal predicate check; all DP0/TP0--3 rows |
| `compute_moe` | 100 / 100 / 100 | 100 unique | direct first-formal predicate check; all DP0/TP0--3 rows |
| `compute_detail` | 100 / 100 / 100 | 100 unique | direct first-formal predicate check; all DP0/TP0--3 rows |
| `batch_after` | 100 / 100 / 100 | 100 unique | all DP0/TP0--3 rows |

The two clean control arms passed the existing batch identity validator. The operator arms were checked by direct request-id and predicate selection because the validator's `ops` mode expects the legacy `op_profile_selected` contract and rejects these operator artifacts; this is a validator compatibility limitation, not a missing batch row. The first-formal batch IDs were stable across TP ranks within each arm: `4048` (`batch_before`), `4525` (`compute_attention`), `4211` (`compute_moe`), `4626` (`compute_detail`), and `4529` (`batch_after`).

### Outer span scale

| Arm | Median (ms) | P90 (ms) | Rank max (ms) | Rank spread (ms) |
| --- | ---: | ---: | ---: | ---: |
| `batch_before` | 82.269920 | 82.272194 | 82.283646 | 0.106010 |
| `compute_attention` | 102.303600 | 102.314400 | 102.349251 | 0.077797 |
| `compute_moe` | 134.537086 | 134.613022 | 134.681473 | 0.656067 |
| `compute_detail` | 103.391888 | 103.402687 | 103.571938 | 0.200836 |
| `batch_after` | 137.100624 | 137.124832 | 137.127487 | 0.304581 |

`batch_before` and `batch_after` are separate clean-control runs. Their 54.830704 ms median difference demonstrates that this suite does not provide an ABBA clean-span estimate. Operator arms are intrusive diagnostic executions and are not clean CUDA references.

### MoE scope totals for first formal batch

| Scope | TP0 (ms) | TP1 (ms) | TP2 (ms) | TP3 (ms) | Count/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `moe_gating` | 4.812352 | 24.023648 | 4.883680 | 4.808768 | 96 |
| `moe_shuffling` | 1.324032 | 5.048608 | 1.325312 | 1.360096 | 48 |
| `moe_grouped_gemm` parent | 16.865632 | 24.452032 | 16.778560 | 16.966240 | 48 |
| `moe_sum` | 2.802240 | 2.835392 | 2.811328 | 2.812800 | 48 |

The detail arm independently recorded child scopes in the same first-formal shape and lane, but it is a separate execution and must not be numerically added to `compute_moe`:

| Scope | TP0 (ms) | TP1 (ms) | TP2 (ms) | TP3 (ms) | Count/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `moe_grouped_gemm_w1` | 11.624992 | 7.977568 | 6.514912 | 10.733152 | 48 |
| `moe_activation` | 6.556608 | 6.549664 | 6.546880 | 6.568224 | 48 |
| `moe_grouped_gemm_w2` | 4.054656 | 4.003712 | 4.037920 | 4.073280 | 48 |

### RCA conclusion

The strongest supported conclusion is that rank-local MoE scope spread is not a stable TP hardware or fixed-rank property. In this run, TP1's `moe_gating` and `moe_shuffling` CUDA-event spans are inflated (`24.023648` and `5.048608` ms versus approximately `4.8` and `1.3` ms on the other TP ranks), and the parent grouped-GEMM envelope is also inflated (`24.452032` ms). The child detail run shows a different distribution (`W1` is largest on TP0 at `11.624992` ms while activation and W2 are nearly equal). This rank and phase movement is consistent with queued device work or host submission timing entering CUDA-event default spans; it is not evidence of a fixed TP1 workload or a single hidden 20 ms kernel.

`moe_sum` is stable across ranks (2.802240--2.835392 ms), so routed reduction is not the source of the observed rank spread. The local input shape remains the known `[4097, 2048]` BF16 after naive DP dispatch; equal shape does not establish equal expert token population or padded GEMM block count.

The suite does not close the clean Frontier-vLLM gap. The clean controls differ by 54.830704 ms across separate arms, and all operator arms exceed the established 70--110 ms diagnostic scale. Therefore no arithmetic is performed between these outer spans, and no operator correction, communication correction, or clean/diagnostic reconciliation is claimed.

The unresolved critical-path structure remains the previously established protocol mismatch: Frontier predicts attention TP AR at about 17.899440 ms while measured vLLM reduced attention TP AR is about 5.54--5.71 ms; Frontier has no independent post-MoE TP4 AR boundary, and naive vLLM dispatch/combine/return includes rank arrival and collective waits. The current suite confirms that local MoE and queued work can move the collective arrival participant, but it does not measure expert population/padding or a shared device timeline in one run.

## 4. Next Evidence Required

The next H200 measurement must use one standard warmed run with the union of MoE scopes and routing logging in the same execution, plus the existing completion capture. It must join, for DP0 TP0--TP3 and one `batch_id`, `moe_shuffling`, W1, activation, W2, `moe_sum`, local-stage completion, combine completion, post-MoE TP AR completion, and next-layer start. The run must record routing `per_expert_tokens` and, if rank skew persists, add only diagnostic `num_tokens_post_padded` metadata immediately after `moe_align_block_size`. This is required to distinguish expert population/padding from queued device work and collective arrival. It remains diagnostic and cannot be used as clean span or Frontier correction until repeated evidence closes the attribution.
