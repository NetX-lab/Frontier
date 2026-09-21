# First formal local-stage RCA

The selected batch is `batch_id=3880`, request `cmpl-pf4096_dc1024:0-0`, DP0 TP0–TP3, with one 4096-token prefill request and one DP dummy token (`batch_dp_token_counts=[4096,1]`). Identity validation passed and all four ranks contain 48 `local_moe_apply`, 48 `combine`, and 48 `tp_ar` rows.

The measured local CUDA scope is the non-chunked `quant_method.apply()` call. All ranks have the same input shape `[4097,2048]`, BF16 dtype, and 8,390,656 input elements. TP1 nevertheless spends `47.901280 ms` across 48 local scopes, compared with `27.315488 ms` (TP0), `33.951136 ms` (TP2), and `29.063264 ms` (TP3). TP1 is the latest local-stage end on 47/48 layers and the latest combine start on 47/48 layers.

The boundary after local MoE is not a host submission problem in this run. The local-stage-end to combine-start interval is approximately `0.002944–0.003008 ms` median on the CUDA stream and `0.030232–0.032006 ms` median in host timestamps. The later post-MoE AR duration changes in the opposite direction: TP1 has only `4.770560 ms` of inclusive AR duration, while earlier ranks record `30.354400`, `23.484864`, and `28.319840 ms`. Those AR values are rank-local inclusive wait scopes and cannot be summed.

Evidence therefore supports this causal chain for this first formal run:

```text
TP1 local MoE execution/queued device work is longer
 -> TP1 reaches combine later
 -> TP0/TP2/TP3 enter the TP AR earlier and wait inside it
 -> TP1's AR duration is shorter while the outer batch boundary stays near 90.69 ms
```

The remaining unknown is inside `quant_method.apply()`: this capture does not split grouped GEMM, activation, reduction, expert token counts, or per-kernel scheduling. The equal outer input shape and uniform routing flag do not by themselves prove equal per-expert work after dispatch. A repeat run and, if needed, a routing-count or kernel-level diagnostic are required before attributing the difference to a specific expert shard, GPU execution variance, or internal queued work.

The diagnostic span includes one per-forward `torch.cuda.synchronize()` and JSONL flush I/O, so it cannot close the clean CUDA gate or justify clean/diagnostic reconciliation.
