# Step-5 linear operator profiles

The Step linear producer uses the existing linear-op profiler. It does not run
the simulator or enable measured Step request execution. These are development
measurements until native timing, coverage and typed admission are qualified.

## Physical ownership

| Work | Producer |
| --- | --- |
| SWA/DSA QKV, attention preparation, output projection and attention internals | Native Step attention profiler, including the projections. The linear profile has no attention operators and constructs no generic attention module. |
| Dense MLP | Linear `mlp_up_proj`, `mlp_act`, `mlp_down_proj`; layers 0, 1, 2 and 91, BF16, width 13824, local width 1728 at TP8. |
| Shared expert | Linear `share_expert_up_proj`, `share_expert_act`, `share_expert_down_proj`; routed layers 3–90, BF16, width 1536, local width 192 at TP8. |
| Routed/shared branch merge | Linear `step5_ffn_merge`; native BF16 + BF16 to FP32 addition, full hidden size 4096 per rank. This is a replicated TP1 row, not an FFN-width-sharded operation. |
| FP32 router, sigmoid/bias top-k and routed experts | Existing MoE profiler; the linear producer does not repeat the router. |
| Representative fused input/post-attention norms and embedding lookup | Existing memory portion of the linear producer, with native Step normalization selected through the model profile. |

The activation is selected by the existing model and activation registries. The
Step resolver reads the canonical dense/shared layer limits: 7.0 in this config.
CUDA calls the image's `swiglustep_and_mul_triton`, which applies FP32 SiLU,
clamps its result and then multiplies the other branch. A zero/unclamped limit
selects ordinary SiLU. CPU formulas are references, not CUDA substitutes.

Step normalization uses `make_step5_rms_norm` with FP32 weights and residuals,
zero-centered weights and BF16 output. The input norm receives FP32 hidden
states; the post-attention norm receives BF16 hidden states. Input preparation
casts are outside the norm timer. A separate residual `add` is not planned
when fused normalization already owns it.

## CPU plan and immutable smoke package

```bash
python -m frontier.profiling.linear_op.main \
  --models stepfun-ai/Step-5-Preview --attn_tp 8 --ffn_tp 8 --moe_tp 1 \
  --num_tokens_list 1 16 64 --precision BF16 --dry_run

python -m frontier.profiling.linear_op.step5_job \
  --stage /data/ycfeng/tmp/step5_attention_20261020/linear_smoke_stage01 \
  --dependency-site /data/ycfeng/tmp/step5_attention_20261020/offline_site01 \
  --dry-run
```

Remove `--dry-run` from the second command to create a new local package. Use
the prepared offline dependency directory; no GPU submission occurs during
staging. The package shares the attention smoke's pinned image, source/mount
description, payload hashing and `submit_with_env.sh` credential wrapper. The
parent, not the implementation agent, submits it and keeps its launcher alive.

The smoke uses one H800, token counts 1/16/64, attention/FFN TP8 and MoE TP1.
The canonical planner also creates the TP1 memory measurements. Each of
`cuda_event` and `record_function` writes six rows: three replicated rows and
three sharded FFN rows, for twelve rows in two CSVs. Replicated result
deduplication must preserve disjoint measured operators, including the merge,
without replacing earlier measurements of overlapping operators.

The fresh output root is:

```text
/mnt/codesign-exp/ycfeng/frontier/step5_profiling/linear_smoke_20261020_01
```

`planned_metadata.json` records the actual producer plan, declared activation
variants and normalization configuration. `summary.json` checks output coverage
without turning reporting errors into native workload failures. Inspect the
native kernel traces and numerical/timing evidence before admitting any row.
Startup/JIT and queueing are not included in a guaranteed runtime estimate.

## Remaining coverage and full-grid gates

This slice does not claim the first layer's unfused `residual=None` norm,
endpoint final norm, LM head/logits, sampling, residual/input/output casts or
the full TP-sharded embedding mask/collective path. The existing main-model
linear producer has no independent endpoint norm/logits contract; the MTP
producer is unsupported for Step and must not be relabeled as that endpoint.
Embedding rows represent lookup work, not qualification of the entire TP8
embedding stage. Communication belongs to the communication producer.

Before a full grid, qualify these native kernels on H800, preserve source,
config and runtime identities, close missing physical scopes and typed
admission, and freeze each role's scheduling and graph configuration. The
proposed P32/D32 topology is TP8/DP4/EP32/PP1, with attention TP8/DP4 and routed
MoE TP1/EP32; this one-GPU smoke does not qualify that deployment. Record
`max_num_seqs`, token/context/chunk limits, KV layout/block size, resolved
capture sizes and actual graph dispatch independently for each role. With
the current `max_num_seqs=1`, capture sizes are `[1,2]`: 16/64-token prefill
is native eager and single-token decode is FULL. Do not force a larger graph
ceiling merely to remove eager host gaps from a calibration file.
