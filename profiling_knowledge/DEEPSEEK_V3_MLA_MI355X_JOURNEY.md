# DeepSeek-V3 on MI355X: From "No MLA Support" to a Real Simulation

This document records how Frontier went from having no working path to simulate a
DeepSeek-V3-class (MLA + block-FP8 MoE) model on AMD MI355X, to a completed
real (non-dummy) end-to-end simulation using genuinely profiled and trained data.
It's a chronological record of root causes and fixes, useful both as a reference
for repeating this on another MLA model/device and as a map of where the rough
edges still are.

## Goal

Run a real `frontier.main` simulation of `deepseek-v3` on the `mi355x` device
SKU, backed entirely by real profiling data collected on MI355X hardware and
predictors trained on that data — no dummy/analytical shortcuts.

## Starting state

- DeepSeek-V3's model config (`data/config/models/deepseek-v3.json`) already
  resolved via Frontier's generic HuggingFace-config.json loader
  (`BaseModelConfig._create_from_hf_json`), and MLA was already a real,
  first-class attention family (`LATENT_MLA_ATTENTION_FAMILY`) with wiring
  into the execution-time predictor.
- What was missing: no MLA-capable attention *profiling backend* existed
  (only `FlashinferAttentionWrapper`, NVIDIA-only, and `TorchSdpaAttentionWrapper`,
  which explicitly raised `NotImplementedError` for MLA), no MI355X profiling
  data for DeepSeek-V3 at all, and no training-side support for MLA's six
  operator scopes.

## 1. Building the MLA profiling backend

Wrote `TorchSdpaMlaAttentionWrapper`
(`frontier/profiling/attention/backends/torch_sdpa_mla_attention_wrapper.py`),
a portable (CUDA + ROCm) MLA attention backend using only
`torch.nn.functional.scaled_dot_product_attention`, implementing the real
weight-absorbed MLA algorithm and its six timed scopes:

- `attn_mla_kv_cache_save` — write the compressed `(c_kv, k_pe)` latent to the paged cache
- `attn_mla_prefill_kv_up_proj` / `attn_mla_prefill` — prefill up-projects the latent to full K/V, then ordinary causal SDPA
- `attn_mla_decode_q_latent_proj` / `attn_mla_decode` / `attn_mla_v_up_proj` — decode absorbs `W_UK` into Q to attend directly against the compressed cache (MQA-style), then up-projects the output back to `v_head_dim`

Wired in as a new `AttentionBackend.TORCH_SDPA_MLA` enum value, and
`main.py`'s output-finalize/validation step (previously hardcoded to
`DENSE_ATTENTION_FAMILY`) was generalized to resolve the model's real
attention family via `bind_attention_family()`.

## 2. Config-resolution bugs found along the way

Several bugs in `frontier/config/model_config.py`'s generic HF-json loader
surfaced only because DeepSeek-V3 was the first model to exercise these paths:

- **FP8 detection**: `QuantizationConfig.from_dict` defaulted
  `is_checkpoint_fp8_serialized=False` unless the JSON explicitly set it — but
  real HF `config.json` files never carry that key (it's an internal vLLM
  concept). Fixed to default it from `quant_method == "fp8"` when absent.
- **MoE detection**: `is_moe = int(cfg.get("num_experts", 0)) > 1` — every
  other MoE config in the repo uses `num_experts`, but DeepSeek-V2/V3 use
  `n_routed_experts` instead. Added it as a fallback alias.

## 3. Sanity check and the first real MLA profiling run

A tiny (`--min/max_batch_size 1`) sanity run on the MI355X box validated the
wrapper end-to-end before committing to a full sweep — the six-scope timing
split showed exactly the expected pattern (prefill scopes real, decode scopes
at timer-overhead noise floor during a prefill row, and vice versa during a
decode row), confirming the algorithm was implemented correctly.

Two bugs found only once real hardware was involved:

- `attention_wrapper.py`'s shared `AttentionWrapper.profile()` method only
  ever emitted the *dense* family's metadata columns
  (`n_q_head`, `prefill_chunk_size`, ...) — never the MLA-specific ones
  (`kv_lora_rank`, `qk_nope_head_dim`, `batch_num_prefill_tokens`, ...), so the
  CSV finalize step rejected the output. Added an MLA-conditional block.
- The full TP-sweep (TP ∈ {1,2,4,8}, chunked-prefill grid) then ran cleanly:
  52/52 configurations, `KV_heads=1` correctly constant across all TP sizes
  (MLA's compressed cache isn't partitioned by TP) while `Q_heads` split
  correctly.

## 4. linear_op profiling: three more vLLM API drift issues

Re-profiling linear ops (`attn_pre_proj`, `attn_post_proj`, `attn_rope`, norms,
MTP) at DeepSeek-V3's real `hidden_size=7168` shape hit a chain of vLLM API
drift in `frontier/profiling/common/parallel_utils/tensor_parallel_layers.py`
— the installed vLLM (`0.16.1.dev`) had moved on from what this file expected:

1. `Fp8LinearOp` (tensor-wise FP8 op) was removed entirely, but it was
   imported in the *same* `try/except` block as symbols the *block-wise* path
   still needed (`vllm_ops`, `GroupShape`, ...) — one missing name broke all
   of them. Split into an independent, non-fatal import.
2. `torch.ops.vllm.apply_w8a8_block_fp8_linear` (a single unified custom op)
   was also gone, replaced by vLLM's own `W8A8BlockFp8LinearOp` class, which
   internally dispatches across Cutlass/Triton/DeepGEMM/aiter kernels per
   platform. Rewired `_apply_fp8_linear` to construct and use that class
   instead, mirroring vLLM's own `Fp8LinearMethod` construction args exactly
   (including AMD's `aiter` support path).
3. `W8A8BlockFp8LinearOp.__init__` turned out to build a vLLM `CustomOp`
   internally, which asserts a live vLLM engine config is set — something
   Frontier's standalone profiling harness never sets up. Wrapped its
   construction in a trivial dummy `set_current_vllm_config(VllmConfig())`
   context, satisfying the assertion without needing a real engine.

Result: 76/76 linear_op configurations profiled successfully, using the real
block-FP8 kernel path (with vLLM's expected "no autotuned kernel config found,
falling back to default" warnings — harmless, just not peak-tuned).

## 5. MoE profiling: real fused-kernel path and a genuine feature bug

- Fixed the same `is_moe` detection bug from step 2, which also made the MoE
  profiling banner initially show `Num Experts: 0`.
- DeepSeek-V3's FP8 quantization forces `use_vllm_kernel=True` in
  `moe_wrapper.py` (the real fused kernel is required for FP8 accuracy) — but
  our command had passed `--disable_load_imbalance`, which is *also* the flag
  that controls kernel selection (its help text says so explicitly). Dropped
  it, restoring the default (`enable_load_imbalance=True`).
- That exposed `invoke_fused_moe_kernel` and `get_config_dtype_str` having
  been renamed (to `dispatch_fused_moe_kernel` and `_get_config_dtype_str`
  respectively — confirmed call-signature-compatible) in
  `moe_vllm_kernel.py`'s own separate vLLM import block — same
  one-broken-name-kills-everything pattern as before, same fix pattern
  (nested try/except, old name first).
- `MoEGatingNetwork`'s `ReplicatedLinear(disable_tp=True)` still queried
  `get_tensor_model_parallel_rank()` during weight construction in this vLLM
  version — `disable_tp` no longer fully bypasses it. Added a one-time,
  idempotent single-process (`world_size=1`) vLLM distributed-group
  initialization (`init_distributed_environment` + `initialize_model_parallel`,
  `gloo` backend) before constructing it — satisfies the real requirement
  the code always intended to skip, rather than fighting vLLM further.

Result: 528/528 MoE configurations profiled (TP × EP × token-count × load-distribution grid).

## 6. Training: MLA support didn't exist yet

`frontier/training/attention_trainer.py` was entirely hardcoded to
`DENSE_ATTENTION_FAMILY` throughout — model list, feature columns, target
columns, dataset routing — despite the underlying helper functions in
`frontier/attention/profiling_mapping.py` already having full MLA support
built in (they're family-parameterized; nobody had ever pointed them at MLA
from the trainer). Added:

- Family detection in `__init__` (`bind_attention_family(model_config)`),
  switching between dense and MLA class-level constants for every
  model-list/feature/target lookup that was previously hardcoded to the dense
  ones.
- MLA prefill/decode phase routing in `train()`, derived programmatically from
  each MLA operator's declared `phases` (not hand-listed), so it stays in
  sync with the family spec automatically.
- Skipped the dense-only "mixed batch" pseudo-models (`attn_prefill_mixed`,
  `attn_decode_in_mixed`) for MLA, since they rely on dense-specific CSV
  columns with no MLA equivalent.

Verified with a synthetic 25-row MLA dataset before touching the remote box —
all six MLA models trained with correct feature/target routing.

`linear_op` training just needed `--is_moe` on the CLI (the trainer already
fully supported it; it just wasn't being passed).

## 7. First simulation attempts: three real, distinct config bugs

Running `frontier.main` surfaced issues that only exist once a real MLA + MoE
model is exercised end-to-end for the first time:

1. **`n_kv_head` semantic bug** — `AttentionWrapper.profile()`'s MLA branch
   recorded the model's raw `num_kv_heads` (128) instead of MLA's
   family-resolved runtime value (always `1`, since MLA's whole point is a
   single shared compressed KV "head"). The simulator's own filter correctly
   expected `1`. Fixed in the profiling wrapper (for future runs) and patched
   the already-collected CSV in place (a metadata correction, not a
   re-measurement — the timing values were never wrong). Also fixed the same
   bug's twin in `attention_trainer.py`'s own filter, which would otherwise
   have broken on the corrected data.
2. **`block_size` mismatch** — the simulator defaults to `16` (dense
   convention); MLA profiling required `32`. Just a missing
   `--vllm_v1_scheduler_config_block_size 32` CLI flag.
3. **MoE gating context mismatch** — profiled with `--gating_runtime_context
   prefill_hot` (copying `profile_moe.sh`'s script default) instead of the
   real default `standalone_legacy`. `prefill_hot` turned out to be an
   empirically-tuned correction specific to Qwen3-MoE's profiling/live-kernel
   mismatch (gated by `should_enable_prefill_hot_moe_gating_contract`, which
   only returns `True` for `qwen3_moe`) and is *only ever additive* on top of
   `standalone_legacy` data — never a substitute. Re-profiled MoE with the
   correct (default) context.
4. **Parallel-domain constraint** — Frontier's co-location architecture
   requires `attn_tp × attn_dp == moe_tp × moe_ep`. Since the full TP×EP grid
   (1,2,4,8 each) had already been profiled and trained for both attention and
   MoE, no new data collection was needed — just picking a self-consistent
   combination (`attn_tp=8, attn_dp=1, moe_tp=1, moe_ep=8`).
5. **Memory capacity** — an initial `attn_tp=1` run correctly hit a realistic
   `FRONTIER_MEMORY_OOM`: DeepSeek-V3's ~1.4TB parameter footprint (at the
   simulated precision mix) genuinely cannot fit on one 288GB MI355X. Not a
   bug — proof the simulator's capacity check works. Resolved by using the
   same `attn_tp=8` topology from the point above.

## Final working command

```bash
python3 -m frontier.main \
  --simulation_mode offline \
  --sys_arch co-location \
  --cc_backend_config_type analytical \
  --cluster_config_num_replicas 1 \
  --replica_config_device mi355x \
  --replica_config_model_name deepseek-v3 \
  --replica_config_attn_tensor_parallel_size 8 \
  --replica_config_attn_data_parallel_size 1 \
  --replica_config_moe_tensor_parallel_size 1 \
  --replica_config_moe_expert_parallel_size 8 \
  --replica_config_num_pipeline_stages 1 \
  --replica_scheduler_config_type vllm_v1 \
  --vllm_v1_scheduler_config_block_size 32 \
  --decode_cuda_graph_mode none \
  --no-vllm_v1_scheduler_config_enable_chunked_prefill \
  --vllm_v1_scheduler_config_max_tokens_in_batch 4096 \
  --vllm_v1_scheduler_config_long_prefill_token_threshold 0 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 16 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 512 \
  --fixed_request_length_generator_config_decode_tokens 128 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 1.0 \
  --metrics_config_output_dir outputs/mi355x_deepseek \
  --metrics_config_run_id deepseek_v3_first_real_sim \
  --metrics_config_write_metrics \
  --metrics_config_store_request_metrics \
  --metrics_config_store_batch_metrics \
  --metrics_config_store_token_completion_metrics \
  --metrics_config_store_utilization_metrics \
  --no-metrics_config_store_plots \
  --no-metrics_config_enable_chrome_trace \
  --no-metrics_config_write_json_trace
```

Output: `outputs/mi355x_deepseek/deepseek_v3/offline_batch/deepseek_v3_first_real_sim/`
(`request_metrics.csv`, `system_metrics.json`, `op_precision_metadata.csv`).

## Known follow-ups

- **Code sync**: all fixes above are applied to the local `/home/dn/Frontier`
  checkout. Steps 1–6's fixes are also on the remote MI355X box; the two
  fixes from step 7 (`n_kv_head`, `standalone_legacy` context) only exist
  locally — push them to the remote box before profiling any other MLA model
  there.
- **Decode-phase MAPE is high** (`attn_mla_decode` ~64%, `attn_mla_decode_q_latent_proj`
  ~22%, `attn_mla_v_up_proj` ~22%) because only 8 decode rows exist in the
  current profiling sweep. A denser decode-side sweep (more batch/KV-cache-size
  points) would materially improve these.
- **This run used TP=1/EP=1-trained MoE alongside TP=8 attention** only
  because the full TP×EP grid was already profiled; if you change topology,
  confirm profiled data exists for that exact combination first — the MoE
  predictor is trained per-(TP,EP), not generalized across it.
- **`prefill_hot` vs `standalone_legacy`**: if DeepSeek-V3-specific gating
  behavior ever needs the `prefill_hot`-style correction, that requires
  extending `should_enable_prefill_hot_moe_gating_contract` to recognize
  `deepseek_v3`'s `model_type`/`model_arch`, not just reusing Qwen3-MoE's flag.
- **Real AITER MLA kernels vs. this doc's `TORCH_SDPA_MLA` reference backend**:
  a real `AiterMlaAttentionWrapper` now exists (server3/server8 only — see
  [AITER_KERNELS.md](AITER_KERNELS.md)), giving actual production-kernel
  timing instead of the portable reference backend built in step 1 above.
  `scripts/profile_deepseek_aiter_mla.sh` wraps both (`--attention-backend
  AITER` or `TORCH_SDPA_MLA`) with a preflight check for which checkout you're
  running on; its default grid is sanity-check scale, not a re-run of this
  doc's full TP×chunked-prefill sweep — widen deliberately if you use it to
  reproduce that.
