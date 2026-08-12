# Four Models on MI355X: Llama-2-7B, Qwen3-30B-A3B, gpt-oss-20b, gpt-oss-120b

Record of taking Frontier from "MI355X collective data exists but nothing can
consume it" to real (non-dummy) end-to-end simulations of four models on
`mi355x`, backed by compute profiles collected on this hardware and predictors
trained on them. Companion to
[MI355X_ROCM_COOKBOOK.md](MI355X_ROCM_COOKBOOK.md) (ROCm environment) and
[DEEPSEEK_V3_MLA_MI355X_JOURNEY.md](DEEPSEEK_V3_MLA_MI355X_JOURNEY.md) (the MLA
equivalent of this document).

## Starting state

- `data/profiling/network/mi355x_8gpu/{all_reduce,send_recv}.csv` already
  collected, but **unusable**: `mi355x` / `mi355x_8gpu` were not registered
  SKUs, and `--replica_config_device` resolves through
  `BaseFixedConfig.create_from_type_string`, which raises on unknown strings.
- No `mi355x` compute profiles for any of the four target models.
- Llama-2-7B and Qwen3-30B-A3B both already resolved — the latter as
  `qwen3-a3b-30b-moe`, which the codebase also aliases from the canonical
  HuggingFace id in `frontier/spec_decode/mtp_registry.py`. Only the two
  gpt-oss models had no config in `data/config/models/`.
- **The `TORCH_SDPA` attention backend the ROCm cookbook describes did not
  exist in this checkout.** The cookbook was written against `/home/dn/Frontier`,
  which is not present on this box. Only `FLASHINFER` (CUDA-only) and `NO_OP`
  (measures nothing) were available, so attention could not be profiled at all.

## 1. Register the device and node SKUs

`MI355X = 9` in `frontier/types/device_sku_type.py`, `MI355X_8GPU = 10` in
`frontier/types/node_sku_type.py`, with matching config dataclasses.
`BaseIntEnum.__str__` lowercases the member name, so these resolve from the
strings `mi355x` and `mi355x_8gpu` — the latter matching the existing network
data directory.

`fp16_tflops=2500` (AMD's published 2.5 PFLOPS dense BF16/FP16; MFU reporting
only, never latency) and `total_memory_gb=288`, which matches what the driver
reports on this box (309220868096 B).

That alone made the existing collective data live: the Vidur CC backend trains
on it at **0.18–0.27% MAPE** for TP=2/4/8.

## 2. Model configs

**Qwen3-30B-A3B needed no new config at all.** An earlier pass added one under
the canonical HuggingFace id and it was removed as a duplicate — worth
recording why, because the reasoning generalises to any "should I add a config
for this model?" question:

- Every field that drives simulated cost (layers, hidden size, head counts,
  expert count, top-k, `moe_intermediate_size`) is identical between the two.
- The model-specific code paths — the prefill-hot gating contract in
  `moe_gating_runtime.py`, the PP payload multiplier in
  `sklearn_execution_time_predictor.py` — test `model_type == "qwen3_moe"`
  first and only fall back to a literal name match, so both configs behave
  identically. Verified, not assumed.
- The only genuine differences, `max_position_embeddings` (40960 vs 262144)
  and `rope_theta` (1e6 vs 1e7), distinguish the original release from
  Qwen3-30B-A3B-Instruct-2507. Neither reaches the collected data: attention
  profiling recorded `max_model_len=4096` either way, and `rope_theta` is not
  an input to any timing path.
- Reusing the repo's name has a concrete benefit: `data/profiling/compute/a800/`
  already holds `qwen3-a3b-30b-moe`, so a800-vs-mi355x comparison for this
  model works. Two names for one architecture would have blocked it.

The one caveat worth knowing: the checked-in `qwen3-a3b-30b-moe.json` carries
Instruct-2507's 262144 context and 1e7 rope_theta while being named for the
base model. That only matters if you simulate contexts beyond 40960; it is left
as-is because existing a800 datasets depend on that file.

gpt-oss did need translation:

| Upstream | Frontier | Why |
|---|---|---|
| `num_local_experts` | `num_experts` | MoE detection reads `num_experts`; without it the model loads as dense. |
| `intermediate_size` (per-expert) | `moe_intermediate_size` | MoE FFN width is read from `moe_intermediate_size`. |
| `quantization_config: mxfp4` | *(omitted)* | `QuantizationConfig` accepts only `{None, "fp8"}` and raises otherwise. |

Both keys are kept side by side so the files still diff cleanly against
upstream, and every deviation is recorded in a `_frontier_deviations` block
inside each JSON.

**Two gpt-oss architecture features are not modeled**, and this is a fidelity
limit rather than a configuration mistake:

- **Alternating sliding-window attention.** The HF loader never reads
  `sliding_window` or `layer_types`, and there is no sliding-window attention
  family. gpt-oss binds to `dense_attention/gqa` — verified — so every layer is
  simulated as full attention. Long-context attention cost and KV-cache
  footprint are both overstated.
- **Attention sinks.** No sink concept exists in any of the three attention
  families (dense, MLA, DSA).

Combined with BF16-instead-of-MXFP4, gpt-oss results describe *a dense-attention
model with gpt-oss's dimensions*, not gpt-oss. Say so when publishing them.

Parameter counts came out right, which is a good end-to-end check on the
translation: 6.5B / 30.3B / 20.4B / 116.6B across 8 devices.

## 3. The `TORCH_SDPA` attention backend

Written from scratch at
`frontier/profiling/attention/backends/torch_sdpa_attention_wrapper.py`, using
only `torch.nn.functional.scaled_dot_product_attention` so it runs on CUDA and
ROCm alike. Emits the same five scopes as the FlashInfer backend, so downstream
training and simulation consume its rows unchanged.

Attention profiling only requires `torch` when the backend is not FlashInfer
(`attention/main.py` gates on this), so this backend runs on the host Python
with no vLLM and no container.

Two bugs found during bring-up, both caught by looking at the numbers rather
than by anything failing:

1. **`kv_cache[:, 0].reshape(...)` copied the entire KV cache on every forward.**
   Slicing the K/V dimension makes the tensor non-contiguous, so `reshape` must
   materialise a copy. `attn_kv_cache_save` measured **3.0 ms** instead of
   0.019 ms — 40x the attention it sits next to — and the write never reached
   the real cache. Fixed by indexing `kv_cache[block_index, kv, block_offset]`
   directly, which is a true scatter. **150x** improvement.
2. **A per-request loop made every measurement launch-overhead bound.**
   Prefill fell only 17% going from 32 heads to 4 (8x less work), and decode was
   flat across TP — i.e. the profile would have told the simulator that
   tensor parallelism does nothing for attention. Since most of the profiling
   grid uses equal sequence lengths, uniform-shape batches now go through a
   single stacked SDPA call, like a real paged kernel. Decode scaling was
   restored (0.279 → 0.164 → 0.120 → 0.097 ms for TP 1→2→4→8) and absolute
   decode times dropped ~3x.

Both the masking math and the batched path were validated against reference
implementations before collection: bottom-right chunked-prefill masking matches
a full-causal reference to ~1e-7, and the batched path matches the per-request
loop to fp16 rounding.

Remaining honest caveats are in the module docstring: ragged batches still loop,
and the paged-cache gather is timed inside the attention scopes because a real
paged kernel would not do it.

## 4. vLLM API drift (three separate breakages)

The container is `rocm/vllm:rocm7.12.0_gfx950-...` with vLLM `0.16.1.dev10`,
much newer than the code was written against. `linear_op` and `moe` both
hard-require an importable vLLM, so these had to be fixed rather than avoided.

1. **`get_rope()` no longer takes `rotary_dim`** (it takes `rope_parameters`).
   Confirmed by inspecting the signature. No code change needed — set
   `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`, which is already wired in.
2. **`moe_impl.py`**: `fused_topk` moved to the `fused_moe` package namespace and
   `get_config_dtype_str` was renamed `_get_config_dtype_str` in
   `fused_moe/config.py`. The other two names in the same
   `from x import (a, b, c)` were fine — one missing name fails the whole
   statement, which is what makes this class of breakage look worse than it is.
   Fixed by importing each name independently with a fallback.
3. **`moe_vllm_kernel.py`**: `invoke_fused_moe_kernel` → `dispatch_fused_moe_kernel`.
   Before aliasing it I compared the new signature against this module's call
   site: all 21 keyword arguments match exactly, so it is a genuine drop-in
   rename, not an adapter. This one gates load-imbalance profiling, so leaving
   it broken (as the cookbook suggested) would have cost the skewed-expert data
   that the simulator's default routing mode depends on.

Plus one that is not an import: **`ReplicatedLinear(disable_tp=True)` still
queries the tensor-parallel group** while constructing weights, so MoE profiling
died instantly with `AssertionError: tensor model parallel group is not
initialized`. Resolved by initialising a one-rank gloo group before construction
— profiling runs one process per GPU, so `world_size=1` is the truthful
description of that process, not a workaround.

## 5. A real hardware bug: Qwen3 at exactly 4000 tokens

`linear_op` profiling of Qwen3-30B-A3B faults the GPU:

```
Memory access fault by GPU node-2 ... Reason: Write access to a read-only page.
```

Characterised rather than guessed at:

- Deterministic across repeated runs.
- **Only `num_tokens=4000`.** 2000, 3000, 3072, 3936, 3968, 4032, 4064 and 4096
  all pass.
- Only TP=1; TP=8 at 4000 passes.
- Only Qwen3 — Llama-2-7B and gpt-oss-20b both pass at 4000.
- Not the MoE path (fails with and without `--is_moe`) and not qk-norm (tested
  with a config that disables it).

The fault kills the whole worker pool, so retrying in place is not an option.
Qwen3 is profiled on the default grid minus that single point — 258 of 259
values, with neighbours at 3968 and 4032 keeping the grid dense there. The
reconstruction is verified equal to the profiler's own grid minus 4000.

This looks like a ROCm/vLLM kernel bug at one specific shape, not a Frontier
bug. Worth reporting upstream if it reproduces outside this stack.

## 6. Collection

One script, three stages. The stage split is the host/container boundary:
attention needs only `torch` (attention/main.py gates the vLLM requirement on
the FlashInfer backend), `linear_op`/`moe` hard-require vLLM, and train/simulate
touches no GPU. Every stage is idempotent — it skips any `(model, op)` whose CSV
already exists, so an interrupted run resumes.

```bash
bash examples/profiling/profile_mi355x.sh attention    # host
bash examples/profiling/profile_mi355x.sh linear_moe   # in the vLLM container
bash examples/profiling/profile_mi355x.sh train_sim    # host
```

The `attention` stage invokes `attention.main` directly instead of using
`profile_attention_chunked_prefill.sh`, because that wrapper hardcodes
`--profile_only_prefill` and the CSV writer overwrites rather than appends —
collecting the phases separately would destroy the first. Passing neither phase
flag profiles both at once. Result: **3072 decode rows per model** where the
release wrapper would have produced zero.

Two grid decisions worth knowing about in the MoE driver:

- **All 16 TP×EP combinations are kept.** The MoE predictor is trained
  per-`(TP, EP)` and never generalises across it, so the combination the
  simulator runs must have been profiled.
- **All three load distributions are requested explicitly.** The profiler
  defaults to `uniform` alone, but the simulator's default
  `--replica_config_moe_routing_mode simulation` models realistic expert-load
  skew. A first pass collected uniform-only and was thrown away and re-run:
  training a grouped-GEMM predictor on uniform rows would have left it
  extrapolating on exactly the skew that mode exists to represent.
- The **token axis** is thinned to ~50 well-spread points instead of 259. It
  feeds a smooth per-token cost curve, so the information loss is negligible
  and it keeps the grid at 7,056 rows/model instead of 37,296.

Collected volumes:

| Model | attention | linear_op | moe |
|---|---|---|---|
| meta-llama/Llama-2-7b-hf | 3388 | 1036 | — (dense) |
| qwen3-a3b-30b-moe | 3388 | 1032 | 7056 |
| openai/gpt-oss-20b | 3388 | 1036 | 7056 |
| openai/gpt-oss-120b | 3388 | 1036 | 7056 |

For MoE models `linear_op` correctly carries no `mlp_*` ops — expert FFN cost
lives in `moe.csv`.

## 7. Results

Predictor accuracy (MAPE), representative across the three MoE models:

| Operator | MAPE |
|---|---|
| `moe_grouped_gemm` | 0.54–0.90% |
| `attn_pre_proj` / `attn_post_proj` / norms | 1.0–5.1% |
| `attn_decode` | 3.8–4.3% |
| `attn_prefill` | 4.2–6.1% |
| `moe_gating_routing_topk` | 4.8–8.0% |
| `attn_kv_cache_save` | **22.8–26.2%** |

`attn_kv_cache_save` is the one weak predictor. It is a ~0.02 ms scatter whose
runtime is dominated by launch overhead and noise, so it is intrinsically hard
to fit; its absolute contribution to TTFT/TPOT is correspondingly small. Worth
improving if KV-cache write cost ever matters specifically.

End-to-end simulation, co-location, TP=8, 32 requests, 512 prefill / 128 decode
tokens, real profiled compute **and** real MI355X collectives
(`--cc_backend_config_type vidur`):

| Model | TTFT (ms) | TPOT (ms) | tok/s | params/device | weights/device |
|---|---|---|---|---|---|
| meta-llama/Llama-2-7b-hf | 218.40 | 11.61 | 12095 | 0.81B | 1.51 GB |
| openai/gpt-oss-20b | 132.87 | 11.69 | 12666 | 2.55B | 4.74 GB |
| qwen3-a3b-30b-moe | 191.32 | 24.72 | 6150 | 3.79B | 7.06 GB |
| openai/gpt-oss-120b | 203.63 | 21.20 | 7073 | 14.58B | 27.15 GB |

Per-device parameter counts multiply back to 6.5B / 20.4B / 30.3B / 116.6B,
matching the published model sizes — a useful independent check that the config
translations in step 2 are right.

## Known limits

- **Absolute latencies are conservative.** Three compounding reasons: the SDPA
  attention backend is a portable reference rather than a tuned paged kernel;
  vLLM ships no tuned grouped-GEMM config for this device string (expect
  `Using default MoE config. Performance might be sub-optimal!` — real
  measurements, untuned kernel); and the runs use
  `--decode_cuda_graph_mode none`. Treat cross-configuration *comparisons* as
  more trustworthy than absolute numbers.
- **gpt-oss is modeled as dense-attention BF16** — see step 2.
- **Prefill barely scales with TP** in the attention profile. At a 64-token
  chunk the kernel is launch-bound on an MI355X, so this is partly real and
  partly a floor imposed by the SDPA backend. A larger
  `--fixed_chunked_prefill_size` would separate the two.
- **`prefill_hot` gating rows were not collected.** They are optional (the
  predictor logs a warning and skips the pseudo-model) and only apply to
  `qwen3_moe`, where they are additive on top of `standalone_legacy` — never a
  substitute. Adding them for Qwen3 requires a second MoE pass and a merge.
- **Only TP=8 predictors are trained.** The profiles cover TP∈{1,2,4,8} and
  EP∈{1,2,4,8}, so other topologies need only a re-run of
  `profile_mi355x.sh train_sim` with `TP=`/`EP=` set — no new collection.
  Keep `attn_tp × attn_dp == moe_tp × moe_ep`.
- **This attention.csv has zero true-mixed-batch rows** — fine for offline/
  low-concurrency simulation, but running any of these four models' real
  serving captures through a concurrent chunked-prefill simulation crashes on
  `attn_decode_in_mixed`. That gap (gpt-oss-20b/120b and qwen3-a3b-30b-moe,
  the three dense/GQA models here) is fixed separately in
  [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md),
  runnable directly via `scripts/profile_true_mixed_batch.sh`.
