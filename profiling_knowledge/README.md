# Profiling Knowledge

Everything learned collecting and validating real-hardware profiling data for Frontier on
MI355X, in one place. Previously scattered as loose `.md` files at the repo root; consolidated
here so it stops growing that way.

This is a working-knowledge / journey record, not the official release documentation — for the
user-facing "how do I run the public `examples/profiling/*.sh` scripts" guide, see
[`docs/profiling/README.md`](../docs/profiling/README.md) instead. The two overlap in places;
where they do, this folder has the more detailed "why," the official guide has the more stable
"how."

## Reading order

If you're new to this, read in this order. If you're looking for one specific thing, the table
below gets you there directly.

1. **[HARDWARE_COOKBOOK.md](HARDWARE_COOKBOOK.md)** — device-agnostic profiling workflow:
   registering a new device SKU, the three compute-profiling categories
   (linear_op/attention/moe), network/collective profiling, training predictors, running a real
   (non-dummy) simulation. Start here if profiling a device/model combination for the first
   time, on *any* hardware.
2. **[MI355X_ROCM_COOKBOOK.md](MI355X_ROCM_COOKBOOK.md)** — the MI355X/ROCm-specific delta on
   top of the above: environment setup, the `CUDA_VISIBLE_DEVICES`-on-AMD gotcha, why
   `TORCH_SDPA` exists (no FlashInfer on ROCm), vLLM API-drift fixes, a real GPU kernel bug at
   exactly 4000 tokens (Qwen3-specific).
3. **[DEEPSEEK_V3_MLA_MI355X_JOURNEY.md](DEEPSEEK_V3_MLA_MI355X_JOURNEY.md)** — building MLA
   attention profiling from nothing (`TorchSdpaMlaAttentionWrapper`), the config-resolution bugs
   that only surface with an MLA+MoE model, and the first real end-to-end DeepSeek-V3 simulation.
4. **[MI355X_FOUR_MODEL_PROFILING.md](MI355X_FOUR_MODEL_PROFILING.md)** — the same kind of
   journey for Llama-2-7B, Qwen3-30B-A3B, gpt-oss-20b, gpt-oss-120b: device/model registration,
   building the dense-family `TORCH_SDPA` backend, more vLLM API drift, results and known limits.
   This is the baseline the rest of this folder's gpt-oss/qwen3 work builds on.
5. **[AITER_KERNELS.md](AITER_KERNELS.md)** — what AITER is, that an MLA-only wrapper already
   exists (on two of five checkouts — see the infra map), and the real, unclosed gap: no
   dense/GQA AITER wrapper exists anywhere, so gpt-oss/qwen3 profiling still uses the portable,
   not-peak-tuned `TORCH_SDPA` reference backend.
6. **[GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md)** — diagnosing
   and fixing a real `attn_decode_in_mixed` simulation crash for gpt-oss/qwen3: the missing
   mixed-batch profiling data, the block_size mismatch, the prediction-grid extrapolation issue,
   an 86-hour lesson about profiling having no checkpointing, and the narrow-vs-wide grid design
   that fixed it in ~15 seconds per model per block_size.
7. **[VALIDATION_TOOL.md](VALIDATION_TOOL.md)** — the `tools/validation/` pipeline: what it does,
   the vLLM-log-parsing and multi-repetition-error-bar work, every new CLI flag added and why,
   and the still-open gap (open-loop/Poisson-rate logs aren't runnable yet).
8. **[REAL_BENCHMARK_DATA_QUALITY.md](REAL_BENCHMARK_DATA_QUALITY.md)** — two real data-quality
   findings in the captured benchmarks themselves: confirming genuine server overload in
   open-loop captures (not a benchmark artifact), and gpt-oss stopping generation at ~6-53% of
   the requested output length depending on backend (not request failure — confirmed via
   `successful_requests == num_prompts` everywhere).
9. **[INFRASTRUCTURE_MAP.md](INFRASTRUCTURE_MAP.md)** — every server/checkout in play, what's
   missing on which one (checked directly, not assumed), SCP patterns, and the
   GPU-discovery/long-running-job gotchas that bit us more than once.

## Scripts

Runnable, configurable versions of the recipes above — for "just run this," not just reading
about it. Every value is an env var (`VAR=... ./scripts/foo.sh`) or CLI flag (`--flag value`);
every script supports `--dry-run` to print the resolved command(s) without touching a GPU.

| Script | What it does | Status |
|---|---|---|
| [`scripts/profile_true_mixed_batch.sh`](scripts/profile_true_mixed_batch.sh) | The gpt-oss/qwen3 true-mixed-batch sweep from [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md#the-final-validated-sweep-command) — loops model × block_size, pins the known workload-shape dimensions, widens the scheduling-dependent ones, renames outputs after each run. | Validated (gpt-oss-20b/120b actually run; qwen3-a3b-30b-moe included by the same recipe but not independently re-run) |
| [`scripts/profile_deepseek_aiter_mla.sh`](scripts/profile_deepseek_aiter_mla.sh) | DeepSeek MLA attention profiling via real AITER kernels (or the portable `TORCH_SDPA_MLA` fallback) — preflight-checks AITER's checkout-dependent availability before running. See [AITER_KERNELS.md](AITER_KERNELS.md). | Preflight check validated; default grid is sanity-check scale, not a validated production sweep — widen deliberately |

## Quick index by question

| If you're asking... | Go to |
|---|---|
| "How do I profile a new device from scratch?" | [HARDWARE_COOKBOOK.md](HARDWARE_COOKBOOK.md) |
| "Why doesn't `nvidia-smi`/FlashInfer work on this AMD box?" | [MI355X_ROCM_COOKBOOK.md](MI355X_ROCM_COOKBOOK.md) |
| "How do I add MLA support for a new model?" | [DEEPSEEK_V3_MLA_MI355X_JOURNEY.md](DEEPSEEK_V3_MLA_MI355X_JOURNEY.md) |
| "What's the baseline gpt-oss/qwen3 MI355X profiling recipe?" | [MI355X_FOUR_MODEL_PROFILING.md](MI355X_FOUR_MODEL_PROFILING.md) |
| "Should I profile with AITER kernels? Where are they?" | [AITER_KERNELS.md](AITER_KERNELS.md) |
| "Why did my gpt-oss/qwen3 simulation crash on `attn_decode_in_mixed`?" | [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md) |
| "What's the exact profiling command, and how big should the grid be?" | [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md#the-final-validated-sweep-command) |
| "How do I run `run_validation.py`? What flags exist?" | [VALIDATION_TOOL.md](VALIDATION_TOOL.md) |
| "Why is my gpt-oss real-vs-sim comparison so far off?" | [REAL_BENCHMARK_DATA_QUALITY.md](REAL_BENCHMARK_DATA_QUALITY.md) |
| "Which server/checkout has X? What's missing where?" | [INFRASTRUCTURE_MAP.md](INFRASTRUCTURE_MAP.md) |

## Still-open gaps (not fixed, tracked so they don't get re-discovered)

- No dense/GQA AITER attention wrapper — [AITER_KERNELS.md](AITER_KERNELS.md).
- Open-loop (Poisson-rate) real logs can't run through the validation tool yet —
  [VALIDATION_TOOL.md](VALIDATION_TOOL.md#known-gap-open-loop-poisson-rate-logs-arent-runnable-yet).
- gpt-oss real captures under-generate tokens relative to the intended workload, especially on
  vLLM (~6% of target) — [REAL_BENCHMARK_DATA_QUALITY.md](REAL_BENCHMARK_DATA_QUALITY.md).
- vLLM's real `block_size` was never independently confirmed the way SGLang's was —
  [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md#confirmed-real-block_size-values-dont-guess-this).
- Checkout drift across `server1`/`server3`/`server8`/both local checkouts — nothing syncs these
  automatically today — [INFRASTRUCTURE_MAP.md](INFRASTRUCTURE_MAP.md).
