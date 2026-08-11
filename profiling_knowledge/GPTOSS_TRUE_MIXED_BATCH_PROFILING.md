# Closing the `attn_decode_in_mixed` Gap for gpt-oss / Qwen3 (Dense/GQA Family)

Record of diagnosing and fixing a real simulation crash for gpt-oss-120b, gpt-oss-20b, and
Qwen3-30B-A3B on MI355X, and the (expensive) lessons learned about sizing a profiling sweep.
Companion to [AITER_KERNELS.md](AITER_KERNELS.md) (a *different*, still-open fidelity gap) and
[MI355X_FOUR_MODEL_PROFILING.md](MI355X_FOUR_MODEL_PROFILING.md) (how these models' *existing*
compute profiles were originally collected).

## The crash, and its real root cause

Running `tools/validation/run_validation.py` against real gpt-oss/Qwen3 vLLM or sglang captures
crashed deep inside simulation:

```
ValueError: attn_decode_in_mixed prediction is required for true mixed batches but not found
for cluster monolithic. Please provide merged attention profiling data via atten_input_file
(Option A) and train attn_decode_in_mixed.
```

`attn_decode_in_mixed` is the predictor for a batch that genuinely mixes prefill (new/continuing
requests) and decode (other in-flight requests) in one scheduling step — exactly what real
chunked-prefill serving does under concurrency. Checked directly: every dense-family model's
existing `attention.csv` (qwen3-a3b-30b-moe, gpt-oss-120b, gpt-oss-20b, and even
deepseek-r1-0528's MLA data) had **zero** `is_true_mixed_batch=True` rows. It had simply never
been profiled.

Two more root causes stacked on top of this before the fix was complete:

1. **`block_size` mismatch.** The dense-family attention loader does an *exact-match* filter on
   `block_size`. DeepSeek's data was collected at `block_size=32`; qwen3/gpt-oss data was
   collected at `block_size=16`. The simulator CLI had no override for this at all — added
   `--block-size` to `run_validation.py`/`frontier_cli_translator.py`.
2. **Prediction-grid extrapolation.** The execution-time predictor (a RandomForest) is trained
   on the profiled grid, then queried via a *separately widened* lookup grid
   (`prediction_max_prefill_chunk_size`/`prediction_max_tokens_per_request`, default 4096). The
   underlying training data only went up to `prefill_chunk_size≈1984` — so even after widening
   the lookup grid to stop the `KeyError`, the model was silently **extrapolating** far outside
   its training range for an 8192-token real prefill. RandomForests cannot extrapolate
   meaningfully; a tree just returns whatever its rightmost leaf learned. This is why the
   profiling sweep below also widens `--max_seq_len` to 9216 (exactly `input_len + output_len`
   for this workload), not just adds mixed-batch coverage.

## Why disabling chunked-prefill was a workaround, not a fix

Before this profiling data existed, the only way to avoid the crash was
`--no-enable-chunked-prefill` — which stops the simulator from ever constructing a mixed batch
at all. That's a real fidelity cost (every real server captured here *does* run with chunked
prefill on), not a neutral setting. Once true-mixed data exists and `attn_decode_in_mixed`
trains successfully, drop that flag — `--enable-chunked-prefill` defaults to on.

## Confirmed real `block_size` values (don't guess this)

| Real backend | Real value | How we know |
|---|---|---|
| SGLang | `page_size=1` | Confirmed directly from a real server's startup logs (single-token-granularity radix-tree cache — architecturally different from vLLM's fixed-size paging, not just "a small block size") |
| vLLM | `block_size=16` (assumed) | Never independently confirmed from vLLM's own startup logs the way SGLang's was — the launch script never sets `--block-size` explicitly, so it's whatever vLLM's own current default is |
| `32` | doesn't match either | This is what the *original* deepseek-r1-0528 profiling data happened to use — not confirmed to match either real backend's actual serving config |

Given this, profile at **block_size ∈ {1, 16, 32}** — 1 for SGLang, 16 for the (unconfirmed)
vLLM assumption, 32 kept for backward compatibility with the existing DeepSeek data.

## The 86-hour lesson: profiling has no checkpointing, size the grid deliberately

A first attempt profiled the *full* auto-generated grid (`--max_seq_len 65536`,
`--enable_chunked_prefill_grid_search`, default dense batch/KV-cache grids, across **two**
models in one invocation) — **115,818 points**. After **86.5 hours** it was only 42% done, the
tmux session died for unrelated reasons, and — critically — **zero data was saved**.
`frontier.profiling.attention.main` holds everything in memory and only calls `df.to_csv(...)`
once, after the *entire* run completes. There is no `--resume`/checkpoint flag. An interrupted
run, however far along, produces nothing.

Root cause of the size: the chunked-prefill dimension's point count is
`sum(max_seq_len // chunk_size for chunk_size in chunk_sizes)` — at `max_seq_len=65536` this
explodes because small chunk sizes get enormous `num_partitions`. Concretely (single model, one
block_size, one TP):

| `max_seq_len` | total points |
|---|---|
| 4096 | 12,252 |
| 9216 | 15,970 |
| 65536 | 57,280 (ceiling — the underlying grid space arrays stop generating new values here; going higher wastes time for zero new coverage) |

### The fix: narrow what's known, widen only what's genuinely uncertain

Not every dimension deserves the same treatment. Classify each one:

**Known from the real workload → pin to an exact value, no grid search:**
- `--max_seq_len 9216` / `--max_model_len 9216` — real `input_len + output_len` is exactly
  8192 + 1024, confirmed in every `config.txt`.
- `--fixed_chunked_prefill_size 8192` — real `input_len` is exactly 8192; there's no actual
  chunking to search across for this workload.
- `--true_mixed_prefill_chunk_sizes 8192` / `--true_mixed_prefill_batch_sizes 1` — the token
  budget equals the prefill length exactly, so there's no room for a second simultaneous
  new-prefill request; `num_prefill_seqs=1` is the only realistic case.
- `--true_mixed_prefill_kv_cache_size 0` — every request here is a fresh prefill, never a
  continuation chunk.
- `--num_tensor_parallel_workers 8` — every real capture uses TP=8, confirmed from server launch
  commands.

**Genuinely uncertain (the actual value at any simulated instant depends on scheduling dynamics
we can't pin down in advance) → widen these, don't guess a narrow value:**
- Decode batch size and decode KV-cache-size, for *both* the regular grid
  (`--batch_size_list`/`--decode_kv_cache_size_list`) and the true-mixed grid
  (`--true_mixed_decode_batch_sizes`/`--true_mixed_decode_kv_cache_sizes`).

This alone took the per-model point count from 115,818 down to **158–325** (see the exact
numbers below) — a ~350-700x reduction — while still closing the actual crash-causing gap and
covering the real workload's exact prefill/context length precisely.

### The output-file overwrite trap

`attention.csv`'s output path depends only on `(device, model, profile_method)` — **not**
`block_size` or TP. Running the sweep three times (once per `block_size`) writes to the exact
same path each time, with a blind `df.to_csv(...)` and no merge step. **Running `block_size=16`
after `block_size=1` silently destroys the `block_size=1` output.** Rename immediately after each
run:

```bash
mv attention.csv attention_block${bs}.csv
mv attention_true_mixed.csv attention_true_mixed_block${bs}.csv
mv attention_combined.csv attention_combined_block${bs}.csv
```

### Which file the simulator actually needs

Checked directly in `sklearn_execution_time_predictor.py`: `true_mixed_df` is filtered **out of
the same dataframe** the predictor loads (`attention_df[attention_df["is_true_mixed_batch"]]`) —
it never reads `attention_true_mixed.csv` as a second file. The simulator needs exactly one
file, and it must be the **combined** one (`attention_combined_block{N}.csv` — standard + mixed
+ true-mixed rows already merged), not the plain standard-only file and not the true-mixed-only
file alone. Point Frontier at it directly (no copy-to-default-path needed) with:

```
--random_forrest_execution_time_predictor_config_atten_input_file <path-to-attention_combined_block{N}.csv>
```

(exposed as `--atten-input-file` on `run_validation.py`/`frontier_cli_translator.py` — see
[VALIDATION_TOOL.md](VALIDATION_TOOL.md)).

## The final, validated sweep command

**Packaged as a runnable script**: [`scripts/profile_true_mixed_batch.sh`](scripts/profile_true_mixed_batch.sh)
reproduces exactly the command below, with every value below exposed as a
configurable env var / CLI flag (model list, block_size list, the pinned
workload-shape dimensions, and the widened batch/KV-cache-size grids), plus
the rename-after-each-run step and the `CUDA_VISIBLE_DEVICES`/
`HIP_VISIBLE_DEVICES` export baked in. Run `./scripts/profile_true_mixed_batch.sh --dry-run`
to see the exact resolved commands without touching a GPU. The manual version
is kept below for reference/copy-paste.

Per model, looped over `block_size ∈ {1, 16, 32}`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # see MI355X_ROCM_COOKBOOK.md gotcha #2 for why both

for bs in 1 16 32; do
  python3 -m frontier.profiling.attention.main \
    --disable_ray --num_gpus 8 --device mi355x \
    --models openai/gpt-oss-20b \
    --num_tensor_parallel_workers 8 --max_pipeline_parallel_size 1 \
    --attention_backend TORCH_SDPA --profile_method cuda_event --precision BF16 \
    --block_size $bs \
    --max_seq_len 9216 --max_model_len 9216 \
    --min_batch_size 1 --max_batch_size 512 \
    --batch_size_list 1 2 4 8 16 32 48 64 96 128 160 192 256 320 384 448 512 \
    --decode_kv_cache_size_list 512 1024 2048 3072 4096 5120 6144 8192 9216 \
    --fixed_chunked_prefill_size 8192 \
    --enable_true_mixed \
    --true_mixed_prefill_batch_sizes 1 \
    --true_mixed_prefill_chunk_sizes 8192 \
    --true_mixed_decode_batch_sizes 1 2 4 8 16 32 48 64 96 128 160 192 256 320 384 448 512 \
    --true_mixed_decode_kv_cache_sizes 512 1024 2048 3072 4096 5120 6144 8192 9216 \
    --true_mixed_prefill_kv_cache_size 0 \
    --yes 2>&1 | tee profile_gptoss20b_block${bs}.log
  mv data/profiling/compute/mi355x/openai/gpt-oss-20b/attention.csv \
     data/profiling/compute/mi355x/openai/gpt-oss-20b/attention_block${bs}.csv
  mv data/profiling/compute/mi355x/openai/gpt-oss-20b/attention_true_mixed.csv \
     data/profiling/compute/mi355x/openai/gpt-oss-20b/attention_true_mixed_block${bs}.csv
  mv data/profiling/compute/mi355x/openai/gpt-oss-20b/attention_combined.csv \
     data/profiling/compute/mi355x/openai/gpt-oss-20b/attention_combined_block${bs}.csv
done
```

Swap `--models openai/gpt-oss-20b` for `openai/gpt-oss-120b` for the other model — nothing else
changes (attention profiling only measures the attention op, and both models share identical
attention dimensions: `hidden_size=2880`, 64 heads, 8 KV heads — they only differ in
`num_hidden_layers`/`num_experts`, which this tool never touches).

**Actual observed result** (both models, all three block sizes): 325 items each, **15 seconds
per run**. Dramatically faster than the failed 86-hour attempt predicted for a much larger grid —
consistent with the point-count reduction above, likely helped further by warm
Triton/kernel-compile caches from running the same model repeatedly back-to-back (the failed run
alternated between two different large models, which may have thrashed those caches instead).

## Sanity-checking the result (do this before trusting new profiling data)

Four checks, all passed on the gpt-oss data:

1. **Reproducibility**: two independently-generated rows can land on the same shape by
   coincidence (here, a degenerate single-chunk case and a dense-grid sample both hit exactly
   `prefill_chunk_size=8192`) — their measured times should be nearly identical. Observed:
   <0.5% apart both times.
2. **Monotonicity**: decode time must increase smoothly with batch size and with KV-cache depth,
   no discontinuities, zeros, or NaNs. Observed: smooth 0.134ms → 2.72ms across batch 1→512.
3. **Mixed vs. non-mixed at a comparable shape**: should show a small, plausible overhead, not an
   identical value (predictor training would be pointless) or a wild multiplier (measurement
   bug). Observed: ~4% overhead consistently.
4. **Cross-model consistency**: since gpt-oss-120b and gpt-oss-20b share identical attention
   dimensions, the *same* shape should give nearly the same timing on both. Observed: 0.13376ms
   (20b) vs. 0.13404ms (120b) at batch=1, KV=4096 — 0.05% apart. This is the strongest of the
   four checks, since it's an independent cross-validation, not just internal consistency.

## Open follow-ups

- vLLM's real `block_size` was never independently confirmed the way SGLang's `page_size=1` was
  — worth getting a direct server-log confirmation the same way, rather than assuming 16.
- The dense/GQA AITER kernel gap (see [AITER_KERNELS.md](AITER_KERNELS.md)) is separate from
  everything above and still open — this document's fix makes the *scheduling* (mixed-batch)
  side of the simulation correct; it does nothing for kernel-level timing fidelity
  (`TORCH_SDPA` vs. real `aiter` kernels).
- The real-benchmark data quality issue in
  [REAL_BENCHMARK_DATA_QUALITY.md](REAL_BENCHMARK_DATA_QUALITY.md) (gpt-oss stopping generation
  far short of the requested 1024 tokens) is independent of this profiling work but affects the
  *same* models — both should be accounted for before trusting a gpt-oss comparison report.
