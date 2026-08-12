# `tools/validation/`: Real-vs-Simulated Serving Benchmark Validation

What the validation tool does, what changed this session, and what's still missing. Code lives
in `tools/validation/`; real capture data lives in `tools/inference_bench/<model>/<engine>/`
(`{closed-loop,open-loop}/run_<label>/{bench_output.txt,config.txt}`).

## Pipeline

```
real_log_parser.py       parse one sglang/vLLM bench_serving log -> BenchmarkResult per concurrency
real_log_aggregator.py   aggregate N repeated runs of the same config -> mean/std/min/max per metric
frontier_cli_translator.py   translate a real record's workload shape -> a frontier.main CLI invocation
metrics_extractor.py     extract Frontier's own system_metrics.json -> a comparable SimResult
compare_plots.py         render real (mean ± std) vs. simulated on one HTML report
run_validation.py         orchestrates all of the above end to end
```

## What changed this session

### vLLM log parsing (originally sglang-only)

`real_log_parser.py` now treats both backends' logs as one superset schema instead of
sglang-only: matches vLLM's bare `Namespace(...)` args line (sglang prefixes it
`benchmark_args=`), maps vLLM's renamed/extra fields (`Maximum request concurrency`,
`Failed requests`), and — the correctness-critical one — **prefers vLLM's `--input-len`/
`--output-len` over `random_input_len`/`random_output_len`**, which are stale argparse defaults
vLLM's launcher leaves behind unused (confirmed via the log's own
`"Sampling input_len from [8192, 8192]..."` line).

### Multi-repetition aggregation with error bars

A single real run is noisy — `real_log_aggregator.py` discovers a directory of repeated
`run_<label>/` subdirectories (same config, launched N times), and `compare_plots.py` now
renders the real side as mean ± std instead of one noisy point per concurrency level. A rep
directory missing `bench_output.txt` entirely (crashed before writing output) is skipped with a
warning, not a hard failure — confirmed necessary: `deepseek/vllm`'s rep08 is exactly this case.

### New CLI flags on `run_validation.py` / `frontier_cli_translator.py`

| Flag | Why |
|---|---|
| `--block-size` | Was hardcoded to 32 (matched only DeepSeek's profiling data); qwen3/gpt-oss need 16 or 1. See [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md). |
| `--max-tokens-in-batch` | The scheduler's per-iteration token budget (vLLM's `--max-num-batched-tokens` / SGLang's `--chunked-prefill-size` — an aggregate cap shared across everything in one step, **not** a per-request limit). Was never overridable; defaults to `max(prefill_tokens, 8192)`. |
| `--enable-chunked-prefill` / `--no-enable-chunked-prefill` | Workaround for missing mixed-batch profiling data (default on now that real data exists for gpt-oss/qwen3). |
| `--atten-input-file` | Points the predictor at a specific profiling CSV (e.g. a block-size-tagged `attention_combined_block16.csv`) instead of the default `data/profiling/compute/<device>/<model>/attention.csv`. Must be the *combined* file — see [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md#which-file-the-simulator-actually-needs). These `_combined_block{N}.csv` files are exactly what `scripts/profile_true_mixed_batch.sh` produces — run it before wiring up `--atten-input-file` if they don't exist yet for your (model, block_size). |

### Bugs fixed along the way

- **Run-id sanitization**: HF-style model names with a `/` (`openai/gpt-oss-120b`) broke both
  the log file path and Frontier's own run-id validation. Fixed by sanitizing the model name the
  same way Frontier does internally before building `run_id`.
- **Prediction-grid widening**: see
  [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md) — the same
  `--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size`/
  `..._prediction_max_tokens_per_request` widening is applied automatically by
  `frontier_cli_translator.py` based on the real record's own prefill/decode lengths.

## Example commands

Closed-loop, single real run:

```bash
python3 -m tools.validation.run_validation \
  tools/inference_bench/oss-120b/sglang/closed-loop \
  --device mi355x --model-name openai/gpt-oss-120b --attn-tp 8 \
  --block-size 1 \
  --atten-input-file data/profiling/compute/mi355x/openai/gpt-oss-120b/attention_combined_block1.csv \
  --output-dir outputs/validation/oss-120b/sglang \
  --log-dir outputs/validation/oss-120b/sglang/logs \
  --report outputs/validation/oss-120b/sglang/comparison_report.html
```

Passing a *directory of* `run_<label>/` reps (instead of one) automatically aggregates with
error bars — no separate flag needed, `load_and_aggregate()` auto-detects single-run vs. group.

## Known gap: open-loop (Poisson-rate) logs aren't runnable yet

`tools/inference_bench/*/{sglang,vllm}/open-loop/` exists (percentage-of-peak sweeps,
`run_..._pctNN/`), but the validation tool cannot process it correctly yet — three real
problems, not yet fixed:

1. **Wrong QPS source.** `frontier_cli_translator.py`'s existing `request_mode="poisson"` path
   uses `real.request_throughput_req_s` (achieved throughput) — built for approximating an
   open-loop-equivalent rate *from a closed-loop run*. For genuine open-loop captures, the
   correct value is `real.request_rate` (the actual injected rate), which can differ from
   achieved throughput by 10x under real overload (confirmed: pct90 of one qwen3/sglang capture
   showed `injected=0.738` vs. `achieved=0.07` req/s — a genuinely saturated real system, not a
   measurement artifact — see [REAL_BENCHMARK_DATA_QUALITY.md](REAL_BENCHMARK_DATA_QUALITY.md)).
2. **`concurrency` is the wrong sweep key for open-loop data.** It's hardcoded as the
   grouping/x-axis/run-id key throughout the aggregator, translator, and plotting code. In every
   open-loop capture, `concurrency` is a constant safety cap (512) across the whole sweep —
   `request_rate` is what actually varies. Using `concurrency` as-is would collapse all
   pct-tier points into one group and collide their `run_id`s (overwriting each other's sim
   output on disk).
3. **The per-iteration token-budget semantics matter more here than for closed-loop.** SGLang's
   real `chunked-prefill-size` (confirmed 16384 for one real DeepSeek server) being *larger*
   than the real prefill length (8192) is exactly what makes mixed batches the expected case for
   open-loop, not an edge case — worth setting `--max-tokens-in-batch` deliberately for these
   runs rather than leaving it at the closed-loop default.

None of this is a design objection — it's a "generalize `concurrency`-as-sweep-key to
`concurrency`-or-`request_rate` depending on mode" refactor across `real_log_aggregator.py`,
`frontier_cli_translator.py`, and `compare_plots.py`, plus fixing which field feeds the
simulated Poisson rate. Not done yet.
