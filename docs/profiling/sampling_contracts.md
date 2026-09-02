# Profiling Sampling Contracts

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-08-23 | Documented attention context boundaries, MoE EP sampling, and incremental dataset refresh rules. |

## Scope

This document describes the sampling behavior shared by the release-facing
profiling examples and the profiling CLIs. It covers input generation and
dataset coverage. GPU memory checks remain target-specific execution checks.

## Attention Length Boundaries

`max_seq_len` defines the automatic profiling envelope. Automatic standard,
mixed, and true-mixed axes derive their anchors and endpoints from this value.

`max_model_len` defines the runtime context boundary. Each complete sequence
must fit this limit. Explicit decode KV or prefill chunk values may exceed
`max_seq_len` when the complete sequence still fits `max_model_len`. Explicit
values that cannot form a runtime-legal workload fail fast.

Standard attention applies a second, target-local physical-capacity filter
after input generation. When a selected `(model, TP)` target cannot execute an
explicit decode KV value, the profiler emits a `RuntimeWarning` naming the
discarded value and the target capacity. Profiling continues when another
selected target retains that value. The command raises `ValueError` when no
selected target retains a requested explicit decode KV value.

The release wrapper keeps both limits explicit:

```bash
bash examples/profiling/profile_attention_chunked_prefill.sh \
  --max-seq-len 4096 \
  --max-model-len 4096 \
  --dry-run
```

When `--max-model-len` is omitted from this wrapper, it uses the resolved
`--max-seq-len` value.

## MoE Expert-Parallel Domain

When `--expert_parallel_sizes` is omitted, the MoE profiler derives every
positive divisor of each model's `num_experts`. This is the runtime-legal EP
domain for that model. For example, a model with 16 experts resolves to
`[1, 2, 4, 8, 16]`.

The release wrapper follows the same default:

```bash
bash examples/profiling/profile_moe.sh --dry-run
```

Pass an explicit subset when a focused run is required:

```bash
bash examples/profiling/profile_moe.sh \
  --ep-sizes "1 2 4" \
  --dry-run
```

Every explicit EP value must be a positive divisor of the selected model's
expert count. The confirmation output reports the resolved per-model domain
and total configuration count before GPU execution.

## Linear-Operator Token Domain

Automatic token generation includes the configured `max_tokens` endpoint.
`extra_num_tokens` extends the requested profiling domain and may include
values above `max_tokens`. These explicit points are executed at the requested
size, so select values that fit the target; allocation or kernel failures stay
visible to the caller.

The release wrapper exposes both the automatic endpoint and explicit extension
points:

```bash
bash examples/profiling/profile_linear_op.sh \
  --max-tokens 4096 \
  --extra-num-tokens "4097 8192" \
  --dry-run
```

## Existing CSV Compatibility

These sampling changes add coverage points; they do not invalidate timing rows
that already match the current CSV feature and target columns. Keep existing
measurements and collect only missing tuples for the device, model, precision,
parallel configuration, and runtime metadata being used.

Use the following refresh policy:

1. Audit the canonical CSV before launching a GPU job.
2. Build the required tuple set from the current sampler and selected model.
3. Profile only tuples absent from the canonical dataset.
4. Merge old and supplemental rows by non-timing feature columns.
5. Replace the full dataset only when the measurement method, kernel/backend,
   precision, model configuration, or required feature schema changes.

For MoE, an EP=1-only CSV remains usable for EP=1 simulation and training. It
does not cover runtime configurations that request EP greater than 1. Add the
missing EP tuples instead of repeating existing EP=1 measurements.
