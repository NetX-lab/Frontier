# Profiling Examples

## Scope

This directory is the release-facing entry point for profiling examples. The migration is **non-destructive**: the legacy internal scripts under `frontier/profiling/example` remain in place because repository rules forbid deleting or moving files without explicit approval.

All examples use the canonical output taxonomy:

```text
data/profiling/compute/<device>/<model>/
├── linear_op.csv
├── attention.csv
└── moe.csv
```

You can override `PYTHON_BIN`, `MODEL`, `DEVICE`, `DATA_DIR_BASE`, and script-specific sizing variables from the shell or via CLI flags. Smoke scripts reject unknown CLI arguments before running dependency checks, so mistyped validation commands fail fast instead of silently validating a different target.

## Operator Coverage

| Script | Operator class | Primary output | Notes |
|--------|----------------|----------------|-------|
| `profile_linear_op.sh` | `linear_op` | `data/profiling/compute/<device>/<model>/linear_op.csv` | Attention projections, normalization, residual, dense FFN, and shared-expert linear work. |
| `profile_attention_chunked_prefill.sh` | `attention` | `data/profiling/compute/<device>/<model>/attention.csv` | Explicit chunked prefill recipe using `--fixed_chunked_prefill_size` and `--enable_chunked_prefill_grid_search`. |
| `profile_moe.sh` | `moe` | `data/profiling/compute/<device>/<model>/moe.csv` | MoE expert compute with routing/gating runtime controls. |

## Support Matrix

| Contract | Release wrapper behavior |
|----------|--------------------------|
| Measurement default | The wrapper default is `PROFILE_METHOD=cuda_event`, producing simulator-ready `CUDA_EVENT` rows. The lower-level implementation CLIs may default to `record_function` for `KERNEL_ONLY`; the wrappers pass `--profile_method "$PROFILE_METHOD"` explicitly. |
| Output taxonomy | All wrapper outputs are routed under `data/profiling/compute/<device>/<model>/`. |
| Dry-run scope | `--dry-run` validates command construction, path routing, argument parsing, and defaults; it does not collect GPU timing data. |
| Downstream CSV smokes | `smoke_simulator_dense_csv.sh` and `smoke_simulator_moe_csv.sh` consume checked-in CSV profiles by default, rather than freshly generated CSVs from the current shell session. |
| MoE routing alignment | The MoE downstream smoke binds `moe_routing_distribution_type=random` to `uniform_topk` rows. A mismatched CSV fails fast instead of falling back to another routing family. |

Current profiler CSVs may include `model_architecture_profile` and
`typed_operator_contracts`. These columns bind each measured operator to its
owner and semantic TP mode. Routed-expert operators use `moe_tp` and
`expert_parallel_size`; shared-expert linear operators remain in
`linear_op.csv` and use their architecture-profile-owned TP domain.

## Quick Validation

Run dry-run command validation without requiring GPU profiling dependencies:

```bash
bash examples/profiling/profile_linear_op.sh --dry-run
bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
bash examples/profiling/profile_moe.sh --dry-run
```

Validate existing profiling metadata:

```bash
bash examples/profiling/smoke_metadata.sh \
  --data_path data/profiling/compute/rtx_pro_6000/qwen2_dense_test
```

## Downstream Simulator Closure

These smoke scripts feed checked-in CSV profiles directly into the simulator with dummy predictor mode disabled. They verify that the simulator can parse and consume profiling outputs:

```bash
bash examples/profiling/smoke_simulator_dense_csv.sh
bash examples/profiling/smoke_simulator_moe_csv.sh
```

The MoE downstream smoke sets `--replica_config_moe_routing_distribution_type random` because the checked-in tiny MoE dataset contains `routing_runtime_path=uniform_topk` rows. Keeping the canonical distribution selector aligned with the CSV contract is required; otherwise the predictor fails fast instead of training against mismatched routing data.

By default, downstream simulator smoke outputs are written under `outputs/examples/profiling-simulator`. For task-local validation artifacts, pass `METRICS_OUTPUT_DIR=<path>` or `--metrics-output-dir <path>` explicitly.

## Chunked Prefill Attention Recipe

`profile_attention_chunked_prefill.sh` demonstrates an attention profiling state that mirrors chunked prefill scheduling. The core knobs are:

```bash
FIXED_CHUNKED_PREFILL_SIZE=64 \
ENABLE_CHUNKED_PREFILL_GRID_SEARCH=true \
bash examples/profiling/profile_attention_chunked_prefill.sh
```

Use a positive `FIXED_CHUNKED_PREFILL_SIZE`; the script fails fast if the value is invalid.
