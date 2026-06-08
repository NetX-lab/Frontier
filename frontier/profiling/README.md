# Frontier Profiling Module

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-07 | Prioritized `examples/profiling/` wrappers in the usage guide and demoted disaggregated profiling examples to guarded historical notes. |
| 2026-06-07 | Pointed release-facing profiling examples to `examples/profiling/` while keeping `frontier/profiling/example/` as legacy/internal reference scripts. |
| 2026-06-06 | Updated profiling environment guidance to use environment_profiling.yml and avoid stale private environment names |
| 2026-04-27 | Clarify that PP receiver-head / prefill consumer-active handoff uses consumer-local `recv_end_ts`; producer `send_end_ts` belongs to the separate producer-send-path family. |
| 2026-03-29 | Record the first MoE uniform-topk data landing: `frontier/profiling/moe/main.py` now exposes `--routing_runtime_path`, MoE profiling rows carry explicit routing-path metadata, and the current canonical `qwen3-a3b-30b-moe/moe.csv` contains both standard and `uniform_topk` TP1/EP1/CUDA_EVENT routing rows. |
| 2026-03-29 | Add the MoE uniform-routing reminder: when runtime enables uniform routing, profiling/modeling must target the uniform-routing runtime path rather than reusing standard `fused_topk/topk_softmax` rows. |
| 2026-03-26 | Add `other_overhead/pp_receiver_head.csv` documentation for the PP8 baseline-only receiver-head family and record that broader single-GPU profiling plus CSV-miss generalization remains future work. |
| 2026-03-13 | Align profiling docs with measurement-aware contract; document `profile_method -> measurement_type` mapping and current defaults |
| 2026-02-16 | Restore CPU overhead section after accidental worktree deletion; add schema v2 columns, legacy defaults, and replay format notes |
| 2026-02-15 | Add CPU overhead profiling section with single-node policy, CSV contract, backend usage, and analytical TP modeling notes |

## Overview

The `frontier/profiling` module is the hardware profiling subsystem of the Frontier LLM inference simulator. It collects timing data for various LLM operations on real GPU hardware, which is then used to train ML-based execution time predictors for accurate simulation.

Release-facing one-click examples now live under `examples/profiling/`. Use those scripts first for turnkey workflows, including `linear_op`, Chunked Prefill attention, MoE, metadata smoke, and downstream simulator CSV smoke. The older `frontier/profiling/example/` scripts remain as legacy/internal references and are not deleted in this branch.

### Role in Frontier Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Frontier Simulator Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │   Profiling  │ ──► │   Training   │ ──► │ Execution Time Predictor │   │
│   │    Module    │     │    Module    │     │                          │   │
│   │              │     │              │     │  (sklearn models for     │   │
│   │ (This module)│     │ (frontier/   │     │   simulation latency     │   │
│   │              │     │  training/)  │     │   prediction)            │   │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘   │
│         │                    │                         │                    │
│         ▼                    ▼                         ▼                    │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │  CSV Files   │     │ Cached ML    │     │   Discrete-Event         │   │
│   │  (profiling  │     │ Models       │     │   Simulation             │   │
│   │   data)      │     │ (cache/)     │     │   (frontier/simulator.py)│   │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Single-GPU Profiling with Weight Sharding**: Profiling executes on a single GPU with TP-style weight sharding, measuring per-shard compute cost without requiring `torch.distributed`.

2. **EP as Distribution Parameter**: Expert Parallelism (EP) determines expert distribution across devices but doesn't change per-expert computation. Profiling uses `num_experts_per_device` for reusable data.

3. **Architecture-Based Organization**: Profiling is organized by model architecture component (attention, linear_op, moe) rather than deployment topology.

4. **Pluggable Backends**: Attention profiling supports multiple backends (FlashInfer, NoOp) for flexibility.

### Measurement Contract

Profiling-time `profile_method` and training-time `measurement_type` are related but not identical:

- `record_function` → `KERNEL_ONLY`
- `cuda_event` → `CUDA_EVENT`
- `kineto` / `perf_counter` are debug-only and must not be used as predictor-training inputs

Current profiling CLIs default to `record_function`, because decode CUDA graph modeling trains a dedicated kernel-only family. When generating eager-family training data, you must explicitly pass `--profile_method cuda_event`.

### MoE Uniform-Routing Reminder

If the target runtime enables MoE uniform routing, profiling and modeling must target the same uniform-routing runtime path.

- vLLM example: `VLLM_MOE_UNIFORM_ROUTING=1`
- Frontier example: a uniform-routing mode such as `uniform_legacy`
- For `moe_gating_routing_topk`, the target path becomes `uniform_topk`, not the standard `fused_topk -> topk_softmax` path

Do not reuse standard routing rows as a surrogate for a uniform-routing runtime. Until a runtime-equivalent uniform-routing profiling contract is materialized, any `routing_topk` gap must be treated as path-mismatch evidence first, not as proof that the standard profiling target is numerically too low.

The current MoE profiling CLI now exposes that contract directly:

- `python -m frontier.profiling.moe.main ... --routing_runtime_path standard_fused_topk`
- `python -m frontier.profiling.moe.main ... --routing_runtime_path uniform_topk`

The resulting `moe.csv` rows must carry explicit routing-path metadata rather than relying on unlabeled standard-path assumptions.

---

## Directory Structure

```
frontier/profiling/
├── README.md                    # This documentation file
├── __init__.py                  # Module initialization
│
├── attention/                   # Attention operation profiling
│   ├── __init__.py
│   ├── main.py                  # Entry point for attention profiling
│   ├── attention_wrapper.py     # AttentionWrapper class
│   ├── attention_input.py       # AttentionInput dataclass
│   ├── mixed_attention_input.py # MixedAttentionInput for varied-length batches
│   ├── sequence_metadata.py     # Sequence metadata structures
│   ├── sequence_proxy.py        # SequenceMetadataProxy for profiling
│   ├── backends/                # Attention backend implementations
│   │   ├── __init__.py          # Backend registry and factory
│   │   ├── base_attention_wrapper.py
│   │   ├── flashinfer_attention_wrapper.py
│   │   └── no_op_attention_wrapper.py
│   ├── README_MIXED_BATCH.md    # Mixed-batch profiling documentation
│   └── PROFILING_TEST_GUIDE.md  # Testing guide
│
├── linear_op/                   # Linear operation profiling (MLP, LayerNorm, etc.)
│   ├── __init__.py
│   ├── main.py                  # Entry point for linear op profiling
│   ├── linear_op_wrapper.py     # LinearOpWrapper class
│   ├── linear_op_impl.py        # GPTModel implementation for profiling
│   ├── README.md                # Detailed linear_op documentation
│   ├── PROFILING_NAN_FIX_REPORT.md
│   ├── PROFILING_QA.md
│   └── ROPE_LLAMA3_FIX.md
│
├── moe/                         # MoE (Mixture of Experts) profiling
│   ├── __init__.py
│   ├── main.py                  # Entry point for MoE profiling
│   ├── moe_wrapper.py           # MoEWrapper class
│   ├── moe_impl.py              # MoE component implementations
│   ├── moe_input.py             # MoELoadImbalanceInput dataclass
│   ├── moe_vllm_kernel.py       # vLLM fused kernel integration
│   ├── load_distribution.py     # Load distribution generation
│   ├── README.md                # Detailed MoE documentation
│   ├── SETUP.md                 # Setup instructions
│   ├── LOAD_IMBALANCE_GUIDE.md  # Load imbalance profiling guide
│   └── CHANGELOG_MULTI_GPU.md   # Multi-GPU support changelog
│
├── collectives/                 # NCCL collective operation profiling
│   ├── __init__.py
│   ├── main.py                  # Entry point for collectives profiling
│   ├── benchmark_runner.py      # Ray-based benchmark runner
│   ├── collectives_wrapper.py   # CollectivesWrapper class
│   ├── collectives_impl.py      # Collective implementations
│   └── collectives_input.py     # CollectivesInput dataclass
│
├── cpu_overhead/                # CPU overhead profiling
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── benchmark_runner.py      # Benchmark runner
│   ├── schema.py                # CSV contract definition
│   ├── validation.py            # Contract validation helpers
│   ├── planning.py              # Single-node TP planning helpers
│   ├── analytical.py            # Analytical TP extrapolation helpers
│   └── backends/                # Backend abstraction
│       ├── base_backend.py
│       ├── sarathi_backend.py
│       ├── vllm_backend.py
│       ├── vllm_mapping.py
│       └── factory.py
│
├── other_overhead/              # PP-specific overhead materialization
│   ├── __init__.py
│   ├── schema.py                # CSV contract definition
│   ├── validation.py            # Contract validation helpers
│   ├── materialize.py           # Raw trace -> canonical CSV materialization
│   └── backends/
│       └── vllm_pp_stage_boundary_mapping.py
│
├── common/                      # Shared utilities and configurations
│   ├── __init__.py
│   ├── model_config.py          # ModelConfig class (profiling-specific)
│   ├── parallel_config.py       # ParallelConfig dataclass
│   ├── cuda_timer.py            # CudaTimer context manager
│   ├── timer_stats_store.py     # TimerStatsStore singleton
│   ├── utils.py                 # Utility functions
│   ├── constants.py             # Constants
│   ├── layers/                  # Common layer implementations
│   │   ├── __init__.py
│   │   ├── activation.py        # Activation functions
│   │   ├── layernorm.py         # LayerNorm implementations
│   │   └── rotary_embedding.py  # RoPE implementations
│   ├── parallel_utils/          # Tensor parallelism utilities
│   │   ├── __init__.py
│   │   ├── parallel_state.py    # Simulated parallel state
│   │   ├── tensor_parallel_layers.py  # TP layer implementations
│   │   ├── tensor_parallel_mappings.py
│   │   └── tensor_parallel_utils.py
│
├── utils/                       # Profiling utilities
│   ├── __init__.py              # ProfileMethod enum, input generators
│   ├── record_function_tracer.py # PyTorch profiler integration
│   └── singleton.py             # Singleton metaclass
│
└── example/                     # Legacy/internal profiling scripts; release-facing wrappers live in examples/profiling/
    ├── README.md                # Legacy script documentation
    ├── test_profiling_attn.sh   # Attention profiling script
    ├── test_profiling_linear_op.sh  # Linear op profiling script
    ├── test_profiling_moe.sh    # MoE profiling script
    ├── test_pd_af_profiling.sh  # PD+AF disaggregation profiling
    └── migration_records/       # Migration documentation
        ├── DEPENDENCY_ANALYSIS_REPORT.md
        ├── MIGRATION_CHANGES.md
        ├── MIGRATION_FINAL_REPORT.md
        ├── MIGRATION_ROADMAP.md
        └── PHASE1_SUMMARY.md
```

---

## CPU Overhead Profiling

### Design policy (single-node first)

CPU overhead profiling follows the same decoupling principle used by Frontier compute profiling:

- Prefer **single-node** measurements (single GPU or multi-GPU in one node).
- Do not require multi-node distributed cluster hardware for baseline CPU overhead datasets.
- For TP degrees that cannot be directly measured on one node, use explicit analytical TP modeling only when enabled.

### CPU overhead CSV contract

Required columns:

- Identity:
  - `model_name`
  - `batch_size`
  - `tensor_parallel_degree`
  - `num_prefill_tokens`
  - `num_decode_tokens`
  - `scheduling_mode` (`sync` / `async`)
- CPU metrics:
  - `schedule_mean`, `schedule_median`
  - `sampler_e2e_mean`, `sampler_e2e_median`
  - `prepare_inputs_e2e_mean`, `prepare_inputs_e2e_median`
  - `process_model_outputs_mean`, `process_model_outputs_median`
  - `ray_comm_time_mean` (residual framework overhead)
- Metadata: `profiling_precision`

Optional provenance columns (if present):

- `cpu_overhead_source` (for example `measured`, `analytical_tp_scaling`)
- `analytical_tp_base_degree`
- `analytical_tp_scale_factor`

Legacy schema-v1 compatibility:

- Legacy CSV missing v2 identity fields is accepted with explicit warning.
- Injected defaults:
  - `num_prefill_tokens = 256`
  - `num_decode_tokens = batch_size * 3`
  - `scheduling_mode = "sync"`

`ray_comm_time_mean` semantics:

- Sarathi backend: residual Ray/framework overhead.
- vLLM replay backend: `step_wall_time - (schedule + prepare_inputs + sampler + process_model_outputs)`.

### Backend and TP policy flags

Main entrypoint:

```bash
python -m frontier.profiling.cpu_overhead.main -h
```

Important flags:

- `--backend {sarathi,vllm}`
- `--single_node_gpu_capacity <int>`: override auto-detected local measurable TP capacity
- `--enable_analytical_tp_modeling`: allow analytical TP extrapolation for missing TP degrees
- `--vllm_cpu_overhead_input_file <path>`: replay input for `--backend vllm` (`.json` / `.jsonl`)

### Reproducible example command

```bash
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m frontier.profiling.cpu_overhead.main \
  --models meta-llama/Llama-2-7b-hf \
  --num_tensor_parallel_workers 1 2 4 8 \
  --max_batch_size 32 \
  --precision FP16 \
  --backend sarathi \
  --single_node_gpu_capacity 4 \
  --enable_analytical_tp_modeling \
  --output_dir data/profiling
```

vLLM replay example:

```bash
python -m frontier.profiling.cpu_overhead.main \
  --backend vllm \
  --vllm_cpu_overhead_input_file tests/profiling/fixtures/cpu_overhead/vllm_replay_example.jsonl \
  --models meta-llama/Llama-2-7b-hf \
  --num_tensor_parallel_workers 2 4 \
  --precision FP16 \
  --max_batch_size 16 \
  --output_dir data/profiling
```

Replay mapper supports both input formats:

- Sarathi-style replay (`prepare_inputs_ms`, `process_model_outputs_ms`)
- vLLM v1 native replay (`preprocess_ms`, `postprocess_ms`, `bookkeep_ms`)

Semantics:

- TP 1/2/4 are measured directly.
- TP 8 is analytically generated from measured rows when enabled.
- If analytical modeling is not enabled and unmeasurable TP exists, profiling fails fast.

## PP Receiver-Head Overhead Materialization

This path is for PP-specific overhead evidence that should stay separate from the
generic `cpu_overhead/` contract.

### Current landed scope

- Canonical CSV:
  - `data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_receiver_head.csv`
- Current v1 runtime scope:
  - dense `llama3.1-8b`
  - `co-location`
  - `tp=1`
  - `pp=8`
  - decode-only consumer stages
- Current lookup policy:
  - exact-match only
  - warning once + return `0` on miss

### Current profiling source

- Raw source:
  - existing `vllm_pp_stage_boundary.jsonl`
- Materialization entrypoint:
  - `frontier.profiling.other_overhead.materialize.materialize_pp_receiver_head_csv(...)`
  - case-local pure-prefill variant:
    - `frontier.profiling.other_overhead.materialize.materialize_pp_prefill_consumer_active_csv(...)`
    - emits the receiver-head schema with `phase_label=prefill` and source `vllm_prefill_consumer_active_replay`

Current target:

- `pp_receiver_head_runtime_ms = consumer.forward_start_ts - consumer_ready_ts`
- `handoff_complete_ts = consumer.recv_end_ts`
- `consumer_ready_ts = max(handoff_complete_ts, previous_consumer_forward_end_ts_same_stage)`
- `producer.send_end_ts` is intentionally excluded from the receiver-head
  critical path because producer-side send tail is modeled by
  `pp_producer_send_path`.

This keeps `pp_stage_boundary_handoff_time` diagnostic-only while allowing the
narrower receiver-head family to participate in active runtime on the baseline
PP8 case.

### Important scope warning

The current landing is **baseline-only**. It should not be read as evidence that
the family is already validated for:

- `PP=2`
- `PP=4`
- `PP=16`
- MoE
- disaggregation
- mixed prefill/decode runtime landing

### Future important work (not implemented yet)

Two extensibility requirements remain intentionally out of scope for this round:

1. A profiling path similar to the existing profiling modules, so a single GPU can
   measure receiver-head family overheads for broader multi-stage / multi-machine
   semantics.
2. A simple non-sklearn miss-handling policy that returns an error-acceptable value
   when the exact CSV key is missing inside a supported PP regime.

Until those are implemented and validated, `pp_receiver_head.csv` should be treated
as a narrow authoritative baseline artifact, not a general PP-size profiling dataset.

---

## Supported Model Architectures

### Dense Models

Standard transformer models without MoE layers:

- **Llama Family**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-70b-hf`, `meta-llama/Meta-Llama-3-8B`, `meta-llama/Meta-Llama-3-70B`
- **Qwen Family**: `Qwen/Qwen-72B`, `Qwen/Qwen2.5-7B-Instruct`
- **Others**: `microsoft/phi-2`, `internlm/internlm-20b`, `codellama/CodeLlama-34b-Instruct-hf`

**Profiling modules**: `attention`, `linear_op`

### MoE Models

Mixture of Experts models with expert parallelism support:

- **Mixtral**: `mixtral_8x7b_moe` (8 experts, topk=2)
- **Qwen2-MoE**: `qwen2_moe_57b_a14b` (64 experts, topk=8)
- **Qwen3-MoE**: `Qwen3-30B-A3B` variants

**Profiling modules**: `attention`, `linear_op` (with `--is_moe` flag), `moe`

### Step2Mini Models

Custom architecture with shared experts and specialized attention:

- **Step2Mini-tiny**: Tiny variant for testing
- **Step2Mini**: Full model

**Special handling**:

- `model_arch` field set to `"step2_mini"`
- Requires `share_expert_dim` configuration
- Additional operations: `attn_inter_norm`, `attn_wq_proj`, `share_expert_*`

**Profiling modules**: `attention`, `linear_op`, `moe`

### Model Configuration

Model configurations are loaded from JSON files in `data/config/models/`:

```json
{
  "name": "Qwen3-30B-A3B-tiny",
  "num_layers": 4,
  "num_q_heads": 32,
  "num_kv_heads": 4,
  "embedding_dim": 2048,
  "mlp_hidden_dim": 8192,
  "is_moe": true,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "model_arch": null,
  "use_qk_norm": true,
  "dtype": "BF16"
}
```

For Step2Mini models:

```json
{
  "model_arch": "step2_mini",
  "share_expert_dim": 4096,
  "share_q_dim": 512,
  "head_dim": 128
}
```

---

## Usage Guide

### Complete Profiling Workflow

```
1. Profile Operations → 2. Train Models → 3. Run Simulation
```

#### Step 1: Profile Operations

Start with the release-facing wrappers under `examples/profiling/`. Use `--dry-run` to verify commands and paths without launching GPU kernels:

```bash
# Set environment
export CUDA_VISIBLE_DEVICES=0
cd /path/to/frontier-vllm-comparison

bash examples/profiling/profile_linear_op.sh --dry-run
bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
bash examples/profiling/profile_moe.sh --dry-run

# Metadata and downstream simulator-consumption smoke checks
bash examples/profiling/smoke_metadata.sh \
  --data_path data/profiling/compute/rtx_pro_6000/qwen2_dense_test
bash examples/profiling/smoke_simulator_dense_csv.sh
bash examples/profiling/smoke_simulator_moe_csv.sh
```

Advanced users can call the package entrypoints directly. Direct CLI usage follows the same taxonomy and should pass `--output_dir data/profiling`:

```bash
# Profile attention operations directly
python -m frontier.profiling.attention.main \
    --models meta-llama/Llama-2-7b-hf \
    --num_gpus 4 \
    --disable_ray \
    --profile_method cuda_event \
    --max_seq_len 4096 \
    --num_tensor_parallel_workers 1 2 4 \
    --output_dir data/profiling

# Profile linear operations directly
python -m frontier.profiling.linear_op.main \
    --models meta-llama/Llama-2-7b-hf \
    --num_gpus 4 \
    --disable_ray \
    --profile_method cuda_event \
    --max_tokens 4096 \
    --num_tensor_parallel_workers 1 2 4 \
    --output_dir data/profiling

# Profile MoE operations directly
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --device a100 \
    --num_gpus 4 \
    --disable_ray \
    --profile_method cuda_event \
    --max_tokens 1024 \
    --num_tensor_parallel_workers 1 2 \
    --expert_parallel_sizes 1 2 4 \
    --output_dir data/profiling
```

#### Step 2: Train Prediction Models

```bash
# Train attention models
python -m frontier.training.cli attention \
    --compute_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \
    --layer_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv \
    --output_dir cache

# Train linear op models
python -m frontier.training.cli linear_op \
    --dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \
    --output_dir cache

# Train MoE models
python -m frontier.training.cli moe \
    --dataset_path data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv \
    --output_dir cache
```

#### Step 3: Run Simulation

```bash
python -m frontier.main \
    --sys_arch co-location \
    --simulation_mode offline \
    --model_name meta-llama/Llama-2-7b-hf \
    --device a100
```

### Profiling for Different System Architectures

#### Co-location (Monolithic)

The current release-facing examples support co-location only. Use the top-level wrappers first:

```bash
# Dense model operator coverage
MODEL=qwen2_dense_test DEVICE=rtx_pro_6000 bash examples/profiling/profile_linear_op.sh --dry-run
MODEL=qwen2_dense_test DEVICE=rtx_pro_6000 bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run

# MoE model operator coverage
MODEL=Qwen3-30B-A3B-tiny DEVICE=rtx_pro_6000 bash examples/profiling/profile_linear_op.sh --dry-run
MODEL=Qwen3-30B-A3B-tiny DEVICE=rtx_pro_6000 bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
MODEL=Qwen3-30B-A3B-tiny DEVICE=rtx_pro_6000 bash examples/profiling/profile_moe.sh --dry-run
```

The legacy/internal scripts under `frontier/profiling/example/` remain available for compatibility and advanced workflows, but they are not the primary release entry point.

#### Historical / guarded architecture notes

The current release-facing examples support co-location only. Disaggregated runtime modes (`pd-disaggregation` and `pd-af-disaggregation`) are guarded out of this release, so their older profiling command snippets are historical/internal references rather than supported example workflows. If you need those paths for internal experimentation, start from `frontier/profiling/example/README.md` and verify the runtime guard behavior separately.

### Execution Modes

| Mode | Flag | Description |
|------|------|-------------|
| Single-GPU | `--num_gpus 1 --disable_ray` | Sequential execution on one GPU |
| Multi-GPU (ProcessPool) | `--num_gpus N --disable_ray` | Parallel execution using multiprocessing |
| Ray Mode | `--num_gpus N` (no `--disable_ray`) | Distributed execution via Ray actors |

**⚠️ Important**: Ray mode is currently broken due to grpcio 1.67.1 incompatibility. Always use `--disable_ray`.

---

## Output Format and Data Structure

### Directory Structure

```
data/profiling/
├── compute/                          # Compute operation profiling
│   └── {device}/                     # Device SKU (a100, h100, rtx_pro_6000, etc.)
│       └── {model_name}/             # Model name
│           ├── attention.csv         # Attention CUDA-event timing data
│           ├── attention_kernel_only.csv
│           ├── linear_op.csv         # Linear operation CUDA-event timing data
│           ├── linear_op_kernel_only.csv
│           ├── moe.csv               # MoE CUDA-event timing data
│           └── moe_kernel_only.csv
└── network/                          # Network/collective profiling
    └── {device}/
        ├── all_reduce.csv
        └── send_recv.csv
```

The canonical profiling CSV schema is
`data/profiling/compute/<device>/<model_name>/<op_name>.csv`. Pass
`--output_dir data/profiling`; the profiling entrypoints append
`compute/<device>/<model_name>/` internally.

### CSV Column Structure

#### Timing Statistics (per operation)

Each operation generates 5 timing columns:

| Column | Description |
|--------|-------------|
| `time_stats.{operation}.min` | Minimum time (ms) |
| `time_stats.{operation}.max` | Maximum time (ms) |
| `time_stats.{operation}.mean` | Mean time (ms) |
| `time_stats.{operation}.median` | Median time (ms) |
| `time_stats.{operation}.std` | Standard deviation (ms) |

#### Linear Op CSV Columns

```csv
time_stats.mlp_up_proj.mean,time_stats.mlp_down_proj.mean,time_stats.mlp_act.mean,
time_stats.input_layernorm.mean,time_stats.post_attention_layernorm.mean,
time_stats.attn_pre_proj.mean,time_stats.attn_post_proj.mean,time_stats.attn_rope.mean,
time_stats.add.mean,...,
num_tokens,num_tensor_parallel_workers,n_head,n_kv_head,n_embd,n_expanded_embd,
vocab_size,use_gated_mlp,model_arch,profiling_precision
```

#### Attention CSV Columns

```csv
time_stats.attn_prefill.mean,time_stats.attn_decode.mean,time_stats.attn_kv_cache_save.mean,...,
batch_size,prefill_chunk_size,kv_cache_size,is_prefill,
num_tensor_parallel_workers,n_embd,n_q_head,n_kv_head,block_size,max_model_len,
attention_backend,is_mixed_batch,mode,total_tokens,max_seq_len,min_seq_len,
avg_seq_len,seq_len_variance,profiling_precision
```

#### MoE CSV Columns

```csv
time_stats.moe_gating_linear.mean,time_stats.moe_gating_routing_topk.mean,
time_stats.moe_shuffling.mean,time_stats.moe_grouped_gemm.mean,...,
num_tokens,num_experts,num_experts_per_device,expert_parallel_size,
routing_runtime_path,routing_assignment_policy,routing_weight_policy,
routing_uses_router_logits,gating_runtime_context,gating_runtime_context_impl,
router_topk,hidden_dim,expert_hidden_dim,use_gated,num_tensor_parallel_workers,
load_distribution,load_imbalance_cv,load_gini_coefficient,measurement_type,
profiling_precision,model_arch
```

MoE profiling writes one consolidated CSV per measurement type:
`moe.csv` for CUDA-event timings and `moe_kernel_only.csv` for kernel-only
timings. Both files keep the split gating columns above so training can model
`moe_gating_linear` and `moe_gating_routing_topk` separately.

### Metadata Columns

| Column | Description |
|--------|-------------|
| `num_tokens` | Number of input tokens |
| `num_tensor_parallel_workers` | Tensor parallelism size |
| `expert_parallel_size` | Expert parallelism size (MoE only) |
| `num_experts_per_device` | Experts per device (MoE only) |
| `profiling_precision` | Data type (FP16, BF16, FP32) |
| `model_arch` | Model architecture identifier |
| `attention_backend` | Attention backend used |

---

## Environment Requirements

### Python Environment

```bash
# Recommended: create the dedicated profiling environment.
conda env create -f environment_profiling.yml
conda activate frontier-profiling

# Or use an existing environment that already provides vLLM and FlashInfer.
pip install -e .
```

### Required Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Core deep learning framework |
| `pandas` | ≥1.5 | Data manipulation |
| `numpy` | ≥1.24 | Numerical operations |
| `tqdm` | ≥4.65 | Progress bars |
| `pyyaml` | ≥6.0 | Configuration files |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flashinfer` | ≥0.1 | FlashInfer attention backend |
| `ray` | ≥2.5 | Distributed profiling (currently broken) |
| `vllm` | ≥0.10 | Fused MoE kernels for accurate profiling |

### GPU Requirements

| Profiling Type | Minimum VRAM | Recommended |
|----------------|--------------|-------------|
| Linear Op | 16 GB | 40 GB |
| Attention | 24 GB | 80 GB |
| MoE | 24 GB | 80 GB |

### CUDA Compatibility

- CUDA 11.8+ required
- CUDA 12.x recommended for best performance
- cuDNN 8.6+ required

---

## Troubleshooting

### Common Issues

#### Issue 1: Ray Mode Crashes

**Error**: `UnknownError: UNKNOWN:ipv4:127.0.0.1:xxxxx: Trying to connect an http1.x server`

**Solution**: Always use `--disable_ray` flag.

```bash
python -m frontier.profiling.moe.main --disable_ray ...
```

#### Issue 2: FlashInfer Not Found

**Error**: `ModuleNotFoundError: No module named 'flashinfer'`

**Solution**: Install FlashInfer or use NO_OP backend:

```bash
pip install flashinfer

# Or use NO_OP backend
python -m frontier.profiling.attention.main --attention_backend NO_OP ...
```

#### Issue 3: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. Reduce `--max_tokens` or `--max_seq_len`
2. Use fewer TP sizes
3. Profile on a GPU with more VRAM
4. Set `CUDA_VISIBLE_DEVICES` to a specific GPU

```bash
export CUDA_VISIBLE_DEVICES=7  # Use GPU with most free memory
python -m frontier.profiling.moe.main --max_tokens 256 ...
```

#### Issue 4: NaN Values in Profiling Data

**Error**: CSV contains NaN timing values

**Solutions**:

1. Ensure model weights are properly initialized
2. Check for numerical overflow with large token counts
3. Use `--precision BF16` for better numerical stability

See `linear_op/PROFILING_NAN_FIX_REPORT.md` for detailed analysis.

#### Issue 5: Missing Operations in Output

**Error**: Expected operations not in CSV

**Solutions**:

1. Verify model configuration matches expected architecture
2. Check `--is_moe` flag for MoE models
3. Ensure `model_arch` is correctly set in JSON config

### Performance Optimization

1. **Start Small**: Use `--max_tokens 256` for initial testing
2. **Batch GPU Usage**: Profile multiple configurations in one run
3. **Use the correct family explicitly**: default `--profile_method record_function` yields `KERNEL_ONLY`; pass `--profile_method cuda_event` when collecting eager / `CUDA_EVENT` data
4. **Parallel Profiling**: Use `--num_gpus 4 --disable_ray` for faster execution

### Debug Tips

1. **Enable Verbose Logging**:

   ```bash
   PYTHONPATH=. python -m frontier.profiling.moe.main --verbose ...
   ```

2. **Check GPU Memory**:

   ```bash
   nvidia-smi -l 1  # Monitor GPU memory during profiling
   ```

3. **Validate Output**:

   ```python
   import pandas as pd
   df = pd.read_csv("moe.csv")
   print(df.describe())  # Check for NaN, outliers
   print(df.groupby(['num_tensor_parallel_workers', 'expert_parallel_size']).size())
   ```

---

## Integration with Other Modules

### Training Module (`frontier/training/`)

Profiling data is consumed by trainers:

```python
from frontier.training.cli import train_moe, train_linear_op, train_attention

# Train models from profiling data
train_moe(dataset_path="data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv")
train_linear_op(dataset_path="data/profiling/compute/a100/llama2-7b/linear_op.csv")
train_attention(
    compute_dataset_path="data/profiling/compute/a100/llama2-7b/linear_op.csv",
    layer_dataset_path="data/profiling/compute/a100/llama2-7b/attention.csv"
)
```

### Execution Time Predictor (`frontier/execution_time_predictor/`)

Trained models are loaded by predictors:

```python
from frontier.execution_time_predictor import SklearnMoEExecutionTimePredictor

predictor = SklearnMoEExecutionTimePredictor(
    model_config=model_config,
    replica_config=replica_config,
)

# Predict execution time
exec_time = predictor.get_execution_time(batch, replica_id=0)
```

### CC Backend (`frontier/cc_backend/`)

Network profiling data is used by VidurCCBackend:

```python
from frontier.cc_backend import CCBackendFactory

backend = CCBackendFactory.create(
    backend_type="vidur",
    profiling_path="data/profiling/network/a100"
)

# Predict collective communication time
allreduce_time = backend.predict_allreduce(data_size=1024*1024, num_devices=8)
```

---

## References

- [MoE Profiling README](moe/README.md) - Detailed MoE profiling documentation
- [Linear Op README](linear_op/README.md) - Linear operation profiling details
- [Example Scripts README](example/README.md) - Script usage documentation
- [AGENTS.md](../../AGENTS.md) - Project-wide development guidelines
