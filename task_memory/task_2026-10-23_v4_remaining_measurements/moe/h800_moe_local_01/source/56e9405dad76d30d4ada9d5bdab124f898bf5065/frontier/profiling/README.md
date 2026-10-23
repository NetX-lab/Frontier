# Frontier Profiling Module

## Overview

The `frontier/profiling` module is the hardware profiling subsystem of the Frontier LLM inference simulator. It measures LLM operators, host (CPU) overheads, kernel gaps and GPU memory on real hardware. The simulator trains its execution-time predictors on these measurements.

Release-facing one-click examples live under `examples/profiling/`. Use those scripts first for turnkey workflows, including `linear_op`, Chunked Prefill attention, MoE, metadata smoke, and downstream simulator CSV smoke. The older `frontier/profiling/example/` scripts remain as legacy/internal references; see `frontier/profiling/example/README.md` for why they do not run as-is.

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

4. **Pluggable Backends**: Attention profiling supports the `FLASHINFER` (default), `FLASHINFER_MLA`, `VLLM_ROCM` and `NO_OP` backends (`--attention_backend`).

### Measurement Contract

Profiling-time `profile_method` and training-time `measurement_type` are related but not identical:

| `--profile_method` | `measurement_type` | Output file | What it times |
| --- | --- | --- | --- |
| `cuda_event` (alias `cuda`) | `CUDA_EVENT` | `<op>.csv`, for example `linear_op.csv` | Wall-clock device time between CUDA events around each operator |
| `record_function` (alias `kernel_only`) | `KERNEL_ONLY` | `<op>_kernel_only.csv` | Sum of the operator's kernel durations from a profiler trace; also records `time_stats.<op>.kernel_count` |
| `device_event` | `DEVICE_EVENT` | `<op>_device_event.csv` | ROCm device events (the ROCm counterpart of `cuda_event`) |

- `kineto` / `perf_counter` are debug-only and must not be used as predictor-training inputs.
- The platform check rejects `cuda_event` on ROCm and `device_event` on CUDA (`validate_profile_method_platform` in `utils/__init__.py`).

The lower-level profiling CLIs default to `record_function`, which writes
`KERNEL_ONLY` rows. The release wrappers under `examples/profiling/` override
that default with `cuda_event` and write simulator-ready `CUDA_EVENT` rows.
Choose the method explicitly when invoking a lower-level CLI.

The simulator prices a step from the event tables (`CUDA_EVENT`, or
`DEVICE_EVENT` on ROCm) by default. It uses the kernel-only tables for decode
steps that replay a CUDA graph, for the PD-AF decode roles, and for the device
stream of an eager step under two-stream pricing (see
[From vLLM CPU-probe logs](#from-vllm-cpu-probe-logs)). A kernel-only step also
pays the per-kernel gap from `kernel_gap.csv` when that file exists (see
[Kernel Gap Profiling](#kernel-gap-profiling)).

### MoE Uniform-Routing Reminder

If the target runtime enables MoE uniform routing, profiling and modeling must target the same uniform-routing runtime path.

- vLLM example: `VLLM_MOE_UNIFORM_ROUTING=1`
- Frontier runtime: `moe_routing_distribution_type` accepts `balanced`,
  `random`, `skewed`, or `zipf`. The checked-in uniform-top-k MoE smoke
  uses `random` because its CSV rows are labeled `uniform_topk`.
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
├── migrate_csv_metadata.py      # Adds the required metadata columns to legacy profiling CSVs
│
├── attention/                   # Attention operation profiling
│   ├── __init__.py
│   ├── main.py                  # Entry point for attention profiling
│   ├── attention_wrapper.py     # AttentionWrapper class
│   ├── attention_input.py       # AttentionInput dataclass
│   ├── mixed_attention_input.py # MixedAttentionInput for varied-length batches
│   ├── true_mixed_batch_input.py # Inputs for true mixed prefill+decode batches
│   ├── metadata_utils.py        # Metadata post-processing for attention rows
│   ├── vllm_mla_profile_importer.py # Imports measured vLLM MLA rows into the attention schema
│   ├── sequence_metadata.py     # Sequence metadata structures
│   ├── sequence_proxy.py        # SequenceMetadataProxy for profiling
│   ├── backends/                # Attention backend implementations
│   │   ├── __init__.py          # AttentionBackend enum and factory
│   │   ├── base_attention_wrapper.py
│   │   ├── flashinfer_attention_wrapper.py
│   │   ├── vllm_rocm_attention_wrapper.py
│   │   └── no_op_attention_wrapper.py
│   ├── README_MIXED_BATCH.md    # Mixed-batch profiling documentation
│   └── PROFILING_TEST_GUIDE.md  # Testing guide
│
├── gdn/                         # Qwen3.5 gated-delta-network (GDN) layer profiling with vLLM
│   ├── README.md
│   ├── main.py                  # Entry point; writes gdn.csv
│   ├── inputs.py                # GDN workload descriptions and schema constants
│   └── vllm_wrapper.py          # vLLM GDN layer wrapper
│
├── kernel_gap/                  # Device idle time between kernels (kernel_gap.csv)
│   ├── main.py                  # Entry point
│   └── schema.py                # Columns shared with the simulator's reader
│
├── linear_op/                   # Linear operation profiling (MLP, LayerNorm, etc.)
│   ├── __init__.py
│   ├── main.py                  # Entry point for linear op profiling
│   ├── linear_op_wrapper.py     # LinearOpWrapper class
│   ├── linear_op_impl.py        # GPTModel implementation for profiling
│   ├── profiling_plan.py        # Separate attention / FFN / MoE TP profiling plan
│   ├── ray_setup_hook.py        # Ray worker setup hook
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
│   ├── vllm_cpu_probe.py        # CSVs from vLLM CPU-probe step logs
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
├── non_kv_cache_overhead/       # Non-KV-cache GPU memory profiling (see its README.md)
│
├── experimental/                # Experimental integrations, not used by the release profilers
│   └── sglang/                  # Selected SGLang/AITER primitives and trace importers
│
├── common/                      # Shared utilities and configurations
│   ├── __init__.py
│   ├── model_config.py          # ModelConfig class (profiling-specific)
│   ├── parallel_config.py       # ParallelConfig dataclass
│   ├── accelerator.py           # CUDA / ROCm accelerator discovery
│   ├── cuda_timer.py            # CudaTimer context manager
│   ├── device_timer.py          # Device-event timing for CUDA and ROCm
│   ├── timer_stats_store.py     # TimerStatsStore singleton
│   ├── typed_contracts.py       # Imports of the typed operator contract schema
│   ├── vllm_compat.py           # Helpers for standalone use of vLLM custom ops
│   ├── utils.py                 # Utility functions
│   ├── constants.py             # Constants
│   ├── layers/                  # Common layer implementations
│   │   ├── __init__.py
│   │   ├── activation.py        # Activation functions
│   │   ├── layernorm.py         # LayerNorm implementations
│   │   └── rotary_embedding.py  # RoPE implementations
│   └── parallel_utils/          # Tensor parallelism utilities
│       ├── __init__.py
│       ├── parallel_state.py    # Simulated parallel state
│       ├── tensor_parallel_layers.py  # TP layer implementations
│       ├── tensor_parallel_mappings.py
│       └── tensor_parallel_utils.py
│
├── utils/                       # Profiling utilities
│   ├── __init__.py              # ProfileMethod enum, output paths, input generators
│   ├── confirmation.py          # Interactive confirmation prompt (`--yes` skips it)
│   ├── record_function_tracer.py # PyTorch profiler integration
│   ├── replicated_ops.py        # Splits results into sharded and replicated rows
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

Optional columns (if present):

- Provenance: `cpu_overhead_source` (for example `measured`, `analytical_tp_scaling`),
  `analytical_tp_base_degree`, `analytical_tp_scale_factor`
- Eager-forward terms written by the CPU-probe path: `forward_launch_mean` / `forward_launch_median`
  and `forward_drain_mean` / `forward_drain_median`
- Row keys written by the CPU-probe path: `pipeline_stage_id` (PP>1 tables) and `engine_idle_ms`
  (tables built with `--engine_idle_edges_ms`)

The simulator reads the eager table
`data/profiling/cpu_overhead/<network_device>/<model>/cpu_overheads.csv` and the kernel-only table
`cpu_overheads_kernel_only.csv` beside it. CPU-overhead modeling is off by default
(`skip_cpu_overhead_modeling=True`); pass
`--no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling` to enable it.

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

- `--models <name> ...`, `--num_tensor_parallel_workers <tp> ...`
- `--max_batch_size <int>` (default `128`), or `--batch_sizes <int> ...` to list the batch sizes explicitly
- `--precision` (alias `--dtype`, default `FP16`)
- `--backend {sarathi,vllm}` (default `sarathi`)
- `--single_node_gpu_capacity <int>`: override auto-detected local measurable TP capacity
- `--enable_analytical_tp_modeling`: allow analytical TP extrapolation for missing TP degrees
- `--vllm_cpu_overhead_input_file <path>`: replay input for `--backend vllm` (`.json` / `.jsonl`)
- `--output_dir <dir>` (default `profiling_outputs`)

The CSV is written to `<output_dir>/cpu_overhead/<YYYY-MM-DD_HH-MM-SS>/<model>/cpu_overhead.csv`.
This is not the path the simulator reads: copy the file to
`data/profiling/cpu_overhead/<network_device>/<model>/cpu_overheads.csv`, or pass its path with
`--random_forrest_execution_time_predictor_config_cpu_overhead_input_file`.

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

With `--single_node_gpu_capacity 4`:

- TP 1/2/4 are measured directly.
- TP 8 is analytically generated from measured rows when enabled.
- If analytical modeling is not enabled and unmeasurable TP exists, profiling fails fast.

vLLM replay example (`vllm_cpu_overhead_records.jsonl` is your own record file; the repository
does not ship one):

```bash
python -m frontier.profiling.cpu_overhead.main \
  --backend vllm \
  --vllm_cpu_overhead_input_file path/to/vllm_cpu_overhead_records.jsonl \
  --models meta-llama/Llama-2-7b-hf \
  --num_tensor_parallel_workers 2 4 \
  --precision FP16 \
  --max_batch_size 16 \
  --output_dir data/profiling
```

Each record carries `model_name`, `batch_size`, `tensor_parallel_degree`, `profiling_precision`,
`schedule_ms`, `sampler_ms` and `step_wall_time_ms`, plus the fields of one of two input formats
(`cpu_overhead/backends/vllm_mapping.py`):

- Sarathi-style replay (`prepare_inputs_ms`, `process_model_outputs_ms`)
- vLLM v1 native replay (`preprocess_ms`, `postprocess_ms`, `bookkeep_ms`)

### From vLLM CPU-probe logs

`vllm_cpu_probe.py` builds the CPU-overhead CSVs from the logs that Frontier's instrumented
vLLM writes when `VLLM_FRONTIER_CPU_PROBE_LOG_PATH` is set. The engine log
`cpu_probe[_dp<d>].jsonl` holds one record per engine step with host timestamps of the
schedule and the output update, and for each scheduled request its tokens, its computed
tokens before the step and its prompt length, with the engine's KV connector role `kv_role`.
Each pipeline stage writes `cpu_probe[_dp<d>]_pp<p>.jsonl`, `_pp0` for a single-stage engine,
with one record per forward: the runner's timestamps, the start and end of the DP token-count
all-reduce and of the KV connector's `start_load_kv`, the CUDA graph mode and the CUDA-event
device time of the forward, timed from the forward-context entry. Logs of an older probe,
without the request-phase or KV-load fields, are rejected.

Frontier prices each stage's forward from its operator tables, and prices itself the waits
between stages, between DP engines and for a PD decode instance's KV load. The terms price
every other host interval of a step, each once, on the stage that runs it:

| Term | Probe interval |
| --- | --- |
| `schedule` | stage 0: `step_start -> schedule_end` |
| `prepare_inputs_e2e` | each stage: dispatch start `-> forward_context_entered`, less the DP wait and the KV-load wait |
| `forward_launch` | each stage, eager forwards only: launch start `-> forward_end` |
| `forward_drain` | each stage, eager forwards only: `forward_end ->` device end, at least `0` |
| `sampler_e2e` | last stage: `max(forward_end, device end) -> sample_end` |
| `process_model_outputs` | last stage: `sample_end ->` next `step_start` when the next step shares a request and starts after `update_end`, else `sample_end -> update_end` |
| `ray_comm_time_mean` | `0` (no Ray hop) |

- Stage 0's dispatch starts at the later of `schedule_end` and the end of its previous
  forward, which leaves out queueing behind the previous batch; a later stage's dispatch
  starts at its `execute_start`, after its receive wait.
- The DP wait is the all-reduce's duration beyond its own latency: the median duration on
  the DP engine that reaches an all-reduce last, over the all-reduces recorded by every DP
  engine of the serving instance.
- The KV-load wait is the KV connector's `start_load_kv`, which runs inside the
  forward-context entry; an engine without a connector waits `0`.
- The launch start is the later of `forward_context_entered` and the upstream stage's device
  end; a stage's device end is its launch start plus its forward's device time.
- Frontier ends an eager forward no earlier than `forward_launch + forward_drain`, the
  in-situ time to the device end; when the predicted device time is longer, the forward
  takes the predicted device time.

Forwards are grouped per stage by `(batch_size, num_prefill_tokens, num_decode_tokens)` and
keyed by `pipeline_stage_id`. A request's tokens are prefill when they recompute tokens an
earlier step computed (a resumed preemption victim) or continue its prompt, and decode
otherwise; on a PD decode instance (`kv_consumer`) a request's first step, which computes the
prompt tokens the connector did not load, is its first decode step, as in Frontier's DECODE
role. A forward that replayed a FULL CUDA graph goes to the
kernel-only file and publishes no `forward_launch` or `forward_drain`; every other forward
goes to the eager file. Each `--cpu_probe_logs` lists the engine logs of one serving
instance, one per DP engine in DP rank order; a PP>1 engine adds `--num_pipeline_stages`.

The DP engines of a MoE instance meet in every layer's expert-parallel all-to-all, so a
forward that runs beside another engine's prompt lasts until that engine's device work ends.
Frontier prices this wait between DP engines itself, from the lanes it simulates, so a forward
whose device interval (launch start to device end) overlaps a forward of another DP engine at
the same pipeline stage that holds prefill tokens is left out of the rows.

With `--engine_idle_edges_ms 0 100 500`, rows are also keyed by `engine_idle_ms`, the
largest edge at or below the engine's idle time before the step: from the latest
`update_end` of its earlier steps, or the end of its latest DP dummy forward, to its
`step_start`. A DP>1 instance reads its dummy forwards from the `dummy_pass` records of
`dp_placement/*.jsonl` beside its engine logs. An engine's first step that schedules tokens
carries the engine's one-time start costs, which a ground-truth run's warmups absorb, so it
is left out of the rows; its terms are still measured, so the step after it keeps its own
dispatch queueing. Without `--engine_idle_edges_ms`, every step is kept. Frontier keys each
batch the same way, from the end of its engine's latest loop iteration or dummy forward to
the batch's schedule time.

```bash
python -m frontier.profiling.cpu_overhead.vllm_cpu_probe \
  --cpu_probe_logs run/prefill/cpu_probe.jsonl \
  --cpu_probe_logs run/decode/cpu_probe.jsonl \
  --model_name llama2_7b_dense_example --tensor_parallel_degree 1 \
  --profiling_precision FP16 --scheduling_mode sync \
  --eager_output_file cpu_overheads.csv \
  --kernel_only_output_file cpu_overheads_kernel_only.csv
```

When the eager table carries `forward_launch` and CPU-overhead modeling is on, the simulator
prices an eager step as the slower of two streams: the device stream from the kernel-only
operator tables and the host stream from `forward_launch`
(`uses_two_stream_eager_pricing` in `frontier/execution_time_predictor/measurement_input_paths.py`).
Such a table must have `pipeline_stage_id` rows for every pipeline stage of the replica's model and
attention TP degree (stage `0` alone for a single-stage replica). PD-AF rejects a table with
`forward_launch`.

## Kernel Gap Profiling

A kernel-only operator table times each operator's kernels and nothing between them.
`frontier.profiling.kernel_gap.main` measures the mean device idle time before a kernel for two
execution modes: `eager` (one forward of the linear-op profiler's decoder block per token count in
`--num_tokens`) and `cuda_graph` (the decode CUDA graphs of a vLLM engine built from
`--vllm_engine_args`). It needs a GPU, `torch` and `vllm`.

```bash
python -m frontier.profiling.kernel_gap.main --device h800 --output_dir data/profiling \
    --model Qwen3-235B-A22B --num_tensor_parallel_workers 8 --num_tokens 16 2048 \
    --vllm_engine_args engine_args.json --vllm_attention_backend FLASHINFER
```

- `engine_args.json` holds the deployment's `vllm.LLM` keyword arguments other than the model, for
  example `{"load_format": "dummy", "skip_tokenizer_init": true, "tensor_parallel_size": 8,
  "compilation_config": {"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}, "max_num_seqs": 64}`.
- `--num_tensor_parallel_workers` must equal the engine's `tensor_parallel_size`.
- Other flags: `--repeats` (default `20`), `--host_ahead_ms` (default `50.0`) and
  `--vllm_prompt_tokens` (default `16`).
- Output: `<output_dir>/compute/<device>/<model>/kernel_gap.csv` with columns `execution_mode`,
  `num_tensor_parallel_workers` and `kernel_gap_us`, plus provenance columns. One file may hold
  rows of several TP sizes.

The simulator reads the rows whose `num_tensor_parallel_workers` equals the replica's attention TP
size and needs exactly one `eager` and one `cuda_graph` row there
(`frontier/execution_time_predictor/kernel_gap.py`). It ignores the file in dummy mode. When the
file exists, a PD-AF run, a model with GDN layers, or a run with speculative decoding fails with
an error; remove the file to run those cases without the kernel gap.

## PP Receiver-Head Overhead Materialization

This path is for PP-specific overhead evidence that should stay separate from the
generic `cpu_overhead/` contract.

### Current landed scope

- Canonical CSV (no file is checked in; the simulator logs a warning and prices `0` when it is absent):
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

### Model names

`--models` (and the simulator's `--replica_config_model_name`) accepts two kinds of names
(`BaseModelConfig.create_from_name` in `frontier/config/model_config.py`):

- Names registered as model config classes in `frontier/config/model_config.py`, for example
  `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-70b-hf`, `meta-llama/Meta-Llama-3-8B`,
  `meta-llama/Meta-Llama-3-70B`, `Qwen/Qwen-72B`, `Qwen/Qwen3-32B`, `microsoft/phi-2`,
  `internlm/internlm-20b` and `codellama/CodeLlama-34b-Instruct-hf`.
- Any other name, loaded from the HuggingFace `config.json` copy at
  `data/config/models/<name>.json`, with `/` in the name replaced by `__`. Checked-in examples
  include `llama2_7b_dense_example`, `qwen2_dense_test`, `mixtral_8x7b_moe` (8 experts, top-2),
  `qwen2_moe_example` (60 experts, top-4), `Qwen3-30B-A3B-tiny` (16 experts, top-8),
  `Qwen3-235B-A22B` (128 experts, top-8), `Step2Mini-tiny` and
  `Qwen3.8-2.4T-A95B-Quark-MXFP4` (a GDN model).

### Profiling modules per model type

| Model type | Profiling modules |
| --- | --- |
| Dense | `attention`, `linear_op` |
| MoE | `attention`, `linear_op` (with `--is_moe`, which skips the dense `mlp_up_proj`, `mlp_down_proj` and `mlp_act` operators), `moe` |
| GDN layers (Qwen3.5, ROCm) | `gdn` (see `gdn/README.md`), plus the modules above for the other layers |

### Architecture profiles

The model architecture profile (`frontier/model_architectures.py`) selects each operator's owning
family and TP domain. The profiles are `generic`, `step2_mini`, `step3_text` and `qwen3_5_moe`.
Profilers write the selected profile into the `model_architecture_profile` column.

Step2Mini (`model_arch: "step2_mini"`) has shared experts and extra attention operators:

- Requires `share_expert_dim` in the model JSON
- Additional operations: `attn_inter_norm`, `attn_wq_proj`, `share_expert_*`
- Profiling modules: `attention`, `linear_op`, `moe`

### Model configuration files

A file under `data/config/models/` is a HuggingFace `config.json` (fields such as `hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`,
`num_experts`, `num_experts_per_tok` and `torch_dtype`). The profiling `ModelConfig` also reads a
few Frontier-specific fields when present, for example for Step2Mini:

```json
{
  "model_type": "step2_mini",
  "model_arch": "step2_mini",
  "share_expert_dim": 2048,
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
# Run from the repository root
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD

bash examples/profiling/profile_linear_op.sh --dry-run
bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
bash examples/profiling/profile_moe.sh --dry-run

# Metadata and downstream simulator-consumption smoke checks
bash examples/profiling/smoke_metadata.sh \
  --data_path data/profiling/compute/rtx_pro_6000/qwen2_dense_test
bash examples/profiling/smoke_simulator_dense_csv.sh
bash examples/profiling/smoke_simulator_moe_csv.sh
```

Advanced users can call the package entrypoints directly. Direct CLI usage follows the same taxonomy: pass `--output_dir data/profiling` and the hardware label with `--device` (the attention and linear-op CLIs default to `unknown`; the MoE CLI requires it). Each run asks for confirmation before it starts; `--yes` skips the prompt.

For standard attention, `max_seq_len` bounds automatically generated profiling
axes. `max_model_len` is the runtime context boundary: an explicit decode KV
value may exceed `max_seq_len` when it remains at most
`max_model_len - 1`. The attention wrapper still enforces the available
physical KV-block capacity during measurement.

```bash
# Profile attention operations directly
python -m frontier.profiling.attention.main \
    --models meta-llama/Llama-2-7b-hf \
    --device h200 \
    --num_gpus 4 \
    --disable_ray \
    --profile_method cuda_event \
    --max_seq_len 4096 \
    --max_model_len 4096 \
    --num_tensor_parallel_workers 1 2 4 \
    --output_dir data/profiling

# Profile linear operations directly
python -m frontier.profiling.linear_op.main \
    --models meta-llama/Llama-2-7b-hf \
    --device h200 \
    --num_gpus 4 \
    --disable_ray \
    --profile_method cuda_event \
    --max_tokens 4096 \
    --num_tensor_parallel_workers 1 2 4 \
    --output_dir data/profiling

# Profile MoE operations directly
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --device h200 \
    --num_gpus 4 \
    --disable_ray \
    --profile_method cuda_event \
    --max_tokens 1024 \
    --num_tensor_parallel_workers 1 2 \
    --expert_parallel_sizes 1 2 4 \
    --output_dir data/profiling
```

#### Step 2: Train Prediction Models (optional)

The simulator trains any missing predictor from the configured CSVs on its first run and caches it
(see Step 3). Run the training CLI separately to pre-warm the cache or to debug a CSV. The examples
below use checked-in H200 tables; `--measurement_type` must match the table's `measurement_type`
column.

```bash
# Train linear op models
python -m frontier.training.cli linear_op \
    --dataset_path data/profiling/compute/h200/llama2_7b_dense_example/linear_op.csv \
    --model_name llama2_7b_dense_example \
    --device h200 \
    --measurement_type CUDA_EVENT \
    --output_dir cache

# Train attention models
python -m frontier.training.cli attention \
    --layer_dataset_path data/profiling/compute/h200/llama2_7b_dense_example/attention.csv \
    --compute_dataset_path data/profiling/compute/h200/llama2_7b_dense_example/linear_op.csv \
    --model_name llama2_7b_dense_example \
    --device h200 \
    --measurement_type CUDA_EVENT \
    --output_dir cache

# Train MoE models
python -m frontier.training.cli moe \
    --dataset_path data/profiling/compute/h200/Qwen3-30B-A3B-tiny/moe.csv \
    --model_name Qwen3-30B-A3B-tiny \
    --device h200 \
    --measurement_type CUDA_EVENT \
    --output_dir cache
```

See `docs/training/README.md` for the full training guide.

#### Step 3: Run Simulation

With no `*_input_file` flags, the simulator reads the default paths built from
`--replica_config_device`, `--replica_config_network_device` and `--replica_config_model_name`
(see [Output Directory Structure](#output-directory-structure)). This command reads the checked-in H200 tables
for `llama2_7b_dense_example`, including the CPU-overhead tables and `kernel_gap.csv`:

```bash
python -m frontier.main \
    --sys_arch co-location \
    --simulation_mode offline \
    --replica_config_model_name llama2_7b_dense_example \
    --replica_config_device h200 \
    --replica_config_network_device h200_dgx \
    --cc_backend_config_type analytical \
    --replica_scheduler_config_type vllm_v1 \
    --request_generator_config_type synthetic \
    --synthetic_request_generator_config_num_requests 4 \
    --length_generator_config_type fixed \
    --fixed_request_length_generator_config_prefill_tokens 128 \
    --fixed_request_length_generator_config_decode_tokens 8 \
    --decode_cuda_graph_mode none \
    --no-random_forrest_execution_time_predictor_config_enable_dummy_mode \
    --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 1024 \
    --no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling
```

The first run trains every predictor it needs and stores it under `--metrics_config_cache_dir`
(default `cache`); on a CPU-only host this can take tens of minutes. Later runs reuse the cache.
To point at other tables, pass their paths with the
`--random_forrest_execution_time_predictor_config_<table>_input_file` flags, for example
`..._linear_op_input_file`, `..._atten_input_file`, `..._moe_input_file`,
`..._cpu_overhead_input_file` and `..._kernel_gap_input_file`.

### Profiling for Different System Architectures

#### Co-location (Monolithic)

Operator profiling does not depend on the serving architecture. Each cluster role reads the tables
of its own device and model and selects the rows of its own TP / EP sizes. Use the top-level
wrappers first:

```bash
# Dense model operator coverage
MODEL=qwen2_dense_test DEVICE=rtx_pro_6000 bash examples/profiling/profile_linear_op.sh --dry-run
MODEL=qwen2_dense_test DEVICE=rtx_pro_6000 bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run

# MoE model operator coverage
MODEL=Qwen3-30B-A3B-tiny DEVICE=rtx_pro_6000 bash examples/profiling/profile_linear_op.sh --dry-run
MODEL=Qwen3-30B-A3B-tiny DEVICE=rtx_pro_6000 bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
MODEL=Qwen3-30B-A3B-tiny DEVICE=rtx_pro_6000 bash examples/profiling/profile_moe.sh --dry-run
```

The downstream simulator smokes of the current release-facing examples support co-location only:
`smoke_simulator_dense_csv.sh` and `smoke_simulator_moe_csv.sh` both run `--sys_arch co-location`.

The legacy/internal scripts under `frontier/profiling/example/` are kept as references, but they are not the release entry point and do not run as-is.

#### Historical / guarded architecture notes

Earlier releases guarded out the disaggregated runtime modes. This release runs
`pd-disaggregation` and `pd-af-disaggregation` in sequential mode
(`--no-enable_parallel_clusters`) from the same operator tables, with these differences:

- PD-AF prices its `DECODE_ATTN` decode steps and its `DECODE_FFN` steps from the kernel-only
  tables (`*_kernel_only.csv`), so profile those roles with `--profile_method record_function`.
- Under PD-AF, a `kernel_gap.csv` file or a CPU-overhead table with `forward_launch` columns stops
  the run with an error (see [Kernel Gap Profiling](#kernel-gap-profiling)).
- `frontier/profiling/example/test_pd_af_profiling.sh` is a legacy per-role profiling script; see
  `frontier/profiling/example/README.md`.

### Execution Modes

| Mode | Flag | Description |
|------|------|-------------|
| Single-GPU | `--num_gpus 1 --disable_ray` | Sequential execution on one GPU |
| Multi-GPU (ProcessPool) | `--num_gpus N --disable_ray` | Parallel execution using multiprocessing |
| Ray Mode | `--num_gpus N` (no `--disable_ray`) | Distributed execution via Ray actors |

**⚠️ Important**: Ray mode is currently broken due to grpcio 1.67.1 incompatibility. Always use `--disable_ray`.

---

## Output Format and Data Structure

### Output Directory Structure

These are the paths the simulator reads by default (`frontier/config/execution_time_predictor_config.py`
and `frontier/cc_backend/cc_backend_config.py`). `<device>` is `--replica_config_device`,
`<network_device>` is `--replica_config_network_device` and `<model>` is
`--replica_config_model_name`.

```
data/profiling/
├── compute/<device>/<model>/
│   ├── linear_op.csv                # CUDA_EVENT linear-op timings
│   ├── linear_op_kernel_only.csv    # KERNEL_ONLY linear-op timings
│   ├── attention.csv
│   ├── attention_kernel_only.csv
│   ├── moe.csv
│   ├── moe_kernel_only.csv
│   ├── gdn.csv                      # GDN layers (frontier.profiling.gdn.main)
│   └── kernel_gap.csv               # Device idle time per kernel (frontier.profiling.kernel_gap.main)
├── cpu_overhead/<network_device>/<model>/
│   ├── cpu_overheads.csv            # Host overheads of eager steps
│   └── cpu_overheads_kernel_only.csv # Host overheads of CUDA-graph steps
├── network/<network_device>/
│   ├── all_reduce.csv               # Read by the vidur CC backend
│   └── send_recv.csv
└── other_overhead/<device>/<model>/
    ├── pp_stage_boundary.csv
    ├── pp_receiver_head.csv
    ├── pp_producer_send_path.csv
    └── pp_prefill_consumer_active.csv
```

On ROCm, `device_event` runs write `<op>_device_event.csv` beside the tables above. The attention
profiler can also write `attention_mixed.csv`, `attention_true_mixed.csv` and
`attention_combined.csv` (see `attention/README_MIXED_BATCH.md`).

The operator profilers write `<output_dir>/compute/<device>/<model>/<op_name>.csv`: pass
`--output_dir data/profiling` and they append `compute/<device>/<model>/` internally. The
CPU-overhead profiler uses a different layout (see [CPU Overhead Profiling](#cpu-overhead-profiling)).

### CSV Column Structure

#### Timing Statistics (per operation)

Each operation generates these columns:

| Column | Description |
|--------|-------------|
| `time_stats.{operation}.min` | Minimum time (ms) |
| `time_stats.{operation}.max` | Maximum time (ms) |
| `time_stats.{operation}.mean` | Mean time (ms) |
| `time_stats.{operation}.median` | Median time (ms) |
| `time_stats.{operation}.std` | Standard deviation (ms) |
| `time_stats.{operation}.count` | Number of timed samples |
| `time_stats.{operation}.kernel_count` | Kernels the operator launches (kernel-only tables only) |

MoE tables add `block_size_m`, `block_size_m_ranges` and `num_tokens_post_padded` statistics for
the grouped GEMM.

#### Linear Op CSV Columns

```csv
time_stats.mlp_up_proj.mean,time_stats.mlp_down_proj.mean,time_stats.mlp_act.mean,
time_stats.input_layernorm.mean,time_stats.post_attention_layernorm.mean,
time_stats.attn_pre_proj.mean,time_stats.attn_post_proj.mean,time_stats.attn_rope.mean,
time_stats.add.mean,...,
num_tokens,num_tensor_parallel_workers,n_head,n_kv_head,n_embd,n_expanded_embd,
vocab_size,use_gated_mlp,model_arch,model_architecture_profile,typed_operator_contracts,
measurement_type,profiling_precision,quant_signature,...
```

`time_stats.add` appears only for models whose residual add is not fused into the norm, and a
`--is_moe` run has no `mlp_*` columns.

#### Attention CSV Columns

```csv
time_stats.attn_prefill.mean,time_stats.attn_decode.mean,time_stats.attn_kv_cache_save.mean,...,
batch_size,prefill_chunk_size,kv_cache_size,is_prefill,
num_tensor_parallel_workers,n_embd,n_q_head,n_kv_head,block_size,max_model_len,
attention_backend,is_mixed_batch,mode,total_tokens,max_seq_len,min_seq_len,
avg_seq_len,seq_len_variance,profiling_precision,measurement_type,model_arch,
model_architecture_profile,quant_signature,...
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
profiling_precision,model_arch,model_architecture_profile,typed_operator_contracts,
quant_signature,...
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
| `model_architecture_profile` | Architecture profile (`generic`, `step2_mini`, `step3_text`, `qwen3_5_moe`); checked against the active model at load time |
| `typed_operator_contracts` | JSON map from operator name to its owning family and TP / EP selection (`linear_op` and `moe` tables) |
| `measurement_type` | `CUDA_EVENT`, `KERNEL_ONLY` or `DEVICE_EVENT` |
| `quant_signature` | Quantization identity of the profiled weights (`none` without quantization) |
| `attention_backend` | Attention backend used |

`python -m frontier.profiling.migrate_csv_metadata` adds the required metadata columns
(`profiling_precision`, `model_arch`, `model_architecture_profile`, `quant_signature`,
`measurement_type`) to a legacy CSV (`--input_csv`) or a directory of CSVs (`--input_dir`).

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

### Dependencies

`environment_profiling.yml` pins the profiling stack: Python 3.10, `torch>=2.7,<2.9`,
`vllm>=0.10,<0.11`, `flashinfer-python>=0.3,<0.4`, `triton>=3,<4`, the conda-forge CUDA 12.8
compiler and runtime (`cuda-nvcc`, `cuda-cudart`, `cuda-cudart-dev`), and the simulator's
`numpy`, `pandas`, `scipy`, `scikit-learn`, `tqdm` and `pyyaml`.

- FlashInfer is needed for the default attention backend; FlashInfer JIT needs a working `nvcc`.
- vLLM is needed for MoE fused kernels, GDN profiling and the `cuda_graph` mode of kernel-gap profiling.
- Ray is optional (see [Execution Modes](#execution-modes)).

### GPU Requirements

| Profiling Type | Minimum VRAM | Recommended |
|----------------|--------------|-------------|
| Linear Op | 16 GB | 40 GB |
| Attention | 24 GB | 80 GB |
| MoE | 24 GB | 80 GB |

### CUDA Compatibility

The profiling environment installs the CUDA 12.8 toolchain from conda-forge. See the FlashInfer
JIT notes in `AGENTS.md` when `nvcc` or `-lcudart` cannot be found.

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
pip install "flashinfer-python>=0.3,<0.4"

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

1. **Preview a Run**: the release wrappers accept `--dry-run`, and each lower-level profiler
   prints its parameter table and asks for confirmation before it starts (`--yes` skips the prompt).

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

The training CLI trains predictors from profiling CSVs (see
[Step 2](#step-2-train-prediction-models-optional) and `docs/training/README.md`):

```bash
python -m frontier.training.cli {linear_op,attention,moe,gdn} --help
```

The `train_*` functions in `frontier/training/cli.py` take the parsed `argparse` namespace of
these subcommands; call the CLI rather than the functions.

### Execution Time Predictor (`frontier/execution_time_predictor/`)

The simulator builds its predictors itself. They read the tables named by the
`--random_forrest_execution_time_predictor_config_*_input_file` flags (defaults in
[Output Directory Structure](#output-directory-structure)), train any missing model and cache it
under `--metrics_config_cache_dir`. MoE models use `SklearnMoEExecutionTimePredictor`
(`frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`).

### CC Backend (`frontier/cc_backend/`)

The `vidur` CC backend trains on `data/profiling/network/<network_device>/all_reduce.csv` and
`send_recv.csv`:

```python
from frontier.cc_backend import CCBackendFactory
from frontier.cc_backend.cc_backend_config import VidurCCBackendConfig
from frontier.types import CCBackendType, ClusterType

backend = CCBackendFactory.create(
    backend_type=CCBackendType.VIDUR,
    config=VidurCCBackendConfig(),
    cluster_type=ClusterType.MONOLITHIC,
    device_type="a100",
    network_device="a100_dgx",
    num_devices=8,
)

# Predicted all-reduce time in ms
allreduce_ms = backend.predict_allreduce(data_size_bytes=1024 * 1024, num_devices=8)
```

---

## References

- [Public profiling guide](../../docs/profiling/README.md) - Release wrappers and downstream simulator checks
- [MoE Profiling README](moe/README.md) - Detailed MoE profiling documentation
- [Linear Op README](linear_op/README.md) - Linear operation profiling details
- [GDN Profiler README](gdn/README.md) - Qwen3.5 GDN layer profiling
- [Non-KV Cache Overhead README](non_kv_cache_overhead/README.md) - GPU memory profiling for KV block planning
- [Example Scripts README](example/README.md) - Legacy script documentation
- [AGENTS.md](../../AGENTS.md) - Project-wide development guidelines
